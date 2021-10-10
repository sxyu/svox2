# First, install svox2
# Then, python opt.py <path_to>/nerf_synthetic/<scene> -t ckpt/<some_name>
# or use launching script:   sh launch.sh <EXP_NAME> <GPU> <DATA_DIR>
import torch
import torch.cuda
import torch.nn.functional as F
import svox2
import json
import imageio
import os
from os import path
import shutil
import gc
import numpy as np
import math
import argparse
from util.dataset import Dataset
from util.util import Timing

from torch.utils.tensorboard import SummaryWriter

from tqdm import tqdm
from typing import NamedTuple, Optional, Union

device = "cuda" if torch.cuda.is_available() else "cpu"

parser = argparse.ArgumentParser()
parser.add_argument('data_dir', type=str)
group = parser.add_argument_group("general")
group.add_argument('--train_dir', '-t', type=str, default='ckpt',
                     help='checkpoint and logging directory')
group.add_argument('--scene_scale', type=float, default=5/6,
                   help='Scene scale; generally 2/3, can be 5/6 for lego')
group.add_argument('--reso', type=int, default=256, help='grid resolution')
group.add_argument('--sh_dim', type=int, default=8,#9,
                   help='SH dimensions, must be square number >=1, <= 16')

group = parser.add_argument_group("optimization")
group.add_argument('--batch_size', type=int,
                   default=20000,#20000,#5000,
                   help='batch size')
group.add_argument('--eval_batch_size', type=int, default=200000, help='evaluation batch size')
group.add_argument('--lr_sigma', type=float, default=2e7, help='SGD lr for sigma')
#  group.add_argument('--lr_color', type=float, default=5e5, help='SGD lr for base color')
#  group.add_argument('--lr_coeff', type=float, default=5e5, help='SGD lr for coeffs')
group.add_argument('--lr_color', type=float, default=2e5, help='sgd lr for base color')
group.add_argument('--lr_coeff', type=float, default=2e5,
            help='SGD lr for coeffs')
group.add_argument('--lr_cubemap', type=float,
                    default=2e2, #2e3, #2e4,#2e5,#2e2,
                    help='SGD lr for cubemap')
group.add_argument('--n_epochs', type=int, default=20)
group.add_argument('--print_every', type=int, default=20, help='print every')
group.add_argument('--cubemap_reso', type=int, default=8,#4,#16,
                   help='cubemap resolution (per face)')

group = parser.add_argument_group("initialization")
group.add_argument('--init_rgb', type=float, default=0.0, help='initialization rgb (pre-sigmoid)')
group.add_argument('--init_sigma', type=float, default=0.1, help='initialization sigma')
group.add_argument('--init_cubemap_mean', type=float, default=0.5,
                   help='initialization of cubemap elements (mean)')
group.add_argument('--init_cubemap_std', type=float, default=0.05,
                   help='initialization of cubemap elements (std)')


group = parser.add_argument_group("misc experiments")
group.add_argument('--no_lerp', action='store_true', default=False,
                    help='use nearest neighbor interp (faster)')
group.add_argument('--perm', action='store_true', default=True,
                    help='sample by permutation of rays (true epoch) instead of '
                         'uniformly random rays')
group.add_argument('--resample_thresh', type=float, default=2.5,
                   help='Resample (upsample to 512) sigma threshold')
group.add_argument('--prox_l1_alpha', type=float, default=0.0,
                   help='proximal L1 per epoch; amount to subtract from sigma')
group.add_argument('--prox_l0', action='store_true', default=False,
                   help='proximal L0 i.e., keep resampling after each epoch')
args = parser.parse_args()

os.makedirs(args.train_dir, exist_ok=True)
summary_writer = SummaryWriter(args.train_dir)


with open(path.join(args.train_dir, 'args.json'), 'w') as f:
    json.dump(args.__dict__, f, indent=2)
shutil.copyfile(__file__, path.join(args.train_dir, 'opt.py'))

torch.manual_seed(20200823)
np.random.seed(20200823)

dset = Dataset(args.data_dir, split="train", device=device,
               permutation=args.perm,
               scene_scale=args.scene_scale)
dset_test = Dataset(args.data_dir, split="test",
                    scene_scale=args.scene_scale)

grid = svox2.SparseGrid(reso=args.reso,
                        radius=1.0,
                        basis_dim=args.sh_dim,
                        use_z_order=True,
                        cubemap_reso=args.cubemap_reso,
                        device=device)
grid.data.data[..., 1:] = args.init_rgb
grid.data.data[..., :1] = args.init_sigma
#  grid.data.data[..., 4:] = 0.0  # DEBUG

grid.cubemap.data.normal_(args.init_cubemap_mean, args.init_cubemap_std)

grid.requires_grad_(True)
step_size = 0.5  # 0.5 of a voxel!
epoch_size = dset.rays.origins.size(0)
batches_per_epoch = (epoch_size-1)//args.batch_size+1

grid.opt.step_size = step_size
grid.opt.sigma_thresh = 1e-8
grid.opt.backend = 'cuvol'

for epoch_id in range(args.n_epochs):
    # Test
    def eval_step():
        # Put in a function to avoid memory leak
        print('Eval step')
        with torch.no_grad():
            im_size = dset_test.h * dset_test.w
            stats_test = {'psnr' : 0.0, 'mse' : 0.0}
            START_IMAGE = 4
            N_IMGS_TO_SAVE = 5
            N_IMGS_TO_EVAL = 20
            img_eval_interval = dset_test.n_images // N_IMGS_TO_EVAL
            img_save_interval = img_eval_interval * (N_IMGS_TO_EVAL // N_IMGS_TO_SAVE)
            gstep_id = epoch_id * batches_per_epoch
            n_images_gen = 0
            for img_id in tqdm(range(START_IMAGE, dset_test.n_images, img_eval_interval)):
                all_rgbs = []
                all_mses = []
                for batch_begin in range(0, im_size, args.eval_batch_size):
                    batch_end = min(batch_begin + args.eval_batch_size, im_size)
                    batch_origins = dset_test.rays.origins[img_id][batch_begin: batch_end].to(device=device)
                    batch_dirs = dset_test.rays.dirs[img_id][batch_begin: batch_end].to(device=device)
                    rgb_gt_test = dset_test.rays.gt[img_id][batch_begin: batch_end].to(device=device)

                    rays = svox2.Rays(batch_origins, batch_dirs)
                    rgb_pred_test = grid.volume_render(rays, use_kernel=True)
                    all_rgbs.append(rgb_pred_test.cpu())
                    all_mses.append(((rgb_gt_test - rgb_pred_test) ** 2).cpu())
                if (img_id - START_IMAGE) % img_save_interval == 0 and len(all_rgbs):
                    im = torch.cat(all_rgbs).view(dset_test.h, dset_test.w, all_rgbs[0].size(-1))
                    summary_writer.add_image(f'test/image_{img_id:04d}',
                            im, global_step=gstep_id, dataformats='HWC')
                    im = None
                mse_num : float = torch.cat(all_mses).mean().item()
                psnr = -10.0 * math.log10(mse_num)
                stats_test['mse'] += mse_num
                stats_test['psnr'] += psnr
                n_images_gen += 1

            stats_test['mse'] /= n_images_gen
            stats_test['psnr'] /= n_images_gen
            for stat_name in stats_test:
                summary_writer.add_scalar('test/' + stat_name,
                        stats_test[stat_name], global_step=gstep_id)
            summary_writer.add_scalar('epoch_id', float(epoch_id), global_step=gstep_id)
            print('eval stats:', stats_test)
    eval_step()
    gc.collect()

    def train_step():
        print('Train step')
        dset.shuffle_rays()
        pbar = tqdm(enumerate(range(0, epoch_size, args.batch_size)), total=batches_per_epoch)
        stats = {"mse" : 0.0, "psnr" : 0.0, "invsqr_mse" : 0.0}
        for iter_id, batch_begin in pbar:
            batch_end = min(batch_begin + args.batch_size, epoch_size)
            batch_origins = dset.rays.origins[batch_begin: batch_end]
            batch_dirs = dset.rays.dirs[batch_begin: batch_end]
            rgb_gt = dset.rays.gt[batch_begin: batch_end]
            rays = svox2.Rays(batch_origins, batch_dirs)
            rgb_pred = grid.volume_render(rays, use_kernel=True, randomize=True)

            mse = F.mse_loss(rgb_gt, rgb_pred)

            # Stats
            mse_num : float = mse.detach().item()
            psnr = -10.0 * math.log10(mse_num)
            stats['mse'] += mse_num
            stats['psnr'] += psnr
            stats['invsqr_mse'] += 1.0 / mse_num ** 2

            if (iter_id + 1) % args.print_every == 0:
                # Print averaged stats
                gstep_id = iter_id + epoch_id * batches_per_epoch
                pbar.set_description(f'epoch {epoch_id}/{args.n_epochs} psnr={psnr:.2f}')
                for stat_name in stats:
                    stat_val = stats[stat_name] / args.print_every
                    summary_writer.add_scalar(stat_name, stat_val, global_step=gstep_id)
                    stats[stat_name] = 0.0

            # Backprop
            mse.backward()

            # Manual SGD step
            grid.data.grad[..., 1:4] *= args.lr_color
            grid.data.grad[..., 4:] *= args.lr_coeff
            grid.data.grad[..., :1] *= args.lr_sigma
            grid.cubemap.grad *= args.lr_cubemap
            grid.data.data -= grid.data.grad
            grid.cubemap.data -= grid.cubemap.grad
            #  grid.data.data[..., 4:] = 0.0  # DEBUG
            del grid.data.grad  # Save memory
            del grid.cubemap.grad  # Save memory

    train_step()
    gc.collect()

    #  ckpt_path = path.join(args.train_dir, f'ckpt_{epoch_id:05d}.npz')
    # Overwrite prev checkpoints since they are very huge
    ckpt_path = path.join(args.train_dir, 'ckpt.npz')
    print('Saving', ckpt_path)
    grid.save(ckpt_path)

    if epoch_id == 0 or args.prox_l0:
        print('Upsampling!!!')
        grid.resample(reso=512, sigma_thresh=args.resample_thresh)

    if args.prox_l1_alpha > 0.0:
        print('ProxL1: sigma -=', args.prox_l1_alpha)
        grid.data.data[..., :1] -= args.prox_l1_alpha

