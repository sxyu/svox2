# Copyright 2021 Alex Yu

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
from util.util import Timing, get_expon_lr_func

from torch.utils.tensorboard import SummaryWriter

from tqdm import tqdm
from typing import NamedTuple, Optional, Union

device = "cuda" if torch.cuda.is_available() else "cpu"

parser = argparse.ArgumentParser()
parser.add_argument('data_dir', type=str)
group = parser.add_argument_group("general")
group.add_argument('--train_dir', '-t', type=str, default='ckpt',
                     help='checkpoint and logging directory')
group.add_argument('--final_reso', type=int, default=512,
                   help='FINAL grid resolution')
group.add_argument('--init_reso', type=int, default=256,
                   help='INITIAL grid resolution')
group.add_argument('--ref_reso', type=int, default=256,
                   help='reference grid resolution (for adjusting lr)')
group.add_argument('--sh_dim', type=int, default=9, help='learned basis dimensions (at most 10)')
group.add_argument('--scene_scale', type=float, default=
                           2/3,
                           help='Scene scale; generally 2/3, can be 5/6 for lego (no longer doing this)')

group = parser.add_argument_group("optimization")
group.add_argument('--batch_size', type=int, default=
                     5000,
                     #100000,
                     #  2000,
                   help='batch size')


# TODO: make the lr higher near the end
group.add_argument('--sigma_optim', choices=['sgd', 'rmsprop'], default='rmsprop', help="Density optimizer")
group.add_argument('--lr_sigma', type=float, default=
                                            #  1e1,
                                            2e1,
                                            #5e1,
                                            #5e1,#2e0,#1e8
        help='SGD/rmsprop lr for sigma')
group.add_argument('--lr_sigma_final', type=float, default=5e-1)
group.add_argument('--lr_sigma_decay_steps', type=int, default=250000)
group.add_argument('--lr_sigma_delay_steps', type=int, default=15000, help="Reverse cosine steps (0 means disable)")
group.add_argument('--lr_sigma_delay_mult', type=float, default=1e-2)


group.add_argument('--sh_optim', choices=['sgd', 'rmsprop'], default='rmsprop', help="SH optimizer")
group.add_argument('--lr_sh', type=float, default=#2e6,
                    1e-3,
                   help='SGD/rmsprop lr for SH')
group.add_argument('--lr_sh_final', type=float,
                      default=#2e6
                      1e-3
                    )
group.add_argument('--lr_sh_decay_steps', type=int, default=250000)
group.add_argument('--lr_sh_delay_steps', type=int, default=0, help="Reverse cosine steps (0 means disable)")
group.add_argument('--lr_sh_delay_mult', type=float, default=1e-2)
group.add_argument('--lr_sh_upscale_factor', type=float, default=1.0)


group.add_argument('--basis_optim', choices=['sgd', 'rmsprop'], default='rmsprop', help="Learned basis optimizer")
group.add_argument('--lr_basis', type=float, default=#2e6,
                    1e-3,
                   help='SGD/rmsprop lr for SH')
group.add_argument('--lr_basis_final', type=float,
                      default=#2e6
                      1e-3
                    )
group.add_argument('--lr_basis_decay_steps', type=int, default=250000)
group.add_argument('--lr_basis_delay_steps', type=int, default=0, help="Reverse cosine steps (0 means disable)")
group.add_argument('--lr_basis_delay_mult', type=float, default=1e-2)


group.add_argument('--n_epochs', type=int, default=30)
group.add_argument('--print_every', type=int, default=20, help='print every')
group.add_argument('--upsamp_every', type=int, default=
                     3 * 12800,
                    help='upsample the grid every x iters')
group.add_argument('--save_every', type=int, default=5,
                   help='save every x epochs')
group.add_argument('--eval_every', type=int, default=1,
                   help='evaluate every x epochs')

group = parser.add_argument_group("initialization")
group.add_argument('--init_rgb', type=float, default=0.0, help='initialization rgb (pre-sigmoid)')
group.add_argument('--init_sigma', type=float, default=0.1, help='initialization sigma')
group.add_argument('--init_basis_mean', type=float, default=1.0,
                   help='initialization learned basis std')
group.add_argument('--init_basis_std', type=float, default=0.1,
                   help='initialization learned basis std')


group = parser.add_argument_group("misc experiments")
group.add_argument('--perm', action='store_true', default=True,
                    help='sample by permutation of rays (true epoch) instead of '
                         'uniformly random rays')
group.add_argument('--sigma_thresh', type=float,
                    default=2.5,
                   help='Resample (upsample to 512) sigma threshold')
group.add_argument('--weight_thresh', type=float,
                    default=0.001,
                   help='Resample (upsample to 512) weight threshold')
group.add_argument('--use_weight_thresh', action='store_true', default=True,
                    help='use weight thresholding')
group.add_argument('--prox_l1_alpha', type=float, default=0.0,
                   help='proximal L1 per epoch; amount to subtract from sigma')
group.add_argument('--prox_l0', action='store_true', default=False,
                   help='proximal L0 i.e., keep resampling after each epoch')
#  group.add_argument('--norand', action='store_true', default=True,
#                     help='disable random')

group.add_argument('--tune_mode', action='store_true', default=False,
                   help='hypertuning mode (do not save, for speed)')

group.add_argument('--rms_beta', type=float, default=0.9)
group.add_argument('--lambda_tv', type=float, default=
                    1e-2)
                    #  1e-3)
group.add_argument('--tv_sparsity', type=float, default=
                        #  0.001)
                        0.01)
                        #  1.0)

group.add_argument('--lambda_tv_sh', type=float, default=0.0)
group.add_argument('--tv_sh_sparsity', type=float, default=0.01)

group.add_argument('--lambda_tv_basis', type=float, default=1e-5)

group.add_argument('--weight_decay_sigma', type=float, default=1.0)
group.add_argument('--weight_decay_sh', type=float, default=1.0)

group.add_argument('--lr_decay', action='store_true', default=True)
group.add_argument('--use_sphere_bound', action='store_true', default=True)
args = parser.parse_args()

assert args.lr_sigma_final <= args.lr_sigma, "lr_sigma must be >= lr_sigma_final"
assert args.lr_sh_final <= args.lr_sh, "lr_sh must be >= lr_sh_final"
assert args.lr_basis_final <= args.lr_basis, "lr_basis must be >= lr_basis_final"

os.makedirs(args.train_dir, exist_ok=True)
summary_writer = SummaryWriter(args.train_dir)

with open(path.join(args.train_dir, 'args.json'), 'w') as f:
    json.dump(args.__dict__, f, indent=2)
    shutil.copyfile(__file__, path.join(args.train_dir, 'opt.py'))

torch.manual_seed(20200823)
np.random.seed(20200823)

reso = args.init_reso
factor = args.ref_reso // reso

dset = Dataset(args.data_dir, split="train", device=device, permutation=args.perm,
               factor=factor,
               scene_scale=args.scene_scale)
dset.shuffle_rays()
dset_test = Dataset(args.data_dir, split="test", scene_scale=args.scene_scale)

grid = svox2.SparseGrid(reso=reso,
                        radius=1.0,
                        basis_dim=args.sh_dim,
                        use_z_order=True,
                        device=device,
                        use_sphere_bound=args.use_sphere_bound)
grid.sh_data.data[:] = args.init_rgb
grid.density_data.data[:] = args.init_sigma
grid.basis_data.data.normal_(mean=args.init_basis_mean, std=args.init_basis_std)
grid.basis_data.data[..., 0] = args.init_basis_mean  # DC init

grid.requires_grad_(True)
step_size = 0.5  # 0.5 of a voxel!
#  step_size = 2.0

grid.opt.step_size = step_size
grid.opt.sigma_thresh = 1e-8
grid.opt.backend = 'cuvol'

gstep_id_base = 0

resample_cameras = [
        svox2.Camera(c2w.to(device=device), dset.focal, dset.focal,
                     dset.w, dset.h) for c2w in dset.c2w
    ] if args.use_weight_thresh else None
ckpt_path = path.join(args.train_dir, 'ckpt.npz')

lr_sigma_func = get_expon_lr_func(args.lr_sigma, args.lr_sigma_final, args.lr_sigma_delay_steps,
                                  args.lr_sigma_delay_mult, args.lr_sigma_decay_steps)
lr_sh_func = get_expon_lr_func(args.lr_sh, args.lr_sh_final, args.lr_sh_delay_steps,
                               args.lr_sh_delay_mult, args.lr_sh_decay_steps)
lr_basis_func = get_expon_lr_func(args.lr_basis, args.lr_basis_final, args.lr_basis_delay_steps,
                               args.lr_basis_delay_mult, args.lr_basis_decay_steps)
lr_sigma_factor = 1.0
lr_sh_factor = 1.0
lr_basis_factor = 1.0

last_upsamp_step = 0

for epoch_id in range(args.n_epochs):
    epoch_size = dset.rays.origins.size(0)
    batches_per_epoch = (epoch_size-1)//args.batch_size+1
    # Test
    def eval_step():
        # Put in a function to avoid memory leak
        print('Eval step')
        with torch.no_grad():
            stats_test = {'psnr' : 0.0, 'mse' : 0.0}

            # Standard set
            #  N_IMGS_TO_SAVE = 5
            #  N_IMGS_TO_EVAL = 20 if epoch_id > 0 else 5
            #  img_eval_interval = dset_test.n_images // N_IMGS_TO_EVAL
            #  img_save_interval = (N_IMGS_TO_EVAL // N_IMGS_TO_SAVE)
            #  img_ids = range(0, dset_test.n_images, img_eval_interval)

            # Special 'very hard' specular + fuzz set
            img_ids = [2, 5, 7, 9, 21,
                       44, 45, 47, 49, 56,
                       80, 88, 99, 115, 120,
                       154]
            img_save_interval = 1

            n_images_gen = 0
            for i, img_id in tqdm(enumerate(img_ids), total=len(img_ids)):
                c2w = dset_test.c2w[img_id].to(device=device)
                cam = svox2.Camera(c2w, dset_test.focal, dset_test.focal,
                                   dset_test.w, dset_test.h)
                rgb_pred_test = grid.volume_render_image(cam, use_kernel=True)
                rgb_gt_test = dset_test.gt[img_id].to(device=device)
                all_mses = ((rgb_gt_test - rgb_pred_test) ** 2).cpu()
                if i % img_save_interval == 0:
                    summary_writer.add_image(f'test/image_{img_id:04d}',
                            rgb_pred_test.cpu(), global_step=gstep_id_base, dataformats='HWC')
                rgb_pred_test = rgb_gt_test = None
                mse_num : float = all_mses.mean().item()
                psnr = -10.0 * math.log10(mse_num)
                stats_test['mse'] += mse_num
                stats_test['psnr'] += psnr
                n_images_gen += 1

            stats_test['mse'] /= n_images_gen
            stats_test['psnr'] /= n_images_gen
            for stat_name in stats_test:
                summary_writer.add_scalar('test/' + stat_name,
                        stats_test[stat_name], global_step=gstep_id_base)
            summary_writer.add_scalar('epoch_id', float(epoch_id), global_step=gstep_id_base)
            print('eval stats:', stats_test)
    if epoch_id % max(factor, args.eval_every) == 0:
        eval_step()
        gc.collect()

    def train_step():
        print('Train step')
        pbar = tqdm(enumerate(range(0, epoch_size, args.batch_size)), total=batches_per_epoch)
        stats = {"mse" : 0.0, "psnr" : 0.0, "invsqr_mse" : 0.0}
        for iter_id, batch_begin in pbar:
            gstep_id = iter_id + gstep_id_base
            lr_sigma = lr_sigma_func(gstep_id) * lr_sigma_factor
            lr_sh = lr_sh_func(gstep_id) * lr_sh_factor
            lr_basis = lr_basis_func(gstep_id) * lr_basis_factor
            if not args.lr_decay:
                lr_sigma = args.lr_sigma * lr_sigma_factor
                lr_sh = args.lr_sh * lr_sh_factor
                lr_basis = args.lr_basis * lr_basis_factor

            batch_end = min(batch_begin + args.batch_size, epoch_size)
            batch_origins = dset.rays.origins[batch_begin: batch_end]
            batch_dirs = dset.rays.dirs[batch_begin: batch_end]
            rgb_gt = dset.rays.gt[batch_begin: batch_end]
            rays = svox2.Rays(batch_origins, batch_dirs)
            rgb_pred = grid.volume_render_fused(rays, rgb_gt)

            mse = F.mse_loss(rgb_gt, rgb_pred)

            # Stats
            mse_num : float = mse.detach().item()
            psnr = -10.0 * math.log10(mse_num)
            stats['mse'] += mse_num
            stats['psnr'] += psnr
            stats['invsqr_mse'] += 1.0 / mse_num ** 2

            if (iter_id + 1) % args.print_every == 0:
                # Print averaged stats
                pbar.set_description(f'epoch {epoch_id}/{args.n_epochs} psnr={psnr:.2f}')
                for stat_name in stats:
                    stat_val = stats[stat_name] / args.print_every
                    summary_writer.add_scalar(stat_name, stat_val, global_step=gstep_id)
                    stats[stat_name] = 0.0
                if args.lambda_tv > 0.0:
                    with torch.no_grad():
                        tv = grid.tv()
                    summary_writer.add_scalar("loss_tv", tv, global_step=gstep_id)
                if args.lambda_tv_sh > 0.0:
                    with torch.no_grad():
                        tv_sh = grid.tv_color()
                    summary_writer.add_scalar("loss_tv_sh", tv_sh, global_step=gstep_id)
                with torch.no_grad():
                    tv_basis = grid.tv_basis()
                summary_writer.add_scalar("loss_tv_basis", tv_basis, global_step=gstep_id)
                summary_writer.add_scalar("lr_sh", lr_sh, global_step=gstep_id)
                summary_writer.add_scalar("lr_sigma", lr_sigma, global_step=gstep_id)

                if args.weight_decay_sh < 1.0:
                    grid.sh_data.data *= args.weight_decay_sigma
                if args.weight_decay_sigma < 1.0:
                    grid.density_data.data *= args.weight_decay_sh

            # Apply TV
            if args.lambda_tv > 0.0:
                grid.inplace_tv_grad(grid.density_data.grad,
                        scaling=args.lambda_tv,
                        sparse_frac=args.tv_sparsity)
            if args.lambda_tv_sh > 0.0:
                grid.inplace_tv_color_grad(grid.sh_data.grad,
                        scaling=args.lambda_tv_sh,
                        sparse_frac=args.tv_sh_sparsity)
            if args.lambda_tv_basis > 0.0:
                tv_basis = grid.tv_basis()
                loss_tv_basis = tv_basis * args.lambda_tv_basis
                loss_tv_basis.backward()

            # Manual SGD/rmsprop step
            grid.optim_density_step(lr_sigma, beta=args.rms_beta, optim=args.sigma_optim)
            grid.optim_sh_step(lr_sh, beta=args.rms_beta, optim=args.sh_optim)
            grid.optim_basis_step(lr_basis, beta=args.rms_beta, optim=args.basis_optim)

    train_step()
    gc.collect()
    gstep_id_base += batches_per_epoch

    #  ckpt_path = path.join(args.train_dir, f'ckpt_{epoch_id:05d}.npz')
    # Overwrite prev checkpoints since they are very huge
    if args.save_every > 0 and (epoch_id + 1) % max(
            factor, args.save_every) == 0 and not args.tune_mode:
        print('Saving', ckpt_path)
        grid.save(ckpt_path)

    if (gstep_id_base - last_upsamp_step) >= args.upsamp_every:
        last_upsamp_step = gstep_id_base
        if reso < args.final_reso or args.prox_l0:
            print('* Upsampling from', reso, 'to', reso * 2)
            non_final = reso < args.final_reso
            if non_final:
                reso *= 2
            use_sparsify = True # reso >= args.ref_reso
            grid.resample(reso=reso,
                    sigma_thresh=args.sigma_thresh if use_sparsify else 0.0,
                    weight_thresh=args.weight_thresh if use_sparsify else 0.0,
                    dilate=1, #use_sparsify,
                    cameras=resample_cameras)
            if non_final:
                #  if reso <= args.ref_reso:
                #  lr_sigma_factor *= 8
                #  else:
                #  lr_sigma_factor *= 4
                lr_sh_factor *= args.lr_sh_upscale_factor
            print('Increased lr to (sigma:)', args.lr_sigma, '(sh:)', args.lr_sh)

        if factor > 1 and reso < args.final_reso:
            factor //= 2
            dset.gen_rays(factor=factor)
            dset.shuffle_rays()

    if args.prox_l1_alpha > 0.0:
        print('ProxL1: sigma -=', args.prox_l1_alpha)
        grid.density_data.data -= args.prox_l1_alpha

    if epoch_id == args.n_epochs - 1:
        print('Final eval and save')
        eval_step()
        if not args.tune_mode:
            grid.save(ckpt_path)
