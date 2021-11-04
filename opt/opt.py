# Copyright 2021 Alex Yu

# First, install svox2
# Then, python opt.py <path_to>/nerf_synthetic/<scene> -t ckpt/<some_name>
# or use launching script:   sh launch.sh <EXP_NAME> <GPU> <DATA_DIR>
import torch
import torch.cuda
import torch.optim
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
import cv2
from util.dataset import datasets
from util.util import Timing, get_expon_lr_func, generate_dirs_equirect, viridis_cmap
from util import config_util

from torch.utils.tensorboard import SummaryWriter

from tqdm import tqdm
from typing import NamedTuple, Optional, Union

device = "cuda" if torch.cuda.is_available() else "cpu"

parser = argparse.ArgumentParser()
config_util.define_common_args(parser)


group = parser.add_argument_group("general")
group.add_argument('--train_dir', '-t', type=str, default='ckpt',
                     help='checkpoint and logging directory')

group.add_argument('--reso',
                        type=str,
                        default=
                        "[[128, 128, 128], [256, 256, 256], [512, 512, 512]]",
                       help='List of grid resolution (will be evaled as json);'
                            'resamples to the next one every upsamp_every iters, then ' +
                            'stays at the last one; ' +
                            'should be a list where each item is a list of 3 ints or an int')
group.add_argument('--upsamp_every', type=int, default=
                     2 * 12800,
                    help='upsample the grid every x iters')


group.add_argument('--basis_type',
                    choices=['sh', '3d_texture', 'mlp'],
                    default='sh',
                    help='Basis function type')

group.add_argument('--basis_reso', type=int, default=32,
                   help='basis grid resolution (only for learned texture)')
group.add_argument('--sh_dim', type=int, default=9, help='SH/learned basis dimensions (at most 10)')

group.add_argument('--mlp_posenc_size', type=int, default=4, help='Positional encoding size if using MLP basis; 0 to disable')
group.add_argument('--mlp_width', type=int, default=32, help='MLP width if using MLP basis')

group.add_argument('--background_nlayers', type=int, default=32, help='Number of background layers (0=disable BG model)')
group.add_argument('--background_reso', type=int, default=512, help='Background resolution')



group = parser.add_argument_group("optimization")
group.add_argument('--batch_size', type=int, default=
                     5000,
                     #100000,
                     #  2000,
                   help='batch size')


# TODO: make the lr higher near the end
group.add_argument('--sigma_optim', choices=['sgd', 'rmsprop'], default='rmsprop', help="Density optimizer")
group.add_argument('--lr_sigma', type=float, default=3e1,
        help='SGD/rmsprop lr for sigma')
group.add_argument('--lr_sigma_final', type=float, default=5e-2)
group.add_argument('--lr_sigma_decay_steps', type=int, default=250000)
group.add_argument('--lr_sigma_delay_steps', type=int, default=
                    15000,
                   help="Reverse cosine steps (0 means disable)")
group.add_argument('--lr_sigma_delay_mult', type=float, default=1e-2)


group.add_argument('--sh_optim', choices=['sgd', 'rmsprop'], default='rmsprop', help="SH optimizer")
group.add_argument('--lr_sh', type=float, default=
                    1e-2,
                   help='SGD/rmsprop lr for SH')
group.add_argument('--lr_sh_final', type=float,
                      default=
                    5e-5
                    )
group.add_argument('--lr_sh_decay_steps', type=int, default=250000)
group.add_argument('--lr_sh_delay_steps', type=int, default=0, help="Reverse cosine steps (0 means disable)")
group.add_argument('--lr_sh_delay_mult', type=float, default=1e-2)
group.add_argument('--lr_sh_upscale_factor', type=float, default=1.0)

group.add_argument('--bg_optim', choices=['sgd', 'rmsprop'], default='rmsprop', help="Density optimizer")
group.add_argument('--lr_sigma_bg', type=float, default=3e-2, help='SGD/rmsprop lr for background')
group.add_argument('--lr_color_bg', type=float, default=1e-2, help='SGD/rmsprop lr for background')

group.add_argument('--basis_optim', choices=['sgd', 'rmsprop'], default='rmsprop', help="Learned basis optimizer")
group.add_argument('--lr_basis', type=float, default=#2e6,
                      1e-3,
                   help='SGD/rmsprop lr for SH')
group.add_argument('--lr_basis_final', type=float,
                      default=
                      1e-4
                    )
group.add_argument('--lr_basis_decay_steps', type=int, default=250000)
group.add_argument('--lr_basis_delay_steps', type=int, default=0,#15000,
                   help="Reverse cosine steps (0 means disable)")
group.add_argument('--lr_basis_begin_step', type=int, default=0)#4 * 12800)
group.add_argument('--lr_basis_delay_mult', type=float, default=1e-2)

group.add_argument('--rms_beta', type=float, default=0.95, help="RMSProp exponential averaging factor")

group.add_argument('--n_iters', type=int, default=20 * 12800, help='number of iters to optimize for')
group.add_argument('--print_every', type=int, default=20, help='print every')
group.add_argument('--save_every', type=int, default=2, #5,
                   help='save every x epochs')
group.add_argument('--eval_every', type=int, default=1,
                   help='evaluate every x epochs')

group.add_argument('--init_sigma', type=float,
                   default=0.1,
                   help='initialization sigma')



group = parser.add_argument_group("misc experiments")
group.add_argument('--perm', action='store_true', default=True,
                    help='sample by permutation of rays (true epoch) instead of '
                         'uniformly random rays')
group.add_argument('--weight_thresh', type=float,
                    default=0.0005,
                   help='Resample (upsample to 512) weight threshold')

group.add_argument('--tune_mode', action='store_true', default=False,
                   help='hypertuning mode (do not save, for speed)')



group = parser.add_argument_group("losses")
# Foreground TV
group.add_argument('--lambda_tv', type=float, default=1e-5)
group.add_argument('--tv_sparsity', type=float, default=0.01)
group.add_argument('--tv_logalpha', action='store_true', default=False,
                   help='Use log(1-exp(-delta * sigma)) as in neural volumes')

group.add_argument('--lambda_tv_sh', type=float, default=1e-3)
group.add_argument('--tv_sh_sparsity', type=float, default=0.01)

group.add_argument('--lambda_l2_sh', type=float, default=0.0)#1e-4)
# End Foreground TV


# Background TV
group.add_argument('--lambda_tv_background_sigma', type=float, default=1e-4)
group.add_argument('--lambda_tv_background_color', type=float, default=1e-4)

group.add_argument('--tv_background_sparsity', type=float, default=0.01)
# End Background TV

group.add_argument('--lambda_tv_basis', type=float, default=0.0, help='Learned basis total variation loss')

group.add_argument('--weight_decay_sigma', type=float, default=1.0)
group.add_argument('--weight_decay_sh', type=float, default=1.0)

group.add_argument('--lr_decay', action='store_true', default=True)

args = parser.parse_args()
config_util.maybe_merge_config_file(args)

assert args.lr_sigma_final <= args.lr_sigma, "lr_sigma must be >= lr_sigma_final"
assert args.lr_sh_final <= args.lr_sh, "lr_sh must be >= lr_sh_final"
assert args.lr_basis_final <= args.lr_basis, "lr_basis must be >= lr_basis_final"

os.makedirs(args.train_dir, exist_ok=True)
summary_writer = SummaryWriter(args.train_dir)

reso_list = json.loads(args.reso)
reso_id = 0

with open(path.join(args.train_dir, 'args.json'), 'w') as f:
    json.dump(args.__dict__, f, indent=2)
    # Changed name to prevent errors
    shutil.copyfile(__file__, path.join(args.train_dir, 'opt_frozen.py'))

torch.manual_seed(20200823)
np.random.seed(20200823)

factor = 1

dset = datasets[args.dataset_type](
               args.data_dir,
               split="train",
               device=device,
               permutation=args.perm,
               factor=factor,
               **config_util.build_data_options(args))
dset.shuffle_rays()
dset_test = datasets[args.dataset_type](
        args.data_dir, split="test", **config_util.build_data_options(args))

grid = svox2.SparseGrid(reso=reso_list[reso_id],
                        center=dset.scene_center,
                        radius=dset.scene_radius,
                        use_sphere_bound=dset.use_sphere_bound,
                        basis_dim=args.sh_dim,
                        use_z_order=True,
                        device=device,
                        basis_reso=args.basis_reso,
                        basis_type=svox2.__dict__['BASIS_TYPE_' + args.basis_type.upper()],
                        mlp_posenc_size=args.mlp_posenc_size,
                        mlp_width=args.mlp_width,
                        background_nlayers=args.background_nlayers,
                        background_reso=args.background_reso)

grid.opt.last_sample_opaque = dset.last_sample_opaque

# DC -> gray; mind the SH scaling!
grid.sh_data.data[:] = 0.0
grid.density_data.data[:] = args.init_sigma

if grid.use_background:
    grid.background_cubemap.data[..., -1] = args.init_sigma
    grid.background_cubemap.data[..., :-1] = 0.0 / svox2.utils.SH_C0

#  grid.sh_data.data[:, 0] = 4.0
#  osh = grid.density_data.data.shape
#  den = grid.density_data.data.view(grid.links.shape)
#  #  den[:] = 0.00
#  #  den[:, :256, :] = 1e9
#  #  den[:, :, 0] = 1e9
#  grid.density_data.data = den.view(osh)

optim_basis_mlp = None

if grid.basis_type == svox2.BASIS_TYPE_3D_TEXTURE:
    #  grid.reinit_learned_bases(init_type='sh')
    #  grid.reinit_learned_bases(init_type='fourier')
    #  grid.reinit_learned_bases(init_type='sg', upper_hemi=True)
    grid.basis_data.data.normal_(mean=0.28209479177387814, std=0.001)

elif grid.basis_type == svox2.BASIS_TYPE_MLP:
    # MLP!
    optim_basis_mlp = torch.optim.Adam(
                    grid.basis_mlp.parameters(),
                    lr=args.lr_basis
                )


grid.requires_grad_(True)
grid.opt.step_size = args.step_size
grid.opt.sigma_thresh = args.sigma_thresh
grid.opt.stop_thresh = args.stop_thresh
grid.opt.background_brightness = args.background_brightness
grid.opt.background_msi_scale = args.background_msi_scale
grid.opt.backend = args.renderer_backend

gstep_id_base = 0

resample_cameras = [
        svox2.Camera(c2w.to(device=device),
                     dset.intrins.fx,
                     dset.intrins.fy,
                     dset.intrins.cx,
                     dset.intrins.cy,
                     width=dset.w,
                     height=dset.h,
                     ndc_coeffs=dset.ndc_coeffs) for c2w in dset.c2w
    ]
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

epoch_id = -1
while True:
    epoch_id += 1
    epoch_size = dset.rays.origins.size(0)
    batches_per_epoch = (epoch_size-1)//args.batch_size+1
    # Test
    def eval_step():
        # Put in a function to avoid memory leak
        print('Eval step')
        with torch.no_grad():
            stats_test = {'psnr' : 0.0, 'mse' : 0.0}

            # Standard set
            N_IMGS_TO_SAVE = min(5, dset_test.n_images)
            N_IMGS_TO_EVAL = min(20 if epoch_id > 0 else 5, dset_test.n_images)
            img_eval_interval = dset_test.n_images // N_IMGS_TO_EVAL
            img_save_interval = (N_IMGS_TO_EVAL // N_IMGS_TO_SAVE)
            img_ids = range(0, dset_test.n_images, img_eval_interval)

            # Special 'very hard' specular + fuzz set
            #  img_ids = [2, 5, 7, 9, 21,
            #             44, 45, 47, 49, 56,
            #             80, 88, 99, 115, 120,
            #             154]
            img_save_interval = 1

            n_images_gen = 0
            for i, img_id in tqdm(enumerate(img_ids), total=len(img_ids)):
                c2w = dset_test.c2w[img_id].to(device=device)
                cam = svox2.Camera(c2w,
                                   dset_test.intrins.fx,
                                   dset_test.intrins.fy,
                                   dset_test.intrins.cx,
                                   dset_test.intrins.cy,
                                   width=dset_test.w,
                                   height=dset_test.h,
                                   ndc_coeffs=dset_test.ndc_coeffs)
                rgb_pred_test = grid.volume_render_image(cam, use_kernel=True)
                rgb_gt_test = dset_test.gt[img_id].to(device=device)
                all_mses = ((rgb_gt_test - rgb_pred_test) ** 2).cpu()
                if i % img_save_interval == 0:
                    img_pred = rgb_pred_test.cpu()
                    img_pred.clamp_max_(1.0)
                    summary_writer.add_image(f'test/image_{img_id:04d}',
                            img_pred, global_step=gstep_id_base, dataformats='HWC')
                    mse_img = all_mses / all_mses.max()
                    summary_writer.add_image(f'test/mse_map_{img_id:04d}',
                            mse_img, global_step=gstep_id_base, dataformats='HWC')

                rgb_pred_test = rgb_gt_test = None
                mse_num : float = all_mses.mean().item()
                psnr = -10.0 * math.log10(mse_num)
                if math.isnan(psnr):
                    print('NAN PSNR', i, img_id)
                stats_test['mse'] += mse_num
                stats_test['psnr'] += psnr
                n_images_gen += 1

            if grid.basis_type == svox2.BASIS_TYPE_3D_TEXTURE or \
               grid.basis_type == svox2.BASIS_TYPE_MLP:
                 # Add spherical map visualization
                EQ_RESO = 256
                eq_dirs = generate_dirs_equirect(EQ_RESO * 2, EQ_RESO)
                eq_dirs = torch.from_numpy(eq_dirs).to(device=device).view(-1, 3)

                if grid.basis_type == svox2.BASIS_TYPE_MLP:
                    sphfuncs = grid._eval_basis_mlp(eq_dirs)
                else:
                    sphfuncs = grid._eval_learned_bases(eq_dirs)
                sphfuncs = sphfuncs.view(EQ_RESO, EQ_RESO*2, -1).permute([2, 0, 1]).cpu().numpy()

                stats = [(sphfunc.min(), sphfunc.mean(), sphfunc.max())
                        for sphfunc in sphfuncs]
                sphfuncs_cmapped = [viridis_cmap(sphfunc) for sphfunc in sphfuncs]
                for im, (minv, meanv, maxv) in zip(sphfuncs_cmapped, stats):
                    cv2.putText(im, f"{minv=:.4f} {meanv=:.4f} {maxv=:.4f}", (10, 20),
                                0, 0.5, [255, 0, 0])
                sphfuncs_cmapped = np.concatenate(sphfuncs_cmapped, axis=0)
                summary_writer.add_image(f'test/spheric',
                        sphfuncs_cmapped, global_step=gstep_id_base, dataformats='HWC')
                # END add spherical map visualization

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
            lr_basis = lr_basis_func(gstep_id - args.lr_basis_begin_step) * lr_basis_factor
            if not args.lr_decay:
                lr_sigma = args.lr_sigma * lr_sigma_factor
                lr_sh = args.lr_sh * lr_sh_factor
                lr_basis = args.lr_basis * lr_basis_factor

            batch_end = min(batch_begin + args.batch_size, epoch_size)
            batch_origins = dset.rays.origins[batch_begin: batch_end]
            batch_dirs = dset.rays.dirs[batch_begin: batch_end]
            #  batch_dirs.normal_() # FIXME
            #  batch_dirs /= batch_dirs.norm(dim=-1).unsqueeze(-1) # FIXME
            rgb_gt = dset.rays.gt[batch_begin: batch_end]
            #  rgb_gt[:] = 0 # FIXME
            rays = svox2.Rays(batch_origins, batch_dirs)

            #  with Timing("volrend_fused"):
            rgb_pred = grid.volume_render_fused(rays, rgb_gt)

            #  with Timing("loss_comp"):
            mse = F.mse_loss(rgb_gt, rgb_pred)

            # Stats
            mse_num : float = mse.detach().item()
            psnr = -10.0 * math.log10(mse_num)
            stats['mse'] += mse_num
            stats['psnr'] += psnr
            stats['invsqr_mse'] += 1.0 / mse_num ** 2

            if (iter_id + 1) % args.print_every == 0:
                # Print averaged stats
                pbar.set_description(f'epoch {epoch_id} psnr={psnr:.2f}')
                for stat_name in stats:
                    stat_val = stats[stat_name] / args.print_every
                    summary_writer.add_scalar(stat_name, stat_val, global_step=gstep_id)
                    stats[stat_name] = 0.0
                #  if args.lambda_tv > 0.0:
                #      with torch.no_grad():
                #          tv = grid.tv(logalpha=args.tv_logalpha, ndc_coeffs=dset.ndc_coeffs)
                #      summary_writer.add_scalar("loss_tv", tv, global_step=gstep_id)
                #  if args.lambda_tv_sh > 0.0:
                #      with torch.no_grad():
                #          tv_sh = grid.tv_color()
                #      summary_writer.add_scalar("loss_tv_sh", tv_sh, global_step=gstep_id)
                #  with torch.no_grad():
                #      tv_basis = grid.tv_basis()
                #  summary_writer.add_scalar("loss_tv_basis", tv_basis, global_step=gstep_id)
                summary_writer.add_scalar("lr_sh", lr_sh, global_step=gstep_id)
                summary_writer.add_scalar("lr_sigma", lr_sigma, global_step=gstep_id)
                summary_writer.add_scalar("lr_basis", lr_basis, global_step=gstep_id)

                if args.weight_decay_sh < 1.0:
                    grid.sh_data.data *= args.weight_decay_sigma
                if args.weight_decay_sigma < 1.0:
                    grid.density_data.data *= args.weight_decay_sh

            # Apply TV/Sparsity regularizers
            if args.lambda_tv > 0.0:
                grid.inplace_tv_grad(grid.density_data.grad,
                        scaling=args.lambda_tv,
                        sparse_frac=args.tv_sparsity,
                        logalpha=args.tv_logalpha,
                        ndc_coeffs=dset.ndc_coeffs)
            if args.lambda_tv_sh > 0.0:
                grid.inplace_tv_color_grad(grid.sh_data.grad,
                        scaling=args.lambda_tv_sh,
                        sparse_frac=args.tv_sh_sparsity,
                        ndc_coeffs=dset.ndc_coeffs)
            if args.lambda_l2_sh > 0.0:
                grid.inplace_l2_color_grad(grid.sh_data.grad,
                        scaling=args.lambda_l2_sh)
            if args.lambda_tv_background_sigma > 0.0 or args.lambda_tv_background_color > 0.0:
                grid.inplace_tv_background_grad(grid.background_cubemap.grad,
                        scaling=args.lambda_tv_background_color,
                        scaling_density=args.lambda_tv_background_sigma,
                        sparse_frac=args.tv_background_sparsity)
            if args.lambda_tv_basis > 0.0:
                tv_basis = grid.tv_basis()
                loss_tv_basis = tv_basis * args.lambda_tv_basis
                loss_tv_basis.backward()
            #  print('nz density', torch.count_nonzero(grid.sparse_grad_indexer).item(),
            #        ' sh', torch.count_nonzero(grid.sparse_sh_grad_indexer).item())

            # Manual SGD/rmsprop step
            grid.optim_density_step(lr_sigma, beta=args.rms_beta, optim=args.sigma_optim)
            grid.optim_sh_step(lr_sh, beta=args.rms_beta, optim=args.sh_optim)
            if grid.use_background:
                grid.optim_background_step(args.lr_sigma_bg, args.lr_color_bg, beta=args.rms_beta, optim=args.bg_optim)
            if gstep_id >= args.lr_basis_begin_step:
                if grid.basis_type == svox2.BASIS_TYPE_3D_TEXTURE:
                    grid.optim_basis_step(lr_basis, beta=args.rms_beta, optim=args.basis_optim)
                elif grid.basis_type == svox2.BASIS_TYPE_MLP:
                    optim_basis_mlp.step()
                    optim_basis_mlp.zero_grad()

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
        if reso_id < len(reso_list) - 1:
            print('* Upsampling from', reso_list[reso_id], 'to', reso_list[reso_id + 1])
            reso_id += 1
            use_sparsify = True
            grid.resample(reso=reso_list[reso_id],
                    #  sigma_thresh=args.sigma_thresh if use_sparsify else 0.0,
                    weight_thresh=args.weight_thresh if use_sparsify else 0.0,
                    dilate=2, #use_sparsify,
                    cameras=resample_cameras)
            if args.lr_sh_upscale_factor > 1:
                lr_sh_factor *= args.lr_sh_upscale_factor
                print('Increased lr to (sigma:)', args.lr_sigma, '(sh:)', args.lr_sh)

        if factor > 1 and reso_id < len(reso_list) - 1:
            print('* Using higher resolution images due to large grid; new factor', factor)
            factor //= 2
            dset.gen_rays(factor=factor)
            dset.shuffle_rays()

    if gstep_id_base >= args.n_iters:
        print('* Final eval and save')
        eval_step()
        if not args.tune_mode:
            grid.save(ckpt_path)
        break
