# Copyright 2021 Alex Yu

# First, install svox2
# Then, python opt.py <path_to>/nerf_synthetic/<scene> -t ckpt/<some_name>
# or use launching script:   sh launch.sh <EXP_NAME> <GPU> <DATA_DIR>
import torch
import torch.cuda
import torch.optim
import torch.nn.functional as F
import svox2

# import sys
# from os import path
# sys.path.append( path.dirname( path.dirname( path.abspath(__file__) ) ) )
# import svox2

import json
import imageio
import os
from os import path
import shutil
import gc
import numpy as np
import math
import cv2
from util.dataset import datasets
from util.util import Timing, get_expon_lr_func, generate_dirs_equirect, viridis_cmap
from util import config_util
import gin
import ast

from warnings import warn
from datetime import datetime
from torch.utils.tensorboard import SummaryWriter

from tqdm import tqdm
from typing import NamedTuple, Optional, Union

device = "cuda" if torch.cuda.is_available() else "cpu"

# parser = argparse.ArgumentParser()
# parser.add_argument('--train_dir', '-t', type=str, default='ckpt',
#                     help='checkpoint and logging directory')
# parser.add_argument('--gin-config', '-c',
#                         type=str,
#                         default=None,
#                         help="Config yaml file (will override args)")


# FLAGS = parser.parse_args()
# args = config_util.setup_train_conf(FLAGS)

args = config_util.setup_train_conf()
USE_KERNEL = not args.nokernel

assert args.lr_sigma_final <= args.lr_sigma, "lr_sigma must be >= lr_sigma_final"
assert args.lr_sh_final <= args.lr_sh, "lr_sh must be >= lr_sh_final"
assert args.lr_basis_final <= args.lr_basis, "lr_basis must be >= lr_basis_final"

os.makedirs(args.train_dir, exist_ok=True)
summary_writer = SummaryWriter(args.train_dir)

# reso_list = json.loads(args.reso)
reso_list = [ast.literal_eval(args.reso[i]) for i in range(len(args.reso))]
reso_id = 0

# with open(path.join(args.train_dir, 'args.json'), 'w') as f:
#     json.dump(args.__dict__, f, indent=2)
    # # Changed name to prevent errors
    # shutil.copyfile(__file__, path.join(args.train_dir, 'opt_frozen.py'))

with open(path.join(args.train_dir, 'args.yaml'), 'w') as file:
    for arg in sorted(vars(args)):
        attr = getattr(args, arg)
        file.write('{} = {}\n'.format(arg, attr))
shutil.copyfile(args.config, path.join(args.train_dir, 'config.yaml'))

torch.manual_seed(20200823)
np.random.seed(20200823)

factor = 1
dset = datasets[args.dataset_type](
               args.data_dir,
               split="train",
               device=device,
               factor=factor,
               n_images=args.n_train,
               **config_util.build_data_options(args))

if args.background_nlayers > 0 and not dset.should_use_background:
    warn('Using a background model for dataset type ' + str(type(dset)) + ' which typically does not use background')

dset_test = datasets[args.dataset_type](
        args.data_dir, split="test", **config_util.build_data_options(args))

global_start_time = datetime.now()


ckpt_npz = path.join(args.train_dir, 'ckpt.npz')

if path.isfile(ckpt_npz) and args.load_ckpt:
    print(f'Resume from ckpt at {ckpt_npz}!')
    grid = svox2.SparseGrid.load(ckpt_npz, device=device)
    assert args.renderer_backend == grid.backend, "Loaded ckpt incompatible with given configs"
    gstep_id_base = grid.step_id
else:
    grid = svox2.SparseGrid(reso=reso_list[reso_id],
                            center=dset.scene_center,
                            radius=dset.scene_radius,
                            use_sphere_bound=dset.use_sphere_bound and not args.nosphereinit,
                            basis_dim=args.sh_dim,
                            use_z_order=True,
                            device=device,
                            basis_reso=args.basis_reso,
                            basis_type=svox2.__dict__['BASIS_TYPE_' + args.basis_type.upper()],
                            mlp_posenc_size=args.mlp_posenc_size,
                            mlp_width=args.mlp_width,
                            background_nlayers=args.background_nlayers,
                            background_reso=args.background_reso,
                            backend=args.renderer_backend,
                            geometry_init=args.geometry_init)

    # DC -> gray; mind the SH scaling!
    grid.sh_data.data[:] = 0.0
    grid.density_data.data[:] = 0.0 if args.lr_fg_begin_step > 0 else args.init_sigma

    if grid.use_background:
        grid.background_data.data[..., -1] = args.init_sigma_bg
        #  grid.background_data.data[..., :-1] = 0.5 / svox2.utils.SH_C0

    #  grid.sh_data.data[:, 0] = 4.0
    #  osh = grid.density_data.data.shape
    #  den = grid.density_data.data.view(grid.links.shape)
    #  #  den[:] = 0.00
    #  #  den[:, :256, :] = 1e9
    #  #  den[:, :, 0] = 1e9
    #  grid.density_data.data = den.view(osh)

    gstep_id_base = 0

optim_basis_mlp = None

if grid.basis_type == svox2.BASIS_TYPE_3D_TEXTURE and not (path.isfile(ckpt_npz) and args.load_ckpt):
    # do not reinit if resuming from ckpt
    grid.reinit_learned_bases(init_type='sh')
    #  grid.reinit_learned_bases(init_type='fourier')
    #  grid.reinit_learned_bases(init_type='sg', upper_hemi=True)
    #  grid.basis_data.data.normal_(mean=0.28209479177387814, std=0.001)

elif grid.basis_type == svox2.BASIS_TYPE_MLP:
    # MLP!
    optim_basis_mlp = torch.optim.Adam(
                    grid.basis_mlp.parameters(),
                    lr=args.lr_basis
                )


grid.requires_grad_(True)
config_util.setup_render_opts(grid.opt, args)
print('Render options', grid.opt)



resample_cameras = [
        svox2.Camera(c2w.to(device=device),
                     dset.intrins.get('fx', i),
                     dset.intrins.get('fy', i),
                     dset.intrins.get('cx', i),
                     dset.intrins.get('cy', i),
                     width=dset.get_image_size(i)[1],
                     height=dset.get_image_size(i)[0],
                     ndc_coeffs=dset.ndc_coeffs) for i, c2w in enumerate(dset.c2w)
    ]
ckpt_path = path.join(args.train_dir, 'ckpt.npz')

lr_sigma_func = get_expon_lr_func(args.lr_sigma, args.lr_sigma_final, args.lr_sigma_delay_steps,
                                  args.lr_sigma_delay_mult, args.lr_sigma_decay_steps)
lr_geo_func = get_expon_lr_func(args.lr_geo, args.lr_geo_final, args.lr_geo_delay_steps,
                                  args.lr_geo_delay_mult, args.lr_geo_decay_steps)
lr_sh_func = get_expon_lr_func(args.lr_sh, args.lr_sh_final, args.lr_sh_delay_steps,
                               args.lr_sh_delay_mult, args.lr_sh_decay_steps)
lr_basis_func = get_expon_lr_func(args.lr_basis, args.lr_basis_final, args.lr_basis_delay_steps,
                               args.lr_basis_delay_mult, args.lr_basis_decay_steps)
lr_sigma_bg_func = get_expon_lr_func(args.lr_sigma_bg, args.lr_sigma_bg_final, args.lr_sigma_bg_delay_steps,
                               args.lr_sigma_bg_delay_mult, args.lr_sigma_bg_decay_steps)
lr_color_bg_func = get_expon_lr_func(args.lr_color_bg, args.lr_color_bg_final, args.lr_color_bg_delay_steps,
                               args.lr_color_bg_delay_mult, args.lr_color_bg_decay_steps)
lr_sigma_factor = 1.0
lr_geo_factor = 1.0
lr_sh_factor = 1.0
lr_basis_factor = 1.0

last_upsamp_step = args.init_iters

if args.enable_random:
    warn("Randomness is enabled for training (normal for LLFF & scenes with background)")

epoch_id = -1
while True:
    dset.shuffle_rays()
    epoch_id += 1
    epoch_size = dset.rays.origins.size(0)
    batches_per_epoch = (epoch_size-1)//args.batch_size+1
    # Test
    def eval_step(step_id=gstep_id_base):
        # Put in a function to avoid memory leak
        print('Eval step')
        with torch.no_grad():
            stats_test = {'psnr' : 0.0, 'mse' : 0.0}

            # Standard set
            # N_IMGS_TO_EVAL = min(20 if epoch_id > 0 else 5, dset_test.n_images)
            N_IMGS_TO_EVAL = args.n_eval
            N_IMGS_TO_SAVE = N_IMGS_TO_EVAL # if not args.tune_mode else 1
            img_eval_interval = dset_test.n_images // N_IMGS_TO_EVAL
            img_save_interval = (N_IMGS_TO_EVAL // N_IMGS_TO_SAVE)
            img_ids = range(0, dset_test.n_images, img_eval_interval)

            # img_ids = [120]

            # Special 'very hard' specular + fuzz set
            #  img_ids = [2, 5, 7, 9, 21,
            #             44, 45, 47, 49, 56,
            #             80, 88, 99, 115, 120,
            #             154]
            #  img_save_interval = 1

            n_images_gen = 0
            for i, img_id in tqdm(enumerate(img_ids), total=len(img_ids)):
                c2w = dset_test.c2w[img_id].to(device=device)
                cam = svox2.Camera(c2w,
                                   dset_test.intrins.get('fx', img_id),
                                   dset_test.intrins.get('fy', img_id),
                                   dset_test.intrins.get('cx', img_id),
                                   dset_test.intrins.get('cy', img_id),
                                   width=dset_test.get_image_size(img_id)[1],
                                   height=dset_test.get_image_size(img_id)[0],
                                   ndc_coeffs=dset_test.ndc_coeffs)
                # cam = svox2.Camera(c2w,
                #                    dset_test.intrins.get('fx', img_id) / 8,
                #                    dset_test.intrins.get('fy', img_id) / 8,
                #                    dset_test.intrins.get('cx', img_id) / 8,
                #                    dset_test.intrins.get('cy', img_id) / 8,
                #                    width=100,
                #                    height=100,
                #                    ndc_coeffs=dset_test.ndc_coeffs)
                rgb_pred_test = grid.volume_render_image(cam, use_kernel=USE_KERNEL, batch_size=10000)
                rgb_gt_test = dset_test.gt[img_id].to(device=device)
                all_mses = ((rgb_gt_test - rgb_pred_test) ** 2).cpu()
                if i % img_save_interval == 0:
                    img_pred = rgb_pred_test.cpu()
                    img_pred.clamp_max_(1.0)
                    summary_writer.add_image(f'test/image_{img_id:04d}',
                            img_pred, global_step=step_id, dataformats='HWC')
                    if args.log_mse_image:
                        mse_img = all_mses / all_mses.max()
                        summary_writer.add_image(f'test/mse_map_{img_id:04d}',
                                mse_img, global_step=step_id, dataformats='HWC')
                    if args.log_depth_map:
                        depth_img = grid.volume_render_depth_image(cam,
                                    args.log_depth_map_use_thresh if
                                    args.log_depth_map_use_thresh else None,
                                    batch_size=10000
                                )
                        depth_img = viridis_cmap(depth_img.cpu())
                        summary_writer.add_image(f'test/depth_map_{img_id:04d}',
                                depth_img,
                                global_step=step_id, dataformats='HWC')

                rgb_pred_test = rgb_gt_test = None
                mse_num : float = all_mses.mean().item()
                psnr = -10.0 * math.log10(mse_num)
                if math.isnan(psnr):
                    print('NAN PSNR', i, img_id, mse_num)
                    assert False 
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
                        sphfuncs_cmapped, global_step=step_id, dataformats='HWC')
                # END add spherical map visualization

            stats_test['mse'] /= n_images_gen
            stats_test['psnr'] /= n_images_gen
            for stat_name in stats_test:
                summary_writer.add_scalar('test/' + stat_name,
                        stats_test[stat_name], global_step=step_id)
            summary_writer.add_scalar('epoch_id', float(epoch_id), global_step=step_id)
            print('eval stats:', stats_test)

            # log train imgs
            img_eval_interval = dset.n_images // N_IMGS_TO_EVAL
            img_ids = range(0, dset.n_images, img_eval_interval)
            for i, img_id in tqdm(enumerate(img_ids), total=len(img_ids)):
                c2w = dset.c2w[img_id].to(device=device)
                cam = svox2.Camera(c2w,
                                   dset.intrins.get('fx', img_id),
                                   dset.intrins.get('fy', img_id),
                                   dset.intrins.get('cx', img_id),
                                   dset.intrins.get('cy', img_id),
                                   width=dset.get_image_size(img_id)[1],
                                   height=dset.get_image_size(img_id)[0],
                                   ndc_coeffs=dset.ndc_coeffs)
                rgb_pred_test = grid.volume_render_image(cam, use_kernel=USE_KERNEL, batch_size=10000)
                rgb_gt_test = dset.gt[img_id].to(device=device)
                all_mses = ((rgb_gt_test - rgb_pred_test) ** 2).cpu()
                if i % img_save_interval == 0:
                    img_pred = rgb_pred_test.cpu()
                    img_pred.clamp_max_(1.0)
                    summary_writer.add_image(f'train/image_{img_id:04d}',
                            img_pred, global_step=step_id, dataformats='HWC')
                    if args.log_mse_image:
                        mse_img = all_mses / all_mses.max()
                        summary_writer.add_image(f'train/mse_map_{img_id:04d}',
                                mse_img, global_step=step_id, dataformats='HWC')
                    if args.log_depth_map:
                        depth_img = grid.volume_render_depth_image(cam,
                                    args.log_depth_map_use_thresh if
                                    args.log_depth_map_use_thresh else None,
                                    batch_size=10000
                                )
                        depth_img = viridis_cmap(depth_img.cpu())
                        summary_writer.add_image(f'train/depth_map_{img_id:04d}',
                                depth_img,
                                global_step=step_id, dataformats='HWC')

    # if epoch_id % max(factor, args.eval_every) == 0 and (epoch_id > 0 or not args.tune_mode):
    # if epoch_id % max(factor, args.eval_every) == 0 and (epoch_id > 0):
    #     # NOTE: we do an eval sanity check, if not in tune_mode
    #     eval_step()
    #     gc.collect()

    def train_step():
        print('Train step')
        pbar = tqdm(enumerate(range(gstep_id_base, gstep_id_base+epoch_size, args.batch_size)), total=batches_per_epoch)
        stats = {"mse" : 0.0, "psnr" : 0.0, "invsqr_mse" : 0.0}
        for iter_id, batch_begin in pbar:
            gstep_id = iter_id + gstep_id_base
            if args.lr_fg_begin_step > 0 and gstep_id == args.lr_fg_begin_step:
                grid.density_data.data[:] = args.init_sigma
            lr_sigma = lr_sigma_func(gstep_id) * lr_sigma_factor
            lr_geo = lr_geo_func(gstep_id) * lr_geo_factor
            lr_sh = lr_sh_func(gstep_id) * lr_sh_factor
            lr_basis = lr_basis_func(gstep_id - args.lr_basis_begin_step) * lr_basis_factor
            lr_sigma_bg = lr_sigma_bg_func(gstep_id - args.lr_basis_begin_step) * lr_basis_factor
            lr_color_bg = lr_color_bg_func(gstep_id - args.lr_basis_begin_step) * lr_basis_factor
            if not args.lr_decay:
                lr_sigma = args.lr_sigma * lr_sigma_factor
                lr_geo = args.lr_geo * lr_geo_factor
                lr_sh = args.lr_sh * lr_sh_factor
                lr_basis = args.lr_basis * lr_basis_factor

            batch_end = min(batch_begin + args.batch_size, epoch_size)
            batch_origins = dset.rays.origins[batch_begin: batch_end]
            batch_dirs = dset.rays.dirs[batch_begin: batch_end]
            rgb_gt = dset.rays.gt[batch_begin: batch_end]
            rays = svox2.Rays(batch_origins, batch_dirs)

            #  with Timing("volrend_fused"):
            if not USE_KERNEL:
                if args.renderer_backend == 'sdf':
                    rgb_pred = grid._sdf_render_gradcheck_lerp(rays, rgb_gt,
                            beta_loss=args.lambda_beta,
                            sparsity_loss=args.lambda_sparsity,
                            randomize=args.enable_random)
                elif args.renderer_backend == 'plane':
                    rgb_pred = grid._plane_render_gradcheck_lerp(rays, rgb_gt,
                            beta_loss=args.lambda_beta,
                            sparsity_loss=args.lambda_sparsity,
                            randomize=args.enable_random)
                else:
                    raise NotImplementedError
            else:
                rgb_pred = grid.volume_render_fused(rays, rgb_gt,
                        beta_loss=args.lambda_beta,
                        sparsity_loss=args.lambda_sparsity,
                        randomize=args.enable_random)

            #  with Timing("loss_comp"):
            mse = F.mse_loss(rgb_gt, rgb_pred)

            if not USE_KERNEL:
                mse.backward()

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
                #      tv_basis = grid.tv_basis() #  summary_writer.add_scalar("loss_tv_basis", tv_basis, global_step=gstep_id)
                summary_writer.add_scalar("lr_sh", lr_sh, global_step=gstep_id)
                summary_writer.add_scalar("lr_sigma", lr_sigma, global_step=gstep_id)
                summary_writer.add_scalar("lr_geo", lr_geo, global_step=gstep_id)
                if grid.basis_type == svox2.BASIS_TYPE_3D_TEXTURE:
                    summary_writer.add_scalar("lr_basis", lr_basis, global_step=gstep_id)
                if grid.use_background:
                    summary_writer.add_scalar("lr_sigma_bg", lr_sigma_bg, global_step=gstep_id)
                    summary_writer.add_scalar("lr_color_bg", lr_color_bg, global_step=gstep_id)

                if args.weight_decay_sh < 1.0:
                    grid.sh_data.data *= args.weight_decay_sigma
                if args.weight_decay_sigma < 1.0:
                    grid.density_data.data *= args.weight_decay_sh

            #  # For outputting the % sparsity of the gradient
            #  indexer = grid.sparse_sh_grad_indexer
            #  if indexer is not None:
            #      if indexer.dtype == torch.bool:
            #          nz = torch.count_nonzero(indexer)
            #      else:
            #          nz = indexer.size()
            #      with open(os.path.join(args.train_dir, 'grad_sparsity.txt'), 'a') as sparsity_file:
            #          sparsity_file.write(f"{gstep_id} {nz}\n")

            # Apply TV/Sparsity regularizers
            if args.lambda_tv > 0.0:
                #  with Timing("tv_inpl"):
                grid.inplace_tv_grad(grid.density_data.grad,
                        scaling=args.lambda_tv,
                        sparse_frac=args.tv_sparsity,
                        logalpha=args.tv_logalpha,
                        ndc_coeffs=dset.ndc_coeffs,
                        contiguous=args.tv_contiguous)
            if args.lambda_tv_sh > 0.0:
                #  with Timing("tv_color_inpl"):
                grid.inplace_tv_color_grad(grid.sh_data.grad,
                        scaling=args.lambda_tv_sh,
                        sparse_frac=args.tv_sh_sparsity,
                        ndc_coeffs=dset.ndc_coeffs,
                        contiguous=args.tv_contiguous)
            if args.lambda_tv_lumisphere > 0.0:
                grid.inplace_tv_lumisphere_grad(grid.sh_data.grad,
                        scaling=args.lambda_tv_lumisphere,
                        dir_factor=args.tv_lumisphere_dir_factor,
                        sparse_frac=args.tv_lumisphere_sparsity,
                        ndc_coeffs=dset.ndc_coeffs)
            if args.lambda_l2_sh > 0.0:
                grid.inplace_l2_color_grad(grid.sh_data.grad,
                        scaling=args.lambda_l2_sh)
            if grid.use_background and (args.lambda_tv_background_sigma > 0.0 or args.lambda_tv_background_color > 0.0):
                grid.inplace_tv_background_grad(grid.background_data.grad,
                        scaling=args.lambda_tv_background_color,
                        scaling_density=args.lambda_tv_background_sigma,
                        sparse_frac=args.tv_background_sparsity,
                        contiguous=args.tv_contiguous)
            if args.lambda_tv_basis > 0.0:
                tv_basis = grid.tv_basis()
                loss_tv_basis = tv_basis * args.lambda_tv_basis
                loss_tv_basis.backward()
            #  print('nz density', torch.count_nonzero(grid.sparse_grad_indexer).item(),
            #        ' sh', torch.count_nonzero(grid.sparse_sh_grad_indexer).item())

            # Manual SGD/rmsprop step
            if gstep_id >= args.lr_fg_begin_step:
                grid.optim_density_step(lr_sigma, beta=args.rms_beta, optim=args.sigma_optim)
                grid.optim_geo_step(lr_geo, beta=args.rms_beta, optim=args.geo_optim)
                grid.optim_sh_step(lr_sh, beta=args.rms_beta, optim=args.sh_optim)
            if grid.use_background:
                grid.optim_background_step(lr_sigma_bg, lr_color_bg, beta=args.rms_beta, optim=args.bg_optim)
            if gstep_id >= args.lr_basis_begin_step:
                if grid.basis_type == svox2.BASIS_TYPE_3D_TEXTURE:
                    grid.optim_basis_step(lr_basis, beta=args.rms_beta, optim=args.basis_optim)
                elif grid.basis_type == svox2.BASIS_TYPE_MLP:
                    optim_basis_mlp.step()
                    optim_basis_mlp.zero_grad()

            if (gstep_id % args.eval_every_iter) == 0: # and gstep_id > 0:
                eval_step(step_id=gstep_id)
                gc.collect()

            if gstep_id >= args.n_iters:
                print('* Final eval and save')
                global_stop_time = datetime.now()
                secs = (global_stop_time - global_start_time).total_seconds()
                timings_file = open(os.path.join(args.train_dir, 'time_mins.txt'), 'a')
                timings_file.write(f"{secs / 60}\n")
                if not args.tune_nosave:
                    grid.save(ckpt_path, step_id=gstep_id)
                exit(0)
            
            if args.save_every > 0 and gstep_id % args.save_every == 0 and not args.tune_mode:
                print('Saving', ckpt_path)
                grid.save(ckpt_path)

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
            if args.tv_early_only > 0:
                print('turning off TV regularization')
                args.lambda_tv = 0.0
                args.lambda_tv_sh = 0.0
            elif args.tv_decay != 1.0:
                args.lambda_tv *= args.tv_decay
                args.lambda_tv_sh *= args.tv_decay

            reso_id += 1
            use_sparsify = True
            z_reso = reso_list[reso_id] if isinstance(reso_list[reso_id], int) else reso_list[reso_id][2]
            grid.resample(reso=reso_list[reso_id],
                    sigma_thresh=args.density_thresh,
                    weight_thresh=args.weight_thresh / z_reso if use_sparsify else 0.0,
                    dilate=2, #use_sparsify,
                    cameras=resample_cameras if args.thresh_type == 'weight' else None,
                    max_elements=args.max_grid_elements)

            if grid.use_background and reso_id <= 1:
                grid.sparsify_background(args.background_density_thresh)

            if args.upsample_density_add:
                grid.density_data.data[:] += args.upsample_density_add

        if factor > 1 and reso_id < len(reso_list) - 1:
            print('* Using higher resolution images due to large grid; new factor', factor)
            factor //= 2
            dset.gen_rays(factor=factor)
            dset.shuffle_rays()

    if gstep_id_base >= args.n_iters:
        print('* Final eval and save')
        # eval_step()
        global_stop_time = datetime.now()
        secs = (global_stop_time - global_start_time).total_seconds()
        timings_file = open(os.path.join(args.train_dir, 'time_mins.txt'), 'a')
        timings_file.write(f"{secs / 60}\n")
        if not args.tune_nosave:
            grid.save(ckpt_path)
        break
