# Copyright 2021 Alex Yu
# Eval

import torch
import svox2
import math
import argparse
import numpy as np
import os
from os import path
from util.dataset import datasets
from util.util import Timing
from util import config_util

import imageio
from tqdm import tqdm

parser = argparse.ArgumentParser()
parser.add_argument('ckpt', type=str)

config_util.define_common_args(parser)

#  parser.add_argument('--eval_batch_size', type=int, default=200000, help='evaluation batch size')
parser.add_argument('--n_eval', '-n', type=int, default=200, help='images to evaluate (equal interval), at most evals every image')
parser.add_argument('--train', action='store_true', default=False, help='render train set')
parser.add_argument('--render_path',
                    action='store_true',
                    default=False,
                    help="Render path instead of test images (no metrics will be given)")
parser.add_argument('--nofg',
                    action='store_true',
                    default=False,
                    help="Do not render foreground (if using BG model)")
parser.add_argument('--nobg',
                    action='store_true',
                    default=False,
                    help="Do not render background (if using BG model)")
args = parser.parse_args()
config_util.maybe_merge_config_file(args)
device = 'cuda:0'

render_dir = path.join(path.dirname(args.ckpt),
            'train_renders' if args.train else 'test_renders')
if args.render_path:
    assert not args.train
    render_dir += '_path'

dset = datasets[args.dataset_type](args.data_dir, split="test_train" if args.train else "test",
                                    **config_util.build_data_options(args))

grid = svox2.SparseGrid.load(args.ckpt, device=device)
grid.opt.last_sample_opaque = dset.last_sample_opaque

if grid.use_background:
    if args.nobg:
        #  grid.background_cubemap.data = grid.background_cubemap.data.cuda()
        grid.background_cubemap.data[..., -1] = 0.0
        render_dir += '_nobg'
    if args.nofg:
        grid.density_data.data[:] = 0.0
        render_dir += '_nofg'

print('Writing to', render_dir)
os.makedirs(render_dir, exist_ok=True)

grid.opt.step_size = args.step_size
grid.opt.sigma_thresh = args.sigma_thresh
grid.opt.stop_thresh = args.stop_thresh
grid.opt.background_brightness = 1.0
grid.opt.backend = args.renderer_backend
grid.opt.background_msi_scale = args.background_msi_scale

with torch.no_grad():
    im_size = dset.h * dset.w
    n_images = dset.render_c2w.size(0) if args.render_path else dset.n_images
    img_eval_interval = max(n_images // args.n_eval, 1)
    avg_psnr = 0.0
    n_images_gen = 0
    cam = svox2.Camera(torch.tensor(0), dset.intrins.fx, dset.intrins.fy,
                       dset.intrins.cx, dset.intrins.cy,
                       dset.w, dset.h,
                       ndc_coeffs=dset.ndc_coeffs)
    c2ws = dset.render_c2w.to(device=device) if args.render_path else dset.c2w.to(device=device)
    for img_id in tqdm(range(0, n_images, img_eval_interval)):
        cam.c2w = c2ws[img_id]
        im = grid.volume_render_image(cam, use_kernel=True)
        im.clamp_(0.0, 1.0)
        if not args.render_path:
            im_gt = dset.gt[img_id].to(device=device)
            mse = (im - im_gt) ** 2
            mse_num : float = mse.mean().item()
            psnr = -10.0 * math.log10(mse_num)
            avg_psnr += psnr
            print(img_id, 'PSNR', psnr)
        #  all_rgbs = []
        #  all_mses = []
        #  for batch_begin in range(0, im_size, args.eval_batch_size):
        #      batch_end = min(batch_begin + args.eval_batch_size, im_size)
        #      batch_origins = dset.rays.origins[img_id][batch_begin: batch_end].to(device=device)
        #      batch_dirs = dset.rays.dirs[img_id][batch_begin: batch_end].to(device=device)
        #      rgb_gt_test = dset.rays.gt[img_id][batch_begin: batch_end].to(device=device)
        #
        #      rays = svox2.Rays(batch_origins, batch_dirs)
        #      rgb_pred_test = grid.volume_render(rays, use_kernel=True, randomize=True)
        #      rgb_pred_test.clamp_(0.0, 1.0)
        #      all_rgbs.append(rgb_pred_test.cpu())
        #      all_mses.append(((rgb_gt_test - rgb_pred_test) ** 2).cpu())
        img_path = path.join(render_dir, f'{img_id:04d}.png');
        im = im.cpu().numpy()
        if not args.render_path:
            im_gt = dset.gt[img_id].numpy()
            im = np.concatenate([im_gt, im], axis=1)
        imageio.imwrite(img_path, (im * 255).astype(np.uint8))
        im = None
        n_images_gen += 1
    avg_psnr /= n_images_gen
    print('average PSNR', avg_psnr)
    with open(path.join(render_dir, 'psnr.txt'), 'w') as f:
        f.write(str(avg_psnr))
