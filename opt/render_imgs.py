import torch
import svox2
import math
import argparse
import numpy as np
import os
from os import path
from util.dataset import Dataset
from util.util import Timing
import imageio
from tqdm import tqdm

parser = argparse.ArgumentParser()
parser.add_argument('ckpt', type=str)
parser.add_argument('data_dir', type=str)
parser.add_argument('--eval_batch_size', type=int, default=200000, help='evaluation batch size')
parser.add_argument('--n_eval', '-n', type=int, default=200, help='images to evaluate (equal interval), at most evals every image')
parser.add_argument('--train', action='store_true', default=False, help='render train set')
parser.add_argument('--scene_scale', type=float, default=5/6,
                   help='Scene scale; generally 2/3, can be 5/6 for lego')
args = parser.parse_args()
device = 'cuda:0'

render_dir = path.join(path.dirname(args.ckpt),
            'train_renders' if args.train else 'test_renders')
print('Writing to', render_dir)
os.makedirs(render_dir, exist_ok=True)

grid = svox2.SparseGrid.load(args.ckpt, device=device)

dset = Dataset(args.data_dir, split="test_train" if args.train else "test", scene_scale=args.scene_scale)

with torch.no_grad():
    im_size = dset.h * dset.w
    img_eval_interval = max(dset.n_images // args.n_eval, 1)
    avg_psnr = 0.0
    n_images_gen = 0
    for img_id in tqdm(range(0, dset.n_images, img_eval_interval)):
        all_rgbs = []
        all_mses = []
        for batch_begin in range(0, im_size, args.eval_batch_size):
            batch_end = min(batch_begin + args.eval_batch_size, im_size)
            batch_origins = dset.rays.origins[img_id][batch_begin: batch_end].to(device=device)
            batch_dirs = dset.rays.dirs[img_id][batch_begin: batch_end].to(device=device)
            rgb_gt_test = dset.rays.gt[img_id][batch_begin: batch_end].to(device=device)

            rays = svox2.Rays(batch_origins, batch_dirs)
            rgb_pred_test = grid.volume_render(rays, use_kernel=True, randomize=True)
            rgb_pred_test.clamp_(0.0, 1.0)
            all_rgbs.append(rgb_pred_test.cpu())
            all_mses.append(((rgb_gt_test - rgb_pred_test) ** 2).cpu())
        if len(all_rgbs):
            im = torch.cat(all_rgbs).view(dset.h, dset.w, all_rgbs[0].size(-1))
            img_path = path.join(render_dir, f'{img_id:04d}.png');
            imageio.imwrite(img_path, (im.cpu().numpy() * 255).astype(np.uint8))
            im = None
        mse_num : float = torch.cat(all_mses).mean().item()
        psnr = -10.0 * math.log10(mse_num)
        avg_psnr += psnr
        print(img_id, 'PSNR', psnr)
        n_images_gen += 1
    avg_psnr /= n_images_gen
    print('average PSNR', avg_psnr)
    with open(path.join(render_dir, 'psnr.txt'), 'w') as f:
        f.write(str(avg_psnr))
