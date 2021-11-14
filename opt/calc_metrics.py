# Calculate metrics on saved images

# Usage: python calc_metrics.py <renders_dir> <dataset_dir>
# Where <renders_dir> is ckpt_dir/test_renders
# or jaxnerf test renders dir

from util.dataset import datasets
from util.util import compute_ssim, viridis_cmap
from util import config_util
from os import path
from glob import glob
import imageio
import math
import argparse
import torch

parser = argparse.ArgumentParser()
parser.add_argument('render_dir', type=str)
parser.add_argument('--crop', type=float, default=1.0, help='center crop')
config_util.define_common_args(parser)
args = parser.parse_args()

if path.isfile(args.render_dir):
    print('please give the test_renders directory (not checkpoint) in the future')
    args.render_dir = path.join(path.dirname(args.render_dir), 'test_renders')

device = 'cuda:0'

import lpips
lpips_vgg = lpips.LPIPS(net="vgg").eval().to(device)

dset = datasets[args.dataset_type](args.data_dir, split="test",
                                    **config_util.build_data_options(args))


im_files = sorted(glob(path.join(args.render_dir, "*.png")))
im_files = [x for x in im_files if not path.basename(x).startswith('disp_')]   # Remove depths
assert len(im_files) == dset.n_images, \
       f'number of images found {len(im_files)} differs from test set images:{dset.n_images}'

avg_psnr = 0.0
avg_ssim = 0.0
avg_lpips = 0.0
n_images_gen = 0
for i, im_path in enumerate(im_files):
    im = torch.from_numpy(imageio.imread(im_path))
    im_gt = dset.gt[i]
    if im.shape[1] >= im_gt.shape[1] * 2:
        # Assume we have some gt/baselines on the left
        im = im[:, -im_gt.shape[1]:]
    im = im.float() / 255
    if args.crop != 1.0:
        del_tb = int(im.shape[0] * (1.0 - args.crop) * 0.5)
        del_lr = int(im.shape[1] * (1.0 - args.crop) * 0.5)
        im = im[del_tb:-del_tb, del_lr:-del_lr]
        im_gt = im_gt[del_tb:-del_tb, del_lr:-del_lr]

    mse = (im - im_gt) ** 2
    mse_num : float = mse.mean().item()
    psnr = -10.0 * math.log10(mse_num)
    ssim = compute_ssim(im_gt, im).item()
    lpips_i = lpips_vgg(im_gt.permute([2, 0, 1]).cuda().contiguous(),
            im.permute([2, 0, 1]).cuda().contiguous(),
            normalize=True).item()

    print(i, 'of', len(im_files), '; PSNR', psnr, 'SSIM', ssim, 'LPIPS', lpips_i)
    avg_psnr += psnr
    avg_ssim += ssim
    avg_lpips += lpips_i
    n_images_gen += 1  # Just to be sure

avg_psnr /= n_images_gen
avg_ssim /= n_images_gen
avg_lpips /= n_images_gen
print('AVERAGES')
print('PSNR:', avg_psnr)
print('SSIM:', avg_ssim)
print('LPIPS:', avg_lpips)
postfix = '_cropped' if args.crop != 1.0 else ''
#  with open(path.join(args.render_dir, f'psnr{postfix}.txt'), 'w') as f:
#      f.write(str(avg_psnr))
#  with open(path.join(args.render_dir, f'ssim{postfix}.txt'), 'w') as f:
#      f.write(str(avg_ssim))
#  with open(path.join(args.render_dir, f'lpips{postfix}.txt'), 'w') as f:
#      f.write(str(avg_lpips))
