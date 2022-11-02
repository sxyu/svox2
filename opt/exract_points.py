# Copyright 2021 Alex Yu
# Render 360 circle path

import torch
import svox2
import svox2.utils
import math
import configargparse
import numpy as np
import os
from os import path
from util.dataset import datasets
from util.util import Timing, compute_ssim, viridis_cmap, pose_spherical
from util import config_util
import sklearn.neighbors as skln

import imageio
import cv2
from tqdm import tqdm
parser = configargparse.ArgumentParser()
parser.add_argument('ckpt', type=str)

config_util.define_common_args(parser)

parser.add_argument('--n_eval', '-n', type=int, default=100000, help='images to evaluate (equal interval), at most evals every image')
parser.add_argument('--traj_type',
                    choices=['spiral', 'circle', 'test'],
                    default='spiral',
                    help="Render a spiral (doubles length, using 2 elevations), or just a cirle")
parser.add_argument(
                "--width", "-W", type=float, default=None, help="Rendering image width (only if not --traj)"
                        )
parser.add_argument(
                    "--height", "-H", type=float, default=None, help="Rendering image height (only if not --traj)"
                            )
parser.add_argument(
	"--num_views", "-N", type=int, default=10,
    help="Number of frames to render"
)

# Path adjustment
parser.add_argument(
    "--offset", type=str, default="0,0,0", help="Center point to rotate around (only if not --traj)"
)
parser.add_argument("--radius", type=float, default=2.0, help="Radius of orbit (only if not --traj)")
parser.add_argument(
    "--elevation",
    type=float,
    default=-90,
    help="Elevation of orbit in deg, negative is above",
)
parser.add_argument(
    "--elevation2",
    type=float,
    default=90,
    help="Max elevation, only for spiral",
)
parser.add_argument(
    "--vec_up",
    type=str,
    default=None,
    help="up axis for camera views (only if not --traj);"
    "3 floats separated by ','; if not given automatically determined",
)
parser.add_argument(
    "--vert_shift",
    type=float,
    default=0.0,
    help="vertical shift by up axis"
)
parser.add_argument(
    "--use_test_cams",
    action='store_true', 
    default=False,
    help="Use test cameras to extract pts"
)
parser.add_argument(
    "--downsample_density",
    type=float,
    default=0.,
    help="density for downsampling the pts, set to 0 to disable"
)
parser.add_argument(
    "--intersect_th",
    type=float,
    default=0.1,
    help="alpha threshold for determining intersections"
)
parser.add_argument(
    "--out_path",
    type=str,
    default='pts.npy'
)

# Camera adjustment
parser.add_argument('--crop',
                    type=float,
                    default=1.0,
                    help="Crop (0, 1], 1.0 = full image")



args = parser.parse_args()
device = 'cuda:0'


dset = datasets[args.dataset_type](args.data_dir, split="test",
                                    **config_util.build_data_options(args))

if args.vec_up is None:
    up_rot = dset.c2w[:, :3, :3].cpu().numpy()
    ups = np.matmul(up_rot, np.array([0, -1.0, 0])[None, :, None])[..., 0]
    args.vec_up = np.mean(ups, axis=0)
    args.vec_up /= np.linalg.norm(args.vec_up)
    print('  Auto vec_up', args.vec_up)
else:
    args.vec_up = np.array(list(map(float, args.vec_up.split(","))))


args.offset = np.array(list(map(float, args.offset.split(","))))
if args.traj_type == 'spiral':
    angles = np.linspace(-180, 180, args.num_views + 1)[:-1]
    elevations = np.linspace(args.elevation, args.elevation2, args.num_views)
    c2ws = [
        pose_spherical(
            angle,
            ele,
            args.radius,
            args.offset,
            vec_up=args.vec_up,
        )
        for ele, angle in zip(elevations, angles)
    ]
    c2ws += [
        pose_spherical(
            angle,
            ele,
            args.radius,
            args.offset,
            vec_up=args.vec_up,
        )
        for ele, angle in zip(reversed(elevations), angles)
    ]
    c2ws = np.stack(c2ws, axis=0)
elif args.traj_type == 'test':
    if args.num_views >= dset.c2w.shape[0]:
        c2ws = dset.c2w.numpy()[:, :4, :4]
    else:
        test_cam_ids = np.round(np.linspace(0, dset.c2w.shape[0] - 1, args.num_views)).astype(int)
        c2ws = dset.c2w.numpy()[test_cam_ids, :4, :4]
else :
    c2ws = [
        pose_spherical(
            angle,
            args.elevation,
            args.radius,
            args.offset,
            vec_up=args.vec_up,
        )
        for angle in np.linspace(-180, 180, args.num_views + 1)[:-1]
    ]
    c2ws = np.stack(c2ws, axis=0)
if args.vert_shift != 0.0:
    c2ws[:, :3, 3] += np.array(args.vec_up) * args.vert_shift
c2ws = torch.from_numpy(c2ws).to(device=device)

if not path.isfile(args.ckpt):
    args.ckpt = path.join(args.ckpt, 'ckpt.npz')

render_out_path = path.join(path.dirname(args.ckpt), 'circle_depths')

# Handle various image transforms
if args.crop != 1.0:
    render_out_path += f'_crop{args.crop}'
if args.vert_shift != 0.0:
    render_out_path += f'_vshift{args.vert_shift}'

grid = svox2.SparseGrid.load(args.ckpt, device=device)
print(grid.center, grid.radius)


config_util.setup_render_opts(grid.opt, args)

# NOTE: no_grad enables the fast image-level rendering kernel for cuvol backend only
# other backends will manually generate rays per frame (slow)
with torch.no_grad():
    n_images = c2ws.size(0)
    img_eval_interval = max(n_images // args.n_eval, 1)
    all_pts = []
    #  if args.near_clip >= 0.0:
    grid.opt.near_clip = 0.0 #args.near_clip
    if args.width is None:
        args.width = dset.get_image_size(0)[1]
    if args.height is None:
        args.height = dset.get_image_size(0)[0]

    for img_id in tqdm(range(0, n_images, img_eval_interval)):
        dset_h, dset_w = args.height, args.width
        im_size = dset_h * dset_w
        w = dset_w if args.crop == 1.0 else int(dset_w * args.crop)
        h = dset_h if args.crop == 1.0 else int(dset_h * args.crop)

        cam = svox2.Camera(c2ws[img_id],
                           dset.intrins.get('fx', 0),
                           dset.intrins.get('fy', 0),
                           w * 0.5,
                           h * 0.5,
                           w, h,
                           ndc_coeffs=(-1.0, -1.0))
        # torch.cuda.synchronize()
        # depth = grid.volume_render_depth_image(cam)
        torch.cuda.synchronize()
        pts = grid.volume_render_extract_pts(cam, sigma_thresh=args.intersect_th, intersect_th=args.intersect_th)
        # depth = grid.volume_render_depth_image(cam, sigma_thresh=args.intersect_th, intersect_th=args.intersect_th) # ,
        torch.cuda.synchronize()
        # imageio.imwrite('d.png', depth.cpu().numpy())

        # depth.clamp_(0.0, 1.0)
        # depth = depth /depth.max()
        pts = pts.cpu().numpy()



        
        # depth = depth.cpu().numpy()
        # depth = (depth * 255).astype(np.uint8)
        all_pts.append(pts)

        # # debug purpose, save pts from only one view
        # np.save(path.join(path.dirname(args.ckpt), 'pts.npy'), pts.astype(np.half))
        # imageio.imsave(path.join(path.dirname(args.ckpt), 'depth.png'), depth)

        # raise NotImplementedError

        
all_pts = np.concatenate(all_pts, 0)
# for dtu dataset, need to rescale the pts
if hasattr(dset, 'pt_rescale'):
    all_pts = dset.world2rescale(all_pts)

if args.downsample_density > 0:
    nn_engine = skln.NearestNeighbors(n_neighbors=1, radius=args.downsample_density, algorithm='kd_tree', n_jobs=-1)
    nn_engine.fit(all_pts)
    rnn_idxs = nn_engine.radius_neighbors(all_pts, radius=args.downsample_density, return_distance=False)
    mask = np.ones(all_pts.shape[0], dtype=np.bool_)
    for curr, idxs in enumerate(rnn_idxs):
        if mask[curr]:
            mask[idxs] = 0
            mask[curr] = 1
    all_pts = all_pts[mask]


np.save(args.out_path, all_pts)




