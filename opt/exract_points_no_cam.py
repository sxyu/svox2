# Copyright 2021 Alex Yu
# Render 360 circle path

import torch
import svox2
import svox2.utils
import math
import open3d as o3d
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

# config_util.define_common_args(parser)

parser.add_argument(
    "--intersect_th",
    type=float,
    default=0.,
    help="alpha threshold for determining intersections"
)
parser.add_argument(
    "--out_path",
    type=str,
    default=None
)
parser.add_argument(
    "--data_path",
    type=str,
    default=None
)
parser.add_argument(
    "--debug_alpha",
    action='store_true', 
    default=False,
    help="Delete ckpt after extraction"
)
parser.add_argument(
    "--extract_nerf",
    action='store_true', 
    default=False,
    help="Delete ckpt after extraction"
)
parser.add_argument(
    "--del_ckpt",
    action='store_true', 
    default=False,
    help="Delete ckpt after extraction"
)
parser.add_argument(
    "--single_lv",
    action='store_true', 
    default=False,
    help="Use only single lv set"
)
parser.add_argument(
    "--downsample_density",
    type=float,
    default=0.001,
    help="density for downsampling the pts, set to 0 to disable"
)
parser.add_argument(
    "--surf_lv_set",
    type=float,
    default=None,
)
parser.add_argument(
    "--n_sample",
    type=int,
    default=10,
    help="density for downsampling the pts, set to 0 to disable"
)

args = parser.parse_args()
device = 'cuda:0'


dset = None
if args.data_path is not None:
    dset = datasets['auto'](args.data_path, split="train")
    scene_scale = dset.scene_scale
else:
    scene_scale = 2./3. # assume blender

if not path.isfile(args.ckpt):
    args.ckpt = path.join(args.ckpt, 'ckpt.npz')

if args.ckpt.endswith('grid.npy'):
    # better marching cube of exported density grid
    print(f'Loading density other nerf: {args.ckpt}')
    density_data = np.load(args.ckpt)[...,0]

    grid = svox2.SparseGrid(reso=density_data.shape,
                            use_sphere_bound=False,
                            device=device)
    
    density_data = torch.from_numpy(density_data.astype(np.float32)).to(device=device).view(-1,1)

    # perform a quick prunning
    reso = grid.links.shape
    valid_mask = density_data > 0
    valid_mask = valid_mask.view(reso)

    for _ in range(int(2)):
        valid_mask = grid._C.dilate(valid_mask)

    valid_mask = valid_mask.view(-1)
    grid.density_data.data = density_data[valid_mask]
    grid.sh_data.data = grid.sh_data.data[valid_mask]

    init_links = (
        torch.cumsum(valid_mask.to(torch.int32), dim=-1).int() - 1
    )
    init_links[~valid_mask] = -1

    grid.links = init_links.view(reso).to(device=device)
    kept_ratio = torch.count_nonzero(valid_mask) / valid_mask.numel()
    print(f'{kept_ratio} of the loaded grids has > 0 density')

    grid.surface_data = None
    grid.surface_type = svox2.__dict__['SURFACE_TYPE_NONE']
    grid.opt.backend = 'cuvol'
else:
    grid = svox2.SparseGrid.load(args.ckpt, device=device)
    print(grid.center, grid.radius)



# NOTE: no_grad enables the fast image-level rendering kernel for cuvol backend only
# other backends will manually generate rays per frame (slow)

if args.extract_nerf:
    grid.surface_data = None
    grid.surface_type = svox2.__dict__['SURFACE_TYPE_NONE']
    grid.opt.backend = 'cuvol'

print('Render options', grid.opt)

# args = parser.parse_args()
# device = 'cuda:0'


# if not path.isfile(args.ckpt):
#     args.ckpt = path.join(args.ckpt, 'ckpt.npz')


# grid = svox2.SparseGrid.load(args.ckpt, device=device)
# print(grid.center, grid.radius)


# # config_util.setup_render_opts(grid.opt, args)


# if args.extract_nerf:
#     grid.surface_data = None
#     grid.surface_type = svox2.__dict__['SURFACE_TYPE_NONE']

# print('Render options', grid.opt)



# grid.extract_mesh(args.out_path, args.intersect_th)

if args.surf_lv_set is None:
    # use level set from grid ckpt
    if grid.level_set_data is None:
        # nerf ckpt, surf lv set doesn't matter
        surf_lv_set = [0.]
    else:
        surf_lv_set = grid.level_set_data.cpu().detach().numpy()
        if args.single_lv:
            surf_lv_set = surf_lv_set[:1]
else:
    surf_lv_set = [args.surf_lv_set]

all_pts = []
for lv_set in surf_lv_set:
    pts = grid.extract_pts(n_sample=args.n_sample, density_thresh=args.intersect_th, scene_scale=scene_scale, to_world=True, surf_lv_set=lv_set)
    pts = pts.cpu().detach().numpy()

    if dset is not None and hasattr(dset, 'pt_rescale'):
        # rescale for DTU
        pts = dset.world2rescale(pts)

    if args.downsample_density > 0 and len(pts) > 0:
        nn_engine = skln.NearestNeighbors(n_neighbors=1, radius=args.downsample_density, algorithm='kd_tree', n_jobs=-1)
        nn_engine.fit(pts)
        rnn_idxs = nn_engine.radius_neighbors(pts, radius=args.downsample_density, return_distance=False)
        mask = np.ones(pts.shape[0], dtype=np.bool_)
        for curr, idxs in enumerate(rnn_idxs):
            if mask[curr]:
                mask[idxs] = 0
                mask[curr] = 1
        pts = pts[mask]

    all_pts.append(pts)

all_pts = np.concatenate(all_pts, axis=0)
if args.downsample_density > 0 and len(all_pts) > 0:
    nn_engine = skln.NearestNeighbors(n_neighbors=1, radius=args.downsample_density, algorithm='kd_tree', n_jobs=-1)
    nn_engine.fit(all_pts)
    rnn_idxs = nn_engine.radius_neighbors(all_pts, radius=args.downsample_density, return_distance=False)
    mask = np.ones(all_pts.shape[0], dtype=np.bool_)
    for curr, idxs in enumerate(rnn_idxs):
        if mask[curr]:
            mask[idxs] = 0
            mask[curr] = 1
    all_pts = all_pts[mask]

print(f'Saving pts to {args.out_path}')
if args.out_path.endswith('txt'):
    np.savetxt(args.out_path, all_pts)
elif args.out_path.endswith('ply'):
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(all_pts)
    o3d.io.write_point_cloud(args.out_path, pcd)
else:
    np.save(args.out_path, all_pts)

if args.del_ckpt:
    os.remove(args.ckpt)




