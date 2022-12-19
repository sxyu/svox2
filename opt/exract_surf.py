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
import mcubes
import cv2
from tqdm import tqdm
parser = configargparse.ArgumentParser()
parser.add_argument('ckpt', type=str)

config_util.define_common_args(parser)

parser.add_argument(
    "--intersect_th",
    type=float,
    default=0.1,
    help="alpha threshold for determining intersections"
)
parser.add_argument(
    "--out_path",
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



args = parser.parse_args()
device = 'cuda:0'


if not path.isfile(args.ckpt):
    args.ckpt = path.join(args.ckpt, 'ckpt.npz')


grid = svox2.SparseGrid.load(args.ckpt, device=device)
print(grid.center, grid.radius)


config_util.setup_render_opts(grid.opt, args)


if args.extract_nerf:
    grid.surface_data = None
    grid.surface_type = svox2.__dict__['SURFACE_TYPE_NONE']
    grid.opt.backend = 'cuvol'

print('Render options', grid.opt)

# grid.extract_mesh(args.out_path, args.intersect_th)
grid.extract_mesh(args.out_path, None)

