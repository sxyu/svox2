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
import imageio

from warnings import warn
from datetime import datetime
from torch.utils.tensorboard import SummaryWriter

from tqdm import tqdm
from typing import NamedTuple, Optional, Union

device = "cuda" if torch.cuda.is_available() else "cpu"
# torch.set_printoptions(sci_mode=False)

args = config_util.setup_train_conf()
USE_KERNEL = not args.nokernel
if args.surface_type is None:
    args.surface_type = 'none'

torch.manual_seed(20200823)
np.random.seed(20200823)

DATASET_TYPE = 'test'
IMG_ID = 0
P_COORD = torch.tensor([ # matlibplot [x, y]
    [400, 467],
    [425, 467]
    ])

P_COORD = None

factor = 1
dset = datasets[args.dataset_type](
            args.data_dir,
            split=DATASET_TYPE,
            device=device,
            factor=factor,
            **config_util.build_data_options(args))

if args.background_nlayers > 0 and not dset.should_use_background:
    warn('Using a background model for dataset type ' + str(type(dset)) + ' which typically does not use background')


ckpt_npz = path.join(args.train_dir, 'ckpt.npz')

if path.isfile(ckpt_npz):
    print('#####################################################')
    print(f'Resume from ckpt at {ckpt_npz}')
    grid = svox2.SparseGrid.load(ckpt_npz, device=device)
    assert svox2.__dict__['SURFACE_TYPE_' + args.surface_type.upper()] == grid.surface_type, "Loaded ckpt incompatible with given configs"
    print(f'Loaded from step {grid.step_id}')
    print('#####################################################')
else: 
    raise NotImplementedError(f'Ckpt {ckpt_npz} not found')

grid.density_data.data[:] = 1e8


optim_basis_mlp = None

if grid.basis_type == svox2.BASIS_TYPE_3D_TEXTURE and not (path.isfile(ckpt_npz) and args.load_ckpt):
    # do not reinit if resuming from ckpt
    grid.reinit_learned_bases(init_type='sh')

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


if args.enable_random:
    warn("Randomness is enabled for training (normal for LLFF & scenes with background)")

epoch_id = -1

with torch.no_grad():

    c2w = dset.c2w[IMG_ID].to(device=device)
    cam = svox2.Camera(c2w,
                        dset.intrins.get('fx', IMG_ID),
                        dset.intrins.get('fy', IMG_ID),
                        dset.intrins.get('cx', IMG_ID),
                        dset.intrins.get('cy', IMG_ID),
                        width=dset.get_image_size(IMG_ID)[1],
                        height=dset.get_image_size(IMG_ID)[0],
                        ndc_coeffs=dset.ndc_coeffs)
    rgb_pred_test = grid.volume_render_image(cam, use_kernel=USE_KERNEL, debug_pixels=P_COORD)
    rgb_pred_test = torch.clamp_max(rgb_pred_test, 1.)
    rgb_pred_test = rgb_pred_test.cpu().detach().numpy()

    if P_COORD is None:
        imageio.imsave(path.join(args.train_dir, f'debug_{grid.step_id}.png'), rgb_pred_test)





