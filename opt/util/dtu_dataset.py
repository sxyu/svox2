# Standard NeRF Blender dataset loader
from .util import Rays, Intrin, select_or_shuffle_rays
from .dataset_base import DatasetBase
import torch
import torch.nn.functional as F
from typing import NamedTuple, Optional, Union
from os import path
import imageio
from tqdm import tqdm
import cv2
import json
import numpy as np
import os
from pathlib import Path

def load_K_Rt_from_P(filename, P=None):
    if P is None:
        lines = open(filename).read().splitlines()
        if len(lines) == 4:
            lines = lines[1:]
        lines = [[x[0], x[1], x[2], x[3]] for x in (x.split(" ") for x in lines)]
        P = np.asarray(lines).astype(np.float32).squeeze()

    out = cv2.decomposeProjectionMatrix(P)
    K = out[0]
    R = out[1]
    t = out[2]

    K = K/K[2,2]
    intrinsics = np.eye(4)
    intrinsics[:3, :3] = K

    pose = np.eye(4, dtype=np.float32)
    pose[:3, :3] = R.transpose()
    pose[:3,3] = (t[:3] / t[3])[:,0]

    return intrinsics, pose

# raise NotImplementedError()

class DTUDataset(DatasetBase):
    """Dataset for a class of objects, where each datapoint is a SceneInstanceDataset."""

    focal: float
    c2w: torch.Tensor  # (n_images, 4, 4)
    gt: torch.Tensor  # (n_images, h, w, 3)
    h: int
    w: int
    n_images: int
    rays: Optional[Rays]
    split: str

    def __init__(
        self,
        root,
        split,
        epoch_size : Optional[int] = None,
        device: Union[str, torch.device] = "cpu",
        scene_scale: Optional[float] = None,  # Scene scaling
        factor: int = 1,                      # Image scaling (on ray gen; use gen_rays(factor) to dynamically change scale)
        scale : Optional[float] = 1.,        # Image scaling (on load)
        permutation: bool = True,
        white_bkgd: bool = True,
        apply_mask: bool = True,
        **kwargs
        ):
        super().__init__()
        assert path.isdir(root), f"'{root}' is not a directory"

        if scene_scale is None:
            scene_scale = 1.0
        if scale is None:
            scale = 1.0
        
        self.scene_radius = [1., 1., 1.]
        self.scene_radius = [.5, .5, .5]
        self.device = device
        self.permutation = permutation
        self.epoch_size = epoch_size
        all_c2w = []
        all_gt = []
        all_mask = []
        

        img_path = Path(root) / 'image'
        mask_path = Path(root) / 'mask'

        image_paths = sorted(img_path.glob('*'))
        mask_paths = sorted(mask_path.glob('*'))
        self.n_images = len(image_paths)

        cam_file_path = Path(root) / 'cameras_large.npz'
        camera_dict = np.load(str(cam_file_path))
        scale_mats = [camera_dict['scale_mat_%d' % idx].astype(np.float32) for idx in range(self.n_images)]
        world_mats = [camera_dict['world_mat_%d' % idx].astype(np.float32) for idx in range(self.n_images)]

        self.pt_rescale = scale_mats[0]

        intrinsics_all = []

        for i in tqdm(range(self.n_images)):
            # load pose
            scale_mat = scale_mats[i]
            world_mat = world_mats[i]
            P = world_mat @ scale_mat
            P = P[:3, :4]
            intrinsics, pose = load_K_Rt_from_P(None, P)
            intrinsics_all.append(torch.from_numpy(intrinsics).float())
            all_c2w.append(torch.from_numpy(pose).float())

            # load rgb
            im_gt = imageio.imread(str(image_paths[i]))
            if scale < 1.0:
                full_size = list(im_gt.shape[:2])
                rsz_h, rsz_w = [round(hw * scale) for hw in full_size]
                im_gt = cv2.resize(im_gt, (rsz_w, rsz_h), interpolation=cv2.INTER_AREA)
            
            # load mask
            im_mask = imageio.imread(str(mask_paths[i]))[..., :3]
            if scale < 1.0:
                full_size = list(im_mask.shape[:2])
                rsz_h, rsz_w = [round(hw * scale) for hw in full_size]
                im_mask = cv2.resize(im_mask, (rsz_w, rsz_h), interpolation=cv2.INTER_AREA)

            if apply_mask:
                im_gt[im_mask==0] = 255 if white_bkgd else 0

            all_gt.append(torch.from_numpy(im_gt))
            all_mask.append(torch.from_numpy(im_mask))

        self.c2w = torch.stack(all_c2w)
        self.c2w[:, :3, 3] *= scene_scale

        self.gt = torch.stack(all_gt).float() / 255.0
        self.mask = torch.stack(all_mask).float() / 255.0
        if self.gt.size(-1) == 4:
            if white_bkgd:
                # Apply alpha channel
                self.gt = self.gt[..., :3] * self.gt[..., 3:] + (1.0 - self.gt[..., 3:])
            else:
                self.gt = self.gt[..., :3]

        _, self.h_full, self.w_full, _ = self.gt.shape

        intrinsics_all = torch.stack(intrinsics_all)
        self.intrins_full : Intrin = Intrin(intrinsics_all[:, 0, 0], 
                                            intrinsics_all[:, 1, 1],
                                            intrinsics_all[:, 0, 2],
                                            intrinsics_all[:, 1, 2]).scale(scale)

        self.split = split
        self.scene_scale = scene_scale
        if self.split == "train":
            self.gen_rays(factor=factor)
        else:
            # Rays are not needed for testing
            self.h, self.w = self.h_full, self.w_full
            self.intrins : Intrin = self.intrins_full

        self.should_use_background = False  # Give warning

    def __len__(self):
        return self.n_images

    def gen_rays(self, factor=1):
        print(" Generating rays, scaling factor", factor)
        # Generate rays
        self.factor = factor
        self.h = self.h_full // factor
        self.w = self.w_full // factor
        true_factor = self.h_full / self.h
        self.intrins = self.intrins_full.scale(1.0 / true_factor)
        yy, xx = torch.meshgrid(
            torch.arange(self.h, dtype=torch.float32) + 0.5,
            torch.arange(self.w, dtype=torch.float32) + 0.5,
        )
        xx = (xx[None, ...] -  self.intrins.cx[:, None, None]) / self.intrins.fx[:, None, None]
        yy = (yy[None, ...] - self.intrins.cy[:, None, None]) / self.intrins.fy[:, None, None]
        zz = torch.ones_like(xx)
        dirs = torch.stack((xx, yy, zz), dim=-1)  # OpenCV convention
        dirs /= torch.norm(dirs, dim=-1, keepdim=True)
        dirs = dirs.reshape(self.intrins.cx.shape[0], -1, 3, 1)
        del xx, yy, zz
        dirs = (self.c2w[:, None, :3, :3] @ dirs)[..., 0]

        if factor != 1:
            gt = F.interpolate(
                self.gt.permute([0, 3, 1, 2]), size=(self.h, self.w), mode="area"
            ).permute([0, 2, 3, 1])
            gt = gt.reshape(self.n_images, -1, 3)
        else:
            gt = self.gt.reshape(self.n_images, -1, 3)
        origins = self.c2w[:, None, :3, 3].expand(-1, self.h * self.w, -1).contiguous()
        if self.split == "train":
            origins = origins.view(-1, 3)
            dirs = dirs.view(-1, 3)
            gt = gt.reshape(-1, 3)

        self.rays_init = Rays(origins=origins, dirs=dirs, gt=gt)
        self.rays = self.rays_init

    def near_far_from_sphere(self, rays_o, rays_d):
        # TODO: find out what this does
        a = torch.sum(rays_d**2, dim=-1, keepdim=True)
        b = 2.0 * torch.sum(rays_o * rays_d, dim=-1, keepdim=True)
        mid = 0.5 * (-b) / a
        near = mid - 1.0
        far = mid + 1.0
        return near, far

    def world2rescale(self, pts: np.ndarray):
        '''
        Transform points in world space to DTU GT space
        '''
        return pts * self.pt_rescale[0,0] + self.pt_rescale[:3,3][None]

    def rescale2world(self, pts: np.ndarray):
        '''
        Transform points in DTU GT space to world space
        '''
        return (pts - self.pt_rescale[:3,3][None]) / self.pt_rescale[0,0]
    





