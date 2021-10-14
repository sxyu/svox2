from .util import  Rays
import torch
import torch.nn.functional as F
from typing import NamedTuple, Optional, Union
from os import path
import imageio
from tqdm import tqdm
import json
import numpy as np

class Dataset():
    """
    NeRF dataset loader
    """
    focal: float
    c2w: torch.Tensor # (n_images, 4, 4)
    gt: torch.Tensor  # (n_images, h, w, 3)
    h: int
    w: int
    n_images: int
    rays: Optional[Rays]
    split: str

    def __init__(self, root, split,
                 device : Union[str, torch.device]='cpu',
                 scene_scale : float = 1.0/1.5,
                 factor : int = 1,
                 permutation : bool = True):
        self.device = device
        self.permutation = permutation
        all_c2w = []
        all_gt = []

        split_name = split if split != 'test_train' else 'train'
        data_path = path.join(root, split_name)
        data_json = path.join(root, 'transforms_' + split_name + '.json')
        print('LOAD DATA', data_path)
        j = json.load(open(data_json, 'r'))

        cam_trans = torch.diag(torch.tensor([1, -1, -1, 1], dtype=torch.float32))

        for frame in tqdm(j['frames']):
            fpath = path.join(data_path, path.basename(frame['file_path']) + '.png')
            c2w = torch.tensor(frame['transform_matrix'], dtype=torch.float32)
            c2w = c2w @ cam_trans  # To OpenCV

            im_gt = imageio.imread(fpath).astype(np.float32) / 255.0
            im_gt = im_gt[..., :3] * im_gt[..., 3:] + (1.0 - im_gt[..., 3:])
            all_c2w.append(c2w)
            all_gt.append(torch.from_numpy(im_gt))
        self.focal_full = float(0.5 * all_gt[0].shape[1] / np.tan(0.5 * j['camera_angle_x']))
        self.c2w = torch.stack(all_c2w)
        self.c2w[:, :3, 3] *= scene_scale
        self.gt = torch.stack(all_gt)
        self.n_images, self.h_full, self.w_full, _ = self.gt.shape
        self.split = split
        self.scene_scale = scene_scale
        if self.split == 'train':
            self.gen_rays(factor=factor)
        else:
            self.h, self.w, self.focal = self.h_full, self.w_full, self.focal_full

    def gen_rays(self, factor=1):
        print(" Generating rays, scaling factor", factor)
        # Generate rays
        self.factor = factor
        self.h = self.h_full // factor
        self.w = self.w_full // factor
        true_factor = self.h_full / self.h
        self.focal = self.focal_full / true_factor
        yy, xx = torch.meshgrid(
            torch.arange(self.h, dtype=torch.float32) + 0.5,
            torch.arange(self.w, dtype=torch.float32) + 0.5,
        )
        xx = (xx - self.w * 0.5) / self.focal
        yy = (yy - self.h * 0.5) / self.focal
        zz = torch.ones_like(xx)
        dirs = torch.stack((xx, yy, zz), dim=-1)  # OpenGL convention (NeRF)
        dirs /= torch.norm(dirs, dim=-1, keepdim=True)
        dirs = dirs.reshape(1, -1, 3, 1)
        del xx, yy, zz
        dirs = (self.c2w[:, None, :3, :3] @ dirs)[..., 0]

        if factor != 1:
            gt = F.interpolate(self.gt.permute([0, 3, 1, 2]),
                                    size=(self.h, self.w),
                                    mode='area').permute([0, 2, 3, 1])
            gt = gt.reshape(self.n_images, -1, 3)
        else:
            gt = self.gt.reshape(self.n_images, -1, 3)
        origins = self.c2w[:, None, :3, 3].expand(-1, self.h * self.w, -1).contiguous()
        if self.split == 'train':
            origins = origins.view(-1, 3)
            dirs = dirs.view(-1, 3)
            gt = gt.reshape(-1, 3)

        self.rays_init = Rays(origins=origins, dirs=dirs, gt=gt)
        self.rays = self.rays_init

    def shuffle_rays(self):
        """
        Shuffle all rays
        """
        if self.split == 'train':
            n_rays = self.rays.origins.size(0)
            if self.permutation:
                print(" Shuffling rays")
                perm = torch.randperm(n_rays)
            else:
                print(" Randomizing rays")
                perm = torch.randint(0, n_rays, (n_rays,))
            self.rays = Rays(origins = self.rays_init.origins[perm].to(device=self.device),
                    dirs = self.rays_init.dirs[perm].to(device=self.device),
                    gt = self.rays_init.gt[perm].to(device=self.device))
