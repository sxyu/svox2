from .util import Rays, Intrin
import torch
import torch.nn.functional as F
from typing import NamedTuple, Optional, Union
from os import path
import imageio
from tqdm import tqdm
import json
import numpy as np


class NeRFDataset:
    """
    NeRF dataset loader
    """

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
        device: Union[str, torch.device] = "cpu",
        scene_scale: float = 2/3,
        factor: int = 1,
        permutation: bool = True,
    ):
        assert path.isdir(root), f"'{root}' is not a directory"

        self.device = device
        self.permutation = permutation
        all_c2w = []
        all_gt = []

        split_name = split if split != "test_train" else "train"
        data_path = path.join(root, split_name)
        data_json = path.join(root, "transforms_" + split_name + ".json")

        print("LOAD DATA", data_path)

        j = json.load(open(data_json, "r"))

        # OpenGL -> OpenCV
        cam_trans = torch.diag(torch.tensor([1, -1, -1, 1], dtype=torch.float32))

        for frame in tqdm(j["frames"]):
            fpath = path.join(data_path, path.basename(frame["file_path"]) + ".png")
            c2w = torch.tensor(frame["transform_matrix"], dtype=torch.float32)
            c2w = c2w @ cam_trans  # To OpenCV

            im_gt = imageio.imread(fpath)
            all_c2w.append(c2w)
            all_gt.append(torch.from_numpy(im_gt))
        focal = float(
            0.5 * all_gt[0].shape[1] / np.tan(0.5 * j["camera_angle_x"])
        )
        self.c2w = torch.stack(all_c2w)
        self.c2w[:, :3, 3] *= scene_scale

        self.gt = torch.stack(all_gt).float() / 255.0
        if self.gt.size(-1) == 4:
            # Apply alpha channel
            self.gt = self.gt[..., :3] * self.gt[..., 3:] + (1.0 - self.gt[..., 3:])

        self.n_images, self.h_full, self.w_full, _ = self.gt.shape
        self.intrins_full : Intrin = Intrin(focal, focal,
                                            self.w_full * 0.5,
                                            self.h_full * 0.5)

        self.split = split
        self.scene_scale = scene_scale
        if self.split == "train":
            self.gen_rays(factor=factor)
        else:
            # Rays are not needed for testing
            self.h, self.w = self.h_full, self.w_full
            self.intrins : Intrin = self.intrins_full

        # Hardcoded; adjust scene_scale to make sure the scene fits in a unit sphere
        self.scene_center = [0.0, 0.0, 0.0]
        self.scene_radius = 1.0
        self.use_sphere_bound = True

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
        xx = (xx - self.intrins.cx) / self.intrins.fx
        yy = (yy - self.intrins.cy) / self.intrins.fy
        zz = torch.ones_like(xx)
        dirs = torch.stack((xx, yy, zz), dim=-1)  # OpenCV convention
        dirs /= torch.norm(dirs, dim=-1, keepdim=True)
        dirs = dirs.reshape(1, -1, 3, 1)
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

    def shuffle_rays(self):
        """
        Shuffle all rays
        """
        if self.split == "train":
            n_rays = self.rays.origins.size(0)
            if self.permutation:
                print(" Shuffling rays")
                perm = torch.randperm(n_rays, device=self.device)
            else:
                print(" Randomizing rays")
                perm = torch.randint(0, n_rays, (n_rays,), device=self.device)
            self.rays = self.rays_init.to(device=self.device)[perm]
