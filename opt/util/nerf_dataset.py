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


class NeRFDataset(DatasetBase):
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
        epoch_size : Optional[int] = None,
        device: Union[str, torch.device] = "cpu",
        scene_scale: Optional[float] = None,
        factor: int = 1,
        scale : Optional[float] = None,
        permutation: bool = True,
        white_bkgd: bool = True,
        n_images = None,
        **kwargs
    ):
        super().__init__()
        assert path.isdir(root), f"'{root}' is not a directory"

        if scene_scale is None:
            scene_scale = 2/3
        if scale is None:
            scale = 1.0
        self.device = device
        self.permutation = permutation
        self.epoch_size = epoch_size
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
            if scale < 1.0:
                full_size = list(im_gt.shape[:2])
                rsz_h, rsz_w = [round(hw * scale) for hw in full_size]
                im_gt = cv2.resize(im_gt, (rsz_w, rsz_h), interpolation=cv2.INTER_AREA)

            all_c2w.append(c2w)
            all_gt.append(torch.from_numpy(im_gt))
        focal = float(
            0.5 * all_gt[0].shape[1] / np.tan(0.5 * j["camera_angle_x"])
        )
        self.c2w = torch.stack(all_c2w)
        self.c2w[:, :3, 3] *= scene_scale

        self.gt = torch.stack(all_gt).float() / 255.0
        if self.gt.size(-1) == 4:
            if white_bkgd:
                # Apply alpha channel
                self.gt = self.gt[..., :3] * self.gt[..., 3:] + (1.0 - self.gt[..., 3:])
            else:
                self.gt = self.gt[..., :3]

        self.n_images, self.h_full, self.w_full, _ = self.gt.shape
        # Choose a subset of training images
        if n_images is not None:
            if n_images > self.n_images:
                print(f'using {self.n_images} available training views instead of the requested {n_images}.')
                n_images = self.n_images
            self.n_images = n_images
            self.gt = self.gt[0:n_images,...]
            self.c2w = self.c2w[0:n_images,...]

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

        self.should_use_background = False  # Give warning

