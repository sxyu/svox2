import torch
import torch.cuda
from typing import NamedTuple, Optional, Union
from dataclasses import dataclass
import numpy as np
import cv2
from matplotlib import pyplot as plt


@dataclass
class Rays:
    origins: torch.Tensor
    dirs: torch.Tensor
    gt: torch.Tensor

    def to(self, *args, **kwargs):
        origins = self.origins.to(*args, **kwargs)
        dirs = self.dirs.to(*args, **kwargs)
        gt = self.gt.to(*args, **kwargs)
        return Rays(origins, dirs, gt)

    def __getitem__(self, key):
        origins = self.origins[key]
        dirs = self.dirs[key]
        gt = self.gt[key]
        return Rays(origins, dirs, gt)

    def __len__(self):
        return self.origins.size(0)

@dataclass
class Intrin:
    fx: float
    fy: float
    cx: float
    cy: float

    def scale(self, scaling: float):
        return Intrin(
                self.fx * scaling,
                self.fy * scaling,
                self.cx * scaling,
                self.cy * scaling
                )


class Timing:
    """
    Timing environment
    usage:
    with Timing("message"):
        your commands here
    will print CUDA runtime in ms
    """

    def __init__(self, name):
        self.name = name

    def __enter__(self):
        self.start = torch.cuda.Event(enable_timing=True)
        self.end = torch.cuda.Event(enable_timing=True)
        self.start.record()

    def __exit__(self, type, value, traceback):
        self.end.record()
        torch.cuda.synchronize()
        print(self.name, "elapsed", self.start.elapsed_time(self.end), "ms")


def get_expon_lr_func(
    lr_init, lr_final, lr_delay_steps=0, lr_delay_mult=1.0, max_steps=1000000
):
    """
    Continuous learning rate decay function. Adapted from JaxNeRF

    The returned rate is lr_init when step=0 and lr_final when step=max_steps, and
    is log-linearly interpolated elsewhere (equivalent to exponential decay).
    If lr_delay_steps>0 then the learning rate will be scaled by some smooth
    function of lr_delay_mult, such that the initial learning rate is
    lr_init*lr_delay_mult at the beginning of optimization but will be eased back
    to the normal learning rate when steps>lr_delay_steps.

    :param conf: config subtree 'lr' or similar
    :param max_steps: int, the number of steps during optimization.
    :return HoF which takes step as input
    """

    def helper(step):
        if step < 0 or (lr_init == 0.0 and lr_final == 0.0):
            # Disable this parameter
            return 0.0
        if lr_delay_steps > 0:
            # A kind of reverse cosine decay.
            delay_rate = lr_delay_mult + (1 - lr_delay_mult) * np.sin(
                0.5 * np.pi * np.clip(step / lr_delay_steps, 0, 1)
            )
        else:
            delay_rate = 1.0
        t = np.clip(step / max_steps, 0, 1)
        log_lerp = np.exp(np.log(lr_init) * (1 - t) + np.log(lr_final) * t)
        return delay_rate * log_lerp

    return helper


def viridis_cmap(gray: np.ndarray):
    """
    Visualize a single-channel image using matplotlib's viridis color map
    yellow is low, blue is high
    :param gray: np.ndarray, (H, W) or (H, W, 1) unscaled
    :return: (H, W, 3) float32 in [0, 1]
    """
    colored = plt.cm.viridis(plt.Normalize()(gray.squeeze()))[..., :-1]
    return colored.astype(np.float32)


def save_img(img: np.ndarray, path: str):
    """Save an image to disk. Image should have values in [0,1]."""
    img = np.array((np.clip(img, 0.0, 1.0) * 255.0).astype(np.uint8))
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    cv2.imwrite(path, img)


def equirect2xyz(uv):
    """
    Convert equirectangular coordinate to unit vector,
    inverse of xyz2equirect
    Args:
        uv: np.ndarray [..., 2] x, y coordinates in image space in [-1.0, 1.0]
    Returns:
        xyz: np.ndarray [..., 3] unit vectors
    """
    lon = uv[..., 0] * np.pi
    lat = uv[..., 1] * (np.pi * 0.5)
    coslat = np.cos(lat)
    return np.stack(
        [
            coslat * np.sin(lon),
            coslat * np.cos(lon),
            np.sin(lat),
        ],
        axis=-1,
    )


def generate_dirs_equirect(w, h):
    x, y = np.meshgrid(  # pylint: disable=unbalanced-tuple-unpacking
        np.arange(w, dtype=np.float32) + 0.5,  # X-Axis (columns)
        np.arange(h, dtype=np.float32) + 0.5,  # Y-Axis (rows)
        indexing="xy",
    )
    uv = np.stack([x * (2.0 / w) - 1.0, y * (2.0 / h) - 1.0], axis=-1)
    camera_dirs = equirect2xyz(uv)
    return camera_dirs
