import torch
import torch.cuda
import torch.nn.functional as F
from typing import Optional, Union, List
from dataclasses import dataclass
import numpy as np
import cv2
from scipy.spatial.transform import Rotation
from scipy.interpolate import CubicSpline
from matplotlib import pyplot as plt
from warnings import warn


@dataclass
class Rays:
    origins: Union[torch.Tensor, List[torch.Tensor]]
    dirs: Union[torch.Tensor, List[torch.Tensor]]
    gt: Union[torch.Tensor, List[torch.Tensor]]

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
    fx: Union[float, torch.Tensor]
    fy: Union[float, torch.Tensor]
    cx: Union[float, torch.Tensor]
    cy: Union[float, torch.Tensor]

    def scale(self, scaling: float):
        return Intrin(
                    self.fx * scaling,
                    self.fy * scaling,
                    self.cx * scaling,
                    self.cy * scaling
                )

    def get(self, field:str, image_id:int=0):
        val = self.__dict__[field]
        return val if isinstance(val, float) else val[image_id].item()


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
    yellow is high value, blue is low
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


def equirect2xyz(uv, rows, cols):
    """
    Convert equirectangular coordinate to unit vector,
    inverse of xyz2equirect
    Taken from Vickie Ye
    Args:
        uv: np.ndarray [..., 2] x, y coordinates in image space in [-1.0, 1.0]
    Returns:
        xyz: np.ndarray [..., 3] unit vectors
    """
    lon = (uv[..., 0] * (1.0 / cols) - 0.5) * (2 * np.pi)
    lat = -(uv[..., 1] * (1.0 / rows) - 0.5) * np.pi
    coslat = np.cos(lat)
    return np.stack(
        [
            coslat * np.sin(lon),
            np.sin(lat),
            coslat * np.cos(lon),
        ],
        axis=-1,
    )

def xyz2equirect(bearings, rows, cols):
    """
    Convert ray direction vectors into equirectangular pixel coordinates.
    Inverse of equirect2xyz.
    Taken from Vickie Ye
    """
    lat = np.arcsin(bearings[..., 1])
    lon = np.arctan2(bearings[..., 0], bearings[..., 2])
    x = cols * (0.5 + lon / 2 / np.pi)
    y = rows * (0.5 - lat / np.pi)
    return np.stack([x, y], axis=-1)

def generate_dirs_equirect(w, h):
    x, y = np.meshgrid(  # pylint: disable=unbalanced-tuple-unpacking
        np.arange(w, dtype=np.float32) + 0.5,  # X-Axis (columns)
        np.arange(h, dtype=np.float32) + 0.5,  # Y-Axis (rows)
        indexing="xy",
    )
    uv = np.stack([x * (2.0 / w) - 1.0, y * (2.0 / h) - 1.0], axis=-1)
    camera_dirs = equirect2xyz(uv)
    return camera_dirs


# Data
def select_or_shuffle_rays(rays_init : Rays,
                 permutation: int = False,
                 epoch_size: Optional[int] = None,
                 device: Union[str, torch.device] = "cpu"):
    n_rays = rays_init.origins.size(0)
    n_samp = n_rays if (epoch_size is None) else epoch_size
    if permutation:
        print(" Shuffling rays")
        indexer = torch.randperm(n_rays, device='cpu')[:n_samp]
    else:
        print(" Selecting random rays")
        indexer = torch.randint(n_rays, (n_samp,), device='cpu')
    return rays_init[indexer].to(device=device)


def compute_ssim(
    img0,
    img1,
    max_val=1.0,
    filter_size=11,
    filter_sigma=1.5,
    k1=0.01,
    k2=0.03,
    return_map=False,
):
    """Computes SSIM from two images.

    This function was modeled after tf.image.ssim, and should produce comparable
    output.

    Args:
      img0: torch.tensor. An image of size [..., width, height, num_channels].
      img1: torch.tensor. An image of size [..., width, height, num_channels].
      max_val: float > 0. The maximum magnitude that `img0` or `img1` can have.
      filter_size: int >= 1. Window size.
      filter_sigma: float > 0. The bandwidth of the Gaussian used for filtering.
      k1: float > 0. One of the SSIM dampening parameters.
      k2: float > 0. One of the SSIM dampening parameters.
      return_map: Bool. If True, will cause the per-pixel SSIM "map" to returned

    Returns:
      Each image's mean SSIM, or a tensor of individual values if `return_map`.
    """
    device = img0.device
    ori_shape = img0.size()
    width, height, num_channels = ori_shape[-3:]
    img0 = img0.view(-1, width, height, num_channels).permute(0, 3, 1, 2)
    img1 = img1.view(-1, width, height, num_channels).permute(0, 3, 1, 2)
    batch_size = img0.shape[0]

    # Construct a 1D Gaussian blur filter.
    hw = filter_size // 2
    shift = (2 * hw - filter_size + 1) / 2
    f_i = ((torch.arange(filter_size, device=device) - hw + shift) / filter_sigma) ** 2
    filt = torch.exp(-0.5 * f_i)
    filt /= torch.sum(filt)

    # Blur in x and y (faster than the 2D convolution).
    # z is a tensor of size [B, H, W, C]
    filt_fn1 = lambda z: F.conv2d(
        z, filt.view(1, 1, -1, 1).repeat(num_channels, 1, 1, 1),
        padding=[hw, 0], groups=num_channels)
    filt_fn2 = lambda z: F.conv2d(
        z, filt.view(1, 1, 1, -1).repeat(num_channels, 1, 1, 1),
        padding=[0, hw], groups=num_channels)

    # Vmap the blurs to the tensor size, and then compose them.
    filt_fn = lambda z: filt_fn1(filt_fn2(z))
    mu0 = filt_fn(img0)
    mu1 = filt_fn(img1)
    mu00 = mu0 * mu0
    mu11 = mu1 * mu1
    mu01 = mu0 * mu1
    sigma00 = filt_fn(img0 ** 2) - mu00
    sigma11 = filt_fn(img1 ** 2) - mu11
    sigma01 = filt_fn(img0 * img1) - mu01

    # Clip the variances and covariances to valid values.
    # Variance must be non-negative:
    sigma00 = torch.clamp(sigma00, min=0.0)
    sigma11 = torch.clamp(sigma11, min=0.0)
    sigma01 = torch.sign(sigma01) * torch.min(
        torch.sqrt(sigma00 * sigma11), torch.abs(sigma01)
    )

    c1 = (k1 * max_val) ** 2
    c2 = (k2 * max_val) ** 2
    numer = (2 * mu01 + c1) * (2 * sigma01 + c2)
    denom = (mu00 + mu11 + c1) * (sigma00 + sigma11 + c2)
    ssim_map = numer / denom
    ssim = torch.mean(ssim_map.reshape([-1, num_channels*width*height]), dim=-1)
    return ssim_map if return_map else ssim


def generate_rays(w, h, focal, camtoworlds, equirect=False):
    """
    Generate perspective camera rays. Principal point is at center.
    Args:
        w: int image width
        h: int image heigth
        focal: float real focal length
        camtoworlds: jnp.ndarray [B, 4, 4] c2w homogeneous poses
        equirect: if true, generates spherical rays instead of pinhole
    Returns:
        rays: Rays a namedtuple(origins [B, 3], directions [B, 3], viewdirs [B, 3])
    """
    x, y = np.meshgrid(  # pylint: disable=unbalanced-tuple-unpacking
        np.arange(w, dtype=np.float32),  # X-Axis (columns)
        np.arange(h, dtype=np.float32),  # Y-Axis (rows)
        indexing="xy",
    )

    if equirect:
        uv = np.stack([x * (2.0 / w) - 1.0, y * (2.0 / h) - 1.0], axis=-1)
        camera_dirs = equirect2xyz(uv)
    else:
        camera_dirs = np.stack(
            [
                (x - w * 0.5) / focal,
                -(y - h * 0.5) / focal,
                -np.ones_like(x),
            ],
            axis=-1,
        )

    #  camera_dirs = camera_dirs / np.linalg.norm(camera_dirs, axis=-1, keepdims=True)

    c2w = camtoworlds[:, None, None, :3, :3]
    camera_dirs = camera_dirs[None, Ellipsis, None]
    directions = np.matmul(c2w, camera_dirs)[Ellipsis, 0]
    origins = np.broadcast_to(
        camtoworlds[:, None, None, :3, -1], directions.shape
    )
    norms = np.linalg.norm(directions, axis=-1, keepdims=True)
    viewdirs = directions / norms
    rays = Rays(
        origins=origins, directions=directions, viewdirs=viewdirs
    )
    return rays


def similarity_from_cameras(c2w):
    """
    Get a similarity transform to normalize dataset
    from c2w (OpenCV convention) cameras

    :param c2w: (N, 4)

    :return T (4,4) , scale (float)
    """
    t = c2w[:, :3, 3]
    R = c2w[:, :3, :3]

    # (1) Rotate the world so that z+ is the up axis
    # we estimate the up axis by averaging the camera up axes
    ups = np.sum(R * np.array([0, -1.0, 0]), axis=-1)
    world_up = np.mean(ups, axis=0)
    world_up /= np.linalg.norm(world_up)

    up_camspace = np.array([0.0, -1.0, 0.0])
    c = (up_camspace * world_up).sum()
    cross = np.cross(world_up, up_camspace)
    skew = np.array([[0.0, -cross[2], cross[1]],
                     [cross[2], 0.0, -cross[0]],
                     [-cross[1], cross[0], 0.0]])
    if c > -1:
        R_align = np.eye(3) + skew + (skew @ skew) * 1 / (1+c)
    else:
        # In the unlikely case the original data has y+ up axis,
        # rotate 180-deg about x axis
        R_align = np.array([[-1.0, 0.0, 0.0],
                            [0.0, 1.0, 0.0],
                            [0.0, 0.0, 1.0]])


    #  R_align = np.eye(3) # DEBUG
    R = (R_align @ R)
    fwds = np.sum(R * np.array([0, 0.0, 1.0]), axis=-1)
    t = (R_align @ t[..., None])[..., 0]

    # (2) Recenter the scene using camera center rays
    # find the closest point to the origin for each camera's center ray
    nearest = t + (fwds * -t).sum(-1)[:, None] * fwds

    # median for more robustness
    translate = -np.median(nearest, axis=0)

    #  translate = -np.mean(t, axis=0)  # DEBUG

    transform = np.eye(4)
    transform[:3, 3] = translate
    transform[:3, :3] = R_align

    # (3) Rescale the scene using camera distances
    scale = 1.0 / np.median(np.linalg.norm(t + translate, axis=-1))
    return transform, scale

def jiggle_and_interp_poses(poses : torch.Tensor,
                            n_inter: int,
                            noise_std : float=0.0):
    """
    For generating a novel trajectory close to known trajectory

    :param poses: torch.Tensor (B, 4, 4)
    :param n_inter: int, number of views to interpolate in total
    :param noise_std: float, default 0
    """
    n_views_in = poses.size(0)
    poses_np = poses.cpu().numpy().copy()
    rot = Rotation.from_matrix(poses_np[:, :3, :3])
    trans = poses_np[:, :3, 3]
    trans += np.random.randn(*trans.shape) * noise_std
    pose_quat = rot.as_quat()

    t_in = np.arange(n_views_in, dtype=np.float32)
    t_out = np.linspace(t_in[0], t_in[-1], n_inter, dtype=np.float32)

    q_new = CubicSpline(t_in, pose_quat)
    q_new : np.ndarray = q_new(t_out)
    q_new = q_new / np.linalg.norm(q_new, axis=-1)[..., None]

    t_new = CubicSpline(t_in, trans)
    t_new = t_new(t_out)

    rot_new = Rotation.from_quat(q_new)
    R_new = rot_new.as_matrix()

    Rt_new = np.concatenate([R_new, t_new[..., None]], axis=-1)
    bottom = np.array([[0.0, 0.0, 0.0, 1.0]], dtype=np.float32)
    bottom = bottom[None].repeat(Rt_new.shape[0], 0)
    Rt_new = np.concatenate([Rt_new, bottom], axis=-2)
    Rt_new = torch.from_numpy(Rt_new).to(device=poses.device, dtype=poses.dtype)
    return Rt_new


# Rather ugly pose generation code, derived from NeRF
def _trans_t(t):
    return np.array(
        [
            [1, 0, 0, 0],
            [0, 1, 0, 0],
            [0, 0, 1, t],
            [0, 0, 0, 1],
        ],
        dtype=np.float32,
    )


def _rot_phi(phi):
    return np.array(
        [
            [1, 0, 0, 0],
            [0, np.cos(phi), -np.sin(phi), 0],
            [0, np.sin(phi), np.cos(phi), 0],
            [0, 0, 0, 1],
        ],
        dtype=np.float32,
    )


def _rot_theta(th):
    return np.array(
        [
            [np.cos(th), 0, -np.sin(th), 0],
            [0, 1, 0, 0],
            [np.sin(th), 0, np.cos(th), 0],
            [0, 0, 0, 1],
        ],
        dtype=np.float32,
    )

def pose_spherical(theta : float, phi : float, radius : float, offset : Optional[np.ndarray]=None,
                   vec_up : Optional[np.ndarray]=None):
    """
    Generate spherical rendering poses, from NeRF. Forgive the code horror
    :return: r (3,), t (3,)
    """
    c2w = _trans_t(radius)
    c2w = _rot_phi(phi / 180.0 * np.pi) @ c2w
    c2w = _rot_theta(theta / 180.0 * np.pi) @ c2w
    c2w = (
        np.array(
            [[-1, 0, 0, 0], [0, 0, 1, 0], [0, 1, 0, 0], [0, 0, 0, 1]],
            dtype=np.float32,
        )
        @ c2w
    )
    if vec_up is not None:
        vec_up = vec_up / np.linalg.norm(vec_up)
        vec_1 = np.array([vec_up[0], -vec_up[2], vec_up[1]])
        vec_2 = np.cross(vec_up, vec_1)

        trans = np.eye(4, 4, dtype=np.float32)
        trans[:3, 0] = vec_1
        trans[:3, 1] = vec_2
        trans[:3, 2] = vec_up
        c2w = trans @ c2w
    c2w = c2w @ np.diag(np.array([1, -1, -1, 1], dtype=np.float32))
    if offset is not None:
        c2w[:3, 3] += offset
    return c2w
