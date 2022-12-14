from .defs import *
from . import utils
import torch
from torch import nn, autograd
import torch.nn.functional as F
from typing import Union, List, NamedTuple, Optional, Tuple
from dataclasses import dataclass
from warnings import warn
from functools import reduce
from tqdm import tqdm
from scipy.spatial.transform import Rotation
import numpy as np
import sympy
from sympy.solvers import solve
from sympy import Symbol
import kaolin

_C = utils._get_c_extension()


@dataclass
class RenderOptions:
    """
    Rendering options, see comments
    available:
    :param backend: str, renderer backend
    :param background_brightness: float
    :param step_size: float, step size for rendering
    :param sigma_thresh: float
    :param stop_thresh: float
    """

    backend: str = 'cuvol'  # One of cuvol, svox1, nvol, surface

    background_brightness: float = 1.0  # [0, 1], the background color black-white

    step_size: float = 0.5  # Step size, in normalized voxels (not used for svox1)
    #  (i.e. 1 = 1 voxel width, different from svox where 1 = grid width!)

    sigma_thresh: float = 1e-10  # Voxels with sigmas < this are ignored, in [0, 1]
    #  make this higher for fast rendering

    stop_thresh: float = (
        1e-7  # Stops rendering if the remaining light intensity/termination, in [0, 1]
    )
    #  probability is <= this much (forward only)
    #  make this higher for fast rendering

    last_sample_opaque: bool = False   # Make the last sample opaque (for forward-facing)

    near_clip: float = 0.0
    use_spheric_clip: bool = False

    random_sigma_std: float = 1.0                   # Noise to add to sigma (only if randomize=True)
    random_sigma_std_background: float = 1.0        # Noise to add to sigma
                                                    # (for the BG model; only if randomize=True)

    surf_fake_sample: bool = False
    surf_fake_sample_min_vox_len: float = 0.1
    no_surf_grad_from_sh: bool = False
    alpha_activation_type: int = EXP_FN

    def _to_cpp(self, randomize: bool = False):
        """
        Generate object to pass to C++
        """
        opt = _C.RenderOptions()
        opt.background_brightness = self.background_brightness
        opt.step_size = self.step_size
        opt.sigma_thresh = self.sigma_thresh
        opt.stop_thresh = self.stop_thresh
        opt.near_clip = self.near_clip
        opt.use_spheric_clip = self.use_spheric_clip

        opt.last_sample_opaque = self.last_sample_opaque
        opt.surf_fake_sample = self.surf_fake_sample
        opt.surf_fake_sample_min_vox_len = self.surf_fake_sample_min_vox_len
        opt.no_surf_grad_from_sh = self.no_surf_grad_from_sh
        opt.alpha_activation_type = self.alpha_activation_type
        #  opt.randomize = randomize
        #  opt.random_sigma_std = self.random_sigma_std
        #  opt.random_sigma_std_background = self.random_sigma_std_background

        #  if randomize:
        #      # For our RNG
        #      UINT32_MAX = 2**32-1
        #      opt._m1 = np.random.randint(0, UINT32_MAX)
        #      opt._m2 = np.random.randint(0, UINT32_MAX)
        #      opt._m3 = np.random.randint(0, UINT32_MAX)
        #      if opt._m2 == opt._m3:
        #          opt._m3 += 1  # Prevent all equal case
        # Note that the backend option is handled in Python
        return opt


@dataclass
class Rays:
    origins: torch.Tensor
    dirs: torch.Tensor

    def _to_cpp(self):
        """
        Generate object to pass to C++
        """
        spec = _C.RaysSpec()
        spec.origins = self.origins
        spec.dirs = self.dirs
        return spec

    def __getitem__(self, key):
        return Rays(self.origins[key], self.dirs[key])

    @property
    def is_cuda(self) -> bool:
        return self.origins.is_cuda and self.dirs.is_cuda

@dataclass
class RayVoxIntersecs:
    voxel_ls: torch.Tensor
    vox_start_i: torch.Tensor
    vox_num: torch.Tensor

    def _to_cpp(self):
        """
        Generate object to pass to C++
        """
        spec = _C.RayVoxIntersecSpec()
        spec.voxel_ls = self.voxel_ls
        spec.vox_start_i = self.vox_start_i
        spec.vox_num = self.vox_num
        return spec

    def __getitem__(self, key):
        return RayVoxIntersecs(self.voxel_ls[key], self.vox_start_i[key], self.vox_num[key])

    @property
    def is_cuda(self) -> bool:
        return self.voxel_ls.is_cuda and self.vox_start_i.is_cuda and self.vox_num.is_cuda


@dataclass
class Camera:
    c2w: torch.Tensor  # OpenCV
    fx: float = 1111.11
    fy: Optional[float] = None
    cx: Optional[float] = None
    cy: Optional[float] = None
    width: int = 800
    height: int = 800

    ndc_coeffs: Union[Tuple[float, float], List[float]] = (-1.0, -1.0)

    @property
    def fx_val(self):
        return self.fx

    @property
    def fy_val(self):
        return self.fx if self.fy is None else self.fy

    @property
    def cx_val(self):
        return self.width * 0.5 if self.cx is None else self.cx

    @property
    def cy_val(self):
        return self.height * 0.5 if self.cy is None else self.cy

    @property
    def using_ndc(self):
        return self.ndc_coeffs[0] > 0.0

    def _to_cpp(self):
        """
        Generate object to pass to C++
        """
        spec = _C.CameraSpec()
        spec.c2w = self.c2w
        spec.fx = self.fx_val
        spec.fy = self.fy_val
        spec.cx = self.cx_val
        spec.cy = self.cy_val
        spec.width = self.width
        spec.height = self.height
        spec.ndc_coeffx = self.ndc_coeffs[0]
        spec.ndc_coeffy = self.ndc_coeffs[1]
        return spec

    @property
    def is_cuda(self) -> bool:
        return self.c2w.is_cuda

    def gen_rays(self) -> Rays:
        """
        Generate the rays for this camera
        :return: (origins (H*W, 3), dirs (H*W, 3))
        """
        origins = self.c2w[None, :3, 3].expand(self.height * self.width, -1).contiguous()
        yy, xx = torch.meshgrid(
            torch.arange(self.height, dtype=torch.float64, device=self.c2w.device) + 0.5,
            torch.arange(self.width, dtype=torch.float64, device=self.c2w.device) + 0.5,
        )
        xx = (xx - self.cx_val) / self.fx_val
        yy = (yy - self.cy_val) / self.fy_val
        zz = torch.ones_like(xx)
        dirs = torch.stack((xx, yy, zz), dim=-1)   # OpenCV
        del xx, yy, zz
        dirs /= torch.norm(dirs, dim=-1, keepdim=True)
        dirs = dirs.reshape(-1, 3, 1)
        dirs = (self.c2w[None, :3, :3].double() @ dirs)[..., 0]
        dirs = dirs.reshape(-1, 3).float()

        if self.ndc_coeffs[0] > 0.0:
            origins, dirs = utils.convert_to_ndc(
                    origins,
                    dirs,
                    self.ndc_coeffs)
            dirs /= torch.norm(dirs, dim=-1, keepdim=True)
        return Rays(origins, dirs)


# BEGIN Differentiable CUDA functions with custom gradient
class _SampleGridAutogradFunction(autograd.Function):
    @staticmethod
    def forward(
        ctx,
        data_density: torch.Tensor,
        data_sh: torch.Tensor,
        grid,
        points: torch.Tensor,
        want_colors: bool,
    ):
        assert not points.requires_grad, "Point gradient not supported"
        out_density, out_sh = _C.sample_grid(grid, points, want_colors)
        ctx.save_for_backward(points)
        ctx.grid = grid
        ctx.want_colors = want_colors
        return out_density, out_sh

    @staticmethod
    def backward(ctx, grad_out_density, grad_out_sh):
        (points,) = ctx.saved_tensors
        print('backward called')
        grad_density_grid = torch.zeros_like(ctx.grid.density_data.data)
        grad_sh_grid = torch.zeros_like(ctx.grid.sh_data.data)
        _C.sample_grid_backward(
            ctx.grid,
            points,
            grad_out_density.contiguous(),
            grad_out_sh.contiguous(),
            grad_density_grid,
            grad_sh_grid,
            ctx.want_colors
        )
        if not ctx.needs_input_grad[0]:
            grad_density_grid = None
        if not ctx.needs_input_grad[1]:
            grad_sh_grid = None

        return grad_density_grid, grad_sh_grid, None, None, None


class _VolumeRenderFunction(autograd.Function):
    @staticmethod
    def forward(
        ctx,
        data_density: torch.Tensor,
        data_sh: torch.Tensor,
        data_basis: torch.Tensor,
        data_background: torch.Tensor,
        grid,
        rays,
        opt,
        backend: str,
        ray_vox=None,
    ):
        cu_fn = _C.__dict__[f"volume_render_{backend}"]
        color = cu_fn(grid, rays, opt)
        ctx.save_for_backward(color)
        ctx.grid = grid
        ctx.rays = rays
        ctx.opt = opt
        ctx.backend = backend
        ctx.basis_data = data_basis
        return color

    @staticmethod
    def backward(ctx, grad_out): 
        # this backward is only used for testing. Use fused MSE method in actual training instead
        # grad_out is the gradient for color
        # import pdb; pdb.set_trace()
        (color_cache,) = ctx.saved_tensors
        cu_fn = _C.__dict__[f"volume_render_{ctx.backend}_backward"]
        grad_density_grid = torch.zeros_like(ctx.grid.density_data.data)
        grad_sh_grid = torch.zeros_like(ctx.grid.sh_data.data)
        if ctx.grid.basis_type == BASIS_TYPE_MLP:
            grad_basis = torch.zeros_like(ctx.basis_data)
        elif ctx.grid.basis_type == BASIS_TYPE_3D_TEXTURE:
            grad_basis = torch.zeros_like(ctx.grid.basis_data.data)
        if ctx.grid.background_data is not None:
            grad_background = torch.zeros_like(ctx.grid.background_data.data)
        grad_holder = _C.GridOutputGrads()
        grad_holder.grad_density_out = grad_density_grid
        grad_holder.grad_sh_out = grad_sh_grid
        if ctx.needs_input_grad[2]:
            grad_holder.grad_basis_out = grad_basis # for SH type, basis needs no grad
        if ctx.grid.background_data is not None and ctx.needs_input_grad[3]:
            grad_holder.grad_background_out = grad_background
        cu_fn(
            ctx.grid,
            ctx.rays,
            ctx.opt,
            grad_out.contiguous(),
            color_cache,
            grad_holder
        )
        ctx.grid = ctx.rays = ctx.opt = None
        if not ctx.needs_input_grad[0]:
            grad_density_grid = None
        if not ctx.needs_input_grad[1]:
            grad_sh_grid = None
        if not ctx.needs_input_grad[2]:
            grad_basis = None
        if not ctx.needs_input_grad[3]:
            grad_background = None
        ctx.basis_data = None

        return grad_density_grid, grad_sh_grid, grad_basis, grad_background, \
               None, None, None, None

class _SurfTravRenderFunction(autograd.Function):
    @staticmethod
    def forward(
        ctx,
        data_density: torch.Tensor,
        data_surface: torch.Tensor,
        data_sh: torch.Tensor,
        data_basis: torch.Tensor,
        data_background: torch.Tensor,
        data_fake_sample_std: torch.Tensor,
        grid,
        rays,
        opt,
        backend: str,
    ):
        cu_fn = _C.__dict__[f"volume_render_{backend}"]
        color = cu_fn(grid, rays, opt)
        ctx.save_for_backward(color)
        ctx.grid = grid
        ctx.rays = rays
        ctx.opt = opt
        ctx.backend = backend
        ctx.basis_data = data_basis
        return color

    @staticmethod
    def backward(ctx, grad_out): 
        # this backward is only used for testing. Use fused MSE method in actual training instead
        # grad_out is the gradient for color
        # import pdb; pdb.set_trace()
        (color_cache,) = ctx.saved_tensors
        cu_fn = _C.__dict__[f"volume_render_{ctx.backend}_backward"]
        grad_density_grid = torch.zeros_like(ctx.grid.density_data.data)
        grad_surface_grid = torch.zeros_like(ctx.grid.surface_data.data)
        grad_sh_grid = torch.zeros_like(ctx.grid.sh_data.data)
        grad_fs_std_grid = torch.zeros_like(ctx.grid.density_data.data[:1])
        if ctx.grid.basis_type == BASIS_TYPE_MLP:
            grad_basis = torch.zeros_like(ctx.basis_data)
        elif ctx.grid.basis_type == BASIS_TYPE_3D_TEXTURE:
            grad_basis = torch.zeros_like(ctx.grid.basis_data.data)
        if ctx.grid.background_data is not None:
            grad_background = torch.zeros_like(ctx.grid.background_data.data)
        grad_holder = _C.GridOutputGrads()
        grad_holder.grad_density_out = grad_density_grid
        grad_holder.grad_surface_out = grad_surface_grid
        grad_holder.grad_sh_out = grad_sh_grid
        grad_holder.grad_fake_sample_std_out = grad_fs_std_grid

        # if torch.is_tensor(ctx.grid.fake_sample_std):
        #     ctx.grid.fake_sample_std.grad = torch.zeros_like(ctx.grid.fake_sample_std.data)
        #     grad_holder.grad_fake_sample_std_out = ctx.grid.fake_sample_std.grad


        if ctx.needs_input_grad[3]:
            grad_holder.grad_basis_out = grad_basis # for SH type, basis needs no grad
        if ctx.grid.background_data is not None and ctx.needs_input_grad[4]:
            grad_holder.grad_background_out = grad_background
        cu_fn(
            ctx.grid,
            ctx.rays,
            ctx.opt,
            grad_out.contiguous(),
            color_cache,
            grad_holder
        )
        ctx.grid = ctx.rays = ctx.opt = None
        if not ctx.needs_input_grad[0]:
            grad_density_grid = None
        if not ctx.needs_input_grad[1]:
            grad_surface_grid = None
        if not ctx.needs_input_grad[2]:
            grad_sh_grid = None
        if not ctx.needs_input_grad[3]:
            grad_basis = None
        if not ctx.needs_input_grad[4]:
            grad_background = None
        if not ctx.needs_input_grad[5]:
            grad_background = None
        ctx.basis_data = None

        return grad_density_grid, grad_surface_grid, grad_sh_grid, grad_basis, grad_background, \
               grad_fs_std_grid, None, None, None, None

class _SurfaceRenderFunction(autograd.Function):
    @staticmethod
    def forward(
        ctx,
        data_density: torch.Tensor,
        data_surface: torch.Tensor,
        data_sh: torch.Tensor,
        data_basis: torch.Tensor,
        data_background: torch.Tensor,
        grid,
        rays,
        ray_vox,
        opt,
        backend: str,
    ):
        cu_fn = _C.__dict__[f"volume_render_{backend}"]
        color = cu_fn(grid, rays, ray_vox, opt)
        ctx.save_for_backward(color)
        ctx.grid = grid
        ctx.rays = rays
        ctx.ray_vox = ray_vox
        ctx.opt = opt
        ctx.backend = backend
        ctx.basis_data = data_basis
        return color

    @staticmethod
    def backward(ctx, grad_out): 
        # this backward is only used for testing. Use fused MSE method in actual training instead
        # grad_out is the gradient for color
        # import pdb; pdb.set_trace()
        (color_cache,) = ctx.saved_tensors
        cu_fn = _C.__dict__[f"volume_render_{ctx.backend}_backward"]
        grad_density_grid = torch.zeros_like(ctx.grid.density_data.data)
        grad_surface_grid = torch.zeros_like(ctx.grid.surface_data.data)
        grad_sh_grid = torch.zeros_like(ctx.grid.sh_data.data)
        if ctx.grid.basis_type == BASIS_TYPE_MLP:
            grad_basis = torch.zeros_like(ctx.basis_data)
        elif ctx.grid.basis_type == BASIS_TYPE_3D_TEXTURE:
            grad_basis = torch.zeros_like(ctx.grid.basis_data.data)
        if ctx.grid.background_data is not None:
            grad_background = torch.zeros_like(ctx.grid.background_data.data)
        grad_holder = _C.GridOutputGrads()
        grad_holder.grad_density_out = grad_density_grid
        grad_holder.grad_surface_out = grad_surface_grid
        grad_holder.grad_sh_out = grad_sh_grid
        if ctx.needs_input_grad[3]:
            grad_holder.grad_basis_out = grad_basis # for SH type, basis needs no grad
        if ctx.grid.background_data is not None and ctx.needs_input_grad[4]:
            grad_holder.grad_background_out = grad_background
        cu_fn(
            ctx.grid,
            ctx.rays,
            ctx.ray_vox,
            ctx.opt,
            grad_out.contiguous(),
            color_cache,
            grad_holder
        )
        ctx.grid = ctx.rays = ctx.opt = None
        if not ctx.needs_input_grad[0]:
            grad_density_grid = None
        if not ctx.needs_input_grad[1]:
            grad_surface_grid = None
        if not ctx.needs_input_grad[2]:
            grad_sh_grid = None
        if not ctx.needs_input_grad[3]:
            grad_basis = None
        if not ctx.needs_input_grad[4]:
            grad_background = None
        ctx.basis_data = None

        return grad_density_grid, grad_surface_grid, grad_sh_grid, grad_basis, grad_background, \
               None, None, None, None, None


class _TotalVariationFunction(autograd.Function):
    @staticmethod
    def forward(
        ctx,
        data: torch.Tensor,
        links: torch.Tensor,
        start_dim: int,
        end_dim: int,
        use_logalpha: bool,
        logalpha_delta: float,
        ignore_edge: bool,
        ndc_coeffs: Tuple[float, float]
    ):
        tv = _C.tv(links, data, start_dim, end_dim, use_logalpha,
                   logalpha_delta, ignore_edge, ndc_coeffs[0], ndc_coeffs[1])
        ctx.save_for_backward(links, data)
        ctx.start_dim = start_dim
        ctx.end_dim = end_dim
        ctx.use_logalpha = use_logalpha
        ctx.logalpha_delta = logalpha_delta
        ctx.ignore_edge = ignore_edge
        ctx.ndc_coeffs = ndc_coeffs
        return tv

    @staticmethod
    def backward(ctx, grad_out):
        links, data = ctx.saved_tensors
        grad_grid = torch.zeros_like(data)
        _C.tv_grad(links, data, ctx.start_dim, ctx.end_dim, 1.0,
                   ctx.use_logalpha, ctx.logalpha_delta,
                   ctx.ignore_edge,
                   ctx.ndc_coeffs[0], ctx.ndc_coeffs[1],
                   grad_grid)
        ctx.start_dim = ctx.end_dim = None
        if not ctx.needs_input_grad[0]:
            grad_grid = None
        return grad_grid, None, None, None,\
               None, None, None, None


# END Differentiable CUDA functions with custom gradient

class _SPC():
    def __init__(self, octree=None, length=None, feature=None, max_level=None, pyramid=None, exsum=None, point_hierarchy=None):
        self.octree = octree
        self.length = length
        self.feature = feature
        self.max_level = max_level
        self.pyramid = pyramid
        self.exsum = exsum
        self.point_hierarchy = point_hierarchy

class SparseGrid(nn.Module):
    """
    Main sparse grid data structure.
    initially it will be a dense grid of resolution <reso>.
    Only float32 is supported.

    :param reso: int or List[int, int, int], resolution for resampled grid, as in the constructor
    :param radius: float or List[float, float, float], the 1/2 side length of the grid, optionally in each direction
    :param center: float or List[float, float, float], the center of the grid
    :param basis_type: int, basis type; may use svox2.BASIS_TYPE_* (1 = SH, 4 = learned 3D texture, 255 = learned MLP)
    :param basis_dim: int, size of basis / number of SH components
                           (must be square number in case of SH)
    :param basis_reso: int, resolution of grid if using BASIS_TYPE_3D_TEXTURE
    :param use_z_order: bool, if true, stores the data initially in a Z-order curve if possible
    :param mlp_posenc_size: int, if using BASIS_TYPE_MLP, then enables standard axis-aligned positional encoding of
                                 given size on MLP; if 0 then does not use positional encoding
    :param mlp_width: int, if using BASIS_TYPE_MLP, specifies MLP width (hidden dimension)
    :param device: torch.device, device to store the grid
    """

    def __init__(
        self,
        reso: Union[int, List[int], Tuple[int, int, int]] = 128,
        radius: Union[float, List[float]] = 1.0,
        center: Union[float, List[float]] = [0.0, 0.0, 0.0],
        basis_type: int = BASIS_TYPE_SH,
        basis_dim: int = 9,  # SH/learned basis size; in SH case, square number
        basis_reso: int = 16,  # Learned basis resolution (x^3 embedding grid)
        use_z_order : bool=False,
        use_sphere_bound : bool=False,
        mlp_posenc_size : int = 0,
        mlp_width : int = 16,
        background_nlayers : int = 0,  # BG MSI layers
        background_reso : int = 256,  # BG MSI cubemap face size
        device: Union[torch.device, str] = "cpu",
        surface_type: int = SURFACE_TYPE_NONE,
        surface_init: str = None, # methods used to init sdf data
        use_octree: bool = True,
        trainable_fake_sample_std: bool = False,
        force_alpha: bool = False, # clamp alpha to be non-trivial to force surface learning
    ):
        super().__init__()
        self.basis_type = basis_type
        self.surface_type = surface_type
        self.step_id = 0
        self.force_alpha = force_alpha
        if basis_type == BASIS_TYPE_SH:
            assert utils.isqrt(basis_dim) is not None, "basis_dim (SH) must be a square number"
        assert (
            basis_dim >= 1 and basis_dim <= utils.MAX_SH_BASIS
        ), f"basis_dim 1-{utils.MAX_SH_BASIS} supported"
        self.basis_dim = basis_dim

        self.mlp_posenc_size = mlp_posenc_size
        self.mlp_width = mlp_width

        self.background_nlayers = background_nlayers
        assert background_nlayers == 0 or background_nlayers > 1, "Please use at least 2 MSI layers (trilerp limitation)"
        self.background_reso = background_reso

        if isinstance(reso, int):
            reso = [reso] * 3
        else:
            assert (
                len(reso) == 3
            ), "reso must be an integer or indexable object of 3 ints"

        if use_z_order and not (reso[0] == reso[1] and reso[0] == reso[2] and utils.is_pow2(reso[0])):
            print("Morton code requires a cube grid of power-of-2 size, ignoring...")
            use_z_order = False

        if isinstance(radius, float) or isinstance(radius, int):
            radius = [radius] * 3
        if isinstance(radius, torch.Tensor):
            radius = radius.to(device="cpu", dtype=torch.float32)
        else:
            radius = torch.tensor(radius, dtype=torch.float32, device="cpu")
        if isinstance(center, torch.Tensor):
            center = center.to(device="cpu", dtype=torch.float32)
        else:
            center = torch.tensor(center, dtype=torch.float32, device="cpu")

        self.radius: torch.Tensor = radius  # CPU
        self.center: torch.Tensor = center  # CPU
        self._offset = 0.5 * (1.0 - self.center / self.radius)
        self._scaling = 0.5 / self.radius

        n3: int = reduce(lambda x, y: x * y, reso)
        if use_z_order:
            init_links = utils.gen_morton(reso[0], device=device, dtype=torch.int32).flatten()
        else:
            init_links = torch.arange(n3, device=device, dtype=torch.int32)

        if use_sphere_bound:
            # only keeping grids in a sphere
            X = torch.arange(reso[0], dtype=torch.float32, device=device) - 0.5
            Y = torch.arange(reso[1], dtype=torch.float32, device=device) - 0.5
            Z = torch.arange(reso[2], dtype=torch.float32, device=device) - 0.5
            X, Y, Z = torch.meshgrid(X, Y, Z)
            points = torch.stack((X, Y, Z), dim=-1).view(-1, 3)
            gsz = torch.tensor(reso)
            roffset = 1.0 / gsz - 1.0
            rscaling = 2.0 / gsz
            points = torch.addcmul(
                roffset.to(device=points.device),
                points,
                rscaling.to(device=points.device),
            )

            norms = points.norm(dim=-1)
            mask = norms <= 1.0 + (3 ** 0.5) / gsz.max()
            self.capacity: int = mask.sum()

            data_mask = torch.zeros(n3, dtype=torch.int32, device=device)
            idxs = init_links[mask].long()
            data_mask[idxs] = 1
            data_mask = torch.cumsum(data_mask, dim=0) - 1

            init_links[mask] = data_mask[idxs].int()
            init_links[~mask] = -1
        else:
            self.capacity = n3

        self.density_data = nn.Parameter(
            torch.zeros(self.capacity, 1, dtype=torch.float32, device=device)
        )
        # Called sh for legacy reasons, but it's just the coeffients for whatever
        # spherical basis functions
        self.sh_data = nn.Parameter(
            torch.zeros(
                self.capacity, self.basis_dim * 3, dtype=torch.float32, device=device
            )
        )

        if self.basis_type == BASIS_TYPE_3D_TEXTURE:
            # Unit sphere embedded in a cube
            self.basis_data = nn.Parameter(
                torch.zeros(
                    basis_reso, basis_reso, basis_reso,
                    self.basis_dim, dtype=torch.float32, device=device
                )
            )
        elif self.basis_type == BASIS_TYPE_MLP:
            D_rgb = mlp_width
            dir_in_dims = 3 + 6 * self.mlp_posenc_size
            # Hard-coded 4 layer MLP
            self.basis_mlp = nn.Sequential(
                nn.Linear(dir_in_dims, D_rgb),
                nn.ReLU(),
                nn.Linear(D_rgb, D_rgb),
                nn.ReLU(),
                nn.Linear(D_rgb, D_rgb),
                nn.ReLU(),
                nn.Linear(D_rgb, self.basis_dim)
            )
            self.basis_mlp = self.basis_mlp.to(device=self.sh_data.device)
            self.basis_mlp.apply(utils.init_weights)
            self.basis_data = nn.Parameter(
                torch.empty(
                    0, 0, 0, 0, dtype=torch.float32, device=device
                ),
                requires_grad=False
            )
        else:
            self.basis_data = nn.Parameter(
                torch.empty(
                    0, 0, 0, 0, dtype=torch.float32, device=device
                ),
                requires_grad=False
            )

        self.background_links: Optional[torch.Tensor]
        self.background_data: Optional[torch.Tensor]
        if self.use_background:
            background_capacity = (self.background_reso ** 2) * 2
            background_links = torch.arange(
                background_capacity,
                dtype=torch.int32, device=device
            ).reshape(self.background_reso * 2, self.background_reso)
            self.register_buffer('background_links', background_links)
            self.background_data = nn.Parameter(
                torch.zeros(
                    background_capacity,
                    self.background_nlayers,
                    4,
                    dtype=torch.float32, device=device
                )
            )
        else:
            self.background_data = nn.Parameter(
                torch.empty(
                    0, 0, 0,
                    dtype=torch.float32, device=device
                ),
                requires_grad=False
            )

        self.register_buffer("links", init_links.view(reso))
        self.links: torch.Tensor

        self.trainable_data = ["density_data", "sh_data", "basis_data", "background_data"]

        self.surface_data = None
        self.fake_sample_std = None # variance parameter for fake samples
        if trainable_fake_sample_std:
            self.fake_sample_std = nn.Parameter(
                    torch.tensor([[1.]], dtype=torch.float32, device=device)
                )
            self.trainable_data.append("fake_sample_std")
        surface_data = None
        self.level_set_data = None
        if surface_type == SURFACE_TYPE_SDF:
            # surface_init = None
            level_sets = torch.tensor([0.])
            level_sets = level_sets.to(device)
            self.level_set_data = level_sets
            if surface_init is None:
                surface_data = torch.zeros(self.capacity, 1, dtype=torch.float32, device=device)
            elif surface_init == 'sphere':
                # method 1: initialize with distance to grid center, then reduce each vertices by the mean from the 8?
                surface_data = torch.zeros(self.capacity, 1, dtype=torch.float32, device=device)
                coords = torch.meshgrid(torch.arange(reso[0]), torch.arange(reso[1]), torch.arange(reso[2]))
                coords = torch.stack(coords).view(3, -1).T
                # grid_center = (torch.tensor(reso) - 1.) / 2
                grid_center = (torch.tensor(reso)) / 2
                rs = torch.sqrt(torch.sum((coords - grid_center)**2, axis=-1)).to(device)

                sphere_rs = torch.arange(0, torch.sqrt(torch.sum((torch.tensor(reso)/2) ** 2)) , 6) + 0.5
                sphere_rs = sphere_rs.to(device)
                dists = rs[:, None] - sphere_rs[None, :]

                links = self.links[coords[:, 0], coords[:, 1], coords[:, 2]]
                surface_data[links.long(), 0] = dists[torch.arange(dists.shape[0]), torch.abs(dists).min(axis=-1).indices]


                # floors = torch.floor(rs - 0.5)
                # # ids where sdf values should be positive
                # pos_ids = torch.arange(coords.shape[0])[floors % 2 == 0]
                # # ids where sdf values should be negative
                # neg_ids = torch.arange(coords.shape[0])[floors % 2 == 1]

                # # fetch links
                # links = self.links[coords[:, 0], coords[:, 1], coords[:, 2]]
                #     # surface_data[links[pos_ids].long(), 0] = rs[pos_ids]
                #     # surface_data[links[neg_ids].long(), 0] = rs[neg_ids] * -1.
                #     surface_data[links[pos_ids].long(), 0] = rs[pos_ids] - (floors[pos_ids] + 0.5)
                #     surface_data[links[neg_ids].long(), 0] = rs[neg_ids] - (floors[neg_ids] + 1.5)
            elif surface_init == 'single_sphere':
                surface_data = torch.zeros(self.capacity, 1, dtype=torch.float32, device=device)
                coords = torch.meshgrid(torch.arange(reso[0]), torch.arange(reso[1]), torch.arange(reso[2]))
                coords = torch.stack(coords).view(3, -1).T
                grid_center = (torch.tensor(reso)) / 2
                rs = torch.sqrt(torch.sum((coords - grid_center)**2, axis=-1)).to(device)

                links = self.links[coords[:, 0], coords[:, 1], coords[:, 2]]
                surface_data[links.long(), 0] = rs[torch.arange(rs.shape[0])] - (torch.norm(grid_center, keepdim=True).to(device) / 2.)

                surface_data = surface_data * 10. / rs.max()


            elif surface_init == 'outwards':
                # method 2: init as random surface facing outwards
                surface_data = torch.rand(self.capacity, 1, dtype=torch.float32, device=device)
                grid_center = torch.tensor(reso) / 2

                ids = torch.meshgrid(torch.arange(reso[0]), torch.arange(reso[1]), torch.arange(reso[2]))
                ids = torch.stack(ids).view(3, -1).T

                abs_max_axis = torch.abs(ids- grid_center.long()).max(axis=-1).indices
                for axis_id in range(len(reso)):
                    # find list of coordinates where the i-th axis is maximum
                    coords = ids[abs_max_axis == axis_id]
                    # list of coords where sdf values need to be negated
                    neg_coords = coords[coords[:,axis_id] % 2 == 0]

                    # fech link ids
                    links = self.links[neg_coords[:, 0], neg_coords[:, 1], neg_coords[:, 2]]
                    # set surface to face outwards
                    surface_data[links.long()] *= -1

            elif surface_init == 'plane_init':
                # method 3: simple plane with fixed driection
                surface_data = torch.rand(self.capacity, 1, dtype=torch.float32, device=device) * 0.1 + 1
                surface_data[self.links[np.arange(1,reso[0],2),:,:].view(-1).long()] *= -1.
            else:
                raise NotImplementedError(f'Surface initialization [{surface_init}] is not supported for grid [{surface_type}]')
        
        elif surface_type == SURFACE_TYPE_PLANE:
            if surface_init == 'random' or surface_init is None:
                surface_data = torch.rand(self.capacity, 4, dtype=torch.float32, device=device) -.5 # allow negative values
                # ax + by + cz + d = 0
                # normalize a,b,c
                # surface_data[:, :3] = surface_data[:, :3] / torch.sqrt(torch.sum(surface_data[:, :3]**2, axis=-1))
                surface_data[:, :3] = surface_data[:, :3] / torch.norm(surface_data[:, :3], dim=-1, keepdim=True)

                # modify d to initialize all vertices to have planes located exactly on them
                coords = torch.meshgrid(torch.arange(reso[0]), torch.arange(reso[1]), torch.arange(reso[2]))
                coords = torch.stack(coords).view(3, -1).T
                links = self.links[coords[:, 0], coords[:, 1], coords[:, 2]]
                surface_data[links.long(), 3] = -torch.sum(coords.to(device) * surface_data[links.long(), :3], axis=-1)
                # surface_data[links.long(), 3] = -torch.sum((coords.to(device) + 0.5) * surface_data[links.long(), :3], axis=-1)
            elif surface_init == 'sphere':
                surface_data = torch.zeros(self.capacity, 4, dtype=torch.float32, device=device)
                grid_center = torch.tensor(reso) / 2

                coords = torch.meshgrid(torch.arange(reso[0]), torch.arange(reso[1]), torch.arange(reso[2]))
                coords = torch.stack(coords).view(3, -1).T
                links = self.links[coords[:, 0], coords[:, 1], coords[:, 2]]

                norm_dirs = (coords - grid_center).to(device)
                norm_dirs = norm_dirs / torch.norm(norm_dirs, dim=-1, keepdim=True)
                norm_dirs = torch.nan_to_num(norm_dirs, 1./np.sqrt(3))

                surface_data[links.long(), :3] = norm_dirs

                # modify d to initialize all vertices to have planes located exactly on them
                surface_data[links.long(), 3] = -torch.sum(coords.to(device) * surface_data[links.long(), :3], axis=-1)

            else:
                raise NotImplementedError(f'Surface initialization [{surface_init}] is not supported for grid [{surface_type}]')
        elif surface_type == SURFACE_TYPE_UDF or surface_type == SURFACE_TYPE_UDF_ALPHA \
            or surface_type == SURFACE_TYPE_UDF_FAKE_SAMPLE:
            # unsigned distance field with fixed level sets
            # udf_alpha: each level set has an alpha, instead of each vertex
            if surface_init is None:
                surface_data = torch.zeros(self.capacity, 1, dtype=torch.float32, device=device)
                level_sets = torch.tensor([64.])
                level_sets = level_sets.to(device)
            elif surface_init == 'sphere':
                surface_data = torch.zeros(self.capacity, 1, dtype=torch.float32, device=device)
                coords = torch.meshgrid(torch.arange(reso[0]), torch.arange(reso[1]), torch.arange(reso[2]))
                coords = torch.stack(coords).view(3, -1).T
                grid_center = (torch.tensor(reso)) / 2
                rs = torch.sqrt(torch.sum((coords - grid_center)**2, axis=-1)).to(device)

                links = self.links[coords[:, 0], coords[:, 1], coords[:, 2]]
                surface_data[links.long(), 0] = rs[torch.arange(rs.shape[0])]

                level_sets = torch.arange(0, torch.sqrt(torch.sum((torch.tensor(reso)/2) ** 2)), 4) + 0.5
                level_sets = level_sets.to(device)

                # # invert softplus activation
                # surface_data = surface_data + torch.log(-torch.expm1(-surface_data))
            elif surface_init == 'single_lv':
                # single level set with single sphere

                # level_sets = torch.norm(grid_center, keepdim=True) / 2.
                level_sets = torch.tensor([64.])
                level_sets = level_sets.to(device)

                surface_data = torch.zeros(self.capacity, 1, dtype=torch.float32, device=device)
                coords = torch.meshgrid(torch.arange(reso[0]), torch.arange(reso[1]), torch.arange(reso[2]))
                coords = torch.stack(coords).view(3, -1).T
                grid_center = (torch.tensor(reso)) / 2
                rs = torch.sqrt(torch.sum((coords - grid_center)**2, axis=-1)).to(device)

                links = self.links[coords[:, 0], coords[:, 1], coords[:, 2]]
                surface_data[links.long(), 0] = rs[torch.arange(rs.shape[0])] - (torch.norm(grid_center, keepdim=True).to(device) / 2. - level_sets[0])


            elif surface_init == 'single_lv_multi_sphere':
                # single level set with multi sphere
                grid_center = (torch.tensor(reso)) / 2
                # level_sets = torch.norm(grid_center, keepdim=True) / 2.
                level_sets = torch.tensor([64.])
                level_sets = level_sets.to(device)

                surface_data = torch.zeros(self.capacity, 1, dtype=torch.float32, device=device)
                coords = torch.meshgrid(torch.arange(reso[0]), torch.arange(reso[1]), torch.arange(reso[2]))
                coords = torch.stack(coords).view(3, -1).T
                # grid_center = (torch.tensor(reso) - 1.) / 2
                grid_center = (torch.tensor(reso)) / 2
                rs = torch.sqrt(torch.sum((coords - grid_center)**2, axis=-1)).to(device)

                sphere_rs = torch.arange(0, torch.sqrt(torch.sum((torch.tensor(reso)/2) ** 2)) , 4) + 0.5
                sphere_rs = sphere_rs.to(device)
                dists = rs[:, None] - sphere_rs[None, :]

                links = self.links[coords[:, 0], coords[:, 1], coords[:, 2]]
                surface_data[links.long(), 0] = dists[torch.arange(dists.shape[0]), torch.abs(dists).min(axis=-1).indices] + level_sets[0]


            
            else:
                raise NotImplementedError(f'Surface initialization [{surface_init}] is not supported for grid [{surface_type}]')

            self.level_set_data = level_sets
            if surface_type == SURFACE_TYPE_UDF_ALPHA:
                self.density_data = nn.Parameter(
                    torch.zeros(level_sets.numel(), 1, dtype=torch.float32, device=device)
                )
            

        elif surface_type == SURFACE_TYPE_VOXEL_FACE:
            surface_data = torch.zeros(self.capacity, 1, dtype=torch.float32, device=device)


        
        if surface_data is not None:
            self.surface_data = nn.Parameter(surface_data)
            self.trainable_data.append('surface_data')

        self.opt = RenderOptions() # set up outside of initializer
        self.sparse_grad_indexer: Optional[torch.Tensor] = None
        self.sparse_sh_grad_indexer: Optional[torch.Tensor] = None
        self.sparse_background_indexer: Optional[torch.Tensor] = None
        self.density_rms: Optional[torch.Tensor] = None
        self.surface_rms: Optional[torch.Tensor] = None
        self.fake_sample_std_rms: Optional[torch.Tensor] = None
        self.sh_rms: Optional[torch.Tensor] = None
        self.background_rms: Optional[torch.Tensor] = None
        self.basis_rms: Optional[torch.Tensor] = None
        self.use_octree = use_octree

        if use_octree:
            # create place holder feature grid
            # this grid is empty and is only used for ray-voxel intersection determination
            feature_grid = torch.ones(1, 1, reso[0], reso[1], reso[2]).to(device)
            octree, length, feature = kaolin.ops.spc.feature_grids_to_spc(feature_grid)
            max_level, pyramids, exsum = kaolin.ops.spc.scan_octrees(octree, length)
            point_hierarchy = kaolin.ops.spc.generate_points(octree, pyramids, exsum)
            
            self.spc = _SPC(octree, length[0], feature, max_level, pyramids[0], exsum, point_hierarchy)

        self._C = _C

        if self.links.is_cuda and use_sphere_bound and _C is not None:
            self.accelerate()

    @property
    def data_dim(self):
        """
        Get the number of channels in the data, including color + density
        (similar to svox 1)
        """
        return self.sh_data.size(1) + 1

    @property
    def basis_reso(self):
        """
        Return the resolution of the learned spherical function data if using
        3D learned texture, or 0 if only using SH
        """
        return self.basis_data.size(0) if self.BASIS_TYPE_3D_TEXTURE else 0

    @property
    def use_background(self):
        return self.background_nlayers > 0

    @property
    def shape(self):
        return list(self.links.shape) + [self.data_dim]

    def _fetch_links(self, links):
        results_sigma = torch.zeros(
            (links.size(0), 1), device=links.device, dtype=torch.float32
        )
        results_sh = torch.zeros(
            (links.size(0), self.sh_data.size(1)),
            device=links.device,
            dtype=torch.float32,
        )
        mask = links >= 0
        idxs = links[mask].long()
        if self.surface_type not in [SURFACE_TYPE_UDF_ALPHA]:
            results_sigma[mask] = self.density_data[idxs]
        results_sh[mask] = self.sh_data[idxs]

        if self.surface_type != SURFACE_TYPE_NONE:
            results_surface = torch.zeros(
                (links.size(0), self.surface_data.shape[-1]), device=links.device, dtype=torch.float32
            )
            results_surface[mask] = self.surface_data[idxs]
            return results_sigma, results_sh, results_surface

        return results_sigma, results_sh

    def sample(self, points: torch.Tensor,
               use_kernel: bool = True,
               grid_coords: bool = False,
               want_colors: bool = True):
        """
        Grid sampling with trilinear interpolation.
        Behaves like torch.nn.functional.grid_sample
        with padding mode border and align_corners=False (better for multi-resolution).

        Any voxel with link < 0 (empty) is considered to have 0 values in all channels
        prior to interpolating.

        :param points: torch.Tensor, (N, 3)
        :param use_kernel: bool, if false uses pure PyTorch version even if on CUDA.
        :param grid_coords: bool, if true then uses grid coordinates ([-0.5, reso[i]-0.5 ] in each dimension);
                                  more numerically exact for resampling
        :param want_colors: bool, if true (default) returns density and colors,
                            else returns density and a dummy tensor to be ignored
                            (much faster)

        :return: (density, color)
        """

        if use_kernel and self.links.is_cuda and _C is not None:
            assert points.is_cuda
            return _SampleGridAutogradFunction.apply(
                self.density_data, self.sh_data, self._to_cpp(grid_coords=grid_coords), points, want_colors
            )
        else:
            if not grid_coords:
                points = self.world2grid(points)
            points.clamp_min_(0.0)
            for i in range(3):
                points[:, i].clamp_max_(self.links.size(i) - 1)
            l = points.to(torch.long)
            for i in range(3):
                l[:, i].clamp_max_(self.links.size(i) - 2)
            wb = points - l
            wa = 1.0 - wb

            lx, ly, lz = l.unbind(-1)
            links000 = self.links[lx, ly, lz]
            links001 = self.links[lx, ly, lz + 1]
            links010 = self.links[lx, ly + 1, lz]
            links011 = self.links[lx, ly + 1, lz + 1]
            links100 = self.links[lx + 1, ly, lz]
            links101 = self.links[lx + 1, ly, lz + 1]
            links110 = self.links[lx + 1, ly + 1, lz]
            links111 = self.links[lx + 1, ly + 1, lz + 1]

            sigma000, rgb000 = self._fetch_links(links000)
            sigma001, rgb001 = self._fetch_links(links001)
            sigma010, rgb010 = self._fetch_links(links010)
            sigma011, rgb011 = self._fetch_links(links011)
            sigma100, rgb100 = self._fetch_links(links100)
            sigma101, rgb101 = self._fetch_links(links101)
            sigma110, rgb110 = self._fetch_links(links110)
            sigma111, rgb111 = self._fetch_links(links111)

            c00 = sigma000 * wa[:, 2:] + sigma001 * wb[:, 2:]
            c01 = sigma010 * wa[:, 2:] + sigma011 * wb[:, 2:]
            c10 = sigma100 * wa[:, 2:] + sigma101 * wb[:, 2:]
            c11 = sigma110 * wa[:, 2:] + sigma111 * wb[:, 2:]
            c0 = c00 * wa[:, 1:2] + c01 * wb[:, 1:2]
            c1 = c10 * wa[:, 1:2] + c11 * wb[:, 1:2]
            samples_sigma = c0 * wa[:, :1] + c1 * wb[:, :1]

            if want_colors:
                c00 = rgb000 * wa[:, 2:] + rgb001 * wb[:, 2:]
                c01 = rgb010 * wa[:, 2:] + rgb011 * wb[:, 2:]
                c10 = rgb100 * wa[:, 2:] + rgb101 * wb[:, 2:]
                c11 = rgb110 * wa[:, 2:] + rgb111 * wb[:, 2:]
                c0 = c00 * wa[:, 1:2] + c01 * wb[:, 1:2]
                c1 = c10 * wa[:, 1:2] + c11 * wb[:, 1:2]
                samples_rgb = c0 * wa[:, :1] + c1 * wb[:, :1]
            else:
                samples_rgb = torch.empty_like(self.sh_data[:0])

            return samples_sigma, samples_rgb


    def sample_surface(self, points: torch.Tensor,
                        use_kernel: bool = True,
                        grid_coords: bool = False,
                        want_colors: bool = True,
                        want_surfaces: bool = True):
        """
        Grid sampling with trilinear interpolation.
        Behaves like torch.nn.functional.grid_sample
        with padding mode border and align_corners=False (better for multi-resolution).

        Any voxel with link < 0 (empty) is considered to have 0 values in all channels
        prior to interpolating.

        :param points: torch.Tensor, (N, 3)
        :param use_kernel: bool, if false uses pure PyTorch version even if on CUDA.
        :param grid_coords: bool, if true then uses grid coordinates ([-0.5, reso[i]-0.5 ] in each dimension);
                                  more numerically exact for resampling
        :param want_colors: bool, if true (default) returns density and colors,
                            else returns density and a dummy tensor to be ignored
                            (much faster)

        :return: (alpha, color)
        """
        # if self.surface_type != SURFACE_TYPE_NONE:
        #     raise NotImplementedError

        if use_kernel and self.links.is_cuda and _C is not None:
            assert points.is_cuda

            return _C.sample_grid_sh_surf(self._to_cpp(grid_coords=grid_coords), points, want_colors, want_surfaces)
        else:
            raise NotImplementedError()
            if not grid_coords:
                points = self.world2grid(points)
            points.clamp_min_(0.0)
            for i in range(3):
                points[:, i].clamp_max_(self.links.size(i) - 1)
            l = points.to(torch.long)
            for i in range(3):
                l[:, i].clamp_max_(self.links.size(i) - 2)
            wb = points - l
            wa = 1.0 - wb

            lx, ly, lz = l.unbind(-1)
            links000 = self.links[lx, ly, lz]
            links001 = self.links[lx, ly, lz + 1]
            links010 = self.links[lx, ly + 1, lz]
            links011 = self.links[lx, ly + 1, lz + 1]
            links100 = self.links[lx + 1, ly, lz]
            links101 = self.links[lx + 1, ly, lz + 1]
            links110 = self.links[lx + 1, ly + 1, lz]
            links111 = self.links[lx + 1, ly + 1, lz + 1]

            sigma000, rgb000 = self._fetch_links(links000)
            sigma001, rgb001 = self._fetch_links(links001)
            sigma010, rgb010 = self._fetch_links(links010)
            sigma011, rgb011 = self._fetch_links(links011)
            sigma100, rgb100 = self._fetch_links(links100)
            sigma101, rgb101 = self._fetch_links(links101)
            sigma110, rgb110 = self._fetch_links(links110)
            sigma111, rgb111 = self._fetch_links(links111)

            c00 = sigma000 * wa[:, 2:] + sigma001 * wb[:, 2:]
            c01 = sigma010 * wa[:, 2:] + sigma011 * wb[:, 2:]
            c10 = sigma100 * wa[:, 2:] + sigma101 * wb[:, 2:]
            c11 = sigma110 * wa[:, 2:] + sigma111 * wb[:, 2:]
            c0 = c00 * wa[:, 1:2] + c01 * wb[:, 1:2]
            c1 = c10 * wa[:, 1:2] + c11 * wb[:, 1:2]
            samples_sigma = c0 * wa[:, :1] + c1 * wb[:, :1]

            if want_colors:
                c00 = rgb000 * wa[:, 2:] + rgb001 * wb[:, 2:]
                c01 = rgb010 * wa[:, 2:] + rgb011 * wb[:, 2:]
                c10 = rgb100 * wa[:, 2:] + rgb101 * wb[:, 2:]
                c11 = rgb110 * wa[:, 2:] + rgb111 * wb[:, 2:]
                c0 = c00 * wa[:, 1:2] + c01 * wb[:, 1:2]
                c1 = c10 * wa[:, 1:2] + c11 * wb[:, 1:2]
                samples_rgb = c0 * wa[:, :1] + c1 * wb[:, :1]
            else:
                samples_rgb = torch.empty_like(self.sh_data[:0])

            return samples_sigma, samples_rgb


    def forward(self, points: torch.Tensor, use_kernel: bool = True):
        return self.sample(points, use_kernel=use_kernel)

    def _volume_render_gradcheck_lerp(self, rays: Rays, return_raylen: bool=False):
        """
        trilerp gradcheck version
        """
        origins = self.world2grid(rays.origins)
        dirs = rays.dirs / torch.norm(rays.dirs, dim=-1, keepdim=True)
        viewdirs = dirs
        B = dirs.size(0)
        assert origins.size(0) == B
        gsz = self._grid_size()
        dirs = dirs * (self._scaling * gsz).to(device=dirs.device)
        delta_scale = 1.0 / dirs.norm(dim=1)
        dirs *= delta_scale.unsqueeze(-1)

        if self.basis_type == BASIS_TYPE_3D_TEXTURE:
            sh_mult = self._eval_learned_bases(viewdirs)
        elif self.basis_type == BASIS_TYPE_MLP:
            sh_mult = torch.sigmoid(self._eval_basis_mlp(viewdirs))
        else:
            sh_mult = utils.eval_sh_bases(self.basis_dim, viewdirs)
        invdirs = 1.0 / dirs

        gsz = self._grid_size()
        gsz_cu = gsz.to(device=dirs.device)
        t1 = (-0.5 - origins) * invdirs # origin + t1 * dir = (-0.5, -0.5, -0.5)
        t2 = (gsz_cu - 0.5 - origins) * invdirs

        t = torch.min(t1, t2)
        t[dirs == 0] = -1e9
        t = torch.max(t, dim=-1).values.clamp_min_(self.opt.near_clip)

        tmax = torch.max(t1, t2)
        tmax[dirs == 0] = 1e9
        tmax = torch.min(tmax, dim=-1).values
        if return_raylen:
            return tmax - t

        log_light_intensity = torch.zeros(B, device=origins.device)
        out_rgb = torch.zeros((B, 3), device=origins.device)
        good_indices = torch.arange(B, device=origins.device)

        origins_ini = origins
        dirs_ini = dirs

        mask = t <= tmax
        good_indices = good_indices[mask]
        origins = origins[mask]
        dirs = dirs[mask]

        #  invdirs = invdirs[mask]
        del invdirs
        t = t[mask]
        sh_mult = sh_mult[mask]
        tmax = tmax[mask]

        while good_indices.numel() > 0:
            pos = origins + t[:, None] * dirs
            pos = pos.clamp_min_(0.0)
            pos[:, 0] = torch.clamp_max(pos[:, 0], gsz_cu[0] - 1)
            pos[:, 1] = torch.clamp_max(pos[:, 1], gsz_cu[1] - 1)
            pos[:, 2] = torch.clamp_max(pos[:, 2], gsz_cu[2] - 1)
            #  print('pym', pos, log_light_intensity)

            l = pos.to(torch.long)
            l.clamp_min_(0)
            l[:, 0] = torch.clamp_max(l[:, 0], gsz_cu[0].long() - 2)
            l[:, 1] = torch.clamp_max(l[:, 1], gsz_cu[1].long() - 2)
            l[:, 2] = torch.clamp_max(l[:, 2], gsz_cu[2].long() - 2)
            pos -= l

            # BEGIN CRAZY TRILERP
            lx, ly, lz = l.unbind(-1)
            links000 = self.links[lx, ly, lz]
            links001 = self.links[lx, ly, lz + 1]
            links010 = self.links[lx, ly + 1, lz]
            links011 = self.links[lx, ly + 1, lz + 1]
            links100 = self.links[lx + 1, ly, lz]
            links101 = self.links[lx + 1, ly, lz + 1]
            links110 = self.links[lx + 1, ly + 1, lz]
            links111 = self.links[lx + 1, ly + 1, lz + 1]

            sigma000, rgb000 = self._fetch_links(links000)
            sigma001, rgb001 = self._fetch_links(links001)
            sigma010, rgb010 = self._fetch_links(links010)
            sigma011, rgb011 = self._fetch_links(links011)
            sigma100, rgb100 = self._fetch_links(links100)
            sigma101, rgb101 = self._fetch_links(links101)
            sigma110, rgb110 = self._fetch_links(links110)
            sigma111, rgb111 = self._fetch_links(links111)

            wa, wb = 1.0 - pos, pos
            c00 = sigma000 * wa[:, 2:] + sigma001 * wb[:, 2:]
            c01 = sigma010 * wa[:, 2:] + sigma011 * wb[:, 2:]
            c10 = sigma100 * wa[:, 2:] + sigma101 * wb[:, 2:]
            c11 = sigma110 * wa[:, 2:] + sigma111 * wb[:, 2:]
            c0 = c00 * wa[:, 1:2] + c01 * wb[:, 1:2]
            c1 = c10 * wa[:, 1:2] + c11 * wb[:, 1:2]
            sigma = c0 * wa[:, :1] + c1 * wb[:, :1]

            c00 = rgb000 * wa[:, 2:] + rgb001 * wb[:, 2:]
            c01 = rgb010 * wa[:, 2:] + rgb011 * wb[:, 2:]
            c10 = rgb100 * wa[:, 2:] + rgb101 * wb[:, 2:]
            c11 = rgb110 * wa[:, 2:] + rgb111 * wb[:, 2:]
            c0 = c00 * wa[:, 1:2] + c01 * wb[:, 1:2]
            c1 = c10 * wa[:, 1:2] + c11 * wb[:, 1:2]
            rgb = c0 * wa[:, :1] + c1 * wb[:, :1]

            # END CRAZY TRILERP

            log_att = (
                -self.opt.step_size
                * torch.relu(sigma[..., 0])
                * delta_scale[good_indices]
            ) # alpha
            weight = torch.exp(log_light_intensity[good_indices]) * (
                1.0 - torch.exp(log_att)
            )
            # [B', 3, n_sh_coeffs]
            rgb_sh = rgb.reshape(-1, 3, self.basis_dim)
            rgb = torch.clamp_min(
                torch.sum(sh_mult.unsqueeze(-2) * rgb_sh, dim=-1) + 0.5,
                0.0,
            )  # [B', 3]
            rgb = weight[:, None] * rgb[:, :3]

            out_rgb[good_indices] += rgb
            log_light_intensity[good_indices] += log_att
            t += self.opt.step_size

            mask = t <= tmax
            good_indices = good_indices[mask]
            origins = origins[mask]
            dirs = dirs[mask]
            #  invdirs = invdirs[mask]
            t = t[mask]
            sh_mult = sh_mult[mask]
            tmax = tmax[mask]

        if self.use_background:
            # Render the MSI background model
            csi = utils.ConcentricSpheresIntersector(
                    gsz_cu,
                    origins_ini,
                    dirs_ini,
                    delta_scale)
            inner_radius = torch.cross(csi.origins, csi.dirs, dim=-1).norm(dim=-1) + 1e-3
            inner_radius = inner_radius.clamp_min(1.0)
            _, t_last = csi.intersect(inner_radius)
            n_steps = int(self.background_nlayers / self.opt.step_size) + 2
            layer_scale = (self.background_nlayers - 1) / (n_steps + 1)

            def fetch_bg_link(lx, ly, lz):
                results = torch.zeros([lx.shape[0], self.background_data.size(-1)],
                                        device=lx.device)
                lnk = self.background_links[lx, ly]
                mask = lnk >= 0
                results[mask] = self.background_data[lnk[mask].long(), lz[mask]]
                return results

            for i in range(n_steps):
                r : float = n_steps / (n_steps - i - 0.5)
                normalized_inv_radius = min((i + 1) * layer_scale, self.background_nlayers - 1)
                layerid = min(int(normalized_inv_radius), self.background_nlayers - 2);
                interp_wt = normalized_inv_radius - layerid;

                active_mask, t = csi.intersect(r)
                active_mask = active_mask & (r >= inner_radius)
                if active_mask.count_nonzero() == 0:
                    continue
                t_sub = t[active_mask]
                t_mid_sub = (t_sub + t_last[active_mask]) * 0.5
                sphpos = csi.origins[active_mask] + \
                         t_mid_sub.unsqueeze(-1) * csi.dirs[active_mask]
                invr_mid = 1.0 / torch.norm(sphpos, dim=-1)
                sphpos *= invr_mid.unsqueeze(-1)

                xy = utils.xyz2equirect(sphpos, self.background_links.size(1))
                z = torch.clamp((1.0 - invr_mid) * self.background_nlayers - 0.5, 0.0,
                               self.background_nlayers - 1);
                points = torch.cat([xy, z.unsqueeze(-1)], dim=-1)
                l = points.to(torch.long)
                l[..., 0].clamp_max_(self.background_links.size(0) - 1)
                l[..., 1].clamp_max_(self.background_links.size(1) - 1)
                l[..., 2].clamp_max_(self.background_nlayers - 2)

                wb = points - l
                wa = 1.0 - wb
                lx, ly, lz = l.unbind(-1)
                lnx = (lx + 1) % self.background_links.size(0)
                lny = (ly + 1) % self.background_links.size(1)
                lnz = lz + 1

                v000 = fetch_bg_link(lx, ly, lz)
                v001 = fetch_bg_link(lx, ly, lnz)
                v010 = fetch_bg_link(lx, lny, lz)
                v011 = fetch_bg_link(lx, lny, lnz)
                v100 = fetch_bg_link(lnx, ly, lz)
                v101 = fetch_bg_link(lnx, ly, lnz)
                v110 = fetch_bg_link(lnx, lny, lz)
                v111 = fetch_bg_link(lnx, lny, lnz)

                c00 = v000 * wa[:, 2:] + v001 * wb[:, 2:]
                c01 = v010 * wa[:, 2:] + v011 * wb[:, 2:]
                c10 = v100 * wa[:, 2:] + v101 * wb[:, 2:]
                c11 = v110 * wa[:, 2:] + v111 * wb[:, 2:]
                c0 = c00 * wa[:, 1:2] + c01 * wb[:, 1:2]
                c1 = c10 * wa[:, 1:2] + c11 * wb[:, 1:2]
                rgba = c0 * wa[:, :1] + c1 * wb[:, :1]

                log_att = -csi.world_step_scale[active_mask] * torch.relu(rgba[:, -1]) * (
                            t_sub - t_last[active_mask]
                        )
                weight = torch.exp(log_light_intensity[active_mask]) * (
                    1.0 - torch.exp(log_att)
                )
                rgb = torch.clamp_min(rgba[:, :3] * utils.SH_C0 + 0.5, 0.0)
                out_rgb[active_mask] += rgb * weight[:, None]
                log_light_intensity[active_mask] += log_att
                t_last[active_mask] = t[active_mask]

        # Add background color
        if self.opt.background_brightness:
            out_rgb += (
                torch.exp(log_light_intensity).unsqueeze(-1)
                * self.opt.background_brightness
            )
        return out_rgb

    def within_grid(self, pts, atol=1e-6):
        '''
        Check whether the given pts is within the voxel grid
        '''
        gsz_cu = self._grid_size().to(pts.dtype).to(device=pts.device)
        return ((pts <= gsz_cu-1. - atol).all(axis=-1) & \
                (pts >= gsz_cu*0. + atol).all(axis=-1))

    def find_next_intersection(self, t, origins, dirs):
        '''
        Find the next intersecting t with voxel planes
        '''
        def floor_int(tensor, atol=1e-6):
            '''
            Floor the given tensor
            if the tensor contains int, reduce the element by 1
            '''
            ft = torch.floor(tensor)
            eq_mask = torch.isclose(ft, tensor, atol=atol)
            # eq_mask = torch.abs(ft - tensor) <= eps
            ft[eq_mask] = ft[eq_mask] - 1
            return ft

        def ceil_int(tensor, atol=1e-6):
            '''
            Ceil the given tensor
            if the tensor contains int, increase the element by 1
            '''
            ft = torch.ceil(tensor)
            eq_mask = torch.isclose(ft, tensor, atol=atol)
            # eq_mask = torch.abs(ft - tensor) <= eps
            ft[eq_mask] = ft[eq_mask] + 1
            return ft

        def find_next_plane(pos, dirs):
            '''
            Use current pos and dirs to find axis of next planes to intersect
            '''
            next_plane = torch.clone(pos)
            for i in range(pos.shape[1]):
                dir_mask = dirs[:,i] >= 0
                next_plane[dir_mask, i] = ceil_int(next_plane[dir_mask, i])
                next_plane[~dir_mask, i] = floor_int(next_plane[~dir_mask, i])
            # torch.clamp_(next_plane, 0., gsz_cu-1.)
            return next_plane
        # current pos
        pos = origins + t[:, None] * dirs

        # find next plane for intersections
        next_plane = find_next_plane(pos, dirs)
        # find adaptive step size that reachs the next intersection between ray and voxel xyz plane
        ada_steps = (next_plane - pos) / dirs
        # remove nan and non-positive steps
        ada_steps.view(-1)[ada_steps.view(-1) <= 0] = torch.inf
        torch.nan_to_num_(ada_steps, torch.inf)
        ada_step = ada_steps.min(axis=-1).values
        next_t = t + ada_step

        # check whether intersection goes outside of voxel
        next_pos = origins + next_t[:, None] * dirs
        next_t[~self.within_grid(next_pos, atol=-1e-8)] = torch.nan

        return next_t
    
    def find_mid_voxel(self, t, next_t, origins, dirs):
        '''
        Use two ray-plane intersections to find the voxel that ray passes through
        '''
        t_mid = (t + next_t) / 2
        pos_mid = origins + t_mid[:, None] * dirs
        return torch.floor(pos_mid).long()

    def find_ray_voxels_intersection(self, t, origins, dirs):
        '''
        Find a list of voxels that given camera ray batch intersects with
        Re-arrange ray and voxels into a batch
        https://stackoverflow.com/questions/33290838/find-if-a-ray-intersects-a-voxel-without-marching
        '''
        gsz = self._grid_size()
        gsz_cu = gsz.to(device=dirs.device)
        B = origins.shape[0]
        voxels_i = torch.zeros(B, device=origins.device, dtype=torch.long)

        if self.use_octree:
            # move camera origin to near clip
            ray_o = origins + dirs * self.opt.near_clip * self._scaling.to(device=dirs.device) * gsz_cu.mean()
            # re-scale and shift camera pos to [-1, +1]
            ray_o = ray_o / (gsz_cu/2) - 1.
            ray_d = dirs / (gsz_cu/2)
            ray_d = ray_d / torch.norm(ray_d, dim=-1, keepdim=True)

            # ugly work arounds to move ray_o outside [-1, +1]
            inside_o_mask = ((ray_o >= -1.) & (ray_o <= 1.)).all(axis=-1)
            ts = (torch.tensor([-1, -1, -1, 1, 1, 1], dtype=ray_o.dtype, device=ray_o.device) - ray_o[inside_o_mask].repeat([1,2])) /  ray_d[inside_o_mask].repeat([1,2])
            ts[ts>0] = -torch.inf # ingore positive ts
            near = torch.zeros_like(ray_o[:,0])
            near[inside_o_mask] = ts.max(axis=-1).values
            ray_o[inside_o_mask] = ray_o[inside_o_mask] + (near[inside_o_mask]-1e-6)[:, None] * ray_d[inside_o_mask]

            
            nugs_ridx, nugs_pidx, nugs_depth = \
                kaolin.render.spc.unbatched_raytrace(
                    self.spc.octree, self.spc.point_hierarchy, self.spc.pyramid, 
                    self.spc.exsum, ray_o.float(), ray_d.float(), self.spc.max_level)

            # remove voxels that are behind the ray (needed due to the work arounds)
            behind_intersect_mask = inside_o_mask[nugs_ridx.long()] & \
                                    (nugs_depth[:, 0] < -near[nugs_ridx.long()])
            nugs_ridx = nugs_ridx[~behind_intersect_mask]
            nugs_pidx = nugs_pidx[~behind_intersect_mask]

            l = kaolin.ops.spc.morton_to_points(nugs_pidx.long() - self.spc.pyramid[1, self.spc.max_level]).long() 
            # filter boundary ls
            valid_l_mask = (l <= gsz_cu-2).all(axis=-1) & (l >= 0).all(axis=-1)
            l = l[valid_l_mask]
            bincount = torch.bincount(nugs_ridx[valid_l_mask])
            voxels_i[:bincount.shape[0]] = bincount
        else:
            voxels = torch.ones((B, torch.sum(gsz_cu).long(), 3), device=origins.device, dtype=torch.long) * -1
            good_indices = torch.arange(B, device=origins.device)

            # record a list of intersecting voxels
            # with utils.Timing('Intersection finding'):
            while good_indices.numel() > 0:
                next_t = self.find_next_intersection(t, origins, dirs)
                l = self.find_mid_voxel(t, next_t, origins, dirs)

                mask = ~torch.isnan(next_t)
                good_indices = good_indices[mask]
                # invalid l can appear due to rounding error
                valid_l_mask = (l[mask] <= gsz_cu-2).all(axis=-1) & (l[mask] >= 0).all(axis=-1)
                
                voxels[good_indices[valid_l_mask], voxels_i[good_indices[valid_l_mask]], :] = l[mask][valid_l_mask]
                voxels_i[good_indices[valid_l_mask]] += 1
                origins = origins[mask]
                dirs = dirs[mask]
                t = next_t[mask]

        # re-arrange voxels into a batch of samples
        VV = voxels_i.sum() # total number of visited voxels
        MV = voxels_i.max() if voxels_i.numel() > 0 else 0 # number of max voxels per ray
        
        if not self.use_octree:
            B_l = voxels[:, :MV, :].reshape(-1, 3)
            visited_l_mask = (torch.arange(MV)[None, :].repeat([B, 1]).to(origins.device) < voxels_i[:, None]).reshape(-1)
            l = B_l[visited_l_mask] # [VV]
            ray_ids = torch.repeat_interleave(torch.arange(voxels_i.shape[0]).to(voxels_i.device), voxels_i)
        else:
            ray_ids = nugs_ridx[valid_l_mask]

        return l, ray_ids.long()


    def _surface_render_gradcheck_lerp(
        self,
        rays: Rays,
        rgb_gt: torch.Tensor = None,
        randomize: bool = False,
        beta_loss: float = 0.0,
        sparsity_loss: float = 0.0,
        return_raylen: bool = False,
        return_depth: bool = False,
        alpha_weighted_norm_loss: bool = False,
        numerical_solution: bool = False,
        dtype = torch.double,
        allow_outside: bool = False, 
        intersect_th: float = 0.1,
        no_surface: bool = False,
        run_backward: bool = False,
        reg: bool = False,
    ):
        """
        gradcheck version for surface rendering

        alpha_weighted_norm_loss: use alpha value to re-weight the surface normal loss
        numerical_solution: use numerical solver to find cubic roots. Currently has no gradient flow!
        allow_outside: allow intersections outside of voxel. Deprecated
        intersect_th: threshold for determining intersections. Used when converting to point clouds
        no_surface: do not use surface to take samples. Use for no_surface_init_iters
        """
        
        ########### Preprocess Camera Rays ###########
        
        origins = self.world2grid(rays.origins).to(dtype)
        dirs = rays.dirs / torch.norm(rays.dirs, dim=-1, keepdim=True).to(dtype)
        viewdirs = dirs
        B = dirs.size(0)
        assert origins.size(0) == B

        gsz = self._grid_size()
        gsz_cu = gsz.to(device=dirs.device).to(dtype)

        dirs = dirs * (self._scaling * gsz).to(device=dirs.device)
        delta_scale = 1.0 / dirs.norm(dim=1)
        dirs *= delta_scale.unsqueeze(-1)

        if self.basis_type == BASIS_TYPE_3D_TEXTURE:
            sh_mult = self._eval_learned_bases(viewdirs)
        elif self.basis_type == BASIS_TYPE_MLP:
            sh_mult = torch.sigmoid(self._eval_basis_mlp(viewdirs))
        else:
            sh_mult = utils.eval_sh_bases(self.basis_dim, viewdirs)

        ########### Find Ray Bounds ###########

        ts_bd = torch.cat([(gsz_cu*0 - origins) / dirs, (gsz_cu-1 - origins) / dirs], axis=-1)
        
        # remove intersection in inverse directions
        ts_bd.view(-1)[ts_bd.view(-1) < 0] = torch.nan

        # remove intersections outside of voxel
        for i in range(ts_bd.shape[1]):
            ts_bd[~self.within_grid(ts_bd[:, i, None] * dirs + origins, atol=-1e-6), i] = torch.nan

        t = torch.nan_to_num(ts_bd, torch.inf).min(axis=-1).values
        tmax = torch.nan_to_num(ts_bd, -torch.inf).max(axis=-1).values
        # for cameras within voxel grids, t_near = 0
        t[self.within_grid(origins)] = 0.

        if return_raylen:
            return tmax - t

        if self.surface_type == SURFACE_TYPE_VOXEL_FACE:
            # take samples at ray-voxel intersections
            # l, ray_ids = self.find_ray_voxels_intersection(t, origins, dirs)

            plane_xs = torch.arange(0, gsz[0], device=origins.device)
            plane_ys = torch.arange(0, gsz[1], device=origins.device)
            plane_zs = torch.arange(0, gsz[2], device=origins.device)

            ts_x = (plane_xs[None, :] - origins[:, 0, None])/dirs[:, 0, None]
            ts_y = (plane_ys[None, :] - origins[:, 1, None])/dirs[:, 1, None]
            ts_z = (plane_zs[None, :] - origins[:, 2, None])/dirs[:, 2, None]

            ts = torch.concat([ts_x, ts_y, ts_z], axis=-1)
            samples = origins[:, None, :] + dirs[:, None, :] * ts[..., None]

            valid_sample_mask = self.within_grid(samples)

            ts = ts[valid_sample_mask]
            samples = samples[valid_sample_mask]
            ray_ids = torch.repeat_interleave(torch.arange(B, device=t.device), torch.count_nonzero(valid_sample_mask,dim=-1))


            l = samples.long()
            lx, ly, lz = l.unbind(-1) 
            links000 = self.links[lx, ly, lz]
            links001 = self.links[lx, ly, lz + 1]
            links010 = self.links[lx, ly + 1, lz]
            links011 = self.links[lx, ly + 1, lz + 1]
            links100 = self.links[lx + 1, ly, lz]
            links101 = self.links[lx + 1, ly, lz + 1]
            links110 = self.links[lx + 1, ly + 1, lz]
            links111 = self.links[lx + 1, ly + 1, lz + 1]

            alpha000, rgb000, _ = self._fetch_links(links000)
            alpha001, rgb001, _ = self._fetch_links(links001)
            alpha010, rgb010, _ = self._fetch_links(links010)
            alpha011, rgb011, _ = self._fetch_links(links011)
            alpha100, rgb100, _ = self._fetch_links(links100)
            alpha101, rgb101, _ = self._fetch_links(links101)
            alpha110, rgb110, _ = self._fetch_links(links110)
            alpha111, rgb111, _ = self._fetch_links(links111)

            pos = samples - l

            wa, wb = 1.0 - pos, pos
            # c00 = alpha000 * wa[:, 2:] + alpha001 * wb[:, 2:]
            # c01 = alpha010 * wa[:, 2:] + alpha011 * wb[:, 2:]
            # c10 = alpha100 * wa[:, 2:] + alpha101 * wb[:, 2:]
            # c11 = alpha110 * wa[:, 2:] + alpha111 * wb[:, 2:]
            # c0 = c00 * wa[:, 1:2] + c01 * wb[:, 1:2]
            # c1 = c10 * wa[:, 1:2] + c11 * wb[:, 1:2]
            # alpha = c0 * wa[:, :1] + c1 * wb[:, :1]
            # alpha = torch.sigmoid(alpha)
            alpha = torch.sigmoid(alpha000)

            c00 = rgb000 * wa[:, 2:] + rgb001 * wb[:, 2:]
            c01 = rgb010 * wa[:, 2:] + rgb011 * wb[:, 2:]
            c10 = rgb100 * wa[:, 2:] + rgb101 * wb[:, 2:]
            c11 = rgb110 * wa[:, 2:] + rgb111 * wb[:, 2:]
            c0 = c00 * wa[:, 1:2] + c01 * wb[:, 1:2]
            c1 = c10 * wa[:, 1:2] + c11 * wb[:, 1:2]
            rgb = c0 * wa[:, :1] + c1 * wb[:, :1]


            # split and pad samples into 2d arrays
            # where first dim is B
            ray_bin = torch.zeros((B), device=origins.device)
            bincount = torch.bincount(ray_ids)
            ray_bin[:bincount.shape[0]] = bincount
            MS = ray_bin.max().long().item() # maximum number of samples per-ray 

            B_rgb = torch.zeros((B, MS, 3), device=origins.device)
            B_alpha = torch.zeros((B, MS), device=origins.device)

            B_to_sample_mask = (torch.arange(MS)[None, :].repeat([B, 1]).to(origins.device) < ray_bin[:, None])

            B_alpha[B_to_sample_mask] = alpha[:, 0].to(B_alpha.dtype) # [N_samples]

            rgb_sh = rgb.reshape(-1, 3, self.basis_dim)
            B_rgb[B_to_sample_mask,:] = torch.clamp_min(
                torch.sum(sh_mult[ray_ids, None, :] * rgb_sh, dim=-1).to(B_rgb.dtype) + 0.5,
                0.0,
            ) # [N_samples, 3]

            assert not torch.isnan(B_alpha).any(), 'NaN detcted in alpha!'

            B_weights = B_alpha * torch.cumprod(
                torch.cat([torch.ones((B_alpha.shape[0], 1)).to(B_alpha.device), torch.clamp(1.-B_alpha, 1e-7, 1-1e-7)], -1), -1
                )[:, :-1] # [B, MS, 3]
            
            out_rgb = torch.sum(B_weights[...,None] * B_rgb, -2)  # [B, 3]

            B_ts = torch.zeros((B, MS), device=origins.device)
            B_ts[B_to_sample_mask] = ts.to(B_ts.dtype) # [N_samples]

            out_depth = torch.sum(B_weights[...,None] * B_ts[...,None], -2) # [B, 1]

            # Add background color
            if self.opt.background_brightness:
                out_rgb += (1-torch.sum(B_weights, -1))[:, None] * self.opt.background_brightness

            out = {
                'rgb': out_rgb,
                'depth': out_depth,
                'extra_loss': {},
                'log_stats': {},
            }

            # # compute density lap loss
            # if alpha.numel() == 0:
            #     density_lap_loss = 0.
            # else:
            #     p_lap = torch.exp(-alpha) + torch.exp(-(1-alpha))
            #     density_lap_loss = torch.mean(-torch.log(p_lap))
            #     # make positive
            #     density_lap_loss = density_lap_loss + torch.log(torch.exp(torch.tensor(-1, device=p_lap.device)) + 1)
            # out['extra_loss']['density_lap_loss'] = density_lap_loss
            # out['log_stats']['density_lap_loss'] = density_lap_loss

            return out



        if no_surface:
            # take samples at fixed step intervals
            valid_mask = tmax < t
            MAX_N_SAMPLES = ((tmax - t) / self.opt.step_size).long().max()

            ts = torch.arange(1, MAX_N_SAMPLES, device=t.device)[None, :].repeat(B,1)
            ts = ts * self.opt.step_size + t[:,None]
            valid_ts_mask = ts < tmax[:,None]

            ts = ts[valid_ts_mask]
            ray_ids = torch.repeat_interleave(torch.arange(B, device=t.device), torch.count_nonzero(valid_ts_mask,dim=-1))

            # Interpolation
            pos = origins[ray_ids] + ts[:, None] * dirs[ray_ids]
            pos = pos.clamp_min_(0.0)
            pos[:, 0] = torch.clamp_max(pos[:, 0], gsz_cu[0] - 1)
            pos[:, 1] = torch.clamp_max(pos[:, 1], gsz_cu[1] - 1)
            pos[:, 2] = torch.clamp_max(pos[:, 2], gsz_cu[2] - 1)

            l = pos.to(torch.long)
            l.clamp_min_(0)
            l[:, 0] = torch.clamp_max(l[:, 0], gsz_cu[0].long() - 2)
            l[:, 1] = torch.clamp_max(l[:, 1], gsz_cu[1].long() - 2)
            l[:, 2] = torch.clamp_max(l[:, 2], gsz_cu[2].long() - 2)
            pos -= l

            lx, ly, lz = l.unbind(-1)
            links000 = self.links[lx, ly, lz]
            links001 = self.links[lx, ly, lz + 1]
            links010 = self.links[lx, ly + 1, lz]
            links011 = self.links[lx, ly + 1, lz + 1]
            links100 = self.links[lx + 1, ly, lz]
            links101 = self.links[lx + 1, ly, lz + 1]
            links110 = self.links[lx + 1, ly + 1, lz]
            links111 = self.links[lx + 1, ly + 1, lz + 1]

            sigma000, rgb000, _ = self._fetch_links(links000)
            sigma001, rgb001, _ = self._fetch_links(links001)
            sigma010, rgb010, _ = self._fetch_links(links010)
            sigma011, rgb011, _ = self._fetch_links(links011)
            sigma100, rgb100, _ = self._fetch_links(links100)
            sigma101, rgb101, _ = self._fetch_links(links101)
            sigma110, rgb110, _ = self._fetch_links(links110)
            sigma111, rgb111, _ = self._fetch_links(links111)

            wa, wb = 1.0 - pos, pos
            c00 = sigma000 * wa[:, 2:] + sigma001 * wb[:, 2:]
            c01 = sigma010 * wa[:, 2:] + sigma011 * wb[:, 2:]
            c10 = sigma100 * wa[:, 2:] + sigma101 * wb[:, 2:]
            c11 = sigma110 * wa[:, 2:] + sigma111 * wb[:, 2:]
            c0 = c00 * wa[:, 1:2] + c01 * wb[:, 1:2]
            c1 = c10 * wa[:, 1:2] + c11 * wb[:, 1:2]
            sigma = c0 * wa[:, :1] + c1 * wb[:, :1]
            sigma = torch.relu(sigma)

            c00 = rgb000 * wa[:, 2:] + rgb001 * wb[:, 2:]
            c01 = rgb010 * wa[:, 2:] + rgb011 * wb[:, 2:]
            c10 = rgb100 * wa[:, 2:] + rgb101 * wb[:, 2:]
            c11 = rgb110 * wa[:, 2:] + rgb111 * wb[:, 2:]
            c0 = c00 * wa[:, 1:2] + c01 * wb[:, 1:2]
            c1 = c10 * wa[:, 1:2] + c11 * wb[:, 1:2]
            rgb = c0 * wa[:, :1] + c1 * wb[:, :1]

            # split and pad samples into 2d arrays
            # where first dim is B
            ray_bin = torch.zeros((B), device=origins.device)
            bincount = torch.bincount(ray_ids)
            ray_bin[:bincount.shape[0]] = bincount
            MS = ray_bin.max().long().item() # maximum number of samples per-ray 

            B_rgb = torch.zeros((B, MS, 3), device=origins.device)
            B_alpha = torch.zeros((B, MS), device=origins.device)

            B_to_sample_mask = (torch.arange(MS)[None, :].repeat([B, 1]).to(origins.device) < ray_bin[:, None])

            d = torch.tensor([1.,1.,1.])
            d = d / torch.norm(d)
            delta_scale = 1./(d * self._scaling * self._grid_size()).norm()

            alpha = 1 - torch.exp(-sigma[:,0] * delta_scale * self.opt.step_size).to(B_rgb.dtype) # [N_samples]
            B_alpha[B_to_sample_mask] = alpha # [N_samples]

            rgb_sh = rgb.reshape(-1, 3, self.basis_dim)
            B_rgb[B_to_sample_mask,:] = torch.clamp_min(
                torch.sum(sh_mult[ray_ids, None, :] * rgb_sh, dim=-1).to(B_rgb.dtype) + 0.5,
                0.0,
            ) # [N_samples, 3]

            assert not torch.isnan(B_alpha).any(), 'NaN detcted in alpha!'

            B_weights = B_alpha * torch.cumprod(
                torch.cat([torch.ones((B_alpha.shape[0], 1)).to(B_alpha.device), torch.clamp(1.-B_alpha, 1e-7, 1-1e-7)], -1), -1
                )[:, :-1] # [B, MS, 3]
            
            out_rgb = torch.sum(B_weights[...,None] * B_rgb, -2)  # [B, 3]

            B_ts = torch.zeros((B, MS), device=origins.device)
            B_ts[B_to_sample_mask] = ts.to(B_ts.dtype) # [N_samples]

            out_depth = torch.sum(B_weights[...,None] * B_ts[...,None], -2) # [B, 1]

            # Add background color
            if self.opt.background_brightness:
                out_rgb += (1-torch.sum(B_weights, -1))[:, None] * self.opt.background_brightness

            out = {
                'rgb': out_rgb,
                'depth': out_depth,
                'extra_loss': {},
                'log_stats': {},
            }

            # compute density lap loss
            if alpha.numel() == 0:
                no_surf_init_density_lap_loss = 0.
            else:
                p_lap = torch.exp(-alpha) + torch.exp(-(1-alpha))
                no_surf_init_density_lap_loss = torch.mean(-torch.log(p_lap))
                # make positive
                no_surf_init_density_lap_loss = no_surf_init_density_lap_loss + torch.log(torch.exp(torch.tensor(-1, device=p_lap.device)) + 1)
            out['extra_loss']['no_surf_init_density_lap_loss'] = no_surf_init_density_lap_loss
            out['log_stats']['no_surf_init_density_lap_loss'] = no_surf_init_density_lap_loss

            return out


        ########### Find Ray-Voxel Intersections ###########
        # with utils.Timing("Ray-voxel intersection finding"):
        l, ray_ids = self.find_ray_voxels_intersection(t, origins, dirs)

        ########### Fetch Vertices Data ###########

        lx, ly, lz = l.unbind(-1) 
        links000 = self.links[lx, ly, lz]
        links001 = self.links[lx, ly, lz + 1]
        links010 = self.links[lx, ly + 1, lz]
        links011 = self.links[lx, ly + 1, lz + 1]
        links100 = self.links[lx + 1, ly, lz]
        links101 = self.links[lx + 1, ly, lz + 1]
        links110 = self.links[lx + 1, ly + 1, lz]
        links111 = self.links[lx + 1, ly + 1, lz + 1]

        # mask out voxels that don't exist
        exist_l_mask = [links >= 0 for links in [links000, links001, links010, links011, links100, links101, links110, links111]]
        exist_l_mask = torch.stack(exist_l_mask).T.all(axis=-1) # [VV]
        VEV = torch.count_nonzero(exist_l_mask) # number of visited and exist voxels
        ray_ids = ray_ids[exist_l_mask]
        l = l[exist_l_mask]
        l_ids = torch.arange(l.shape[0]).to(l.device)

        alpha000, rgb000, surface000 = self._fetch_links(links000[exist_l_mask]) # [VEV, ...]
        alpha001, rgb001, surface001 = self._fetch_links(links001[exist_l_mask])
        alpha010, rgb010, surface010 = self._fetch_links(links010[exist_l_mask])
        alpha011, rgb011, surface011 = self._fetch_links(links011[exist_l_mask])
        alpha100, rgb100, surface100 = self._fetch_links(links100[exist_l_mask])
        alpha101, rgb101, surface101 = self._fetch_links(links101[exist_l_mask])
        alpha110, rgb110, surface110 = self._fetch_links(links110[exist_l_mask])
        alpha111, rgb111, surface111 = self._fetch_links(links111[exist_l_mask])


        ########### Find Ray-Surface Intersections ###########
        # with utils.Timing("Cubic root finding"):
        if self.surface_type in [SURFACE_TYPE_SDF, SURFACE_TYPE_UDF, SURFACE_TYPE_UDF_ALPHA, SURFACE_TYPE_UDF_FAKE_SAMPLE]:
            if self.surface_type in [SURFACE_TYPE_UDF, SURFACE_TYPE_UDF_ALPHA, SURFACE_TYPE_UDF_FAKE_SAMPLE]:
                fn = torch.nn.Softplus()
                # fn = torch.relu
                surface000 = fn(surface000)
                surface001 = fn(surface001)
                surface010 = fn(surface010)
                surface011 = fn(surface011)
                surface100 = fn(surface100)
                surface101 = fn(surface101)
                surface110 = fn(surface110)
                surface111 = fn(surface111)

            surface_values = torch.stack(
                    [surface000, surface001, surface010, surface011, surface100, surface101, surface110, surface111],
                    dim=-1)

            # re-scale all coordinates to [0,1]
            # this prevents grad scale issue
            grid_rescale = 256. / gsz_cu
            grid_rescale = gsz_cu / gsz_cu
            # scale = 256. / gsz_cu
            l = l * grid_rescale
            origins = origins * grid_rescale
            dirs = dirs * grid_rescale
            lx, ly, lz = (l).unbind(-1)
            ox, oy, oz = (origins[ray_ids].to(dtype)).unbind(-1)
            vx, vy, vz = (dirs[ray_ids].to(dtype)).unbind(-1)

            close_planes = l + (dirs[ray_ids]<0).long()
            far_planes = l + (~(dirs[ray_ids]<0)).long()

            close_t = torch.nan_to_num((close_planes - origins[ray_ids]) / dirs[ray_ids], -torch.inf).max(axis=-1).values
            far_t = torch.nan_to_num((far_planes - origins[ray_ids]) / dirs[ray_ids], torch.inf).min(axis=-1).values

            ox, oy, oz = ((origins[ray_ids] + close_t[:, None] * dirs[ray_ids]).to(dtype)).unbind(-1)

            # ox, oy, oz: ray origins
            # vx, vy, vz: ray dirs
            # lx, ly, lz: voxel coordinates
            a00 = surface000[:,0].to(dtype) * (1-oz+lz) + surface001[:,0].to(dtype) * (oz-lz)
            a01 = surface010[:,0].to(dtype) * (1-oz+lz) + surface011[:,0].to(dtype) * (oz-lz)
            a10 = surface100[:,0].to(dtype) * (1-oz+lz) + surface101[:,0].to(dtype) * (oz-lz)
            a11 = surface110[:,0].to(dtype) * (1-oz+lz) + surface111[:,0].to(dtype) * (oz-lz)

            b00 = -surface000[:,0].to(dtype) + surface001[:,0].to(dtype)
            b01 = -surface010[:,0].to(dtype) + surface011[:,0].to(dtype)
            b10 = -surface100[:,0].to(dtype) + surface101[:,0].to(dtype)
            b11 = -surface110[:,0].to(dtype) + surface111[:,0].to(dtype)

            c0 = a00*(1-oy+ly) + a01*(oy-ly)
            c1 = a10*(1-oy+ly) + a11*(oy-ly)

            d0 = -(a00*vy - vz*b00*(1-oy+ly)) + (a01*vy + vz*b01*(oy-ly))
            d1 = -(a10*vy - vz*b10*(1-oy+ly)) + (a11*vy + vz*b11*(oy-ly))

            e0 = -vy*vz*b00 + vy*vz*b01
            e1 = -vy*vz*b10 + vy*vz*b11

            f3 = -e0*vx + e1*vx
            f2 = -d0*vx+e0*(1-ox+lx) + d1*vx+e1*(ox-lx)
            f1 = -c0*vx + d0*(1-ox+lx) + c1*vx + d1*(ox-lx)
            f0 = c0*(1-ox+lx) + c1*(ox-lx)
            # f3 * t^3 + f2 * t^2 + f1 * t + f0 = level_set gives the surface

            # if self.surface_type in [SURFACE_TYPE_UDF, SURFACE_TYPE_UDF_ALPHA, SURFACE_TYPE_UDF_FAKE_SAMPLE]:
            # find the list of possible level sets
            lv_set_mask = (self.level_set_data >= surface_values.min(axis=-1).values) & \
                (self.level_set_data <= surface_values.max(axis=-1).values) # [VEV, N_level_sets]

            if self.opt.surf_fake_sample:
                # do not filter out voxels by surface scalars if we render fake samples
                lv_set_mask = torch.ones_like(lv_set_mask).bool()

            if self.surface_type in [SURFACE_TYPE_UDF_FAKE_SAMPLE]:
                # if no valid lv set in range, use closest one
                no_lv_mask = ~lv_set_mask.any(axis=-1)
                if torch.count_nonzero(no_lv_mask) > 0:
                    udf_avgs = torch.mean(surface_values[no_lv_mask], axis=-1)
                    dists = torch.abs(self.level_set_data[None, :] - udf_avgs)
                    lv_set_mask[no_lv_mask, dists.argmin(axis=-1)] = True # TODO check if this is ok!

            lv_sets = self.level_set_data[None, :].repeat(lv_set_mask.shape[0], 1)[lv_set_mask]
            lv_set_ids = torch.arange(self.level_set_data.shape[0], device=surface_values.device)[None, :].repeat(lv_set_mask.shape[0], 1)[lv_set_mask]
            N_EQ = lv_sets.shape[0] # total number of equations to solve
            if lv_set_mask.numel() == 0:
                M_LV = 0
            else:
                M_LV = torch.count_nonzero(lv_set_mask, dim=-1).max() # maximum number of level sets in one voxel
            lv_set_bincount = torch.count_nonzero(lv_set_mask,axis=-1)

            ray_ids = torch.repeat_interleave(ray_ids, lv_set_bincount)
            l_ids = torch.repeat_interleave(l_ids, lv_set_bincount)

            f3 = torch.repeat_interleave(f3, lv_set_bincount)
            f2 = torch.repeat_interleave(f2, lv_set_bincount)
            f1 = torch.repeat_interleave(f1, lv_set_bincount)
            f0 = torch.repeat_interleave(f0, lv_set_bincount)

            f0 = f0 - lv_sets

            close_t = torch.repeat_interleave(close_t, lv_set_bincount)

            # negative roots are considered no roots and are filtered out later
            ts = torch.ones([f0.numel(), 3]).to(dtype).to(device=dirs.device) * -1 # [VV, 3]

            fs = torch.stack([f0,f1,f2,f3]).T
            # np.save('fs.npy', fs.cpu().detach().numpy())

            # solve cubic equations
            numerical_solution = False
            if numerical_solution:
                # TODO use Aesara instead https://aesara.readthedocs.io/en/latest/
                print('using slow numerical solver for cubic functions!')
                print('no gradient is enabled at the moment!')
                ts = torch.ones([f0.numel(), 3]).to(device=dirs.device) * -1
                for i in range(f0.shape[0]):
                    x = Symbol('x')
                    a_np = f3.cpu().detach().numpy() 
                    b_np = f2.cpu().detach().numpy()
                    c_np = f1.cpu().detach().numpy()
                    d_np = f0.cpu().detach().numpy()
                    solutions = sympy.solveset(a_np[i] * x**3 + b_np[i] * x**2 + c_np[i] * x + d_np[i], x, domain=sympy.S.Reals)
                    solutions = torch.tensor(list(solutions),dtype=torch.float32).to(device=dirs.device)
                    ts[i, :solutions.shape[0]] = solutions
            else:

                # analyical solution for f0 + _t*f1 + (_t**2)*f2 + (_t**3)*f3 = 0
                # https://github.com/shril/CubicEquationSolver/blob/master/CubicEquationSolver.py

                # check for trivial a and b -- reduce to linear or polynomial solutions
                # no_solution_mask = (f3 == 0.) & (f2 == 0.) & (f1 == 0.)
                # linear_mask = (f3 == 0.) & (f2 == 0.) & (~no_solution_mask)
                # quad_mask = (f3 == 0.) & (~linear_mask) & (~no_solution_mask)
                # cubic_mask = (~quad_mask) & (~linear_mask) & (~no_solution_mask)

                atol = 1e-10
                no_solution_mask = torch.isclose(f3, torch.zeros_like(f3), atol=atol) \
                        & torch.isclose(f2, torch.zeros_like(f2), atol=atol) \
                        & torch.isclose(f1, torch.zeros_like(f1), atol=atol)
                linear_mask = torch.isclose(f3, torch.zeros_like(f3), atol=atol) \
                        & torch.isclose(f2, torch.zeros_like(f2), atol=atol) \
                        & (~no_solution_mask)
                quad_mask = torch.isclose(f3, torch.zeros_like(f3), atol=atol) \
                        & (~linear_mask) & (~no_solution_mask)
                cubic_mask = (~quad_mask) & (~linear_mask) & (~no_solution_mask)


                ##### Linear Roots #####
                if ts[linear_mask].numel() > 0:
                    ts[linear_mask, 0] = (-f0[linear_mask] * 1.0) / f1[linear_mask]

                ##### Quadratic Roots #####
                if ts[quad_mask].numel() > 0:
                    _b, _c, _d = f2[quad_mask], f1[quad_mask], f0[quad_mask]
                    D = _c**2 - 4.0 * _b * _d

                    # two real roots
                    D_mask = D > 0 
                    sqrt_D = torch.sqrt(D[D_mask])
                    ids = torch.arange(quad_mask.shape[0])[quad_mask][D_mask]
                    t0 = (-_c[D_mask] - sqrt_D) / (2.0 * _b[D_mask])
                    t1 = (-_c[D_mask] + sqrt_D) / (2.0 * _b[D_mask])
                    ts[ids, 0] = torch.min(torch.stack([t0,t1]), axis=0).values
                    ts[ids, 1] = torch.max(torch.stack([t0,t1]), axis=0).values

                    # otherwise, has no real roots

                ##### Cubic Roots #####

                cubic_ids = torch.arange(ts.shape[0])[cubic_mask]

                # normalize 
                norm_term = f3[cubic_mask]
                a = f3[cubic_mask] / norm_term
                b = f2[cubic_mask] / norm_term
                c = f1[cubic_mask] / norm_term
                d = f0[cubic_mask] / norm_term

                def cond_cbrt(x, eps=1e-10):
                    '''
                    Compute cubic root of x based on sign
                    '''
                    ret = torch.zeros_like(x)
                    ret[x >= 0] = torch.pow(torch.clamp_min_(x[x >= 0], eps), 1/3.)
                    ret[x < 0] = torch.pow(torch.clamp_min_(-x[x < 0], eps), 1/3.) * -1
                    return ret

                Q = ((b**2) - 3.*c) / 9.
                R = (2.*(b**3) - 9.*b*c + 27.*d) /54.

                # # all three roots are real and equal
                # _mask1 = ((f == 0) & (g == 0) & (h == 0))
                # ts[cubic_ids[_mask1], 0] = cond_cbrt(d[_mask1]/a[_mask1])

                # all three roots are real 
                _mask2 = (R)**2 < (Q)**3
                _b, _Q, _R = b[_mask2], Q[_mask2], R[_mask2]

                eps = 1e-10
                theta = torch.acos(torch.clamp(_R / torch.sqrt((_Q)**3), -1+eps, 1-eps))
                # theta = torch.acos((_R / torch.sqrt((_Q)**3)))
                
                ts[cubic_ids[_mask2], 0] = -2. * torch.sqrt(_Q) * torch.cos(theta/3.) - _b/3.
                ts[cubic_ids[_mask2], 1] = -2. * torch.sqrt(_Q) * torch.cos((theta - 2.*torch.pi)/3.) - _b/3.
                ts[cubic_ids[_mask2], 2] = -2. * torch.sqrt(_Q) * torch.cos((theta + 2.*torch.pi)/3.) - _b/3.

                # only one root is real
                _mask3 = ~_mask2
                __b, __Q, __R = b[_mask3], Q[_mask3], R[_mask3]

                A = -torch.sign(__R) * (torch.abs(__R) + torch.sqrt(torch.clamp_min((__R)**2 - (__Q)**3, 1e-8))) ** (1./3.)
                # A = -torch.sign(__R) * (torch.abs(__R) + torch.sqrt(((__R)**2 - (__Q)**3))) ** (1./3.)
                _B = __Q/A
                _B[A== 0.] = 0.

                ts[cubic_ids[_mask3], 0] = (A+_B) - __b/3.


                

                assert not torch.isnan(ts).any(), 'NaN detcted in cubic roots'
                assert torch.isfinite(ts).all(), 'Inf detcted in cubic roots'

            assert (((ts[:,0] <= ts[:,1]) | (ts[:,1]==-1.)) & ((ts[:,1] <= ts[:,2])| (ts[:,2]==-1.)) ).all()

            N_INTERSECT = ts.shape[1]
            # ts = torch.sort(ts, dim=-1).values
            close_t = close_t[:, None].repeat(1, N_INTERSECT)
            samples = origins[ray_ids, None, :] + (ts[..., None] + close_t[..., None]) * dirs[ray_ids, None, :] # [VEV, N_INTERSECT, 3]
            ray_ids = ray_ids[:, None].repeat(1, N_INTERSECT)
            l_ids = l_ids[:, None].repeat(1, N_INTERSECT)
            lv_set_ids = lv_set_ids[:, None].repeat(1, N_INTERSECT)

            def check_solution(ts=ts):
                return ts**3 * f3[:, None].repeat(1, N_INTERSECT) + ts**2 * f2[:, None].repeat(1, N_INTERSECT) + \
                    ts * f1[:, None].repeat(1, N_INTERSECT) + f0[:, None].repeat(1, N_INTERSECT)


            # filter out roots with t that's smaller than near clip
            # Note that scaling is not exactly correct when grid reso are not the same
            neg_roots_mask = (ts + close_t) < self.opt.near_clip * self._scaling.to(device=ts.device) * gsz_cu.mean()


            if allow_outside:
                invalid_sample_mask = ~self.within_grid(samples) | (torch.isnan(samples)).any(axis=-1)
            else:
                # remove all samples outside of the voxel
                invalid_sample_mask = (samples < l[l_ids]).any(axis=-1) | (samples > l[l_ids]+grid_rescale).any(axis=-1) | (torch.isnan(samples)).any(axis=-1)
            
            valid_sample_mask = (~neg_roots_mask) & (~invalid_sample_mask)


            # if self.surface_type in [SURFACE_TYPE_UDF_FAKE_SAMPLE]:
            if self.opt.surf_fake_sample:
                # mask of ray-voxel pair where no valid intersection exists
                # where we take one fake sample at the mid of the ray passing through the voxel
                fake_sample_mask = ~(valid_sample_mask.any(axis=-1, keepdim=True)).repeat(1,N_INTERSECT)
                fake_sample_mask[:, 1:] = False
                # find two intersections between ray and voxel surfaces
                _l = l[l_ids[fake_sample_mask]]
                _os = origins[ray_ids[fake_sample_mask]]
                _ds = dirs[ray_ids[fake_sample_mask]]
                
                # if dirs is negative, closest plane has larger coord
                close_planes = _l + (_ds<0).long()
                far_planes = _l + (~(_ds<0)).long()

                _close_t = torch.nan_to_num((close_planes - _os) / _ds, -torch.inf).max(axis=-1).values
                _far_t = torch.nan_to_num((far_planes - _os) / _ds, torch.inf).min(axis=-1).values

                ts[fake_sample_mask] = (_close_t + _far_t)/2 - close_t[fake_sample_mask]
                samples[fake_sample_mask, :] = _os + ts[fake_sample_mask][:, None] * _ds

                valid_sample_mask = valid_sample_mask | fake_sample_mask

                # store ids of fake samples
                fake_sample_ids = torch.zeros(fake_sample_mask.numel(), device=fake_sample_mask.device, dtype=torch.long)
                fake_sample_ids[valid_sample_mask.view(-1)] = torch.arange(
                    torch.count_nonzero(valid_sample_mask),
                    device=fake_sample_mask.device)

                fake_sample_ids = fake_sample_ids.view(fake_sample_mask.shape)[fake_sample_mask]

            ts = ts[valid_sample_mask] # [VEV * N_INTERSECT]
            close_t = close_t[valid_sample_mask] # [VEV * N_INTERSECT]
            ray_ids = ray_ids[valid_sample_mask] # [VEV * N_INTERSECT]
            l_ids = l_ids[valid_sample_mask] # [VEV * N_INTERSECT]
            lv_set_ids = lv_set_ids[valid_sample_mask] # [VEV * N_INTERSECT]
            samples = samples[valid_sample_mask, :] # [VEV * N_INTERSECT, 3]
            l = l[l_ids, :]


            ts_raw = ts
            # ts = torch.concat([ts_raw[:1].detach().clone(), ts_raw[1:2], ts_raw[2:].detach().clone()], dim=-1)
            ts = ts_raw
            samples = origins[ray_ids, :] + (ts[..., None] + close_t[...,None]) * dirs[ray_ids, :] # [VEV * N_INTERSECT, 3]
        
        elif self.surface_type == SURFACE_TYPE_PLANE:
            # raise NotImplementedError
            # plane: ax + by + cz + d = 0
            a, b, c, d = torch.mean(torch.stack(
                [surface000, surface001, surface010, surface011, surface100, surface101, surface110, surface111]
                ), axis=0).to(dtype).unbind(-1)

            # a, b, c, d = plane000.to(dtype).unbind(-1)

            # thresholding to force the plane to stay within its voxel
            # abs(torch.sum((l + 0.5) * torch.stack([a,b,c]).T, axis=-1) + d) <= sqrt(3 * (0.5 ** 2))
            th = 0.3
            xyz_term = torch.sum((l + 0.5) * torch.stack([a,b,c]).T, axis=-1)
            d = torch.clamp(d, -th - xyz_term, th - xyz_term)

            ox, oy, oz = origins[ray_ids].to(dtype).unbind(-1)
            vx, vy, vz = dirs[ray_ids].to(dtype).unbind(-1)

            ts = -(a*ox + b*oy + c*oz + d) / (a*vx+b*vy+c*vz)
            samples = origins[ray_ids] + ts[..., None] * dirs[ray_ids] # [VEV, 3]

            # filter out roots with negative t
            neg_roots_mask = ts < 0

            if allow_outside:
                # invalid_sample_mask = ~self.within_grid(samples) | (torch.isnan(samples)).any(axis=-1)
                invalid_sample_mask = (torch.isnan(samples)).any(axis=-1)
            else:
                # remove all samples outside of the voxel
                invalid_sample_mask = (samples < l).any(axis=-1) | (samples > l+1).any(axis=-1) | (torch.isnan(samples)).any(axis=-1)

            valid_sample_mask = (~neg_roots_mask) & (~invalid_sample_mask)
            ts = ts[valid_sample_mask]
            ray_ids = ray_ids[valid_sample_mask]
            samples = samples[valid_sample_mask, :]

        else:
            raise NotImplementedError(f'Gird surface type {self.surface_type} is not supported for grad check rendering')


        ########### Interpolate Alpha & SH ###########

        def check_sample_surface(samples=samples, l=l, l_ids=l_ids):
            wa, wb = grid_rescale - (samples - l), (samples - l)
            wa, wb = wa / (wa + wb), wb / (wa + wb)
            c00 = surface000[l_ids] * wa[:, 2:] + surface001[l_ids] * wb[:, 2:]
            c01 = surface010[l_ids] * wa[:, 2:] + surface011[l_ids] * wb[:, 2:]
            c10 = surface100[l_ids] * wa[:, 2:] + surface101[l_ids] * wb[:, 2:]
            c11 = surface110[l_ids] * wa[:, 2:] + surface111[l_ids] * wb[:, 2:]
            c0 = c00 * wa[:, 1:2] + c01 * wb[:, 1:2]
            c1 = c10 * wa[:, 1:2] + c11 * wb[:, 1:2]
            surface = c0 * wa[:, :1] + c1 * wb[:, :1]
            return surface
 
        # interpolate opacity
        wa, wb = grid_rescale - (samples - l), (samples - l)
        # wa, wb = grid_rescale - (samples.detach().clone() - l), (samples.detach().clone() - l)
        wa, wb = wa / (wa + wb), wb / (wa + wb)
        if self.surface_type == SURFACE_TYPE_UDF_ALPHA:
            alpha = self.density_data[lv_set_ids]
        else:
            c00 = alpha000[l_ids] * wa[:, 2:] + alpha001[l_ids] * wb[:, 2:]
            c01 = alpha010[l_ids] * wa[:, 2:] + alpha011[l_ids] * wb[:, 2:]
            c10 = alpha100[l_ids] * wa[:, 2:] + alpha101[l_ids] * wb[:, 2:]
            c11 = alpha110[l_ids] * wa[:, 2:] + alpha111[l_ids] * wb[:, 2:]
            c0 = c00 * wa[:, 1:2] + c01 * wb[:, 1:2]
            c1 = c10 * wa[:, 1:2] + c11 * wb[:, 1:2]
            alpha_raw = c0 * wa[:, :1] + c1 * wb[:, :1]
        # post sigmoid activation
        if self.opt.alpha_activation_type == SIGMOID_FN:
            alpha = torch.sigmoid(alpha_raw)
        else:
            alpha = 1 - torch.exp(-torch.relu(alpha_raw))
        # alpha = alpha.detach().clone()
        # alpha.requires_grad = True


        # if self.surface_type == SURFACE_TYPE_UDF_FAKE_SAMPLE:
        if self.opt.surf_fake_sample:
            # use naive biased formula to get alpha for fake samples
            lv_sets = self.level_set_data[lv_set_ids[fake_sample_ids]]

            # surface_norm = torch.norm(surface_values[l_ids[fake_sample_ids]], dim=-1)
            surface_norm = torch.std(surface_values[l_ids[fake_sample_ids]], dim=-1, unbiased=False)

            n_surfaces = torch.permute(surface_values[l_ids[fake_sample_ids]], [0,2,1]) / surface_norm[:,None,:]
            
            # find the UDF value at the fake sample
            c00 = n_surfaces[:,0] * wa[fake_sample_ids, 2:] + n_surfaces[:,1] * wb[fake_sample_ids, 2:]
            c01 = n_surfaces[:,2] * wa[fake_sample_ids, 2:] + n_surfaces[:,3] * wb[fake_sample_ids, 2:]
            c10 = n_surfaces[:,4] * wa[fake_sample_ids, 2:] + n_surfaces[:,5] * wb[fake_sample_ids, 2:]
            c11 = n_surfaces[:,6] * wa[fake_sample_ids, 2:] + n_surfaces[:,7] * wb[fake_sample_ids, 2:]
              
            
            c0 = c00 * wa[fake_sample_ids, 1:2] + c01 * wb[fake_sample_ids, 1:2]
            c1 = c10 * wa[fake_sample_ids, 1:2] + c11 * wb[fake_sample_ids, 1:2]
            surface_scalar = c0 * wa[fake_sample_ids, :1] + c1 * wb[fake_sample_ids, :1]

            surface_scalar = surface_scalar[:,0] - lv_sets # SDF-like values

            alpha_before_rw = alpha

            fake_sample_reweight = torch.exp(-.5 * (surface_scalar/self.fake_sample_std)**2).view(-1)
            _alpha = alpha.clone()
            _alpha[fake_sample_ids] = alpha[fake_sample_ids] * fake_sample_reweight[:, None]
            alpha = _alpha

        # interpolate rgb
        wa, wb = grid_rescale - (samples - l), (samples - l)
        # wa, wb = grid_rescale - (samples.detach().clone() - l), (samples.detach().clone() - l)
        wa, wb = wa / (wa + wb), wb / (wa + wb)
        c00 = rgb000[l_ids] * wa[:, 2:] + rgb001[l_ids] * wb[:, 2:]
        c01 = rgb010[l_ids] * wa[:, 2:] + rgb011[l_ids] * wb[:, 2:]
        c10 = rgb100[l_ids] * wa[:, 2:] + rgb101[l_ids] * wb[:, 2:]
        c11 = rgb110[l_ids] * wa[:, 2:] + rgb111[l_ids] * wb[:, 2:]
        c0 = c00 * wa[:, 1:2] + c01 * wb[:, 1:2]
        c1 = c10 * wa[:, 1:2] + c11 * wb[:, 1:2]
        # rgb = c0 * wa[:, :1] + c1 * wb[:, :1]
        rgb_raw = c0 * wa[:, :1] + c1 * wb[:, :1]

        # rgb.requires_grad = True
        # rgb = rgb_raw.detach().clone()
        # rgb[:, 0] = rgb_raw[:, 0]
        # rgb = torch.concat([rgb_raw[:,:1].detach().clone(), rgb_raw[:,1:2], rgb_raw[:,2:].detach().clone()], dim=-1)
        rgb = rgb_raw

        ########### Volume Rendering ###########

        # alpha = torch.concat([alpha[:2].detach().clone(), alpha[2:3], alpha[3:].detach().clone()])

        # split and pad samples into 2d arrays
        # where first dim is B
        ray_bin = torch.zeros((B), device=origins.device)
        bincount = torch.bincount(ray_ids)
        ray_bin[:bincount.shape[0]] = bincount
        MS = ray_bin.max().long().item() # maximum number of samples per-ray 

        B_rgb = torch.zeros((B, MS, 3), device=origins.device)
        B_alpha = torch.zeros((B, MS), device=origins.device)

        B_to_sample_mask = (torch.arange(MS)[None, :].repeat([B, 1]).to(origins.device) < ray_bin[:, None])
        B_alpha[B_to_sample_mask] = alpha[:,0].to(B_rgb.dtype) # [N_samples]

        rgb_sh = rgb.reshape(-1, 3, self.basis_dim)
        B_rgb[B_to_sample_mask,:] = torch.clamp_min(
            torch.sum(sh_mult[ray_ids, None, :] * rgb_sh, dim=-1).to(B_rgb.dtype) + 0.5,
            0.0,
        ) # [N_samples, 3]


        assert not torch.isnan(B_alpha).any(), 'NaN detcted in alpha!'

        B_T = torch.cumprod(
            torch.cat([torch.ones((B_alpha.shape[0], 1)).to(B_alpha.device), torch.clamp(1.-B_alpha, 1e-7, 1-1e-7)], -1), -1
            )[:, :-1]

        B_weights = B_alpha * B_T # [B, MS, 3]
        
        out_rgb = torch.sum(B_weights[...,None] * B_rgb, -2)  # [B, 3]

        # Add background color
        if self.opt.background_brightness:
            out_rgb += (1-torch.sum(B_weights, -1))[:, None] * self.opt.background_brightness

        out = {
            'rgb': out_rgb,
            'log_stats': {},
            'extra_loss': {}
        }

        if return_depth:
            B_ts = torch.zeros((B, MS), device=origins.device)
            B_ts[B_to_sample_mask] = (ts + close_t).to(B_ts.dtype) # [N_samples]

            out_depth = torch.sum(B_weights[...,None] * B_ts[...,None], -2) # [B, 1]

            out['depth'] = out_depth
        
        if reg:
            if self.surface_type in [SURFACE_TYPE_UDF, SURFACE_TYPE_UDF_ALPHA, SURFACE_TYPE_UDF_FAKE_SAMPLE]:
                out['log_stats']['m_lv_sets'] = M_LV

                # caluclate udf_var_loss
                # this loss is to reduce variance in UDF values in the same voxel
                N_lv_sets = torch.count_nonzero(lv_set_mask, dim=-1)
                if N_lv_sets.numel() == 0:
                    udf_var_loss = 0.
                else:
                    udf_vars = torch.var(surface_values,dim=-1)[:,0]
                    udf_var_loss = torch.mean(torch.clamp_min((N_lv_sets - 1), 0.) * udf_vars)
                out['extra_loss']['udf_var_loss'] = udf_var_loss
                out['log_stats']['udf_var_loss'] = udf_var_loss

            # compute density lap loss
            if alpha.numel() == 0:
                density_lap_loss = 0.
            else:
                p_lap = torch.exp(-alpha) + torch.exp(-(1-alpha))
                density_lap_loss = torch.mean(-torch.log(p_lap))
                # make positive
                density_lap_loss = density_lap_loss + torch.log(torch.exp(torch.tensor(-1, device=p_lap.device)) + 1)
            out['extra_loss']['density_lap_loss'] = density_lap_loss
            out['log_stats']['density_lap_loss'] = density_lap_loss

            # compute normal loss
            # note that this should be replaced by a special form of TV loss really...
            if alpha.numel() == 0:
                normal_loss = 0.
            else:
                x,y,z = (l/grid_rescale).long().unbind(-1)
                x,y,z = torch.clamp(x.long(), 0, self.links.shape[0]-3), \
                        torch.clamp(y.long(), 0, self.links.shape[1]-3), \
                        torch.clamp(z.long(), 0, self.links.shape[2]-3)


                coords = torch.tensor([
                    [0,0,0],
                    [0,0,1],
                    [0,1,0],
                    [0,1,1],
                    [1,0,0],
                    [1,0,1],
                    [1,1,0],
                    [1,1,1],

                    [0,0,2],
                    [0,1,2],
                    [1,0,2],
                    [1,1,2],

                    [0,2,0],
                    [0,2,1],
                    [1,2,0],
                    [1,2,1],

                    [2,0,0],
                    [2,0,1],
                    [2,1,0],
                    [2,1,1],
                ], dtype=torch.long, device=l.device)

                links=torch.zeros([3,3,3,l.shape[0]], dtype=torch.long, device=l.device)
                alphas=torch.zeros([3,3,3,l.shape[0], 1], dtype=self.density_data.dtype, device=l.device)
                surfaces=torch.zeros([3,3,3,l.shape[0], 1], dtype=self.surface_data.dtype, device=l.device)

                for i in range(coords.shape[0]):
                    links[coords[i,0], coords[i,1], coords[i,2]] = self.links[x+coords[i,0], y+coords[i,1], z+coords[i,2]]
                    alphas[coords[i,0], coords[i,1], coords[i,2]], _ , \
                        surfaces[coords[i,0], coords[i,1], coords[i,2]] = self._fetch_links(self.links[x+coords[i,0], y+coords[i,1], z+coords[i,2]])

                def find_normal(norm_xyz):
                    x,y,z = norm_xyz.unbind(-1)

                    dx = ((surfaces[x+1,y,z]+surfaces[x+1,y,z+1]+surfaces[x+1,y+1,z]+surfaces[x+1,y+1,z+1]) - \
                        (surfaces[x,y,z]+surfaces[x,y,z+1]+surfaces[x,y+1,z]+surfaces[x,y+1,z+1])) /4
                    dy = ((surfaces[x,y+1,z]+surfaces[x,y+1,z+1]+surfaces[x+1,y+1,z]+surfaces[x+1,y+1,z+1]) - \
                        (surfaces[x,y,z]+surfaces[x,y,z+1]+surfaces[x+1,y,z]+surfaces[x+1,y,z+1]))/4
                    dz = ((surfaces[x,y,z+1]+surfaces[x,y+1,z+1]+surfaces[x+1,y,z+1]+surfaces[x+1,y+1,z+1]) - \
                        (surfaces[x,y,z]+surfaces[x,y+1,z]+surfaces[x+1,y,z]+surfaces[x+1,y+1,z]))/4

                    normals = torch.stack([dx, dy, dz], dim=-1)
                    normals = normals / torch.clamp(torch.norm(normals, dim=-1, keepdim=True), 1e-10)

                    # check if there is non-exist vertex
                    coords = torch.tensor([
                        [0,0,0],
                        [0,0,1],
                        [0,1,0],
                        [0,1,1],
                        [1,0,0],
                        [1,0,1],
                        [1,1,0],
                        [1,1,1],
                        ], dtype=torch.long, device=l.device)
                    ver_xyzs = norm_xyz[None, :] + coords
                    valid_mask = torch.ones(links.shape[-1], device=links.device).bool()
                    for i in range(ver_xyzs.shape[0]):
                        valid_mask = (valid_mask) & (links[ver_xyzs[i,0], ver_xyzs[i,1], ver_xyzs[i,2]] >= 0)

                    alpha_v = [alphas[ver_xyzs[i,0], ver_xyzs[i,1], ver_xyzs[i,2]] for i in range(ver_xyzs.shape[0])]
                    alpha_v = torch.concat(alpha_v, axis=-1).mean(dim=-1)

                    return normals, valid_mask, torch.sigmoid(alpha_v.detach().clone())

                # find normals
                norm_xyzs = torch.tensor([[0,0,0], [0,0,1], [0,1,0], [1,0,0]], dtype=torch.long, device=l.device)
                norm000, mask000, alpha_v000 = find_normal(norm_xyzs[0])
                norm001, mask001, alpha_v001 = find_normal(norm_xyzs[1])
                norm010, mask010, alpha_v010 = find_normal(norm_xyzs[2])
                norm100, mask100, alpha_v100 = find_normal(norm_xyzs[3])


                # check connectivity of surfaces
                face001 = torch.concat([surfaces[0,0,1], surfaces[0,1,1], surfaces[1,0,1], surfaces[1,1,1]], axis=-1)
                face010 = torch.concat([surfaces[0,1,0], surfaces[0,1,1], surfaces[1,1,0], surfaces[1,1,1]], axis=-1)
                face100 = torch.concat([surfaces[1,0,0], surfaces[1,0,1], surfaces[1,1,0], surfaces[1,1,1]], axis=-1)
                con001 = torch.count_nonzero(
                    (self.level_set_data[None, :] >= face001.min(axis=-1, keepdim=True).values) & \
                    (self.level_set_data[None, :] <= face001.max(axis=-1, keepdim=True).values),
                    axis=-1
                    ) > 0
                con010 = torch.count_nonzero(
                    (self.level_set_data[None, :] >= face010.min(axis=-1, keepdim=True).values) & \
                    (self.level_set_data[None, :] <= face010.max(axis=-1, keepdim=True).values),
                    axis=-1
                    ) > 0
                con100 = torch.count_nonzero(
                    (self.level_set_data[None, :] >= face100.min(axis=-1, keepdim=True).values) & \
                    (self.level_set_data[None, :] <= face100.max(axis=-1, keepdim=True).values),
                    axis=-1
                    ) > 0

                norm_dz = torch.norm(norm001 - norm000, dim=-1)
                norm_dy = torch.norm(norm010 - norm000, dim=-1)
                norm_dx = torch.norm(norm100 - norm000, dim=-1)
                # filter out gradients on non-exist voxel or non-connected surfaces
                norm_dz[(~mask001)|(~mask000)|(~con001)] = 0.
                norm_dy[(~mask010)|(~mask000)|(~con010)] = 0.
                norm_dx[(~mask100)|(~mask000)|(~con100)] = 0.

                if alpha_weighted_norm_loss:
                    # use alpha value to re-weight the normal loss
                    norm_dz = norm_dz * alpha_v000[:, None] * alpha_v001[:, None]
                    norm_dy = norm_dy * alpha_v000[:, None] * alpha_v010[:, None]
                    norm_dx = norm_dx * alpha_v100[:, None] * alpha_v100[:, None]

                normal_loss = torch.mean(torch.concat([norm_dx,norm_dy,norm_dz],axis=-1))

            out['extra_loss']['normal_loss'] = normal_loss
            out['log_stats']['normal_loss'] = normal_loss

        if B_alpha.numel() == 0:
            out['intersections'] = torch.ones((0, 3), device=origins.device)
            out['intersect_alphas'] = torch.ones((0), device=origins.device)
        else:
            B_samples = torch.ones((B, MS, 3), device=origins.device) * -1.
            B_samples[B_to_sample_mask] = (samples / grid_rescale).to(B_samples.dtype)
            sample_mask = B_alpha > intersect_th
            B_samples[~sample_mask, :] = -1.
            # ids = torch.argmax(sample_mask.long(), dim=-1)
            # intersects = B_samples[torch.arange(B_samples.shape[0]), ids]
            # intersects = intersects[sample_mask.any(axis=-1)]
            intersects = B_samples[sample_mask, :]
            # intersect_alphas = B_weights[sample_mask]
            intersect_alphas = B_alpha[sample_mask]
            out['intersections'] = self.grid2world(intersects)
            out['intersect_alphas'] = intersect_alphas

        # run_backward = True

        if run_backward:
            alpha.retain_grad()
            alpha_raw.retain_grad()
            alpha000.retain_grad()
            B_T.retain_grad()
            B_weights.retain_grad()
            samples.retain_grad()
            # B_rgb.retain_grad()
            # rgb.retain_grad()
            rgb_raw.retain_grad()
            ts.retain_grad()
            ts_raw.retain_grad()
            out['rgb'].retain_grad()
            _R.retain_grad() 
            Q.retain_grad() 
            R.retain_grad() 
            theta.retain_grad() 
            # _S.retain_grad()
            # _T.retain_grad()
            # _U.retain_grad()
            # _g.retain_grad()
            # _h.retain_grad()
            # g.retain_grad()
            # h.retain_grad()
            f3.retain_grad()
            f2.retain_grad()
            f1.retain_grad()
            f0.retain_grad()
            a.retain_grad()
            b.retain_grad()
            c.retain_grad()
            d.retain_grad()

            surface000.retain_grad()
            surface001.retain_grad()
            surface010.retain_grad()
            surface011.retain_grad()
            surface100.retain_grad()
            surface101.retain_grad()
            surface110.retain_grad()
            surface111.retain_grad()
            # surface_scalar.retain_grad()

            # alpha_before_rw.retain_grad()
            # n_surfaces.retain_grad()
            # surface_values.retain_grad()

            s = torch.nn.functional.mse_loss(out['rgb'], torch.zeros_like(out['rgb']))
            s.backward()

            # accum = torch.sum(out['rgb'] * out['rgb'].grad)
            # for i in range(alpha.shape[0]):
            #     total_color = torch.sum(out['rgb'].grad * B_rgb[0, i])
            #     accum = accum - B_weights[0, i] * total_color
            #     grad_alpha = accum / (alpha[i]-1) + total_color * B_T[0,i]


            # add fused surface normal loss

            # ex_l = torch.tensor([
            #     [6,1,14],
            #     [5,1,14],
            # ], dtype=l.dtype, device=l.device)

            # l = torch.concat([ex_l, l], axis=0)

            # cells = l[:, 0] * self.links.shape[2] * self.links.shape[1] + l[:, 1] * self.links.shape[1] + l[:, 2]
            # self._surface_normal_loss_grad_check(cells, 0.1, connectivity_check=False, ignore_empty=True)

        # [3,9,7]

        # torch.sum(out_rgb.grad * (-1 / (1-alpha[0]) * (B_T[0,1]*alpha[1]*B_rgb[0,1] + B_T[0,2]*alpha[2]*B_rgb[0,2]) + (B_T[0,0]*B_rgb[0,0])))
        # torch.sum(out_rgb.grad * (-1 / (1-alpha[1]) * (B_T[0,2]*alpha[2]*B_rgb[0,2]) + (B_T[0,1]*B_rgb[0,1])))

        # dc_da = (-1 / (1-alpha[0]) * (B_T[0,1]*alpha[1]*(B_rgb[0,1] - self.opt.background_brightness) + B_T[0,2]*alpha[2]*(B_rgb[0,2]-self.opt.background_brightness)) + (B_T[0,0]*(B_rgb[0,0]-self.opt.background_brightness)))


        return out

    def prune_grid(
            self,
            density_raw_thres=1,
            dilate=2,
            prune_surf=True
        ):
        with torch.no_grad():
            device = self.density_data.device
            reso = self.links.shape
            X = torch.arange(reso[0])
            Y = torch.arange(reso[1])
            Z = torch.arange(reso[2])
            coords = torch.stack((torch.meshgrid(X, Y, Z)), dim=-1).view(-1, 3).to(device)

            def safe_fetch_data(xyz, data, default=0):
                out = torch.full([xyz.shape[0], data.shape[-1]], default, 
                    device=data.device, dtype=data.dtype)
                edge_mask = (xyz >= 0).all(axis=-1) & (xyz < torch.tensor(reso, device=xyz.device)[None, :]).all(axis=-1)

                x, y, z = xyz[edge_mask].unbind(-1)
                links = self.links[x,y,z]
                valid_mask = links >= 0

                idx = torch.arange(out.shape[0])[edge_mask][valid_mask]
                out[idx] = data[links[valid_mask].long(), :]

                return out

            density_data = safe_fetch_data(coords, self.density_data.data)
            sh_data = safe_fetch_data(coords, self.sh_data.data)
            valid_mask = density_data > density_raw_thres
            valid_mask = valid_mask.view(reso)

            if self.surface_data is not None and prune_surf:
                # prune vertex where the sign of surf is same as all its adjacent vertices
                surface_data = safe_fetch_data(coords, self.surface_data.data)

                X = torch.tensor([-1,0,1], device=device)
                offset = torch.stack((torch.meshgrid(X, X, X)), dim=-1).view(-1, 3)

                # def safe_fetch_surf(xyz, default=0):
                #     out = torch.full([xyz.shape[0]], default, 
                #         device=self.surface_data.device, dtype=self.surface_data.dtype)
                #     edge_mask = (xyz >= 0).all(axis=-1) & (xyz < torch.tensor(reso, device=xyz.device)[None, :]).all(axis=-1)

                #     x, y, z = xyz[edge_mask].unbind(-1)
                #     links = self.links[x,y,z]

                #     valid_mask = links >= 0

                #     idx = torch.arange(out.shape[0])[edge_mask][valid_mask]
                #     out[idx] = self.surface_data.data[links[valid_mask].long(), 0]

                #     return out

                def sign_not_equal(s1, s2):
                    # if any of the s is 0, we assume the sign is not equal
                    return (s1 == 0) | (s2 == 0) | (torch.sign(s1) != torch.sign(s2))

                
                valid_surf_mask = torch.zeros_like(coords[:, :1]).bool()
                for i in range(offset.shape[0]):
                    neighbors = safe_fetch_data(coords + offset[i, None, :], self.surface_data.data)
                    valid_surf_mask |= sign_not_equal(surface_data, neighbors)

                valid_mask &= valid_surf_mask.view(reso)


            for _ in range(int(dilate)):
                valid_mask = _C.dilate(valid_mask)

            valid_mask = valid_mask.view(-1)
            
            self.density_data = nn.Parameter(density_data[valid_mask])
            self.sh_data = nn.Parameter(sh_data[valid_mask])
            if self.surface_data is not None: 
                self.surface_data = nn.Parameter(surface_data[valid_mask])

            init_links = (
                torch.cumsum(valid_mask.to(torch.int32), dim=-1).int() - 1
            )
            init_links[~valid_mask] = -1

            self.links = init_links.view(reso).to(device=device)
            kept_ratio = torch.count_nonzero(valid_mask) / valid_mask.numel()
            print(f'{kept_ratio} of the grids are kept after nerf init!')

            return kept_ratio.item


    def _init_surface_from_density(
        self, 
        alpha_lv_sets=0.5, 
        reset_alpha=False, 
        init_alpha=0.1, 
        surface_rescale=0.1, 
        alpha_clip_thresh=1e-6,
        reset_all=False,
        prune_threshold=1e-8,
        dilate=2,
        use_z_order=True):
        '''
        Initialize surface data from density values
        '''
        with torch.no_grad():

            if self.opt.alpha_activation_type == SIGMOID_FN:
                device = self.density_data.device
                # reset surface level set
                # currently only support single lv set
                self.level_set_data = torch.tensor([0. if self.surface_type == SURFACE_TYPE_SDF else 64.], device=device)
                # convert alpha lv set into alpha raw values
                alpha_lv_sets = torch.tensor(alpha_lv_sets, device=device)
                alpha_lv_sets = torch.logit(alpha_lv_sets)

                # convert density to alphas
                d = torch.tensor([1.,1.,1.])
                d = d / torch.norm(d)
                delta_scale = 1./(d * self._scaling * self._grid_size()).norm()

                alpha_data = 1 - torch.exp(-torch.clamp_min(self.density_data.data, 0) * delta_scale * self.opt.step_size)
                raw_alpha_data = torch.logit(torch.clamp(alpha_data, 1e-10, 1-1e-7))
                if reset_all:
                    init_alpha = utils.logit_np(0.01)
                    self.density_data = nn.Parameter(torch.full_like(alpha_data, init_alpha))
                    self.sh_data = nn.Parameter(torch.zeros_like(self.sh_data.data))
                elif reset_alpha:
                    # self.density_data = nn.Parameter(torch.logit(torch.ones_like(self.density_data.data) * init_alpha).to(torch.float32))
                    
                    new_density_data = torch.full_like(alpha_data, utils.logit_np(0.1), dtype=torch.float32)
                    new_density_data[alpha_data < alpha_clip_thresh] = -25.
                    self.density_data = nn.Parameter(new_density_data)
                else:
                    self.density_data = nn.Parameter(raw_alpha_data.detach().clone().to(torch.float32))


                # copy alpha data then shift according to lv set
                surface_data = raw_alpha_data.detach().clone()
                surface_data = (surface_data - alpha_lv_sets)*surface_rescale + self.level_set_data[0]
                self.surface_data = nn.Parameter(surface_data.to(torch.float32))

            else:
                # alpha_raw is activated via exp (similiar to density)
                # alpha lv set is used as sigma lv set
                device = self.density_data.device

                self.level_set_data = torch.tensor([0. if self.surface_type == SURFACE_TYPE_SDF else 64.], device=device)

                surface_data = self.density_data.detach().clone()
                surface_data = (surface_data - alpha_lv_sets)
                self.surface_data = nn.Parameter(surface_data.to(torch.float32))
                # surface_data = (surface_data - alpha_lv_sets)*surface_rescale + self.level_set_data[0]

                self.prune_grid(prune_threshold, dilate, prune_surf=True)
                
                # better rescale using average grad of surface data
                rand_cells = self._get_rand_cells_non_empty(0.1, contiguous=True)
                xyz = rand_cells
                z = (xyz % self.links.shape[2]).long()
                xy = xyz / self.links.shape[2]
                y = (xy % self.links.shape[1]).long()
                x = (xy / self.links.shape[1]).long()

                # filter out cells at the edge
                edge_mask = (x >= self.links.shape[0] - 1) | (y >= self.links.shape[1] - 1) | (z >= self.links.shape[2] - 1)

                x, y, z = x[~edge_mask], y[~edge_mask], z[~edge_mask]            

                link000 = self.links[x,y,z]
                link100 = self.links[x+1,y,z]
                link010 = self.links[x,y+1,z]
                link001 = self.links[x,y,z+1]

                # filter out cells that are near empty cell
                invalid_mask = (link000 < 0) | (link100 < 0) | (link010 < 0) | (link001 < 0)
                x, y, z = x[~invalid_mask], y[~invalid_mask], z[~invalid_mask]

                link000 = link000[~invalid_mask].long()
                link100 = link100[~invalid_mask].long()
                link010 = link010[~invalid_mask].long()
                link001 = link001[~invalid_mask].long()

                surf000 = self.surface_data[link000]
                surf100 = self.surface_data[link100]
                surf010 = self.surface_data[link010]
                surf001 = self.surface_data[link001]
                
                # note that we assume same aspect ratio for xyz when converting sdf from grid coord to world coord
                # this allows fake sample distance to be calculated easier
                gsz = self._grid_size().mean()
                h = 2.0 * self.radius.mean() / gsz
                h = h.to(device)

                norm_grad = torch.sqrt(
                    ((surf100 - surf000) / h) ** 2. + \
                    ((surf010 - surf000) / h) ** 2. + \
                    ((surf001 - surf000) / h) ** 2.
                )

                # surface_data = surface_data/norm_grad.mean() + self.level_set_data[0]
                # self.surface_data = nn.Parameter(surface_data.to(torch.float32))
                self.surface_data.data = self.surface_data.data/norm_grad.mean() + self.level_set_data[0]

                pass






    def _volume_render_gradcheck_nvol_lerp(self, rays: Rays, return_raylen: bool=False):
        """
        trilerp gradcheck version
        """
        origins = self.world2grid(rays.origins)
        dirs = rays.dirs / torch.norm(rays.dirs, dim=-1, keepdim=True)
        viewdirs = dirs
        B = dirs.size(0)
        assert origins.size(0) == B
        gsz = self._grid_size() # gives grid dimension
        # converting length of dirs to match grid size
        # this is mainly to get the delta scale which converts between grid length to world length?
        dirs = dirs * (self._scaling * gsz).to(device=dirs.device) 
        delta_scale = 1.0 / dirs.norm(dim=1)
        dirs *= delta_scale.unsqueeze(-1)

        if self.basis_type == BASIS_TYPE_3D_TEXTURE:
            sh_mult = self._eval_learned_bases(viewdirs)
        elif self.basis_type == BASIS_TYPE_MLP:
            sh_mult = torch.sigmoid(self._eval_basis_mlp(viewdirs))
        else:
            sh_mult = utils.eval_sh_bases(self.basis_dim, viewdirs)
        # TODO: what is invdirs and t??? This seems to calculate near/far for all rays but how?
        invdirs = 1.0 / dirs

        gsz = self._grid_size()
        gsz_cu = gsz.to(device=dirs.device)
        t1 = (-0.5 - origins) * invdirs
        t2 = (gsz_cu - 0.5 - origins) * invdirs

        t = torch.min(t1, t2)
        t[dirs == 0] = -1e9 # why?
        t = torch.max(t, dim=-1).values.clamp_min_(self.opt.near_clip)

        tmax = torch.max(t1, t2)
        tmax[dirs == 0] = 1e9
        tmax = torch.min(tmax, dim=-1).values
        if return_raylen:
            return tmax - t

        total_alpha = torch.zeros(B, device=origins.device)
        out_rgb = torch.zeros((B, 3), device=origins.device)
        good_indices = torch.arange(B, device=origins.device)

        origins_ini = origins
        dirs_ini = dirs

        # mask out ts that are out of bound
        mask = t <= tmax
        good_indices = good_indices[mask]
        origins = origins[mask]
        dirs = dirs[mask]

        #  invdirs = invdirs[mask]
        del invdirs
        t = t[mask]
        sh_mult = sh_mult[mask]
        tmax = tmax[mask]


        while good_indices.numel() > 0:
            pos = origins + t[:, None] * dirs
            pos = pos.clamp_min_(0.0)
            pos[:, 0] = torch.clamp_max(pos[:, 0], gsz_cu[0] - 1)
            pos[:, 1] = torch.clamp_max(pos[:, 1], gsz_cu[1] - 1)
            pos[:, 2] = torch.clamp_max(pos[:, 2], gsz_cu[2] - 1)

            l = pos.to(torch.long)
            l.clamp_min_(0)
            l[:, 0] = torch.clamp_max(l[:, 0], gsz_cu[0] - 2)
            l[:, 1] = torch.clamp_max(l[:, 1], gsz_cu[1] - 2)
            l[:, 2] = torch.clamp_max(l[:, 2], gsz_cu[2] - 2)
            pos -= l # decimal part of the indices -- used to interpolate

            # BEGIN CRAZY TRILERP
            lx, ly, lz = l.unbind(-1)
            links000 = self.links[lx, ly, lz]
            links001 = self.links[lx, ly, lz + 1]
            links010 = self.links[lx, ly + 1, lz]
            links011 = self.links[lx, ly + 1, lz + 1]
            links100 = self.links[lx + 1, ly, lz]
            links101 = self.links[lx + 1, ly, lz + 1]
            links110 = self.links[lx + 1, ly + 1, lz]
            links111 = self.links[lx + 1, ly + 1, lz + 1]

            sigma000, rgb000 = self._fetch_links(links000) # rgb here is actually SH coefficients
            sigma001, rgb001 = self._fetch_links(links001)
            sigma010, rgb010 = self._fetch_links(links010)
            sigma011, rgb011 = self._fetch_links(links011)
            sigma100, rgb100 = self._fetch_links(links100)
            sigma101, rgb101 = self._fetch_links(links101)
            sigma110, rgb110 = self._fetch_links(links110)
            sigma111, rgb111 = self._fetch_links(links111)

            wa, wb = 1.0 - pos, pos
            c00 = sigma000 * wa[:, 2:] + sigma001 * wb[:, 2:]
            c01 = sigma010 * wa[:, 2:] + sigma011 * wb[:, 2:]
            c10 = sigma100 * wa[:, 2:] + sigma101 * wb[:, 2:]
            c11 = sigma110 * wa[:, 2:] + sigma111 * wb[:, 2:]
            c0 = c00 * wa[:, 1:2] + c01 * wb[:, 1:2]
            c1 = c10 * wa[:, 1:2] + c11 * wb[:, 1:2]
            sigma = c0 * wa[:, :1] + c1 * wb[:, :1]

            c00 = rgb000 * wa[:, 2:] + rgb001 * wb[:, 2:]
            c01 = rgb010 * wa[:, 2:] + rgb011 * wb[:, 2:]
            c10 = rgb100 * wa[:, 2:] + rgb101 * wb[:, 2:]
            c11 = rgb110 * wa[:, 2:] + rgb111 * wb[:, 2:]
            c0 = c00 * wa[:, 1:2] + c01 * wb[:, 1:2]
            c1 = c10 * wa[:, 1:2] + c11 * wb[:, 1:2]
            rgb = c0 * wa[:, :1] + c1 * wb[:, :1]

            # END CRAZY TRILERP

            log_att = (
                -self.opt.step_size
                * torch.relu(sigma[..., 0])
                * delta_scale[good_indices]
            )
            #  weight = torch.exp(log_light_intensity[good_indices]) * (
            #      1.0 - torch.exp(log_att)
            #  )
            delta_alpha = 1.0 - torch.exp(log_att)
            total_alpha_sub = total_alpha[good_indices] # accumulated alpha along each ray
            new_total_alpha = torch.clamp_max(total_alpha_sub + delta_alpha, 1.0)
            weight = new_total_alpha - total_alpha_sub
            total_alpha[good_indices] = new_total_alpha

            # [B', 3, n_sh_coeffs]
            rgb_sh = rgb.reshape(-1, 3, self.basis_dim)
            rgb = torch.clamp_min(
                torch.sum(sh_mult.unsqueeze(-2) * rgb_sh, dim=-1) + 0.5,
                0.0,
            )  # [B', 3]
            rgb = weight[:, None] * rgb[:, :3]

            out_rgb[good_indices] += rgb
            t += self.opt.step_size

            mask = t <= tmax
            good_indices = good_indices[mask]
            origins = origins[mask]
            dirs = dirs[mask]
            t = t[mask]
            sh_mult = sh_mult[mask]
            tmax = tmax[mask]

        # Add background color
        if self.opt.background_brightness:
            out_rgb += (
               (1.0 - total_alpha).unsqueeze(-1)
                * self.opt.background_brightness
            )
        return out_rgb


    def volume_render(
        self, rays: Rays, use_kernel: bool = True, randomize: bool = False,
        return_raylen: bool=False, **kwargs
        # sdf_return_depth: bool=False
    ):
        """
        Standard volume rendering. See grid.opt.* (RenderOptions) for configs.

        :param rays: Rays, (origins (N, 3), dirs (N, 3))
        :param use_kernel: bool, if false uses pure PyTorch version even if on CUDA.
        :param randomize: bool, whether to enable randomness
        :param return_raylen: bool, if true then only returns the length of the
                                    ray-cube intersection and quits
        :return: (N, 3), predicted RGB
        """
        if use_kernel and self.links.is_cuda and _C is not None and not return_raylen:
            assert rays.is_cuda
            basis_data = self._eval_basis_mlp(rays.dirs) if self.basis_type == BASIS_TYPE_MLP \
                                                         else None
            if self.opt.backend == 'surface':
                # with utils.Timing('ours preprocessing'):
                B = rays.origins.shape[0]

                # convert ray o/d to grid space
                # TODO: do it only once here?
                origins = self.world2grid(rays.origins)
                dirs = rays.dirs / torch.norm(rays.dirs, dim=-1, keepdim=True)
                dirs = dirs * (self._scaling * self._grid_size()).to(device=dirs.device)
                delta_scale = 1.0 / dirs.norm(dim=1)
                dirs *= delta_scale.unsqueeze(-1)

                voxel_ls, ray_ids = self.find_ray_voxels_intersection(None, origins, dirs)

                vox_nums = torch.bincount(ray_ids, minlength=B)
                vox_start_i = torch.cumsum(vox_nums, dim=0)
                # shift one
                vox_start_i = torch.concat([torch.tensor([0], device=vox_start_i.device), vox_start_i])[:-1]

                ray_vox = RayVoxIntersecs(voxel_ls.to(torch.int32), vox_start_i.to(torch.int32), vox_nums.to(torch.int32))

                return {'rgb':  _SurfaceRenderFunction.apply(
                    self.density_data,
                    self.surface_data,
                    self.sh_data,
                    basis_data,
                    self.background_data if self.use_background else None,
                    self._to_cpp(replace_basis_data=basis_data),
                    rays._to_cpp(),
                    ray_vox._to_cpp(),
                    self.opt._to_cpp(randomize=randomize),
                    self.opt.backend,
                )}
            elif self.opt.backend == 'surf_trav':
                
                return {'rgb':  _SurfTravRenderFunction.apply(
                    self.density_data,
                    self.surface_data,
                    self.sh_data,
                    basis_data,
                    self.background_data if self.use_background else None,
                    self.fake_sample_std,
                    self._to_cpp(replace_basis_data=basis_data),
                    rays._to_cpp(),
                    self.opt._to_cpp(randomize=randomize),
                    self.opt.backend,
                )}
            else:
                return {'rgb':  _VolumeRenderFunction.apply(
                    self.density_data,
                    self.sh_data,
                    basis_data,
                    self.background_data if self.use_background else None,
                    self._to_cpp(replace_basis_data=basis_data),
                    rays._to_cpp(),
                    self.opt._to_cpp(randomize=randomize),
                    self.opt.backend
                )}
        else:
            warn("Using slow volume rendering, should only be used for debugging")
            if self.surface_type != SURFACE_TYPE_NONE:
                return self._surface_render_gradcheck_lerp(rays, return_raylen=return_raylen, **kwargs)
            elif self.opt.backend == 'nvol':
                return {'rgb': self._volume_render_gradcheck_nvol_lerp(rays, return_raylen=return_raylen)}
            else:
                return {'rgb': self._volume_render_gradcheck_lerp(rays, return_raylen=return_raylen)}


    def volume_render_fused(
        self,
        rays: Rays,
        rgb_gt: torch.Tensor,
        randomize: bool = False,
        beta_loss: float = 0.0,
        sparsity_loss: float = 0.0,
        fused_surf_norm_reg_scale: float = 0.0,
        fused_surf_norm_reg_con_check: bool = True,
        fused_surf_norm_reg_ignore_empty: bool = False,
        no_surface: bool = False,
    ):
        """
        Standard volume rendering with fused MSE gradient generation,
            given a ground truth color for each pixel.
        Will update the *.grad tensors for each parameter
        You can then subtract the grad manually or use the optim_*_step methods

        See grid.opt.* (RenderOptions) for configs.

        :param rays: Rays, (origins (N, 3), dirs (N, 3))
        :param rgb_gt: (N, 3), GT pixel colors, each channel in [0, 1]
        :param randomize: bool, whether to enable randomness
        :param beta_loss: float, weighting for beta loss to add to the gradient.
                                 (fused into the backward pass).
                                 This is average voer the rays in the batch.
                                 Beta loss also from neural volumes:
                                 [Lombardi et al., ToG 2019]
        :param no_surface: bool, whether using no surface initialization.
        :return: (N, 3), predicted RGB
        """
        assert (
            _C is not None and self.sh_data.is_cuda
        ), "CUDA extension is currently required for fused"
        assert rays.is_cuda
        grad = self._get_data_grads()
        rgb_out = torch.zeros_like(rgb_gt)
        basis_data : Optional[torch.Tensor] = None
        if self.basis_type == BASIS_TYPE_MLP:
            with torch.enable_grad():
                basis_data = self._eval_basis_mlp(rays.dirs)
            grad['basis_data'] = torch.empty_like(basis_data)

        self.sparse_grad_indexer = torch.zeros((self.density_data.size(0),),
                dtype=torch.bool, device=self.density_data.device)

        grad_holder = _C.GridOutputGrads()
        grad_holder.grad_density_out = grad['density_data']
        if 'surface_data' in grad:
            grad_holder.grad_surface_out = grad['surface_data']
        grad_holder.grad_sh_out = grad['sh_data']
        if self.basis_type != BASIS_TYPE_SH:
            grad_holder.grad_basis_out = grad['basis_data']
        grad_holder.mask_out = self.sparse_grad_indexer
        if self.use_background:
            grad_holder.grad_background_out = grad['background_data']
            self.sparse_background_indexer = torch.zeros(list(self.background_data.shape[:-1]),
                    dtype=torch.bool, device=self.background_data.device)
            grad_holder.mask_background_out = self.sparse_background_indexer
        if 'fake_sample_std' in grad:
            grad_holder.grad_fake_sample_std_out = grad['fake_sample_std']

        backend = self.opt.backend
        if no_surface and self.opt.backend in ['surface', 'surf_trav']:
            # no surface init, train as if nerf
            backend = 'cuvol'
        cu_fn = _C.__dict__[f"volume_render_{backend}_fused"]

        #  with utils.Timing("actual_render"):
        if backend == 'surface':
            B = rays.origins.shape[0]

            # convert ray o/d to grid space
            # TODO: do it only once here?
            origins = self.world2grid(rays.origins)
            dirs = rays.dirs / torch.norm(rays.dirs, dim=-1, keepdim=True)
            dirs = dirs * (self._scaling * self._grid_size()).to(device=dirs.device)
            delta_scale = 1.0 / dirs.norm(dim=1)
            dirs *= delta_scale.unsqueeze(-1)

            voxel_ls, ray_ids = self.find_ray_voxels_intersection(None, origins, dirs)

            vox_nums = torch.bincount(ray_ids, minlength=B)
            vox_start_i = torch.cumsum(vox_nums, dim=0)
            # shift one
            vox_start_i = torch.concat([torch.tensor([0], device=vox_start_i.device), vox_start_i])[:-1]


            ray_vox = RayVoxIntersecs(voxel_ls.to(torch.int32), vox_start_i.to(torch.int32), vox_nums.to(torch.int32))

            cu_fn(
                self._to_cpp(replace_basis_data=basis_data),
                rays._to_cpp(),
                ray_vox._to_cpp(),
                self.opt._to_cpp(randomize=randomize),
                rgb_gt,
                beta_loss,
                sparsity_loss,
                rgb_out,
                grad_holder
            )
        elif backend == 'surf_trav':
            cu_fn(
                self._to_cpp(replace_basis_data=basis_data),
                rays._to_cpp(),
                self.opt._to_cpp(randomize=randomize),
                rgb_gt,
                beta_loss,
                sparsity_loss,
                fused_surf_norm_reg_scale,
                fused_surf_norm_reg_con_check,
                fused_surf_norm_reg_ignore_empty,
                rgb_out,
                grad_holder
            )
        else: 
            cu_fn(
                self._to_cpp(replace_basis_data=basis_data),
                rays._to_cpp(),
                self.opt._to_cpp(randomize=randomize),
                rgb_gt,
                beta_loss,
                sparsity_loss,
                rgb_out,
                grad_holder
            )
        if self.basis_type == BASIS_TYPE_MLP:
            # Manually trigger MLP backward!
            basis_data.backward(grad['basis_data'])

        self.sparse_sh_grad_indexer = self.sparse_grad_indexer.clone()
        return {'rgb': rgb_out}


    def volume_render_image(
        self, camera: Camera, use_kernel: bool = True, randomize: bool = False,
        batch_size : int = 5000,
        return_raylen: bool=False,
        debug_pixels: list=None, # a list of pixel coords to render, only for debugging
        **kwargs,
    ):
        """
        Standard volume rendering (entire image version).
        See grid.opt.* (RenderOptions) for configs.

        :param camera: Camera
        :param use_kernel: bool, if false uses pure PyTorch version even if on CUDA.
        :param randomize: bool, whether to enable randomness
        :return: (H, W, 3), predicted RGB image
        """
        imrend_fn_name = f"volume_render_{self.opt.backend}_image"
        if kwargs.get('no_surface', False) and self.opt.backend in ['surface', 'surf_trav']:
            imrend_fn_name = "volume_render_cuvol_image"
        if (self.basis_type != BASIS_TYPE_MLP) and (_C is not None) and (imrend_fn_name in _C.__dict__) and \
            (not torch.is_grad_enabled()) and (not return_raylen) and use_kernel:
            # Use the fast image render kernel if available
            cu_fn = _C.__dict__[imrend_fn_name]
            return cu_fn(
                self._to_cpp(),
                camera._to_cpp(),
                self.opt._to_cpp()
            )
        else:
            # Manually generate rays for now
            rays = camera.gen_rays()
            if debug_pixels is None:
                all_rgb_out = []
                for batch_start in range(0, camera.height * camera.width, batch_size):
                    rgb_out_part = self.volume_render(rays[batch_start:batch_start+batch_size],
                                                    use_kernel=use_kernel,
                                                    randomize=randomize,
                                                    return_raylen=return_raylen,
                                                    **kwargs)['rgb']
                    all_rgb_out.append(rgb_out_part)

                all_rgb_out = torch.cat(all_rgb_out, dim=0)
                return all_rgb_out.view(camera.height, camera.width, -1)
            else:
                ray_ids = debug_pixels[:, 0]  + debug_pixels[:, 1] * camera.width
                rgb_out_part = self.volume_render(rays[ray_ids],
                                use_kernel=use_kernel,
                                randomize=randomize,
                                return_raylen=return_raylen,
                                **kwargs)['rgb']
                return rgb_out_part

    def volume_render_depth(self, rays: Rays, sigma_thresh: Optional[float] = None, **kwargs):
        """
        Volumetric depth rendering for rays

        :param rays: Rays, (origins (N, 3), dirs (N, 3))
        :param sigma_thresh: Optional[float]. If None then finds the standard expected termination
                                              (NOTE: this is the absolute length along the ray, not the z-depth as usually expected);
                                              else then finds the first point where sigma strictly exceeds sigma_thresh

        :return: (N,)
        """
        backend = self.opt.backend
        if kwargs.get('no_surface', False) and self.opt.backend in ['surface', 'surf_trav']:
            backend = "cuvol"

        if backend in ['surface', 'surf_trav']:
            if sigma_thresh is None:
                cu_fn = _C.__dict__[f"volume_render_expected_term_{backend}"]
                return cu_fn(
                        self._to_cpp(),
                        rays._to_cpp(),
                        self.opt._to_cpp())
            else:
                cu_fn = _C.__dict__[f"volume_render_sigma_thresh_{backend}"]
                return cu_fn(
                        self._to_cpp(),
                        rays._to_cpp(),
                        self.opt._to_cpp(),
                        sigma_thresh)

        if self.surface_type != SURFACE_TYPE_NONE and not kwargs.get('no_surface', False) and self.surface_data is not None:
            out = self._surface_render_gradcheck_lerp(rays, return_depth=True, **kwargs)
            return out['depth']

        if sigma_thresh is None:
            return _C.volume_render_expected_term(
                    self._to_cpp(),
                    rays._to_cpp(),
                    self.opt._to_cpp())
        else:
            return _C.volume_render_sigma_thresh(
                    self._to_cpp(),
                    rays._to_cpp(),
                    self.opt._to_cpp(),
                    sigma_thresh)

    def volume_render_depth_image(self, camera: Camera, sigma_thresh: Optional[float] = None, batch_size: int = 5000, **kwargs):
        """
        Volumetric depth rendering for full image

        :param camera: Camera, a single camera
        :param sigma_thresh: Optional[float]. If None then finds the standard expected termination
                                              (NOTE: this is the absolute length along the ray, not the z-depth as usually expected);
                                              else then finds the first point where sigma strictly exceeds sigma_thresh

        :return: depth (H, W)
        """
        rays = camera.gen_rays()
        all_depths = []
        for batch_start in range(0, camera.height * camera.width, batch_size):
            depths = self.volume_render_depth(rays[batch_start: batch_start + batch_size], sigma_thresh, **kwargs)
            all_depths.append(depths)
        all_depth_out = torch.cat(all_depths, dim=0)
        return all_depth_out.view(camera.height, camera.width)


    def volume_render_extract_pts(self, camera: Camera, sigma_thresh: Optional[float] = None, batch_size: int = 5000, **kwargs):
        """
        Volume render and caculate depth for each camera ray, then store a 3D point for each ray

        :param camera: Camera, a single camera
        :param sigma_thresh: Optional[float]. If None then finds the standard expected termination
                                              (NOTE: this is the absolute length along the ray, not the z-depth as usually expected);
                                              else then finds the first point where sigma strictly exceeds sigma_thresh

        :return: [N, 3] points array
        """
        rays = camera.gen_rays()
        if self.surface_type == SURFACE_TYPE_NONE or self.opt.backend in ['surf_trav'] or self.surface_data is None:
            # TODO: check whether extracting from depth gives inaccuracy
            # extrac points from depth
            all_depths = []
            for batch_start in range(0, camera.height * camera.width, batch_size):
                depths = self.volume_render_depth(rays[batch_start: batch_start + batch_size], sigma_thresh, **kwargs)
                all_depths.append(depths)
            all_depth_out = torch.cat(all_depths, dim=0)
            
            all_pts = rays.origins + rays.dirs * all_depth_out[:,None]
            # filter 0 depth
            all_pts = all_pts[all_depth_out!=0.]
        else:
            # extract from intersections
            all_pts = []
            for batch_start in range(0, camera.height * camera.width, batch_size):
                pts = self._surface_render_gradcheck_lerp(rays[batch_start: batch_start + batch_size], **kwargs)['intersections']
                all_pts.append(pts)
            all_pts = torch.cat(all_pts, dim=0)
            
        return all_pts

    def volume_render_extract_pts_with_alpha(self, 
    camera: Camera, 
    sigma_thresh: Optional[float] = None, 
    batch_size: int = 5000, 
    max_sample: int = 20,
    **kwargs):
        """
        Volume render and caculate depth for each camera ray, then store a 3D point for each ray

        :param camera: Camera, a single camera
        :param sigma_thresh: Optional[float]. If None then finds the standard expected termination
                                              (NOTE: this is the absolute length along the ray, not the z-depth as usually expected);
                                              else then finds the first point where sigma strictly exceeds sigma_thresh

        :return: [N, 3] points array
        """
        rays = camera.gen_rays()
        all_pts = []
        all_alpha = []
        if self.surface_type == SURFACE_TYPE_NONE:
            raise NotImplementedError
        elif self.opt.backend in ['surf_trav']:
            for batch_start in range(0, camera.height * camera.width, batch_size):
                ray_batch =  rays[batch_start: batch_start + batch_size]
                cu_fn = _C.__dict__[f"extract_pts_surf_trav"]
                depths, alphas =  cu_fn(
                                    self._to_cpp(),
                                    ray_batch._to_cpp(),
                                    self.opt._to_cpp(),
                                    max_sample, # max sample per ray
                                    sigma_thresh)

                mask = depths != 0.
                all_alpha.append(alphas[mask])
                pts = ray_batch.origins[:, None, :] + ray_batch.dirs[:, None, :] * depths[..., None]
                all_pts.append(pts[mask])
            
        else:
            # extract from intersections
            for batch_start in range(0, camera.height * camera.width, batch_size):
                out = self._surface_render_gradcheck_lerp(rays[batch_start: batch_start + batch_size], **kwargs)
                all_pts.append(out['intersections'])
                all_alpha.append(out['intersect_alphas'])

        all_pts = torch.cat(all_pts, dim=0)
        all_alpha = torch.cat(all_alpha, dim=0)
            
        return all_pts, all_alpha

    def resample(
        self,
        reso: Union[int, List[int]],
        sigma_thresh: float = 5.0,
        weight_thresh: float = 0.01,
        dilate: int = 2,
        cameras: Optional[List[Camera]] = None,
        use_z_order: bool=False,
        accelerate: bool=True,
        weight_render_stop_thresh: float = 0.2, # SHOOT, forgot to turn this off for main exps..
        max_elements:int=0,
        batch_size:int=720720
    ):
        """
        Resample and sparsify the grid; used to increase the resolution
        :param reso: int or List[int, int, int], resolution for resampled grid, as in the constructor
        :param sigma_thresh: float, threshold to apply on the sigma (if using sigma thresh i.e. cameras NOT given)
        :param weight_thresh: float, threshold to apply on the weights (if using weight thresh i.e. cameras given)
        :param dilate: int, if true applies dilation of size <dilate> to the 3D mask for nodes to keep in the grid
                             (keep neighbors in all 28 directions, including diagonals, of the desired nodes)
        :param cameras: Optional[List[Camera]], optional list of cameras in OpenCV convention (if given, uses weight thresholding)
        :param use_z_order: bool, if true, stores the data initially in a Z-order curve if possible
        :param accelerate: bool, if true (default), calls grid.accelerate() after resampling
                           to build distance transform table (only if on CUDA)
        :param weight_render_stop_thresh: float, stopping threshold for grid weight render in [0, 1];
                                                 0.0 = no thresholding, 1.0 = hides everything.
                                                 Useful for force-cutting off
                                                 junk that contributes very little at the end of a ray
        :param max_elements: int, if nonzero, an upper bound on the number of elements in the
                upsampled grid; we will adjust the threshold to match it
        """
        with torch.no_grad():
            device = self.links.device
            if isinstance(reso, int):
                reso = [reso] * 3
            else:
                assert (
                    len(reso) == 3
                ), "reso must be an integer or indexable object of 3 ints"

            if use_z_order and not (reso[0] == reso[1] and reso[0] == reso[2] and utils.is_pow2(reso[0])):
                print("Morton code requires a cube grid of power-of-2 size, ignoring...")
                use_z_order = False

            self.capacity: int = reduce(lambda x, y: x * y, reso)
            curr_reso = self.links.shape
            dtype = torch.float32
            reso_facts = [0.5 * curr_reso[i] / reso[i] for i in range(3)]
            X = torch.linspace(
                reso_facts[0] - 0.5,
                curr_reso[0] - reso_facts[0] - 0.5,
                reso[0],
                dtype=dtype,
            )
            Y = torch.linspace(
                reso_facts[1] - 0.5,
                curr_reso[1] - reso_facts[1] - 0.5,
                reso[1],
                dtype=dtype,
            )
            Z = torch.linspace(
                reso_facts[2] - 0.5,
                curr_reso[2] - reso_facts[2] - 0.5,
                reso[2],
                dtype=dtype,
            )
            X, Y, Z = torch.meshgrid(X, Y, Z)
            points = torch.stack((X, Y, Z), dim=-1).view(-1, 3)

            if use_z_order:
                morton = utils.gen_morton(reso[0], dtype=torch.long).view(-1)
                points[morton] = points.clone()
            points = points.to(device=device)

            use_weight_thresh = cameras is not None

            
            all_sample_vals_density = []
            print('Pass 1/2 (density)')
            for i in tqdm(range(0, len(points), batch_size)):
                sample_vals_density, _ = self.sample(
                    points[i : i + batch_size],
                    grid_coords=True,
                    want_colors=False
                )
                sample_vals_density = sample_vals_density
                all_sample_vals_density.append(sample_vals_density)
            self.density_data.grad = None
            self.sh_data.grad = None
            self.sparse_grad_indexer = None
            self.sparse_sh_grad_indexer = None
            self.density_rms = None
            self.sh_rms = None

            sample_vals_density = torch.cat(
                    all_sample_vals_density, dim=0).view(reso)
            del all_sample_vals_density
            if use_weight_thresh:
                gsz = torch.tensor(reso)
                offset = (self._offset * gsz - 0.5).to(device=device)
                scaling = (self._scaling * gsz).to(device=device)
                max_wt_grid = torch.zeros(reso, dtype=torch.float32, device=device)
                print(" Grid weight render", sample_vals_density.shape)
                for i, cam in enumerate(cameras):
                    _C.grid_weight_render(
                        sample_vals_density, cam._to_cpp(),
                        0.5,
                        weight_render_stop_thresh,
                        #  self.opt.last_sample_opaque,
                        False,
                        offset, scaling, max_wt_grid
                    )
                    #  if i % 5 == 0:
                    #      # FIXME DEBUG
                    #      tmp_wt_grid = torch.zeros(reso, dtype=torch.float32, device=device)
                    #      import os
                    #      os.makedirs('wmax_vol', exist_ok=True)
                    #      _C.grid_weight_render(
                    #          sample_vals_density, cam._to_cpp(),
                    #          0.5,
                    #          0.0,
                    #          self.opt.last_sample_opaque,
                    #          offset, scaling, tmp_wt_grid
                    #      )
                    #  np.save(f"wmax_vol/wmax_view{i:05d}.npy", tmp_wt_grid.detach().cpu().numpy())
                #  import sys
                #  sys.exit(0)
                sample_vals_mask = max_wt_grid >= weight_thresh
                if max_elements > 0 and max_elements < max_wt_grid.numel() \
                                    and max_elements < torch.count_nonzero(sample_vals_mask):
                    # To bound the memory usage
                    weight_thresh_bounded = torch.topk(max_wt_grid.view(-1),
                                     k=max_elements, sorted=False).values.min().item()
                    weight_thresh = max(weight_thresh, weight_thresh_bounded)
                    print(' Readjusted weight thresh to fit to memory:', weight_thresh)
                    sample_vals_mask = max_wt_grid >= weight_thresh
                del max_wt_grid
            else:
                sample_vals_mask = sample_vals_density >= sigma_thresh
                if max_elements > 0 and max_elements < sample_vals_density.numel() \
                                    and max_elements < torch.count_nonzero(sample_vals_mask):
                    # To bound the memory usage
                    sigma_thresh_bounded = torch.topk(sample_vals_density.view(-1),
                                     k=max_elements, sorted=False).values.min().item()
                    sigma_thresh = max(sigma_thresh, sigma_thresh_bounded)
                    print(' Readjusted sigma thresh to fit to memory:', sigma_thresh)
                    sample_vals_mask = sample_vals_density >= sigma_thresh

                if self.opt.last_sample_opaque:
                    # Don't delete the last z layer
                    sample_vals_mask[:, :, -1] = 1

            if dilate:
                for i in range(int(dilate)):
                    sample_vals_mask = _C.dilate(sample_vals_mask)
            sample_vals_mask = sample_vals_mask.view(-1)
            sample_vals_density = sample_vals_density.view(-1)
            sample_vals_density = sample_vals_density[sample_vals_mask]
            cnz = torch.count_nonzero(sample_vals_mask).item()

            # Now we can get the colors for the sparse points
            points = points[sample_vals_mask]
            print('Pass 2/2 (color), eval', cnz, 'sparse pts')
            all_sample_vals_sh = []
            for i in tqdm(range(0, len(points), batch_size)):
                _, sample_vals_sh = self.sample(
                    points[i : i + batch_size],
                    grid_coords=True,
                    want_colors=True
                )
                all_sample_vals_sh.append(sample_vals_sh)

            sample_vals_sh = torch.cat(all_sample_vals_sh, dim=0) if len(all_sample_vals_sh) else torch.empty_like(self.sh_data[:0])
            del self.density_data
            del self.sh_data
            del all_sample_vals_sh

            if use_z_order:
                inv_morton = torch.empty_like(morton)
                inv_morton[morton] = torch.arange(morton.size(0), dtype=morton.dtype)
                inv_idx = inv_morton[sample_vals_mask]
                init_links = torch.full(
                    (sample_vals_mask.size(0),), fill_value=-1, dtype=torch.int32
                )
                init_links[inv_idx] = torch.arange(inv_idx.size(0), dtype=torch.int32)
            else:
                init_links = (
                    torch.cumsum(sample_vals_mask.to(torch.int32), dim=-1).int() - 1
                )
                init_links[~sample_vals_mask] = -1

            self.capacity = cnz
            print(" New cap:", self.capacity)
            del sample_vals_mask
            print('density', sample_vals_density.shape, sample_vals_density.dtype)
            print('sh', sample_vals_sh.shape, sample_vals_sh.dtype)
            print('links', init_links.shape, init_links.dtype)
            self.density_data = nn.Parameter(sample_vals_density.view(-1, 1).to(device=device))

            self.sh_data = nn.Parameter(sample_vals_sh.to(device=device))
            self.links = init_links.view(reso).to(device=device)

            if accelerate and self.links.is_cuda:
                self.accelerate()





    def resample_surface(
        self,
        reso: Union[int, List[int]],
        alpha_thresh: float = 1e-6,
        weight_thresh: float = 0.01,
        dilate: int = 2,
        cameras: Optional[List[Camera]] = None,
        use_z_order: bool=False,
        accelerate: bool=True,
        weight_render_stop_thresh: float = 0.2, # SHOOT, forgot to turn this off for main exps..
        max_elements:int=0,
        batch_size:int=720720,
        alpha_empty_val:float = -20. 
    ):
        """
        Resample and sparsify the grid; used to increase the resolution
        :param reso: int or List[int, int, int], resolution for resampled grid, as in the constructor
        :param alpha_thresh: float, threshold to apply on the sigma (if using sigma thresh i.e. cameras NOT given)
        :param weight_thresh: float, threshold to apply on the weights (if using weight thresh i.e. cameras given)
        :param dilate: int, if true applies dilation of size <dilate> to the 3D mask for nodes to keep in the grid
                             (keep neighbors in all 28 directions, including diagonals, of the desired nodes)
        :param cameras: Optional[List[Camera]], optional list of cameras in OpenCV convention (if given, uses weight thresholding)
        :param use_z_order: bool, if true, stores the data initially in a Z-order curve if possible
        :param accelerate: bool, if true (default), calls grid.accelerate() after resampling
                           to build distance transform table (only if on CUDA)
        :param weight_render_stop_thresh: float, stopping threshold for grid weight render in [0, 1];
                                                 0.0 = no thresholding, 1.0 = hides everything.
                                                 Useful for force-cutting off
                                                 junk that contributes very little at the end of a ray
        :param max_elements: int, if nonzero, an upper bound on the number of elements in the
                upsampled grid; we will adjust the threshold to match it
        """
        with torch.no_grad():
            device = self.links.device
            if isinstance(reso, int):
                reso = [reso] * 3
            else:
                assert (
                    len(reso) == 3
                ), "reso must be an integer or indexable object of 3 ints"

            if use_z_order and not (reso[0] == reso[1] and reso[0] == reso[2] and utils.is_pow2(reso[0])):
                print("Morton code requires a cube grid of power-of-2 size, ignoring...")
                use_z_order = False

            self.capacity: int = reduce(lambda x, y: x * y, reso)
            curr_reso = self.links.shape
            dtype = torch.float32

            # upscale_factor = [reso[i] / curr_reso[i] for i in range(3)]
            # assert upscale_factor[0] == upscale_factor[1] and upscale_factor[0] == upscale_factor[2], \
            #     "Currently only support ratio invariant surface upsampling"


            reso_facts = [0. * curr_reso[i] / reso[i] for i in range(3)]
            X = torch.linspace(
                reso_facts[0],
                curr_reso[0] - reso_facts[0] - 0.5,
                reso[0],
                dtype=dtype,
            )
            Y = torch.linspace(
                reso_facts[1],
                curr_reso[1] - reso_facts[1] - 0.5,
                reso[1],
                dtype=dtype,
            )
            Z = torch.linspace(
                reso_facts[2],
                curr_reso[2] - reso_facts[2] - 0.5,
                reso[2],
                dtype=dtype,
            )

            # reso_facts = [0.5 * curr_reso[i] / reso[i] for i in range(3)]
            # X = torch.linspace(
            #     reso_facts[0] - 0.5,
            #     curr_reso[0] - reso_facts[0] - 0.5,
            #     reso[0],
            #     dtype=dtype,
            # )
            # Y = torch.linspace(
            #     reso_facts[1] - 0.5,
            #     curr_reso[1] - reso_facts[1] - 0.5,
            #     reso[1],
            #     dtype=dtype,
            # )
            # Z = torch.linspace(
            #     reso_facts[2] - 0.5,
            #     curr_reso[2] - reso_facts[2] - 0.5,
            #     reso[2],
            #     dtype=dtype,
            # )

            X, Y, Z = torch.meshgrid(X, Y, Z)
            points = torch.stack((X, Y, Z), dim=-1).view(-1, 3)

            if use_z_order:
                morton = utils.gen_morton(reso[0], dtype=torch.long).view(-1)
                points[morton] = points.clone()
            points = points.to(device=device)

            use_weight_thresh = cameras is not None

            
            all_sample_vals_density = []
            print('Pass 1/2 (density)')
            for i in tqdm(range(0, len(points), batch_size)):
                sample_vals_density = _C.sample_grid_raw_alpha(
                    self._to_cpp(grid_coords=True),
                    points[i : i + batch_size],
                    alpha_empty_val if self.opt.alpha_activation_type == SIGMOID_FN else 0.
                )
                sample_vals_density = sample_vals_density
                all_sample_vals_density.append(sample_vals_density)
            self.density_data.grad = None
            self.surface_data.grad = None
            self.sh_data.grad = None
            self.sparse_grad_indexer = None
            self.sparse_sh_grad_indexer = None
            self.density_rms = None
            self.sh_rms = None

            sample_vals_density = torch.cat(
                    all_sample_vals_density, dim=0).view(reso)
            del all_sample_vals_density
            if use_weight_thresh:
                gsz = torch.tensor(reso)
                offset = (self._offset * gsz - 0.5).to(device=device)
                scaling = (self._scaling * gsz).to(device=device)
                max_wt_grid = torch.zeros(reso, dtype=torch.float32, device=device)
                print(" Grid weight render", sample_vals_density.shape)
                for i, cam in enumerate(cameras):
                    _C.grid_weight_render(
                        sample_vals_density, cam._to_cpp(),
                        0.5,
                        weight_render_stop_thresh,
                        #  self.opt.last_sample_opaque,
                        False,
                        offset, scaling, max_wt_grid
                    )
                sample_vals_mask = max_wt_grid >= weight_thresh
                if max_elements > 0 and max_elements < max_wt_grid.numel() \
                                    and max_elements < torch.count_nonzero(sample_vals_mask):
                    # To bound the memory usage
                    weight_thresh_bounded = torch.topk(max_wt_grid.view(-1),
                                     k=max_elements, sorted=False).values.min().item()
                    weight_thresh = max(weight_thresh, weight_thresh_bounded)
                    print(' Readjusted weight thresh to fit to memory:', weight_thresh)
                    sample_vals_mask = max_wt_grid >= weight_thresh
                del max_wt_grid
            else:
                if self.opt.alpha_activation_type == SIGMOID_FN:
                    # convert alpha value to raw value
                    alpha_thresh = np.log(alpha_thresh / (1. - alpha_thresh))
                sample_vals_mask = sample_vals_density >= alpha_thresh
                if max_elements > 0 and max_elements < sample_vals_density.numel() \
                                    and max_elements < torch.count_nonzero(sample_vals_mask):
                    # To bound the memory usage
                    alpha_thresh_bounded = torch.topk(sample_vals_density.view(-1),
                                     k=max_elements, sorted=False).values.min().item()
                    alpha_thresh = max(alpha_thresh, alpha_thresh_bounded)
                    print(' Readjusted alpha thresh to fit to memory:', alpha_thresh)
                    sample_vals_mask = sample_vals_density >= alpha_thresh

                if self.opt.last_sample_opaque:
                    # Don't delete the last z layer
                    sample_vals_mask[:, :, -1] = 1

            if dilate:
                for i in range(int(dilate)):
                    sample_vals_mask = _C.dilate(sample_vals_mask)
            sample_vals_mask = sample_vals_mask.view(-1)
            sample_vals_density = sample_vals_density.view(-1)
            sample_vals_density = sample_vals_density[sample_vals_mask]
            cnz = torch.count_nonzero(sample_vals_mask).item()

            grid_ratio = cnz / sample_vals_mask.numel()

            print(f'{grid_ratio} of the grids are kept!')

            # Now we can get the colors for the sparse points
            points = points[sample_vals_mask]
            print('Pass 2/2 (color), eval', cnz, 'sparse pts')
            all_sample_vals_sh = []
            all_sample_vals_surf = []
            for i in tqdm(range(0, len(points), batch_size)):
                sample_vals_sh, sample_vals_surf = self.sample_surface(
                    points[i : i + batch_size],
                    grid_coords=True,
                    want_colors=True
                )
                all_sample_vals_sh.append(sample_vals_sh)
                all_sample_vals_surf.append(sample_vals_surf)

            sample_vals_sh = torch.cat(all_sample_vals_sh, dim=0) if len(all_sample_vals_sh) else torch.empty_like(self.sh_data[:0])
            sample_vals_surf = torch.cat(all_sample_vals_surf, dim=0) if len(all_sample_vals_surf) else torch.empty_like(self.surface_data[:0])
            # del self.density_data
            # del self.surface_data
            # del self.sh_data
            del all_sample_vals_sh
            del all_sample_vals_surf

            if use_z_order:
                inv_morton = torch.empty_like(morton)
                inv_morton[morton] = torch.arange(morton.size(0), dtype=morton.dtype)
                inv_idx = inv_morton[sample_vals_mask]
                init_links = torch.full(
                    (sample_vals_mask.size(0),), fill_value=-1, dtype=torch.int32
                )
                init_links[inv_idx] = torch.arange(inv_idx.size(0), dtype=torch.int32)
            else:
                init_links = (
                    torch.cumsum(sample_vals_mask.to(torch.int32), dim=-1).int() - 1
                )
                init_links[~sample_vals_mask] = -1

            self.capacity = cnz
            print(" New cap:", self.capacity)
            del sample_vals_mask
            print('density', sample_vals_density.shape, sample_vals_density.dtype)
            print('sh', sample_vals_sh.shape, sample_vals_sh.dtype)
            print('links', init_links.shape, init_links.dtype)
            self.density_data = nn.Parameter(sample_vals_density.view(-1, 1).to(device=device))
            self.sh_data = nn.Parameter(sample_vals_sh.to(device=device))
            self.surface_data = nn.Parameter(sample_vals_surf.view(-1, 1).to(device=device))
            self.links = init_links.view(reso).to(device=device)

            if accelerate and self.links.is_cuda:
                self.accelerate()

            return grid_ratio



    def sparsify_background(
        self,
        sigma_thresh: float = 1.0,
        dilate: int = 1,  # BEFORE resampling!
    ):
        device = self.background_links.device
        sigma_mask = torch.zeros(list(self.background_links.shape) + [self.background_nlayers],
                dtype=torch.bool, device=device).view(-1, self.background_nlayers)
        nonempty_mask = self.background_links.view(-1) >= 0
        data_mask = self.background_data[..., -1] >= sigma_thresh
        sigma_mask[nonempty_mask] = data_mask
        sigma_mask = sigma_mask.view(list(self.background_links.shape) + [self.background_nlayers])
        for _ in range(int(dilate)):
            sigma_mask = _C.dilate(sigma_mask)

        sigma_mask = sigma_mask.any(-1) & nonempty_mask.view(self.background_links.shape)
        self.background_links[~sigma_mask] = -1
        retain_vals = self.background_links[sigma_mask]
        self.background_links[sigma_mask] = torch.arange(retain_vals.size(0),
                dtype=torch.int32, device=device)
        self.background_data = nn.Parameter(
                    self.background_data.data[retain_vals.long()]
                )


    def resize(self, basis_dim: int):
        """
        Modify the size of the data stored in the voxels. Called expand/shrink in svox 1.

        :param basis_dim: new basis dimension, must be square number
        """
        assert utils.isqrt(basis_dim) is not None, "basis_dim (SH) must be a square number"
        assert (
            basis_dim >= 1 and basis_dim <= utils.MAX_SH_BASIS
        ), f"basis_dim 1-{utils.MAX_SH_BASIS} supported"
        old_basis_dim = self.basis_dim
        self.basis_dim = basis_dim
        device = self.sh_data.device
        old_data = self.sh_data.data.cpu()

        shrinking = basis_dim < old_basis_dim
        sigma_arr = torch.tensor([0])
        if shrinking:
            shift = old_basis_dim
            arr = torch.arange(basis_dim)
            remap = torch.cat([arr, shift + arr, 2 * shift + arr])
        else:
            shift = basis_dim
            arr = torch.arange(old_basis_dim)
            remap = torch.cat([arr, shift + arr, 2 * shift + arr])

        del self.sh_data
        new_data = torch.zeros((old_data.size(0), 3 * basis_dim + 1), device="cpu")
        if shrinking:
            new_data[:] = old_data[..., remap]
        else:
            new_data[..., remap] = old_data
        new_data = new_data.to(device=device)
        self.sh_data = nn.Parameter(new_data)
        self.sh_rms = None

    def accelerate(self):
        """
        Accelerate
        """
        assert (
            _C is not None and self.links.is_cuda
        ), "CUDA extension is currently required for accelerate"
        _C.accel_dist_prop(self.links)

    def world2grid(self, points):
        """
        World coordinates to grid coordinates. Grid coordinates are
        normalized to [0, n_voxels] in each side

        :param points: (N, 3)
        :return: (N, 3)
        """
        gsz = self._grid_size()
        # offset = self._offset * gsz - 0.5
        offset = self._offset * gsz
        scaling = self._scaling * gsz
        return torch.addcmul(
            offset.to(device=points.device), points, scaling.to(device=points.device)
        )

    def grid2world(self, points):
        """
        Grid coordinates to world coordinates. Grid coordinates are
        normalized to [0, n_voxels] in each side

        :param points: (N, 3)
        :return: (N, 3)
        """
        gsz = self._grid_size()
        # roffset = self.radius * (1.0 / gsz - 1.0) + self.center
        roffset = self.radius * (-1.0) + self.center
        rscaling = 2.0 * self.radius / gsz
        return torch.addcmul(
            roffset.to(device=points.device), points, rscaling.to(device=points.device)
        )

    def save(self, path: str, compress: bool = False, step_id: int = 0):
        """
        Save to a path
        """
        save_fn = np.savez_compressed if compress else np.savez
        data = {
            "radius":self.radius.numpy(),
            "center":self.center.numpy(),
            "links":self.links.cpu().numpy(),
            "density_data":self.density_data.data.cpu().numpy(),
            "sh_data":self.sh_data.data.cpu().numpy().astype(np.float16),
            "step_id": step_id
        }
        if self.surface_type != SURFACE_TYPE_NONE:
            data['surface_data'] = self.surface_data.data.cpu().numpy()
        if self.level_set_data is not None:
            data['level_set_data'] = self.level_set_data.cpu().numpy()
        if torch.is_tensor(self.fake_sample_std):
            data['fake_sample_std'] = self.fake_sample_std.data.cpu().numpy()
        if self.basis_type == BASIS_TYPE_3D_TEXTURE:
            data['basis_data'] = self.basis_data.data.cpu().numpy()
        elif self.basis_type == BASIS_TYPE_MLP:
            utils.net_to_dict(data, "basis_mlp", self.basis_mlp)
            data['mlp_posenc_size'] = np.int32(self.mlp_posenc_size)
            data['mlp_width'] = np.int32(self.mlp_width)

        if self.use_background:
            data['background_links'] = self.background_links.cpu().numpy()
            data['background_data'] = self.background_data.data.cpu().numpy()
        data['basis_type'] = self.basis_type
        data['surface_type'] = self.surface_type

        save_fn(
            path,
            **data
        )

    def save_sdf(self, path: str = './sdf_grid.npy'):
        assert self.surface_type == SURFACE_TYPE_SDF, 'onlt supported for sdf surface_type!'

        sdf_voxel = {
            'sdf_data': self.sdf_data.data.cpu().numpy(),
            'links': self.links.cpu().numpy()
        }

        np.save(path, sdf_voxel)


    @classmethod
    def load(cls, path: str, device: Union[torch.device, str] = "cpu"):
        """
        Load from path
        """
        z = np.load(path, allow_pickle=True)
        if "data" in z.keys():
            # Compatibility
            all_data = z.f.data
            sh_data = all_data[..., 1:]
            density_data = all_data[..., :1]
        else:
            sh_data = z.f.sh_data
            density_data = z.f.density_data
            surface_type = z.f.surface_type.item()
            if surface_type != SURFACE_TYPE_NONE:
                surface_data = z.f.surface_data

        if 'background_data' in z:
            background_data = z['background_data']
            background_links = z['background_links']
        else:
            background_data = None

        links = z.f.links
        basis_dim = (sh_data.shape[1]) // 3
        radius = z.f.radius.tolist() if "radius" in z.files else [1.0, 1.0, 1.0]
        center = z.f.center.tolist() if "center" in z.files else [0.0, 0.0, 0.0]
        grid = cls(
            list(links.shape),
            radius=radius,
            center=center,
            basis_dim=basis_dim,
            use_z_order=False,
            device=device,
            basis_type=z['basis_type'].item() if 'basis_type' in z else BASIS_TYPE_SH,
            mlp_posenc_size=z['mlp_posenc_size'].item() if 'mlp_posenc_size' in z else 0,
            mlp_width=z['mlp_width'].item() if 'mlp_width' in z else 16,
            background_nlayers=0,
            surface_type=surface_type
        )
        if "step_id" in z.keys():
            grid.step_id = z.f.step_id.item()
        if sh_data.dtype != np.float32:
            sh_data = sh_data.astype(np.float32)
        if density_data.dtype != np.float32:
            density_data = density_data.astype(np.float32)
        if surface_type != SURFACE_TYPE_NONE and surface_data.dtype != np.float32:
            surface_data = surface_data.astype(np.float32)
        sh_data = torch.from_numpy(sh_data).to(device=device)
        density_data = torch.from_numpy(density_data).to(device=device)
        grid.sh_data = nn.Parameter(sh_data)
        grid.density_data = nn.Parameter(density_data)
        if surface_type != SURFACE_TYPE_NONE:
            surface_data = torch.from_numpy(surface_data).to(device=device)
            grid.surface_data = nn.Parameter(surface_data)
        if 'level_set_data' in z:
            grid.level_set_data = torch.from_numpy(z.f.level_set_data.astype(np.float32)).to(device=device)
        if 'fake_sample_std' in z:
            grid.fake_sample_std = torch.from_numpy(z.f.fake_sample_std.astype(np.float32)).to(device=device)

        grid.links = torch.from_numpy(links).to(device=device)
        grid.capacity = grid.sh_data.size(0)

        # Maybe load basis_data
        if grid.basis_type == BASIS_TYPE_MLP:
            utils.net_from_dict(z, "basis_mlp", grid.basis_mlp)
            grid.basis_mlp = grid.basis_mlp.to(device=device)
        elif grid.basis_type == BASIS_TYPE_3D_TEXTURE or \
            "basis_data" in z.keys():
            # Note: Checking for basis_data for compatibility with earlier vers
            # where basis_type not stored
            basis_data = torch.from_numpy(z.f.basis_data).to(device=device)
            grid.basis_type = BASIS_TYPE_3D_TEXTURE
            grid.basis_data = nn.Parameter(basis_data)
        else:
            grid.basis_data = nn.Parameter(grid.basis_data.data.to(device=device))

        if background_data is not None:
            background_data = torch.from_numpy(background_data).to(device=device)
            grid.background_nlayers = background_data.shape[1]
            grid.background_reso = background_links.shape[1]
            grid.background_data = nn.Parameter(background_data)
            grid.background_links = torch.from_numpy(background_links).to(device=device)
        else:
            grid.background_data.data = grid.background_data.data.to(device=device)

        if grid.links.is_cuda and _C is not None:
            grid.accelerate()
        return grid

    def to_svox1(self, device: Union[torch.device, str, None] = None):
        """
        Convert the grid to a svox 1 octree. Requires svox (pip install svox)

        :param device: device to put the octree. None = grid data's device
        """
        assert (
            self.is_cubic_pow2
        ), "Grid must be cubic and power-of-2 to be compatible with svox octree"
        if device is None:
            device = self.sh_data.device
        import svox

        n_refine = int(np.log2(self.links.size(0))) - 1

        t = svox.N3Tree(
            data_format=f"SH{self.basis_dim}",
            init_refine=0,
            radius=self.radius.tolist(),
            center=self.center.tolist(),
            device=device,
        )

        curr_reso = self.links.shape
        dtype = torch.float32
        X = (torch.arange(curr_reso[0], dtype=dtype, device=device) + 0.5) / curr_reso[
            0
        ]
        Y = (torch.arange(curr_reso[1], dtype=dtype, device=device) + 0.5) / curr_reso[
            0
        ]
        Z = (torch.arange(curr_reso[2], dtype=dtype, device=device) + 0.5) / curr_reso[
            0
        ]
        X, Y, Z = torch.meshgrid(X, Y, Z)
        points = torch.stack((X, Y, Z), dim=-1).view(-1, 3)

        mask = self.links.view(-1) >= 0
        points = points[mask.to(device=device)]
        index = svox.LocalIndex(points)
        print("n_refine", n_refine)
        for i in tqdm(range(n_refine)):
            t[index].refine()

        t[index, :-1] = self.sh_data.data.to(device=device)
        t[index, -1:] = self.density_data.data.to(device=device)
        if self.surface_type != SURFACE_TYPE_NONE:
            raise NotImplementedError
        return t

    def tv(self, logalpha: bool=False, logalpha_delta: float=2.0,
           ndc_coeffs: Tuple[float, float] = (-1.0, -1.0)):
        """
        Compute total variation over sigma,
        similar to Neural Volumes [Lombardi et al., ToG 2019]

        :return: torch.Tensor, size scalar, the TV value (sum over channels,
                 mean over voxels)
        """
        assert (
            _C is not None and self.sh_data.is_cuda
        ), "CUDA extension is currently required for total variation"
        assert not logalpha, "No longer supported"

        if self.surface_type != SURFACE_TYPE_NONE:
            raise NotImplementedError

        return _TotalVariationFunction.apply(
                self.density_data, self.links, 0, 1, logalpha, logalpha_delta,
                False, ndc_coeffs)

    def tv_color(self,
                 start_dim: int = 0, end_dim: Optional[int] = None,
                 logalpha: bool=False, logalpha_delta: float=2.0,
                 ndc_coeffs: Tuple[float, float] = (-1.0, -1.0)):
        """
        Compute total variation on color

        :param start_dim: int, first color channel dimension to compute TV over (inclusive).
                          Default 0.
        :param end_dim: int, last color channel dimension to compute TV over (exclusive).
                          Default None = all dimensions until the end.

        :return: torch.Tensor, size scalar, the TV value (sum over channels,
                 mean over voxels)
        """
        assert (
            _C is not None and self.sh_data.is_cuda
        ), "CUDA extension is currently required for total variation"
        assert not logalpha, "No longer supported"
        if end_dim is None:
            end_dim = self.sh_data.size(1)
        end_dim = end_dim + self.sh_data.size(1) if end_dim < 0 else end_dim
        start_dim = start_dim + self.sh_data.size(1) if start_dim < 0 else start_dim
        return _TotalVariationFunction.apply(
            self.sh_data, self.links, start_dim, end_dim, logalpha, logalpha_delta,
            True,
            ndc_coeffs
        )

    def tv_basis(self):
        bd = self.basis_data
        return torch.mean(torch.sqrt(1e-5 +
                    (bd[:-1, :-1, 1:] - bd[:-1, :-1, :-1]) ** 2 +
                    (bd[:-1, 1:, :-1] - bd[:-1, :-1, :-1]) ** 2 +
                    (bd[1:, :-1, :-1] - bd[:-1, :-1, :-1]) ** 2).sum(dim=-1))

    def inplace_tv_grad(self, grad: torch.Tensor,
                        scaling: float = 1.0,
                        sparse_frac: float = 0.01,
                        logalpha: bool=False, logalpha_delta: float=2.0,
                        ndc_coeffs: Tuple[float, float] = (-1.0, -1.0),
                        contiguous: bool = True
                    ):
        """
        Add gradient of total variation for sigma as in Neural Volumes
        [Lombardi et al., ToG 2019]
        directly into the gradient tensor, multiplied by 'scaling'
        """
        # if self.surface_type != SURFACE_TYPE_NONE:
        #     raise NotImplementedError

        assert (
            _C is not None and self.density_data.is_cuda and grad.is_cuda
        ), "CUDA extension is currently required for total variation"

        assert not logalpha, "No longer supported"
        rand_cells = self._get_rand_cells(sparse_frac, contiguous=contiguous)
        if rand_cells is not None:
            if rand_cells.size(0) > 0:
                _C.tv_grad_sparse(self.links, self.density_data,
                        rand_cells,
                        self._get_sparse_grad_indexer(),
                        0, 1, scaling,
                        logalpha, logalpha_delta,
                        False,
                        self.opt.last_sample_opaque,
                        ndc_coeffs[0], ndc_coeffs[1],
                        grad)
        else:
            _C.tv_grad(self.links, self.density_data, 0, 1, scaling,
                    logalpha, logalpha_delta,
                    False,
                    ndc_coeffs[0], ndc_coeffs[1],
                    grad)
            self.sparse_grad_indexer : Optional[torch.Tensor] = None

    # def inplace_alpha_lap_grad(self, grad: torch.Tensor,
    #                     scaling: float = 1.0,
    #                     sparse_frac: float = 0.01,
    #                     ndc_coeffs: Tuple[float, float] = (-1.0, -1.0),
    #                     density_is_sigma: bool = False,
    #                     contiguous: bool = True,
    #                     use_kernel: bool = True
    #                 ):
    #     # if self.surface_type != SURFACE_TYPE_NONE:
    #     #     raise NotImplementedError

    #     if not use_kernel:
    #         raise NotImplementedError

    #     assert (
    #         _C is not None and self.density_data.is_cuda and grad.is_cuda
    #     ), "CUDA extension is currently required for alpha lap"

    #     rand_cells = self._get_rand_cells(sparse_frac, contiguous=contiguous)
    #     if rand_cells is not None:
    #         if rand_cells.size(0) > 0:
    #             world_step = 0
    #             if density_is_sigma:
    #                 # density data stores sigma, need to give world step

    #                 # note the world step would be incorrect if 3 dims of grid are not equal

    #                 d = torch.tensor([1.,1.,1.])
    #                 d = d / torch.norm(d)
    #                 delta_scale = 1./(d * self._scaling * self._grid_size()).norm()
    #                 world_step = delta_scale * self.opt.step_size

    #             _C.alpha_lap_grad_sparse(self.links, self.density_data,
    #                     rand_cells,
    #                     self._get_sparse_grad_indexer(),
    #                     0, 1, scaling,
    #                     ndc_coeffs[0], ndc_coeffs[1],
    #                     world_step,
    #                     grad)
    #     else:
    #         raise NotImplementedError

    def _alpha_surf_sparsify_grad_check(self, rand_cells, scale_alpha, scale_surf, surf_decrease, surf_thresh, device='cuda'):
        xyz = rand_cells
        z = (xyz % self.links.shape[2]).long()
        xy = xyz / self.links.shape[2]
        y = (xy % self.links.shape[1]).long()
        x = (xy / self.links.shape[1]).long()


        links = self.links[x,y,z]
        alpha_raws, _ , surfaces = self._fetch_links(links)

        alpha_raws_de = alpha_raws.detach().clone()

        alpha_loss = torch.log(torch.sigmoid(alpha_raws))
        surface_loss = torch.where(alpha_raws < surf_thresh, 
        torch.exp(-alpha_raws_de) / (1+torch.exp(-alpha_raws_de)), torch.zeros_like(alpha_loss)) * surfaces
        if not surf_decrease:
            surface_loss = -surface_loss

        loss = torch.mean(scale_alpha * alpha_loss + scale_surf * surface_loss)
        loss.backward()

        return loss

    def inplace_alpha_surf_sparsify_grad(self, 
                        grad_alpha: torch.Tensor,
                        grad_surf: torch.Tensor,
                        scaling_alpha: float = 1.0,
                        scaling_surf: float = 1.0,
                        sparse_frac: float = 0.01,
                        surf_sparse_decrease: bool = True,
                        surf_sparse_thresh: float = 0.01,
                        alpha_sparsify_bound: float = 1e-6,
                        surf_sparsify_bound: float = -1,
                        contiguous: bool = True
                    ):
        '''
        Inplace sparsify grad for alpha and surf
        Alpha is sparsified by log(alpha)
        Surface is sparsified by reducing/increasing values depending on surf_sparse_decrease

        surf_sparse_decrease: sparsify surface scalars by decreasing or increasing the values
        '''
        # if self.surface_type != SURFACE_TYPE_NONE:
        #     raise NotImplementedError


        assert (
            _C is not None and self.density_data.is_cuda
        ), "CUDA extension is currently required for alpha lap"

        if sparse_frac * self.links.size(0) * self.links.size(1) * self.links.size(2) < 1.:
            return

        rand_cells = self._get_rand_cells(sparse_frac, contiguous=contiguous)

        # grid_size = self.links.size(0) * self.links.size(1) * self.links.size(2)
        # # sparse_num = max(int(sparse_frac * grid_size), 1)
        # sparse_num = 100
        # start = np.random.randint(0, grid_size)
        # rand_cells = torch.arange(start, start + sparse_num, dtype=torch.int32, device=
        #                                 self.links.device)

        # if start > grid_size - sparse_num:
        #     rand_cells[grid_size - sparse_num - start:] -= grid_size
            

        if rand_cells is not None:
            if rand_cells.size(0) > 0:
                _C.alpha_surf_sparsify_grad_sparse(self.links, 
                        self.density_data,
                        self.surface_data,
                        rand_cells,
                        self._get_sparse_grad_indexer(),
                        scaling_alpha, scaling_surf,
                        surf_sparse_decrease,
                        utils.logit_np(surf_sparse_thresh),
                        utils.logit_np(alpha_sparsify_bound),
                        surf_sparsify_bound,
                        grad_alpha,
                        grad_surf)
        else:
            raise NotImplementedError

    def inplace_tv_surface_grad(self, grad: torch.Tensor,
                                scaling: float = 1.0,
                                sparse_frac: float = 0.01,
                                logalpha: bool=False, logalpha_delta: float=2.0,
                                ndc_coeffs: Tuple[float, float] = (-1.0, -1.0),
                                contiguous: bool = True
                            ):
        """
        Add gradient of total variation for sigma as in Neural Volumes
        [Lombardi et al., ToG 2019]
        directly into the gradient tensor, multiplied by 'scaling'
        """

        assert (
            _C is not None and self.surface_data.is_cuda and grad.is_cuda
        ), "CUDA extension is currently required for total variation"

        assert not logalpha, "No longer supported"
        rand_cells = self._get_rand_cells(sparse_frac, contiguous=contiguous)
        if rand_cells is not None:
            if rand_cells.size(0) > 0:
                _C.tv_grad_sparse(self.links, self.surface_data,
                        rand_cells,
                        self._get_sparse_grad_indexer(),
                        0, 1, scaling,
                        logalpha, logalpha_delta,
                        False,
                        self.opt.last_sample_opaque,
                        ndc_coeffs[0], ndc_coeffs[1],
                        grad)
        else:
            _C.tv_grad(self.links, self.surface_data, 0, 1, scaling,
                    logalpha, logalpha_delta,
                    False,
                    ndc_coeffs[0], ndc_coeffs[1],
                    grad)
            self.sparse_grad_indexer : Optional[torch.Tensor] = None

    def _surface_normal_loss_grad_check(
        self, 
        rand_cells, 
        scaling, 
        device='cuda', 
        connectivity_check=True, 
        alpha_weighted_norm_loss=False,
        ignore_empty=False,
        ):
        xyz = rand_cells
        z = (xyz % self.links.shape[2]).long()
        xy = xyz / self.links.shape[2]
        y = (xy % self.links.shape[1]).long()
        x = (xy / self.links.shape[1]).long()

        coords = torch.tensor([
            [0,0,0],
            [0,0,1],
            [0,1,0],
            [0,1,1],
            [1,0,0],
            [1,0,1],
            [1,1,0],
            [1,1,1],

            [0,0,2],
            [0,1,2],
            [1,0,2],
            [1,1,2],

            [0,2,0],
            [0,2,1],
            [1,2,0],
            [1,2,1],

            [2,0,0],
            [2,0,1],
            [2,1,0],
            [2,1,1],
        ], dtype=torch.long, device=device)

        links=torch.zeros([3,3,3,xyz.shape[0]], dtype=torch.long, device=device)
        alphas=torch.zeros([3,3,3,xyz.shape[0], 1], dtype=self.density_data.dtype, device=device)
        surfaces=torch.zeros([3,3,3,xyz.shape[0], 1], dtype=self.surface_data.dtype, device=device)

        for i in range(coords.shape[0]):
            def maybe_get_link(x,y,z):
                _links = torch.ones_like(x, dtype=torch.long) * -1
                invalid_xyz_mask = (torch.stack([x,y,z], axis=-1) >= torch.tensor(self.links.shape, device=device)).any(axis=-1)
                _links[~invalid_xyz_mask] = self.links[x[~invalid_xyz_mask], y[~invalid_xyz_mask], z[~invalid_xyz_mask]].long()

                return _links

            links[coords[i,0], coords[i,1], coords[i,2]] = maybe_get_link(x+coords[i,0], y+coords[i,1], z+coords[i,2])
            alphas[coords[i,0], coords[i,1], coords[i,2]], _ , \
                surfaces[coords[i,0], coords[i,1], coords[i,2]] = self._fetch_links(links[coords[i,0], coords[i,1], coords[i,2]])

        def find_normal(norm_xyz):
            x,y,z = norm_xyz.unbind(-1)

            dx = ((surfaces[x+1,y,z]+surfaces[x+1,y,z+1]+surfaces[x+1,y+1,z]+surfaces[x+1,y+1,z+1]) - \
                (surfaces[x,y,z]+surfaces[x,y,z+1]+surfaces[x,y+1,z]+surfaces[x,y+1,z+1])) /4
            dy = ((surfaces[x,y+1,z]+surfaces[x,y+1,z+1]+surfaces[x+1,y+1,z]+surfaces[x+1,y+1,z+1]) - \
                (surfaces[x,y,z]+surfaces[x,y,z+1]+surfaces[x+1,y,z]+surfaces[x+1,y,z+1]))/4
            dz = ((surfaces[x,y,z+1]+surfaces[x,y+1,z+1]+surfaces[x+1,y,z+1]+surfaces[x+1,y+1,z+1]) - \
                (surfaces[x,y,z]+surfaces[x,y+1,z]+surfaces[x+1,y,z]+surfaces[x+1,y+1,z]))/4

            normals = torch.stack([dx, dy, dz], dim=-1)
            # normals = normals / torch.clamp(torch.norm(normals, dim=-1, keepdim=True), 1e-10)

            # check if there is non-exist vertex
            coords = torch.tensor([
                [0,0,0],
                [0,0,1],
                [0,1,0],
                [0,1,1],
                [1,0,0],
                [1,0,1],
                [1,1,0],
                [1,1,1],
                ], dtype=torch.long, device=device)
            ver_xyzs = norm_xyz[None, :] + coords
            valid_mask = torch.ones(links.shape[-1], device=links.device).bool()
            for i in range(ver_xyzs.shape[0]):
                valid_mask = (valid_mask) & (links[ver_xyzs[i,0], ver_xyzs[i,1], ver_xyzs[i,2]] >= 0)

            empty_mask = ((surfaces[x,y,z] <= 0.) & (surfaces[x,y,z+1] <= 0.) & (surfaces[x,y+1,z] <= 0.) & (surfaces[x,y+1,z+1] <= 0.) & (surfaces[x+1,y,z] <= 0.) & (surfaces[x+1,y,z+1] <= 0.) & (surfaces[x+1,y+1,z] <= 0.) & (surfaces[x+1,y+1,z+1] <= 0.)) | \
                        ((surfaces[x,y,z] >= 0.) & (surfaces[x,y,z+1] >= 0.) & (surfaces[x,y+1,z] >= 0.) & (surfaces[x,y+1,z+1] >= 0.) & (surfaces[x+1,y,z] >= 0.) & (surfaces[x+1,y,z+1] >= 0.) & (surfaces[x+1,y+1,z] >= 0.) & (surfaces[x+1,y+1,z+1] >= 0.))


            alpha_v = [alphas[ver_xyzs[i,0], ver_xyzs[i,1], ver_xyzs[i,2]] for i in range(ver_xyzs.shape[0])]
            alpha_v = torch.concat(alpha_v, axis=-1).mean(dim=-1)

            return normals, valid_mask, empty_mask[:,0], torch.sigmoid(alpha_v.detach().clone())

        # find normals
        norm_xyzs = torch.tensor([[0,0,0], [0,0,1], [0,1,0], [1,0,0]], dtype=torch.long, device=device)
        norm000, mask000, empty000, alpha_v000 = find_normal(norm_xyzs[0])
        norm001, mask001, empty001, alpha_v001 = find_normal(norm_xyzs[1])
        norm010, mask010, empty010, alpha_v010 = find_normal(norm_xyzs[2])
        norm100, mask100, empty100, alpha_v100 = find_normal(norm_xyzs[3])

        Norm000 = norm000 / torch.clamp(torch.norm(norm000, dim=-1, keepdim=True), 1e-10)
        Norm001 = norm001 / torch.clamp(torch.norm(norm001, dim=-1, keepdim=True), 1e-10)
        Norm010 = norm010 / torch.clamp(torch.norm(norm010, dim=-1, keepdim=True), 1e-10)
        Norm100 = norm100 / torch.clamp(torch.norm(norm100, dim=-1, keepdim=True), 1e-10)

        norm_dz = torch.norm(Norm001 - Norm000, dim=-1)**2
        norm_dy = torch.norm(Norm010 - Norm000, dim=-1)**2
        norm_dx = torch.norm(Norm100 - Norm000, dim=-1)**2

        norm_count = torch.ones_like(norm_dz) * 3

        if connectivity_check:
            # check connectivity of surfaces
            face001 = torch.concat([surfaces[0,0,1], surfaces[0,1,1], surfaces[1,0,1], surfaces[1,1,1]], axis=-1)
            face010 = torch.concat([surfaces[0,1,0], surfaces[0,1,1], surfaces[1,1,0], surfaces[1,1,1]], axis=-1)
            face100 = torch.concat([surfaces[1,0,0], surfaces[1,0,1], surfaces[1,1,0], surfaces[1,1,1]], axis=-1)
            con001 = torch.count_nonzero(
                (self.level_set_data[None, :] >= face001.min(axis=-1, keepdim=True).values) & \
                (self.level_set_data[None, :] <= face001.max(axis=-1, keepdim=True).values),
                axis=-1
                ) > 0
            con010 = torch.count_nonzero(
                (self.level_set_data[None, :] >= face010.min(axis=-1, keepdim=True).values) & \
                (self.level_set_data[None, :] <= face010.max(axis=-1, keepdim=True).values),
                axis=-1
                ) > 0
            con100 = torch.count_nonzero(
                (self.level_set_data[None, :] >= face100.min(axis=-1, keepdim=True).values) & \
                (self.level_set_data[None, :] <= face100.max(axis=-1, keepdim=True).values),
                axis=-1
                ) > 0
        else:
            con001 = torch.ones_like(mask000).bool()
            con010 = torch.ones_like(mask000).bool()
            con100 = torch.ones_like(mask000).bool()

        if not ignore_empty:
            empty000 = torch.zeros_like(mask000).bool()
            empty001 = torch.zeros_like(mask000).bool()
            empty010 = torch.zeros_like(mask000).bool()
            empty100 = torch.zeros_like(mask000).bool()
            

        norm_count[(~mask001)|(~mask000)|(~con001) | (empty000 & empty001) ] -= 1
        norm_count[(~mask010)|(~mask000)|(~con010) | (empty000 & empty010) ] -= 1
        norm_count[(~mask100)|(~mask000)|(~con100) | (empty000 & empty100) ] -= 1

        # filter out gradients on non-exist voxel or non-connected surfaces
        norm_dz[(~mask001)|(~mask000)|(~con001) | (empty000 & empty001) ] = 0.
        norm_dy[(~mask010)|(~mask000)|(~con010) | (empty000 & empty010) ] = 0.
        norm_dx[(~mask100)|(~mask000)|(~con100) | (empty000 & empty100) ] = 0.



        if alpha_weighted_norm_loss:
            # use alpha value to re-weight the normal loss
            norm_dz = norm_dz * alpha_v000[:, None] * alpha_v001[:, None]
            norm_dy = norm_dy * alpha_v000[:, None] * alpha_v010[:, None]
            norm_dx = norm_dx * alpha_v100[:, None] * alpha_v100[:, None]

        normal_loss = torch.where(norm_count!=0, (norm_dx+norm_dy+norm_dz) / norm_count, torch.zeros_like(norm_count))

        normal_loss = scaling * torch.mean(normal_loss)

        surfaces.retain_grad()
        norm000.retain_grad()
        norm001.retain_grad()
        norm010.retain_grad()
        norm100.retain_grad()
        Norm000.retain_grad()
        Norm001.retain_grad()
        Norm010.retain_grad()
        Norm100.retain_grad()

        normal_loss.backward()

        return normal_loss


    def _surface_eikonal_loss_grad_check(self, rand_cells, scaling, device='cuda'):
        xyz = rand_cells
        z = (xyz % self.links.shape[2]).long()
        xy = xyz / self.links.shape[2]
        y = (xy % self.links.shape[1]).long()
        x = (xy / self.links.shape[1]).long()

        coords = torch.tensor([
            [0,0,0],
            [0,0,1],
            [0,1,0],
            [0,1,1],
            [1,0,0],
            [1,0,1],
            [1,1,0],
            [1,1,1],
        ], dtype=torch.long, device=device)

        links=torch.zeros([2,2,2,xyz.shape[0]], dtype=torch.long, device=device)
        alphas=torch.zeros([2,2,2,xyz.shape[0], 1], dtype=self.density_data.dtype, device=device)
        surfaces=torch.zeros([2,2,2,xyz.shape[0], 1], dtype=self.surface_data.dtype, device=device)

        for i in range(coords.shape[0]):
            def maybe_get_link(x,y,z):
                _links = torch.ones_like(x, dtype=torch.long) * -1
                invalid_xyz_mask = (torch.stack([x,y,z], axis=-1) >= torch.tensor(self.links.shape, device=device)).any(axis=-1)
                _links[~invalid_xyz_mask] = self.links[x[~invalid_xyz_mask], y[~invalid_xyz_mask], z[~invalid_xyz_mask]].long()

                return _links

            links[coords[i,0], coords[i,1], coords[i,2]] = maybe_get_link(x+coords[i,0], y+coords[i,1], z+coords[i,2])
            alphas[coords[i,0], coords[i,1], coords[i,2]], _ , \
                surfaces[coords[i,0], coords[i,1], coords[i,2]] = self._fetch_links(links[coords[i,0], coords[i,1], coords[i,2]])

        def find_normal(norm_xyz):
            x,y,z = norm_xyz.unbind(-1)

            dx = ((surfaces[x+1,y,z]+surfaces[x+1,y,z+1]+surfaces[x+1,y+1,z]+surfaces[x+1,y+1,z+1]) - \
                (surfaces[x,y,z]+surfaces[x,y,z+1]+surfaces[x,y+1,z]+surfaces[x,y+1,z+1])) /4
            dy = ((surfaces[x,y+1,z]+surfaces[x,y+1,z+1]+surfaces[x+1,y+1,z]+surfaces[x+1,y+1,z+1]) - \
                (surfaces[x,y,z]+surfaces[x,y,z+1]+surfaces[x+1,y,z]+surfaces[x+1,y,z+1]))/4
            dz = ((surfaces[x,y,z+1]+surfaces[x,y+1,z+1]+surfaces[x+1,y,z+1]+surfaces[x+1,y+1,z+1]) - \
                (surfaces[x,y,z]+surfaces[x,y+1,z]+surfaces[x+1,y,z]+surfaces[x+1,y+1,z]))/4

            normals = torch.stack([dx, dy, dz], dim=-1)
            # normals = normals / torch.clamp(torch.norm(normals, dim=-1, keepdim=True), 1e-10)

            # check if there is non-exist vertex
            coords = torch.tensor([
                [0,0,0],
                [0,0,1],
                [0,1,0],
                [0,1,1],
                [1,0,0],
                [1,0,1],
                [1,1,0],
                [1,1,1],
                ], dtype=torch.long, device=device)
            ver_xyzs = norm_xyz[None, :] + coords
            valid_mask = torch.ones(links.shape[-1], device=links.device).bool()
            for i in range(ver_xyzs.shape[0]):
                valid_mask = (valid_mask) & (links[ver_xyzs[i,0], ver_xyzs[i,1], ver_xyzs[i,2]] >= 0)

            alpha_v = [alphas[ver_xyzs[i,0], ver_xyzs[i,1], ver_xyzs[i,2]] for i in range(ver_xyzs.shape[0])]
            alpha_v = torch.concat(alpha_v, axis=-1).mean(dim=-1)

            return normals, valid_mask, torch.sigmoid(alpha_v.detach().clone())

        # find normals
        norm_xyzs = torch.tensor([[0,0,0], [0,0,1], [0,1,0], [1,0,0]], dtype=torch.long, device=device)
        norm000, mask000, alpha_v000 = find_normal(norm_xyzs[0])


        Norm000 = torch.clamp(torch.norm(norm000, dim=-1), 1e-10)
        Norm000[~mask000] = 1.


        eikonal_loss = scaling * torch.sum((1-Norm000)**2)


        eikonal_loss.backward()

        return eikonal_loss


    def _surface_viscosity_loss_grad_check(self, rand_cells, scaling, device='cuda', eta=1e-2):
        xyz = rand_cells
        z = (xyz % self.links.shape[2]).long()
        xy = xyz / self.links.shape[2]
        y = (xy % self.links.shape[1]).long()
        x = (xy / self.links.shape[1]).long()

        # filter out cells at the edge
        edge_mask = (x < 1) | (y < 1) | (z < 1) | \
                    (x >= self.links.shape[0] - 1) | (y >= self.links.shape[1] - 1) | (z >= self.links.shape[2] - 1)

        x, y, z = x[~edge_mask], y[~edge_mask], z[~edge_mask]            

        link000 = self.links[x,y,z]
        link_100 = self.links[x-1,y,z]
        link100 = self.links[x+1,y,z]
        link0_10 = self.links[x,y-1,z]
        link010 = self.links[x,y+1,z]
        link00_1 = self.links[x,y,z-1]
        link001 = self.links[x,y,z+1]

        # filter out cells that are near empty cell
        invalid_mask = (link000 < 0) | (link_100 < 0) | (link100 < 0) | \
                       (link0_10 < 0) | (link010 < 0) | (link00_1 < 0) | (link001 < 0)

        x, y, z = x[~invalid_mask], y[~invalid_mask], z[~invalid_mask]

        link000 = link000[~invalid_mask].long()
        link_100 = link_100[~invalid_mask].long()
        link100 = link100[~invalid_mask].long()
        link0_10 = link0_10[~invalid_mask].long()
        link010 = link010[~invalid_mask].long()
        link00_1 = link00_1[~invalid_mask].long()
        link001 = link001[~invalid_mask].long()


        surf000 = self.surface_data[link000]
        surf_100 = self.surface_data[link_100]
        surf100 = self.surface_data[link100]
        surf0_10 = self.surface_data[link0_10]
        surf010 = self.surface_data[link010]
        surf00_1 = self.surface_data[link00_1]
        surf001 = self.surface_data[link001]

        # note that we assume same aspect ratio for xyz when converting sdf from grid coord to world coord
        # this allows fake sample distance to be calculated easier
        gsz = self._grid_size().mean()
        h = 2.0 * self.radius.mean() / gsz
        h = h.to(device)

        norm_grad = torch.sqrt(
            ((surf100 - surf_100) / (2. * h)) ** 2. + \
            ((surf010 - surf0_10) / (2. * h)) ** 2. + \
            ((surf001 - surf00_1) / (2. * h)) ** 2.
        )

        lap = (surf100 + surf_100 - 2.*surf000) / (h ** 2.) + \
              (surf010 + surf0_10 - 2.*surf000) / (h ** 2.) + \
              (surf001 + surf00_1 - 2.*surf000) / (h ** 2.)


        # vis_loss = ((norm_grad - 1)) ** 2.
        vis_loss = ((norm_grad - 1) * torch.sign(surf000) - eta * lap) ** 2.
        vis_loss = scaling * torch.sum(vis_loss) / rand_cells.shape[0]

        vis_loss.backward()

        return vis_loss, norm_grad.mean()



    def _surface_sign_change_grad_check(self, rand_cells, scaling, device='cuda'):
        xyz = rand_cells
        z = (xyz % self.links.shape[2]).long()
        xy = xyz / self.links.shape[2]
        y = (xy % self.links.shape[1]).long()
        x = (xy / self.links.shape[1]).long()

        v0 = self.surface_data[self.links[x,y,z].long()]

        valid_count = torch.zeros_like(v0)
        L = torch.zeros_like(v0)

        invalid_mask = (torch.stack([x+1,y,z], axis=-1) >= torch.tensor(self.links.shape, device=device)).any(axis=-1)
        vx = v0.clone().detach()
        vx[~invalid_mask] = self.surface_data[self.links[x[~invalid_mask]+1, y[~invalid_mask], z[~invalid_mask]].long()]
        ss_mask = v0 * vx < 0.
        L[ss_mask] += torch.abs(v0[ss_mask] - vx[ss_mask]) * self.links.shape[0] / 256.
        valid_count[ss_mask] += 1.
        
        invalid_mask = (torch.stack([x,y+1,z], axis=-1) >= torch.tensor(self.links.shape, device=device)).any(axis=-1)
        vy = v0.clone().detach()
        vy[~invalid_mask] = self.surface_data[self.links[x[~invalid_mask], y[~invalid_mask]+1, z[~invalid_mask]].long()]
        ss_mask = v0 * vy < 0.
        L[ss_mask] += torch.abs(v0[ss_mask] - vy[ss_mask]) * self.links.shape[1] / 256.
        valid_count[ss_mask] += 1.

        invalid_mask = (torch.stack([x,y,z+1], axis=-1) >= torch.tensor(self.links.shape, device=device)).any(axis=-1)
        vz = v0.clone().detach()
        vz[~invalid_mask] = self.surface_data[self.links[x[~invalid_mask], y[~invalid_mask], z[~invalid_mask]+1].long()]
        ss_mask = v0 * vz < 0.
        L[ss_mask] += torch.abs(v0[ss_mask] - vz[ss_mask]) * self.links.shape[2] / 256.
        valid_count[ss_mask] += 1.

        
        mask = valid_count != 0.
        L[mask] = L[mask] / valid_count[mask]

        sign_change_loss = scaling * torch.mean(L)

        sign_change_loss.backward()

        return sign_change_loss




    def inplace_surface_sign_change_grad(self, grad: torch.Tensor,
                                scaling: float = 1.0,
                                sparse_frac: float = 0.01,
                                contiguous: bool = True,
                                use_kernel: bool = True,
                            ):
        '''
        Penalize surface sign change
        '''

        if self.surface_data is None:
            return
        
        if not use_kernel:
            # pytorch version
            rand_cells = self._get_rand_cells(sparse_frac, contiguous=contiguous)
            return self._surface_sign_change_grad_check(rand_cells, scaling / rand_cells.shape[0])

        else:
            assert (
                _C is not None and self.surface_data.is_cuda and grad.is_cuda
            ), "CUDA extension is currently required for total variation"

            rand_cells = self._get_rand_cells(sparse_frac, contiguous=contiguous)
            if rand_cells is not None:
                if rand_cells.size(0) > 0:
                    _C.surf_sign_change_grad_sparse(self.links, self.surface_data,
                            rand_cells,
                            self._get_sparse_grad_indexer(),
                            0, 1, scaling,
                            grad)
            else:
                raise NotImplementedError
                self.sparse_grad_indexer : Optional[torch.Tensor] = None


    def inplace_surface_normal_grad(self, grad: torch.Tensor,
                                scaling: float = 1.0,
                                eikonal_scale: float = 0.,
                                sparse_frac: float = 0.01,
                                ndc_coeffs: Tuple[float, float] = (-1.0, -1.0),
                                contiguous: bool = True,
                                use_kernel: bool = True,
                                connectivity_check: bool = True,
                                ignore_empty: bool = False,
                                **kwargs
                            ):
        if self.surface_data is None:
            return
        
        if not use_kernel:
            # pytorch version
            rand_cells = self._get_rand_cells(sparse_frac, contiguous=contiguous)
            return self._surface_normal_loss_grad_check(rand_cells, scaling, connectivity_check=connectivity_check, ignore_empty=ignore_empty, **kwargs)

        else:
            assert (
                _C is not None and self.surface_data.is_cuda and grad.is_cuda
            ), "CUDA extension is currently required for total variation"

            rand_cells = self._get_rand_cells(sparse_frac, contiguous=contiguous)
            if rand_cells is not None:
                if rand_cells.size(0) > 0:
                    _C.surface_normal_grad_sparse(self.links, self.surface_data,
                            rand_cells,
                            self._get_sparse_grad_indexer(),
                            self.level_set_data[0],
                            0, 1, scaling, eikonal_scale,
                            ndc_coeffs[0], ndc_coeffs[1],
                            connectivity_check,
                            ignore_empty,
                            grad)
            else:
                _C.surface_normal_grad(self.links, self.surface_data, self.level_set_data[0],
                        0, 1, scaling,
                        ndc_coeffs[0], ndc_coeffs[1],
                        grad)
                self.sparse_grad_indexer : Optional[torch.Tensor] = None

    def inplace_tv_color_grad(
        self,
        grad: torch.Tensor,
        start_dim: int = 0,
        end_dim: Optional[int] = None,
        scaling: float = 1.0,
        sparse_frac: float = 0.01,
        logalpha: bool=False,
        logalpha_delta: float=2.0,
        ndc_coeffs: Tuple[float, float] = (-1.0, -1.0),
        contiguous: bool = True
    ):
        """
        Add gradient of total variation for color
        directly into the gradient tensor, multiplied by 'scaling'

        :param start_dim: int, first color channel dimension to compute TV over (inclusive).
                          Default 0.
        :param end_dim: int, last color channel dimension to compute TV over (exclusive).
                          Default None = all dimensions until the end.
        """
        assert (
            _C is not None and self.sh_data.is_cuda and grad.is_cuda
        ), "CUDA extension is currently required for total variation"
        assert not logalpha, "No longer supported"
        if end_dim is None:
            end_dim = self.sh_data.size(1)
        end_dim = end_dim + self.sh_data.size(1) if end_dim < 0 else end_dim
        start_dim = start_dim + self.sh_data.size(1) if start_dim < 0 else start_dim

        rand_cells = self._get_rand_cells(sparse_frac, contiguous=contiguous)
        if rand_cells is not None:
            if rand_cells.size(0) > 0:
                indexer = self._get_sparse_sh_grad_indexer()
                #  with utils.Timing("actual_tv_color"):
                _C.tv_grad_sparse(self.links, self.sh_data,
                                  rand_cells,
                                  indexer,
                                  start_dim, end_dim, scaling,
                                  logalpha,
                                  logalpha_delta,
                                  True,
                                  False,
                                  ndc_coeffs[0], ndc_coeffs[1],
                                  grad)
        else:
            _C.tv_grad(self.links, self.sh_data, start_dim, end_dim, scaling,
                    logalpha,
                    logalpha_delta,
                    True,
                    ndc_coeffs[0], ndc_coeffs[1],
                    grad)
            self.sparse_sh_grad_indexer = None

    def inplace_tv_lumisphere_grad(
        self,
        grad: torch.Tensor,
        start_dim: int = 0,
        end_dim: Optional[int] = None,
        scaling: float = 1.0,
        sparse_frac: float = 0.01,
        logalpha: bool=False,
        logalpha_delta: float=2.0,
        ndc_coeffs: Tuple[float, float] = (-1.0, -1.0),
        dir_factor: float=1.0,
        dir_perturb_radians: float=0.05
    ):
        assert (
            _C is not None and self.sh_data.is_cuda and grad.is_cuda
        ), "CUDA extension is currently required for total variation"
        assert self.basis_type != BASIS_TYPE_MLP, "MLP not supported"
             #  SparseGridSpec& grid,
             #  torch::Tensor rand_cells,
             #  torch::Tensor sample_dirs,
             #  float scale,
             #  float ndc_coeffx,
             #  float ndc_coeffy,
             #  float dir_factor,
             #  GridOutputGrads& grads) {
        rand_cells = self._get_rand_cells(sparse_frac)
        grad_holder = _C.GridOutputGrads()

        indexer = self._get_sparse_sh_grad_indexer()
        assert indexer is not None
        grad_holder.mask_out = indexer
        grad_holder.grad_sh_out = grad

        batch_size = rand_cells.size(0)

        dirs = torch.randn(3, device=rand_cells.device)
        dirs /= torch.norm(dirs)

        if self.basis_type == BASIS_TYPE_3D_TEXTURE:
            sh_mult = self._eval_learned_bases(dirs[None])
        elif self.basis_type == BASIS_TYPE_MLP:
            sh_mult = torch.sigmoid(self._eval_basis_mlp(dirs[None]))
        else:
            sh_mult = utils.eval_sh_bases(self.basis_dim, dirs[None])
        sh_mult = sh_mult[0]

        if dir_factor > 0.0:
            axis = torch.randn((batch_size, 3))
            axis /= torch.norm(axis, dim=-1, keepdim=True)
            axis *= dir_perturb_radians
            R = Rotation.from_rotvec(axis.numpy()).as_matrix()
            R = torch.from_numpy(R).float().to(device=rand_cells.device)
            dirs_perturb = (R * dirs.unsqueeze(-2)).sum(-1)
        else:
            dirs_perturb = dirs # Dummy, since it won't be used

        if self.basis_type == BASIS_TYPE_3D_TEXTURE:
            sh_mult_u = self._eval_learned_bases(dirs_perturb[None])
        elif self.basis_type == BASIS_TYPE_MLP:
            sh_mult_u = torch.sigmoid(self._eval_basis_mlp(dirs_perturb[None]))
        else:
            sh_mult_u = utils.eval_sh_bases(self.basis_dim, dirs_perturb[None])
        sh_mult_u = sh_mult_u[0]

        _C.lumisphere_tv_grad_sparse(
                          self._to_cpp(),
                          rand_cells,
                          sh_mult,
                          sh_mult_u,
                          scaling,
                          ndc_coeffs[0], ndc_coeffs[1],
                          dir_factor,
                          grad_holder)


    def inplace_l2_color_grad(
        self,
        grad: torch.Tensor,
        start_dim: int = 0,
        end_dim: Optional[int] = None,
        scaling: float = 1.0,
    ):
        """
        Add gradient of L2 regularization for color
        directly into the gradient tensor, multiplied by 'scaling'
        (no CUDA extension used)

        :param start_dim: int, first color channel dimension to compute TV over (inclusive).
                          Default 0.
        :param end_dim: int, last color channel dimension to compute TV over (exclusive).
                          Default None = all dimensions until the end.
        """
        with torch.no_grad():
            if end_dim is None:
                end_dim = self.sh_data.size(1)
            end_dim = end_dim + self.sh_data.size(1) if end_dim < 0 else end_dim
            start_dim = start_dim + self.sh_data.size(1) if start_dim < 0 else start_dim

            if self.sparse_sh_grad_indexer is None:
                scale = scaling / self.sh_data.size(0)
                grad[:, start_dim:end_dim] += scale * self.sh_data[:, start_dim:end_dim]
            else:
                indexer = self._maybe_convert_sparse_grad_indexer(sh=True)
                nz : int = torch.count_nonzero(indexer).item() if indexer.dtype == torch.bool else \
                           indexer.size(0)
                scale = scaling / nz
                grad[indexer, start_dim:end_dim] += scale * self.sh_data[indexer, start_dim:end_dim]

    def inplace_tv_background_grad(
        self,
        grad: torch.Tensor,
        scaling: float = 1.0,
        scaling_density: Optional[float] = None,
        sparse_frac: float = 0.01,
        contiguous: bool = False
    ):
        """
        Add gradient of total variation for color
        directly into the gradient tensor, multiplied by 'scaling'
        """
        assert (
            _C is not None and self.sh_data.is_cuda and grad.is_cuda
        ), "CUDA extension is currently required for total variation"

        rand_cells_bg = self._get_rand_cells_background(sparse_frac, contiguous)
        indexer = self._get_sparse_background_grad_indexer()
        if scaling_density is None:
            scaling_density = scaling
        _C.msi_tv_grad_sparse(
                          self.background_links,
                          self.background_data,
                          rand_cells_bg,
                          indexer,
                          scaling,
                          scaling_density,
                          grad)

    def inplace_tv_basis_grad(
        self,
        grad: torch.Tensor,
        scaling: float = 1.0
    ):
        bd = self.basis_data
        tv_val = torch.mean(torch.sqrt(1e-5 +
                    (bd[:-1, :-1, 1:] - bd[:-1, :-1, :-1]) ** 2 +
                    (bd[:-1, 1:, :-1] - bd[:-1, :-1, :-1]) ** 2 +
                    (bd[1:, :-1, :-1] - bd[:-1, :-1, :-1]) ** 2).sum(dim=-1))
        tv_val_scaled = tv_val * scaling
        tv_val_scaled.backward()

    def optim_density_step(self, lr: float, beta: float=0.9, epsilon: float = 1e-8,
                             optim : str='rmsprop'):
        """
        Execute RMSprop or sgd step on density
        """
        assert (
            _C is not None and self.sh_data.is_cuda
        ), "CUDA extension is currently required for optimizers"

        indexer = self._maybe_convert_sparse_grad_indexer()
        if optim == 'rmsprop':
            if (
                self.density_rms is None
                or self.density_rms.shape != self.density_data.shape
            ):
                del self.density_rms
                self.density_rms = torch.zeros_like(self.density_data.data) # FIXME init?
            _C.rmsprop_step(
                self.density_data.data,
                self.density_rms,
                self.density_data.grad,
                indexer,
                beta,
                lr,
                epsilon,
                -1e9,
                lr
            )
        elif optim == 'sgd':
            _C.sgd_step(
                self.density_data.data,
                self.density_data.grad,
                indexer,
                lr,
                lr
            )
        else:
            raise NotImplementedError(f'Unsupported optimizer {optim}')

    def optim_surface_step(self, lr: float, beta: float=0.9, epsilon: float = 1e-8,
                             optim : str='rmsprop'):
        """
        Execute RMSprop or sgd step on density
        """
        if self.surface_type != SURFACE_TYPE_NONE and self.surface_type != SURFACE_TYPE_VOXEL_FACE:
            surface_data = self.surface_data
        else:
            # No surface data used!
            return

        assert (
            _C is not None and surface_data.is_cuda
        ), "CUDA extension is currently required for optimizers"


        indexer = self._maybe_convert_sparse_grad_indexer()
        if optim == 'rmsprop':
            if (
                self.surface_rms is None
                or self.surface_rms.shape != surface_data.shape
            ):
                del self.surface_rms
                self.surface_rms = torch.zeros_like(surface_data.data) # FIXME init?
            _C.rmsprop_step(
                surface_data.data,
                self.surface_rms,
                surface_data.grad,
                indexer,
                beta,
                lr,
                epsilon,
                -1e9,
                lr
            )
        elif optim == 'sgd':
            _C.sgd_step(
                surface_data.data,
                surface_data.grad,
                indexer,
                lr,
                lr
            )
        else:
            raise NotImplementedError(f'Unsupported optimizer {optim}')


    def optim_fake_sample_std_step(self, lr: float, beta: float=0.9, epsilon: float = 1e-8,
                             optim : str='rmsprop', lambda_l2:float = 0., lambda_l1:float = 0.):
        """
        Execute RMSprop or sgd step on fake sample std
        Note that this function also adds L1/L2 reg to std gradient
        """

        if self.fake_sample_std is None:
            return


        assert (
            _C is not None and self.fake_sample_std.is_cuda
        ), "CUDA extension is currently required for optimizers"

        # add L1 and L2 loss on std
        # l2 = lambda_l2 * (self.fake_sample_std) ** 2
        self.fake_sample_std.grad += 2 * lambda_l2 * self.fake_sample_std + lambda_l1 * torch.sign(self.fake_sample_std)


        indexer = torch.empty((), device=self.density_data.device)
        if optim == 'rmsprop':
            if (
                self.fake_sample_std_rms is None
                or self.fake_sample_std_rms.shape != self.fake_sample_std.shape
            ):
                del self.fake_sample_std_rms
                self.fake_sample_std_rms = torch.zeros_like(self.fake_sample_std.data) # FIXME init?
            _C.rmsprop_step(
                self.fake_sample_std.data,
                self.surface_rms,
                self.fake_sample_std.grad,
                indexer,
                beta,
                lr,
                epsilon,
                -1e9,
                lr
            )
        elif optim == 'sgd':
            _C.sgd_step(
                self.fake_sample_std.data,
                self.fake_sample_std.grad,
                indexer,
                lr,
                lr
            )
        else:
            raise NotImplementedError(f'Unsupported optimizer {optim}')



    def optim_sh_step(self, lr: float, beta: float=0.9, epsilon: float = 1e-8,
                      optim: str = 'rmsprop'):
        """
        Execute RMSprop/SGD step on SH
        """
        assert (
            _C is not None and self.sh_data.is_cuda
        ), "CUDA extension is currently required for optimizers"

        indexer = self._maybe_convert_sparse_grad_indexer(sh=True)
        if optim == 'rmsprop':
            if self.sh_rms is None or self.sh_rms.shape != self.sh_data.shape:
                del self.sh_rms
                self.sh_rms = torch.zeros_like(self.sh_data.data) # FIXME init?
            _C.rmsprop_step(
                self.sh_data.data,
                self.sh_rms,
                self.sh_data.grad,
                indexer,
                beta,
                lr,
                epsilon,
                -1e9,
                lr
            )
        elif optim == 'sgd':
            _C.sgd_step(
                self.sh_data.data, self.sh_data.grad, indexer, lr, lr
            )
        else:
            raise NotImplementedError(f'Unsupported optimizer {optim}')

    def optim_background_step(self,
                              lr_sigma: float,
                              lr_color: float,
                              beta: float=0.9, epsilon: float = 1e-8,
                              optim : str='rmsprop'):
        """
        Execute RMSprop or sgd step on density
        """
        assert (
            _C is not None and self.sh_data.is_cuda
        ), "CUDA extension is currently required for optimizers"

        indexer = self._maybe_convert_sparse_grad_indexer(bg=True)
        n_chnl = self.background_data.size(-1)
        if optim == 'rmsprop':
            if (
                self.background_rms is None
                or self.background_rms.shape != self.background_data.shape
            ):
                del self.background_rms
                self.background_rms = torch.zeros_like(self.background_data.data) # FIXME init?
            _C.rmsprop_step(
                self.background_data.data.view(-1, n_chnl),
                self.background_rms.view(-1, n_chnl),
                self.background_data.grad.view(-1, n_chnl),
                indexer,
                beta,
                lr_color,
                epsilon,
                -1e9,
                lr_sigma
            )
        elif optim == 'sgd':
            _C.sgd_step(
                self.background_data.data.view(-1, n_chnl),
                self.background_data.grad.view(-1, n_chnl),
                indexer,
                lr_color,
                lr_sigma
            )
        else:
            raise NotImplementedError(f'Unsupported optimizer {optim}')

    def optim_basis_step(self, lr: float, beta: float=0.9, epsilon: float = 1e-8,
                         optim: str = 'rmsprop'):
        """
        Execute RMSprop/SGD step on SH
        """
        assert (
            _C is not None and self.sh_data.is_cuda
        ), "CUDA extension is currently required for optimizers"

        if optim == 'rmsprop':
            if self.basis_rms is None or self.basis_rms.shape != self.basis_data.shape:
                del self.basis_rms
                self.basis_rms = torch.zeros_like(self.basis_data.data)
            self.basis_rms.mul_(beta).addcmul_(self.basis_data.grad, self.basis_data.grad, value = 1.0 - beta)
            denom = self.basis_rms.sqrt().add_(epsilon)
            self.basis_data.data.addcdiv_(self.basis_data.grad, denom, value=-lr)
        elif optim == 'sgd':
            self.basis_data.grad.mul_(lr)
            self.basis_data.data -= self.basis_data.grad
        else:
            raise NotImplementedError(f'Unsupported optimizer {optim}')
        self.basis_data.grad.zero_()

    @property
    def basis_type_name(self):
        if self.basis_type == BASIS_TYPE_SH:
            return "SH"
        elif self.basis_type == BASIS_TYPE_3D_TEXTURE:
            return "3D_TEXTURE"
        elif self.basis_type == BASIS_TYPE_MLP:
            return "MLP"
        return "UNKNOWN"

    def __repr__(self):
        return (
            f"svox2.SparseGrid(basis_type={self.basis_type_name}, "
            + f"basis_dim={self.basis_dim}, "
            + f"reso={list(self.links.shape)}, "
            + f"capacity:{self.sh_data.size(0)})"
        )

    def is_cubic_pow2(self):
        """
        Check if the current grid is cubic (same in all dims) with power-of-2 size.
        This allows for conversion to svox 1 and Z-order curve (Morton code)
        """
        reso = self.links.shape
        return reso[0] == reso[1] and reso[0] == reso[2] and utils.is_pow2(reso[0])

    def _to_cpp(self, grid_coords: bool = False, replace_basis_data: Optional[torch.Tensor] = None):
        """
        Generate object to pass to C++
        """
        gspec = _C.SparseGridSpec()
        gspec.density_data = self.density_data
        gspec.surface_type = self.surface_type
        if self.surface_type != SURFACE_TYPE_NONE:
            gspec.surface_data = self.surface_data
            gspec.level_set_data = self.level_set_data
        gspec.sh_data = self.sh_data
        gspec.links = self.links
        if grid_coords:
            gspec._offset = torch.zeros_like(self._offset)
            gspec._scaling = torch.ones_like(self._offset)
        else:
            gsz = self._grid_size()
            # gspec._offset = self._offset * gsz - 0.5
            gspec._offset = self._offset * gsz
            gspec._scaling = self._scaling * gsz

        gspec.basis_dim = self.basis_dim
        gspec.basis_type = self.basis_type
        if replace_basis_data:
            gspec.basis_data = replace_basis_data
        elif self.basis_type == BASIS_TYPE_3D_TEXTURE:
            gspec.basis_data = self.basis_data

        if self.use_background:
            gspec.background_links = self.background_links
            gspec.background_data = self.background_data

        if self.fake_sample_std is not None:
            if torch.is_tensor(self.fake_sample_std):
                gspec.fake_sample_std = self.fake_sample_std.item()
            else:
                gspec.fake_sample_std = self.fake_sample_std
        return gspec

    def _grid_size(self):
        return torch.tensor(self.links.shape, device="cpu", dtype=torch.float32)

    def _get_data_grads(self):
        ret = {}
        for subitem in self.trainable_data:
            param = self.__getattr__(subitem)
            if not param.requires_grad:
                ret[subitem] = torch.zeros_like(param.data)
            else:
                if (
                    not hasattr(param, "grad")
                    or param.grad is None
                    or param.grad.shape != param.data.shape
                ): # if grad attribute is none or invalid, assign new one to it
                    if hasattr(param, "grad"):
                        del param.grad
                    param.grad = torch.zeros_like(param.data)
                ret[subitem] = param.grad
        return ret

    def _get_sparse_grad_indexer(self):
        indexer = self.sparse_grad_indexer
        if indexer is None:
            indexer = torch.empty((0,), dtype=torch.bool, device=self.density_data.device)
        return indexer

    def _get_sparse_sh_grad_indexer(self):
        indexer = self.sparse_sh_grad_indexer
        if indexer is None:
            indexer = torch.empty((0,), dtype=torch.bool, device=self.density_data.device)
        return indexer

    def _get_sparse_background_grad_indexer(self):
        indexer = self.sparse_background_indexer
        if indexer is None:
            indexer = torch.empty((0, 0, 0, 0), dtype=torch.bool,
                            device=self.density_data.device)
        return indexer

    def _maybe_convert_sparse_grad_indexer(self, sh=False, bg=False):
        """
        Automatically convert sparse grad indexer from mask to
        indices, if it is efficient
        """
        indexer = self.sparse_sh_grad_indexer if sh else self.sparse_grad_indexer
        if bg:
            indexer = self.sparse_background_indexer
            if indexer is not None:
                indexer = indexer.view(-1)
        if indexer is None:
            return torch.empty((), device=self.density_data.device)
        if (
            indexer.dtype == torch.bool and
            torch.count_nonzero(indexer).item()
            < indexer.size(0) // 8
        ):
            # Highly sparse (use index)
            indexer = torch.nonzero(indexer.flatten(), as_tuple=False).flatten()
        return indexer

    def _get_rand_cells(self, sparse_frac: float, force: bool = False, contiguous:bool=True):
        if sparse_frac < 1.0 or force:
            assert self.sparse_grad_indexer is None or self.sparse_grad_indexer.dtype == torch.bool, \
                   "please call sparse loss after rendering and before gradient updates"
            grid_size = self.links.size(0) * self.links.size(1) * self.links.size(2)
            sparse_num = max(int(sparse_frac * grid_size), 1)
            if contiguous:
                start = np.random.randint(0, grid_size)
                arr = torch.arange(start, start + sparse_num, dtype=torch.int32, device=
                                                self.links.device)

                if start > grid_size - sparse_num:
                    arr[grid_size - sparse_num - start:] -= grid_size
                return arr
            else:
                return torch.randint(0, grid_size, (sparse_num,), dtype=torch.int32, device=
                                                self.links.device)
        return None

    def _get_rand_cells_non_empty(self, sparse_frac:float, frac_of_remaining:bool=True, contiguous:bool=True):
        '''
        frac_of_remaining: frac is on remain cells
        '''
        grid_size = self.links.size(0) * self.links.size(1) * self.links.size(2)
        non_empty_ids = torch.where(self.links.view(-1) >= 0)[0].int()
        non_empty_num = len(non_empty_ids)
        if frac_of_remaining:
            sparse_num = int(non_empty_num * sparse_frac)
        else:
            sparse_num = int(sparse_frac * grid_size)
            if sparse_num > non_empty_num:
                sparse_num = int(non_empty_num)
        if contiguous:
            start = np.random.randint(0, non_empty_num - sparse_num)
            arr = non_empty_ids[torch.arange(start, start + sparse_num, dtype=torch.long, device=
                                                    self.links.device)]
            return arr
        else:
            return non_empty_ids[torch.randint(0, non_empty_num, (sparse_num,), dtype=torch.long, device=
                                                self.links.device)]
        

    def _get_rand_cells_background(self, sparse_frac: float, contiguous:bool=True):
        assert self.use_background, "Can only use sparse background loss if using background"
        assert self.sparse_background_indexer is None or self.sparse_background_indexer.dtype == torch.bool, \
               "please call sparse loss after rendering and before gradient updates"
        grid_size = self.background_links.size(0) \
                    * self.background_links.size(1) \
                    * self.background_data.size(1)
        sparse_num = max(int(sparse_frac * grid_size), 1)
        if contiguous:
            start = np.random.randint(0, grid_size)# - sparse_num + 1)
            arr = torch.arange(start, start + sparse_num, dtype=torch.int32, device=
                                            self.links.device)
            if start > grid_size - sparse_num:
                arr[grid_size - sparse_num - start:] -= grid_size
            return arr
        else:
            return torch.randint(0, grid_size, (sparse_num,), dtype=torch.int32, device=
                                            self.links.device)

    def _eval_learned_bases(self, dirs: torch.Tensor):
        basis_data = self.basis_data.permute([3, 2, 1, 0])[None]
        samples = F.grid_sample(basis_data, dirs[None, None, None], mode='bilinear', padding_mode='zeros', align_corners=True)
        samples = samples[0, :, 0, 0, :].permute([1, 0])
        #  dc = torch.full_like(samples[:, :1], fill_value=0.28209479177387814)
        #  samples = torch.cat([dc, samples], dim=-1)
        return samples

    def _eval_basis_mlp(self, dirs: torch.Tensor):
        if self.mlp_posenc_size > 0:
            dirs_enc = utils.posenc(
                dirs,
                None,
                0,
                self.mlp_posenc_size,
                include_identity=True,
                enable_ipe=False,
            )
        else:
            dirs_enc = dirs
        return self.basis_mlp(dirs_enc)

    def reinit_learned_bases(self,
            init_type: str = 'sh',
            sg_lambda_max: float = 1.0,
            upper_hemi: bool = False):
        """
        Initialize learned basis using either SH orrandom spherical Gaussians
        with concentration parameter sg_lambda (max magnitude) and
        normalization constant sg_sigma

        Spherical Gaussians formula for reference:
        :math:`Output = \sigma_{i}{exp ^ {\lambda_i * (\dot(\mu_i, \dirs) - 1)}`

        :param upper_hemi: bool, (SG only) whether to only place Gaussians in z <= 0 (note directions are flipped)
        """

        init_type = init_type.lower()
        n_comps = self.basis_data.size(-1)

        basis_reso = self.basis_data.size(0)
        ax = torch.linspace(-1.0, 1.0, basis_reso, dtype=torch.float32)
        X, Y, Z = torch.meshgrid(ax, ax, ax)
        points = torch.stack((X, Y, Z), dim=-1).view(-1, 3)
        points /= points.norm(dim=-1).unsqueeze(-1)

        if init_type == 'sh':
            assert utils.isqrt(n_comps) is not None, \
                   "n_comps (learned basis SH init) must be a square number; maybe try SG init"
            sph_vals = utils.eval_sh_bases(n_comps, points)
        elif init_type == 'sg':
            # Low-disparity direction sampling
            u1 = torch.arange(0, n_comps) + torch.rand((n_comps,))
            u1 /= n_comps
            u1 = u1[torch.randperm(n_comps)]
            u2 = torch.arange(0, n_comps) + torch.rand((n_comps,))
            u2 /= n_comps
            sg_dirvecs = utils.spher2cart(u1 * np.pi, u2 * np.pi * 2)
            if upper_hemi:
                sg_dirvecs[..., 2] = -torch.abs(sg_dirvecs[..., 2])

            # Concentration parameters (0 = DC -> infty = point)
            sg_lambdas = torch.rand_like(sg_dirvecs[:, 0]) * sg_lambda_max
            sg_lambdas[0] = 0.0  # Assure DC

            # L2-Normalization
            sg_sigmas : np.ndarray = np.sqrt(sg_lambdas / (np.pi * (1.0 - np.exp(-4 * sg_lambdas))))
            sg_sigmas[sg_lambdas == 0.0] = 1.0 / np.sqrt(4 * np.pi)
            # L1-Normalization
            #  sg_sigmas : np.ndarray = sg_lambdas / (2 * np.pi * (1.0 - np.exp(-2 * sg_lambdas)))
            #  sg_sigmas[sg_lambdas == 0.0] = 1.0 / (2 *  (1.0 - 1.0 / np.exp(1)) * np.pi)
            sph_vals = utils.eval_sg_at_dirs(sg_lambdas, sg_dirvecs, points) * sg_sigmas
        elif init_type == 'fourier':
            # Low-disparity direction sampling
            u1 = torch.arange(0, n_comps) + torch.rand((n_comps,))
            u1 /= n_comps
            u1 = u1[torch.randperm(n_comps)]
            u2 = torch.arange(0, n_comps) + torch.rand((n_comps,))
            u2 /= n_comps
            fourier_dirvecs = utils.spher2cart(u1 * np.pi, u2 * np.pi * 2)
            fourier_freqs = torch.linspace(0.0, 1.0, n_comps + 1)[:-1]
            fourier_freqs += torch.rand_like(fourier_freqs) * (fourier_freqs[1] - fourier_freqs[0])
            fourier_freqs = torch.exp(fourier_freqs)
            fourier_freqs = fourier_freqs[torch.randperm(n_comps)]
            fourier_scale = 1.0 / torch.sqrt(2 * np.pi - torch.cos(fourier_freqs) * torch.sin(fourier_freqs) / fourier_freqs)
            four_phases = torch.rand_like(fourier_freqs) * np.pi * 2

            dots = (points[:, None] * fourier_dirvecs[None]).sum(-1)
            dots *= fourier_freqs
            sins = torch.sin(dots + four_phases)
            sph_vals = sins * fourier_scale

        else:
            raise NotImplementedError("Unsupported initialization", init_type)
        self.basis_data.data[:] = sph_vals.view(
                    basis_reso, basis_reso, basis_reso, n_comps).to(device=self.basis_data.device)
