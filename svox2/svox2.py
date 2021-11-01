from .utils import (
    isqrt,
    eval_sh_bases,
    gen_morton,
    is_pow2,
    spher2cart,
    eval_sg_at_dirs,
    MAX_SH_BASIS,
    _get_c_extension,
)
import torch
from torch import nn, autograd
import torch.nn.functional as F
from typing import Union, List, NamedTuple, Optional, Tuple
from dataclasses import dataclass
from warnings import warn
from functools import reduce
from tqdm import tqdm
import numpy as np

_C = _get_c_extension()


@dataclass
class RenderOptions:
    """
    Rendering options, see comments
    available:
    :param backend: str, one of lerp, cuvol, nn
    :param background_brightness: float
    :param step_size: float, step size for backend lerp or cuvol only
                      (for estimating the integral; in nearest-neighbor case the integral is exactly computed)
    :param sigma_thresh: float
    :param stop_thresh: float
    """

    backend: str = "cuvol"  # One of lerp, cuvol, nn
    # nn is nearest neighbor (very fast)
    # cuvol is basic lerp version from cuvol
    #   (fast for small batch when sparse)
    # lerp is coalesced lerp (fast for larger batch)

    background_brightness: float = 1.0  # [0, 1], the background color black-white

    step_size: float = (
        0.5  # Step size, in normalized voxels; only used if backend = lerp or cuvol
    )
    #  (i.e. 1 = 1 voxel width, different from svox where 1 = grid width!)

    sigma_thresh: float = 1e-10  # Voxels with sigmas < this are ignored, in [0, 1]
    #  make this higher for fast rendering

    stop_thresh: float = (
        1e-7  # Stops rendering if the remaining light intensity/termination, in [0, 1]
    )
    #  probability is <= this much (forward only)
    #  make this higher for fast rendering

    last_sample_opaque: bool = False   # Make the last sample opaque (for forward-facing)

    def _to_cpp(self, randomize: bool = False):
        """
        Generate object to pass to C++
        """
        opt = _C.RenderOptions()
        opt.background_brightness = self.background_brightness
        opt.step_size = self.step_size
        opt.sigma_thresh = self.sigma_thresh
        opt.stop_thresh = self.stop_thresh
        opt.last_sample_opaque = self.last_sample_opaque
        #  opt.randomize = randomize
        #  UINT32_MAX = 2**32-1
        #  opt._m1 = np.random.randint(0, UINT32_MAX)
        #  opt._m2 = np.random.randint(0, UINT32_MAX)
        #  opt._m3 = np.random.randint(0, UINT32_MAX)
        # Note that the backend option is handled in Python
        return opt


@dataclass
class Camera:
    c2w: torch.Tensor  # OpenCV
    fx: float
    fy: float
    cx: float
    cy: float
    width: int
    height: int

    ndc_coeffs: Union[Tuple[float, float], List[float]] = (-1.0, -1.0)

    @property
    def using_ndc(self):
        return self.ndc_coeffs[0] > 0.0

    def _to_cpp(self):
        """
        Generate object to pass to C++
        """
        spec = _C.CameraSpec()
        spec.c2w = self.c2w
        spec.fx = self.fx
        spec.fy = self.fy
        spec.cx = self.cx
        spec.cy = self.cy
        spec.width = self.width
        spec.height = self.height
        spec.ndc_coeffx = self.ndc_coeffs[0]
        spec.ndc_coeffy = self.ndc_coeffs[1]
        return spec

    @property
    def is_cuda(self):
        return self.c2w.is_cuda


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

    @property
    def is_cuda(self):
        return self.origins.is_cuda and self.dirs.is_cuda


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
        return color

    @staticmethod
    def backward(ctx, grad_out):
        (color_cache,) = ctx.saved_tensors
        cu_fn = _C.__dict__[f"volume_render_{ctx.backend}_backward"]
        grad_density_grid = torch.zeros_like(ctx.grid.density_data.data)
        grad_sh_grid = torch.zeros_like(ctx.grid.sh_data.data)
        grad_basis = torch.zeros_like(ctx.grid.basis_data.data)
        # TODO save the sparse mask
        sparse_mask = cu_fn(
            ctx.grid,
            ctx.rays,
            ctx.opt,
            grad_out.contiguous(),
            color_cache,
            grad_density_grid,
            grad_sh_grid,
            grad_basis,
        )
        ctx.grid = ctx.rays = ctx.opt = None
        if not ctx.needs_input_grad[0]:
            grad_density_grid = None
        if not ctx.needs_input_grad[1]:
            grad_sh_grid = None
        if not ctx.needs_input_grad[2]:
            grad_basis = None

        return grad_density_grid, grad_sh_grid, grad_basis, None, None, None, None


class _VolumeRenderImageFunction(autograd.Function):
    @staticmethod
    def forward(
        ctx,
        data_density: torch.Tensor,
        data_sh: torch.Tensor,
        data_basis: torch.Tensor,
        grid,
        cam,
        opt,
        backend: str,
    ):
        cu_fn = _C.__dict__[f"volume_render_{backend}_image"]
        color = cu_fn(grid, cam, opt)
        ctx.save_for_backward(color)
        ctx.grid = grid
        ctx.cam = cam
        ctx.opt = opt
        ctx.backend = backend
        return color

    @staticmethod
    def backward(ctx, grad_out):
        (color_cache,) = ctx.saved_tensors
        cu_fn = _C.__dict__[f"volume_render_{ctx.backend}_image_backward"]
        grad_density_grid = torch.zeros_like(ctx.grid.density_data.data)
        grad_sh_grid = torch.zeros_like(ctx.grid.sh_data.data)
        grad_basis = torch.zeros_like(ctx.grid.basis_data.data)
        # TODO save the sparse mask
        sparse_mask = cu_fn(
            ctx.grid,
            ctx.cam,
            ctx.opt,
            grad_out.contiguous(),
            color_cache,
            grad_density_grid,
            grad_sh_grid,
            grad_basis,
        )
        ctx.grid = ctx.cam = ctx.opt = None
        if not ctx.needs_input_grad[0]:
            grad_density_grid = None
        if not ctx.needs_input_grad[1]:
            grad_sh_grid = None
        if not ctx.needs_input_grad[2]:
            grad_basis = None
        return grad_density_grid, grad_sh_grid, grad_basis, None, None, None, None


class _TotalVariationFunction(autograd.Function):
    @staticmethod
    def forward(
        ctx, data: torch.Tensor, links: torch.Tensor, start_dim: int, end_dim: int,
        use_logalpha: bool, logalpha_delta: float,
        ignore_edge: bool
    ):
        tv = _C.tv(links, data, start_dim, end_dim, use_logalpha, logalpha_delta, ignore_edge)
        ctx.save_for_backward(links, data)
        ctx.start_dim = start_dim
        ctx.end_dim = end_dim
        ctx.use_logalpha = use_logalpha
        ctx.logalpha_delta = logalpha_delta
        ctx.ignore_edge = ignore_edge
        return tv

    @staticmethod
    def backward(ctx, grad_out):
        links, data = ctx.saved_tensors
        grad_grid = torch.zeros_like(data)
        _C.tv_grad(links, data, ctx.start_dim, ctx.end_dim, 1.0,
                   ctx.use_logalpha, ctx.logalpha_delta,
                   ctx.ignore_edge, grad_grid)
        ctx.start_dim = ctx.end_dim = None
        if not ctx.needs_input_grad[0]:
            grad_grid = None
        return grad_grid, None, None, None


class _SparsityLossFunction(autograd.Function):
    @staticmethod
    def forward(
        ctx, data: torch.Tensor, links: torch.Tensor, delta: float = 0.05
    ):
        tv = _C.sparsity(links, data, delta)
        ctx.save_for_backward(links, data)
        ctx.delta = delta
        return tv

    @staticmethod
    def backward(ctx, grad_out):
        links, data = ctx.saved_tensors
        grad_grid = torch.zeros_like(data)
        _C.sparsity_grad(links, data, 1.0, ctx.delta, grad_grid)
        ctx.start_dim = ctx.end_dim = None
        if not ctx.needs_input_grad[0]:
            grad_grid = None
        return grad_grid, None, None, None


# END Differentiable CUDA functions with custom gradient


class SparseGrid(nn.Module):
    """
    Main sparse grid data structure.
    initially it will be a dense grid of resolution <reso>.
    Only float32 is supported.

    :param reso: int or List[int, int, int], resolution for resampled grid, as in the constructor
    :param radius: float or List[float, float, float], the 1/2 side length of the grid, optionally in each direction
    :param center: float or List[float, float, float], the center of the grid
    :param use_learned_basis: bool, whether to use learned spherical function (false = SH)
    :param basis_dim: int, size of basis / number of SH components
                           (must be square number in case of SH)
    :param basis_reso: int, resolution of learned spherical function
    :param use_z_order: bool, if true, stores the data initially in a Z-order curve if possible
    :param device: torch.device, device to store the grid
    """

    def __init__(
        self,
        reso: Union[int, List[int]] = 128,
        radius: Union[float, List[float]] = 1.0,
        center: Union[float, List[float]] = [0.0, 0.0, 0.0],
        use_learned_basis: bool = True,
        basis_dim: int = 9,  # SH/learned basis size; in SH case, square number
        basis_reso: int = 16,  # Learned basis resolution (x^3 embedding grid)
        use_z_order : bool=False,
        use_sphere_bound : bool=False,
        device: Union[torch.device, str] = "cpu",
    ):
        super().__init__()
        self.use_learned_basis = use_learned_basis
        if not use_learned_basis:
            assert isqrt(basis_dim) is not None, "basis_dim (SH) must be a square number"
        assert (
            basis_dim >= 1 and basis_dim <= MAX_SH_BASIS
        ), f"basis_dim 1-{MAX_SH_BASIS} supported"
        self.basis_dim = basis_dim

        if isinstance(reso, int):
            reso = [reso] * 3
        else:
            assert (
                len(reso) == 3
            ), "reso must be an integer or indexable object of 3 ints"

        if use_z_order and not (reso[0] == reso[1] and reso[0] == reso[2] and is_pow2(reso[0])):
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
            init_links = gen_morton(reso[0], device=device, dtype=torch.int32).flatten()
        else:
            init_links = torch.arange(n3, device=device, dtype=torch.int32)

        if use_sphere_bound:
            X = torch.arange(reso[0], dtype=torch.float32, device=device) - 0.5
            Y = torch.arange(reso[1], dtype=torch.float32, device=device) - 0.5
            Z = torch.arange(reso[2], dtype=torch.float32, device=device) - 0.5
            X, Y, Z = torch.meshgrid(X, Y, Z)
            points = torch.stack((X, Y, Z), dim=-1).view(-1, 3)
            gsz = torch.tensor(reso)
            roffset = 1.0 / gsz - 1.0
            rscaling = 2.0 / gsz
            print(points.dtype, roffset.dtype, rscaling.dtype)
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
        # representation
        self.sh_data = nn.Parameter(
            torch.zeros(
                self.capacity, self.basis_dim * 3, dtype=torch.float32, device=device
            )
        )

        if use_learned_basis:
            # Unit sphere embedded in a cube
            self.basis_data = nn.Parameter(
                torch.zeros(
                    basis_reso, basis_reso, basis_reso,
                    self.basis_dim, dtype=torch.float32, device=device
                )
            )
        else:
            self.basis_data = nn.Parameter(
                torch.zeros(
                    (0, 0, 0, 0), dtype=torch.float32, device=device
                ),
                requires_grad=False
            )

        self.register_buffer("links", init_links.view(reso))
        self.links: torch.Tensor
        self.opt = RenderOptions()
        self.sparse_grad_indexer: Optional[torch.Tensor] = None
        self.sparse_sh_grad_indexer: Optional[torch.Tensor] = None
        self.density_rms: Optional[torch.Tensor] = None
        self.sh_rms: Optional[torch.Tensor] = None
        self.basis_rms: Optional[torch.Tensor] = None

        if self.links.is_cuda and use_sphere_bound:
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
        Return the resolution of the learned spherical function data,
        or 0 if only using SH
        """
        return self.basis_data.size(0) if self.use_learned_basis else 0

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
        results_sigma[mask] = self.density_data[idxs]
        results_sh[mask] = self.sh_data[idxs]
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
                self.density_data, self.sh_data, self._to_cpp(grid_coords), points, want_colors
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

    def forward(self, points: torch.Tensor, use_kernel: bool = True):
        return self.sample(points, use_kernel=use_kernel)

    def _volume_render_gradcheck_lerp(self, rays: Rays):
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

        if self.use_learned_basis:
            sh_mult = self._eval_learned_bases(viewdirs)
        else:
            sh_mult = eval_sh_bases(self.basis_dim, viewdirs)
        invdirs = 1.0 / dirs
        invdirs[dirs == 0] = 1e9

        gsz = self._grid_size()
        t1 = (-0.5 - origins) * invdirs
        t2 = (gsz.to(device=dirs.device) - 0.5 - origins) * invdirs
        t = torch.max(torch.min(t1, t2), dim=-1).values.clamp_min_(0.0)
        tmax = torch.min(torch.max(t1, t2), dim=-1).values

        log_light_intensity = torch.zeros(B, device=origins.device)
        out_rgb = torch.zeros((B, 3), device=origins.device)
        good_indices = torch.arange(B, device=origins.device)

        mask = t < tmax
        good_indices = good_indices[mask]
        origins = origins[mask]
        dirs = dirs[mask]
        invdirs = invdirs[mask]
        t = t[mask]
        sh_mult = sh_mult[mask]
        tmax = tmax[mask]

        grid_cen_x = (self.links.size(0) - 1) * 0.5;
        grid_cen_y = (self.links.size(1) - 1) * 0.5;
        while good_indices.numel() > 0:
            pos = origins + t[:, None] * dirs

            l = pos.to(torch.long)
            l.clamp_min_(0)
            l[:, 0].clamp_max_(gsz[0] - 2)
            l[:, 1].clamp_max_(gsz[1] - 2)
            l[:, 2].clamp_max_(gsz[2] - 2)
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
            )
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

            mask = t < tmax
            good_indices = good_indices[mask]
            origins = origins[mask]
            dirs = dirs[mask]
            invdirs = invdirs[mask]
            t = t[mask]
            sh_mult = sh_mult[mask]
            tmax = tmax[mask]
        out_rgb += (
            torch.exp(log_light_intensity).unsqueeze(-1)
            * self.opt.background_brightness
        )
        return out_rgb

    def volume_render(
        self, rays: Rays, use_kernel: bool = True, randomize: bool = False
    ):
        """
        Standard volume rendering. See grid.opt.* (RenderOptions) for configs.

        :param rays: Rays, (origins (N, 3), dirs (N, 3))
        :param use_kernel: bool, if false uses pure PyTorch version even if on CUDA.
        :param randomize: bool, whether to use randomness (now ignored)
        :return: (N, 3), predicted RGB
        """
        assert self.opt.backend in ["cuvol"]  # , 'lerp', 'nn']
        if use_kernel and self.links.is_cuda and _C is not None:
            assert rays.is_cuda
            return _VolumeRenderFunction.apply(
                self.density_data,
                self.sh_data,
                self.basis_data,
                self._to_cpp(),
                rays._to_cpp(),
                self.opt._to_cpp(randomize=randomize),
                self.opt.backend,
            )
        else:
            warn("Using slow volume rendering, should only be used for debugging")
            return self._volume_render_gradcheck_lerp(rays)

    def volume_render_fused(
        self, rays: Rays, rgb_gt: torch.Tensor, randomize: bool = False
    ):
        """
        Standard volume rendering with fused MSE gradient generation,
            given a ground truth color for each pixel.
        Parameter gradients will be updated (no need for backward()).

        See grid.opt.* (RenderOptions) for configs.
        Requires CUDA. Always uses cuvol backend.

        :param rays: Rays, (origins (N, 3), dirs (N, 3))
        :param rgb_gt: (N, 3), GT pixel colors, each channel in [0, 1]
        :param randomize: bool, whether to use randomness (now ignored)
        :return: (N, 3), predicted RGB
        """
        assert (
            _C is not None and self.sh_data.is_cuda
        ), "CUDA extension is currently required for fused"
        assert rays.is_cuda
        grad_density, grad_sh, grad_basis = self._get_data_grads()
        rgb_out = torch.zeros_like(rgb_gt)
        self.sparse_grad_indexer: torch.Tensor = _C.volume_render_cuvol_fused(
            self._to_cpp(),
            rays._to_cpp(),
            self.opt._to_cpp(),
            rgb_gt,
            rgb_out,
            grad_density,
            grad_sh,
            grad_basis,
        )
        self.sparse_sh_grad_indexer = self.sparse_grad_indexer.clone()
        return rgb_out

    def volume_render_image(
        self, camera: Camera, use_kernel: bool = True, randomize: bool = False
    ):
        """
        Standard volume rendering (entire image version).
        See grid.opt.* (RenderOptions) for configs.

        :param camera: Camera
        :param use_kernel: bool, if false uses pure PyTorch version even if on CUDA.
        :param randomize: bool, whether to use randomness (now ignored)
        :return: (H, W, 3), predicted RGB image
        """
        assert self.opt.backend in ["cuvol"]  # , 'lerp', 'nn']
        if use_kernel and self.links.is_cuda and _C is not None:
            assert camera.is_cuda
            return _VolumeRenderImageFunction.apply(
                self.density_data,
                self.sh_data,
                self.basis_data,
                self._to_cpp(),
                camera._to_cpp(),
                self.opt._to_cpp(randomize=randomize),
                self.opt.backend,
            )
        else:
            raise NotImplementedError(
                "Pure PyTorch image rendering not implemented, " + "please use rays"
            )

    def resample(
        self,
        reso: Union[int, List[int]],
        sigma_thresh: float = 5.0,
        weight_thresh: float = 0.01,
        dilate: int = 2,
        cameras: Optional[List[Camera]] = None,
        use_z_order=False,
        accelerate=True,
        weight_render_stop_thresh: float = 0.05
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
        """
        with torch.no_grad():
            device = self.links.device
            if isinstance(reso, int):
                reso = [reso] * 3
            else:
                assert (
                    len(reso) == 3
                ), "reso must be an integer or indexable object of 3 ints"

            if use_z_order and not (reso[0] == reso[1] and reso[0] == reso[2] and is_pow2(reso[0])):
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
                morton = gen_morton(reso[0], dtype=torch.long).view(-1)
                points[morton] = points.clone()
            points = points.to(device=device)

            use_weight_thresh = cameras is not None

            batch_size = 720720
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
                for cam in tqdm(cameras):
                    _C.grid_weight_render(
                        sample_vals_density, cam._to_cpp(),
                        0.5,
                        weight_render_stop_thresh,
                        self.opt.last_sample_opaque,
                        offset, scaling, max_wt_grid
                    )
                sample_vals_mask = max_wt_grid > weight_thresh
                del max_wt_grid
            else:
                sample_vals_mask = sample_vals_density > sigma_thresh

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

    def resize(self, basis_dim: int):
        """
        Modify the size of the data stored in the voxels. Called expand/shrink in svox 1.

        :param basis_dim: new basis dimension, must be square number
        """
        assert isqrt(basis_dim) is not None, "basis_dim (SH) must be a square number"
        assert (
            basis_dim >= 1 and basis_dim <= MAX_SH_BASIS
        ), f"basis_dim 1-{MAX_SH_BASIS} supported"
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
        offset = self._offset * gsz - 0.5
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
        roffset = self.radius * (1.0 / gsz - 1.0) + self.center
        rscaling = 2.0 * self.radius / gsz
        return torch.addcmul(
            roffset.to(device=points.device), points, rscaling.to(device=points.device)
        )

    def save(self, path: str, compress: bool = False):
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
        }
        if self.use_learned_basis:
            data['basis_data'] = self.basis_data.data.cpu().numpy()
        save_fn(
            path,
            **data
        )

    @classmethod
    def load(cls, path: str, device: Union[torch.device, str] = "cpu"):
        """
        Load from path
        """
        z = np.load(path)
        if "data" in z.keys():
            # Compatibility
            all_data = z.f.data
            sh_data = all_data[..., 1:]
            density_data = all_data[..., :1]
        else:
            sh_data = z.f.sh_data
            density_data = z.f.density_data
        links = z.f.links
        basis_dim = (sh_data.shape[1]) // 3
        radius = z.f.radius.tolist() if "radius" in z.files else [1.0, 1.0, 1.0]
        center = z.f.center.tolist() if "center" in z.files else [0.0, 0.0, 0.0]
        grid = cls(
            1,
            radius=radius,
            center=center,
            basis_dim=basis_dim,
            use_z_order=False,
            device="cpu",
            use_learned_basis=False
        )
        if sh_data.dtype != np.float32:
            sh_data = sh_data.astype(np.float32)
        if density_data.dtype != np.float32:
            density_data = density_data.astype(np.float32)
        sh_data = torch.from_numpy(sh_data).to(device=device)
        density_data = torch.from_numpy(density_data).to(device=device)
        grid.sh_data = nn.Parameter(sh_data)
        grid.density_data = nn.Parameter(density_data)
        grid.links = torch.from_numpy(links).to(device=device)
        grid.capacity = grid.sh_data.size(0)

        # Maybe load basis_data
        if "basis_data" in z.keys():
            basis_data = torch.from_numpy(z.f.basis_data).to(device=device)
            grid.use_learned_basis = True
            grid.basis_data = nn.Parameter(basis_data)
        else:
            grid.basis_data = nn.Parameter(grid.basis_data.data.to(device=device))
        if grid.links.is_cuda:
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
        return t

    def tv(self, logalpha: bool=True, logalpha_delta: float=2.0):
        """
        Compute total variation over sigma,
        similar to Neural Volumes [Lombardi et al., ToG 2019]

        :return: torch.Tensor, size scalar, the TV value (sum over channels,
                 mean over voxels)
        """
        assert (
            _C is not None and self.sh_data.is_cuda
        ), "CUDA extension is currently required for total variation"
        return _TotalVariationFunction.apply(
                self.density_data, self.links, 0, 1, logalpha, logalpha_delta)

    def sparsity(self, delta: float=0.01):
        """
        Compute sparsity loss over sigma,

        :return: torch.Tensor, size scalar, the sparsity loss value
        """
        assert (
            _C is not None and self.sh_data.is_cuda
        ), "CUDA extension is currently required for total variation"
        return _SparsityLossFunction.apply(
                self.density_data, self.links, delta)

    def tv_color(self,
                 start_dim: int = 0, end_dim: Optional[int] = None,
                 logalpha: bool=False, logalpha_delta: float=2.0):
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
        if end_dim is None:
            end_dim = self.sh_data.size(1)
        end_dim = end_dim + self.sh_data.size(1) if end_dim < 0 else end_dim
        start_dim = start_dim + self.sh_data.size(1) if start_dim < 0 else start_dim
        return _TotalVariationFunction.apply(
            self.sh_data, self.links, start_dim, end_dim, logalpha, logalpha_delta
        )

    def tv_basis(self):
        bd = self.basis_data
        return torch.mean(torch.sqrt(1e-5 +
                    (bd[:-1, :-1, 1:] - bd[:-1, :-1, :-1]) ** 2 +
                    (bd[:-1, 1:, :-1] - bd[:-1, :-1, :-1]) ** 2 +
                    (bd[1:, :-1, :-1] - bd[:-1, :-1, :-1]) ** 2).sum(dim=-1))

    def inplace_tv_grad(self, grad: torch.Tensor,
                        scaling: float = 1.0,
                        sparse_frac: float = 1.0,
                        logalpha: bool=True, logalpha_delta: float=2.0
                    ):
        """
        Add gradient of total variation for sigma as in Neural Volumes
        [Lombardi et al., ToG 2019]
        directly into the gradient tensor, multiplied by 'scaling'
        """
        assert (
            _C is not None and self.density_data.is_cuda and grad.is_cuda
        ), "CUDA extension is currently required for total variation"

        rand_cells = self._get_rand_cells(sparse_frac)
        if rand_cells is not None:
            _C.tv_grad_sparse(self.links, self.density_data,
                    rand_cells,
                    self._get_sparse_grad_indexer(),
                    0, 1, scaling,
                    logalpha, logalpha_delta,
                    False,
                    grad)
        else:
            _C.tv_grad(self.links, self.density_data, 0, 1, scaling,
                    logalpha, logalpha_delta,
                    False,
                    grad)
            self.sparse_grad_indexer : Optional[torch.Tensor] = None

    def inplace_tv_color_grad(
        self,
        grad: torch.Tensor,
        start_dim: int = 0,
        end_dim: Optional[int] = None,
        scaling: float = 1.0,
        sparse_frac: float = 1.0,
        logalpha: bool=False,
        logalpha_delta: float=2.0
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
        if end_dim is None:
            end_dim = self.sh_data.size(1)
        end_dim = end_dim + self.sh_data.size(1) if end_dim < 0 else end_dim
        start_dim = start_dim + self.sh_data.size(1) if start_dim < 0 else start_dim

        rand_cells = self._get_rand_cells(sparse_frac)
        if rand_cells is not None:
            _C.tv_grad_sparse(self.links, self.sh_data,
                              rand_cells,
                              self._get_sparse_sh_grad_indexer(),
                              start_dim, end_dim, scaling,
                              logalpha,
                              logalpha_delta,
                              True,
                              grad)
        else:
            _C.tv_grad(self.links, self.sh_data, start_dim, end_dim, scaling,
                    logalpha,
                    logalpha_delta,
                    True,
                    grad)
            self.sparse_sh_grad_indexer = None

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

        self._maybe_convert_sparse_grad_indexer()
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
                self._get_sparse_grad_indexer(),
                beta,
                lr,
                epsilon,
            )
        elif optim == 'sgd':
            _C.sgd_step(
                self.density_data.data,
                self.density_data.grad,
                self._get_sparse_grad_indexer(),
                lr,
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

        self._maybe_convert_sparse_sh_grad_indexer()
        if optim == 'rmsprop':
            if self.sh_rms is None or self.sh_rms.shape != self.sh_data.shape:
                del self.sh_rms
                self.sh_rms = torch.zeros_like(self.sh_data.data) # FIXME init?
            _C.rmsprop_step(
                self.sh_data.data,
                self.sh_rms,
                self.sh_data.grad,
                self._get_sparse_sh_grad_indexer(),
                beta,
                lr,
                epsilon,
            )
        elif optim == 'sgd':
            _C.sgd_step(
                self.sh_data.data, self.sh_data.grad, self._get_sparse_sh_grad_indexer(), lr
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

    def __repr__(self):
        return (
            f"svox2.SparseGrid(basis_dim={self.basis_dim}, "
            + f"reso={list(self.links.shape)}, "
            + f"capacity:{self.sh_data.size(0)})"
        )

    def is_cubic_pow2(self):
        """
        Check if the current grid is cubic (same in all dims) with power-of-2 size.
        This allows for conversion to svox 1 and Z-order curve (Morton code)
        """
        reso = self.links.shape
        return reso[0] == reso[1] and reso[0] == reso[2] and is_pow2(reso[0])

    def _to_cpp(self, grid_coords: bool = False):
        """
        Generate object to pass to C++
        """
        gspec = _C.SparseGridSpec()
        gspec.density_data = self.density_data
        gspec.sh_data = self.sh_data
        gspec.links = self.links
        if grid_coords:
            gspec._offset = torch.zeros_like(self._offset)
            gspec._scaling = torch.ones_like(self._offset)
        else:
            gsz = self._grid_size()
            gspec._offset = self._offset * gsz - 0.5
            gspec._scaling = self._scaling * gsz

        gspec.basis_dim = self.basis_dim
        gspec.use_learned_basis = self.use_learned_basis
        gspec.basis_data = self.basis_data
        return gspec

    def _grid_size(self):
        return torch.tensor(self.links.shape, device="cpu", dtype=torch.float32)

    def _get_data_grads(self):
        ret = []
        for subitem in ["density_data", "sh_data", "basis_data"]:
            param = self.__getattr__(subitem)
            if not param.requires_grad:
                ret.append(torch.zeros_like(param.data))
            else:
                if (
                    not hasattr(param, "grad")
                    or param.grad is None
                    or param.grad.shape != param.data.shape
                ):
                    if hasattr(param, "grad"):
                        del param.grad
                    param.grad = torch.zeros_like(param.data)
                ret.append(param.grad)
        return ret

    def _get_sparse_grad_indexer(self):
        indexer = self.sparse_grad_indexer
        if indexer is None:
            indexer = torch.empty((), device=self.density_data.device)
        return indexer

    def _get_sparse_sh_grad_indexer(self):
        indexer = self.sparse_sh_grad_indexer
        if indexer is None:
            indexer = torch.empty((), device=self.density_data.device)
        return indexer

    def _maybe_convert_sparse_grad_indexer(self):
        """
        Automatically convert sparse grad indexer from mask to
        indices, if it is efficient
        """
        if (
            self.sparse_grad_indexer is not None and
            self.sparse_grad_indexer.dtype == torch.bool and
            torch.count_nonzero(self.sparse_grad_indexer).item()
            < self.sparse_grad_indexer.size(0) // 8
        ):
            # Highly sparse (use index)
            self.sparse_grad_indexer = torch.nonzero(
                self.sparse_grad_indexer.flatten(), as_tuple=False
            ).flatten()

    def _maybe_convert_sparse_sh_grad_indexer(self):
        """
        Automatically convert sparse grad indexer from mask to
        indices, if it is efficient
        """
        if (
            self.sparse_sh_grad_indexer is not None and
            self.sparse_sh_grad_indexer.dtype == torch.bool and
            torch.count_nonzero(self.sparse_sh_grad_indexer).item()
            < self.sparse_sh_grad_indexer.size(0) // 8
        ):
            # Highly sparse (use index)
            self.sparse_sh_grad_indexer = torch.nonzero(
                self.sparse_sh_grad_indexer.flatten(), as_tuple=False
            ).flatten()

    def _get_rand_cells(self, sparse_frac: float):
        if sparse_frac < 1.0:
            assert self.sparse_grad_indexer is None or self.sparse_grad_indexer.dtype == torch.bool, \
                   "please call sparse loss after rendering and before gradient updates"
            grid_size = (self.links.size(0) - 1) * (self.links.size(1) - 1) * (self.links.size(2) - 1)
            sparse_num = max(int(sparse_frac * grid_size), 1)
            return torch.randint(0, grid_size, (sparse_num,), dtype=torch.int32, device=
                                            self.links.device)
        return None

    def _eval_learned_bases(self, dirs: torch.Tensor):
        basis_data = self.basis_data.permute([3, 2, 1, 0])[None]
        samples = F.grid_sample(basis_data, dirs[None, None, None], mode='bilinear', padding_mode='zeros', align_corners=True)
        samples = samples[0, :, 0, 0, :].permute([1, 0])
        #  dc = torch.ones_like(samples[:, :1])
        #  samples = torch.cat([dc, samples], dim=-1)
        return samples

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
            assert isqrt(n_comps) is not None, \
                   "n_comps (learned basis SH init) must be a square number; maybe try SG init"
            sph_vals = eval_sh_bases(n_comps, points)
        elif init_type == 'sg':
            # Low-disparity direction sampling
            u1 = torch.arange(0, n_comps) + torch.rand((n_comps,))
            u1 /= n_comps
            u1 = u1[torch.randperm(n_comps)]
            u2 = torch.arange(0, n_comps) + torch.rand((n_comps,))
            u2 /= n_comps
            sg_dirvecs = spher2cart(u1 * np.pi, u2 * np.pi * 2)
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
            sph_vals = eval_sg_at_dirs(sg_lambdas, sg_dirvecs, points) * sg_sigmas
        elif init_type == 'fourier':
            # Low-disparity direction sampling
            u1 = torch.arange(0, n_comps) + torch.rand((n_comps,))
            u1 /= n_comps
            u1 = u1[torch.randperm(n_comps)]
            u2 = torch.arange(0, n_comps) + torch.rand((n_comps,))
            u2 /= n_comps
            fourier_dirvecs = spher2cart(u1 * np.pi, u2 * np.pi * 2)
            fourier_freqs = torch.linspace(0.0, 2.0, n_comps + 1)[:-1]
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
