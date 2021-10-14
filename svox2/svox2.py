from .utils import isqrt, eval_sh_bases, gen_morton, is_pow2, MAX_SH_BASIS, _get_c_extension
import torch
from torch import nn, autograd
import torch.nn.functional as F
from typing import Union, List, NamedTuple, Optional
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
    :param step_epsilon: float
    :param step_size: float, step size for backend lerp or cuvol only
                      (for estimating the integral; in nearest-neighbor case the integral is exactly computed)
    :param sigma_thresh: float
    :param stop_thresh: float
    """
    backend : str = 'cuvol'           # One of lerp, cuvol, nn
                                      # nn is nearest neighbor (very fast)
                                      # cuvol is basic lerp version from cuvol
                                      #   (fast for small batch when sparse)
                                      # lerp is coalesced lerp (fast for larger batch)

    background_brightness : float = 1.0   # [0, 1], the background color black-white

    step_epsilon : float = 1e-3           # Epsilon added to voxel steps, in normalized voxels
                                          #  (i.e. 1 = 1 voxel width, different from svox where 1 = grid width!)
                                          #  (needed in current traversal method for safety,
                                          #   set it to something like 1e-1 for fast rendering.
                                          #   Probably do not set it below 1e-4, increase if getting stuck)

    step_size : float = 0.5               # Step size, in normalized voxels; only used if backend = lerp or cuvol
                                          #  (i.e. 1 = 1 voxel width, different from svox where 1 = grid width!)

    sigma_thresh : float = 1e-10          # Voxels with sigmas < this are ignored, in [0, 1]
                                          #  make this higher for fast rendering

    stop_thresh : float = 1e-7            # Stops rendering if the remaining light intensity/termination, in [0, 1]
                                          #  probability is <= this much (forward only)
                                          #  make this higher for fast rendering

    def _to_cpp(self, randomize: bool = False):
        """
        Generate object to pass to C++
        """
        opt = _C.RenderOptions()
        opt.background_brightness = self.background_brightness
        opt.step_epsilon = self.step_epsilon
        opt.step_size = self.step_size
        opt.sigma_thresh = self.sigma_thresh
        opt.stop_thresh = self.stop_thresh
        opt.randomize = randomize
        UINT32_MAX = 2**32-1
        opt._m1 = np.random.randint(0, UINT32_MAX)
        opt._m2 = np.random.randint(0, UINT32_MAX)
        opt._m3 = np.random.randint(0, UINT32_MAX)
        # Note that the backend option is handled in Python
        return opt

@dataclass
class Camera:
    c2w : torch.Tensor # OpenCV
    fx : float
    fy : float
    width : int
    height : int

    def _to_cpp(self):
        """
        Generate object to pass to C++
        """
        spec = _C.CameraSpec()
        spec.c2w = self.c2w
        spec.fx = self.fx
        spec.fy = self.fy
        spec.width = self.width
        spec.height = self.height
        return spec

    @property
    def is_cuda(self):
        return self.c2w.is_cuda


@dataclass
class Rays:
    origins : torch.Tensor
    dirs : torch.Tensor

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
    def forward(ctx, data : torch.Tensor, grid, points : torch.Tensor):
        assert not points.requires_grad, "Point gradient not supported"
        out = _C.sample_grid(grid, points)
        ctx.save_for_backward(points)
        ctx.grid = grid
        return out

    @staticmethod
    def backward(ctx, grad_out):
        points, = ctx.saved_tensors
        grad_grid = _C.sample_grid_backward(
            ctx.grid, points, grad_out.contiguous()
        )
        if not ctx.needs_input_grad[0]:
            grad_grid = None
        return grad_grid, None, None


class _VolumeRenderFunction(autograd.Function):
    @staticmethod
    def forward(ctx,
                data : torch.Tensor,
                grid,
                rays,
                opt,
                backend : str):
        cu_fn = _C.__dict__[f'volume_render_{backend}']
        color = cu_fn(grid, rays, opt)
        ctx.save_for_backward(color)
        ctx.grid = grid
        ctx.rays = rays
        ctx.opt = opt
        ctx.backend = backend
        return color

    @staticmethod
    def backward(ctx, grad_out):
        color_cache, = ctx.saved_tensors
        cu_fn = _C.__dict__[f'volume_render_{ctx.backend}_backward']
        grad_grid = cu_fn(
            ctx.grid, ctx.rays, ctx.opt,
            grad_out.contiguous(),
            color_cache,
        )
        ctx.grid = ctx.rays = ctx.opt = None
        if not ctx.needs_input_grad[0]:
            grad_grid = None
        return grad_grid, None, None, None, None


class _VolumeRenderImageFunction(autograd.Function):
    @staticmethod
    def forward(ctx,
                data : torch.Tensor,
                grid,
                cam,
                opt,
                backend : str):
        cu_fn = _C.__dict__[f'volume_render_{backend}_image']
        color = cu_fn(grid, cam, opt)
        ctx.save_for_backward(color)
        ctx.grid = grid
        ctx.cam = cam
        ctx.opt = opt
        ctx.backend = backend
        return color

    @staticmethod
    def backward(ctx, grad_out):
        color_cache, = ctx.saved_tensors
        cu_fn = _C.__dict__[f'volume_render_{ctx.backend}_image_backward']
        grad_grid = cu_fn(
            ctx.grid, ctx.cam, ctx.opt,
            grad_out.contiguous(),
            color_cache,
        )
        ctx.grid = ctx.cam = ctx.opt = None
        if not ctx.needs_input_grad[0]:
            grad_grid = None
        return grad_grid, None, None, None, None

class _TotalVariationFunction(autograd.Function):
    @staticmethod
    def forward(ctx,
                data : torch.Tensor,
                links : torch.Tensor,
                start_dim : int,
                end_dim : int):
        tv = _C.tv(links, data, start_dim, end_dim)
        ctx.save_for_backward(links, data)
        ctx.start_dim = start_dim
        ctx.end_dim = end_dim
        return tv

    @staticmethod
    def backward(ctx, grad_out):
        links, data = ctx.saved_tensors
        grad_grid = torch.zeros_like(data)
        _C.tv_grad(links, data, ctx.start_dim, ctx.end_dim, 1.0, grad_grid)
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
    :param use_z_order: bool, if true, stores the data initially in a Z-order curve if possible
    :param device: torch.device, device to store the grid
    """
    def __init__(self,
            reso : Union[int, List[int]]=128,
            radius : Union[float, List[float]]=1.0,
            center : Union[float, List[float]]=[0.0, 0.0, 0.0],
            basis_dim : int = 9, # SH size; square number
            use_z_order=False,
            device : Union[torch.device, str]="cpu"):
        super().__init__()

        assert isqrt(basis_dim) is not None, "basis_dim (SH) must be a square number"
        assert basis_dim >= 1 and basis_dim <= MAX_SH_BASIS, \
                f"basis_dim 1-{MAX_SH_BASIS} supported"
        self.basis_dim = basis_dim

        if isinstance(reso, int):
            reso = [reso] * 3
        else:
            assert len(reso) == 3, \
                   "reso must be an integer or indexable object of 3 ints"
        self.capacity : int = reduce(lambda x,y: x*y, reso)

        if not (reso[0] == reso[1] and reso[0] == reso[2] and is_pow2(reso[0])):
            print("Morton code requires a cube grid of power-of-2 size, ignoring...")
            use_z_order = False
        if use_z_order:
            init_links = gen_morton(reso[0], device=device, dtype=torch.int32)
        else:
            init_links = torch.arange(self.capacity, device=device, dtype=torch.int32)

        self.register_buffer("links", init_links.view(reso))
        self.links : torch.Tensor
        self.data = nn.Parameter(torch.zeros(
            self.capacity, self.basis_dim * 3 + 1, dtype=torch.float32, device=device))

        if isinstance(radius, float) or isinstance(radius, int):
            radius = [radius] * 3
        if isinstance(radius, torch.Tensor):
            radius = radius.to(device='cpu', dtype=torch.float32)
        else:
            radius = torch.tensor(radius, dtype=torch.float32, device='cpu')
        if isinstance(center, torch.Tensor):
            center = center.to(device='cpu', dtype=torch.float32)
        else:
            center = torch.tensor(center, dtype=torch.float32, device='cpu')

        self.radius : torch.Tensor = radius  # CPU
        self.center : torch.Tensor = center  # CPU
        self._offset = 0.5 * (1.0 - self.center / self.radius)
        self._scaling = 0.5 / self.radius

        self.opt = RenderOptions()


    @property
    def data_dim(self):
        """
        Get the number of channels in the data (data.size(1))
        """
        return self.data.size(1)

    @property
    def shape(self):
        return list(self.links.shape) + [self.data.size(1)]

    def _fetch_links(self, links):
        result = torch.empty((links.size(0), self.data.size(1)), device=links.device, dtype=torch.float32)
        mask = links >= 0
        result[mask] = self.data[links[mask].long()]
        result[~mask] = 0.0
        return result

    def sample(self, points : torch.Tensor, use_kernel : bool = True, grid_coords = False):
        """
        Grid sampling with trilinear interpolation.
        Behaves like torch.nn.functional.grid_sample
        with padding mode border and align_corners=False (better for multi-resolution).

        Any voxel with link < 0 (empty) is considered to have 0 values in all channels
        prior to interpolating.

        :param points: torch.Tensor, (N, 3)
        :param use_kernel: bool, if false uses pure PyTorch version even if on CUDA.
        """
        if use_kernel and self.links.is_cuda and _C is not None:
            assert points.is_cuda
            return _SampleGridAutogradFunction.apply(self.data,
                    self._to_cpp(grid_coords), points)
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

            rgba000 = self._fetch_links(links000)
            rgba001 = self._fetch_links(links001)
            rgba010 = self._fetch_links(links010)
            rgba011 = self._fetch_links(links011)
            rgba100 = self._fetch_links(links100)
            rgba101 = self._fetch_links(links101)
            rgba110 = self._fetch_links(links110)
            rgba111 = self._fetch_links(links111)

            c00 = rgba000 * wa[:, 2:] + rgba001 * wb[:, 2:]
            c01 = rgba010 * wa[:, 2:] + rgba011 * wb[:, 2:]
            c10 = rgba100 * wa[:, 2:] + rgba101 * wb[:, 2:]
            c11 = rgba110 * wa[:, 2:] + rgba111 * wb[:, 2:]

            c0 = c00 * wa[:, 1:2] + c01 * wb[:, 1:2]
            c1 = c10 * wa[:, 1:2] + c11 * wb[:, 1:2]

            samples = c0 * wa[:, :1] + c1 * wb[:, :1]
            return samples

    def forward(self, points : torch.Tensor, use_kernel : bool = True):
        return self.sample(points, use_kernel=use_kernel)

    def _volume_render_gradcheck_lerp(self, rays : Rays):
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

        sh_mult = eval_sh_bases(self.basis_dim, viewdirs)
        invdirs = 1.0 / dirs
        invdirs[dirs == 0] = 1e9

        gsz = self._grid_size()
        t1 = (- origins) * invdirs
        t2 = (gsz.to(device=dirs.device) - 1.0 - origins) * invdirs
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

            rgba000 = self._fetch_links(links000)
            rgba001 = self._fetch_links(links001)
            rgba010 = self._fetch_links(links010)
            rgba011 = self._fetch_links(links011)
            rgba100 = self._fetch_links(links100)
            rgba101 = self._fetch_links(links101)
            rgba110 = self._fetch_links(links110)
            rgba111 = self._fetch_links(links111)

            wa, wb = 1.0 - pos, pos
            c00 = rgba000 * wa[:, 2:] + rgba001 * wb[:, 2:]
            c01 = rgba010 * wa[:, 2:] + rgba011 * wb[:, 2:]
            c10 = rgba100 * wa[:, 2:] + rgba101 * wb[:, 2:]
            c11 = rgba110 * wa[:, 2:] + rgba111 * wb[:, 2:]

            c0 = c00 * wa[:, 1:2] + c01 * wb[:, 1:2]
            c1 = c10 * wa[:, 1:2] + c11 * wb[:, 1:2]

            rgba = c0 * wa[:, :1] + c1 * wb[:, :1]
            # END CRAZY TRILERP

            log_att = - self.opt.step_size * torch.relu(rgba[..., 0]) * delta_scale[good_indices]
            weight = torch.exp(log_light_intensity[good_indices]) * (
                        1.0 - torch.exp(log_att))
            rgb = rgba[:, 1:]
            # [B', 3, n_sh_coeffs]
            rgb_sh = rgb.reshape(-1, 3, self.basis_dim)
            rgb = torch.sigmoid(torch.sum(sh_mult.unsqueeze(-2) * rgb_sh, dim=-1))   # [B', 3]
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
        out_rgb += torch.exp(log_light_intensity).unsqueeze(-1) * \
                   self.opt.background_brightness
        return out_rgb

    def volume_render(self, rays : Rays,
                      use_kernel : bool = True,
                      randomize : bool = False):
        """
        Standard volume rendering. See grid.opt.* (RenderOptions) for configs.

        :param rays: Rays, (origins (N, 3), dirs (N, 3))
        :param use_kernel: bool, if false uses pure PyTorch version even if on CUDA.
        :return: (N, 3) RGB
        """
        assert self.opt.backend in ['cuvol']#, 'lerp', 'nn']
        if use_kernel and self.links.is_cuda and _C is not None:
            assert rays.is_cuda
            return _VolumeRenderFunction.apply(self.data, self._to_cpp(),
                                               rays._to_cpp(),
                                               self.opt._to_cpp(randomize=randomize),
                                               self.opt.backend)
        else:
            warn("Using slow volume rendering, should only be used for debugging")
            return self._volume_render_gradcheck_lerp(rays)

    def volume_render_image(self, camera : Camera,
                            use_kernel : bool = True,
                            randomize : bool = False):
        """
        Standard volume rendering (entire image version).
        See grid.opt.* (RenderOptions) for configs.

        :param camera: Camera, (origins (N, 3), dirs (N, 3))
        :param use_kernel: bool, if false uses pure PyTorch version even if on CUDA.
        :return: (N, 3) RGB
        """
        assert self.opt.backend in ['cuvol']#, 'lerp', 'nn']
        if use_kernel and self.links.is_cuda and _C is not None:
            assert camera.is_cuda
            return _VolumeRenderImageFunction.apply(self.data, self._to_cpp(),
                                               camera._to_cpp(),
                                               self.opt._to_cpp(randomize=randomize),
                                               self.opt.backend)
        else:
            raise NotImplementedError("Pure PyTorch image rendering not implemented, " +
                    "please use rays")

    def resample(self,
                 reso : Union[int, List[int]],
                 sigma_thresh : float = 5.0,
                 weight_thresh : float = 0.01,
                 dilate : int = 2,
                 cameras : Optional[List[Camera]] = None,
                 ):
        """
        Resample and sparsify the grid; used to increase the resolution
        :param reso: int or List[int, int, int], resolution for resampled grid, as in the constructor
        :param sigma_thresh: float, threshold to apply on the sigma (if using sigma thresh i.e. cameras NOT given)
        :param weight_thresh: float, threshold to apply on the weights (if using weight thresh i.e. cameras given)
        :param dilate: int, if true applies dilation of size <dilate> to the 3D mask for nodes to keep in the grid
                             (keep neighbors in all 28 directions, including diagonals, of the desired nodes)
        :param cameras: Optional[List[Camera]], optional list of cameras in OpenCV convention (if given, uses weight thresholding)
        """
        device = self.links.device
        if isinstance(reso, int):
            reso = [reso] * 3
        else:
            assert len(reso) == 3, \
                   "reso must be an integer or indexable object of 3 ints"
        self.capacity : int = reduce(lambda x,y: x*y, reso)
        curr_reso = self.links.shape
        dtype = torch.float32
        reso_facts = [0.5 * curr_reso[i] / reso[i] for i in range(3)]
        X = torch.linspace(reso_facts[0] - 0.5, curr_reso[0] - reso_facts[0] - 0.5, reso[0], dtype=dtype)
        Y = torch.linspace(reso_facts[1] - 0.5, curr_reso[1] - reso_facts[1] - 0.5, reso[1], dtype=dtype)
        Z = torch.linspace(reso_facts[2] - 0.5, curr_reso[2] - reso_facts[2] - 0.5, reso[2], dtype=dtype)
        X, Y, Z = torch.meshgrid(X, Y, Z)
        points = torch.stack((X, Y, Z), dim=-1).view(-1, 3)

        use_weight_thresh = cameras is not None
        pre_mask_samples = not (dilate or use_weight_thresh)

        batch_size = 720720
        all_sample_vals = []
        all_sample_vals_mask = []
        for i in tqdm(range(0, len(points), batch_size)):
            sample_vals = self.sample(points[i:i+batch_size].to(device=device), grid_coords=True)
            if not use_weight_thresh:
                sample_vals_mask = sample_vals[:, 0] > sigma_thresh
                if pre_mask_samples:
                    sample_vals = sample_vals[sample_vals_mask]
                    sample_vals_mask = sample_vals_mask.cpu()
            else:
                sample_vals_mask = sample_vals[:, 0].clone()
            sample_vals = sample_vals.cpu()
            all_sample_vals.append(sample_vals)
            all_sample_vals_mask.append(sample_vals_mask)
        del self.data

        sample_vals_mask = torch.cat(all_sample_vals_mask, dim=0)
        if use_weight_thresh:
            sigmas = sample_vals_mask.view(reso)
            gsz = torch.tensor(reso)
            offset = (self._offset * gsz - 0.5).to(device=device)
            scaling = (self._scaling * gsz).to(device=device)
            max_wt_grid = torch.zeros(reso, dtype=torch.float32, device=device)
            print(' Grid weight render', sigmas.shape)
            for cam in tqdm(cameras):
                _C.grid_weight_render(sigmas, cam._to_cpp(), 0.5, offset, scaling, max_wt_grid)
            print(' max/mean/min weight', max_wt_grid.max(), max_wt_grid.mean(), max_wt_grid.min())
            sample_vals_mask = max_wt_grid.view(-1) > weight_thresh
            del sigmas
        del all_sample_vals_mask
        if dilate:
            sample_vals_mask = sample_vals_mask.view(reso).cuda()
            for i in range(int(dilate)):
                sample_vals_mask = _C.dilate(sample_vals_mask)
            sample_vals_mask = sample_vals_mask.view(-1).cpu()
        sample_vals = torch.cat(all_sample_vals, dim=0)
        del all_sample_vals
        if not pre_mask_samples:
            sample_vals = sample_vals[sample_vals_mask]
        init_links = torch.cumsum(sample_vals_mask.to(torch.int32), dim=-1).int() - 1
        init_links[~sample_vals_mask] = -1

        self.capacity = sample_vals_mask.sum().item()
        print(' New cap:', self.capacity)
        del sample_vals_mask
        self.data = nn.Parameter(sample_vals.to(device=device))
        self.links = init_links.view(reso).to(device=device)

    def resize(self, basis_dim : int):
        """
        Modify the size of the data stored in the voxels. Called expand/shrink in svox 1.

        :param basis_dim: new basis dimension, must be square number
        """
        assert isqrt(basis_dim) is not None, "basis_dim (SH) must be a square number"
        assert basis_dim >= 1 and basis_dim <= MAX_SH_BASIS, \
                f"basis_dim 1-{MAX_SH_BASIS} supported"
        old_basis_dim = self.basis_dim
        self.basis_dim = basis_dim
        device = self.data.device
        old_data = self.data.data.cpu()

        shrinking = basis_dim < old_basis_dim
        data_dim = 3 * basis_dim + 1
        old_data_dim = 3 * old_basis_dim + 1
        sigma_arr = torch.tensor([0])
        if shrinking:
            shift = old_basis_dim
            arr = torch.arange(basis_dim) + 1
            remap = torch.cat([sigma_arr, arr, shift + arr, 2 * shift + arr])
        else:
            shift = basis_dim
            arr = torch.arange(old_basis_dim) + 1
            remap = torch.cat([sigma_arr, arr, shift + arr, 2 * shift + arr])

        del self.data
        new_data = torch.zeros((old_data.size(0), 3 * basis_dim + 1), device='cpu')
        if shrinking:
            new_data[:] = old_data[..., remap]
        else:
            new_data[..., remap] = old_data
        new_data = new_data.to(device=device)
        self.data = nn.Parameter(new_data)


    def world2grid(self, points):
        """
        World coordinates to grid coordinates. Grid coordinates are
        normalized to [0, n_voxels] in each side

        :param points: (N, 3)
        :return: (N, 3)
        """
        gsz = self._grid_size()
        offset = self._offset * gsz - 0.5;
        scaling = self._scaling * gsz
        return torch.addcmul(offset.to(device=points.device),
                             points,
                             scaling.to(device=points.device))

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
        return torch.addcmul(roffset.to(device=points.device),
                             points,
                             rscaling.to(device=points.device))

    def save(self, path : str, compress : bool = False):
        """
        Save to a path
        """
        save_fn = np.savez_compressed if compress else np.savez
        save_fn(path,
                radius=self.radius.numpy(),
                center=self.center.numpy(),
                links=self.links.cpu().numpy(),
                data=self.data.data.cpu().numpy().astype(np.float16))

    @classmethod
    def load(cls, path : str, device : Union[torch.device, str]="cpu"):
        """
        Load from path
        """
        z = np.load(path)
        data = z.f.data
        links = z.f.links
        basis_dim = (data.shape[1] - 1) // 3
        radius = z.f.radius.tolist() if 'radius' in z.files else [1.0, 1.0, 1.0]
        center = z.f.center.tolist() if 'center' in z.files else [0.0, 0.0, 0.0]
        grid = cls(1,
            radius=radius,
            center=center,
            basis_dim = basis_dim,
            use_z_order = False,
            device='cpu')
        if data.dtype != np.float32:
            data = data.astype(np.float32)
        data = torch.from_numpy(data).to(device=device)
        grid.data = nn.Parameter(data)
        grid.links = torch.from_numpy(links).to(device=device)
        grid.capacity = grid.data.size(0)
        return grid

    def to_svox1(self, device : Union[torch.device, str, None]=None):
        """
        Convert the grid to a svox 1 octree. Requires svox (pip install svox)

        :param device: device to put the octree. None = grid data's device
        """
        assert self.is_cubic_pow2, \
               "Grid must be cubic and power-of-2 to be compatible with svox octree"
        if device is None:
            device = self.data.device
        import svox
        n_refine = int(np.log2(self.links.size(0))) - 1

        t = svox.N3Tree(data_format=f'SH{self.basis_dim}',
                        init_refine=0,
                        radius=self.radius.tolist(),
                        center=self.center.tolist(),
                        device=device)

        curr_reso = self.links.shape
        dtype = torch.float32
        X = (torch.arange(curr_reso[0], dtype=dtype, device=device) + 0.5) / curr_reso[0]
        Y = (torch.arange(curr_reso[1], dtype=dtype, device=device) + 0.5) / curr_reso[0]
        Z = (torch.arange(curr_reso[2], dtype=dtype, device=device) + 0.5) / curr_reso[0]
        X, Y, Z = torch.meshgrid(X, Y, Z)
        points = torch.stack((X, Y, Z), dim=-1).view(-1, 3)

        mask = self.links.view(-1) >= 0
        points = points[mask.to(device=device)]
        index = svox.LocalIndex(points)
        print('n_refine', n_refine)
        for i in tqdm(range(n_refine)):
            t[index].refine()

        t[index, :-1] = self.data.data[:, 1:].to(device=device)
        t[index, -1] = self.data.data[:, 0].to(device=device)
        return t

    def tv(self, start_dim : int = 0, end_dim : Optional[int] = None):
        """
        Compute L1 total variation as in Neural Volumes [Lombardi et al., ToG 2019]
        
        :param start_dim: int, first channel dimension to compute TV over (inclusive).
                          Default 0.
        :param end_dim: int, last channel dimension to compute TV over (exclusive).
                          Default None = data_dim.

        :return: torch.Tensor, size scalar, the TV value (sum over channels,
                 mean over voxels)
        """
        assert _C is not None and self.data.is_cuda, \
                "CUDA extension is currently required for total variation"
        if end_dim is None: 
            end_dim = self.data_dim
        end_dim = end_dim + self.data_dim if end_dim < 0 else end_dim
        start_dim = start_dim + self.data_dim if start_dim < 0 else start_dim
        return _TotalVariationFunction.apply(self.data, self.links, start_dim, end_dim)

    def inplace_tv_grad(self, grad : torch.Tensor,
                        start_dim : int = 0,
                        end_dim : Optional[int] = None,
                        scaling : float = 1.0):
        """
        Add gradient of L1 total variation as in Neural Volumes [Lombardi et al., ToG 2019]
        directly into the gradient tensor, multiplied by 'scaling'
        """
        assert _C is not None and self.data.is_cuda and grad.is_cuda, \
                "CUDA extension is currently required for total variation"
        if end_dim is None: 
            end_dim = self.data_dim
        end_dim = end_dim + self.data_dim if end_dim < 0 else end_dim
        start_dim = start_dim + self.data_dim if start_dim < 0 else start_dim
        _C.tv_grad(self.links, self.data, start_dim, end_dim, scaling, grad)

    def __repr__(self):
        return (f"svox2.SparseGrid(basis_dim={self.basis_dim}, " +
                f"reso={list(self.links.shape)}, " +
                f"capacity:{self.data.size(0)})")

    def is_cubic_pow2(self):
        """
        Check if the current grid is cubic (same in all dims) with power-of-2 size.
        This allows for conversion to svox 1 and Z-order curve (Morton code)
        """
        reso = self.links.shape
        return (reso[0] == reso[1] and reso[0] == reso[2] and is_pow2(reso[0]))

    def _to_cpp(self, grid_coords : bool = False):
        """
        Generate object to pass to C++
        """
        gspec = _C.SparseGridSpec()
        gspec.data = self.data
        gspec.links = self.links
        if grid_coords:
            gspec._offset = torch.zeros_like(self._offset)
            gspec._scaling = torch.ones_like(self._offset)
        else:
            gsz = self._grid_size()
            gspec._offset = self._offset * gsz - 0.5
            gspec._scaling = self._scaling * gsz
        gspec.basis_dim = self.basis_dim
        return gspec

    def _grid_size(self):
        return torch.tensor(self.links.shape, device='cpu', dtype=torch.float32)
