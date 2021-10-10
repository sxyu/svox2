import svox2
import torch
from util import Timing

torch.random.manual_seed(0)

device = 'cuda:0'
dtype = torch.float32
grid = svox2.SparseGrid(
                     reso=128,
                     center=[0.0, 0.0, 0.0],
                     radius=[1.0, 1.0, 1.0],
                     basis_dim=8,
                     use_z_order=True,
                     device=device)
grid.opt.sigma_thresh = 0.0
grid.opt.stop_thresh = 0.0

print(grid.data.shape)
grid.data.data.normal_()
grid.data.data[..., 0] = 0.1

ENABLE_TORCH_CHECK = True
N_RAYS = 1000#200 * 200
#  origins = torch.full((N_RAYS, 3), fill_value=0.0, device=device, dtype=dtype)
origins = torch.zeros((N_RAYS, 3), device=device, dtype=dtype)
dirs : torch.Tensor = torch.randn((N_RAYS, 3), device=device, dtype=dtype)
dirs /= torch.norm(dirs, dim=-1, keepdim=True)
rays = svox2.Rays(origins, dirs)

grid.requires_grad_(True)

samps = grid.volume_render(rays, use_kernel=True)
#  sampt = grid.volume_render(grid, origins, dirs, use_kernel=False)

with Timing("ours"):
    samps = grid.volume_render(rays, use_kernel=True)
s = samps.sum()
with Timing("ours_backward"):
    s.backward()
grid_grad_s = grid.data.grad.clone().cpu()
cubemap_grad_s = grid.cubemap.grad.clone().cpu()
grid.data.grad = None
grid.cubemap.grad = None

if ENABLE_TORCH_CHECK:
    with Timing("torch"):
        sampt = grid.volume_render(rays, use_kernel=False)
    s = sampt.sum()
    with Timing("torch_backward"):
        s.backward()
    grid_grad_t = grid.data.grad.clone().cpu()
    cubemap_grad_t = grid.cubemap.grad.clone().cpu()

    E = torch.abs(grid_grad_s-grid_grad_t)
    E_cube = torch.abs(cubemap_grad_s-cubemap_grad_t)

    print('err', torch.abs(samps - sampt).max())
    print()
    print('err_grad\n', E.max())
    print(' mean\n', E.mean())
    print('err_grad_c\n', E[..., 1:].max())
    print('err_grad_sigma\n', E[..., 0].max())
    print()
    print('err_grad_cubemap\n', E_cube.max())
    print()
    print()
    print('g_ours min/max\n', grid_grad_s.min(), grid_grad_s.max())
    print('g_torch min/max\n', grid_grad_t.min(), grid_grad_t.max())
    print('g_ours sigma min/max\n', grid_grad_s[..., 0].min(), grid_grad_s[..., 0].max())
    print('g_torch sigma min/max\n', grid_grad_t[..., 0].min(), grid_grad_t[..., 0].max())
