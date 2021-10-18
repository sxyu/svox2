import svox2
import torch
from util import Timing

#  torch.random.manual_seed(0)
torch.random.manual_seed(1234)

device = 'cuda:0'
dtype = torch.float32
grid = svox2.SparseGrid(
                     reso=128,
                     center=[0.0, 0.0, 0.0],
                     radius=[1.0, 1.0, 1.0],
                     basis_dim=9,
                     use_z_order=True,
                     device=device)
grid.opt.sigma_thresh = 0.0
grid.opt.stop_thresh = 0.0

print(grid.sh_data.shape)
grid.sh_data.data.normal_()
grid.density_data.data[:] = 0.1

ENABLE_TORCH_CHECK = True
N_RAYS = 5000 #200 * 200
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
grid_sh_grad_s = grid.sh_data.grad.clone().cpu()
grid_density_grad_s = grid.density_data.grad.clone().cpu()
grid.sh_data.grad = None
grid.density_data.grad = None

if ENABLE_TORCH_CHECK:
    with Timing("torch"):
        sampt = grid.volume_render(rays, use_kernel=False)
    s = sampt.sum()
    with Timing("torch_backward"):
        s.backward()
    grid_sh_grad_t = grid.sh_data.grad.clone().cpu()
    grid_density_grad_t = grid.density_data.grad.clone().cpu()

    E = torch.abs(grid_sh_grad_s-grid_sh_grad_t)
    Ed = torch.abs(grid_density_grad_s-grid_density_grad_t)
    print('err', torch.abs(samps - sampt).max())
    print('err_sh_grad\n', E.max())
    print(' mean\n', E.mean())
    print('err_density_grad\n', Ed.max())
    print(' mean\n', Ed.mean())
    print()
    print('g_ours min/max\n', grid_sh_grad_s.min(), grid_sh_grad_s.max())
    print('g_torch min/max\n', grid_sh_grad_t.min(), grid_sh_grad_t.max())
    print('g_ours sigma min/max\n', grid_density_grad_s[..., 0].min(), grid_density_grad_s[..., 0].max())
    print('g_torch sigma min/max\n', grid_density_grad_t[..., 0].min(), grid_density_grad_t[..., 0].max())
