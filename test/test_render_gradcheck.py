import svox2
import torch
import torch.nn.functional as F
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
                     device=device,
                     background_nlayers=16,
                     basis_type=svox2.BASIS_TYPE_SH)
grid.opt.sigma_thresh = 0.0
grid.opt.stop_thresh = 0.0

print(grid.sh_data.shape)
#  grid.sh_data.data.normal_()
grid.sh_data.data[..., 0] = 0.5
grid.sh_data.data[..., 1:].normal_(std=0.01)
grid.density_data.data[:] = 0.1

if grid.use_background:
	grid.background_cubemap.data[..., -1] = 0.5
	grid.background_cubemap.data[..., :-1] = torch.randn_like(
            grid.background_cubemap.data[..., :-1]) * 0.01

if grid.basis_type == svox2.BASIS_TYPE_3D_TEXTURE:
    grid.basis_data.data.normal_()
    grid.basis_data.data += 1.0

ENABLE_TORCH_CHECK = True
N_RAYS = 5000 #200 * 200
#  origins = torch.full((N_RAYS, 3), fill_value=0.0, device=device, dtype=dtype)
origins = torch.zeros((N_RAYS, 3), device=device, dtype=dtype)
dirs : torch.Tensor = torch.randn((N_RAYS, 3), device=device, dtype=dtype)
dirs /= torch.norm(dirs, dim=-1, keepdim=True)
rays = svox2.Rays(origins, dirs)

rgb_gt = torch.zeros((N_RAYS, 3), device=device, dtype=dtype)

#  grid.requires_grad_(True)

#  samps = grid.volume_render(rays, use_kernel=True)
#  sampt = grid.volume_render(grid, origins, dirs, use_kernel=False)

with Timing("ours"):
    samps = grid.volume_render(rays, use_kernel=True)
s = F.mse_loss(samps, rgb_gt)
print(s)
print('bkwd..')
with Timing("ours_backward"):
    s.backward()
grid_sh_grad_s = grid.sh_data.grad.clone().cpu()
grid_density_grad_s = grid.density_data.grad.clone().cpu()
grid.sh_data.grad = None
grid.density_data.grad = None
if grid.basis_type == svox2.BASIS_TYPE_3D_TEXTURE:
    grid_basis_grad_s = grid.basis_data.grad.clone().cpu()
    grid.basis_data.grad = None
if grid.use_background:
    grid_bg_grad_s = grid.background_cubemap.grad.clone().cpu()
    grid.background_cubemap.grad = None

if ENABLE_TORCH_CHECK:
    with Timing("torch"):
        sampt = grid.volume_render(rays, use_kernel=False)
    s = F.mse_loss(sampt, rgb_gt)
    with Timing("torch_backward"):
        s.backward()
    grid_sh_grad_t = grid.sh_data.grad.clone().cpu()
    grid_density_grad_t = grid.density_data.grad.clone().cpu()
    if grid.basis_type == svox2.BASIS_TYPE_3D_TEXTURE:
        grid_basis_grad_t = grid.basis_data.grad.clone().cpu()
    if grid.use_background:
        grid_bg_grad_t = grid.background_cubemap.grad.clone().cpu()

    E = torch.abs(grid_sh_grad_s-grid_sh_grad_t)
    Ed = torch.abs(grid_density_grad_s-grid_density_grad_t)
    if grid.basis_type == svox2.BASIS_TYPE_3D_TEXTURE:
        Eb = torch.abs(grid_basis_grad_s-grid_basis_grad_t)
    if grid.use_background:
        Ebg = torch.abs(grid_bg_grad_s-grid_bg_grad_t)
    print('err', torch.abs(samps - sampt).max())
    print('err_sh_grad\n', E.max())
    print(' mean\n', E.mean())
    print('err_density_grad\n', Ed.max())
    print(' mean\n', Ed.mean())
    if grid.basis_type == svox2.BASIS_TYPE_3D_TEXTURE:
        print('err_basis_grad\n', Eb.max())
        print(' mean\n', Eb.mean())
    if grid.use_background:
        print('err_background_grad\n', Ebg.max())
        print(' mean\n', Ebg.mean())
    print()
    print('g_ours sh min/max\n', grid_sh_grad_s.min(), grid_sh_grad_s.max())
    print('g_torch sh min/max\n', grid_sh_grad_t.min(), grid_sh_grad_t.max())
    print('g_ours sigma min/max\n', grid_density_grad_s.min(), grid_density_grad_s.max())
    print('g_torch sigma min/max\n', grid_density_grad_t.min(), grid_density_grad_t.max())
    if grid.basis_type == svox2.BASIS_TYPE_3D_TEXTURE:
        print('g_ours basis min/max\n', grid_basis_grad_s.min(), grid_basis_grad_s.max())
        print('g_torch basis min/max\n', grid_basis_grad_t.min(), grid_basis_grad_t.max())
    if grid.use_background:
        print('g_ours bg min/max\n', grid_bg_grad_s.min(), grid_bg_grad_s.max())
        print('g_torch bg min/max\n', grid_bg_grad_t.min(), grid_bg_grad_t.max())
