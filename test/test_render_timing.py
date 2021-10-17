import svox2
import torch
from util import Timing

torch.random.manual_seed(0)

device = 'cuda:0'
dtype = torch.float32
grid = svox2.SparseGrid(
                     reso=256,
                     center=[0.0, 0.0, 0.0],
                     radius=[1.0, 1.0, 1.0],
                     basis_dim=9,
                     use_z_order=True,
                     device=device)
grid.opt.sigma_thresh = 0.0
grid.opt.stop_thresh = 0.0

grid.sh_data.data.normal_()
grid.density_data.data[:] = 0.1

N_RAYS = 200 * 200
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
