# nvprof -f --profile-from-start off --quiet --metrics all --events all -o prof.nvvp python prof.py
# then use nvvp to open prof.nvvp
import svox2
import torch
import numpy as np
from util import Timing
from matplotlib import pyplot as plt

import torch.cuda.profiler as profiler
import pyprof

device='cuda:0'

GRID_FILE = 'lego.npy'
grid = svox2.SparseGrid(reso=256, device='cpu', radius=1.3256)
data = torch.from_numpy(np.load(GRID_FILE)).view(-1, grid.data_dim)
grid.sh_data.data = data[..., 1:]
grid.density_data.data = data[..., :1]
grid = grid.cuda()
#  grid.data.data[..., 0] += 0.1

N_RAYS = 5000
#  origins = torch.full((N_RAYS, 3), fill_value=0.0, device=device, dtype=dtype)
origins = torch.zeros((N_RAYS, 3), device=device, dtype=torch.float32)
dirs : torch.Tensor = torch.randn((N_RAYS, 3), device=device, dtype=torch.float32)
dirs /= torch.norm(dirs, dim=-1, keepdim=True)
rays = svox2.Rays(origins, dirs)

grid.requires_grad_(True)

samps = grid.volume_render(rays, use_kernel=True)
#  sampt = grid.volume_render(grid, origins, dirs, use_kernel=False)

pyprof.init()
with torch.autograd.profiler.emit_nvtx():
    profiler.start()
    samps = grid.volume_render(rays, use_kernel=True)
    s = samps.sum()
    s.backward()
    profiler.stop()
