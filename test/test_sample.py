import svox2
import torch
import numpy as np
from util import Timing

torch.random.manual_seed(0)

device = 'cuda:0'

#  GRID_FILE = 'lego.npy'
#  grid = svox2.SparseGrid(reso=256, device='cpu', radius=1.3256)
#  grid.data.data = torch.from_numpy(np.load(GRID_FILE)).view(-1, grid.data_dim)
#  grid = grid.cuda()

grid = svox2.SparseGrid(reso=256, center=[0.0, 0.0, 0.0],
                        radius=1.0, device=device)
grid.sh_data.data.normal_(0.0, 1.0)
grid.density_data.data.normal_(0.1, 0.05).clamp_min_(0.0)
#  grid.density_data.data[:] = 1.0
#  grid = torch.rand((2, 2, 2, 4), device=device, dtype=torch.float32)

N_POINTS = 5000 * 1024
points = torch.rand(N_POINTS, 3, device=device) * 2 - 1
#  points = torch.tensor([[0.49, 0.49, 0.49], [0.9985, 0.4830, 0.4655]], device=device)
#  points.clamp_(-0.999, 0.999)

_ = grid.sample(points)
_ = grid.sample(points, use_kernel=False)

grid.requires_grad_(True)

with Timing("ours"):
    sigma_c, rgb_c = grid.sample(points)

s = sigma_c.sum() + rgb_c.sum()
with Timing("our_back"):
    s.backward()
gdo = grid.density_data.grad.clone()
gso = grid.sh_data.grad.clone()
grid.density_data.grad = None
grid.sh_data.grad = None

with Timing("torch"):
    sigma_t, rgb_t = grid.sample(points, use_kernel=False)
s = sigma_t.sum() + rgb_t.sum()
with Timing("torch_back"):
    s.backward()
gdt = grid.density_data.grad.clone()
gst = grid.sh_data.grad.clone()

#  print('c\n', sampc)
#  print('t\n', sampt)
print('err_sigma\n', torch.abs(sigma_t-sigma_c).max())
print('err_rgb\n', torch.abs(rgb_t-rgb_c).max())
print('err_grad_sigma\n', torch.abs(gdo-gdt).max())
print('err_grad_rgb\n', torch.abs(gso-gst).max())
