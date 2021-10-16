import svox2
import torch
from util import Timing

torch.random.manual_seed(0)

device = 'cuda:0'
grid = svox2.SparseGrid(reso=128, center=[0.0, 0.0, 0.0],
                        radius=1.0, device=device)
grid.data.data[..., 1:].normal_(0.0, 1.0)
grid.data.data[..., 0] = 1.0
#  grid = torch.rand((2, 2, 2, 4), device=device, dtype=torch.float32)

N_POINTS = 5000 * 1024
points = torch.rand(N_POINTS, 3, device=device) * 2 - 1
#  points = torch.tensor([[0.49, 0.49, 0.49], [0.9985, 0.4830, 0.4655]], device=device)
#  points.clamp_(-0.999, 0.999)

sampc = grid.sample(points)
sampt = grid.sample(points, use_kernel=False)

grid.requires_grad_(True)

with Timing("ours"):
    sampc = grid.sample(points)

s = sampc.sum()
with Timing("our_back"):
    s.backward()
go = grid.data.grad.clone()
grid.data.grad = None

with Timing("torch"):
    sampt = grid.sample(points, use_kernel=False)
s = sampt.sum()
with Timing("torch_back"):
    s.backward()
gt = grid.data.grad.clone()

#  print('c\n', sampc)
#  print('t\n', sampt)
print('err\n', torch.abs(sampt-sampc).max())
print('err_grad\n', torch.abs(go-gt).max())
