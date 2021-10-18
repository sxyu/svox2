import torch
import svox2

device = 'cuda:0'

torch.random.manual_seed(0)
g = svox2.SparseGrid(center=[0.0, 0.0, 0.0],
                     radius=[1.0, 1.0, 1.0],
                     device=device)

g.opt.sigma_thresh = 0.0
g.opt.stop_thresh = 0.0

g.sh_data.data.normal_()
g.density_data.data[..., 0] = 1.5
g.sh_data.data[..., 0] += 1.0

N_RAYS = 5000

origins = torch.zeros(N_RAYS, 3, device=device) #torch.randn(N_RAYS, 3) * 0.01
dirs = torch.randn(N_RAYS, 3, device=device)
dirs = dirs / torch.norm(dirs, dim=-1).unsqueeze(-1)

rays = svox2.Rays(origins=origins, dirs=dirs)

rgb_gt = g.volume_render(rays, use_kernel=False)
torch.cuda.synchronize()
rgb = g.volume_render(rays, use_kernel=True)
torch.cuda.synchronize()

E = torch.abs(rgb - rgb_gt)
err = E.max().detach().item()
print(err)
