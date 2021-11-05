import torch
import svox2

device = 'cuda:0'

torch.random.manual_seed(199)
g = svox2.SparseGrid(center=[0.0, 0.0, 0.0],
                     radius=[1.0, 1.0, 1.0],
                     device=device,
                     basis_type=svox2.BASIS_TYPE_SH,
                     background_nlayers=32)

g.opt.sigma_thresh = 0.0
g.opt.stop_thresh = 0.0
g.opt.background_brightness = 1.0

g.sh_data.data.normal_()
g.density_data.data[..., 0] = 1.1
g.sh_data.data[..., 0] = 0.5
g.sh_data.data[..., 1:] = torch.randn_like(g.sh_data.data[..., 1:]) * 0.01

if g.use_background:
    g.background_cubemap.data[..., -1] = 1.0
    g.background_cubemap.data[..., :-1] = torch.randn_like(
            g.background_cubemap.data[..., :-1]) * 0.01
    g.background_cubemap.data[..., :-1] = 0.5

g.basis_data.data.normal_()
g.basis_data.data *= 10.0
#  print('use frustum?', g.use_frustum)

N_RAYS = 30000

#  origins = torch.full((N_RAYS, 3), fill_value=-3.0, device=device) #torch.randn(N_RAYS, 3) * 0.01
origins = torch.zeros(N_RAYS, 3, device=device) * 3
#  origins[:, 0] *= 1.1
dirs = torch.randn(N_RAYS, 3, device=device)
#  dirs = torch.ones(N_RAYS, 3, device=device)
#  dirs[:, 0] *= 2
dirs[0] = 1.0
dirs = dirs / torch.norm(dirs, dim=-1).unsqueeze(-1)

rays = svox2.Rays(origins=origins, dirs=dirs)

rgb = g.volume_render(rays, use_kernel=True)
torch.cuda.synchronize()
rgb_gt = g.volume_render(rays, use_kernel=False)
torch.cuda.synchronize()

E = torch.abs(rgb - rgb_gt)
err = E.max().detach().item()
print(err)
