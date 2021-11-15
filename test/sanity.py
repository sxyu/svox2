import torch
import svox2

device = 'cuda:0'


torch.random.manual_seed(4000)
g = svox2.SparseGrid(center=[0.0, 0.0, 0.0],
                     radius=[1.0, 1.0, 1.0],
                     device=device,
                     basis_type=svox2.BASIS_TYPE_SH,
                     background_nlayers=0)

g.opt.backend = 'nvol'
g.opt.sigma_thresh = 0.0
g.opt.stop_thresh = 0.0
g.opt.background_brightness = 1.0

g.sh_data.data.normal_()
g.density_data.data[..., 0] = 0.1
g.sh_data.data[..., 0] = 0.5
g.sh_data.data[..., 1:] = torch.randn_like(g.sh_data.data[..., 1:]) * 0.01

if g.use_background:
    g.background_data.data[..., -1] = 1.0
    g.background_data.data[..., :-1] = torch.randn_like(
            g.background_data.data[..., :-1]) * 0.01
    #  g.background_data.data[..., :-1] = 0.5

g.basis_data.data.normal_()
g.basis_data.data *= 10.0
#  print('use frustum?', g.use_frustum)

N_RAYS = 1

#  origins = torch.randn(N_RAYS, 3, device=device) * 3
#  dirs = torch.randn(N_RAYS, 3, device=device)
#  origins = origins[27513:27514]
#  dirs = dirs[27513:27514]

origins = torch.tensor([[-3.8992738723754883, 4.844727993011475, 4.323856830596924]], device='cuda:0')
dirs = torch.tensor([[1.1424630880355835, -1.2679963111877441, -0.8437137603759766]], device='cuda:0')
dirs = dirs / torch.norm(dirs, dim=-1).unsqueeze(-1)

rays = svox2.Rays(origins=origins, dirs=dirs)

rgb = g.volume_render(rays, use_kernel=True)
torch.cuda.synchronize()
rgb_gt = g.volume_render(rays, use_kernel=False)
torch.cuda.synchronize()

E = torch.abs(rgb - rgb_gt)
err = E.max().detach().item()
print(err)
