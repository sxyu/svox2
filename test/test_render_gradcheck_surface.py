import svox2
import torch
import svox2.csrc as _C
import torch.nn.functional as F
from util import Timing
import numpy as np

import os
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'

torch.random.manual_seed(2)
#  torch.random.manual_seed(8289)

# test gradient
# x = torch.tensor(-2.).requires_grad_()
# x.retain_grad()
# eq = (x ** (3.))
# eq.backward()
# print(x.grad)

torch.autograd.set_detect_anomaly(True)

device = 'cuda:0'
surface_type = svox2.SURFACE_TYPE_SDF
dtype = torch.float32
grid = svox2.SparseGrid(
                     reso=128,
                     center=[0.0, 0.0, 0.0],
                     radius=[1.0, 1.0, 1.0],
                     basis_dim=9,
                     use_z_order=True,
                     device=device,
                     background_nlayers=0,
                     basis_type=svox2.BASIS_TYPE_SH,
                     surface_type=surface_type,
                     use_sphere_bound=False,
                     trainable_fake_sample_std=True,
                     surface_init='sphere',
                     )



# grid.opt.backend = 'surface'
grid.opt.backend = 'surf_trav'
grid.opt.sigma_thresh = -20
grid.opt.stop_thresh = 0.0
grid.opt.background_brightness = 1.0
grid.opt.near_clip = 0.0
grid.opt.step_size = 0.01
# grid.opt.only_outward_intersect = True
grid.opt.truncated_vol_render = True
grid.truncated_vol_render_a = 5

# grid.level_set_data = torch.tensor([0.1, 0, -0.1], dtype=grid.surface_data.dtype, device=grid.surface_data.device)
grid.level_set_data = torch.tensor([0.], dtype=grid.surface_data.dtype, device=grid.surface_data.device)

grid.opt.surf_fake_sample = True
# grid.fake_sample_std = 1
grid.opt.surf_fake_sample_min_vox_len = 0.
grid.opt.limited_fake_sample = True

grid.opt.alpha_activation_type = 1

lambda_l2 = 0
lambda_l1 = 1

print(grid.opt)

print(grid.sh_data.shape)
#  grid.sh_data.data.normal_()
grid.sh_data.data[..., 0] = 0.5
grid.sh_data.data[..., 1:].normal_(std=0.1)
grid.density_data.data[:] = 10

# grid.sh_data.data[:].normal_(std=0.1)
grid.density_data.data[:].normal_(mean=0.5, std=0.1)
grid.surface_data.data += torch.rand_like(grid.surface_data.data) -0.5


if grid.use_background:
	grid.background_data.data[..., -1] = 0.5
	grid.background_data.data[..., :-1] = torch.randn_like(
            grid.background_data.data[..., :-1]) * 0.01

if grid.basis_type == svox2.BASIS_TYPE_3D_TEXTURE:
    grid.basis_data.data.normal_()
    grid.basis_data.data += 1.0



# pts = grid.extract_pts(density_thresh=-100)

# pretrained_ckpt_path = '/home/tw554/plenoxels/opt/ckpt/nerf/lego/ckpt.npz'

# z = np.load(pretrained_ckpt_path, allow_pickle=True)
# sh_data = z.f.sh_data
# density_data = z.f.density_data
# links = z.f.links
# sh_data = torch.from_numpy(sh_data.astype(np.float32)).to(device=device)
# density_data = torch.from_numpy(density_data.astype(np.float32)).to(device=device)
# grid.sh_data.data = sh_data
# grid.density_data.data = density_data

# grid.links = torch.from_numpy(links).to(device=device)
# grid.capacity = grid.sh_data.size(0)
# grid.accelerate()

ENABLE_TORCH_CHECK = True
#  N_RAYS = 5000 #200 * 200
N_RAYS = 200 * 200
# N_RAYS = 100
origins = torch.randn((N_RAYS, 3), device=device, dtype=dtype) * 3
dirs = torch.randn((N_RAYS, 3), device=device, dtype=dtype)

origins = torch.tensor(np.load('/home/tw554/plenoxels/test/test_origins.npy'), device=device, dtype=dtype)
dirs = torch.tensor(np.load('/home/tw554/plenoxels/test/test_dirs.npy'), device=device, dtype=dtype)

dirs /= torch.norm(dirs, dim=-1, keepdim=True)

#  start = 71
#  end = 72
#  origins = origins[start:end]
#  dirs = dirs[start:end]
#  print(origins.tolist(), dirs.tolist())

#  breakpoint()
rays = svox2.Rays(origins, dirs)


# IDX = torch.tensor([2655], dtype=torch.long) # 
# IDX = torch.tensor([2000], dtype=torch.long) # 
IDX = torch.tensor([2500], dtype=torch.long) # 
# IDX = torch.arange(0, 2500, dtype=torch.long)
rays.origins = rays.origins[IDX, ...]
rays.dirs = rays.dirs[IDX, ...]

# Note that the gradient is never correct on all rays !!!
# this is due to numerical issues

rgb_gt = torch.zeros((rays.origins.size(0), 3), device=device, dtype=dtype)


grid.requires_grad_(True)

# out = grid.volume_render(rays, use_kernel=True, no_surface=False)
# out = grid.volume_render_normal(rays)
# print(out)

# with Timing("ours"):
#     out = grid.volume_render(rays, use_kernel=True, no_surface=False)
#     samps = out['rgb']
# s = F.mse_loss(samps, rgb_gt) * lambda_l2 + torch.abs(samps - rgb_gt).mean() * lambda_l1

# print(s)
# print('bkwd..')
# with Timing("ours_backward"):
#     s.backward()

lambda_sparsity_loss = 0.
lambda_l_entropy = 0.
lambda_conv_mode_samp = 1.
with Timing("ours"):
    out = grid.volume_render_fused(rays, rgb_gt,
            lambda_l2 = lambda_l2,
            lambda_l1 = lambda_l1,
            lambda_l_dist = 0.,
            lambda_l_entropy = lambda_l_entropy,
            no_norm_weight_l_entropy = True,
            lambda_l_samp_dist = 0.,
            sparsity_loss=lambda_sparsity_loss,
            surf_sparse_alpha_thresh=0.1,
            lambda_inplace_surf_sparse=0.,
            lambda_inwards_norm_loss=0.,
            lambda_conv_mode_samp=lambda_conv_mode_samp,
            no_surface=False)

samps = out['rgb']


# grid.inplace_surface_normal_grad(grid.surface_data.grad, use_kernel=False)


grid_sh_grad_s = grid.sh_data.grad.clone().cpu()
grid_density_grad_s = grid.density_data.grad.clone().cpu()
grid_surface_grad_s = grid.surface_data.grad.clone().cpu()
grid.sh_data.grad = None
grid.density_data.grad = None
grid.surface_data.grad = None
if grid.basis_type == svox2.BASIS_TYPE_3D_TEXTURE:
    grid_basis_grad_s = grid.basis_data.grad.clone().cpu()
    grid.basis_data.grad = None
if grid.use_background:
    grid_bg_grad_s = grid.background_data.grad.clone().cpu()
    grid.background_data.grad = None

if grid.fake_sample_std.requires_grad:
    fake_sample_std_grad_s = grid.fake_sample_std.grad.clone().cpu()
    grid.fake_sample_std.grad = None

# grid.opt.surf_fake_sample = True

if ENABLE_TORCH_CHECK:
    with Timing("torch"):
        out = grid.volume_render(rays, use_kernel=False, allow_outside=False, run_backward=True, 
                                 lambda_l_dist=0, lambda_l_entropy=lambda_l_entropy, sparsity_loss=lambda_sparsity_loss,
                                 lambda_conv_mode_samp=lambda_conv_mode_samp)
        sampt = out['rgb']
    s = F.mse_loss(sampt, rgb_gt) * lambda_l2 + torch.abs(sampt - rgb_gt).mean() * lambda_l1
    print(s)
    # with Timing("torch_backward"):
        # s.backward()
    grid_sh_grad_t = grid.sh_data.grad.clone().cpu() if grid.sh_data.grad is not None else torch.zeros_like(grid_sh_grad_s)
    grid_density_grad_t = grid.density_data.grad.clone().cpu() if grid.density_data.grad is not None else torch.zeros_like(grid_density_grad_s)
    grid_surface_grad_t = grid.surface_data.grad.clone().cpu() if grid.surface_data.grad is not None else torch.zeros_like(grid_surface_grad_s)
    if grid.basis_type == svox2.BASIS_TYPE_3D_TEXTURE:
        grid_basis_grad_t = grid.basis_data.grad.clone().cpu()
    if grid.use_background:
        grid_bg_grad_t = grid.background_data.grad.clone().cpu() if grid.background_data.grad is not None else torch.zeros_like(grid_bg_grad_s)

    E = torch.abs(grid_sh_grad_s-grid_sh_grad_t)
    Ed = torch.abs(grid_density_grad_s-grid_density_grad_t)
    Esurface = torch.abs(grid_surface_grad_s-grid_surface_grad_t)
    if grid.basis_type == svox2.BASIS_TYPE_3D_TEXTURE:
        Eb = torch.abs(grid_basis_grad_s-grid_basis_grad_t)
    if grid.use_background:
        Ebg = torch.abs(grid_bg_grad_s-grid_bg_grad_t)
    print('err', torch.abs(samps - sampt).max())
    print('err_sh_grad\n', E.max())
    print(' mean\n', E.mean())
    print('err_density_grad\n', Ed.max())
    print(' mean\n', Ed.mean())
    print('err_surface_grad\n', Esurface.max())
    print(' mean\n', Esurface.mean())
    if grid.basis_type == svox2.BASIS_TYPE_3D_TEXTURE:
        print('err_basis_grad\n', Eb.max())
        print(' mean\n', Eb.mean())
    if grid.use_background:
        print('err_background_grad\n', Ebg.max())
        print(' mean\n', Ebg.mean())
    if grid.fake_sample_std.requires_grad and grid.fake_sample_std.grad is not None:
        fake_sample_std_grad_t = grid.fake_sample_std.grad.clone().cpu()
        
        E_fake_sample_std = torch.abs(fake_sample_std_grad_s - fake_sample_std_grad_t)
        print('E_fake_sample_std\n', E_fake_sample_std.max())
        print('mean\n', E_fake_sample_std.mean())
    # check whether surface grad signs are correct
    surface_correct_sign = ((grid_surface_grad_s > 0) & (grid_surface_grad_t > 0)) | ((grid_surface_grad_s < 0) & (grid_surface_grad_t < 0)) | (grid_surface_grad_s==0) | (grid_surface_grad_t==0) 
    print('Surface grad correct sign:', surface_correct_sign.all())
    print()
    print('g_ours sh min/max\n', grid_sh_grad_s.min(), grid_sh_grad_s.max())
    print('g_torch sh min/max\n', grid_sh_grad_t.min(), grid_sh_grad_t.max())
    print('g_ours sigma min/max\n', grid_density_grad_s.min(), grid_density_grad_s.max())
    print('g_torch sigma min/max\n', grid_density_grad_t.min(), grid_density_grad_t.max())
    print('g_ours surface min/max\n', grid_surface_grad_s.min(), grid_surface_grad_s.max())
    print('g_torch surface min/max\n', grid_surface_grad_t.min(), grid_surface_grad_t.max())
    if grid.basis_type == svox2.BASIS_TYPE_3D_TEXTURE:
        print('g_ours basis min/max\n', grid_basis_grad_s.min(), grid_basis_grad_s.max())
        print('g_torch basis min/max\n', grid_basis_grad_t.min(), grid_basis_grad_t.max())
    if grid.use_background:
        print('g_ours bg min/max\n', grid_bg_grad_s.min(), grid_bg_grad_s.max())
        print('g_torch bg min/max\n', grid_bg_grad_t.min(), grid_bg_grad_t.max())


# [12,4,31]