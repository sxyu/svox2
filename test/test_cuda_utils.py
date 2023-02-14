import svox2
import torch
import torch.nn.functional as F
import svox2.csrc as _C
from util import Timing
import numpy as np

import os
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'

torch.random.manual_seed(2)

def grad_sign_check(g1, g2):
    return ((g1 > 0) & (g2 > 0)) | ((g1 < 0) & (g2 < 0)) | (g1==0) | (g2==0)

def cubic_root_grad_py(fs, st_ids, dtype=torch.double):
    # analyical solution for f0 + _t*f1 + (_t**2)*f2 + (_t**3)*f3 = 0
    # https://github.com/shril/CubicEquationSolver/blob/master/CubicEquationSolver.py

    # check for trivial a and b -- reduce to linear or polynomial solutions
    # no_solution_mask = (f3 == 0.) & (f2 == 0.) & (f1 == 0.)
    # linear_mask = (f3 == 0.) & (f2 == 0.) & (~no_solution_mask)
    # quad_mask = (f3 == 0.) & (~linear_mask) & (~no_solution_mask)
    # cubic_mask = (~quad_mask) & (~linear_mask) & (~no_solution_mask)

    f0,f1,f2,f3 = fs.to(dtype).unbind(-1)
    ts = torch.ones([f0.numel(), 3]).to(dtype).to(device=f0.device) * -1

    atol = 1e-10
    no_solution_mask = torch.isclose(f3, torch.zeros_like(f3), atol=atol) \
            & torch.isclose(f2, torch.zeros_like(f2), atol=atol) \
            & torch.isclose(f1, torch.zeros_like(f1), atol=atol)
    linear_mask = torch.isclose(f3, torch.zeros_like(f3), atol=atol) \
            & torch.isclose(f2, torch.zeros_like(f2), atol=atol) \
            & (~no_solution_mask)
    quad_mask = torch.isclose(f3, torch.zeros_like(f3), atol=atol) \
            & (~linear_mask) & (~no_solution_mask)
    cubic_mask = (~quad_mask) & (~linear_mask) & (~no_solution_mask)


    ##### Linear Roots #####
    if ts[linear_mask].numel() > 0:
        ts[linear_mask, 0] = (-f0[linear_mask] * 1.0) / f1[linear_mask]

    ##### Quadratic Roots #####
    if ts[quad_mask].numel() > 0:
        _b, _c, _d = f2[quad_mask], f1[quad_mask], f0[quad_mask]
        D = _c**2 - 4.0 * _b * _d

        # two real roots
        D_mask = D > 0 
        sqrt_D = torch.sqrt(D[D_mask])
        ids = torch.arange(quad_mask.shape[0])[quad_mask][D_mask]
        t0 = (-_c[D_mask] - sqrt_D) / (2.0 * _b[D_mask])
        t1 = (-_c[D_mask] + sqrt_D) / (2.0 * _b[D_mask])
        ts[ids, 0] = torch.min(torch.stack([t0,t1]), axis=0).values
        ts[ids, 1] = torch.max(torch.stack([t0,t1]), axis=0).values

        # otherwise, has no real roots

    ##### Cubic Roots #####

    cubic_ids = torch.arange(ts.shape[0])[cubic_mask]

    # normalize 
    norm_term = f3[cubic_mask]
    a = f3[cubic_mask] / norm_term
    b = f2[cubic_mask] / norm_term
    c = f1[cubic_mask] / norm_term
    d = f0[cubic_mask] / norm_term

    def cond_cbrt(x, eps=1e-15):
        '''
        Compute cubic root of x based on sign
        '''
        ret = torch.zeros_like(x)
        ret[x >= 0] = torch.pow(torch.clamp_min_(x[x >= 0], eps), 1/3.)
        ret[x < 0] = torch.pow(torch.clamp_min_(-x[x < 0], eps), 1/3.) * -1
        return ret

    f = ((3.*c/a) - ((b**2.) / (a**2.))) / 3.                      
    g = (((2.*(b**3.)) / (a**3.)) - ((9.*b*c) / (a**2.)) + (27.*d/a)) / 27.                 
    h = ((g**2.) / 4. + (f**3.) / 27.) 

    # all three roots are real and equal
    _mask1 = ((f == 0) & (g == 0) & (h == 0))
    ts[cubic_ids[_mask1], 0] = cond_cbrt(d[_mask1]/a[_mask1])

    # all three roots are real 
    _mask2 = (h <= 0) & (~((f == 0) & (g == 0) & (h == 0)))
    _a, _b, _g, _h = a[_mask2], b[_mask2], g[_mask2], h[_mask2]
    
    _i = torch.sqrt(((_g ** 2.) / 4.) - _h)   
    _j = _i ** (1 / 3.)
    eps = 1e-10
    _k = torch.acos(torch.clamp(-(_g / (2 * _i)), -1+eps, 1-eps))              
    # _k = torch.acos(-(_g / (2 * _i)))        
    _L = _j * -1                              
    _M = torch.cos(_k / 3.)       
    _N = np.sqrt(3) * torch.sin(_k / 3.)    
    _P = (_b / (3. * _a)) * -1

    ts[cubic_ids[_mask2], 0] = _L * (_M + _N) + _P
    ts[cubic_ids[_mask2], 1] = _L * (_M - _N) + _P
    ts[cubic_ids[_mask2], 2] = -2 *_L * _M + _P # 2 * _j * torch.cos(_k / 3.) - (_b / (3. * _a))

    # only one root is real
    _mask3 = (h > 0)
    _a, _b, _g, _h = a[_mask3], b[_mask3], g[_mask3], h[_mask3]

    # _R = -(_g.detach().clone() / 2.) + torch.sqrt(_h)    
    # _S = cond_cbrt(_R)

    # _T = -(_g.detach().clone() / 2.) - torch.sqrt(_h)
    # _U = cond_cbrt(_T).detach().clone()

    _R = -(_g / 2.) + torch.sqrt(_h)    
    _S = cond_cbrt(_R)#.detach().clone()

    _T = -(_g / 2.) - torch.sqrt(_h)
    _U = cond_cbrt(_T)

    ts[cubic_ids[_mask3], 0] = (_S + _U) - (_b / (3. * _a))
    # # the rest two are complex roots:
    # ts[cubic_ids[_mask3], 1] = -(_S + _U) / 2 - (_b / (3. * _a)) + (_S - _U) * np.sqrt(3) * 0.5j
    # ts[cubic_ids[_mask3], 2] = -(_S + _U) / 2 - (_b / (3. * _a)) - (_S - _U) * np.sqrt(3) * 0.5j

    assert not torch.isnan(ts).any(), 'NaN detcted in cubic roots'
    assert torch.isfinite(ts).all(), 'Inf detcted in cubic roots'

    ts.retain_grad()
    _U.retain_grad()
    _T.retain_grad()
    _S.retain_grad()
    _R.retain_grad()
    a.retain_grad()
    b.retain_grad()
    c.retain_grad()
    d.retain_grad()
    f.retain_grad()
    g.retain_grad()
    h.retain_grad()
    # D.retain_grad()

    loss = torch.sum(ts[torch.arange(ts.shape[0], device=ts.device), st_ids])
    loss.backward()

    return ts

def cubic_root_grad_py_vieta(fs, st_ids, dtype=torch.double):
    # analyical solution for f0 + _t*f1 + (_t**2)*f2 + (_t**3)*f3 = 0
    # https://github.com/shril/CubicEquationSolver/blob/master/CubicEquationSolver.py

    # check for trivial a and b -- reduce to linear or polynomial solutions
    # no_solution_mask = (f3 == 0.) & (f2 == 0.) & (f1 == 0.)
    # linear_mask = (f3 == 0.) & (f2 == 0.) & (~no_solution_mask)
    # quad_mask = (f3 == 0.) & (~linear_mask) & (~no_solution_mask)
    # cubic_mask = (~quad_mask) & (~linear_mask) & (~no_solution_mask)

    f0,f1,f2,f3 = fs.to(dtype).unbind(-1)
    ts = torch.ones([f0.numel(), 3]).to(dtype).to(device=f0.device) * -1

    atol = 1e-10
    no_solution_mask = torch.isclose(f3, torch.zeros_like(f3), atol=atol) \
            & torch.isclose(f2, torch.zeros_like(f2), atol=atol) \
            & torch.isclose(f1, torch.zeros_like(f1), atol=atol)
    linear_mask = torch.isclose(f3, torch.zeros_like(f3), atol=atol) \
            & torch.isclose(f2, torch.zeros_like(f2), atol=atol) \
            & (~no_solution_mask)
    quad_mask = torch.isclose(f3, torch.zeros_like(f3), atol=atol) \
            & (~linear_mask) & (~no_solution_mask)
    cubic_mask = (~quad_mask) & (~linear_mask) & (~no_solution_mask)


    ##### Linear Roots #####
    if ts[linear_mask].numel() > 0:
        ts[linear_mask, 0] = (-f0[linear_mask] * 1.0) / f1[linear_mask]

    ##### Quadratic Roots #####
    if ts[quad_mask].numel() > 0:
        _b, _c, _d = f2[quad_mask], f1[quad_mask], f0[quad_mask]
        D = _c**2 - 4.0 * _b * _d

        # two real roots
        D_mask = D > 0 
        sqrt_D = torch.sqrt(D[D_mask])
        ids = torch.arange(quad_mask.shape[0])[quad_mask][D_mask]
        t0 = (-_c[D_mask] - sqrt_D) / (2.0 * _b[D_mask])
        t1 = (-_c[D_mask] + sqrt_D) / (2.0 * _b[D_mask])
        ts[ids, 0] = torch.min(torch.stack([t0,t1]), axis=0).values
        ts[ids, 1] = torch.max(torch.stack([t0,t1]), axis=0).values

        # otherwise, has no real roots

    ##### Cubic Roots #####

    cubic_ids = torch.arange(ts.shape[0])[cubic_mask]

    # normalize 
    norm_term = f3[cubic_mask]
    a = f3[cubic_mask] / norm_term
    b = f2[cubic_mask] / norm_term
    c = f1[cubic_mask] / norm_term
    d = f0[cubic_mask] / norm_term

    def cond_cbrt(x, eps=1e-15):
        '''
        Compute cubic root of x based on sign
        '''
        ret = torch.zeros_like(x)
        ret[x >= 0] = torch.pow(torch.clamp_min_(x[x >= 0], eps), 1/3.)
        ret[x < 0] = torch.pow(torch.clamp_min_(-x[x < 0], eps), 1/3.) * -1
        return ret

    Q = ((b**2) - 3.*c) / 9.
    R = (2.*(b**3) - 9.*b*c + 27.*d) /54.

    # # all three roots are real and equal
    # _mask1 = ((f == 0) & (g == 0) & (h == 0))
    # ts[cubic_ids[_mask1], 0] = cond_cbrt(d[_mask1]/a[_mask1])

    # all three roots are real 
    _mask2 = (R)**2 < (Q)**3
    _b, _Q, _R = b[_mask2], Q[_mask2], R[_mask2]

    eps = 1e-10
    # theta = torch.acos(torch.clamp(_R / torch.sqrt((_Q)**3), -1+eps, 1-eps))
    theta = torch.acos((_R / torch.sqrt((_Q)**3)))
    
    ts[cubic_ids[_mask2], 0] = -2. * torch.sqrt(_Q) * torch.cos(theta/3.) - _b/3.
    ts[cubic_ids[_mask2], 1] = -2. * torch.sqrt(_Q) * torch.cos((theta - 2.*torch.pi)/3.) - _b/3.
    ts[cubic_ids[_mask2], 2] = -2. * torch.sqrt(_Q) * torch.cos((theta + 2.*torch.pi)/3.) - _b/3.

    # only one root is real
    _mask3 = ~_mask2
    __b, __Q, __R = b[_mask3], Q[_mask3], R[_mask3]

    # A = -torch.sign(__R) * (torch.abs(__R) + torch.sqrt(torch.clamp_min((__R)**2 - (__Q)**3, 1e-8))) ** (1./3.)
    A = -torch.sign(__R) * (torch.abs(__R) + torch.sqrt(((__R)**2 - (__Q)**3))) ** (1./3.)
    _B = __Q/A
    _B[A== 0.] = 0.

    ts[cubic_ids[_mask3], 0] = (A+_B) - __b/3.


    assert not torch.isnan(ts).any(), 'NaN detcted in cubic roots'
    assert torch.isfinite(ts).all(), 'Inf detcted in cubic roots'

    ts.retain_grad()
    _b.retain_grad()
    theta.retain_grad()
    _Q.retain_grad()
    _R.retain_grad()
    a.retain_grad()
    b.retain_grad()
    c.retain_grad()
    d.retain_grad()
    # D.retain_grad()

    loss = torch.sum(ts[torch.arange(ts.shape[0], device=ts.device), st_ids])
    loss.backward()

    return ts



def test_cubic_root_grad():
    device = 'cuda'
    dtype = torch.float32
    N_RAYS = 10000
    fs = torch.randn((N_RAYS, 4), device=device, dtype=dtype)
    # IDX = torch.tensor([800], dtype=torch.long, device=device) # st_id 1

    fs = np.load('./fs.npy')
    fs = torch.tensor(fs, device=device, dtype=dtype)
    IDX = torch.tensor(torch.arange(0,40), dtype=torch.long, device=device) # 
    IDX = torch.tensor([2], dtype=torch.long, device=device) # 
    fs = fs[IDX,:].clone()
    fs.requires_grad = True

    st_ids = torch.zeros_like(fs[:,0]).long() + 1

    ts_cu = torch.ones([N_RAYS, 3]).to(dtype).to(device=fs.device) * -1

    # run cuda
    fs_grad_cu = fs.clone()
    fs_grad_cu[:] = 1.
    fs_grad_cu = _C.test_cubic_root_grad(fs.to(torch.float64), st_ids.to(torch.int32), fs_grad_cu, ts_cu, True)


    # run pytorch
    # fs.grad[:] = 0.
    ts_py = cubic_root_grad_py(fs, st_ids)

    fs_grad_py = fs.grad

    abs_error = torch.abs(fs_grad_py - fs_grad_cu)

    print('###### Cubic Root Grad Checking ######')
    print(f'Max error: {abs_error.max()}')
    print(f'Mean error: {abs_error.mean()}')
    print(f'Sign check: {grad_sign_check(fs_grad_py,fs_grad_cu).all()}')
    print()


def test_surface_norm_grad():
    device = 'cuda:0'
    surface_type = svox2.SURFACE_TYPE_SDF
    grid = svox2.SparseGrid(
                        reso=16, #[16,32,64],
                        center=[0.0, 0.0, 0.0],
                        radius=[1.0, 1.0, 1.0],
                        basis_dim=9,
                        use_z_order=True,
                        device=device,
                        background_nlayers=0,
                        basis_type=svox2.BASIS_TYPE_SH,
                        surface_type=surface_type,
                        use_sphere_bound=False,
                        surface_init='sphere')

    grid.surface_data.data += torch.randn_like(grid.surface_data.data)

    sparse_frac = 0.9

    rand_cells = grid._get_rand_cells(sparse_frac, contiguous=True)
    # rand_cells = torch.arange(1335, 1347, dtype=torch.int32, device=device)
    # rand_cells = torch.tensor([1335], dtype=torch.int32, device=device)

    scaling = 1
    eikonal_scale = 0.
    ndc_coeffs = [1,1]

    check_con = False
    ignore_empty = False
    use_l1 = True

    # cuda grad
    grad_cu = torch.zeros_like(grid.surface_data)
    _C.surface_normal_grad_sparse(grid.links, grid.surface_data,
            rand_cells,
            grid._get_sparse_grad_indexer(),
            grid.level_set_data[0],
            0, 1, scaling, eikonal_scale,
            ndc_coeffs[0], ndc_coeffs[1],
            check_con, ignore_empty, use_l1,
            grad_cu)

    # pytorch grad
    loss = grid._surface_normal_loss_grad_check(rand_cells, scaling, connectivity_check=check_con, ignore_empty=ignore_empty, use_l1=use_l1)

    grad_py = grid.surface_data.grad.clone()

    abs_error = torch.abs(grad_py - grad_cu)
    print('###### Cubic Root Grad Checking ######')
    print(f'Max error: {abs_error.max()}')
    print(f'Max error percent: {abs_error.max() / torch.abs(grad_py).max()}')
    print(f'Mean error: {abs_error.mean()}')
    print(f'Sign check: {grad_sign_check(grad_py,grad_cu).all()}')
    print()

    
    pass


def test_surface_eikonal_grad():
    device = 'cuda:0'
    surface_type = svox2.SURFACE_TYPE_SDF
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
                        surface_init='sphere')

    sparse_frac = 0.9

    rand_cells = grid._get_rand_cells(sparse_frac, contiguous=True)
    # rand_cells = torch.arange(0, 100, dtype=torch.int32, device=device)
    # rand_cells = torch.tensor([10,20], dtype=torch.int32, device=device)

    scaling = 0.
    eikonal_scale = 1.
    ndc_coeffs = [1,1]

    # cuda grad
    # grad_cu = torch.zeros_like(grid.surface_data)
    # _C.surface_normal_grad_sparse(grid.links, grid.surface_data,
    #         rand_cells,
    #         grid._get_sparse_grad_indexer(),
    #         grid.level_set_data[0],
    #         0, 1, scaling, eikonal_scale,
    #         ndc_coeffs[0], ndc_coeffs[1],
    #         grad_cu)

    # pytorch grad
    loss = grid._surface_eikonal_loss_grad_check(rand_cells, eikonal_scale)

    grad_py = grid.surface_data.grad.clone()

    abs_error = torch.abs(grad_py - grad_cu)
    print('###### Cubic Root Grad Checking ######')
    print(f'Max error: {abs_error.max()}')
    print(f'Mean error: {abs_error.mean()}')
    print(f'Sign check: {grad_sign_check(grad_py,grad_cu).all()}')
    print()

    
    pass

def test_alpha_surf_sparsify_grad():
    device = 'cuda:0'
    surface_type = svox2.SURFACE_TYPE_SDF
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
                        surface_init='sphere')

    sparse_frac = 0.9

    grid.density_data.data[:].normal_(mean=-0.5, std=0.5)

    rand_cells = grid._get_rand_cells(sparse_frac, contiguous=True)
    # rand_cells = torch.arange(0, 100, dtype=torch.int32, device=device)
    # rand_cells = torch.tensor([10,20], dtype=torch.int32, device=device)

    scale_alpha = 1.
    scale_surf = 1.

    surf_thresh = 1 # this is raw alpha value

    surf_decrease = False

    # cuda grad
    grad_alpha_cu = torch.zeros_like(grid.surface_data)
    grad_surf_cu = torch.zeros_like(grid.surface_data)
    _C.alpha_surf_sparsify_grad_sparse(grid.links, 
            grid.density_data,
            grid.surface_data,
            rand_cells,
            grid._get_sparse_grad_indexer(),
            scale_alpha, scale_surf,
            surf_decrease,
            surf_thresh,
            grad_alpha_cu, grad_surf_cu)

# 1. (arg0: at::Tensor, arg1: at::Tensor, arg2: at::Tensor, arg3: at::Tensor, arg4: float, 
# arg5: int, arg6: int, arg7: float, arg8: float, arg9: float, arg10: float, arg11: bool, arg12: bool, arg13: at::Tensor) -> None

    # pytorch grad
    loss = grid._alpha_surf_sparsify_grad_check(rand_cells, scale_alpha, scale_surf, surf_decrease, surf_thresh)

    grad_alpha_py = grid.density_data.grad.clone()
    grad_surf_py = grid.surface_data.grad.clone()

    abs_error = torch.abs(grad_alpha_py - grad_alpha_cu)
    print('###### Alpha Grad Checking ######')
    print(f'Max error: {abs_error.max()}')
    print(f'Mean error: {abs_error.mean()}')
    # print(f'Sign check: {grad_sign_check(grad_py,grad_cu).all()}')
    print()

    abs_error_surf = torch.abs(grad_surf_py - grad_surf_cu)
    print('###### Surf Grad Checking ######')
    print(f'Max error: {abs_error_surf.max()}')
    print(f'Mean error: {abs_error_surf.mean()}')
    # print(f'Sign check: {grad_sign_check(grad_py,grad_cu).all()}')
    print()

    
    pass



def test_sign_change_grad():
    device = 'cuda:0'
    surface_type = svox2.SURFACE_TYPE_SDF
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
                        surface_init='sphere')

    sparse_frac = 0.5

    rand_cells = grid._get_rand_cells(sparse_frac, contiguous=True)
    # rand_cells = torch.arange(0, 100, dtype=torch.int32, device=device)
    # rand_cells = torch.tensor([10,20], dtype=torch.int32, device=device)

    scaling = 10

    # cuda grad
    grad_cu = torch.zeros_like(grid.surface_data)
    _C.surf_sign_change_grad_sparse(grid.links, grid.surface_data,
            rand_cells,
            grid._get_sparse_grad_indexer(),
            0, 1, scaling,
            grad_cu)

    # pytorch grad
    loss = grid._surface_sign_change_grad_check(rand_cells, scaling)

    grad_py = grid.surface_data.grad.clone()

    abs_error = torch.abs(grad_py - grad_cu)
    print('###### Cubic Root Grad Checking ######')
    print(f'Max error: {abs_error.max()}')
    print(f'Mean error: {abs_error.mean()}')
    print(f'Sign check: {grad_sign_check(grad_py,grad_cu).all()}')
    print()

    
    pass



test_surface_eikonal_grad()
# test_sign_change_grad()
# test_surface_norm_grad()
# test_cubic_root_grad()

# test_alpha_surf_sparsify_grad()