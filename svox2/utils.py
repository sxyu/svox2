from functools import partial
import torch
from torch import nn
from typing import Optional, Tuple
import numpy as np
from dataclasses import dataclass
import math

def inthroot(x : int, n : int):
    if x <= 0:
        return None
    lo, hi = 1, x
    while lo <= hi:
        mi = lo + (hi - lo) // 2
        p = mi ** n
        if p == x:
            return mi
        elif p > x:
            hi = mi - 1
        else:
            lo = mi + 1
    return None

isqrt = partial(inthroot, n=2)

def is_pow2(x : int):
    return x > 0 and (x & (x - 1)) == 0

def _get_c_extension():
    from warnings import warn
    try:
        import svox2.csrc as _C
        if not hasattr(_C, "sample_grid"):
            _C = None
    except:
        _C = None

    if _C is None:
        warn("CUDA extension svox2.csrc could not be loaded! " +
             "Operations will be slow.\n" +
             "Please do not import svox in the svox2 source directory.")
    return _C


# Morton code (Z-order curve)
def _expand_bits(v):
    v = (v | (v << 16)) & 0x030000FF
    v = (v | (v <<  8)) & 0x0300F00F
    v = (v | (v <<  4)) & 0x030C30C3
    v = (v | (v <<  2)) & 0x09249249
    return v

def _unexpand_bits(v):
    v &= 0x49249249
    v = (v | (v >> 2)) & 0xc30c30c3
    v = (v | (v >> 4)) & 0xf00f00f
    v = (v | (v >> 8)) & 0xff0000ff
    v = (v | (v >> 16)) & 0x0000ffff
    return v


def morton_code_3(x, y, z):
    xx = _expand_bits(x)
    yy = _expand_bits(y)
    zz = _expand_bits(z)
    return (xx << 2) + (yy << 1) + zz

def inv_morton_code_3(code):
    x = _unexpand_bits(code >> 2)
    y = _unexpand_bits(code >> 1)
    z = _unexpand_bits(code)
    return x, y, z

def gen_morton(D, device='cpu', dtype=torch.long):
    assert is_pow2(D), "Morton code requires power of 2 reso"
    arr = torch.arange(D, device=device, dtype=dtype)
    X, Y, Z = torch.meshgrid(arr, arr, arr)
    mort = morton_code_3(X, Y, Z)
    return mort


# SH

SH_C0 = 0.28209479177387814
SH_C1 = 0.4886025119029199
SH_C2 = [
    1.0925484305920792,
    -1.0925484305920792,
    0.31539156525252005,
    -1.0925484305920792,
    0.5462742152960396
]
SH_C3 = [
    -0.5900435899266435,
    2.890611442640554,
    -0.4570457994644658,
    0.3731763325901154,
    -0.4570457994644658,
    1.445305721320277,
    -0.5900435899266435
]
SH_C4 = [
    2.5033429417967046,
    -1.7701307697799304,
    0.9461746957575601,
    -0.6690465435572892,
    0.10578554691520431,
    -0.6690465435572892,
    0.47308734787878004,
    -1.7701307697799304,
    0.6258357354491761,
]

MAX_SH_BASIS = 10
def eval_sh_bases(basis_dim : int, dirs : torch.Tensor):
    """
    Evaluate spherical harmonics bases at unit directions,
    without taking linear combination.
    At each point, the final result may the be
    obtained through simple multiplication.

    :param basis_dim: int SH basis dim. Currently, 1-25 square numbers supported
    :param dirs: torch.Tensor (..., 3) unit directions

    :return: torch.Tensor (..., basis_dim)
    """
    result = torch.empty((*dirs.shape[:-1], basis_dim), dtype=dirs.dtype, device=dirs.device)
    result[..., 0] = SH_C0
    if basis_dim > 1:
        x, y, z = dirs.unbind(-1)
        result[..., 1] = -SH_C1 * y;
        result[..., 2] = SH_C1 * z;
        result[..., 3] = -SH_C1 * x;
        if basis_dim > 4:
            xx, yy, zz = x * x, y * y, z * z
            xy, yz, xz = x * y, y * z, x * z
            result[..., 4] = SH_C2[0] * xy;
            result[..., 5] = SH_C2[1] * yz;
            result[..., 6] = SH_C2[2] * (2.0 * zz - xx - yy);
            result[..., 7] = SH_C2[3] * xz;
            result[..., 8] = SH_C2[4] * (xx - yy);

            if basis_dim > 9:
                result[..., 9] = SH_C3[0] * y * (3 * xx - yy);
                result[..., 10] = SH_C3[1] * xy * z;
                result[..., 11] = SH_C3[2] * y * (4 * zz - xx - yy);
                result[..., 12] = SH_C3[3] * z * (2 * zz - 3 * xx - 3 * yy);
                result[..., 13] = SH_C3[4] * x * (4 * zz - xx - yy);
                result[..., 14] = SH_C3[5] * z * (xx - yy);
                result[..., 15] = SH_C3[6] * x * (xx - 3 * yy);

                if basis_dim > 16:
                    result[..., 16] = SH_C4[0] * xy * (xx - yy);
                    result[..., 17] = SH_C4[1] * yz * (3 * xx - yy);
                    result[..., 18] = SH_C4[2] * xy * (7 * zz - 1);
                    result[..., 19] = SH_C4[3] * yz * (7 * zz - 3);
                    result[..., 20] = SH_C4[4] * (zz * (35 * zz - 30) + 3);
                    result[..., 21] = SH_C4[5] * xz * (7 * zz - 3);
                    result[..., 22] = SH_C4[6] * (xx - yy) * (7 * zz - 1);
                    result[..., 23] = SH_C4[7] * xz * (xx - 3 * yy);
                    result[..., 24] = SH_C4[8] * (xx * (xx - 3 * yy) - yy * (3 * xx - yy));
    return result

# BEGIN Cubemaps
@dataclass
class CubemapCoord:
    ax : torch.Tensor
    ori : torch.Tensor
    u : torch.Tensor
    v : torch.Tensor

    def query_in(self, cubemap : torch.Tensor):
        face = self.ax * 2 + self.ori
        #  print(cubemap.shape, face.min(), face.max(), ' ',
        #        self.u.min(), self.u.max(),
        if cubemap.ndim == 4:
            return cubemap[face, self.u, self.v]
        else:
            return cubemap[torch.arange(face.size(0), device=face.device), face, self.u, self.v]

    def clone(self):
        return CubemapCoord(
                self.ax.clone(),
                self.ori.clone(),
                self.u.clone(),
                self.v.clone())

@dataclass
class CubemapBilerpQuery:
    i00: CubemapCoord
    i01: CubemapCoord
    i10: CubemapCoord
    i11: CubemapCoord
    du: torch.Tensor
    dv: torch.Tensor


def dir_to_cubemap_coord(xyz : torch.Tensor,
                   face_reso : int,
                   eac : bool = True) -> CubemapCoord:
    """
    Convert a direction on a sphere (not necessarily normalized)

    :param xyz: direction (not necessarily normalized)
    :param face_reso: int, resolution of cubemap face
    :param eac: bool, if true (default) then uses equi-angular cubemaps (EAC)
                      instead of standard cubemap; see
      https://blog.google/products/google-ar-vr/bringing-pixels-front-and-center-vr-video/

    :return: CubemapCoord
    """
    xyz = xyz.float()
    maxv, ax = torch.max(torch.abs(xyz), dim=-1)
    xyz = xyz * (1.0 / maxv.unsqueeze(-1))
    if eac:
        xyz_eac = torch.atan(xyz) * (4 / math.pi)
    else:
        xyz_eac = xyz

    arr = torch.arange(ax.size(0))
    ud = (ax ^ 1) & 1
    vd = (ax ^ 2) & 2
    u_eac = xyz_eac[arr, ud]
    v_eac = xyz_eac[arr, vd]

    ori = (xyz_eac[arr, ax] >= 0).long()

    u = ((u_eac + 1) * face_reso - 1.0) * 0.5
    v = ((v_eac + 1) * face_reso - 1.0) * 0.5
    return CubemapCoord(ax, ori, u, v)

def cubemap_build_query(idx : CubemapCoord, face_reso : int,
                        mode : str = 'linear') ->  CubemapBilerpQuery:
    """
    Compute the points on the cubemap for bilinear sampling
    given a cubemap coordinate from dir_to_cubemap_coord;
    to be used with cubemap_sample.

    :param idx: CubemapCoord, cube map coordinate from dir_to_cubemap_coord
    :param face_reso: int, resolution of cubemap face
    :param mode: str, interpolation mode; one of nearest, linear_simple, linear;
                      linear_simple interpolates per-face, while linear also
                      interpolates across edges (this is the only one supported
                      in CUDA)

    :return: CubemapBilerpQuery
    """
    if mode == 'nearest':
        uf = torch.floor(idx.u + 0.5).long().clamp_(0, face_reso - 1)
        vf = torch.floor(idx.v + 0.5).long().clamp_(0, face_reso - 1)
        idx_ul = CubemapCoord(idx.ax, idx.ori, uf, vf)
        # Corner: triple average
        return CubemapBilerpQuery(idx_ul, idx_ul, idx_ul, idx_ul,
                    torch.zeros_like(idx.u), torch.zeros_like(idx.v))
    elif mode == 'linear_simple':
        u = idx.u.clamp(0, face_reso - 2)
        v = idx.v.clamp(0, face_reso - 2)
        uf = torch.floor(u).long()
        vf = torch.floor(v).long()
        uc = uf + 1
        vc = vf + 1
        du = u - uf
        dv = v - vf
        return CubemapBilerpQuery(
                    CubemapCoord(idx.ax, idx.ori, uf, vf),
                    CubemapCoord(idx.ax, idx.ori, uf, vc),
                    CubemapCoord(idx.ax, idx.ori, uc, vf),
                    CubemapCoord(idx.ax, idx.ori, uc, vc),
                    du, dv)
    elif mode == 'linear':
        uf = torch.floor(idx.u).long()
        vf = torch.floor(idx.v).long()
        uc = uf + 1
        vc = vf + 1

        m0u = uf < 0
        m0v = vf < 0
        m1u = uc > (face_reso - 1)
        m1v = vc > (face_reso - 1)

        ud = (idx.ax ^ 1) & 1
        vd = (idx.ax ^ 2) & 2

        def _index_across_sides(nidx : CubemapCoord, uori, vori, mu, mv):
            mdiagonal = mu & mv
            # FIXME not quite correct (matches CUDA impl)
            mu = mu & (~mdiagonal)
            mv = mv & (~mdiagonal)
            nidx.u[mdiagonal] = nidx.u[mdiagonal].clamp(0, face_reso - 1)
            nidx.v[mdiagonal] = nidx.v[mdiagonal].clamp(0, face_reso - 1)

            def _index_across_one_side(mask, d, ori, other_coord):
                nax = d[mask]

                nud = (nax ^ 1) & 1
                #  nvd = (nax ^ 2) & 2

                ax_is_u = nud == nidx.ax[mask]
                ax_is_v = ~ax_is_u

                ax_is_u_m = torch.zeros_like(mask)
                ax_is_u_m[mask] = ax_is_u

                ax_is_v_m = torch.zeros_like(mask)
                ax_is_v_m[mask] = ax_is_v

                nidx.u[ax_is_v_m] = other_coord[ax_is_v_m]
                nidx.v[ax_is_u_m] = other_coord[ax_is_u_m]

                nidx.u[ax_is_u_m] = nidx.ori[ax_is_u_m] * (face_reso - 1)
                nidx.v[ax_is_v_m] = nidx.ori[ax_is_v_m] * (face_reso - 1)

                nidx.ax[mask] = nax
                nidx.ori[mask] = ori

            _index_across_one_side(mu, ud, uori, nidx.v)
            _index_across_one_side(mv, vd, vori, nidx.u)
            return nidx

        i00 = _index_across_sides(CubemapCoord(idx.ax, idx.ori, uf, vf).clone(),
                0, 0, m0u, m0v)
        i01 = _index_across_sides(CubemapCoord(idx.ax, idx.ori, uf, vc).clone(),
                0, 1, m0u, m1v)
        i10 = _index_across_sides(CubemapCoord(idx.ax, idx.ori, uc, vf).clone(),
                1, 0, m1u, m0v)
        i11 = _index_across_sides(CubemapCoord(idx.ax, idx.ori, uc, vc).clone(),
                1, 1, m1u, m1v)

        du = idx.u - uf
        dv = idx.v - vf
        return CubemapBilerpQuery(
                    i00,
                    i01,
                    i10,
                    i11,
                    du,
                    dv)
    else:
        raise NotImplementedError()


def cubemap_sample(cubemap: torch.Tensor, idx4 : CubemapBilerpQuery):
    """
    Perform bilinear sampling on a cubemap given a query from cubemap_build_query

    :param cubemap: torch.Tensor float
            (6, face_reso, face_reso, C)
            or
            (B, 6, face_reso, face_reso, C)
    :param idx4: CubemapBilerpQuery from cubemap_build_query where
                 each tensor has batch size B

    :return: (B, C)
    """
    face_reso = cubemap.size(2)
    v00 = idx4.i00.query_in(cubemap)
    v01 = idx4.i01.query_in(cubemap)
    v10 = idx4.i10.query_in(cubemap)
    v11 = idx4.i11.query_in(cubemap)
    du = idx4.du.view([-1] + (v00.dim() - 1) * [1])
    dv = idx4.dv.view([-1] + (v00.dim() - 1) * [1])

    r0 = v00 * (1 - dv) + v01 * dv
    r1 = v10 * (1 - dv) + v11 * dv
    return r0 * (1 - du) + r1 * du

# END Cubemaps

# Ray-sphere intersector for MSI
class ConcentricSpheresIntersector:
    def __init__(self,
            size : torch.Tensor,
            rorigins : torch.Tensor,
            rdirs : torch.Tensor,
            rworld_step : torch.Tensor):
        sphere_scaling = 2.0 / size

        origins = (rorigins + 0.5) * sphere_scaling - 1.0
        dirs = rdirs * sphere_scaling
        inorm = 1.0 / dirs.norm(dim=-1)

        self.world_step_scale = rworld_step * inorm
        dirs *= inorm.unsqueeze(-1)

        self.q2a : torch.Tensor = 2 * (dirs * dirs).sum(-1)
        self.qb : torch.Tensor = 2 * (origins * dirs).sum(-1)
        self.f = self.qb.square() - 2 * self.q2a * (origins * origins).sum(-1)
        self.origins = origins
        self.dirs = dirs

    def intersect(self, r : float):
        """
        Find far intersection of all rays with sphere of radius r
        """
        det = self._det(r)
        success_mask = det >= 0
        result = torch.zeros_like(self.q2a)
        result[success_mask] = (-self.qb[success_mask] +
                torch.sqrt(det[success_mask])) / self.q2a[success_mask]
        return success_mask, result

    def intersect_near(self, r : float):
        """
        Find near intersection of all rays with sphere of radius r
        """
        det = self._det(r)
        success_mask = det >= 0
        result = torch.zeros_like(self.q2a)
        result[success_mask] = (-self.qb[success_mask] -
                torch.sqrt(det[success_mask])) / self.q2a[success_mask]
        return success_mask, result

    def _det(self, r : float) -> torch.Tensor:
        return self.f + 2 * self.q2a * (r * r)


def memlog(device='cuda'):
    # Memory debugging
    print(torch.cuda.memory_summary(device))
    import gc
    for obj in gc.get_objects():
        try:
            if torch.is_tensor(obj) or (
                    hasattr(obj, 'data') and torch.is_tensor(obj.data)):
                if str(obj.device) != 'cpu':
                    print(obj.device, '{: 10}'.format(obj.numel()),
                            obj.dtype,
                            obj.size(), type(obj))
        except:
            pass


def spher2cart(theta : torch.Tensor, phi : torch.Tensor):
    """Convert spherical coordinates into Cartesian coordinates on unit sphere."""
    x = torch.sin(theta) * torch.cos(phi)
    y = torch.sin(theta) * torch.sin(phi)
    z = torch.cos(theta)
    return torch.stack([x, y, z], dim=-1)

def eval_sg_at_dirs(sg_lambda : torch.Tensor, sg_mu : torch.Tensor, dirs : torch.Tensor):
    """
    Evaluate spherical Gaussian functions at unit directions
    using learnable SG basis,
    without taking linear combination
    Works with torch.
    ... Can be 0 or more batch dimensions.
    N is the number of SG basis we use.
    :math:`Output = \sigma_{i}{exp ^ {\lambda_i * (\dot(\mu_i, \dirs) - 1)}`

    :param sg_lambda: The sharpness of the SG lobes. (N), positive
    :param sg_mu: The directions of the SG lobes. (N, 3), unit vector
    :param dirs: jnp.ndarray unit directions (..., 3)

    :return: (..., N)
    """
    product = torch.einsum(
        "ij,...j->...i", sg_mu, dirs)  # [..., N]
    basis = torch.exp(torch.einsum(
        "i,...i->...i", sg_lambda, product - 1))  # [..., N]
    return basis

def init_weights(m):
    if type(m) == nn.Linear:
        nn.init.xavier_uniform_(m.weight)
        m.bias.data.fill_(0.0)


def cross_broadcast(x : torch.Tensor, y : torch.Tensor):
    """
    Cross broadcasting for 2 tensors

    :param x: torch.Tensor
    :param y: torch.Tensor, should have the same ndim as x
    :return: tuple of cross-broadcasted tensors x, y. Any dimension where the size of x or y is 1
             is expanded to the maximum size in that dimension among the 2.
             Formally, say the shape of x is (a1, ... an)
             and of y is (b1, ... bn);
             then the result has shape (a'1, ... a'n), (b'1, ... b'n)
             where
                :code:`a'i = ai if (ai > 1 and bi > 1) else max(ai, bi)`
                :code:`b'i = bi if (ai > 1 and bi > 1) else max(ai, bi)`
    """
    assert x.ndim == y.ndim, "Only available if ndim is same for all tensors"
    max_shape = [(-1 if (a > 1 and b > 1) else max(a,b)) for i, (a, b)
                    in enumerate(zip(x.shape, y.shape))]
    shape_x = [max(a, m) for m, a in zip(max_shape, x.shape)]
    shape_y = [max(b, m) for m, b in zip(max_shape, y.shape)]
    x = x.broadcast_to(shape_x)
    y = y.broadcast_to(shape_y)
    return x, y

def posenc(
    x: torch.Tensor,
    cov_diag: Optional[torch.Tensor],
    min_deg: int,
    max_deg: int,
    include_identity: bool = True,
    enable_ipe: bool = True,
    cutoff: float = 1.0,
):
    """
    Positional encoding function. Adapted from jaxNeRF
    (https://github.com/google-research/google-research/tree/master/jaxnerf).
    With support for mip-NeFF IPE (by passing cov_diag != 0, keeping enable_ipe=True).
    And BARF-nerfies frequency attenuation (setting cutoff)

    Cat x with a positional encoding of x with scales 2^[min_deg, max_deg-1],
    Instead of computing [sin(x), cos(x)], we use the trig identity
    cos(x) = sin(x + pi/2) and do one vectorized call to sin([x, x+pi/2]).

    :param x: torch.Tensor (..., D), variables to be encoded. Note that x should be in [-pi, pi].
    :param cov_diag: torch.Tensor (..., D), diagonal cov for each variable to be encoded (IPE)
    :param min_deg: int, the minimum (inclusive) degree of the encoding.
    :param max_deg: int, the maximum (exclusive) degree of the encoding. if min_deg >= max_deg,
                         positional encoding is disabled.
    :param include_identity: bool, if true then concatenates the identity
    :param enable_ipe: bool, if true then uses cov_diag to compute IPE, if available.
                             Note cov_diag = 0 will give the same effect.
    :param cutoff: float, in [0, 1], a relative frequency cutoff as in BARF/nerfies. 1 = all frequencies,
                          0 = no frequencies

    :return: encoded torch.Tensor (..., D * (max_deg - min_deg) * 2 [+ D if include_identity]),
                     encoded variables.
    """
    if min_deg >= max_deg:
        return x
    scales = torch.tensor([2 ** i for i in range(min_deg, max_deg)], device=x.device)
    half_enc_dim = x.shape[-1] * scales.shape[0]
    shapeb = list(x.shape[:-1]) + [half_enc_dim]  # (..., D * (max_deg - min_deg))
    xb = torch.reshape((x[..., None, :] * scales[:, None]), shapeb)
    four_feat = torch.sin(
        torch.cat([xb, xb + 0.5 * np.pi], dim=-1)
    )  # (..., D * (max_deg - min_deg) * 2)
    if enable_ipe and cov_diag is not None:
        # Apply integrated positional encoding (IPE)
        xb_var = torch.reshape((cov_diag[..., None, :] * scales[:, None] ** 2), shapeb)
        xb_var = torch.tile(xb_var, (2,))  # (..., D * (max_deg - min_deg) * 2)
        four_feat = four_feat * torch.exp(-0.5 * xb_var)
    if cutoff < 1.0:
        # BARF/nerfies, could be made cleaner
        cutoff_mask = _cutoff_mask(
            scales, cutoff * (max_deg - min_deg)
        )  # (max_deg - min_deg,)
        four_feat = four_feat.view(shapeb[:-1] + [2, scales.shape[0], x.shape[-1]])
        four_feat = four_feat * cutoff_mask[..., None]
        four_feat = four_feat.view(shapeb[:-1] + [2 * scales.shape[0] * x.shape[-1]])
    if include_identity:
        four_feat = torch.cat([x] + [four_feat], dim=-1)
    return four_feat


def net_to_dict(out_dict : dict,
                prefix : str,
                model : nn.Module):
    for child in model.named_children():
        layer_name = child[0]
        layer_params = {}
        for param in child[1].named_parameters():
            param_name = param[0]
            param_value = param[1].data.cpu().numpy()
            out_dict['pt__' + prefix + '__' + layer_name + '__' + param_name] = param_value

def net_from_dict(in_dict,
                  prefix : str,
                  model : nn.Module):
    for child in model.named_children():
        layer_name = child[0]
        layer_params = {}
        for param in child[1].named_parameters():
            param_name = param[0]
            value = in_dict['pt__' + prefix + '__' + layer_name + '__' + param_name]
            param_value = param[1].data[:] = torch.from_numpy(value).to(
                    device=param[1].data.device)


def convert_to_ndc(origins, directions, ndc_coeffs, near: float = 1.0):
    """Convert a set of rays to NDC coordinates."""
    # Shift ray origins to near plane, not sure if needed
    t = (near - origins[Ellipsis, 2]) / directions[Ellipsis, 2]
    origins = origins + t[Ellipsis, None] * directions

    dx, dy, dz = directions.unbind(-1)
    ox, oy, oz = origins.unbind(-1)

    # Projection
    o0 = ndc_coeffs[0] * (ox / oz)
    o1 = ndc_coeffs[1] * (oy / oz)
    o2 = 1 - 2 * near / oz

    d0 = ndc_coeffs[0] * (dx / dz - ox / oz)
    d1 = ndc_coeffs[1] * (dy / dz - oy / oz)
    d2 = 2 * near / oz;

    origins = torch.stack([o0, o1, o2], -1)
    directions = torch.stack([d0, d1, d2], -1)
    return origins, directions


def xyz2equirect(bearings, reso):
    """
    Convert ray direction vectors into equirectangular pixel coordinates.
    Inverse of equirect2xyz.
    Taken from Vickie Ye
    """
    lat = torch.asin(bearings[..., 1])
    lon = torch.atan2(bearings[..., 0], bearings[..., 2])
    x = reso * 2 * (0.5 + lon / 2 / np.pi)
    y = reso * (0.5 - lat / np.pi)
    return torch.stack([x, y], dim=-1)

class Timing:
    """
    Timing environment
    usage:
    with Timing("message"):
        your commands here
    will print CUDA runtime in ms
    """

    def __init__(self, name):
        self.name = name

    def __enter__(self):
        self.start = torch.cuda.Event(enable_timing=True)
        self.end = torch.cuda.Event(enable_timing=True)
        self.start.record()

    def __exit__(self, type, value, traceback):
        self.end.record()
        torch.cuda.synchronize()
        print(self.name, "elapsed", self.start.elapsed_time(self.end), "ms")

