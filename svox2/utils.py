from functools import partial
import torch

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

C0 = 0.28209479177387814
C1 = 0.4886025119029199
C2 = [
    1.0925484305920792,
    -1.0925484305920792,
    0.31539156525252005,
    -1.0925484305920792,
    0.5462742152960396
]
C3 = [
    -0.5900435899266435,
    2.890611442640554,
    -0.4570457994644658,
    0.3731763325901154,
    -0.4570457994644658,
    1.445305721320277,
    -0.5900435899266435
]
C4 = [
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

MAX_SH_BASIS = 16
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
    result[..., 0] = C0
    if basis_dim > 1:
        x, y, z = dirs.unbind(-1)
        result[..., 1] = -C1 * y;
        result[..., 2] = C1 * z;
        result[..., 3] = -C1 * x;
        if basis_dim > 4:
            xx, yy, zz = x * x, y * y, z * z
            xy, yz, xz = x * y, y * z, x * z
            result[..., 4] = C2[0] * xy;
            result[..., 5] = C2[1] * yz;
            result[..., 6] = C2[2] * (2.0 * zz - xx - yy);
            result[..., 7] = C2[3] * xz;
            result[..., 8] = C2[4] * (xx - yy);

            if basis_dim > 9:
                result[..., 9] = C3[0] * y * (3 * xx - yy);
                result[..., 10] = C3[1] * xy * z;
                result[..., 11] = C3[2] * y * (4 * zz - xx - yy);
                result[..., 12] = C3[3] * z * (2 * zz - 3 * xx - 3 * yy);
                result[..., 13] = C3[4] * x * (4 * zz - xx - yy);
                result[..., 14] = C3[5] * z * (xx - yy);
                result[..., 15] = C3[6] * x * (xx - 3 * yy);

                if basis_dim > 16:
                    result[..., 16] = C4[0] * xy * (xx - yy);
                    result[..., 17] = C4[1] * yz * (3 * xx - yy);
                    result[..., 18] = C4[2] * xy * (7 * zz - 1);
                    result[..., 19] = C4[3] * yz * (7 * zz - 3);
                    result[..., 20] = C4[4] * (zz * (35 * zz - 30) + 3);
                    result[..., 21] = C4[5] * xz * (7 * zz - 3);
                    result[..., 22] = C4[6] * (xx - yy) * (7 * zz - 1);
                    result[..., 23] = C4[7] * xz * (xx - 3 * yy);
                    result[..., 24] = C4[8] * (xx * (xx - 3 * yy) - yy * (3 * xx - yy));
    return result

# Cubemaps
def cubemap2xyz(index : torch.Tensor, uv : torch.Tensor):
    """
    Cubemap coords to directional vector (nor normalized, max dim=1)

    :param index: integer Tensor (B,), 0-5 for face (x+, x-, y+, y-, z+, z-)
    :param uv: float Tensor (B, 2), [-1, 1]^2 xy coord in each face

    :return: float tensor (B, 3), directions
    """
    xyz = torch.empty((uv.size(0), 3), dtype=uv.dtype, device=uv.device)
    ones = torch.ones_like(uv[:, 0])
    m = index == 0
    xyz[m] = torch.stack([ones[m], uv[m, 1], -uv[m, 0]], dim=-1)
    m = index == 1
    xyz[m] = torch.stack([-ones[m], uv[m, 1], uv[m, 0]], dim=-1)
    m = index == 2
    xyz[m] = torch.stack([uv[m, 0], ones[m], -uv[m, 1]], dim=-1)
    m = index == 3
    xyz[m] = torch.stack([uv[m, 0], -ones[m], uv[m, 1]], dim=-1)
    m = index == 4
    xyz[m] = torch.stack([uv[m, 0], uv[m, 1], ones[m]], dim=-1)
    m = index == 5
    xyz[m] = torch.stack([-uv[m, 0], uv[m, 1], -ones[m]], dim=-1)
    return xyz

def xyz2cubemap(xyz :  torch.Tensor):
    """
    Vector (not necessarily normalized) to cubemap coords

    :param xyz: float tensor (B, 3), directions

    :return: index, long Tensor (B,), 0-5 for face (x+, x-, y+, y-, z+, z-);
             uv, float Tensor (B, 2), [-1, 1]^2 xy coord in each face
    """
    x, y, z = xyz.unbind(-1)
    abs_x = torch.abs(x)
    abs_y = torch.abs(y)
    abs_z = torch.abs(z)
    x_pos = x > 0
    y_pos = y > 0
    z_pos = z > 0

    max_axis = torch.max(torch.max(abs_x, abs_y), abs_z)
    x_max_mask = (abs_x >= abs_y) & (abs_x >= abs_z)
    y_max_mask = (~x_max_mask) & (abs_y >= abs_z)
    z_max_mask = (~x_max_mask) & (~y_max_mask)

    index = torch.empty(x.shape, dtype=torch.long, device=xyz.device)
    uv = torch.empty_like(xyz[:, :2])
    uv[x_max_mask, 0] = z[x_max_mask]
    uv[x_max_mask & x_pos, 0] *= -1
    uv[x_max_mask, 1] = y[x_max_mask]
    index[x_max_mask & x_pos] = 0
    index[x_max_mask & ~x_pos] = 1

    uv[y_max_mask, 0] = x[y_max_mask]
    uv[y_max_mask, 1] = z[y_max_mask]
    uv[y_max_mask & y_pos, 1] *= -1
    index[y_max_mask & y_pos] = 2
    index[y_max_mask & ~y_pos] = 3

    uv[z_max_mask, 0] = x[z_max_mask]
    uv[z_max_mask & ~z_pos, 0] *= -1
    uv[z_max_mask, 1] = y[z_max_mask]
    index[z_max_mask & z_pos] = 4
    index[z_max_mask & ~z_pos] = 5

    uv = uv / max_axis.unsqueeze(-1)
    return index, uv

