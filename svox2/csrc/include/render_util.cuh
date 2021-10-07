// Copyright 2021 Alex Yu
#pragma once

#include "cuda_util.cuh"
#include <cstdint>

namespace {
namespace device {

// SH Coefficients from https://github.com/google/spherical-harmonics
__device__ __constant__ const float C0 = 0.28209479177387814;
__device__ __constant__ const float C1 = 0.4886025119029199;
__device__ __constant__ const float C2[] = {
    1.0925484305920792,
    -1.0925484305920792,
    0.31539156525252005,
    -1.0925484305920792,
    0.5462742152960396
};

__device__ __constant__ const float C3[] = {
    -0.5900435899266435,
    2.890611442640554,
    -0.4570457994644658,
    0.3731763325901154,
    -0.4570457994644658,
    1.445305721320277,
    -0.5900435899266435
};

// __device__ __constant__ const float C4[] = {
//     2.5033429417967046,
//     -1.7701307697799304,
//     0.9461746957575601,
//     -0.6690465435572892,
//     0.10578554691520431,
//     -0.6690465435572892,
//     0.47308734787878004,
//     -1.7701307697799304,
//     0.6258357354491761,
// };

__device__ __inline__ void calc_sh(
    const int basis_dim,
    const float* __restrict__ dir,
    float* __restrict__ out) {
    out[0] = C0;
    const float x = dir[0], y = dir[1], z = dir[2];
    const float xx = x * x, yy = y * y, zz = z * z;
    const float xy = x * y, yz = y * z, xz = x * z;
    switch (basis_dim) {
        case 16:
            out[9] = C3[0] * y * (3 * xx - yy);
            out[10] = C3[1] * xy * z;
            out[11] = C3[2] * y * (4 * zz - xx - yy);
            out[12] = C3[3] * z * (2 * zz - 3 * xx - 3 * yy);
            out[13] = C3[4] * x * (4 * zz - xx - yy);
            out[14] = C3[5] * z * (xx - yy);
            out[15] = C3[6] * x * (xx - 3 * yy);
            [[fallthrough]];
        case 9:
            out[4] = C2[0] * xy;
            out[5] = C2[1] * yz;
            out[6] = C2[2] * (2.0 * zz - xx - yy);
            out[7] = C2[3] * xz;
            out[8] = C2[4] * (xx - yy);
            [[fallthrough]];
        case 4:
            out[1] = -C1 * y;
            out[2] = C1 * z;
            out[3] = -C1 * x;
    }
}

struct CubemapIndex {
    int32_t face;
    int32_t ui;
    int32_t vi;
    float u;
    float v;
};

__device__ __inline__ void get_cubemap_index(
    const torch::PackedTensorAccessor32<float, 4, torch::RestrictPtrTraits> cubemap,
    const float* __restrict__ dir,
    CubemapIndex* __restrict__ idx) {
    const float x = dir[0], y = dir[1], z = dir[2];
    const float absx = fabsf(x), absy = fabsf(y), absz = fabsf(z);
    const int cubemap_reso = cubemap.size(1);
    float u, v, max_axis;

    if (absx >= absy && absx >= absz) {
        max_axis = absx;
        v = y;
        if (x >= 0) {
            idx->face = 0;
            u = -z;
        } else {
            idx->face = 1;
            u = z;
        }
    } else if (absy >= absz) {
        max_axis = absy;
        u = x;
        if (y >= 0) {
            idx->face = 2;
            v = -z;
        } else {
            idx->face = 3;
            v = z;
        }
    } else {
        max_axis = absz;
        v = y;
        if (z >= 0) {
            idx->face = 4;
            u = x;
        } else {
            idx->face = 5;
            u = -x;
        }
    }

    u = min(max(0.5f * cubemap_reso * (u / max_axis + 1.f) - 0.5f, 0.f), cubemap_reso - 1.0f);
    v = min(max(0.5f * cubemap_reso * (v / max_axis + 1.f) - 0.5f, 0.f), cubemap_reso - 1.0f);
    idx->ui = min((int32_t) u, cubemap_reso - 2);
    idx->vi = min((int32_t) v, cubemap_reso - 2);
    idx->u = u - idx->ui;
    idx->v = v - idx->vi;
}

__device__ __inline__ void eval_cubemap(
    const torch::PackedTensorAccessor32<float, 4, torch::RestrictPtrTraits> cubemap,
    const CubemapIndex& __restrict__ idx,
    float* __restrict__ out) {

    const float* __restrict__ ptr_00 = &cubemap[idx.face][idx.ui][idx.vi][0];
    const float* __restrict__ ptr_01 = &cubemap[idx.face][idx.ui][idx.vi + 1][0];
    const float* __restrict__ ptr_10 = &cubemap[idx.face][idx.ui + 1][idx.vi][0];
    const float* __restrict__ ptr_11 = &cubemap[idx.face][idx.ui + 1][idx.vi + 1][0];
    for (int i = 0; i < cubemap.size(3); ++i) {
        const float l0 = lerp(ptr_00[i], ptr_01[i], idx.v);
        const float l1 = lerp(ptr_10[i], ptr_11[i], idx.v);
        out[i] = lerp(l0, l1, idx.u);
    }
}

__device__ __inline__ void eval_cubemap_backward(
    torch::PackedTensorAccessor32<float, 4, torch::RestrictPtrTraits> grad_cubemap,
    const CubemapIndex& __restrict__ idx,
    const float* __restrict__ grad_out) {

    float* __restrict__ ptr00 = &grad_cubemap[idx.face][idx.ui][idx.vi][0];
    float* __restrict__ ptr01 = &grad_cubemap[idx.face][idx.ui][idx.vi + 1][0];
    float* __restrict__ ptr10 = &grad_cubemap[idx.face][idx.ui + 1][idx.vi][0];
    float* __restrict__ ptr11 = &grad_cubemap[idx.face][idx.ui + 1][idx.vi + 1][0];

    for (int i = 0; i < grad_cubemap.size(3); ++i) {
        const float axo = (1.f - idx.u) * grad_out[i];
        atomicAdd(ptr00 + i, (1.f - idx.v) * axo);
        atomicAdd(ptr01 + i, idx.v * axo);
        const float bxo = idx.u * grad_out[i];
        atomicAdd(ptr10 + i, (1.f - idx.v) * bxo);
        atomicAdd(ptr11 + i, idx.v * bxo);
    }
}

__host__ __device__ __inline__ static float _norm(
                float* dir) {
    return sqrtf(dir[0] * dir[0] + dir[1] * dir[1] + dir[2] * dir[2]);
}

__device__ __inline__ float _intersect_aabb_unit(
        const float* __restrict__ cen,
        const float* __restrict__ invdir) {
    // Intersect unit AABB
    float tmax = 1e9f;
    float t1, t2;
#pragma unroll
    for (int i = 0; i < 3; ++i) {
        t1 = - cen[i] * invdir[i];
        t2 = t1 +  invdir[i];
        tmax = min(tmax, max(t1, t2));
    }
    return tmax;
}

__device__ __inline__ float _get_delta_scale(
    const float* __restrict__ scaling,
    float* __restrict__ dir) {
    dir[0] *= scaling[0];
    dir[1] *= scaling[1];
    dir[2] *= scaling[2];
    float delta_scale = 1.f / _norm(dir);
    dir[0] *= delta_scale;
    dir[1] *= delta_scale;
    dir[2] *= delta_scale;
    return delta_scale;
}

} // namespace device
} // namespace
