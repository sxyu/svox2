// Copyright 2021 Alex Yu
#pragma once

#include <cstdint>
#include "data_spec_packed.cuh"

namespace {
namespace device {

template<class data_type_t, class voxel_index_t>
__device__ __inline__ float trilerp_one(
        const data_type_t* __restrict__ data,
        int reso, int stride,
        const voxel_index_t* __restrict__ l,
        const float* __restrict__ pos,
        const int idx) {
    const int offz = stride;
    const int offy = reso * stride;
    const int offx = reso * offy;
    const data_type_t* __restrict__ data_ptr = data + (offx * l[0] +
                                                    offy * l[1] +
                                                    offz * l[2]
                                                    + idx);

    const float ix0y0 = lerp(data_ptr[0], data_ptr[offz], pos[2]);
    const float ix0y1 = lerp(data_ptr[offy], data_ptr[offy + offz], pos[2]);
    const float ix0 = lerp(ix0y0, ix0y1, pos[1]);
    const float ix1y0 = lerp(data_ptr[offx], data_ptr[offx + offz], pos[2]);
    const float ix1y1 = lerp(data_ptr[offy + offx],
                             data_ptr[offy + offx + offz], pos[2]);
    const float ix1 = lerp(ix1y0, ix1y1, pos[1]);
    return lerp(ix0, ix1, pos[0]);
}

__device__ __inline__ float compute_skip_dist(
        SingleRaySpec& __restrict__ ray,
        const int32_t* __restrict__ links,
        int offx, int offy,
        int pos_offset = 0) {
    const int32_t link_val = links[offx * (ray.l[0] + pos_offset) + offy * (ray.l[1] + pos_offset) +
                                (ray.l[2] + pos_offset)];
    if (link_val >= -1) return 0.f; // Not worth

    const uint32_t dist = -link_val;
    const uint32_t cell_ul_shift = (dist - 1);
    const uint32_t cell_side_len = (1 << cell_ul_shift) - 1.f;

    // AABB intersection
    // Consider caching the invdir for the ray
    float tmin = 0.f;
    float tmax = 1e9f;
#pragma unroll
    for (int i = 0; i < 3; ++i) {
        int ul = (((ray.l[i] + pos_offset) >> cell_ul_shift) << cell_ul_shift);
        ul -= ray.l[i] + pos_offset;

        const float invdir = 1.0 / ray.dir[i];
        const float t1 = (ul - ray.pos[i] + pos_offset) * invdir;
        const float t2 = (ul + cell_side_len - ray.pos[i] + pos_offset) * invdir;
        if (ray.dir[i] != 0.f) {
            tmin = max(tmin, min(t1, t2));
            tmax = min(tmax, max(t1, t2));
        }
    }

//     const uint32_t cell_ul_shift = 1 - dist;
//     const uint32_t cell_br_shift = -cell_ul_shift;
//
//     // AABB intersection
//     // Consider caching the invdir for the ray
//     float tmin = 0.f;
//     float tmax = 1e9f;
// #pragma unroll
//     for (int i = 0; i < 3; ++i) {
//         const float invdir = 1.0 / ray.dir[i];
//         const float t1 = (cell_ul_shift - ray.pos[i] + pos_offset) * invdir;
//         const float t2 = (cell_br_shift - ray.pos[i] + pos_offset) * invdir;
//         if (ray.dir[i] != 0.f) {
//             tmin = max(tmin, min(t1, t2));
//             tmax = min(tmax, max(t1, t2));
//         }
//     }

    if (tmin > 0.f) {
        // Somehow the origin is not in the cube
        // Should not happen for distance transform

        // If using geometric distances:
        // will happen near the lowest vertex of a cell,
        // since l is always the lowest neighbor
        return 0.f;
    }
    return tmax;
}

// trilerp with links
template<class data_type_t, class voxel_index_t>
__device__ __inline__ float trilerp_cuvol_one(
        const int32_t* __restrict__ links,
        const data_type_t* __restrict__ data,
        int offx, int offy, size_t stride,
        const voxel_index_t* __restrict__ l,
        const float* __restrict__ pos,
        const int idx) {
    const int32_t* __restrict__ link_ptr = links + (offx * l[0] + offy * l[1] + l[2]);

#define MAYBE_READ_LINK(u) ((link_ptr[u] >= 0) ? data[link_ptr[u] * stride + idx] : 0.f)
    const float ix0y0 = lerp(MAYBE_READ_LINK(0), MAYBE_READ_LINK(1), pos[2]);
    const float ix0y1 = lerp(MAYBE_READ_LINK(offy), MAYBE_READ_LINK(offy + 1), pos[2]);
    const float ix0 = lerp(ix0y0, ix0y1, pos[1]);
    const float ix1y0 = lerp(MAYBE_READ_LINK(offx), MAYBE_READ_LINK(offx + 1), pos[2]);
    const float ix1y1 = lerp(MAYBE_READ_LINK(offy + offx),
                             MAYBE_READ_LINK(offy + offx + 1), pos[2]);
    const float ix1 = lerp(ix1y0, ix1y1, pos[1]);
    return lerp(ix0, ix1, pos[0]);
#undef MAYBE_READ_LINK
}

template<class data_type_t, class voxel_index_t>
__device__ __inline__ void trilerp_backward_one(
        data_type_t* __restrict__ grad_data,
        int reso, int stride,
        const voxel_index_t* __restrict__ l,
        const float* __restrict__ pos,
        float grad_out,
        const int idx) {
    const float ay = 1.f - pos[1], az = 1.f - pos[2];
    float xo = (1.0f - pos[0]) * grad_out;

    const int offz = stride;
    const int offy = reso * stride;
    const int offx = reso * offy;
    data_type_t* __restrict__ grad_data_ptr = grad_data + (offx * l[0] +
                                                    offy * l[1] +
                                                    offz * l[2]
                                                    + idx);

#define ADD_WT(u, val) atomicAdd(&grad_data_ptr[u], val)
    ADD_WT(0, ay * az * xo);
    ADD_WT(offz, ay * pos[2] * xo);
    ADD_WT(offy, pos[1] * az * xo);
    ADD_WT(offy + offz, pos[1] * pos[2] * xo);

    xo = pos[0] * grad_out;
    ADD_WT(offx, ay * az * xo);
    ADD_WT(offx + offz, ay * pos[2] * xo);
    ADD_WT(offx + offy, pos[1] * az * xo);
    ADD_WT(offx + offy + offz, pos[1] * pos[2] * xo);
#undef ADD_WT
}

template<class data_type_t, class voxel_index_t>
__device__ __inline__ void trilerp_backward_cuvol_one(
        const int32_t* __restrict__ links,
        data_type_t* __restrict__ grad_data,
        int offx, int offy, size_t stride,
        const voxel_index_t* __restrict__ l,
        const float* __restrict__ pos,
        float grad_out,
        const int idx) {
    const float ay = 1.f - pos[1], az = 1.f - pos[2];
    float xo = (1.0f - pos[0]) * grad_out;

    const int32_t* __restrict__ link_ptr = links + (offx * l[0] + offy * l[1] + l[2]);

#define MAYBE_ADD_LINK(u, val) if (link_ptr[u] >= 0) { \
              atomicAdd(&grad_data[link_ptr[u] * stride + idx], val); \
        }
    MAYBE_ADD_LINK(0, ay * az * xo);
    MAYBE_ADD_LINK(1, ay * pos[2] * xo);
    MAYBE_ADD_LINK(offy, pos[1] * az * xo);
    MAYBE_ADD_LINK(offy + 1, pos[1] * pos[2] * xo);

    xo = pos[0] * grad_out;
    MAYBE_ADD_LINK(offx + 0, ay * az * xo);
    MAYBE_ADD_LINK(offx + 1, ay * pos[2] * xo);
    MAYBE_ADD_LINK(offx + offy, pos[1] * az * xo);
    MAYBE_ADD_LINK(offx + offy + 1, pos[1] * pos[2] * xo);
#undef MAYBE_ADD_LINK
}

template<class data_type_t, class voxel_index_t>
__device__ __inline__ void trilerp_backward_cuvol_one_density(
        const int32_t* __restrict__ links,
        data_type_t* __restrict__ grad_data_out,
        bool* __restrict__ mask_out,
        int offx, int offy,
        const voxel_index_t* __restrict__ l,
        const float* __restrict__ pos,
        float grad_out) {
    const float ay = 1.f - pos[1], az = 1.f - pos[2];
    float xo = (1.0f - pos[0]) * grad_out;

    const int32_t* __restrict__ link_ptr = links + (offx * l[0] + offy * l[1] + l[2]);

#define MAYBE_ADD_LINK_DEN(u, val) if (link_ptr[u] >= 0) { \
              atomicAdd(&grad_data_out[link_ptr[u]], val); \
              mask_out[link_ptr[u]] = true; \
        }
    MAYBE_ADD_LINK_DEN(0, ay * az * xo);
    MAYBE_ADD_LINK_DEN(1, ay * pos[2] * xo);
    MAYBE_ADD_LINK_DEN(offy, pos[1] * az * xo);
    MAYBE_ADD_LINK_DEN(offy + 1, pos[1] * pos[2] * xo);

    xo = pos[0] * grad_out;
    MAYBE_ADD_LINK_DEN(offx + 0, ay * az * xo);
    MAYBE_ADD_LINK_DEN(offx + 1, ay * pos[2] * xo);
    MAYBE_ADD_LINK_DEN(offx + offy, pos[1] * az * xo);
    MAYBE_ADD_LINK_DEN(offx + offy + 1, pos[1] * pos[2] * xo);
#undef MAYBE_ADD_LINK_DEN
}

// Spherical functions

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
        // 16 not supported rn due to warp size
        // case 16:
        //     out[9] = C3[0] * y * (3 * xx - yy);
        //     out[10] = C3[1] * xy * z;
        //     out[11] = C3[2] * y * (4 * zz - xx - yy);
        //     out[12] = C3[3] * z * (2 * zz - 3 * xx - 3 * yy);
        //     out[13] = C3[4] * x * (4 * zz - xx - yy);
        //     out[14] = C3[5] * z * (xx - yy);
        //     out[15] = C3[6] * x * (xx - 3 * yy);
        //     [[fallthrough]];
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

__device__ __inline__ void calc_sphfunc(
    const PackedSparseGridSpec& grid,
    const int lane_id,
    const int ray_id,
    const float* __restrict__ dir, // Pre-normalized
    float* __restrict__ out) {
    // Placeholder
    if (grid.basis_type == BASIS_TYPE_3D_TEXTURE) {
        float p[3];
        int32_t l[3];
        for (int j = 0; j < 3; ++j) {
            // Note: this is align_corners=True behavior
            // (vs align_corners=False in the sigma/coeff grid trilerp)
            p[j] = (dir[j] * 0.5f + 0.5f) * (grid.basis_reso - 1.f);
            p[j] = min(max(p[j], 0.f), grid.basis_reso - 1.f);
            l[j] = min(static_cast<int32_t>(p[j]), grid.basis_reso - 2);
            p[j] -= static_cast<float>(l[j]);
        }

        if (lane_id > 0 && lane_id < grid.basis_dim) {
            out[lane_id] =
                    fmaxf(
                        trilerp_one(
                        grid.basis_data,
                        grid.basis_reso,
                        grid.basis_dim - 1,
                        l, p,
                        lane_id - 1),
                        0.f);
        }
        out[0] = C0;
    } else if (grid.basis_type == BASIS_TYPE_MLP) {
        const float* __restrict__ basis_ptr = grid.basis_data + grid.basis_dim * ray_id;
        if (lane_id < grid.basis_dim) {
            out[lane_id] = _SIGMOID(basis_ptr[lane_id]);
        }
    } else {
        calc_sh(grid.basis_dim, dir, out);
    }
}

__device__ __inline__ void calc_sphfunc_backward(
    const PackedSparseGridSpec& grid,
    const int lane_id,
    const int ray_id,
    const float* __restrict__ dir, // Pre-normalized
    const float* __restrict__ output_saved,
    const float* __restrict__ grad_output,
    float* __restrict__ grad_basis_data) {
    // Placeholder
    if (grid.basis_type == BASIS_TYPE_3D_TEXTURE) {
        float p[3];
        int32_t l[3];
        for (int j = 0; j < 3; ++j) {
            // Note: this is align_corners=True behavior
            // (vs align_corners=False in the sigma/coeff grid trilerp)
            p[j] = (dir[j] * 0.5f + 0.5f) * (grid.basis_reso - 1.f);
            p[j] = min(max(p[j], 0.f), grid.basis_reso - 1.f);
            l[j] = min(static_cast<int32_t>(p[j]), grid.basis_reso - 2);
            p[j] -= static_cast<float>(l[j]);
        }

        __syncwarp((1U << grid.sh_data_dim) - 1);
        if (lane_id > 0 && lane_id < grid.basis_dim && output_saved[lane_id] > 0.f) {
            trilerp_backward_one<float, int32_t>(grad_basis_data,
                    grid.basis_reso,
                    grid.basis_dim - 1,
                    l, p,
                    grad_output[lane_id],
                    lane_id - 1);
        }
    } else if (grid.basis_type == BASIS_TYPE_MLP) {
        float* __restrict__ grad_basis_ptr = grad_basis_data + grid.basis_dim * ray_id;
        if (lane_id < grid.basis_dim) {
            const float sig = output_saved[lane_id];
            grad_basis_ptr[lane_id] =
                sig * (1.f - sig) * grad_output[lane_id];
        }
    } else {
        // nothing needed
    }
}

__device__ __inline__ static float _norm(
                const float* __restrict__ dir) {
    // return sqrtf(dir[0] * dir[0] + dir[1] * dir[1] + dir[2] * dir[2]);
    return norm3df(dir[0], dir[1], dir[2]);
}

__device__ __inline__ static float _dot(
                const float* __restrict__ x,
                const float* __restrict__ y) {
    return x[0] * y[0] + x[1] * y[1] + x[2] * y[2];
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

__device__ __inline__ static void _normalize(
                float* dir) {
    float norm = _norm(dir);
    dir[0] /= norm; dir[1] /= norm; dir[2] /= norm;
}

__device__ __inline__ static void world2ndc(
        const PackedCameraSpec& __restrict__ cam,
        float* __restrict__ dir,
        float* __restrict__ cen,
        float near = 1.f) {
    // Shift ray origins to near plane, not sure if needed
    const float t = (near - cen[2]) / dir[2];
#pragma unroll 3
    for (int i = 0; i < 3; ++i) {
        cen[i] = cen[i] + t * dir[i];
    }

    dir[0] = cam.ndc_coeffx * (dir[0] / dir[2] - cen[0] / cen[2]);
    dir[1] = cam.ndc_coeffy * (dir[1] / dir[2] - cen[1] / cen[2]);
    dir[2] = 2 * near / cen[2];

    cen[0] = cam.ndc_coeffx * (cen[0] / cen[2]);
    cen[1] = cam.ndc_coeffy * (cen[1] / cen[2]);
    cen[2] = 1 - 2 * near / cen[2];

    _normalize(dir);
}

__device__ __inline__ void cam2world_ray(
    int ix, int iy,
    float* dir,
    float* origin,
    const PackedCameraSpec& __restrict__ cam) {
    // OpenCV convention (contrary to svox 1, which uses OpenGL)
    float x = (ix + 0.5f - cam.cx) / cam.fx;
    float y = (iy + 0.5f - cam.cy) / cam.fy;
    float z = sqrtf(x * x + y * y + 1.0);
    x /= z; y /= z; z = 1.0f / z;
    dir[0] = cam.c2w[0][0] * x + cam.c2w[0][1] * y + cam.c2w[0][2] * z;
    dir[1] = cam.c2w[1][0] * x + cam.c2w[1][1] * y + cam.c2w[1][2] * z;
    dir[2] = cam.c2w[2][0] * x + cam.c2w[2][1] * y + cam.c2w[2][2] * z;
    origin[0] = cam.c2w[0][3]; origin[1] = cam.c2w[1][3]; origin[2] = cam.c2w[2][3];

    if (cam.ndc_coeffx > 0.f)
        world2ndc(cam, dir, origin);
}

__device__ __inline__ void ray_find_bounds(
        SingleRaySpec& __restrict__ ray,
        const PackedSparseGridSpec& __restrict__ grid,
        const RenderOptions& __restrict__ opt) {
    // Warning: modifies ray.origin
    transform_coord(ray.origin, grid._scaling, grid._offset);
    // Warning: modifies ray.dir
    ray.world_step = _get_delta_scale(grid._scaling, ray.dir) * opt.step_size;

    ray.tmin = 0.0f;
    ray.tmax = 2e3f;
    for (int i = 0; i < 3; ++i) {
        const float invdir = 1.0 / ray.dir[i];
        const float t1 = (-0.5f - ray.origin[i]) * invdir;
        const float t2 = (grid.size[i] - 0.5f  - ray.origin[i]) * invdir;
        if (ray.dir[i] != 0.f) {
            ray.tmin = max(ray.tmin, min(t1, t2));
            ray.tmax = min(ray.tmax, max(t1, t2));
        }
    }
}

} // namespace device
} // namespace
