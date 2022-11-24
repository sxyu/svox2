// Copyright 2021 Alex Yu
#pragma once

#include <cstdint>
#include "data_spec_packed.cuh"
#include "random_util.cuh"

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

#define MAYBE_READ_LINK(u) ((link_ptr[u] >= 0) ? data[link_ptr[u] * stride + idx] : 0.f) // fetch data only if link is non negative
    const float ix0y0 = lerp(MAYBE_READ_LINK(0), MAYBE_READ_LINK(1), pos[2]);            // stride is last dim of the data
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
__device__ __inline__ void trilerp_backward_cuvol_one(
        const int32_t* __restrict__ links,
        data_type_t* __restrict__ grad_data,
        int offx, int offy, size_t stride,
        const voxel_index_t* __restrict__ l,
        const float* __restrict__ pos,
        float grad_out,
        const int idx) {
    const float ay = 1.f - pos[1], az = 1.f - pos[2];
    float xo = (1.0f - pos[0]) * grad_out; // az * d_mse/d_sh

    const int32_t* __restrict__ link_ptr = links + (offx * l[0] + offy * l[1] + l[2]);

#define MAYBE_ADD_LINK(u, val) if (link_ptr[u] >= 0) { \
              atomicAdd(&grad_data[link_ptr[u] * stride + idx], val); \
        }
    MAYBE_ADD_LINK(0, ay * az * xo); // 000
    MAYBE_ADD_LINK(1, ay * pos[2] * xo); // 001
    MAYBE_ADD_LINK(offy, pos[1] * az * xo); // 010
    MAYBE_ADD_LINK(offy + 1, pos[1] * pos[2] * xo); // 011

    xo = pos[0] * grad_out; // pz * d_mse/d_sh
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
        const float* __restrict__ pos, // pos changes during running... because of shared memory?
        float grad_out) {
    const float ay = 1.f - pos[1], az = 1.f - pos[2];
    float xo = (1.0f - pos[0]) * grad_out; // used as if az * d_mse/d_sig

    const int32_t* __restrict__ link_ptr = links + (offx * l[0] + offy * l[1] + l[2]);

#define MAYBE_ADD_LINK_DEN(u, val) if (link_ptr[u] >= 0) { \
              atomicAdd(&grad_data_out[link_ptr[u]], val); \
              if (mask_out != nullptr) \
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

template<class data_type_t, class voxel_index_t>
__device__ __inline__ void trilerp_backward_one_pos(
        const int32_t* __restrict__ links,
        const data_type_t* __restrict__ data,
        int offx, int offy, size_t stride,
        const voxel_index_t* __restrict__ l,
        const float* __restrict__ pos,
        const int idx,
        float grad_in, // d_mse/d_sh or d_mse/d_alpha
        data_type_t* __restrict__ grad_out
        )
/*
Find gradient wrt to the sample location (pos)
*/  
{
    const int32_t* __restrict__ link_ptr = links + (offx * l[0] + offy * l[1] + l[2]);

#define READ_LINK(u) (data[link_ptr[u] * stride + idx])

    // const data_type_t* __restrict__ data_ptr = data + (offx * l[0] +
    //                                                 offy * l[1] +
    //                                                 offz * l[2]
    //                                                 + idx);

    const float ix0y0 = lerp(READ_LINK(0), READ_LINK(1), pos[2]);
    const float ix0y1 = lerp(READ_LINK(offy), READ_LINK(offy + 1), pos[2]);
    const float ix0 = lerp(ix0y0, ix0y1, pos[1]);
    const float ix1y0 = lerp(READ_LINK(offx), READ_LINK(offx + 1), pos[2]);
    const float ix1y1 = lerp(READ_LINK(offy + offx),
                             READ_LINK(offy + offx + 1), pos[2]);
    const float ix1 = lerp(ix1y0, ix1y1, pos[1]);

    // s000 = READ_LINK[0], s001 = READ_LINK[1]
    // s010 = READ_LINK[offy], s011 = READ_LINK[offy+1]
    // s100 = READ_LINK[offx], s101 = READ_LINK[offx+1]
    // s110 = READ_LINK[offx+offy], s111 = READ_LINK[offx+offy+1]

    // dx
    grad_out[0] += grad_in * (ix1 - ix0);
    // dy
    grad_out[1] += grad_in * ((1-pos[0]) * (ix0y1-ix0y0) + (pos[0]) * (ix1y1-ix1y0));
    // dz
    grad_out[2] += grad_in * ((1-pos[0]) * ((1-pos[1])*(READ_LINK(1)-READ_LINK(0)) + (pos[1])*(READ_LINK(offy+1)-READ_LINK(offy))) +
                             (pos[0]) * ((1-pos[1])*(READ_LINK(offx+1)-READ_LINK(offx)) + (pos[1])*(READ_LINK(offx+offy+1)-READ_LINK(offx+offy))));
#undef READ_LINK
    ASSERT_NUM(grad_out[0]);
    ASSERT_NUM(grad_out[1]);
    ASSERT_NUM(grad_out[2]);
}

// Trilerp with xy links & wrapping (background)
template<class data_type_t, class voxel_index_t>
__device__ __inline__ float trilerp_bg_one(
        const int32_t* __restrict__ links,
        const data_type_t* __restrict__ data,
        int reso,
        int nlayers,
        int nchannels,
        const voxel_index_t* __restrict__ l,
        const float* __restrict__ pos,
        const int idx) {
#define MAYBE_READ_LINK2(varname, u) \
    float varname; \
    { \
        int link = links[u]; \
        if (link >= 0) { \
            const float* __restrict__ dptr = &data[(link * nlayers + l[2]) * nchannels + idx]; \
            varname = lerp(dptr[0], dptr[nchannels], pos[2]); \
        } else { \
            varname = 0.f; \
        } \
    }
    const int ny = l[1] < (reso - 1) ? (l[1] + 1) : 0;
    MAYBE_READ_LINK2(ix0y0, reso * l[0] + l[1]);
    MAYBE_READ_LINK2(ix0y1, reso * l[0] + ny);
    const float ix0 = lerp(ix0y0, ix0y1, pos[1]);

    const int nx = l[0] < (2 * reso - 1) ? (l[0] + 1) : 0;
    MAYBE_READ_LINK2(ix1y0, reso * nx + l[1]);
    MAYBE_READ_LINK2(ix1y1, reso * nx + ny);
    const float ix1 = lerp(ix1y0, ix1y1, pos[1]);
    return lerp(ix0, ix1, pos[0]);
#undef MAYBE_READ_LINK2
}

template<class data_type_t, class voxel_index_t>
__device__ __inline__ void trilerp_backward_bg_one(
        const int32_t* __restrict__ links,
        data_type_t* __restrict__ grad_data_out,
        bool* __restrict__ mask_out,
        int reso,
        int nlayers,
        int nchannels,
        const voxel_index_t* __restrict__ l,
        const float* __restrict__ pos,
        float grad_out,
        const int idx) {
    const float ay = 1.f - pos[1], az = 1.f - pos[2];

#define MAYBE_ADD_LINK2(u, valexpr) \
        { \
            int link = links[u]; \
            if (link >= 0) { \
                link *= nlayers; \
                float* __restrict__ gdptr = &grad_data_out[(link + l[2]) \
                                                             * nchannels + idx]; \
                const float val = (valexpr); \
                atomicAdd(gdptr, val * az); \
                atomicAdd(gdptr + nchannels, val * pos[2]); \
                if (mask_out != nullptr) { \
                    bool* __restrict__ mptr = &mask_out[link + l[2]]; \
                    mptr[0] = mptr[1] = true; \
                } \
            } \
        }

    const int ny = l[1] < (reso - 1) ? (l[1] + 1) : 0;
    float xo = (1.0f - pos[0]) * grad_out;
    MAYBE_ADD_LINK2(reso * l[0] + l[1], ay * xo);
    MAYBE_ADD_LINK2(reso * l[0] + ny, pos[1] * xo);

    xo = pos[0] * grad_out;
    const int nx = l[0] < (2 * reso - 1) ? (l[0] + 1) : 0;
    MAYBE_ADD_LINK2(reso * nx + l[1], ay * xo);
    MAYBE_ADD_LINK2(reso * nx + ny, pos[1] * xo);

#undef MAYBE_READ_LINK2
}

// Compute the amount to skip for negative link values
__device__ __inline__ float compute_skip_dist(
        SingleRaySpec& __restrict__ ray,
        const int32_t* __restrict__ links,
        int offx, int offy,
        int pos_offset = 0) {
    const int32_t link_val = links[offx * (ray.l[0] + pos_offset) +
                                   offy * (ray.l[1] + pos_offset) +
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

        if (lane_id < grid.basis_dim) {
            out[lane_id] =
                    fmaxf(
                        trilerp_one(
                        grid.basis_data,
                        grid.basis_reso,
                        grid.basis_dim,
                        l, p,
                        lane_id),
                        0.f);
        }
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
    if (grad_basis_data == nullptr) return;
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
        if (lane_id < grid.basis_dim && output_saved[lane_id] > 0.f) {
            trilerp_backward_one<float, int32_t>(grad_basis_data,
                    grid.basis_reso,
                    grid.basis_dim,
                    l, p,
                    grad_output[lane_id],
                    lane_id);
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
    float delta_scale = _rnorm(dir);
    dir[0] *= delta_scale;
    dir[1] *= delta_scale;
    dir[2] *= delta_scale;
    return delta_scale;
}

__device__ __inline__ static void _normalize(
                float* __restrict__ dir) {
    const float rnorm = _rnorm(dir);
    dir[0] *= rnorm; dir[1] *= rnorm; dir[2] *= rnorm;
}

__device__ __inline__ static void _unitvec2equirect(
        const float* __restrict__ unit_dir,
        int reso,
        float* __restrict__ xy) {
    const float lat = asinf(unit_dir[1]);
    const float lon = atan2f(unit_dir[0], unit_dir[2]);
    xy[0] = reso * 2 * (0.5 + lon * 0.5 * M_1_PI);
    xy[1] = reso * (0.5 - lat * M_1_PI);
}

__device__ __inline__ static void _equirect2unitvec(
        float x, float y,
        int reso,
        float* __restrict__ unit_dir) {
    const float lon = (x * (1.0 / (reso * 2)) - 0.5) * (2 * M_PI);
    const float lat = -(y * (1.0 / reso) - 0.5) * M_PI;
    const float coslat = cosf(lat);
    unit_dir[0] = coslat * sinf(lon);
    unit_dir[1] = sinf(lat);
    unit_dir[2] = coslat * cosf(lon);
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
        cen[i] = fmaf(t, dir[i], cen[i]);
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
    const PackedCameraSpec& __restrict__ cam,
    // Outputs
    float* __restrict__ dir,
    float* __restrict__ origin) {
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

struct ConcentricSpheresIntersector {
    __device__
        ConcentricSpheresIntersector(
                const float* __restrict__ origin,
                const float* __restrict__ dir)
    {
        q2a = 2 * _dot(dir, dir);
        qb = 2 * _dot(origin, dir);
        f = qb * qb - 2 * q2a * _dot(origin, origin);
    }

    // Get the far intersection, which we want for rendering MSI
    __device__
    bool intersect(float r, float* __restrict__ out, bool near=false) {
        float det = _det(r);
        if (det < 0) return false;
        if (near) {
            *out = (-qb - sqrtf(det)) / q2a;
        } else {
            *out = (-qb + sqrtf(det)) / q2a;
        }
        return true;
    }

    __device__ __host__
    float _det (float r) {
        return f + 2 * q2a * r * r;
    }

    float q2a, qb, f;
};

__device__ __inline__ void ray_find_bounds(
        SingleRaySpec& __restrict__ ray,
        const PackedSparseGridSpec& __restrict__ grid,
        const RenderOptions& __restrict__ opt,
        uint32_t ray_id) {
    // Warning: modifies ray.origin
    transform_coord(ray.origin, grid._scaling, grid._offset);
    // Warning: modifies ray.dir
    ray.world_step = _get_delta_scale(grid._scaling, ray.dir) * opt.step_size;

    if (opt.use_spheric_clip) {
        // Horrible hack
        const float sphere_scaling[3] {
            2.f / float(grid.size[0]),
            2.f / float(grid.size[1]),
            2.f / float(grid.size[2])
        };
        float sph_origin[3], sph_dir[3];

#pragma unroll 3
        for (int i = 0; i < 3; ++i) {
            sph_origin[i] = fmaf(ray.origin[i] + 0.5f, sphere_scaling[i], -1.f);
            sph_dir[i] = ray.dir[i] * sphere_scaling[i];
        }

        ConcentricSpheresIntersector csi(sph_origin, sph_dir);
        if (!csi.intersect(1.f, &ray.tmax) || !csi.intersect(1.f - opt.near_clip, &ray.tmin, true)) {
            ray.tmin = 1e-9f;
            ray.tmax = 0.f;
        }
    } else {
        ray.tmin = opt.near_clip / ray.world_step * opt.step_size;
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

    // if (opt.randomize && opt.random_sigma_std > 0.0) {
    //     // Seed the RNG
    //     ray.rng.x = opt._m1 ^ ray_id;
    //     ray.rng.y = opt._m2 ^ ray_id;
    //     ray.rng.z = opt._m3 ^ ray_id;
    // }
}

__device__ __inline__ void ray_find_bounds_bg(
        SingleRaySpec& __restrict__ ray,
        const PackedSparseGridSpec& __restrict__ grid,
        const RenderOptions& __restrict__ opt,
        uint32_t ray_id) {
    // Warning: modifies ray.origin
    transform_coord(ray.origin, grid._scaling, grid._offset);
    // Warning: modifies ray.dir
    ray.world_step = _get_delta_scale(grid._scaling, ray.dir);// * opt.step_size;

    const float sphere_scaling[3] {
        2.f / float(grid.size[0]),
        2.f / float(grid.size[1]),
        2.f / float(grid.size[2])
    };

#pragma unroll 3
    for (int i = 0; i < 3; ++i) {
        ray.origin[i] = fmaf(ray.origin[i] + 0.5f, sphere_scaling[i], -1.f);
        ray.dir[i] = ray.dir[i] * sphere_scaling[i];
    }

    const float inorm = _rnorm(ray.dir);
    ray.world_step *= inorm;
#pragma unroll 3
    for (int i = 0; i < 3; ++i) {
        ray.dir[i] *= inorm;
    }

    // float q2a = 2 * _dot(ray.dir, ray.dir);
    // float qb = 2 * _dot(ray.origin, ray.dir);
    // float f = qb * qb - 2 * q2a * _dot(ray.origin, ray.origin);
    // const float det = f + 2 * q2a * opt.background_msi_scale * opt.background_msi_scale;
    //
    // if (det < 0.f) {
    //     ray.tmin = opt.background_msi_scale;
    // } else {
    //     ray.tmin = (-qb + sqrtf(det)) / q2a;
    // }

    // if (opt.randomize && opt.random_sigma_std_background > 0) {
    //     // Seed the RNG (hacks)
    //     ray.rng.x = opt._m2 ^ (ray_id - 1);
    //     ray.rng.y = opt._m3 ^ (ray_id - 1);
    //     ray.rng.z = opt._m1 ^ (ray_id - 1);
    // }
}



__device__ __inline__ void surface_to_cubic_equation(
    const double* __restrict__ surface,
    const double* __restrict__ origin,
    const double* __restrict__ dir,
    const int32_t* __restrict__ l,
    double* __restrict__ outs
){

    double const a00 = surface[0b000] * (1-origin[2]+l[2]) + surface[0b001] * (origin[2]-l[2]);
    double const a01 = surface[0b010] * (1-origin[2]+l[2]) + surface[0b011] * (origin[2]-l[2]);
    double const a10 = surface[0b100] * (1-origin[2]+l[2]) + surface[0b101] * (origin[2]-l[2]);
    double const a11 = surface[0b110] * (1-origin[2]+l[2]) + surface[0b111] * (origin[2]-l[2]);

    double const b00 = -surface[0b000] + surface[0b001];
    double const b01 = -surface[0b010] + surface[0b011];
    double const b10 = -surface[0b100] + surface[0b101];
    double const b11 = -surface[0b110] + surface[0b111];

    double const c0 = a00*(1-origin[1]+l[1]) + a01*(origin[1]-l[1]);
    double const c1 = a10*(1-origin[1]+l[1]) + a11*(origin[1]-l[1]);

    double const d0 = -(a00*dir[1] - dir[2]*b00*(1-origin[1]+l[1])) + (a01*dir[1] + dir[2]*b01*(origin[1]-l[1]));
    double const d1 = -(a10*dir[1] - dir[2]*b10*(1-origin[1]+l[1])) + (a11*dir[1] + dir[2]*b11*(origin[1]-l[1]));

    double const e0 = -dir[1]*dir[2]*b00 + dir[1]*dir[2]*b01;
    double const e1 = -dir[1]*dir[2]*b10 + dir[1]*dir[2]*b11;

    outs[3] = -e0*dir[0] + e1*dir[0];
    outs[2] = -d0*dir[0]+e0*(1-origin[0]+l[0]) + d1*dir[0]+e1*(origin[0]-l[0]);
    outs[1] = -c0*dir[0] + d0*(1-origin[0]+l[0]) + c1*dir[0] + d1*(origin[0]-l[0]);
    outs[0] = c0*(1-origin[0]+l[0]) + c1*(origin[0]-l[0]);
}


__device__ __inline__ enum BasisType cubic_equation_solver(
    double const f0,
    double const f1,
    double const f2,
    double const f3,
    float const eps,
    double const eps_double,
    double* __restrict__ outs
){
    if (_CLOSE_TO_ZERO(f3, eps_double)){
        if (_CLOSE_TO_ZERO(f2, eps_double)){
            if (_CLOSE_TO_ZERO(f1, eps_double)){
                // no solution
                return CUBIC_TYPE_NO_ROOT; 
            } else {
                // linear case
                outs[0] = -f0 / f1;
                // ASSERT_NUM(outs[0]);
                return CUBIC_TYPE_LINEAR;
            }
        } else {
            // polynomial case
            // _b, _c, _d = f2[quad_mask], f1[quad_mask], f0[quad_mask]
            double const D = _SQR(f1) - 4.0 * f2 * f0;
            double const sqrt_D = sqrt(D);
            if (D > 0){
                if (f2 > 0){
                    outs[0] = (-f1 - sqrt_D) / (2 * f2);
                    outs[1] = (-f1 + sqrt_D) / (2 * f2);
                }else{
                    outs[0] = (-f1 + sqrt_D) / (2 * f2);
                    outs[1] = (-f1 - sqrt_D) / (2 * f2);
                }

                // assert(!isnan(outs[0]));
                // assert(!isnan(outs[1]));
                if (_CLOSE_TO_ZERO(outs[0] - outs[1], eps_double)){
                    // if two roots are too similiar (D==0), then just take one
                    outs[1] = -1;
                    return CUBIC_TYPE_POLY_ONE_R;
                }
                return CUBIC_TYPE_POLY;
            }
            return CUBIC_TYPE_NO_ROOT;
        }
    } else {
        // cubic case
        double const norm_term = f3;
        double const a = f3 / norm_term;
        double const b = f2 / norm_term;
        double const c = f1 / norm_term;
        double const d = f0 / norm_term;

        double const f = ((3*c/a) - (_SQR(b) / _SQR(a))) / 3;                      
        double const g = (((2*_CUBIC(b)) / _CUBIC(a)) - ((9*b*c) / _SQR(a)) + (27*d/a)) / 27;                 
        double const h = (_SQR(g) / 4 + _CUBIC(f) / 27);
        // #define norm_term (static_cast<double>(f3))
        // #define a (static_cast<double>(f3) / norm_term)
        // #define b (static_cast<double>(f2) / norm_term)
        // #define c (static_cast<double>(f1) / norm_term)
        // #define d (static_cast<double>(f0) / norm_term)

        // #define f (((3*c/a) - (_SQR(b) / _SQR(a))) / 3)
        // #define g ((((2*_CUBIC(b)) / _CUBIC(a)) - ((9*b*c) / _SQR(a)) + (27*d/a)) / 27)
        // #define h ((_SQR(g) / 4 + _CUBIC(f) / 27))
        // -inf + inf create nan!

        if ((_CLOSE_TO_ZERO(f, eps_double)) & (_CLOSE_TO_ZERO(g, eps_double)) & (_CLOSE_TO_ZERO(h, eps_double))){
            // all three roots are real and equal
            outs[0] = _COND_CBRT(d/a);
            // if ((isnan(outs[0])) | (!isfinite(outs[0]))){
            //     printf("a=%f\n", a);
            //     printf("b=%f\n", b);
            //     printf("c=%f\n", c);
            //     printf("d=%f\n", d);
            //     printf("g=%f\n", g);
            //     printf("h=%f\n", h);
            //     printf("f=%f\n", f);
            // }
            // assert(!isnan(outs[0]));

            return CUBIC_TYPE_CUBIC_ONE_R;

        } else if (h <= 0){
            // all three roots are real and distinct
            // note that if h==0, gradient cannot be computed
            if (h==0){
                return CUBIC_TYPE_NO_ROOT;
            }

            double const _i = sqrt((_SQR(g) / 4.) - h);   
            double const _j = cbrt(_i);
            double const _k = acos(-(g / (2 * _i)));
            double const _M = cos(_k / 3.);       
            double const _N = sqrt(3) * sin(_k / 3.);
            double const _P = (b / (3. * a)) * -1;         

            // #define _i (sqrt((_SQR(g) / 4.) - h))
            // #define _j (cbrt(_i))
            // #define _k (acos(-(g / (2 * _i))))
            // #define _M (cos(_k / 3.))
            // #define _N (sqrt(3) * sin(_k / 3.))
            // #define _P ((b / (3. * a)) * -1)

            outs[0] = -1 *_j * (_M + _N) + _P;
            outs[1] = -1 *_j * (_M - _N) + _P;
            outs[2] = 2 * _j * _M + _P;
            // if (isnan(outs[0]) | isnan(outs[1]) | isnan(outs[2]) | (!isfinite(outs[0])) | (!isfinite(outs[1])) | (!isfinite(outs[2]))){
            //     printf("a=%f\n", a);
            //     printf("b=%f\n", b);
            //     printf("c=%f\n", c);
            //     printf("d=%f\n", d);
            //     printf("g=%f\n", g);
            //     printf("h=%f\n", h);
            //     printf("f=%f\n", f);

            //     printf("_i=%f\n", _i);
            //     printf("_j=%f\n", _j);
            //     printf("_k=%f\n", _k);
            //     printf("_M=%f\n", _M);
            //     printf("_N=%f\n", _N);
            //     printf("_P=%f\n", _P);
            // }
            // assert(!isnan(outs[0]));
            // assert(!isnan(outs[1]));
            // assert(!isnan(outs[2]));

            return CUBIC_TYPE_CUBIC_THREE_R;
        } else {
            // only one real root
            double const _R = -(g / 2.) + sqrt(h);
            double const _S = _COND_CBRT(_R);

            double const _T = -(g / 2.) - sqrt(h);
            double const _U = _COND_CBRT(_T);
            // #define _R (-(g / 2.) + sqrt(h))
            // #define _S (_COND_CBRT(_R))

            // #define _T (-(g / 2.) - sqrt(h))
            // #define _U (_COND_CBRT(_T))

            outs[0] = (_S + _U) - (b / (3. * a));

            // if ((isnan(outs[0])) | (!isfinite(outs[0]))){
            //     printf("a=%f\n", a);
            //     printf("b=%f\n", b);
            //     printf("c=%f\n", c);
            //     printf("d=%f\n", d);
            //     printf("g=%f\n", g);
            //     printf("h=%f\n", h);
            //     printf("f=%f\n", f);
            //     printf("_R=%f\n", _R);
            //     printf("_S=%f\n", _S);
            //     printf("_T=%f\n", _T);
            //     printf("_U=%f\n", _U);
            // }
            //     assert(!isnan(outs[0]));

            return CUBIC_TYPE_CUBIC_ONE_R_;
        }

    }
}

// #undef norm_term 
// #undef a  
// #undef b  
// #undef c  
// #undef d  

// #undef f  
// #undef g  
// #undef h

// #undef _i
// #undef _j
// #undef _k
// #undef _M
// #undef _N
// #undef _P

// #undef _R
// #undef _S
// #undef _T
#undef _U

__device__ __inline__ void calc_cubic_root_grad(
    enum BasisType cubic_root_type,
    int st_id,
    double* __restrict__ const fs,
    float* __restrict__ grad_fs // storing grad_st
){
    //////////////////////// Find Gradient of Cubic Root //////////////////////
    if (cubic_root_type == CUBIC_TYPE_LINEAR){
        // linear case
        grad_fs[0] *= static_cast<float>(-1. / fs[1]);
        grad_fs[1] *= static_cast<float>(fs[0] / _SQR(fs[1]));
        grad_fs[2] = 0.f;
        grad_fs[3] = 0.f;

    }else if (cubic_root_type == CUBIC_TYPE_POLY_ONE_R){
        double const D = _SQR(fs[1]) - 4. * fs[2] * fs[0];
        double const sqrt_D = sqrt(D);
        double const dt0_dD = 1 / (4.*fs[2]*sqrt_D);
        grad_fs[0] *= static_cast<float>(-1/sqrt_D);
        grad_fs[1] *= static_cast<float>(((-1) / (2*fs[2]) + (dt0_dD * 2 * fs[1])));
        grad_fs[2] *= static_cast<float>(((fs[1] - sqrt_D) / (4*_SQR(fs[2])) + (dt0_dD * (-4) * fs[0])));
        grad_fs[3] = 0.f;

    }else if (cubic_root_type == CUBIC_TYPE_POLY){
        double const  D = _SQR(fs[1]) - 4.0 * fs[2] * fs[0];
        double const  sqrt_D = sqrt(D);
        double const  sqr_f2 = _SQR(fs[2]);

        #define __POLY_ROOT_GRAD_S \
                double const dt_dD = -1 / (4*fs[2]*sqrt_D); \
                grad_fs[0] *= static_cast<float>(1/sqrt_D); \
                grad_fs[1] *= static_cast<float>(((-1) / (2*fs[2]) + (dt_dD * 2 * fs[1]))); \
                grad_fs[2] *= static_cast<float>(((fs[1] + sqrt_D) / (2*sqr_f2) + (dt_dD * (-4) * fs[0])));

        #define __POLY_ROOT_GRAD_L \
                double const dt_dD = 1 / (4*fs[2]*sqrt_D); \
                grad_fs[0] *= static_cast<float>(-1/sqrt_D); \
                grad_fs[1] *= static_cast<float>(((-1) / (2*fs[2]) + (dt_dD * 2 * fs[1]))); \
                grad_fs[2] *= static_cast<float>(((fs[1] - sqrt_D) / (2*sqr_f2) + (dt_dD * (-4) * fs[0]))); \

        if (fs[2] > 0){
            if (st_id == 0){
                // st[0] = (-f1 - sqrt_D) / (2 * f2);
                __POLY_ROOT_GRAD_S
            }else{
                // st[1] = (-f1 + sqrt_D) / (2 * f2);
                __POLY_ROOT_GRAD_L
            }
        }else{
            if (st_id == 0){
                // st[0] = (-f1 + sqrt_D) / (2 * f2);
                __POLY_ROOT_GRAD_S
            }else{
                // st[1] = (-f1 - sqrt_D) / (2 * f2);
                __POLY_ROOT_GRAD_L
            }
        }
        grad_fs[3] = 0.f;

    }else{
        // macros for cubic gradient
        double const  norm_term = fs[3];
        double const  a = fs[3] / norm_term;
        double const  b = fs[2] / norm_term;
        double const  c = fs[1] / norm_term;
        double const  d = fs[0] / norm_term;

        #define __Db_Df3 ((-fs[2]) / fs[3] / fs[3])
        #define __Db_Df2 (1. / fs[3])
        #define __Db_Df1 (0.)
        #define __Db_Df0 (0.)
        #define __Dc_Df3 ((-fs[1]) / fs[3] / fs[3])
        #define __Dc_Df2 (0.)
        #define __Dc_Df1 (1. / fs[3])
        #define __Dc_Df0 (0.)
        #define __Dd_Df3 ((-fs[0]) / fs[3] / fs[3])
        #define __Dd_Df2 (0.)
        #define __Dd_Df1 (0.)
        #define __Dd_Df0 (1. / fs[3])
        // double const volatile __Db_Df3 = ((-fs[2]) / fs[3] / fs[3]);
        // double const volatile __Db_Df2 = (1. / fs[3]);
        // double const volatile __Db_Df1 = (0.);
        // double const volatile __Db_Df0 = (0.);
        // double const volatile __Dc_Df3 = ((-fs[1]) / fs[3] / fs[3]);
        // double const volatile __Dc_Df2 = (0.);
        // double const volatile __Dc_Df1 = (1. / fs[3]);
        // double const volatile __Dc_Df0 = (0.);
        // double const volatile __Dd_Df3 = ((-fs[0]) / fs[3] / fs[3]);
        // double const volatile __Dd_Df2 = (0.);
        // double const volatile __Dd_Df1 = (0.);
        // double const volatile __Dd_Df0 = (1. / fs[3]);

        double const  f = ((3.*c/a) - (_SQR(b) / _SQR(a))) / 3.;                      
        double const  g = (((2.*_CUBIC(b)) / _CUBIC(a)) - ((9.*b*c) / _SQR(a)) + (27.*d/a)) / 27.;                 
        double const  h = (_SQR(g) / 4. + _CUBIC(f) / 27.);

        #define __Df_Db (-2.*b/(3.*_SQR(a)))
        #define __Df_Dc (1./a)
        #define __Df_Dd (0.)
        #define __Dg_Db (-c/(3.*_SQR(a)) + 2.*_SQR(b)/(9.*_CUBIC(a)))
        #define __Dg_Dc (-b/(3.*_SQR(a)))
        #define __Dg_Dd (1./a)

        #define __Dh_Df (_SQR(f)/9.)
        #define __Dh_Dg (g/2.)
        #define __Dh_Db (__Dh_Df * __Df_Db + __Dh_Dg * __Dg_Db)
        #define __Dh_Dc (__Dh_Df * __Df_Dc + __Dh_Dg * __Dg_Dc)
        #define __Dh_Dd (__Dh_Df * __Df_Dd + __Dh_Dg * __Dg_Dd)
        
        #define __Dg_Df3 (__Dg_Db * __Db_Df3 + __Dg_Dc * __Dc_Df3 + __Dg_Dd * __Dd_Df3)
        #define __Dg_Df2 (__Dg_Db * __Db_Df2 + __Dg_Dc * __Dc_Df2 + __Dg_Dd * __Dd_Df2)
        #define __Dg_Df1 (__Dg_Db * __Db_Df1 + __Dg_Dc * __Dc_Df1 + __Dg_Dd * __Dd_Df1)
        #define __Dg_Df0 (__Dg_Db * __Db_Df0 + __Dg_Dc * __Dc_Df0 + __Dg_Dd * __Dd_Df0)
        #define __Dh_Df3 (__Dh_Db * __Db_Df3 + __Dh_Dc * __Dc_Df3 + __Dh_Dd * __Dd_Df3)
        #define __Dh_Df2 (__Dh_Db * __Db_Df2 + __Dh_Dc * __Dc_Df2 + __Dh_Dd * __Dd_Df2)
        #define __Dh_Df1 (__Dh_Db * __Db_Df1 + __Dh_Dc * __Dc_Df1 + __Dh_Dd * __Dd_Df1)
        #define __Dh_Df0 (__Dh_Db * __Db_Df0 + __Dh_Dc * __Dc_Df0 + __Dh_Dd * __Dd_Df0)
        // double const volatile __Df_Db = (-2.*b/(3.*_SQR(a)));
        // double const volatile __Df_Dc = (1./a);
        // double const volatile __Df_Dd = (0.);
        // double const volatile __Dg_Db = (-c/(3.*_SQR(a)) + 2.*_SQR(b)/(9.*_CUBIC(a)));
        // double const volatile __Dg_Dc = (-b/(3.*_SQR(a)));
        // double const volatile __Dg_Dd = (1./a);

        // double const volatile __Dh_Df = (_SQR(f)/9.);
        // double const volatile __Dh_Dg = (g/2.);
        // double const volatile __Dh_Db = (__Dh_Df * __Df_Db + __Dh_Dg * __Dg_Db);
        // double const volatile __Dh_Dc = (__Dh_Df * __Df_Dc + __Dh_Dg * __Dg_Dc);
        // double const volatile __Dh_Dd = (__Dh_Df * __Df_Dd + __Dh_Dg * __Dg_Dd);

        // double const volatile __Dg_Df3 = (__Dg_Db * __Db_Df3 + __Dg_Dc * __Dc_Df3 + __Dg_Dd * __Dd_Df3);
        // double const volatile __Dg_Df2 = (__Dg_Db * __Db_Df2 + __Dg_Dc * __Dc_Df2 + __Dg_Dd * __Dd_Df2);
        // double const volatile __Dg_Df1 = (__Dg_Db * __Db_Df1 + __Dg_Dc * __Dc_Df1 + __Dg_Dd * __Dd_Df1);
        // double const volatile __Dg_Df0 = (__Dg_Db * __Db_Df0 + __Dg_Dc * __Dc_Df0 + __Dg_Dd * __Dd_Df0);
        // double const volatile __Dh_Df3 = (__Dh_Db * __Db_Df3 + __Dh_Dc * __Dc_Df3 + __Dh_Dd * __Dd_Df3);
        // double const volatile __Dh_Df2 = (__Dh_Db * __Db_Df2 + __Dh_Dc * __Dc_Df2 + __Dh_Dd * __Dd_Df2);
        // double const volatile __Dh_Df1 = (__Dh_Db * __Db_Df1 + __Dh_Dc * __Dc_Df1 + __Dh_Dd * __Dd_Df1);
        // double const volatile __Dh_Df0 = (__Dh_Db * __Db_Df0 + __Dh_Dc * __Dc_Df0 + __Dh_Dd * __Dd_Df0);


        if (cubic_root_type == CUBIC_TYPE_CUBIC_ONE_R){
            // cubic with three real and equal roots
            double const d_con_cbrt = _D_COND_CBRT(d/a);
            grad_fs[0] *= static_cast<float>(d_con_cbrt / a * __Dd_Df0); // a=1 can be further simplified
            grad_fs[1] = 0;
            grad_fs[2] = 0;
            grad_fs[3] *= static_cast<float>(d_con_cbrt / a * __Dd_Df3);
        }else if (cubic_root_type == CUBIC_TYPE_CUBIC_THREE_R){
            // cubic with three real and distinct roots
            double const  _i = sqrt((_SQR(g) / 4.) - h);   
            double const  _j = cbrt(_i);
            double const  _k = acos(-(g / (2. * _i)));
            double const  _M = cos(_k/3.);
            double const  _N = sqrt(3.) * sin(_k / 3.);
            // double const  _P = (b / (3. * a)) * -1.;   

            #define __Dj_Dg (g/(12.* pow(_SQR(g)/4. - h, 5./6.)))
            #define __Dj_Dh (-1./(6.* pow(_SQR(g)/4. - h, 5./6.)))

            #define __Dk_Dg (-(_SQR(g)/(8.*pow(_SQR(g)/4. - h, 3./2.)) - 1./(2.*sqrt(_SQR(g)/4. - h)))/sqrt(-_SQR(g)/(4.*(_SQR(g)/4. - h)) + 1.))
            #define __Dk_Dh (g/(4.*pow(_SQR(g)/4. - h, 3./2.)*sqrt(-_SQR(g)/(4.*(_SQR(g)/4. - h)) + 1.)))

            
            double  __Dst_Dj = 0.;
            double  __Dst_Dk = 0.;
            double  __Dst_Db_ = 0.;

            if (st_id == 0){
                // st[0]
                __Dst_Dj = (-_M - _N);
                __Dst_Dk = (-_j*(-sin(_k/3.)/3. + sqrt(3.)*_M/3.));
                __Dst_Db_ = (-1./(3.*a));
            }else if (st_id == 1){
                // st[1]
                __Dst_Dj = (-_M + _N);
                __Dst_Dk = (-_j*(-sin(_k/3.)/3. - sqrt(3.)*_M/3.));
                __Dst_Db_ = (-1./(3.*a));
            }else{
                // st[2]
                __Dst_Dj = (2.*_M);
                __Dst_Dk = (-2.*_j*sin(_k/3.)/3.);
                __Dst_Db_ = (-1./(3.*a));
            }


            grad_fs[0] *= static_cast<float>(
                __Dst_Dj * (__Dj_Dg * __Dg_Df0 
                            +__Dj_Dh * __Dh_Df0) 
                + __Dst_Dk * (__Dk_Dg * __Dg_Df0 
                                +__Dk_Dh * __Dh_Df0) 
                + __Dst_Db_ * __Db_Df0);

            grad_fs[1] *= static_cast<float>(
                __Dst_Dj * (__Dj_Dg * __Dg_Df1 
                            +__Dj_Dh * __Dh_Df1) 
                + __Dst_Dk * (__Dk_Dg * __Dg_Df1 
                                +__Dk_Dh * __Dh_Df1) 
                + __Dst_Db_ * __Db_Df1);

            grad_fs[2] *= static_cast<float>(
                __Dst_Dj * (__Dj_Dg * __Dg_Df2 
                            +__Dj_Dh * __Dh_Df2) 
                + __Dst_Dk * (__Dk_Dg * __Dg_Df2 
                                +__Dk_Dh * __Dh_Df2) 
                + __Dst_Db_ * __Db_Df2);

            grad_fs[3] *= static_cast<float>(
                __Dst_Dj * (__Dj_Dg * __Dg_Df3 
                            +__Dj_Dh * __Dh_Df3) 
                + __Dst_Dk * (__Dk_Dg * __Dg_Df3 
                                +__Dk_Dh * __Dh_Df3) 
                + __Dst_Db_ * __Db_Df3);

            ASSERT_NUM(grad_fs[0]);
            ASSERT_NUM(grad_fs[1]);
            ASSERT_NUM(grad_fs[2]);
            ASSERT_NUM(grad_fs[3]);


        }else{
            // CUBIC_TYPE_CUBIC_ONE_R_: cubic with a single real root
            double const  _R = -(g / 2.) + sqrt(h);
            // double const  _S = _COND_CBRT(_R);

            double const  _T = -(g / 2.) - sqrt(h);
            // double const  _U = _COND_CBRT(_T);

            // #define Dst_DS (1)
            // #define Dst_DU (1)
            #define DS_DR (_D_COND_CBRT(_R))
            #define DU_DT (_D_COND_CBRT(_T))

            #define __Dst_DR (_D_COND_CBRT(_R))
            #define __Dst_DT (_D_COND_CBRT(_T))
            #define __Dst_Db_ (-1./(3.*a))

            #define __DR_Dh (1./(2.*sqrt(h)))
            #define __DR_Dg (-0.5)
            #define __DT_Dh (-1./(2.*sqrt(h)))
            #define __DT_Dg (-0.5)
            // double const volatile __Dst_DR = (_D_COND_CBRT(_R));
            // double const volatile __Dst_DT = (_D_COND_CBRT(_T));
            // double const volatile __Dst_Db_ = (-1./(3.*a));

            // double const volatile __DR_Dh = (1./(2.*sqrt(h)));
            // double const volatile __DR_Dg = (-0.5);
            // double const volatile __DT_Dh = (-1./(2.*sqrt(h)));
            // double const volatile __DT_Dg = (-0.5);

            grad_fs[0] *= static_cast<float>(
                __Dst_DR * (__DR_Dg * __Dg_Df0 
                            +__DR_Dh * __Dh_Df0) 
                + __Dst_DT * (__DT_Dg * __Dg_Df0 
                                +__DT_Dh * __Dh_Df0) 
                + __Dst_Db_ * __Db_Df0);

            grad_fs[1] *= static_cast<float>(
                __Dst_DR * (__DR_Dg * __Dg_Df1 
                            +__DR_Dh * __Dh_Df1) 
                + __Dst_DT * (__DT_Dg * __Dg_Df1 
                                +__DT_Dh * __Dh_Df1) 
                + __Dst_Db_ * __Db_Df1);

            grad_fs[2] *= static_cast<float>(
                __Dst_DR * (__DR_Dg * __Dg_Df2 
                            +__DR_Dh * __Dh_Df2) 
                + __Dst_DT * (__DT_Dg * __Dg_Df2 
                                +__DT_Dh * __Dh_Df2) 
                + __Dst_Db_ * __Db_Df2);

            grad_fs[3] *= static_cast<float>(
                __Dst_DR * (__DR_Dg * __Dg_Df3 
                            +__DR_Dh * __Dh_Df3) 
                + __Dst_DT * (__DT_Dg * __Dg_Df3 
                                +__DT_Dh * __Dh_Df3) 
                + __Dst_Db_ * __Db_Df3);

            ASSERT_NUM(grad_fs[0]);
            ASSERT_NUM(grad_fs[1]);
            ASSERT_NUM(grad_fs[2]);
            ASSERT_NUM(grad_fs[3]);
        }

    }


}

__device__ __inline__ void calc_surface_grad(
    const float* __restrict__ origin,
    const float* __restrict__ dir,
    const int32_t* __restrict__ l,
    float* __restrict__ const grad_fs,
    float* __restrict__ grad_surface
){
    grad_surface[0b000] = 
            grad_fs[0] * ((l[0] - origin[0] + 1)*(l[1] - origin[1] + 1)*(l[2] - origin[2] + 1))
        + grad_fs[1] * (dir[0]*(l[1] - origin[1] + 1)*(-l[2] + origin[2] - 1) + (-dir[1]*(l[2] - origin[2] + 1) - dir[2]*(l[1] - origin[1] + 1))*(l[0] - origin[0] + 1))
        + grad_fs[2] * (dir[0]*(dir[1]*(l[2] - origin[2] + 1) + dir[2]*(l[1] - origin[1] + 1)) + dir[1]*dir[2]*(l[0] - origin[0] + 1))
        + grad_fs[3] * (-dir[0]*dir[1]*dir[2]);

    grad_surface[0b001] = 
            grad_fs[0] * ((-l[2] + origin[2])*(l[0] - origin[0] + 1)*(l[1] - origin[1] + 1)) 
        + grad_fs[1] * (dir[0]*(l[2] - origin[2])*(l[1] - origin[1] + 1) + (-dir[1]*(-l[2] + origin[2]) + dir[2]*(l[1] - origin[1] + 1))*(l[0] - origin[0] + 1))
        + grad_fs[2] * (dir[0]*(dir[1]*(-l[2] + origin[2]) - dir[2]*(l[1] - origin[1] + 1)) - dir[1]*dir[2]*(l[0] - origin[0] + 1))
        + grad_fs[3] * (dir[0]*dir[1]*dir[2]);

    grad_surface[0b010] = 
            grad_fs[0] * ((-l[1] + origin[1])*(l[0] - origin[0] + 1)*(l[2] - origin[2] + 1)) 
        + grad_fs[1] * (dir[0]*(l[1] - origin[1])*(l[2] - origin[2] + 1) + (dir[1]*(l[2] - origin[2] + 1) - dir[2]*(-l[1] + origin[1]))*(l[0] - origin[0] + 1))
        + grad_fs[2] * (dir[0]*(-dir[1]*(l[2] - origin[2] + 1) + dir[2]*(-l[1] + origin[1])) - dir[1]*dir[2]*(l[0] - origin[0] + 1))
        + grad_fs[3] * (dir[0]*dir[1]*dir[2]);

    grad_surface[0b011] = 
            grad_fs[0] * ((-l[1] + origin[1])*(-l[2] + origin[2])*(l[0] - origin[0] + 1)) 
        + grad_fs[1] * (dir[0]*(l[1] - origin[1])*(-l[2] + origin[2]) + (dir[1]*(-l[2] + origin[2]) + dir[2]*(-l[1] + origin[1]))*(l[0] - origin[0] + 1))
        + grad_fs[2] * (dir[0]*(-dir[1]*(-l[2] + origin[2]) - dir[2]*(-l[1] + origin[1])) + dir[1]*dir[2]*(l[0] - origin[0] + 1))
        + grad_fs[3] * (-dir[0]*dir[1]*dir[2]);

    grad_surface[0b100] = 
            grad_fs[0] * ((-l[0] + origin[0])*(l[1] - origin[1] + 1)*(l[2] - origin[2] + 1)) 
        + grad_fs[1] * (dir[0]*(l[1] - origin[1] + 1)*(l[2] - origin[2] + 1) + (-l[0] + origin[0])*(-dir[1]*(l[2] - origin[2] + 1) - dir[2]*(l[1] - origin[1] + 1)))
        + grad_fs[2] * (dir[0]*(-dir[1]*(l[2] - origin[2] + 1) - dir[2]*(l[1] - origin[1] + 1)) + dir[1]*dir[2]*(-l[0] + origin[0]))
        + grad_fs[3] * (dir[0]*dir[1]*dir[2]);

    grad_surface[0b101] = 
            grad_fs[0] * ((-l[0] + origin[0])*(-l[2] + origin[2])*(l[1] - origin[1] + 1)) 
        + grad_fs[1] * (dir[0]*(-l[2] + origin[2])*(l[1] - origin[1] + 1) + (-l[0] + origin[0])*(-dir[1]*(-l[2] + origin[2]) + dir[2]*(l[1] - origin[1] + 1)))
        + grad_fs[2] * (dir[0]*(-dir[1]*(-l[2] + origin[2]) + dir[2]*(l[1] - origin[1] + 1)) - dir[1]*dir[2]*(-l[0] + origin[0]))
        + grad_fs[3] * (-dir[0]*dir[1]*dir[2]);

    grad_surface[0b110] = 
            grad_fs[0] * ((-l[0] + origin[0])*(-l[1] + origin[1])*(l[2] - origin[2] + 1)) 
        + grad_fs[1] * (dir[0]*(-l[1] + origin[1])*(l[2] - origin[2] + 1) + (-l[0] + origin[0])*(dir[1]*(l[2] - origin[2] + 1) - dir[2]*(-l[1] + origin[1])))
        + grad_fs[2] * (dir[0]*(dir[1]*(l[2] - origin[2] + 1) - dir[2]*(-l[1] + origin[1])) - dir[1]*dir[2]*(-l[0] + origin[0]))
        + grad_fs[3] * (-dir[0]*dir[1]*dir[2]);

    grad_surface[0b111] = 
            grad_fs[0] * ((-l[0] + origin[0])*(-l[1] + origin[1])*(-l[2] + origin[2])) 
        + grad_fs[1] * (dir[0]*(-l[1] + origin[1])*(-l[2] + origin[2]) + (-l[0] + origin[0])*(dir[1]*(-l[2] + origin[2]) + dir[2]*(-l[1] + origin[1])))
        + grad_fs[2] * (dir[0]*(dir[1]*(-l[2] + origin[2]) + dir[2]*(-l[1] + origin[1])) + dir[1]*dir[2]*(-l[0] + origin[0]))
        + grad_fs[3] * (dir[0]*dir[1]*dir[2]);

    // if (isnan(grad_surface[0b000])){
    //     printf("!grad_fs[0]: %f\n", grad_fs[0]);
    //     printf("grad_fs[1]: %f\n", grad_fs[1]);
    //     printf("grad_fs[2]: %f\n", grad_fs[2]);
    //     printf("grad_fs[3]: %f\n", grad_fs[3]);

    //     printf("ray.l[0]: %f\n", ray.l[0]);
    //     printf("ray.l[1]: %f\n", ray.l[1]);
    //     printf("ray.l[2]: %f\n", ray.l[2]);

    //     printf("ray.origin[0]: %f\n", ray.origin[0]);
    //     printf("ray.origin[1]: %f\n", ray.origin[1]);
    //     printf("ray.origin[2]: %f\n", ray.origin[2]);

    //     printf("ray.dir[0]: %f\n", ray.dir[0]);
    //     printf("ray.dir[1]: %f\n", ray.dir[1]);
    //     printf("ray.dir[2]: %f\n", ray.dir[2]);

    //     assert(!isnan(grad_surface[0b000]));
    // }

    ASSERT_NUM(grad_surface[0b000]);
    ASSERT_NUM(grad_surface[0b001]);
    ASSERT_NUM(grad_surface[0b010]);
    ASSERT_NUM(grad_surface[0b011]);
    ASSERT_NUM(grad_surface[0b100]);
    ASSERT_NUM(grad_surface[0b101]);
    ASSERT_NUM(grad_surface[0b110]);
    ASSERT_NUM(grad_surface[0b111]);

}

template<class data_type_t, class voxel_index_t>
__device__ __inline__ void assign_surface_grad(
    const int32_t* __restrict__ links,
    data_type_t* __restrict__ grad_surface_out,
    bool* __restrict__ mask_out,
    int const offx, int const offy,
    const voxel_index_t* __restrict__ l,
    float* __restrict__ grad_surface
){

    const int32_t* __restrict__ link_ptr = links + (offx * l[0] + offy * l[1] + l[2]);
    #define MAYBE_ADD_LINK(u, val) if (link_ptr[u] >= 0) { \
                atomicAdd(&grad_surface_out[link_ptr[u]], val); \
                if (mask_out != nullptr) \
                    mask_out[link_ptr[u]] = true; \
            }

    MAYBE_ADD_LINK(0, grad_surface[0b000]);
    MAYBE_ADD_LINK(1, grad_surface[0b001]);
    MAYBE_ADD_LINK(offy, grad_surface[0b010]);
    MAYBE_ADD_LINK(offy + 1, grad_surface[0b011]);
    MAYBE_ADD_LINK(offx + 0, grad_surface[0b100]);
    MAYBE_ADD_LINK(offx + 1, grad_surface[0b101]);
    MAYBE_ADD_LINK(offx + offy, grad_surface[0b110]);
    MAYBE_ADD_LINK(offx + offy + 1, grad_surface[0b111]);

    #undef MAYBE_ADD_LINK
}


__device__ __inline__ void _split_add_surface_norm_grad(
    const int x, const int y, const int z,
    const float* __restrict__ grad_n,
    const float scale,
    const int32_t* __restrict__ links,
    int const offx, int const offy,
    const size_t ddim,
    const int idx,
    bool* __restrict__ mask_out,
    float* __restrict__ grad_data
){
    /**
     * Note that this is a helper function only used when caluclating surface normal grad
     * 
    */
    const int32_t* __restrict__ link_ptr = links + (offx * x + offy * y + z);

    #define MAYBE_ADD_LINK(u, val) if (val != 0.f) { \
                atomicAdd(&grad_data[link_ptr[u]], val); \
                if (mask_out != nullptr) \
                    mask_out[link_ptr[u]] = true; \
            }


    float const grad000 = -0.25f * grad_n[0] + -0.25f * grad_n[1] + -0.25f * grad_n[2]; 
    MAYBE_ADD_LINK(0, scale*grad000);
    float const grad001 = -0.25f * grad_n[0] + -0.25f * grad_n[1] + 0.25f * grad_n[2]; 
    MAYBE_ADD_LINK(1, scale*grad001);
    float const grad010 = -0.25f * grad_n[0] + 0.25f * grad_n[1] + -0.25f * grad_n[2]; 
    MAYBE_ADD_LINK(offy, scale*grad010);
    float const grad011 = -0.25f * grad_n[0] + 0.25f * grad_n[1] + 0.25f * grad_n[2]; 
    MAYBE_ADD_LINK(offy + 1, scale*grad011);
    float const grad100 = 0.25f * grad_n[0] + -0.25f * grad_n[1] + -0.25f * grad_n[2]; 
    MAYBE_ADD_LINK(offx + 0, scale*grad100);
    float const grad101 = 0.25f * grad_n[0] + -0.25f * grad_n[1] + 0.25f * grad_n[2]; 
    MAYBE_ADD_LINK(offx + 1, scale*grad101);
    float const grad110 = 0.25f * grad_n[0] + 0.25f * grad_n[1] + -0.25f * grad_n[2]; 
    MAYBE_ADD_LINK(offx + offy, scale*grad110);
    float const grad111 = 0.25f * grad_n[0] + 0.25f * grad_n[1] + 0.25f * grad_n[2]; 
    MAYBE_ADD_LINK(offx + offy + 1, scale*grad111);


    #undef MAYBE_ADD_LINK

}

__device__ __inline__ void add_surface_normal_grad(
        // const torch::PackedTensorAccessor32<int32_t, 3, torch::RestrictPtrTraits> links,
        // const torch::PackedTensorAccessor64<float, 2, torch::RestrictPtrTraits> data,
        const int32_t* __restrict__ links,
        const float* __restrict__ surface_data,
        const int* __restrict__ size,
        const int x, const int y, const int z,
        int const offx, int const offy,
        float const lv_set,
        float scale,
        bool con_check,
        bool ignore_empty,
        // Output
        bool* __restrict__ mask_out,
        float* __restrict__ grad_data) {

    #define __FETCH_LINK(_x,_y,_z) (links[(_x)*offx + (_y)*offy + (_z)])
    // always put brackets around input macro parameters!

    #define __GRID_EXIST(x,y,z) (\
    (x < size[0] - 1) && (y < size[1] - 1) && (z < size[2] - 1) && \
    (__FETCH_LINK(x,y,z) >= 0) && (__FETCH_LINK(x,y,z+1) >= 0) && (__FETCH_LINK(x,y+1,z) >= 0) && (__FETCH_LINK(x,y+1,z+1) >= 0) && (__FETCH_LINK(x+1,y,z) >= 0) && (__FETCH_LINK(x+1,y,z+1) >= 0) && (__FETCH_LINK(x+1,y+1,z) >= 0) && (__FETCH_LINK(x+1,y+1,z+1) >= 0))

    if (!__GRID_EXIST(x,y,z)) return;

    // float scaling[3];
    // CALCULATE_RAY_SCALE(scaling, size[0], size[1], size[2]); // scale = links.size / 256.f

    // const float* dptr = data.data();
    // const size_t ddim = data.size(1);
    const int idx = 0;
    const size_t ddim = 1;

    #define __FETCH_DATA(x,y,z) (surface_data[__FETCH_LINK(x,y,z) * ddim + idx])

    #define __CHECK_EMPTY(x,y,z) (((__FETCH_DATA(x,y,z) <= lv_set) && (__FETCH_DATA(x,y,z+1) <= lv_set) && (__FETCH_DATA(x,y+1,z) <= lv_set) && (__FETCH_DATA(x,y+1,z+1) <= lv_set) && \ 
                                   (__FETCH_DATA(x+1,y,z) <= lv_set) && (__FETCH_DATA(x+1,y,z+1) <= lv_set) && (__FETCH_DATA(x+1,y+1,z) <= lv_set) && (__FETCH_DATA(x+1,y+1,z+1) <= lv_set)) || \
                                  ((__FETCH_DATA(x,y,z) >= lv_set) && (__FETCH_DATA(x,y,z+1) >= lv_set) && (__FETCH_DATA(x,y+1,z) >= lv_set) && (__FETCH_DATA(x,y+1,z+1) >= lv_set) && \ 
                                   (__FETCH_DATA(x+1,y,z) >= lv_set) && (__FETCH_DATA(x+1,y,z+1) >= lv_set) && (__FETCH_DATA(x+1,y+1,z) >= lv_set) && (__FETCH_DATA(x+1,y+1,z+1) >= lv_set)))

    bool const empty000 = ignore_empty ? __CHECK_EMPTY(x,y,z) : false;

    // __FETCH_DATA(x+1,y,z);
    #define __CALC_DX(x,y,z) (((__FETCH_DATA(x+1,y,z)+__FETCH_DATA(x+1,y,z+1)+__FETCH_DATA(x+1,y+1,z)+__FETCH_DATA(x+1,y+1,z+1)) - \
         (__FETCH_DATA(x,y,z)+__FETCH_DATA(x,y,z+1)+__FETCH_DATA(x,y+1,z)+__FETCH_DATA(x,y+1,z+1))) /4)
    #define __CALC_DY(x,y,z) (((__FETCH_DATA(x,y+1,z)+__FETCH_DATA(x,y+1,z+1)+__FETCH_DATA(x+1,y+1,z)+__FETCH_DATA(x+1,y+1,z+1)) - \
         (__FETCH_DATA(x,y,z)+__FETCH_DATA(x,y,z+1)+__FETCH_DATA(x+1,y,z)+__FETCH_DATA(x+1,y,z+1)))/4)
    #define __CALC_DZ(x,y,z) (((__FETCH_DATA(x,y,z+1)+__FETCH_DATA(x,y+1,z+1)+__FETCH_DATA(x+1,y,z+1)+__FETCH_DATA(x+1,y+1,z+1)) - \
         (__FETCH_DATA(x,y,z)+__FETCH_DATA(x,y+1,z)+__FETCH_DATA(x+1,y,z)+__FETCH_DATA(x+1,y+1,z)))/4)

    // printf("__FETCH_DATA(x+1,y,z): %f\n", __FETCH_DATA(x+1,y,z));
    // printf("__FETCH_DATA(x+1,y,z+1): %f\n", __FETCH_DATA(x+1,y,z+1));
    // printf("__FETCH_DATA(x+1,y+1,z): %f\n", __FETCH_DATA(x+1,y+1,z));
    // printf("__FETCH_DATA(x+1,y+1,z+1): %f\n", __FETCH_DATA(x+1,y+1,z+1));
    // printf("__FETCH_DATA(x,y,z): %f\n", __FETCH_DATA(x,y,z));
    // printf("__FETCH_DATA(x,y,z+1): %f\n", __FETCH_DATA(x,y,z+1));
    // printf("__FETCH_DATA(x,y+1,z): %f\n", __FETCH_DATA(x,y+1,z));
    // printf("__FETCH_DATA(x,y+1,z+1): %f\n\n", __FETCH_DATA(x,y+1,z+1));

    // printf("__FETCH_LINK(x+1,y,z): %d\n", __FETCH_LINK(x+1,y,z));
    // printf("__FETCH_LINK(x+1,y,z+1): %d\n", __FETCH_LINK(x+1,y,z+1));
    // printf("__FETCH_LINK(x+1,y+1,z): %d\n", __FETCH_LINK(x+1,y+1,z));
    // printf("__FETCH_LINK(x+1,y+1,z+1): %d\n", __FETCH_LINK(x+1,y+1,z+1));
    // printf("__FETCH_LINK(x,y,z): %d\n", __FETCH_LINK(x,y,z));
    // printf("__FETCH_LINK(x,y,z+1): %d\n", __FETCH_LINK(x,y,z+1));
    // printf("__FETCH_LINK(x,y+1,z): %d\n", __FETCH_LINK(x,y+1,z));
    // printf("__FETCH_LINK(x,y+1,z+1): %d\n\n", __FETCH_LINK(x,y+1,z+1));

    // printf("(links[x*offx + (y+1)*offy + z]): %d\n", links[x*offx + (y+1)*offy + z]);

    float _norm000[3] = {
        __CALC_DX(x,y,z),
        __CALC_DY(x,y,z),
        __CALC_DZ(x,y,z)
    }; // unnormalized normal

    float _norm001[3];
    float _norm010[3];
    float _norm100[3];

    bool skips[] = {false, false, false};
    int norm_count = 0;


    #define __GRID_CONNECTED(s0, s1, s2, s3) (!(((s0 <= lv_set) && (s1 <= lv_set) && (s2 <= lv_set) && (s3 <= lv_set)) || \
                                                ((s0 >= lv_set) && (s1 >= lv_set) && (s2 >= lv_set) && (s3 >= lv_set))))

                                            
    // bool volatile ex1 = (__GRID_EXIST(x,y,z+1)); 
    // bool volatile con1 = (__GRID_CONNECTED(__FETCH_DATA(x,y,z+1), __FETCH_DATA(x,y+1,z+1), __FETCH_DATA(x+1,y,z+1), __FETCH_DATA(x+1,y+1,z+1))); 

    if ((__GRID_EXIST(x,y,z+1)) && \
        ((!con_check) || __GRID_CONNECTED(__FETCH_DATA(x,y,z+1), __FETCH_DATA(x,y+1,z+1), __FETCH_DATA(x+1,y,z+1), __FETCH_DATA(x+1,y+1,z+1))) && \
        ((!ignore_empty) || ((!empty000) && (!__CHECK_EMPTY(x,y,z+1))))
        ){
        _norm001[0] = __CALC_DX(x,y,z+1);
        _norm001[1] = __CALC_DY(x,y,z+1);
        _norm001[2] = __CALC_DZ(x,y,z+1);
        norm_count += 1;
    }else{
        skips[2] = true;
    }

    if ((__GRID_EXIST(x,y+1,z)) && \
        ((!con_check) || __GRID_CONNECTED(__FETCH_DATA(x,y+1,z), __FETCH_DATA(x,y+1,z+1), __FETCH_DATA(x+1,y+1,z), __FETCH_DATA(x+1,y+1,z+1))) && \
        ((!ignore_empty) || ((!empty000) && (!__CHECK_EMPTY(x,y+1,z))))
        ){
        _norm010[0] = __CALC_DX(x,y+1,z);
        _norm010[1] = __CALC_DY(x,y+1,z);
        _norm010[2] = __CALC_DZ(x,y+1,z);
        norm_count += 1;
    }else{
        skips[1] = true;
    }
    if ((__GRID_EXIST(x+1,y,z)) && \
        ((!con_check) || __GRID_CONNECTED(__FETCH_DATA(x+1,y,z), __FETCH_DATA(x+1,y,z+1), __FETCH_DATA(x+1,y+1,z), __FETCH_DATA(x+1,y+1,z+1))) && \
        ((!ignore_empty) || ((!empty000) && (!__CHECK_EMPTY(x+1,y,z))))
        ){
        _norm100[0] = __CALC_DX(x+1,y,z);
        _norm100[1] = __CALC_DY(x+1,y,z);
        _norm100[2] = __CALC_DZ(x+1,y,z);
        norm_count += 1;
    }else{
        skips[0] = true;
    }

    float const N0 = _NORM3(_norm000);
    // if ((isnan(N0) || (isinf(N0)))){
    //     printf("N0: %f\n", N0);
    //     printf("_norm000[0]: %f\n", _norm000[0]);
    //     printf("_norm000[1]: %f\n", _norm000[1]);
    //     printf("_norm000[2]: %f\n", _norm000[2]);
    // }
    ASSERT_NUM(N0);
    // apply normal difference loss gradient
    for (int i=0; i<3; ++i){
        if (skips[i]){
            continue;
        }
        float const *_n1 = (i==0) ? _norm100 : ((i==1) ? _norm010 : _norm001);
        float const N1 = _NORM3(_n1);
        ASSERT_NUM(N1);

        // dE/d0x, dE/d0y, dE/d0z
        float const d0[] = {
                (_norm000[0]/N0 - _n1[0]/N1) * \
                (-2.f*_SQR(_norm000[0])/_CUBIC(N0) + 2.f/N0) \ 
                + -2.f*_norm000[0]*_norm000[1]*(_norm000[1]/N0 - _n1[1]/N1) / _CUBIC(N0) \
                + -2.f*_norm000[0]*_norm000[2]*(_norm000[2]/N0 - _n1[2]/N1) / _CUBIC(N0),
                (_norm000[1]/N0 - _n1[1]/N1) * \ 
                (-2.f*_SQR(_norm000[1])/_CUBIC(N0) + 2.f/N0) \ 
                + -2.f*_norm000[0]*_norm000[1]*(_norm000[0]/N0 - _n1[0]/N1) / _CUBIC(N0) \
                + -2.f*_norm000[1]*_norm000[2]*(_norm000[2]/N0 - _n1[2]/N1) / _CUBIC(N0),
                (_norm000[2]/N0 - _n1[2]/N1) * \ 
                (-2.f*_SQR(_norm000[2])/_CUBIC(N0) + 2.f/N0) \ 
                + -2.f*_norm000[0]*_norm000[2]*(_norm000[0]/N0 - _n1[0]/N1) / _CUBIC(N0) \
                + -2.f*_norm000[1]*_norm000[2]*(_norm000[1]/N0 - _n1[1]/N1) / _CUBIC(N0)
        };

        ASSERT_NUM(d0[0]);
        ASSERT_NUM(d0[1]);
        ASSERT_NUM(d0[2]);

        _split_add_surface_norm_grad(x, y, z, d0, 
                          scale * 1.f/norm_count, links, offx, offy, ddim, idx, mask_out, grad_data);
        

        float const d1[] = {
                (_norm000[0]/N0 - _n1[0]/N1) * \ 
                (2.f*_SQR(_n1[0])/_CUBIC(N1) - 2.f/N1) \
                + 2.f*_n1[0]*_n1[1]*(_norm000[1]/N0 - _n1[1]/N1) / _CUBIC(N1) \
                + 2.f*_n1[0]*_n1[2]*(_norm000[2]/N0 - _n1[2]/N1) / _CUBIC(N1),
                (_norm000[1]/N0 - _n1[1]/N1) * \ 
                (2.f*_SQR(_n1[1])/_CUBIC(N1) - 2.f/N1) \ 
                + 2.f*_n1[0]*_n1[1]*(_norm000[0]/N0 - _n1[0]/N1) / _CUBIC(N1) \
                + 2.f*_n1[1]*_n1[2]*(_norm000[2]/N0 - _n1[2]/N1) / _CUBIC(N1),
                (_norm000[2]/N0 - _n1[2]/N1) * \ 
                (2.f*_SQR(_n1[2])/_CUBIC(N1) - 2.f/N1) \ 
                + 2.f*_n1[0]*_n1[2]*(_norm000[0]/N0 - _n1[0]/N1) / _CUBIC(N1) \
                + 2.f*_n1[1]*_n1[2]*(_norm000[1]/N0 - _n1[1]/N1) / _CUBIC(N1)
        };

        ASSERT_NUM(d1[0]);
        ASSERT_NUM(d1[1]);
        ASSERT_NUM(d1[2]);

        float const ux = (i==0) ? x+1:x,
                    uy = (i==1) ? y+1:y,
                    uz = (i==2) ? z+1:z;

        _split_add_surface_norm_grad(ux, uy, uz, d1,
                          scale * 1.f/norm_count, links, offx, offy, ddim, idx, mask_out, grad_data);

    }

    // // apply eikonal constraint gradient
    // if (eikonal_scale > 0.f){
    //     float const d_eki[] = {
    //         -2*_norm000[0]*(1 - N0)/N0,
    //         -2*_norm000[1]*(1 - N0)/N0,
    //         -2*_norm000[2]*(1 - N0)/N0
    //     };

    //     ASSERT_NUM(d_eki[0]);
    //     ASSERT_NUM(d_eki[1]);
    //     ASSERT_NUM(d_eki[2]);

    //     _add_surface_grad(x, y, z, d_eki, 
    //                       eikonal_scale, links, ddim, idx, nullptr, grad_data);
    // }
}



} // namespace device
} // namespace
