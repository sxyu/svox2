#pragma once
#include "cuda_util.cuh"
#include <cmath>
#include <cstdint>

#define _AXIS(x) (x>>1)
#define _ORI(x) (x&1)
#define _FACE(axis, ori) uint8_t((axis << 1) | ori)

namespace {
namespace device {

struct CubemapCoord {
    uint8_t face;
    float uv[2];
};

struct CubemapLocation {
    uint8_t face;
    int16_t uv[2];
};

struct CubemapBilerpQuery {
    CubemapLocation ptr[2][2];
    float duv[2];
};

struct ConcentricSpheresIntersector {
    __device__
        ConcentricSpheresIntersector(
                const int* size,
                const float* __restrict__ rorigin,
                const float* __restrict__ rdir,
                float rworld_step)
    {
        const float sphere_scaling[3] {
            2.f / float(size[0]),
            2.f / float(size[1]),
            2.f / float(size[2])
        };

#pragma unroll 3
        for (int i = 0; i < 3; ++i) {
            origin[i] = fmaf(rorigin[i] + 0.5f, sphere_scaling[i], -1.f);
            dir[i] = rdir[i] * sphere_scaling[i];
        }
        float inorm = 1.f / _norm(dir);
        world_step_scale = rworld_step * inorm;
#pragma unroll 3
        for (int i = 0; i < 3; ++i) {
            dir[i] *= inorm;
        }

        q2a = 2 * _dot(dir, dir);
        qb = 2 * _dot(origin, dir);
        f = qb * qb - 2 * q2a * _dot(origin, origin);
    }

    // Get the far intersection, which we want for rendering MSI
    __device__
    bool intersect(float r, float* __restrict__ out) {
        float det = _det(r);
        if (det < 0) return false;
        *out = (-qb + sqrtf(det)) / q2a;
        return true;
    }

    __device__
    bool intersect_near(float r, float* __restrict__ out) {
        float det = _det(r);
        if (det < 0) return false;
        *out = (-qb - sqrtf(det)) / q2a;
        return true;
    }

    __device__ __host__
    float _det (float r) {
        return f + 2 * q2a * r * r;
    }

    float origin[3], dir[3];
    float world_step_scale;
    float q2a, qb, f;
};

__device__ __host__ __inline__ CubemapCoord
    dir_to_cubemap_coord(const float* __restrict__ xyz_o,
            int face_reso,
            bool eac = true) {
    float maxv;
    int ax;
    float xyz[3] = {xyz_o[0], xyz_o[1], xyz_o[2]};
    if (fabsf(xyz[0]) > fabsf(xyz[1]) && fabsf(xyz[0]) > fabsf(xyz[2])) {
        ax = 0; maxv = xyz[0];
    } else if (fabsf(xyz[1]) > fabsf(xyz[2])) {
        ax = 1; maxv = xyz[1];
    } else {
        ax = 2; maxv = xyz[2];
    }
    const float recip = 1.f / fabsf(maxv);
    xyz[0] *= recip;
    xyz[1] *= recip;
    xyz[2] *= recip;

    if (eac) {
#pragma unroll 3
        for (int i = 0; i < 3; ++i) {
            xyz[i] = atanf(xyz[i]) * (4 * M_1_PI);
        }
    }

    CubemapCoord idx;
    idx.uv[0] = ((xyz[(ax ^ 1) & 1] + 1) * face_reso - 1) * 0.5;
    idx.uv[1] = ((xyz[(ax ^ 2) & 2] + 1) * face_reso - 1) * 0.5;
    const int ori = xyz[ax] >= 0;
    idx.face = _FACE(ax, ori);

    return idx;
}

__device__ __host__ __inline__ CubemapBilerpQuery
    cubemap_build_query(
                const CubemapCoord& idx,
                int face_reso) {
    const int uv_idx[2] ={ (int)floorf(idx.uv[0]), (int)floorf(idx.uv[1]) };

    bool m[2][2];
    m[0][0] = uv_idx[0] < 0;
    m[0][1] = uv_idx[0] > face_reso - 2;
    m[1][0] = uv_idx[1] < 0;
    m[1][1] = uv_idx[1] > face_reso - 2;

    const int face = idx.face;
    const int ax = _AXIS(face);
    const int ori = _ORI(face);
    // if ax is one of {0, 1, 2}, this trick gets the 2
    //  of {0, 1, 2} other than ax
    const int uvd[2] = {((ax ^ 1) & 1), ((ax ^ 2) & 2)};
    int uv_ori[2];

    CubemapBilerpQuery result;
    result.duv[0] = idx.uv[0] - uv_idx[0];
    result.duv[1] = idx.uv[1] - uv_idx[1];

#pragma unroll 2
    for (uv_ori[0] = 0; uv_ori[0] < 2; ++uv_ori[0]) {
#pragma unroll 2
        for (uv_ori[1] = 0; uv_ori[1] < 2; ++uv_ori[1]) {
            CubemapLocation& nidx = result.ptr[uv_ori[0]][uv_ori[1]];
            nidx.face = face;
            nidx.uv[0] = uv_idx[0] + uv_ori[0];
            nidx.uv[1] = uv_idx[1] + uv_ori[1];

            const bool mu = m[0][uv_ori[0]];
            const bool mv = m[1][uv_ori[1]];

            int edge_idx = -1;
            if (mu) {
                // Crosses edge in u-axis
                if (mv) {
                    // FIXME: deal with corners properly, right now
                    // just clamps, resulting in a little artifact
                    // at each cube corner
                    nidx.uv[0] = min(max(nidx.uv[0], 0), face_reso - 1);
                    nidx.uv[1] = min(max(nidx.uv[1], 0), face_reso - 1);
                } else {
                    edge_idx = 0;
                }
            } else if (mv) {
                // Crosses edge in v-axis
                edge_idx = 1;
            }
            if (~edge_idx) {
                const int nax = uvd[edge_idx];
                const int16_t other_coord = nidx.uv[1 - edge_idx];

                // Determine directions in the new face
                const int nud = (nax ^ 1) & 1;
                // const int nvd = (nax ^ 2) & 2;

                if (nud == ax) {
                    nidx.uv[0] = ori ? (face_reso - 1) : 0;
                    nidx.uv[1] = other_coord;
                } else {
                    nidx.uv[0] = other_coord;
                    nidx.uv[1] = ori ? (face_reso - 1) : 0;
                }

                nidx.face = _FACE(nax, uv_ori[edge_idx]);
            }
            // Interior point: nothing needs to be done

        }
    }

    return result;
}

__device__ __host__ __inline__ float
    cubemap_sample(
                const float* __restrict__ cubemap, // (6, face_reso, face_reso, n_channels)
                const CubemapBilerpQuery& query,
                int face_reso,
                int n_channels,
                int chnl_id) {

        // NOTE: assuming address will fit in int32
        const int stride1 = face_reso * n_channels;
        const int stride0 = face_reso * stride1;
        const CubemapLocation& p00 = query.ptr[0][0];
        const float v00 = cubemap[p00.face * stride0  + p00.uv[0] * stride1 + p00.uv[1] * n_channels + chnl_id];
        const CubemapLocation& p01 = query.ptr[0][1];
        const float v01 = cubemap[p01.face * stride0  + p01.uv[0] * stride1 + p01.uv[1] * n_channels + chnl_id];
        const CubemapLocation& p10 = query.ptr[1][0];
        const float v10 = cubemap[p10.face * stride0  + p10.uv[0] * stride1 + p10.uv[1] * n_channels + chnl_id];
        const CubemapLocation& p11 = query.ptr[1][1];
        const float v11 = cubemap[p11.face * stride0  + p11.uv[0] * stride1 + p11.uv[1] * n_channels + chnl_id];

        const float val0 = lerp(v00, v01, query.duv[1]);
        const float val1 = lerp(v10, v11, query.duv[1]);

        return lerp(val0, val1, query.duv[0]);
    }

__device__ __inline__ void
    cubemap_sample_backward(
                float* __restrict__ cubemap_grad, // (6, face_reso, face_reso, n_channels)
                const CubemapBilerpQuery& query,
                int face_reso,
                int n_channels,
                float grad_out,
                int chnl_id,
                bool* __restrict__ mask_out = nullptr) {

        // NOTE: assuming address will fit in int32
        const float bu = query.duv[0], bv = query.duv[1];
        const float au = 1.f - bu, av = 1.f - bv;

#define _ADD_CUBEVERT(i, j, val) { \
            const CubemapLocation& p00 = query.ptr[i][j]; \
            const int idx = (p00.face * face_reso + p00.uv[0]) * face_reso + p00.uv[1]; \
            float* __restrict__ v00 = &cubemap_grad[idx * n_channels + chnl_id]; \
            atomicAdd(v00, val); \
            if (mask_out != nullptr) { \
                mask_out[idx] = true; \
            } \
        }

        _ADD_CUBEVERT(0, 0, au * av * grad_out);
        _ADD_CUBEVERT(0, 1, au * bv * grad_out);
        _ADD_CUBEVERT(1, 0, bu * av * grad_out);
        _ADD_CUBEVERT(1, 1, bu * bv * grad_out);
#undef _ADD_CUBEVERT

    }

}  // namespace device
}  // namespace
