#pragma once
#include "cuda_util.cuh"
#include <cmath>
#include <cstdint>

#define _AXIS(x) (x>>1)
#define _ORI(x) (x&1)
#define _FACE(axis, ori) uint8_t((axis << 1) | ori)

namespace {
namespace device {

struct CubemapIndex {
    uint8_t face;
    float uv[2];
};

struct CubemapPointer {
    uint8_t face;
    uint16_t uv[2];
};

struct CubemapBilerpIndex {
    CubemapPointer ptr[2][2];
    float duv[2];
};

__device__ __host__ __inline__ CubemapIndex
    dir_to_cubemap_index(const float* __restrict__ xyz_o,
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

    CubemapIndex idx;
    idx.uv[0] = ((xyz[(ax ^ 1) & 1] + 1) * face_reso - 1) * 0.5;
    idx.uv[1] = ((xyz[(ax ^ 2) & 2] + 1) * face_reso - 1) * 0.5;
    const int ori = xyz[ax] >= 0;
    idx.face = _FACE(ax, ori);
    return idx;
}

__device__ __host__ __inline__ CubemapBilerpIndex
    cubemap_find_interp_pts(
                const CubemapIndex& idx,
                int face_reso) {
    const int uv_idx[2] ={ (int)floorf(idx.uv[0]), (int)floorf(idx.uv[1]) };

    bool m[2][2];
    m[0][0] = uv_idx[0] < 0;
    m[0][1] = uv_idx[0] > face_reso - 2;
    m[1][0] = uv_idx[1] < 0;
    m[1][1] = uv_idx[1] > face_reso - 2;

    const int ax = _AXIS(idx.face);
    const int ori = _ORI(idx.face);
    // if ax is one of {0, 1, 2}, this trick gets the 2
    //  of {0, 1, 2} other than ax
    const int uvd[2] = {((ax ^ 1) & 1), ((ax ^ 2) & 2)};
    int uv_ori[2];

    CubemapBilerpIndex result;

#pragma unroll 2
    for (uv_ori[0] = 0; uv_ori[0] < 2; ++uv_ori[0]) {
#pragma unroll 2
        for (uv_ori[1] = 0; uv_ori[1] < 2; ++uv_ori[1]) {
            CubemapPointer& nidx = result.ptr[uv_ori[0]][uv_ori[1]];
            nidx.face = idx.face;
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
                    continue;
                } else {
                    edge_idx = 0;
                }
            } else if (mv) {
                // Crosses edge in v-axis
                edge_idx = 1;
            }
            if (~edge_idx) {
                const int nax = uvd[edge_idx];
                const uint16_t other_coord = nidx.uv[1 - edge_idx];

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

    result.duv[0] = idx.uv[0] - uv_idx[0];
    result.duv[1] = idx.uv[1] - uv_idx[1];
    return result;
}

}  // namespace device
}  // namespace
