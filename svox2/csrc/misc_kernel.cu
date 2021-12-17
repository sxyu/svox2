// Copyright 2021 Alex Yu
// Miscellaneous kernels (3D mask dilate, weight thresholding)

#include <torch/extension.h>
#include <cstdint>
#include <cstdio>
#include <iostream>
#include "cuda_util.cuh"
#include "render_util.cuh"
#include "data_spec_packed.cuh"
#include "cubemap_util.cuh"

namespace {

const int MISC_CUDA_THREADS = 256;
const int MISC_MIN_BLOCKS_PER_SM = 4;
namespace device {

// Can also implement using convs.....
__launch_bounds__(MISC_CUDA_THREADS, MISC_MIN_BLOCKS_PER_SM)
__global__ void dilate_kernel(
        const torch::PackedTensorAccessor32<bool, 3, torch::RestrictPtrTraits> grid,
        // Output
        torch::PackedTensorAccessor32<bool, 3, torch::RestrictPtrTraits> out_grid) {
    CUDA_GET_THREAD_ID(tid, grid.size(0) * grid.size(1) * grid.size(2));

    const int z = tid % grid.size(2);
    const int xy = tid / grid.size(2);
    const int y = xy % grid.size(1);
    const int x = xy / grid.size(1);

    int xl = max(x - 1, 0), xr = min(x + 1, (int) grid.size(0) - 1);
    int yl = max(y - 1, 0), yr = min(y + 1, (int) grid.size(1) - 1);
    int zl = max(z - 1, 0), zr = min(z + 1, (int) grid.size(2) - 1);

    out_grid[x][y][z] =
        grid[xl][yl][zl] | grid[xl][yl][z] | grid[xl][yl][zr] |
        grid[xl][y][zl] | grid[xl][y][z] | grid[xl][y][zr] |
        grid[xl][yr][zl] | grid[xl][yr][z] | grid[xl][yr][zr] |

        grid[x][yl][zl] | grid[x][yl][z] | grid[x][yl][zr] |
        grid[x][y][zl] | grid[x][y][z] | grid[x][y][zr] |
        grid[x][yr][zl] | grid[x][yr][z] | grid[x][yr][zr] |

        grid[xr][yl][zl] | grid[xr][yl][z] | grid[xr][yl][zr] |
        grid[xr][y][zl] | grid[xr][y][z] | grid[xr][y][zr] |
        grid[xr][yr][zl] | grid[xr][yr][z] | grid[xr][yr][zr];
}

// Probably can speed up the following functions, however they are really not
// the bottleneck

// ** Distance transforms
// TODO: Maybe replace this with an euclidean distance transform eg PBA
// Actual L-infty distance transform; turns out this is slower than the geometric way
__launch_bounds__(MISC_CUDA_THREADS, MISC_MIN_BLOCKS_PER_SM)
__global__ void accel_linf_dist_transform_kernel(
        torch::PackedTensorAccessor32<int32_t, 3, torch::RestrictPtrTraits> grid,
        int32_t* __restrict__ tmp,
        int d2) {
    const int d0 = d2 == 0 ? 1 : 0;
    const int d1 = 3 - d0 - d2;
    CUDA_GET_THREAD_ID(tid, grid.size(d0) * grid.size(d1));
    int32_t* __restrict__ tmp_ptr = tmp + tid * grid.size(d2);
    int l[3];
    l[d0] = tid / grid.size(1);
    l[d1] = tid % grid.size(1);
    l[d2] = 0;
    const int INF = 0x3f3f3f3f;
    int d01_dist = INF;
    int d2_dist = INF;
    for (; l[d2] < grid.size(d2); ++l[d2]) {
        const int val = grid[l[0]][l[1]][l[2]];
        int32_t cdist = max(- val, 0);
        if (d2 == 0 && cdist)
            cdist = INF;
        const int32_t pdist = max(d2_dist, d01_dist);

        if (cdist < pdist) {
            d01_dist = cdist;
            d2_dist = 0;
        }
        tmp_ptr[l[d2]] = min(cdist, pdist);
        ++d2_dist;
    }

    l[d2] = grid.size(d2) - 1;
    d01_dist = INF;
    d2_dist = INF;
    for (; l[d2] >= 0; --l[d2]) {
        const int val = grid[l[0]][l[1]][l[2]];
        int32_t cdist = max(- val, 0);
        if (d2 == 0 && cdist)
            cdist = INF;
        const int32_t pdist = max(d2_dist, d01_dist);

        if (cdist < pdist) {
            d01_dist = cdist;
            d2_dist = 0;
        }
        if (cdist) {
            grid[l[0]][l[1]][l[2]] = -min(tmp_ptr[l[d2]], min(cdist, pdist));
        }
        ++d2_dist;
    }
}

// Geometric L-infty distance transform-ish thing
__launch_bounds__(MISC_CUDA_THREADS, MISC_MIN_BLOCKS_PER_SM)
__global__ void accel_dist_set_kernel(
        const torch::PackedTensorAccessor32<int32_t, 3, torch::RestrictPtrTraits> grid,
        bool* __restrict__ tmp) {
    int sz_x = grid.size(0);
    int sz_y = grid.size(1);
    int sz_z = grid.size(2);
    CUDA_GET_THREAD_ID(tid, sz_x * sz_y * sz_z);

    int z = tid % grid.size(2);
    const int xy = tid / grid.size(2);
    int y = xy % grid.size(1);
    int x = xy / grid.size(1);

    bool* tmp_base = tmp;

    if (grid[x][y][z] >= 0) {
        while (sz_x > 1 && sz_y > 1 && sz_z > 1) {
            // Propagate occupied cell throughout the temp tree parent nodes
            x >>= 1;
            y >>= 1;
            z >>= 1;
            sz_x = int_div2_ceil(sz_x);
            sz_y = int_div2_ceil(sz_y);
            sz_z = int_div2_ceil(sz_z);

            const int idx = x * sz_y * sz_z + y * sz_z + z;
            // printf("s %d  %d %d %d  %d\n", tid, x, y, z, idx);
            tmp_base[idx] = true;
            tmp_base += sz_x * sz_y * sz_z;
        }
    }
}

__launch_bounds__(MISC_CUDA_THREADS, MISC_MIN_BLOCKS_PER_SM)
__global__ void accel_dist_prop_kernel(
        torch::PackedTensorAccessor32<int32_t, 3, torch::RestrictPtrTraits> grid,
        const bool* __restrict__ tmp) {
    int sz_x = grid.size(0);
    int sz_y = grid.size(1);
    int sz_z = grid.size(2);
    CUDA_GET_THREAD_ID(tid, sz_x * sz_y * sz_z);

    int z = tid % grid.size(2);
    const int xy = tid / grid.size(2);
    int y = xy % grid.size(1);
    int x = xy / grid.size(1);
    const bool* tmp_base = tmp;
    int32_t* __restrict__ val = &grid[x][y][z];

    if (*val < 0) {
        int result = -1;
        while (sz_x > 1 && sz_y > 1 && sz_z > 1) {
            // Find the lowest set parent cell if it exists
            x >>= 1;
            y >>= 1;
            z >>= 1;
            sz_x = int_div2_ceil(sz_x);
            sz_y = int_div2_ceil(sz_y);
            sz_z = int_div2_ceil(sz_z);

            const int idx = x * sz_y * sz_z + y * sz_z + z;
            // printf("p %d  %d %d %d  %d tb[%d] , %d %d %d\n", tid, x, y, z, idx, tmp_base[idx],
            //         sz_x, sz_y, sz_z);
            if (tmp_base[idx]) {
                break;
            }
            result -= 1;
            tmp_base += sz_x * sz_y * sz_z;
        }
        *val = result;
    }
}

// Fast single-channel rendering for weight-thresholding
__device__ __inline__ void grid_trace_ray(
    const torch::PackedTensorAccessor32<float, 3, torch::RestrictPtrTraits>
        data,
        SingleRaySpec ray,
        const float* __restrict__ offset,
        const float* __restrict__ scaling,
        float step_size,
        float stop_thresh,
        bool last_sample_opaque,
    torch::PackedTensorAccessor32<float, 3, torch::RestrictPtrTraits>
        grid_weight) {

    // Warning: modifies ray.origin
    transform_coord(ray.origin, scaling, offset);
    // Warning: modifies ray.dir
    const float world_step = _get_delta_scale(scaling, ray.dir) * step_size;

    float t, tmax;
    {
        float t1, t2;
        t = 0.0f;
        tmax = 2e3f;
#pragma unroll 3
        for (int i = 0; i < 3; ++i) {
            const float invdir = 1.0 / ray.dir[i];
            t1 = (-0.5f - ray.origin[i]) * invdir;
            t2 = (data.size(i) - 0.5f  - ray.origin[i]) * invdir;
            if (ray.dir[i] != 0.f) {
                t = max(t, min(t1, t2));
                tmax = min(tmax, max(t1, t2));
            }
        }
    }

    if (t > tmax) {
        // Ray doesn't hit box
        return;
    }
    float pos[3];
    int32_t l[3];

    float log_light_intensity = 0.f;
    const int stride0 = data.size(1) * data.size(2);
    const int stride1 = data.size(2);
    while (t <= tmax) {
#pragma unroll 3
        for (int j = 0; j < 3; ++j) {
            pos[j] = ray.origin[j] + t * ray.dir[j];
            pos[j] = min(max(pos[j], 0.f), data.size(j) - 1.f);
            l[j] = (int32_t) pos[j];
            l[j] = min(l[j], data.size(j) - 2);
            pos[j] -= l[j];
        }

        float log_att;
        const int idx = l[0] * stride0 + l[1] * stride1 + l[2];

        float sigma;
        {
            // Trilerp
            const float* __restrict__ sigma000 = data.data() + idx;
            const float* __restrict__ sigma100 = sigma000 + stride0;
            const float ix0y0 = lerp(sigma000[0], sigma000[1], pos[2]);
            const float ix0y1 = lerp(sigma000[stride1], sigma000[stride1 + 1], pos[2]);
            const float ix1y0 = lerp(sigma100[0], sigma100[1], pos[2]);
            const float ix1y1 = lerp(sigma100[stride1], sigma100[stride1 + 1], pos[2]);
            const float ix0 = lerp(ix0y0, ix0y1, pos[1]);
            const float ix1 = lerp(ix1y0, ix1y1, pos[1]);
            sigma = lerp(ix0, ix1, pos[0]);
        }
        if (last_sample_opaque && t + step_size > tmax) {
            sigma += 1e9;
            log_light_intensity = 0.f;
        }

        if (sigma > 1e-8f) {
            log_att = -world_step * sigma;
            const float weight = _EXP(log_light_intensity) * (1.f - _EXP(log_att));
            log_light_intensity += log_att;
            float* __restrict__ max_wt_ptr_000 = grid_weight.data() + idx;
            atomicMax(max_wt_ptr_000, weight);
            atomicMax(max_wt_ptr_000 + 1, weight);
            atomicMax(max_wt_ptr_000 + stride1, weight);
            atomicMax(max_wt_ptr_000 + stride1 + 1, weight);
            float* __restrict__ max_wt_ptr_100 = max_wt_ptr_000 + stride0;
            atomicMax(max_wt_ptr_100, weight);
            atomicMax(max_wt_ptr_100 + 1, weight);
            atomicMax(max_wt_ptr_100 + stride1, weight);
            atomicMax(max_wt_ptr_100 + stride1 + 1, weight);

            if (_EXP(log_light_intensity) < stop_thresh) {
                break;
            }
        }
        t += step_size;
    }
}

// __global__ void sample_cubemap_kernel(
//     const torch::PackedTensorAccessor32<float, 4, torch::RestrictPtrTraits>
//         cubemap,
//     const torch::PackedTensorAccessor32<float, 2, torch::RestrictPtrTraits>
//         dirs,
//     int Q,
//     bool eac,
//     // Output
//     torch::PackedTensorAccessor32<float, 2, torch::RestrictPtrTraits>
//         result) {
//     CUDA_GET_THREAD_ID(tid, Q);
//
//     const int chnl_id = tid % cubemap.size(3);
//     const int ray_id = tid / cubemap.size(3);
//
//     const int face_reso = cubemap.size(1);
//
//     CubemapCoord coord = dir_to_cubemap_coord(dirs[ray_id].data(), face_reso, eac);
//     CubemapBilerpQuery query = cubemap_build_query(coord, face_reso);
//     result[ray_id][chnl_id] = cubemap_sample(
//             cubemap.data(),
//             query,
//             face_reso,
//             cubemap.size(3),
//             chnl_id);
// }

__launch_bounds__(MISC_CUDA_THREADS, MISC_MIN_BLOCKS_PER_SM)
__global__ void grid_weight_render_kernel(
    const torch::PackedTensorAccessor32<float, 3, torch::RestrictPtrTraits>
        data,
    PackedCameraSpec cam,
    float step_size,
    float stop_thresh,
    bool last_sample_opaque,
    const float* __restrict__ offset,
    const float* __restrict__ scaling,
    torch::PackedTensorAccessor32<float, 3, torch::RestrictPtrTraits>
        grid_weight) {
    CUDA_GET_THREAD_ID(tid, cam.width * cam.height);
    int iy = tid / cam.width, ix = tid % cam.width;
    float dir[3], origin[3];
    cam2world_ray(ix, iy, cam, dir, origin);
    grid_trace_ray(
        data,
        SingleRaySpec(origin, dir),
        offset,
        scaling,
        step_size,
        stop_thresh,
        last_sample_opaque,
        grid_weight);
}

}  // namespace device
}  // namespace

torch::Tensor dilate(torch::Tensor grid) {
    DEVICE_GUARD(grid);
    CHECK_INPUT(grid);
    TORCH_CHECK(!grid.is_floating_point());
    TORCH_CHECK(grid.ndimension() == 3);

    int Q = grid.size(0) * grid.size(1) * grid.size(2);

    const int blocks = CUDA_N_BLOCKS_NEEDED(Q, MISC_CUDA_THREADS);
    torch::Tensor result = torch::empty_like(grid);
    device::dilate_kernel<<<blocks, MISC_CUDA_THREADS>>>(
            grid.packed_accessor32<bool, 3, torch::RestrictPtrTraits>(),
            // Output
            result.packed_accessor32<bool, 3, torch::RestrictPtrTraits>());
    return result;
}

void accel_dist_prop(torch::Tensor grid) {
    // Grid here is links array from the sparse grid
    DEVICE_GUARD(grid);
    CHECK_INPUT(grid);
    TORCH_CHECK(!grid.is_floating_point());
    TORCH_CHECK(grid.ndimension() == 3);

    int64_t sz_x = grid.size(0);
    int64_t sz_y = grid.size(1);
    int64_t sz_z = grid.size(2);

    int Q = grid.size(0) * grid.size(1) * grid.size(2);

    const int blocks = CUDA_N_BLOCKS_NEEDED(Q, MISC_CUDA_THREADS);

    int64_t req_size = 0;
    while (sz_x > 1 && sz_y > 1 && sz_z > 1) {
        sz_x = int_div2_ceil(sz_x);
        sz_y = int_div2_ceil(sz_y);
        sz_z = int_div2_ceil(sz_z);
        req_size += sz_x * sz_y * sz_z;
    }

    auto tmp_options = torch::TensorOptions()
                  .dtype(torch::kBool)
                  .layout(torch::kStrided)
                  .device(grid.device())
                  .requires_grad(false);
    torch::Tensor tmp = torch::zeros({req_size}, tmp_options);
    device::accel_dist_set_kernel<<<blocks, MISC_CUDA_THREADS>>>(
            grid.packed_accessor32<int32_t, 3, torch::RestrictPtrTraits>(),
            tmp.data_ptr<bool>());

    device::accel_dist_prop_kernel<<<blocks, MISC_CUDA_THREADS>>>(
            grid.packed_accessor32<int32_t, 3, torch::RestrictPtrTraits>(),
            tmp.data_ptr<bool>());


    // int32_t* tmp;
    // const int req_size = sz_x * sz_y * sz_z;
    // cuda(Malloc(&tmp, req_size * sizeof(int32_t)));
    // cuda(MemsetAsync(tmp, 0, req_size * sizeof(int32_t)));
    //
    // {
    //     for (int d2 = 0; d2 < 3; ++d2) {
    //         int d0 = d2 == 0 ? 1 : 0;
    //         int d1 = 3 - d0 - d2;
    //         int Q = grid.size(d0) * grid.size(d1);
    //
    //         const int blocks = CUDA_N_BLOCKS_NEEDED(Q, MISC_CUDA_THREADS);
    //
    //         device::accel_linf_dist_transform_kernel<<<blocks, MISC_CUDA_THREADS>>>(
    //                 grid.packed_accessor32<int32_t, 3, torch::RestrictPtrTraits>(),
    //                 tmp,
    //                 d2);
    //     }
    // }

    // cuda(Free(tmp));
    CUDA_CHECK_ERRORS;
}

void grid_weight_render(
    torch::Tensor data, CameraSpec& cam,
    float step_size,
    float stop_thresh,
    bool last_sample_opaque,
    torch::Tensor offset, torch::Tensor scaling,
    torch::Tensor grid_weight_out) {
    DEVICE_GUARD(data);
    CHECK_INPUT(data);
    CHECK_INPUT(offset);
    CHECK_INPUT(scaling);
    CHECK_INPUT(grid_weight_out);
    cam.check();
    const size_t Q = size_t(cam.width) * cam.height;

    const int blocks = CUDA_N_BLOCKS_NEEDED(Q, MISC_CUDA_THREADS);

    device::grid_weight_render_kernel<<<blocks, MISC_CUDA_THREADS>>>(
        data.packed_accessor32<float, 3, torch::RestrictPtrTraits>(),
        cam,
        step_size,
        stop_thresh,
        last_sample_opaque,
        offset.data_ptr<float>(),
        scaling.data_ptr<float>(),
        grid_weight_out.packed_accessor32<float, 3, torch::RestrictPtrTraits>());
    CUDA_CHECK_ERRORS;
}

// For debugging
// void sample_cubemap(torch::Tensor cubemap, // (6, R, R, C)
//                     torch::Tensor dirs,
//                     bool eac,
//                     // Output
//                     torch::Tensor result) {
//     DEVICE_GUARD(cubemap);
//     CHECK_INPUT(cubemap);
//     CHECK_INPUT(dirs);
//     CHECK_INPUT(result);
//     TORCH_CHECK(cubemap.ndimension() == 4);
//     TORCH_CHECK(cubemap.size(0) == 6);
//     TORCH_CHECK(cubemap.size(1) == cubemap.size(2));
//
//     const size_t Q = size_t(dirs.size(0)) * cubemap.size(3);
//     const int blocks = CUDA_N_BLOCKS_NEEDED(Q, MISC_CUDA_THREADS);
//
//     device::sample_cubemap_kernel<<<blocks, MISC_CUDA_THREADS>>>(
//         cubemap.packed_accessor32<float, 4, torch::RestrictPtrTraits>(),
//         dirs.packed_accessor32<float, 2, torch::RestrictPtrTraits>(),
//         Q,
//         eac,
//         // Output
//         result.packed_accessor32<float, 2, torch::RestrictPtrTraits>());
//     CUDA_CHECK_ERRORS;
// }
