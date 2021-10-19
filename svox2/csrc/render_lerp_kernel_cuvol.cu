// Copyright 2021 Alex Yu
#include <torch/extension.h>
#include "cuda_util.cuh"
#include "random_util.cuh"
#include "data_spec_packed.cuh"
#include "render_util.cuh"

#include <iostream>
#include <cstdint>
#include <tuple>

namespace {
const int WARP_SIZE = 32;

const int TRACE_RAY_CUDA_THREADS = 256;
const int TRACE_RAY_CUDA_RAYS_PER_BLOCK = TRACE_RAY_CUDA_THREADS / WARP_SIZE;

const int TRACE_RAY_BKWD_CUDA_THREADS = 256;
const int TRACE_RAY_BKWD_CUDA_RAYS_PER_BLOCK = TRACE_RAY_BKWD_CUDA_THREADS / WARP_SIZE;

const int TRACE_RAY_FUSED_CUDA_THREADS = 256;
const int TRACE_RAY_FUSED_CUDA_RAYS_PER_BLOCK = TRACE_RAY_FUSED_CUDA_THREADS / WARP_SIZE;

const int MIN_BLOCKS_PER_SM = 4;
typedef cub::WarpReduce<float> WarpReducef;

torch::Tensor init_mask(torch::Tensor ref_data) {
    auto mask_options =
        torch::TensorOptions()
            .dtype(at::ScalarType::Bool)
            .layout(torch::kStrided)
            .device(ref_data.device())
            .requires_grad(false);
    return torch::zeros({ref_data.size(0)}, mask_options);
}

namespace device {

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


// * For ray rendering
__device__ __inline__ void trace_ray_cuvol(
        const PackedSparseGridSpec& __restrict__ grid,
        SingleRaySpec& __restrict__ ray,
        const RenderOptions& __restrict__ opt,
        uint32_t lane_id,
        float* __restrict__ sphfunc_val,
        WarpReducef::TempStorage& __restrict__ temp_storage,
        float* __restrict__ out) {
    if (lane_id >= grid.sh_data_dim)
        return;
    const uint32_t lane_colorgrp_id = lane_id % grid.basis_dim;
    const uint32_t lane_colorgrp = lane_id / grid.basis_dim;

    if (ray.tmin > ray.tmax) {
        out[lane_colorgrp] = opt.background_brightness;
        return;
    }

    float t = ray.tmin;
    float outv = 0.f;

    float light_intensity = 0.f;
    while (t <= ray.tmax) {
#pragma unroll 3
        for (int j = 0; j < 3; ++j) {
            ray.pos[j] = fmaf(t, ray.dir[j], ray.origin[j]);
            ray.pos[j] = min(max(ray.pos[j], 0.f), grid.size[j] - 1.f);
            ray.l[j] = min(static_cast<int32_t>(ray.pos[j]), grid.size[j] - 2);
            ray.pos[j] -= static_cast<float>(ray.l[j]);
        }

        const float sigma = trilerp_cuvol_one(
                grid.links, grid.density_data,
                grid.stride_x,
                grid.size[2],
                1,
                ray.l, ray.pos,
                0);
        if (sigma > opt.sigma_thresh) {
            float lane_color = trilerp_cuvol_one(
                            grid.links,
                            grid.sh_data,
                            grid.stride_x,
                            grid.size[2],
                            grid.sh_data_dim,
                            ray.l, ray.pos, lane_id);
            lane_color *= sphfunc_val[lane_colorgrp_id];

            const float pcnt = ray.world_step * sigma;
            const float weight = _EXP(light_intensity) * (1.f - _EXP(-pcnt));
            light_intensity -= pcnt;

            float lane_color_total = WarpReducef(temp_storage).HeadSegmentedSum(
                                           lane_color, lane_colorgrp_id == 0);
            outv += weight * _SIGMOID(lane_color_total);
            if (_EXP(light_intensity) < opt.stop_thresh) {
                break;
            }
        }
        t += opt.step_size;
    }
    outv += _EXP(light_intensity) * opt.background_brightness;
    if (lane_colorgrp_id == 0) {
        out[lane_colorgrp] = outv;
    }
}

__device__ __inline__ void trace_ray_cuvol_backward(
        const PackedSparseGridSpec& __restrict__ grid,
        const float* __restrict__ grad_output,
        const float* __restrict__ color_cache,
        SingleRaySpec& __restrict__ ray,
        const RenderOptions& __restrict__ opt,
        uint32_t lane_id,
        float* __restrict__ sphfunc_val,
        WarpReducef::TempStorage& __restrict__ temp_storage,
        bool* __restrict__ mask_out,
        float* __restrict__ grad_density_data_out,
        float* __restrict__ grad_sh_data_out
        ) {
    if (lane_id >= grid.sh_data_dim)
        return;
    const uint32_t lane_colorgrp_id = lane_id % grid.basis_dim;
    const uint32_t lane_colorgrp = lane_id / grid.basis_dim;
    const uint32_t leader_mask = 1U | (1U << grid.basis_dim) | (1U << (2 * grid.basis_dim));

    if (ray.tmin > ray.tmax) return;
    float t = ray.tmin;

    const float gout = grad_output[lane_colorgrp];

    float accum = fmaf(color_cache[0], grad_output[0],
                      fmaf(color_cache[1], grad_output[1],
                           color_cache[2] * grad_output[2]));

    float light_intensity = 0.f;
    // remat samples
    while (t <= ray.tmax) {
#pragma unroll 3
        for (int j = 0; j < 3; ++j) {
            ray.pos[j] = fmaf(t, ray.dir[j], ray.origin[j]);
            ray.pos[j] = min(max(ray.pos[j], 0.f), grid.size[j] - 1.f);
            ray.l[j] = min(static_cast<int32_t>(ray.pos[j]), grid.size[j] - 2);
            ray.pos[j] -= static_cast<float>(ray.l[j]);
        }

        const float sigma = trilerp_cuvol_one(
                grid.links,
                grid.density_data,
                grid.stride_x,
                grid.size[2],
                1,
                ray.l, ray.pos,
                0);
        if (sigma > opt.sigma_thresh) {
            float lane_color = trilerp_cuvol_one(
                            grid.links,
                            grid.sh_data,
                            grid.stride_x,
                            grid.size[2],
                            grid.sh_data_dim,
                            ray.l, ray.pos, lane_id);
            lane_color *= sphfunc_val[lane_colorgrp_id];

            const float pcnt = ray.world_step * sigma;
            const float weight = _EXP(light_intensity) * (1.f - _EXP(-pcnt));
            light_intensity -= pcnt;

            const float lane_color_total = WarpReducef(temp_storage).HeadSegmentedSum(
                                           lane_color, lane_colorgrp_id == 0);
            float sigmoid = _SIGMOID(lane_color_total);

            float total_color = sigmoid * gout;
            float total_color_c1 = __shfl_sync(leader_mask, total_color, grid.basis_dim);
            total_color += __shfl_sync(leader_mask, total_color, 2 * grid.basis_dim);
            total_color += total_color_c1;

            sigmoid = __shfl_sync(0xffffffff, sigmoid, lane_colorgrp * grid.basis_dim);
            const float curr_grad_color = sphfunc_val[lane_colorgrp_id] *
                                weight * sigmoid *
                                (1.f - sigmoid) * gout;

//             float total_color = 0.f;
// #pragma unroll 3
//             for (int j = 0; j < 3; ++j) {
//                 const int off = j * grid.basis_dim + 1;
//                 float tmp = 0.f;
//                 for (int i = 0; i < grid.basis_dim; ++i) {
//                     tmp += sphfunc_val[i] * interp_val[off + i];
//                 }
//                 const float sigmoid = _SIGMOID(tmp);
//                 total_color += sigmoid * grad_output[j];
//
//                 const float tmp2 = weight * sigmoid * (1.f - sigmoid)  * grad_output[j];
//                 for (int i = 0; i < grid.basis_dim; ++i) {
//                     curr_grad[off + i] = sphfunc_val[i] * tmp2;
//                 }
//             }
            accum -= weight * total_color;
            float curr_grad_sigma = ray.world_step * (
                    total_color * _EXP(light_intensity) - accum);
            trilerp_backward_cuvol_one(grid.links, grad_sh_data_out,
                    grid.stride_x,
                    grid.size[2],
                    grid.sh_data_dim,
                    ray.l, ray.pos,
                    curr_grad_color, lane_id);
            if (lane_id == 0) {
                trilerp_backward_cuvol_one_density(
                        grid.links,
                        grad_density_data_out,
                        mask_out,
                        grid.stride_x,
                        grid.size[2],
                        ray.l, ray.pos, curr_grad_sigma);
            }
            if (_EXP(light_intensity) < opt.stop_thresh) {
                break;
            }
        }
        t += opt.step_size;
    }
}


// BEGIN KERNELS

__launch_bounds__(TRACE_RAY_CUDA_THREADS, MIN_BLOCKS_PER_SM)
__global__ void render_ray_kernel(
        PackedSparseGridSpec grid,
        PackedRaysSpec rays,
        RenderOptions opt,
        torch::PackedTensorAccessor32<float, 2, torch::RestrictPtrTraits> out) {
    CUDA_GET_THREAD_ID(tid, int(rays.origins.size(0)) * WARP_SIZE);
    const int ray_id = tid >> 5;
    const int ray_blk_id = threadIdx.x >> 5;
    const int lane_id = threadIdx.x & 0x1F;
    __shared__ float sphfunc_val[TRACE_RAY_CUDA_RAYS_PER_BLOCK][9];
    __shared__ SingleRaySpec ray_spec[TRACE_RAY_CUDA_RAYS_PER_BLOCK];
    __shared__ typename WarpReducef::TempStorage temp_storage[
        TRACE_RAY_CUDA_RAYS_PER_BLOCK];
    if (lane_id == 0) {
        ray_spec[ray_blk_id].set(rays.origins[ray_id].data(),
                rays.dirs[ray_id].data());
        calc_sphfunc(SPHFUNC_TYPE_SH, grid.basis_dim,
                     ray_spec[ray_blk_id].dir, sphfunc_val[ray_blk_id]);
        ray_find_bounds(ray_spec[ray_blk_id], grid, opt);
    }
    // printf("ray %d lane %d\n", ray_id, lane_id);
    trace_ray_cuvol(
        grid,
        ray_spec[ray_blk_id],
        opt,
        lane_id,
        sphfunc_val[ray_blk_id],
        temp_storage[ray_blk_id],
        out[ray_id].data());
}

__launch_bounds__(TRACE_RAY_BKWD_CUDA_THREADS, MIN_BLOCKS_PER_SM)
__global__ void render_ray_backward_kernel(
    PackedSparseGridSpec grid,
    const torch::PackedTensorAccessor32<float, 2, torch::RestrictPtrTraits>
        grad_output,
    torch::PackedTensorAccessor32<float, 2, torch::RestrictPtrTraits> color_cache,
        PackedRaysSpec rays,
        RenderOptions opt,
    bool* __restrict__ mask_out,
    float* __restrict__ grad_density_data_out,
    float* __restrict__ grad_sh_data_out
        ) {
    CUDA_GET_THREAD_ID(tid, int(rays.origins.size(0)) * WARP_SIZE);
    const int ray_id = tid >> 5;
    const int ray_blk_id = threadIdx.x >> 5;
    const int lane_id = threadIdx.x & 0x1F;
    __shared__ float sphfunc_val[TRACE_RAY_BKWD_CUDA_RAYS_PER_BLOCK][9];
    __shared__ SingleRaySpec ray_spec[TRACE_RAY_BKWD_CUDA_RAYS_PER_BLOCK];
    __shared__ typename WarpReducef::TempStorage temp_storage[
        TRACE_RAY_CUDA_RAYS_PER_BLOCK];
    if (lane_id == 0) {
        ray_spec[ray_blk_id].set(rays.origins[ray_id].data(),
                                 rays.dirs[ray_id].data());
        calc_sphfunc(SPHFUNC_TYPE_SH, grid.basis_dim,
                     ray_spec[ray_blk_id].dir, sphfunc_val[ray_blk_id]);
        ray_find_bounds(ray_spec[ray_blk_id], grid, opt);
    }
    trace_ray_cuvol_backward(
        grid,
        grad_output[ray_id].data(),
        color_cache[ray_id].data(),
        ray_spec[ray_blk_id],
        opt,
        lane_id,
        sphfunc_val[ray_blk_id],
        temp_storage[ray_blk_id],
        mask_out,
        grad_density_data_out,
        grad_sh_data_out);
}

__launch_bounds__(TRACE_RAY_FUSED_CUDA_THREADS, MIN_BLOCKS_PER_SM)
__global__ void render_ray_fused_kernel(
        PackedSparseGridSpec grid,
        PackedRaysSpec rays,
        RenderOptions opt,
        const float* __restrict__ rgb_gt,
        bool* __restrict__ mask_out,
        float* __restrict__ rgb_out,
        float* __restrict__ grad_density_data_out,
        float* __restrict__ grad_sh_data_out) {
    CUDA_GET_THREAD_ID(tid, int(rays.origins.size(0)) * WARP_SIZE);
    const int ray_id = tid >> 5;
    const int ray_blk_id = threadIdx.x >> 5;
    const int lane_id = threadIdx.x & 0x1F;

    if (lane_id >= grid.sh_data_dim)
        return;

    __shared__ float sphfunc_val[TRACE_RAY_FUSED_CUDA_RAYS_PER_BLOCK][9];
    __shared__ float rgb_val[TRACE_RAY_FUSED_CUDA_RAYS_PER_BLOCK][3];
    __shared__ float grad_out[TRACE_RAY_FUSED_CUDA_RAYS_PER_BLOCK][3];
    __shared__ SingleRaySpec ray_spec[TRACE_RAY_FUSED_CUDA_RAYS_PER_BLOCK];
    __shared__ typename WarpReducef::TempStorage temp_storage[2][
        TRACE_RAY_CUDA_RAYS_PER_BLOCK];
    ray_spec[ray_blk_id].set(rays.origins[ray_id].data(),
            rays.dirs[ray_id].data());
    calc_sphfunc(SPHFUNC_TYPE_SH, grid.basis_dim,
                 ray_spec[ray_blk_id].dir, sphfunc_val[ray_blk_id]);
    ray_find_bounds(ray_spec[ray_blk_id], grid, opt);

    trace_ray_cuvol(
        grid,
        ray_spec[ray_blk_id],
        opt,
        lane_id,
        sphfunc_val[ray_blk_id],
        temp_storage[0][ray_blk_id],
        rgb_val[ray_blk_id]);

#pragma unroll 3
    for (int i = 0; i < 3; ++i) {
        const float resid = rgb_val[ray_blk_id][i] - rgb_gt[ray_id * 3 + i];
        grad_out[ray_blk_id][i] = 2.f * resid / (3 * int(rays.origins.size(0)));
    }
    trace_ray_cuvol_backward(
        grid,
        grad_out[ray_blk_id],
        rgb_val[ray_blk_id],
        ray_spec[ray_blk_id],
        opt,
        lane_id,
        sphfunc_val[ray_blk_id],
        temp_storage[1][ray_blk_id],
        mask_out,
        grad_density_data_out,
        grad_sh_data_out);
    if (lane_id < 3) {
        rgb_out[ray_id * 3 + lane_id] = rgb_val[ray_blk_id][lane_id];
    }
}

__launch_bounds__(TRACE_RAY_CUDA_THREADS, MIN_BLOCKS_PER_SM)
__global__ void render_image_kernel(
        PackedSparseGridSpec grid,
        PackedCameraSpec cam,
        RenderOptions opt,
        torch::PackedTensorAccessor32<float, 3, torch::RestrictPtrTraits> out) {
    CUDA_GET_THREAD_ID(tid, cam.width * cam.height * WARP_SIZE);
    const int ray_id = tid >> 5;
    const int ray_blk_id = threadIdx.x >> 5;
    const int lane_id = threadIdx.x & 0x1F;
    int iy = ray_id / cam.width, ix = ray_id % cam.width;
    float dir[3], origin[3];
    cam2world_ray(ix, iy, dir, origin, cam);
    __shared__ float sphfunc_val[TRACE_RAY_CUDA_RAYS_PER_BLOCK][9];
    __shared__ typename WarpReducef::TempStorage temp_storage[
        TRACE_RAY_CUDA_RAYS_PER_BLOCK];
    __shared__ SingleRaySpec ray_spec[TRACE_RAY_CUDA_RAYS_PER_BLOCK];
    ray_spec[ray_blk_id].set(origin, dir);
    calc_sphfunc(SPHFUNC_TYPE_SH, grid.basis_dim,
                 dir, sphfunc_val[ray_blk_id]);
    ray_find_bounds(ray_spec[ray_blk_id], grid, opt);
    trace_ray_cuvol(
        grid,
        ray_spec[ray_blk_id],
        opt,
        lane_id,
        sphfunc_val[ray_blk_id],
        temp_storage[ray_blk_id],
        &out[iy][ix][0]);
}

__launch_bounds__(TRACE_RAY_BKWD_CUDA_THREADS, MIN_BLOCKS_PER_SM)
__global__ void render_image_backward_kernel(
        PackedSparseGridSpec grid,
        const torch::PackedTensorAccessor32<float, 3, torch::RestrictPtrTraits>
            grad_output,
        torch::PackedTensorAccessor32<float, 3, torch::RestrictPtrTraits> color_cache,
        PackedCameraSpec cam,
        RenderOptions opt,
        bool* __restrict__ mask_out,
        float* __restrict__ grad_density_data_out,
        float* __restrict__ grad_sh_data_out) {
    CUDA_GET_THREAD_ID(tid, cam.width * cam.height * WARP_SIZE);
    const int ray_id = tid >> 5;
    const int ray_blk_id = threadIdx.x >> 5;
    const int lane_id = threadIdx.x & 0x1F;
    int iy = ray_id / cam.width, ix = ray_id % cam.width;
    float dir[3], origin[3];
    cam2world_ray(ix, iy, dir, origin, cam);
    __shared__ float sphfunc_val[TRACE_RAY_BKWD_CUDA_RAYS_PER_BLOCK][9];
    __shared__ SingleRaySpec ray_spec[TRACE_RAY_BKWD_CUDA_RAYS_PER_BLOCK];
    __shared__ typename WarpReducef::TempStorage temp_storage[
        TRACE_RAY_CUDA_RAYS_PER_BLOCK];
    ray_spec[ray_blk_id].set(origin, dir);
    calc_sphfunc(SPHFUNC_TYPE_SH, grid.basis_dim,
                 dir, sphfunc_val[ray_blk_id]);
    ray_find_bounds(ray_spec[ray_blk_id], grid, opt);
    trace_ray_cuvol_backward(
        grid,
        grad_output[iy][ix].data(),
        color_cache[iy][ix].data(),
        ray_spec[ray_blk_id],
        opt,
        lane_id,
        sphfunc_val[ray_blk_id],
        temp_storage[ray_blk_id],
        mask_out,
        grad_density_data_out,
        grad_sh_data_out);
}
}  // namespace device
}  // namespace

torch::Tensor volume_render_cuvol(SparseGridSpec& grid, RaysSpec& rays, RenderOptions& opt) {
    DEVICE_GUARD(grid.sh_data);
    grid.check();
    rays.check();

    const auto Q = rays.origins.size(0);

    const int cuda_n_threads = TRACE_RAY_CUDA_THREADS;
    const int blocks = CUDA_N_BLOCKS_NEEDED(Q * WARP_SIZE, cuda_n_threads);
    torch::Tensor results = torch::empty_like(rays.origins);
    // printf("CB %d %d\n", cuda_n_threads, blocks);
    device::render_ray_kernel<<<blocks, cuda_n_threads>>>(
            grid, rays, opt,
            // Output
            results.packed_accessor32<float, 2, torch::RestrictPtrTraits>());
    CUDA_CHECK_ERRORS;
    return results;
}

torch::Tensor volume_render_cuvol_backward(
        SparseGridSpec& grid,
        RaysSpec& rays,
        RenderOptions& opt,
        torch::Tensor grad_out,
        torch::Tensor color_cache,
        torch::Tensor grad_density_out,
        torch::Tensor grad_sh_out) {

    DEVICE_GUARD(grid.sh_data);
    grid.check();
    rays.check();
    CHECK_INPUT(grad_density_out);
    CHECK_INPUT(grad_sh_out);
    const auto Q = rays.origins.size(0);

    const int cuda_n_threads_render_backward = TRACE_RAY_BKWD_CUDA_THREADS;
    const int blocks = CUDA_N_BLOCKS_NEEDED(Q * WARP_SIZE, cuda_n_threads_render_backward);

    torch::Tensor sparse_mask = init_mask(grid.density_data);

    device::render_ray_backward_kernel<<<blocks,
           cuda_n_threads_render_backward>>>(
            grid,
            grad_out.packed_accessor32<float, 2, torch::RestrictPtrTraits>(),
            color_cache.packed_accessor32<float, 2, torch::RestrictPtrTraits>(),
            rays, opt,
            // Output
            sparse_mask.data_ptr<bool>(),
            grad_density_out.data_ptr<float>(),
            grad_sh_out.data_ptr<float>());
    
    CUDA_CHECK_ERRORS;
    return sparse_mask;
}

torch::Tensor volume_render_cuvol_fused(
        SparseGridSpec& grid,
        RaysSpec& rays,
        RenderOptions& opt,
        torch::Tensor rgb_gt,
        torch::Tensor rgb_out,
        torch::Tensor grad_density_out,
        torch::Tensor grad_sh_out) {

    DEVICE_GUARD(grid.sh_data);
    CHECK_INPUT(rgb_gt);
    CHECK_INPUT(rgb_out);
    CHECK_INPUT(grad_density_out);
    CHECK_INPUT(grad_sh_out);
    TORCH_CHECK(grad_density_out.size(0) == grid.density_data.size(0));
    TORCH_CHECK(grad_sh_out.size(0) == grid.sh_data.size(0));
    grid.check();
    rays.check();
    const auto Q = rays.origins.size(0);

    const int blocks = CUDA_N_BLOCKS_NEEDED(Q * WARP_SIZE, TRACE_RAY_FUSED_CUDA_THREADS);

    torch::Tensor sparse_mask = init_mask(grid.density_data);

    device::render_ray_fused_kernel<<<blocks, TRACE_RAY_FUSED_CUDA_THREADS>>>(
            grid,
            rays,
            opt,
            rgb_gt.data_ptr<float>(),
            // Output
            sparse_mask.data_ptr<bool>(),
            rgb_out.data_ptr<float>(),
            grad_density_out.data_ptr<float>(),
            grad_sh_out.data_ptr<float>());
    CUDA_CHECK_ERRORS;
    return sparse_mask;
}

torch::Tensor volume_render_cuvol_image(SparseGridSpec& grid, CameraSpec& cam, RenderOptions& opt) {
    DEVICE_GUARD(grid.sh_data);
    grid.check();
    cam.check();

    const size_t Q = size_t(cam.width) * cam.height;

    const int cuda_n_threads = TRACE_RAY_CUDA_THREADS;
    const int blocks = CUDA_N_BLOCKS_NEEDED(Q * WARP_SIZE, cuda_n_threads);
    torch::Tensor result = torch::zeros({cam.height, cam.width, 3}, grid.sh_data.options());

    device::render_image_kernel<<<blocks, cuda_n_threads>>>(
            grid, cam, opt,
            // Output
            result.packed_accessor32<float, 3, torch::RestrictPtrTraits>());
    CUDA_CHECK_ERRORS;
    return result;
}

torch::Tensor volume_render_cuvol_image_backward(
        SparseGridSpec& grid,
        CameraSpec& cam,
        RenderOptions& opt,
        torch::Tensor grad_out,
        torch::Tensor color_cache,
        torch::Tensor grad_density_out,
        torch::Tensor grad_sh_out) {

    DEVICE_GUARD(grid.sh_data);
    grid.check();
    cam.check();
    CHECK_INPUT(grad_density_out);
    CHECK_INPUT(grad_sh_out);
    const size_t Q = size_t(cam.width) * cam.height;

    const int cuda_n_threads_render_backward = TRACE_RAY_BKWD_CUDA_THREADS;
    const int blocks = CUDA_N_BLOCKS_NEEDED(Q * WARP_SIZE, cuda_n_threads_render_backward);

    torch::Tensor sparse_mask = init_mask(grid.density_data);

    torch::Tensor result_density = torch::zeros_like(grid.density_data);
    torch::Tensor result_sh = torch::zeros_like(grid.sh_data);
    device::render_image_backward_kernel<<<blocks,
           cuda_n_threads_render_backward>>>(
            grid,
            grad_out.packed_accessor32<float, 3, torch::RestrictPtrTraits>(),
            color_cache.packed_accessor32<float, 3, torch::RestrictPtrTraits>(),
            cam,
            opt,
            // Output
            sparse_mask.data_ptr<bool>(),
            grad_density_out.data_ptr<float>(),
            grad_sh_out.data_ptr<float>());

    CUDA_CHECK_ERRORS;
    return sparse_mask;
}
