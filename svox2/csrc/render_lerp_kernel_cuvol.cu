// Copyright 2021 Alex Yu
#include <torch/extension.h>
#include "cuda_util.cuh"
#include "random_util.cuh"
#include "data_spec_packed.cuh"
#include "render_util.cuh"

#include <cstdint>
#include <tuple>

namespace {
const int WARP_SIZE = 32;
const int TRACE_RAY_CUDA_THREADS = 768;
const int TRACE_RAY_CUDA_RAYS_PER_BLOCK = TRACE_RAY_CUDA_THREADS / WARP_SIZE;
const int TRACE_RAY_BKWD_CUDA_THREADS = 448;
const int TRACE_RAY_BKWD_CUDA_RAYS_PER_BLOCK = TRACE_RAY_BKWD_CUDA_THREADS / WARP_SIZE;
typedef cub::WarpReduce<float> WarpReducef;

namespace device {

template<class data_index_t>
__device__ __inline__ float trilerp_cuvol_one(
        torch::PackedTensorAccessor32<int32_t, 3, torch::RestrictPtrTraits> links,
        torch::GenericPackedTensorAccessor<float, 2, torch::RestrictPtrTraits, data_index_t> data,
        const int32_t* __restrict__ l,
        const float* __restrict__ pos,
        const int idx) {
    const float val000 = (links[l[0]][l[1]][l[2]] >= 0 ?
                          data[links[l[0]][l[1]][l[2]]][idx] : 0.f),

               val001 = (links[l[0]][l[1]][l[2] + 1] >= 0 ?
                          data[links[l[0]][l[1]][l[2] + 1]][idx] : 0.f),

               val010 = (links[l[0]][l[1] + 1][l[2]] >= 0 ?
                          data[links[l[0]][l[1] + 1][l[2]]][idx] : 0.f),

               val011 = (links[l[0]][l[1] + 1][l[2] + 1] >= 0 ?
                          data[links[l[0]][l[1] + 1][l[2] + 1]][idx] : 0.f),

               val100 = (links[l[0] + 1][l[1]][l[2]] >= 0 ?
                          data[links[l[0] + 1][l[1]][l[2]]][idx] : 0.f),

               val101 = (links[l[0] + 1][l[1]][l[2] + 1] >= 0 ?
                          data[links[l[0] + 1][l[1]][l[2] + 1]][idx] : 0.f),

               val110 = (links[l[0] + 1][l[1] + 1][l[2]] >= 0 ?
                          data[links[l[0] + 1][l[1] + 1][l[2]]][idx] : 0.f),

               val111 = (links[l[0] + 1][l[1] + 1][l[2] + 1] >= 0 ?
                          data[links[l[0] + 1][l[1] + 1][l[2] + 1]][idx] : 0.f);
    const float ix0y0 = lerp(val000, val001, pos[2]);
    const float ix0y1 = lerp(val010, val011, pos[2]);
    const float ix1y0 = lerp(val100, val101, pos[2]);
    const float ix1y1 = lerp(val110, val111, pos[2]);
    const float ix0 = lerp(ix0y0, ix0y1, pos[1]);
    const float ix1 = lerp(ix1y0, ix1y1, pos[1]);
    return lerp(ix0, ix1, pos[0]);
}

__device__ __inline__ void trilerp_backward_cuvol_one(
        torch::PackedTensorAccessor32<int32_t, 3, torch::RestrictPtrTraits> links,
        float* __restrict__ grad_data,
        const int32_t* __restrict__ l,
        const float* __restrict__ pos,
        float grad_out,
        const int64_t stride,
        const int idx) {
    const float ay = 1.f - pos[1], az = 1.f - pos[2];
    float xo = (1.0f - pos[0]) * grad_out;
    if (links[l[0]][l[1]][l[2]] >= 0) {
        atomicAdd(grad_data + stride * links[l[0]][l[1]][l[2]] + idx, ay * az * xo);
    }
    if (links[l[0]][l[1]][l[2] + 1] >= 0) {
        atomicAdd(grad_data + stride * links[l[0]][l[1]][l[2] + 1] + idx, ay * pos[2] * xo);
    }
    if (links[l[0]][l[1] + 1][l[2]] >= 0) {
        atomicAdd(grad_data + stride * links[l[0]][l[1] + 1][l[2]] + idx, pos[1] * az * xo);
    }
    if (links[l[0]][l[1] + 1][l[2] + 1] >= 0) {
        atomicAdd(grad_data + stride * links[l[0]][l[1] + 1][l[2] + 1] + idx, pos[1] * pos[2] * xo);
    }

    xo = pos[0] * grad_out;
    if (links[l[0] + 1][l[1]][l[2]] >= 0) {
        atomicAdd(grad_data + stride * links[l[0] + 1][l[1]][l[2]] + idx, ay * az * xo);
    }
    if (links[l[0] + 1][l[1]][l[2] + 1] >= 0) {
        atomicAdd(grad_data + stride * links[l[0] + 1][l[1]][l[2] + 1] + idx, ay * pos[2] * xo);
    }
    if (links[l[0] + 1][l[1] + 1][l[2]] >= 0) {
        atomicAdd(grad_data + stride * links[l[0] + 1][l[1] + 1][l[2]] + idx, pos[1] * az * xo);
    }
    if (links[l[0] + 1][l[1] + 1][l[2] + 1] >= 0) {
        atomicAdd(grad_data + stride * links[l[0] + 1][l[1] + 1][l[2] + 1] + idx, pos[1] * pos[2] * xo);
    }
}


// * For ray rendering
__device__ __inline__ void trace_ray_cuvol(
        const PackedSparseGridSpec& __restrict__ grid,
        const SingleRaySpec& __restrict__ ray,
        const RenderOptions& __restrict__ opt,
        uint32_t lane_id,
        float* __restrict__ sphfunc_val,
        WarpReducef::TempStorage& __restrict__ temp_storage,
        float* __restrict__ out) {
    if (lane_id >= grid.sh_data.size(1))
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
    float pos[3];
    int32_t l[3];
    while (t <= ray.tmax) {
#pragma unroll 3
        for (int j = 0; j < 3; ++j) {
            pos[j] = ray.origin[j] + t * ray.dir[j];
            pos[j] = min(max(pos[j], 0.f), grid.links.size(j) - 1.f);
            l[j] = (int32_t) pos[j];
            l[j] = min(l[j], (int32_t)grid.links.size(j) - 2);
            pos[j] -= l[j];
        }

        const float sigma = trilerp_cuvol_one(grid.links, grid.density_data, l, pos, 0);
        if (sigma > opt.sigma_thresh) {
            float lane_color = trilerp_cuvol_one(grid.links,
                            grid.sh_data, l, pos, lane_id);
            lane_color *= sphfunc_val[lane_colorgrp_id];

            const float pcnt = ray.world_step * sigma;
            const float weight = _EXP(light_intensity) * (1.f - _EXP(-pcnt));
            light_intensity -= pcnt;

            float lane_color_total = WarpReducef(temp_storage).HeadSegmentedSum(
                                           lane_color, lane_colorgrp_id == 0);
            outv += weight * _SIGMOID(lane_color_total);
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
        const SingleRaySpec& __restrict__ ray,
        const RenderOptions& __restrict__ opt,
        uint32_t lane_id,
        float* __restrict__ sphfunc_val,
        WarpReducef::TempStorage& __restrict__ temp_storage,
        float* __restrict__ grad_density_data_out,
        float* __restrict__ grad_sh_data_out
        ) {
    if (lane_id >= grid.sh_data.size(1))
        return;
    const uint32_t lane_colorgrp_id = lane_id % grid.basis_dim;
    const uint32_t lane_colorgrp = lane_id / grid.basis_dim;

    if (ray.tmin > ray.tmax) return;
    float t = ray.tmin;

    float pos[3];
    int32_t l[3];
    float gout = grad_output[lane_colorgrp];

    float accum = color_cache[0] * grad_output[0] +
                  color_cache[1] * grad_output[1] +
                  color_cache[2] * grad_output[2];

    float light_intensity = 0.f;
    // remat samples
    while (t <= ray.tmax) {
#pragma unroll 3
        for (int j = 0; j < 3; ++j) {
            pos[j] = ray.origin[j] + t * ray.dir[j];
            pos[j] = min(max(pos[j], 0.f), grid.links.size(j) - 1.f);
            l[j] = (int32_t) pos[j];
            l[j] = min(l[j], grid.links.size(j) - 2);
            pos[j] -= l[j];
        }

        const float sigma = trilerp_cuvol_one(grid.links, grid.density_data, l, pos, 0);
        if (sigma > opt.sigma_thresh) {
            float lane_color = trilerp_cuvol_one(grid.links,
                            grid.sh_data, l, pos, lane_id);
            lane_color *= sphfunc_val[lane_colorgrp_id];

            const float pcnt = ray.world_step * sigma;
            const float weight = _EXP(light_intensity) * (1.f - _EXP(-pcnt));
            light_intensity -= pcnt;

            const float lane_color_total = WarpReducef(temp_storage).HeadSegmentedSum(
                                           lane_color, lane_colorgrp_id == 0);
            float sigmoid = _SIGMOID(lane_color_total);

            float total_color = sigmoid * gout;
            float total_color_c1 = __shfl_sync(0xffffffff, total_color, grid.basis_dim);
            total_color += __shfl_sync(0xffffffff, total_color, 2 * grid.basis_dim);
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
            trilerp_backward_cuvol_one(grid.links, grad_sh_data_out, l, pos,
                    curr_grad_color, grid.sh_data.size(1), lane_id);
            if (lane_id == 0) {
                trilerp_backward_cuvol_one(grid.links, grad_density_data_out,
                        l, pos, curr_grad_sigma, 1, 0);
            }
        }
        t += opt.step_size;
    }
}


// BEGIN KERNELS

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

__global__ void render_ray_backward_kernel(
    PackedSparseGridSpec grid,
    const torch::PackedTensorAccessor32<float, 2, torch::RestrictPtrTraits>
        grad_output,
    torch::PackedTensorAccessor32<float, 2, torch::RestrictPtrTraits> color_cache,
        PackedRaysSpec rays,
        RenderOptions opt,
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
        grad_density_data_out,
        grad_sh_data_out);
}

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
    if (lane_id == 0) {
        ray_spec[ray_blk_id].set(origin, dir);
        calc_sphfunc(SPHFUNC_TYPE_SH, grid.basis_dim,
                     dir, sphfunc_val[ray_blk_id]);
        ray_find_bounds(ray_spec[ray_blk_id], grid, opt);
    }
    trace_ray_cuvol(
        grid,
        ray_spec[ray_blk_id],
        opt,
        lane_id,
        sphfunc_val[ray_blk_id],
        temp_storage[ray_blk_id],
        &out[iy][ix][0]);
}

__global__ void render_image_backward_kernel(
        PackedSparseGridSpec grid,
        const torch::PackedTensorAccessor32<float, 3, torch::RestrictPtrTraits>
            grad_output,
        torch::PackedTensorAccessor32<float, 3, torch::RestrictPtrTraits> color_cache,
        PackedCameraSpec cam,
        RenderOptions opt,
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
    if (lane_id == 0) {
        ray_spec[ray_blk_id].set(origin, dir);
        calc_sphfunc(SPHFUNC_TYPE_SH, grid.basis_dim,
                     dir, sphfunc_val[ray_blk_id]);
        ray_find_bounds(ray_spec[ray_blk_id], grid, opt);
    }
    trace_ray_cuvol_backward(
        grid,
        grad_output[iy][ix].data(),
        color_cache[iy][ix].data(),
        ray_spec[ray_blk_id],
        opt,
        lane_id,
        sphfunc_val[ray_blk_id],
        temp_storage[ray_blk_id],
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

std::tuple<torch::Tensor,torch::Tensor> volume_render_cuvol_backward(
        SparseGridSpec& grid,
        RaysSpec& rays,
        RenderOptions& opt,
        torch::Tensor grad_out,
        torch::Tensor color_cache) {

    DEVICE_GUARD(grid.sh_data);
    grid.check();
    rays.check();
    const auto Q = rays.origins.size(0);

    const int cuda_n_threads_render_backward = TRACE_RAY_BKWD_CUDA_THREADS;
    const int blocks = CUDA_N_BLOCKS_NEEDED(Q * WARP_SIZE, cuda_n_threads_render_backward);

    torch::Tensor result_density = torch::zeros_like(grid.density_data);
    torch::Tensor result_sh = torch::zeros_like(grid.sh_data);
    device::render_ray_backward_kernel<<<blocks,
           cuda_n_threads_render_backward>>>(
            grid,
            grad_out.packed_accessor32<float, 2, torch::RestrictPtrTraits>(),
            color_cache.packed_accessor32<float, 2, torch::RestrictPtrTraits>(),
            rays, opt,
            // Output
            result_density.data_ptr<float>(),
            result_sh.data_ptr<float>());
    CUDA_CHECK_ERRORS;
    return std::tuple<torch::Tensor, torch::Tensor>{result_density, result_sh};
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

std::tuple<torch::Tensor,torch::Tensor> volume_render_cuvol_image_backward(
        SparseGridSpec& grid,
        CameraSpec& cam,
        RenderOptions& opt,
        torch::Tensor grad_out,
        torch::Tensor color_cache) {

    DEVICE_GUARD(grid.sh_data);
    grid.check();
    cam.check();
    const size_t Q = size_t(cam.width) * cam.height;

    const int cuda_n_threads_render_backward = TRACE_RAY_BKWD_CUDA_THREADS;
    const int blocks = CUDA_N_BLOCKS_NEEDED(Q * WARP_SIZE, cuda_n_threads_render_backward);

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
            result_density.data_ptr<float>(),
            result_sh.data_ptr<float>());
    CUDA_CHECK_ERRORS;
    return std::tuple<torch::Tensor, torch::Tensor>{result_density, result_sh};
}
