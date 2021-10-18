// Copyright 2021 Alex Yu
#include <torch/extension.h>
#include "cuda_util.cuh"
#include "render_util.cuh"
#include "random_util.cuh"
#include "data_spec_packed.cuh"

#include <cstdint>
#include <tuple>

namespace {
const int TRACE_RAY_CUDA_THREADS = 768;
const int TRACE_RAY_CUDA_RAYS_PER_BLOCK = TRACE_RAY_CUDA_THREADS / 32;
typedef cub::WarpReduce<float> WarpReducef;

namespace device {

__device__ __constant__ const float EMPTY_CELL_DATA[] = {
    0.f, 0.f, 0.f, 0.f, 0.f,
    0.f, 0.f, 0.f, 0.f, 0.f,
    0.f, 0.f, 0.f, 0.f, 0.f,
    0.f, 0.f, 0.f, 0.f, 0.f,
    0.f, 0.f, 0.f, 0.f, 0.f,
    0.f, 0.f,
};

template<class data_index_t>
__device__ __inline__ void trilerp_cuvol(
        torch::PackedTensorAccessor32<int32_t, 3, torch::RestrictPtrTraits> links,
        torch::GenericPackedTensorAccessor<float, 2, torch::RestrictPtrTraits, data_index_t> data,
        const int32_t* __restrict__ l,
        const float* __restrict__ pos,
        float* __restrict__ out,
        const int min_idx,
        const int max_idx) {
    const float* ptr000 = (links[l[0]][l[1]][l[2]] >= 0 ?
                          &data[links[l[0]][l[1]][l[2]]][0] : EMPTY_CELL_DATA),

               * ptr001 = (links[l[0]][l[1]][l[2] + 1] >= 0 ?
                          &data[links[l[0]][l[1]][l[2] + 1]][0] : EMPTY_CELL_DATA),

               * ptr010 = (links[l[0]][l[1] + 1][l[2]] >= 0 ?
                          &data[links[l[0]][l[1] + 1][l[2]]][0] : EMPTY_CELL_DATA),

               * ptr011 = (links[l[0]][l[1] + 1][l[2] + 1] >= 0 ?
                          &data[links[l[0]][l[1] + 1][l[2] + 1]][0] : EMPTY_CELL_DATA),

               * ptr100 = (links[l[0] + 1][l[1]][l[2]] >= 0 ?
                          &data[links[l[0] + 1][l[1]][l[2]]][0] : EMPTY_CELL_DATA),

               * ptr101 = (links[l[0] + 1][l[1]][l[2] + 1] >= 0 ?
                          &data[links[l[0] + 1][l[1]][l[2] + 1]][0] : EMPTY_CELL_DATA),

               * ptr110 = (links[l[0] + 1][l[1] + 1][l[2]] >= 0 ?
                          &data[links[l[0] + 1][l[1] + 1][l[2]]][0] : EMPTY_CELL_DATA),

               * ptr111 = (links[l[0] + 1][l[1] + 1][l[2] + 1] >= 0 ?
                          &data[links[l[0] + 1][l[1] + 1][l[2] + 1]][0] : EMPTY_CELL_DATA);
#pragma unroll
    for (int j = min_idx; j < max_idx; ++j) {
        const float ix0y0 = lerp(ptr000[j], ptr001[j], pos[2]);
        const float ix0y1 = lerp(ptr010[j], ptr011[j], pos[2]);
        const float ix1y0 = lerp(ptr100[j], ptr101[j], pos[2]);
        const float ix1y1 = lerp(ptr110[j], ptr111[j], pos[2]);
        const float ix0 = lerp(ix0y0, ix0y1, pos[1]);
        const float ix1 = lerp(ix1y0, ix1y1, pos[1]);
        out[j] = lerp(ix0, ix1, pos[0]);
    }
}

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

__device__ __inline__ void trilerp_backward_cuvol(
        torch::PackedTensorAccessor32<int32_t, 3, torch::RestrictPtrTraits> links,
        float* __restrict__ grad_data,
        const int32_t* __restrict__ l,
        const float* __restrict__ pos,
        const float* __restrict__ grad_out,
        const int64_t stride,
        const int max_idx) {
    const float ax = 1.f - pos[0], ay = 1.f - pos[1], az = 1.f - pos[2];
    const float bx = pos[0], by = pos[1], bz = pos[2];

    float zeros[27];
    float* ptr000 = links[l[0]][l[1]][l[2]] >= 0 ?
                    grad_data + stride * links[l[0]][l[1]][l[2]] : zeros,

         * ptr001 = links[l[0]][l[1]][l[2] + 1] >= 0 ?
                    grad_data + stride * links[l[0]][l[1]][l[2] + 1] : zeros,

         * ptr010 = links[l[0]][l[1] + 1][l[2]] >= 0 ?
                    grad_data + stride * links[l[0]][l[1] + 1][l[2]] : zeros,

         * ptr011 = links[l[0]][l[1] + 1][l[2] + 1] >= 0 ?
                    grad_data + stride * links[l[0]][l[1] + 1][l[2] + 1] : zeros,

         * ptr100 = links[l[0] + 1][l[1]][l[2]] >= 0 ?
                    grad_data + stride * links[l[0] + 1][l[1]][l[2]] : zeros,

         * ptr101 = links[l[0] + 1][l[1]][l[2] + 1] >= 0 ?
                    grad_data + stride * links[l[0] + 1][l[1]][l[2] + 1] : zeros,

         * ptr110 = links[l[0] + 1][l[1] + 1][l[2]] >= 0 ?
                    grad_data + stride * links[l[0] + 1][l[1] + 1][l[2]] : zeros,

         * ptr111 = links[l[0] + 1][l[1] + 1][l[2] + 1] >= 0 ?
                    grad_data + stride * links[l[0] + 1][l[1] + 1][l[2] + 1] : zeros;

#pragma unroll
    for (int j = 0; j < max_idx; ++j) {
        const float axo = ax * grad_out[j];
        atomicAdd(ptr000 + j, ay * az * axo);
        atomicAdd(ptr001 + j, ay * bz * axo);
        atomicAdd(ptr010 + j, by * az * axo);
        atomicAdd(ptr011 + j, by * bz * axo);
        const float bxo = bx * grad_out[j];
        atomicAdd(ptr100 + j, ay * az * bxo);
        atomicAdd(ptr101 + j, ay * bz * bxo);
        atomicAdd(ptr110 + j, by * az * bxo);
        atomicAdd(ptr111 + j, by * bz * bxo);
    }
}

__device__ __inline__ bool intersect_global_aabb(
            const PackedSparseGridSpec& __restrict__ grid,
            SingleRaySpec& __restrict__ ray,
            float* __restrict__ t, float* __restrict__ tmax
        ) {
    *t = 0.0f; 
    *tmax = 1e9f; 
#pragma unroll 3     
    for (int i = 0; i < 3; ++i) { 
        const float invdir = 1.0 / ray.dir[i]; 
        const float t1 = (- ray.origin[i]) * invdir; 
        const float t2 = (grid.links.size(i) - 1.f  - ray.origin[i]) * invdir;
        if (ray.dir[i] != 0.f) {
            *t = max(*t, min(t1, t2)); 
            *tmax = min(*tmax, max(t1, t2)); 
        }
    } 
    return *t > *tmax;
}


// * For ray rendering
__device__ __inline__ void trace_ray_cuvol(
        const PackedSparseGridSpec& __restrict__ grid,
        SingleRaySpec& __restrict__ ray,
        RenderOptions& __restrict__ opt,
        uint32_t lane_id,
        float* __restrict__ sphfunc_val,
        WarpReducef::TempStorage& __restrict__ temp_storage,
        float* __restrict__ out) {
    if (lane_id >= grid.sh_data.size(1))
        return;
    const uint32_t lane_colorgrp_id = lane_id % grid.basis_dim;
    const uint32_t lane_colorgrp = lane_id / grid.basis_dim;

    // Warning: modifies ray.origin
    transform_coord(ray.origin, grid._scaling, grid._offset);
    // Warning: modifies ray.dir
    const float world_step = _get_delta_scale(grid._scaling, ray.dir) * opt.step_size;

    float t, tmax;
    if (intersect_global_aabb(grid, ray, &t, &tmax)) {
        // Ray doesn't hit box
        out[0] = out[1] = out[2] = opt.background_brightness;
        return;
    }
    float outv = 0.f;

    if (lane_id == 0) {
        calc_sh(grid.basis_dim, ray.vdir, sphfunc_val);
    }

    float light_intensity = 0.f;
    float pos[3];
    int32_t l[3];
    while (t <= tmax) {
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

            const float pcnt = world_step * sigma;
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
        const torch::TensorAccessor<float, 1, torch::RestrictPtrTraits, int32_t>
            grad_output,
        const torch::TensorAccessor<float, 1, torch::RestrictPtrTraits, int32_t>
            color_cache,
            SingleRaySpec ray,
            RenderOptions& __restrict__ opt,
        float* __restrict__ grad_density_data_out,
        float* __restrict__ grad_sh_data_out
        ) {

    // Warning: modifies ray.origin
    transform_coord(ray.origin, grid._scaling, grid._offset);
    // Warning: modifies ray.dir
    const float world_step = _get_delta_scale(grid._scaling, ray.dir) * opt.step_size;

    float t, tmax;
    if (intersect_global_aabb(grid, ray, &t, &tmax))
        return;

    float sphfunc_val[9];
    calc_sh(grid.basis_dim, ray.vdir, sphfunc_val);

    float pos[3], interp_val[28];
    int32_t l[3];
    float accum = color_cache[0] * grad_output[0] +
                  color_cache[1] * grad_output[1] +
                  color_cache[2] * grad_output[2];

    float light_intensity = 0.f;
    float curr_grad[28];
    // remat samples
    while (t <= tmax) {
#pragma unroll 3
        for (int j = 0; j < 3; ++j) {
            pos[j] = ray.origin[j] + t * ray.dir[j];
            pos[j] = min(max(pos[j], 0.f), grid.links.size(j) - 1.f);
            l[j] = (int32_t) pos[j];
            l[j] = min(l[j], grid.links.size(j) - 2);
            pos[j] -= l[j];
        }

        trilerp_cuvol(grid.links, grid.density_data, l, pos, interp_val, 0, 1);
        float sigma = interp_val[0];
        if (sigma > opt.sigma_thresh) {
            trilerp_cuvol(grid.links, grid.sh_data, l, pos, interp_val + 1, 0,
                          grid.sh_data.size(1));
            const float weight = __expf(light_intensity) * (1.f - __expf(
                        -world_step * sigma));
            light_intensity -= world_step * sigma;

            float total_color = 0.f;
#pragma unroll 3
            for (int j = 0; j < 3; ++j) {
                const int off = j * grid.basis_dim + 1;
                float tmp = 0.f;
                for (int i = 0; i < grid.basis_dim; ++i) {
                    tmp += sphfunc_val[i] * interp_val[off + i];
                }
                const float sigmoid = _SIGMOID(tmp);
                total_color += sigmoid * grad_output[j];

                const float tmp2 = weight * sigmoid * (1.f - sigmoid)  * grad_output[j];
                for (int i = 0; i < grid.basis_dim; ++i) {
                    curr_grad[off + i] = sphfunc_val[i] * tmp2;
                }
            }
            accum -= weight * total_color;
            curr_grad[0] = world_step * (
                    total_color * __expf(light_intensity) - accum);
            trilerp_backward_cuvol(grid.links, grad_density_data_out, l, pos, curr_grad,
                                   1, 1);
            trilerp_backward_cuvol(grid.links, grad_sh_data_out, l, pos, curr_grad + 1,
                                   grid.sh_data.size(1), grid.sh_data.size(1));
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
    CUDA_GET_THREAD_ID(tid, rays.origins.size(0) * 32);
    const int ray_id = tid >> 5;
    const int ray_blk_id = threadIdx.x >> 5;
    const int lane_id = threadIdx.x & 0x1F;
    __shared__ float sphfunc_val[TRACE_RAY_CUDA_RAYS_PER_BLOCK][9];
    __shared__ SingleRaySpec ray_spec[TRACE_RAY_CUDA_RAYS_PER_BLOCK];
    __shared__ typename WarpReducef::TempStorage temp_storage[
        TRACE_RAY_CUDA_RAYS_PER_BLOCK];
    if (lane_id == 0) {
        ray_spec[ray_blk_id].set(rays.origins[ray_id].data(), rays.dirs[ray_id].data());
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
    CUDA_GET_THREAD_ID(tid, rays.origins.size(0));
    trace_ray_cuvol_backward(
        grid,
        grad_output[tid],
        color_cache[tid],
        SingleRaySpec(&rays.origins[tid][0], &rays.dirs[tid][0]),
        opt,
        grad_density_data_out,
        grad_sh_data_out);
}

__global__ void render_image_kernel(
        PackedSparseGridSpec grid,
        PackedCameraSpec cam,
        RenderOptions opt,
        torch::PackedTensorAccessor32<float, 3, torch::RestrictPtrTraits> out) {
    CUDA_GET_THREAD_ID(tid, cam.width * cam.height * 32);
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
    CUDA_GET_THREAD_ID(tid, cam.width * cam.height);
    int iy = tid / cam.width, ix = tid % cam.width;
    float dir[3], origin[3];
    cam2world_ray(ix, iy, dir, origin, cam);
    trace_ray_cuvol_backward(
        grid,
        grad_output[iy][ix],
        color_cache[iy][ix],
        SingleRaySpec(origin, dir),
        opt,
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
    const int blocks = CUDA_N_BLOCKS_NEEDED(Q * 32, cuda_n_threads);
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

    const int cuda_n_threads_render_backward = 448;
    const int blocks = CUDA_N_BLOCKS_NEEDED(Q, cuda_n_threads_render_backward);

    torch::Tensor result_density = torch::zeros_like(grid.density_data);
    torch::Tensor result_sh = torch::zeros_like(grid.sh_data);
    device::render_ray_backward_kernel<<<blocks,
           cuda_n_threads_render_backward>>>(
            grid,
            grad_out.packed_accessor32<float, 2, torch::RestrictPtrTraits>(),
            color_cache.packed_accessor32<float, 2, torch::RestrictPtrTraits>(),
            rays, opt,
            // Output
            result_density.data<float>(),
            result_sh.data<float>());
    CUDA_CHECK_ERRORS;
    return std::tuple<torch::Tensor, torch::Tensor>{result_density, result_sh};
}

torch::Tensor volume_render_cuvol_image(SparseGridSpec& grid, CameraSpec& cam, RenderOptions& opt) {
    DEVICE_GUARD(grid.sh_data);
    grid.check();
    cam.check();

    const size_t Q = size_t(cam.width) * cam.height;

    const int cuda_n_threads = TRACE_RAY_CUDA_THREADS;
    const int blocks = CUDA_N_BLOCKS_NEEDED(Q * 32, cuda_n_threads);
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

    const int cuda_n_threads_render_backward = 448;
    const int blocks = CUDA_N_BLOCKS_NEEDED(Q, cuda_n_threads_render_backward);

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
            result_density.data<float>(),
            result_sh.data<float>());
    CUDA_CHECK_ERRORS;
    return std::tuple<torch::Tensor, torch::Tensor>{result_density, result_sh};
}
