// Copyright 2021 Alex Yu
#include <torch/extension.h>
#include "cuda_util.cuh"
#include "render_util.cuh"
#include "random_util.cuh"
#include "data_spec_packed.cuh"

#include <cstdint>
#include <tuple>

namespace {
namespace device {

__device__ __constant__ const float EMPTY_CELL_DATA[] = {
    0.f, 0.f, 0.f, 0.f, 0.f,
    0.f, 0.f, 0.f, 0.f, 0.f,
    0.f, 0.f, 0.f, 0.f, 0.f,
    0.f, 0.f, 0.f, 0.f, 0.f,
    0.f, 0.f, 0.f, 0.f, 0.f,
    0.f, 0.f, 0.f,
};

__device__ __inline__ void trilerp_cuvol(
        torch::PackedTensorAccessor32<int32_t, 3, torch::RestrictPtrTraits> links,
        torch::PackedTensorAccessor64<float, 2, torch::RestrictPtrTraits> data,
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

    float zeros[28];
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

// * For ray rendering
__device__ __inline__ void trace_ray_cuvol(
        const PackedSparseGridSpec& __restrict__ grid,
        SingleRaySpec ray,
        RenderOptions& __restrict__ opt,
        uint32_t ray_id,
        float* __restrict__ out) {

    // Warning: modifies ray.origin
    transform_coord(ray.origin, grid._scaling, grid._offset);
    // Warning: modifies ray.dir
    const float world_step = _get_delta_scale(grid._scaling, ray.dir) * opt.step_size;

    float t, tmax;
    float invdir[3];

#pragma unroll 3
    for (int i = 0; i < 3; ++i) {
        invdir[i] = 1.0 / ray.dir[i];
        if (ray.dir[i] == 0.f)
            invdir[i] = 1e9f;
    }

    {
        float t1, t2;
        t = 0.0f;
        tmax = 1e9f;
#pragma unroll 3
        for (int i = 0; i < 3; ++i) {
            t1 = (- ray.origin[i]) * invdir[i];
            t2 = (grid.links.size(i) - 1.f  - ray.origin[i]) * invdir[i];
            t = max(t, min(t1, t2));
            tmax = min(tmax, max(t1, t2));
        }
    }
    if (t > tmax) {
        // Ray doesn't hit box
        out[0] = out[1] = out[2] = opt.background_brightness;
        return;
    }
    out[0] = out[1] = out[2] = 0.f;

    float sphfunc_val[9];
    calc_sh(grid.basis_dim, ray.vdir, sphfunc_val);

    float light_intensity = 0.f;
    float pos[3], interp_val[28];
    int32_t l[3];
    while (t <= tmax) {
#pragma unroll 3
        for (int j = 0; j < 3; ++j) {
            pos[j] = ray.origin[j] + t * ray.dir[j];
            pos[j] = min(max(pos[j], 0.f), grid.links.size(j) - 1.f);
            l[j] = (int32_t) pos[j];
            l[j] = min(l[j], grid.links.size(j) - 2);
            pos[j] -= l[j];
        }

        trilerp_cuvol(grid.links, grid.data, l, pos, interp_val, 0, 1);

        float sigma = interp_val[0];
        if (sigma > opt.sigma_thresh) {
            trilerp_cuvol(grid.links, grid.data, l, pos, interp_val, 1, grid.data.size(1));
            const float pcnt = world_step * sigma;
            const float weight = __expf(light_intensity) * (1.f - __expf(-pcnt));
            light_intensity -= pcnt;

#pragma unroll 3
            for (int j = 0; j < 3; ++j) {
                int off = j * grid.basis_dim + 1;
                float tmp = 0.f;
                for (int i = 0; i < grid.basis_dim; ++i) {
                    tmp += sphfunc_val[i] * interp_val[off + i];
                }
                out[j] += weight * _SIGMOID(tmp);
            }
        }
        t += opt.step_size;
    }
    const float alpha = __expf(light_intensity) * opt.background_brightness;
#pragma unroll 3
    for (int j = 0; j < 3; ++j) {
        out[j] += alpha;
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
        uint32_t ray_id,
        float* __restrict__ grad_data_out
        ) {

    // Warning: modifies ray.origin
    transform_coord(ray.origin, grid._scaling, grid._offset);
    // Warning: modifies ray.dir
    const float world_step = _get_delta_scale(grid._scaling, ray.dir) * opt.step_size;

    float t, tmax;
    float invdir[3];

#pragma unroll 3
    for (int i = 0; i < 3; ++i) {
        invdir[i] = 1.0 / ray.dir[i];
        if (ray.dir[i] == 0.f)
            invdir[i] = 1e9f;
    }

    {
        float t1, t2;
        t = 0.0f;
        tmax = 1e9f;
#pragma unroll 3
        for (int i = 0; i < 3; ++i) {
            t1 = (- ray.origin[i]) * invdir[i];
            t2 = (grid.links.size(i) - 1.f - ray.origin[i]) * invdir[i];
            t = max(t, min(t1, t2));
            tmax = min(tmax, max(t1, t2));
        }
    }
    if (t > tmax) {
        // Ray doesn't hit box
        return;
    }

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

        trilerp_cuvol(grid.links, grid.data, l, pos, interp_val, 0, 1);
        float sigma = interp_val[0];
        if (sigma > opt.sigma_thresh) {
            trilerp_cuvol(grid.links, grid.data, l, pos, interp_val, 1,
                          grid.data.size(1));
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
            trilerp_backward_cuvol(grid.links, grad_data_out, l, pos, curr_grad,
                                   grid.data.size(1), grid.data.size(1));
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
    CUDA_GET_THREAD_ID(tid, rays.origins.size(0));
    trace_ray_cuvol(
        grid,
        SingleRaySpec(&rays.origins[tid][0], &rays.dirs[tid][0]),
        opt,
        tid,
        &out[tid][0]);
}

__global__ void render_ray_backward_kernel(
    PackedSparseGridSpec grid,
    const torch::PackedTensorAccessor32<float, 2, torch::RestrictPtrTraits>
        grad_output,
    torch::PackedTensorAccessor32<float, 2, torch::RestrictPtrTraits> color_cache,
        PackedRaysSpec rays,
        RenderOptions opt,
    float* __restrict__ grad_data_out
        ) {
    CUDA_GET_THREAD_ID(tid, rays.origins.size(0));
    trace_ray_cuvol_backward(
        grid,
        grad_output[tid],
        color_cache[tid],
        SingleRaySpec(&rays.origins[tid][0], &rays.dirs[tid][0]),
        opt,
        tid,
        grad_data_out);
}

__global__ void render_image_kernel(
        PackedSparseGridSpec grid,
        PackedCameraSpec cam,
        RenderOptions opt,
        torch::PackedTensorAccessor32<float, 3, torch::RestrictPtrTraits> out) {
    CUDA_GET_THREAD_ID(tid, cam.width * cam.height);
    int iy = tid / cam.width, ix = tid % cam.width;
    float dir[3], origin[3];
    cam2world_ray(ix, iy, dir, origin, cam);
    trace_ray_cuvol(
        grid,
        SingleRaySpec(origin, dir),
        opt,
        tid,
        &out[iy][ix][0]);
}

__global__ void render_image_backward_kernel(
        PackedSparseGridSpec grid,
        const torch::PackedTensorAccessor32<float, 3, torch::RestrictPtrTraits>
            grad_output,
        torch::PackedTensorAccessor32<float, 3, torch::RestrictPtrTraits> color_cache,
        PackedCameraSpec cam,
        RenderOptions opt,
        float* __restrict__ grad_data_out) {
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
        tid,
        grad_data_out);
}
}  // namespace device
}  // namespace

torch::Tensor volume_render_cuvol(SparseGridSpec& grid, RaysSpec& rays, RenderOptions& opt) {
    DEVICE_GUARD(grid.data);
    grid.check();
    rays.check();

    const auto Q = rays.origins.size(0);

    const int cuda_n_threads = 768;
    const int blocks = CUDA_N_BLOCKS_NEEDED(Q, cuda_n_threads);
    torch::Tensor results = torch::empty_like(rays.origins);
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
        torch::Tensor color_cache) {

    DEVICE_GUARD(grid.data);
    grid.check();
    rays.check();
    const auto Q = rays.origins.size(0);

    const int cuda_n_threads_render_backward = 448;
    const int blocks = CUDA_N_BLOCKS_NEEDED(Q, cuda_n_threads_render_backward);

    torch::Tensor result = torch::zeros_like(grid.data);
    device::render_ray_backward_kernel<<<blocks,
           cuda_n_threads_render_backward>>>(
            grid,
            grad_out.packed_accessor32<float, 2, torch::RestrictPtrTraits>(),
            color_cache.packed_accessor32<float, 2, torch::RestrictPtrTraits>(),
            rays, opt,
            // Output
            result.data<float>());
    CUDA_CHECK_ERRORS;
    return result;
}

torch::Tensor volume_render_cuvol_image(SparseGridSpec& grid, CameraSpec& cam, RenderOptions& opt) {
    DEVICE_GUARD(grid.data);
    grid.check();
    cam.check();

    const size_t Q = size_t(cam.width) * cam.height;

    const int cuda_n_threads = 768;
    const int blocks = CUDA_N_BLOCKS_NEEDED(Q, cuda_n_threads);
    torch::Tensor result = torch::zeros({cam.height, cam.width, 3}, grid.data.options());

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
        torch::Tensor color_cache) {

    DEVICE_GUARD(grid.data);
    grid.check();
    cam.check();
    const size_t Q = size_t(cam.width) * cam.height;

    const int cuda_n_threads_render_backward = 448;
    const int blocks = CUDA_N_BLOCKS_NEEDED(Q, cuda_n_threads_render_backward);

    torch::Tensor result = torch::zeros_like(grid.data);
    device::render_image_backward_kernel<<<blocks,
           cuda_n_threads_render_backward>>>(
            grid,
            grad_out.packed_accessor32<float, 3, torch::RestrictPtrTraits>(),
            color_cache.packed_accessor32<float, 3, torch::RestrictPtrTraits>(),
            cam,
            opt,
            // Output
            result.data<float>());
    CUDA_CHECK_ERRORS;
    return result;
}
