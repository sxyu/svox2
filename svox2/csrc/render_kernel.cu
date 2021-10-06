// Copyright 2021 Alex Yu
#include <torch/extension.h>
#include <cstdint>
#include "cuda_util.cuh"
#include "render_util.cuh"
#include "data_spec_packed.cuh"

namespace {
namespace device {

__device__ __inline__ void trace_ray(
        const PackedSparseGridSpec& __restrict__ grid,
        SingleRaySpec ray,
        RenderOptions& __restrict__ opt,
        float* __restrict__ out) {
    // Warning: modifies ray.origin
    transform_coord(ray.origin, grid._scaling, grid._offset);
    // Warning: modifies ray.dir
    const float delta_scale = _get_delta_scale(grid._scaling, ray.dir);

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
            t1 = (1e-3f - ray.origin[i]) * invdir[i];
            t2 = (grid.links.size(i) - 1.f - 1e-3f - ray.origin[i]) * invdir[i];
            t = max(t, min(t1, t2));
            tmax = min(tmax, max(t1, t2));
        }
    }

    if (tmax < 0 || t > tmax) {
        // Ray doesn't hit box
#pragma unroll 3
        for (int j = 0; j < 3; ++j) {
            out[j] = opt.background_brightness;
        }
        return;
    } else {
#pragma unroll 3
        for (int j = 0; j < 3; ++j) {
            out[j] = 0.f;
        }
        float pos[3];
        int32_t l[3];
        float basis_fn[16];
        // vdir is unscaled unit dir in world space, for calculating spherical function
        calc_sh(grid.basis_dim, ray.vdir, basis_fn);

        float log_light_intensity = 0.f;
        while (t < tmax) {
#pragma unroll 3
            for (int j = 0; j < 3; ++j) {
                pos[j] = ray.origin[j] + t * ray.dir[j];
                l[j] = (int32_t) pos[j];
                pos[j] -= l[j];
            }

            const float delta_t = _intersect_aabb_unit(pos, invdir) + opt.step_epsilon;
            const int32_t link = grid.links[l[0]][l[1]][l[2]];
            t += delta_t;
            if (link >= 0) {
                const float* __restrict__ sample_val = &grid.data[link][0];

                float sigma = sample_val[0];
                if (sigma > opt.sigma_thresh) {
                    const float log_transmit = -delta_t * delta_scale * sigma;
                    const float light_intensity = expf(log_light_intensity);
                    const float weight = light_intensity * (1.f - expf(log_transmit));
#pragma unroll 3
                    for (int k = 0; k < 3; ++k) {
                        const int off = k * grid.basis_dim + 1;
                        float tmp = 0.0;
                        for (int i = 0; i < grid.basis_dim; ++i) {
                            tmp += basis_fn[i] * sample_val[off + i];
                        }
                        out[k] += weight * _SIGMOID(tmp);
                    }
                    log_light_intensity += log_transmit;

                    if (light_intensity <= opt.stop_thresh) {
                        // Full opacity, stop
                        float scale = 1.0 / (1.0 - light_intensity);
                        for (int j = 0; j < 3; ++j) {
                            out[j] *= scale;
                        }
                        return;
                    }
                }
            }
        }
#pragma unroll 3
        for (int j = 0; j < 3; ++j) {
            out[j] += expf(log_light_intensity) * opt.background_brightness;
        }
    }
}

__device__ __inline__ void trace_ray_backward(
    const PackedSparseGridSpec& __restrict__ grid,
    const torch::TensorAccessor<float, 1, torch::RestrictPtrTraits, int32_t>
        grad_output,
    const torch::TensorAccessor<float, 1, torch::RestrictPtrTraits, int32_t>
        color_cache,
        SingleRaySpec ray,
        RenderOptions& __restrict__ opt,
    float* __restrict__ grad_data_out) {
    // Warning: modifies ray.origin
    transform_coord(ray.origin, grid._scaling, grid._offset);
    // Warning: modifies ray.dir
    const float delta_scale = _get_delta_scale(grid._scaling, ray.dir);

    float t, tmax;
    float invdir[3];

#pragma unroll
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
            t1 = (1e-3f - ray.origin[i]) * invdir[i];
            t2 = (grid.links.size(i) - 1.f - 1e-3f - ray.origin[i]) * invdir[i];
            t = max(t, min(t1, t2));
            tmax = min(tmax, max(t1, t2));
        }
    }

    if (tmax < 0 || t > tmax) {
        // Ray doesn't hit box
        return;
    } else {
        float pos[3];
        int32_t l[3];
        float basis_fn[16];
        calc_sh(grid.basis_dim, ray.vdir, basis_fn);

        float accum = color_cache[0] * grad_output[0] +
            color_cache[1] * grad_output[1] +
            color_cache[2] * grad_output[2];
        float log_light_intensity = 0.f;
        while (t < tmax) {
#pragma unroll 3
            for (int j = 0; j < 3; ++j) {
                pos[j] = ray.origin[j] + t * ray.dir[j];
                l[j] = (int32_t) pos[j];
                pos[j] -= l[j];
            }
            const int32_t link = grid.links[l[0]][l[1]][l[2]];
            float delta_t = _intersect_aabb_unit(pos, invdir) + opt.step_epsilon;
            t += delta_t;
            if (link >= 0) {
                const float* __restrict__ sample_val = &grid.data[link][0];

                float sigma = sample_val[0];
                if (sigma > 0.0) {
                    // Reuse offset on gradient
                    float* __restrict__ grad_tree_val = grad_data_out + (
                            sample_val - grid.data.data());

                    delta_t *= delta_scale;
                    const float log_transmit = -delta_t * sigma;
                    const float weight = expf(log_light_intensity) * (1.f - expf(log_transmit));

                    float total_color = 0.f;
#pragma unroll 3
                    for (int k = 0; k < 3; ++ k) {
                        const int off = k * grid.basis_dim + 1;
                        float tmp = 0.0;
                        for (int i = 0; i < grid.basis_dim; ++i) {
                            tmp += basis_fn[i] * sample_val[off + i];
                        }
                        const float sigmoid = _SIGMOID(tmp);

                        tmp = weight * sigmoid * (1.f - sigmoid)  * grad_output[k];
                        for (int i = 0; i < grid.basis_dim; ++i) {
                            atomicAdd(&grad_tree_val[off + i], basis_fn[i] * tmp);
                        }
                        total_color += sigmoid * grad_output[k];
                    }
                    log_light_intensity += log_transmit;
                    accum -= weight * total_color;
                    atomicAdd(&grad_tree_val[0],
                            delta_t * (total_color *
                                expf(log_light_intensity) - accum));
                }
            }
        }
    }
}  // trace_ray_backward


// ** Kernels

__global__ void render_ray_nn_kernel(
        PackedSparseGridSpec grid,
        PackedRaysSpec rays,
        RenderOptions opt,
    torch::PackedTensorAccessor32<float, 2, torch::RestrictPtrTraits>
        out) {
    CUDA_GET_THREAD_ID(tid, rays.origins.size(0));
    trace_ray(
        grid,
        SingleRaySpec(&rays.origins[tid][0], &rays.dirs[tid][0]),
        opt,
        &out[tid][0]);
}


__global__ void render_ray_nn_backward_kernel(
    PackedSparseGridSpec grid,
    const torch::PackedTensorAccessor32<float, 2, torch::RestrictPtrTraits>
        grad_output,
    torch::PackedTensorAccessor32<float, 2, torch::RestrictPtrTraits> color_cache,
        PackedRaysSpec rays,
        RenderOptions opt,
    float* __restrict__ grad_data_out
        ) {
    CUDA_GET_THREAD_ID(tid, rays.origins.size(0));
    trace_ray_backward(
        grid,
        grad_output[tid],
        color_cache[tid],
        SingleRaySpec(&rays.origins[tid][0], &rays.dirs[tid][0]),
        opt,
        grad_data_out);
}

}  // namespace device
}  // namespace

torch::Tensor volume_render_nn(SparseGridSpec& grid, RaysSpec& rays, RenderOptions& opt) {
    DEVICE_GUARD(grid.data);
    grid.check();
    rays.check();
    const auto Q = rays.origins.size(0);

    const int cuda_n_threads = 512;
    const int blocks = CUDA_N_BLOCKS_NEEDED(Q, cuda_n_threads);
    torch::Tensor result = torch::zeros({Q, 3}, rays.origins.options());
    AT_DISPATCH_FLOATING_TYPES(rays.origins.type(), __FUNCTION__, [&] {
            device::render_ray_nn_kernel<<<blocks, cuda_n_threads>>>(
                    grid, rays, opt,
                    result.packed_accessor32<float, 2, torch::RestrictPtrTraits>());
    });
    CUDA_CHECK_ERRORS;
    return result;
}

torch::Tensor volume_render_nn_backward(
        SparseGridSpec& grid,
        RaysSpec& rays,
        RenderOptions& opt,
        torch::Tensor grad_out,
        torch::Tensor color_cache) {
    DEVICE_GUARD(grid.data);
    grid.check();
    rays.check();
    CHECK_INPUT(grad_out);
    CHECK_INPUT(color_cache);

    const int Q = rays.origins.size(0);

    const int cuda_n_threads = 512;
    const int blocks = CUDA_N_BLOCKS_NEEDED(Q, cuda_n_threads);
    torch::Tensor result = torch::zeros_like(grid.data);
    AT_DISPATCH_FLOATING_TYPES(rays.origins.type(), __FUNCTION__, [&] {
            device::render_ray_nn_backward_kernel<<<blocks, cuda_n_threads>>>(
                grid,
                grad_out.packed_accessor32<float, 2, torch::RestrictPtrTraits>(),
                color_cache.packed_accessor32<float, 2, torch::RestrictPtrTraits>(),
                rays,
                opt,
                result.data<float>());
    });
    CUDA_CHECK_ERRORS;
    return result;
}
