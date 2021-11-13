// Copyright 2021 Alex Yu
// nearest-neighbor baseline
// Background is NOT supported
#include <torch/extension.h>
#include <cstdint>
#include "cuda_util.cuh"
#include "render_util.cuh"
#include "data_spec_packed.cuh"

namespace {
namespace device {
// From old version (name is hacky whatever)
struct BasicSingleRaySpec {
    __device__ BasicSingleRaySpec(const float* __restrict__ origin, const float* __restrict__ dir)
        : origin{origin[0], origin[1], origin[2]},
        dir{dir[0], dir[1], dir[2]},
        vdir(dir) {}
    float origin[3];
    float dir[3];
    const float* __restrict__ vdir;
};

__device__ __inline__ float compute_skip_dist_nn(
        const BasicSingleRaySpec& __restrict__ ray,
        const float* __restrict__ invdir,
        const float* __restrict__ pos,
        const int32_t* __restrict__ l,
        int32_t link_val) {
    const uint32_t dist = -link_val;
    const uint32_t cell_ul_shift = (dist - 1);
    const uint32_t cell_side_len = (1 << cell_ul_shift);

    // AABB intersection
    // Consider caching the invdir for the ray
    float tmax = 1e9f;
#pragma unroll
    for (int i = 0; i < 3; ++i) {
        int ul = (((l[i]) >> cell_ul_shift) << cell_ul_shift);
        ul -= l[i];

        const float t1 = (ul - pos[i]) * invdir[i];
        const float t2 = (ul + cell_side_len - pos[i]) * invdir[i];
        if (ray.dir[i] != 0.f) {
            tmax = fminf(tmax, fmaxf(t1, t2));
        }
    }
    return tmax;
}

__device__ __inline__ void trace_ray(
        const PackedSparseGridSpec& __restrict__ grid,
        BasicSingleRaySpec ray,
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
            ray.origin[i] += 0.5f;  // Fix offset of nn vs lerp
            t1 = (0.0f - ray.origin[i]) * invdir[i];
            t2 = (grid.size[i] - 1.f - ray.origin[i]) * invdir[i];
            t = fmaxf(t, fminf(t1, t2));
            tmax = fminf(tmax, fmaxf(t1, t2));
        }
    }

    if (t > tmax) {
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
        float basis_fn[9];
        // vdir is unscaled unit dir in world space, for calculating spherical function
        calc_sh(grid.basis_dim, ray.vdir, basis_fn);

        float log_transmittance = 0.f;
        while (t < tmax) {
#pragma unroll 3
            for (int j = 0; j < 3; ++j) {
                pos[j] = ray.origin[j] + t * ray.dir[j];
                pos[j] = fminf(fmaxf(pos[j], 0.f), grid.size[j] - 1.f);
                l[j] = min(static_cast<int32_t>(pos[j]), grid.size[j] - 1);
                pos[j] -= l[j];
            }

            const int32_t link = grid.links[
                (l[0] * grid.size[1] +  l[1]) * grid.size[2] + l[2]
            ];
            if (link >= 0) {
                const float delta_t = _intersect_aabb_unit(pos, invdir) + 1e-2f;
                t += delta_t;
                float sigma = grid.density_data[link];
                if (opt.last_sample_opaque && t + opt.step_size > tmax) {
                    sigma += 1e9;
                }
                if (sigma > opt.sigma_thresh) {
                    const float* __restrict__ sample_val = &grid.sh_data[size_t(link) * grid.sh_data_dim];
                    const float log_transmit = -delta_t * delta_scale * sigma;
                    const float transmittance = expf(log_transmittance);
                    const float weight = transmittance * (1.f - expf(log_transmit));
#pragma unroll 3
                    for (int k = 0; k < 3; ++k) {
                        const int off = k * grid.basis_dim;
                        float tmp = 0.5f;
                        for (int i = 0; i < grid.basis_dim; ++i) {
                            tmp += basis_fn[i] * sample_val[off + i];
                        }
                        out[k] += weight * fmaxf(tmp, 0.f);
                    }
                    log_transmittance += log_transmit;

                    if (transmittance <= opt.stop_thresh) {
                        // Full opacity, stop
                        float scale = 1.0 / (1.0 - transmittance);
                        for (int j = 0; j < 3; ++j) {
                            out[j] *= scale;
                        }
                        return;
                    }
                }
            } else {
                float skip = fmaxf(compute_skip_dist_nn(ray,
                       invdir,
                       pos,
                       l, link), 0.f);
                t += skip + 1e-2f;
            }
        }
#pragma unroll 3
        for (int j = 0; j < 3; ++j) {
            out[j] += expf(log_transmittance) * opt.background_brightness;
        }
    }
}

__device__ __inline__ void trace_ray_backward(
    const PackedSparseGridSpec& __restrict__ grid,
    const float* __restrict__ grad_output,
    const float* __restrict__ color_cache,
        BasicSingleRaySpec ray,
        RenderOptions& __restrict__ opt,
    PackedGridOutputGrads& __restrict__ grads) {
    // Warning: modifies ray.origin
    transform_coord(ray.origin, grid._scaling, grid._offset);
    // Warning: modifies ray.dir
    const float delta_scale = _get_delta_scale(grid._scaling, ray.dir);

    float t, tmax;
    float invdir[3];

#pragma unroll
    for (int i = 0; i < 3; ++i) {
        invdir[i] = 1.0 / ray.dir[i];
        if (ray.dir[i] == 0.0f) {
            invdir[i] = 1e9f;
        }
    }
    {
        float t1, t2;
        t = 0.0f;
        tmax = 1e9f;
#pragma unroll 3
        for (int i = 0; i < 3; ++i) {
            ray.origin[i] += 0.5f;  // Fix offset of nn vs lerp
            t1 = (0.0f - ray.origin[i]) * invdir[i];
            t2 = (grid.size[i] - 1.f - ray.origin[i]) * invdir[i];
            if (ray.dir[i] != 0.0f) {
                t = fmaxf(t, fminf(t1, t2));
                tmax = fminf(tmax, fmaxf(t1, t2));
            }
        }
    }

    if (t > tmax) {
        // Ray doesn't hit box
        return;
    } else {
        float pos[3];
        int32_t l[3];
        float basis_fn[9];
        calc_sh(grid.basis_dim, ray.vdir, basis_fn);

        float accum = color_cache[0] * grad_output[0] +
            color_cache[1] * grad_output[1] +
            color_cache[2] * grad_output[2];
        float log_transmittance = 0.f;
        while (t < tmax) {
#pragma unroll 3
            for (int j = 0; j < 3; ++j) {
                pos[j] = ray.origin[j] + t * ray.dir[j];
                pos[j] = fminf(fmaxf(pos[j], 0.f), grid.size[j] - 1.f);
                l[j] = min(static_cast<int32_t>(pos[j]), grid.size[j] - 1);
                pos[j] -= l[j];
            }
            const int32_t link = grid.links[
                (l[0] * grid.size[1] +  l[1]) * grid.size[2] + l[2]
            ];
            if (link >= 0) {
                float delta_t = _intersect_aabb_unit(pos, invdir) + 1e-2f;
                t += delta_t;
                float sigma = grid.density_data[link];
                if (opt.last_sample_opaque && t + opt.step_size > tmax) {
                    sigma += 1e9;
                }
                if (sigma > opt.sigma_thresh) {
                    const float* __restrict__ sample_val = &grid.sh_data[size_t(link) * grid.sh_data_dim];
                    float* __restrict__ grad_sample_val = &grads.grad_sh_out[size_t(link) * grid.sh_data_dim];
                    delta_t *= delta_scale;
                    const float log_transmit = -delta_t * sigma;
                    const float weight = expf(log_transmittance) * (1.f - expf(log_transmit));

                    float total_color = 0.f;
#pragma unroll 3
                    for (int k = 0; k < 3; ++ k) {
                        const int off = k * grid.basis_dim;
                        float tmp = 0.5f;
                        for (int i = 0; i < grid.basis_dim; ++i) {
                            tmp += basis_fn[i] * sample_val[off + i];
                        }

                        if (tmp > 0.f) {
                            total_color += tmp * grad_output[k];
                            tmp = weight * grad_output[k];
                            for (int i = 0; i < grid.basis_dim; ++i) {
                                atomicAdd(&grad_sample_val[off + i],
                                        basis_fn[i] * tmp);
                            }
                        }
                    }
                    log_transmittance += log_transmit;
                    accum -= weight * total_color;
                    if (grads.mask_out != nullptr) {
                        grads.mask_out[link] = true;
                    }
                    atomicAdd(&grads.grad_density_out[link],
                            delta_t * (total_color *
                                expf(log_transmittance) - accum));
                    if (expf(log_transmittance) <= opt.stop_thresh) {
                        return;
                    }
                }
            } else {
                t += fmaxf(compute_skip_dist_nn(ray,
                       invdir,
                       pos,
                       l, link), 0.f) + 1e-2f;
            }
        }
    }
}  // trace_ray_backward


// ** Kernels

__global__ void render_ray_svox1_kernel(
        PackedSparseGridSpec grid,
        PackedRaysSpec rays,
        RenderOptions opt,
    torch::PackedTensorAccessor32<float, 2, torch::RestrictPtrTraits>
        out) {
    CUDA_GET_THREAD_ID(tid, rays.origins.size(0));
    trace_ray(
        grid,
        BasicSingleRaySpec(&rays.origins[tid][0], &rays.dirs[tid][0]),
        opt,
        &out[tid][0]);
}


__global__ void render_ray_svox1_backward_kernel(
    PackedSparseGridSpec grid,
    const float* __restrict__ grad_output,
    const float* __restrict__ color_cache,
        PackedRaysSpec rays,
        RenderOptions opt,
    bool grad_out_is_rgb,
    PackedGridOutputGrads grads
        ) {
    CUDA_GET_THREAD_ID(tid, rays.origins.size(0));

    float grad_out[3];
    if (grad_out_is_rgb) {
        const float norm_factor = 2.f / (3 * int(rays.origins.size(0)));
#pragma unroll 3
        for (int i = 0; i < 3; ++i) {
            const float resid = color_cache[tid * 3 + i] - grad_output[tid * 3 + i];
            grad_out[i] = resid * norm_factor;
        }
    } else {
#pragma unroll 3
        for (int i = 0; i < 3; ++i) {
            grad_out[i] = grad_output[tid * 3 + i];
        }
    }

    trace_ray_backward(
        grid,
        grad_out,
        color_cache + tid * 3,
        BasicSingleRaySpec(&rays.origins[tid][0], &rays.dirs[tid][0]),
        opt,
        grads);
}

}  // namespace device
}  // namespace

torch::Tensor volume_render_svox1(SparseGridSpec& grid, RaysSpec& rays, RenderOptions& opt) {
    DEVICE_GUARD(grid.sh_data);
    TORCH_CHECK(grid.basis_type == BASIS_TYPE_SH); // Only supporting SH for now
    grid.check();
    rays.check();
    const auto Q = rays.origins.size(0);

    const int cuda_n_threads = 512;
    const int blocks = CUDA_N_BLOCKS_NEEDED(Q, cuda_n_threads);
    torch::Tensor result = torch::zeros({Q, 3}, rays.origins.options());
    device::render_ray_svox1_kernel<<<blocks, cuda_n_threads>>>(
            grid, rays, opt,
            result.packed_accessor32<float, 2, torch::RestrictPtrTraits>());
    CUDA_CHECK_ERRORS;
    return result;
}

void volume_render_svox1_backward(
        SparseGridSpec& grid,
        RaysSpec& rays,
        RenderOptions& opt,
        torch::Tensor grad_out,
        torch::Tensor color_cache,
        GridOutputGrads& grads) {
    DEVICE_GUARD(grid.sh_data);
    grid.check();
    rays.check();
    grads.check();
    CHECK_INPUT(grad_out);
    CHECK_INPUT(color_cache);

    const int Q = rays.origins.size(0);

    const int cuda_n_threads = 512;
    const int blocks = CUDA_N_BLOCKS_NEEDED(Q, cuda_n_threads);
    device::render_ray_svox1_backward_kernel<<<blocks, cuda_n_threads>>>(
        grid,
        grad_out.data_ptr<float>(),
        color_cache.data_ptr<float>(),
        rays,
        opt,
        false,
        grads);
    CUDA_CHECK_ERRORS;
}

void volume_render_svox1_fused(
        SparseGridSpec& grid,
        RaysSpec& rays,
        RenderOptions& opt,
        torch::Tensor rgb_gt,
        float _,  // not supported
        float _2, // not supported
        torch::Tensor rgb_out,
        GridOutputGrads& grads) {

    DEVICE_GUARD(grid.sh_data);
    CHECK_INPUT(rgb_gt);
    CHECK_INPUT(rgb_out);
    grid.check();
    rays.check();
    grads.check();
    const auto Q = rays.origins.size(0);

    const int cuda_n_threads = 512;
    {
        const int blocks = CUDA_N_BLOCKS_NEEDED(Q, cuda_n_threads);
        device::render_ray_svox1_kernel<<<blocks, cuda_n_threads>>>(
                grid, rays, opt,
                // Output
                rgb_out.packed_accessor32<float, 2, torch::RestrictPtrTraits>());
        CUDA_CHECK_ERRORS;
    }
    {
        const int blocks = CUDA_N_BLOCKS_NEEDED(Q, cuda_n_threads);
        device::render_ray_svox1_backward_kernel<<<blocks, cuda_n_threads>>>(
                grid,
                rgb_gt.data_ptr<float>(),
                rgb_out.data_ptr<float>(),
                rays, opt,
                true,
                // Output
                grads);
        CUDA_CHECK_ERRORS;
    }
}
