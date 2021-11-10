// Copyright 2021 Alex Yu
#include <torch/extension.h>
#include "cuda_util.cuh"
#include "data_spec_packed.cuh"
#include "render_util.cuh"
#include "cubemap_util.cuh"

#include <iostream>
#include <cstdint>
#include <tuple>

namespace {
const int WARP_SIZE = 32;

const int TRACE_RAY_CUDA_THREADS = 128;
const int TRACE_RAY_CUDA_RAYS_PER_BLOCK = TRACE_RAY_CUDA_THREADS / WARP_SIZE;

const int TRACE_RAY_BKWD_CUDA_THREADS = 128;
const int TRACE_RAY_BKWD_CUDA_RAYS_PER_BLOCK = TRACE_RAY_BKWD_CUDA_THREADS / WARP_SIZE;

const int MIN_BLOCKS_PER_SM = 8;

const int TRACE_RAY_BG_CUDA_THREADS = 256;
const int MIN_BG_BLOCKS_PER_SM = 4;
typedef cub::WarpReduce<float> WarpReducef;

namespace device {


// * For ray rendering
__device__ __inline__ void trace_ray_cuvol(
        const PackedSparseGridSpec& __restrict__ grid,
        SingleRaySpec& __restrict__ ray,
        const RenderOptions& __restrict__ opt,
        uint32_t lane_id,
        float* __restrict__ sphfunc_val,
        WarpReducef::TempStorage& __restrict__ temp_storage,
        float* __restrict__ out,
        float* __restrict__ out_log_transmit) {
    const uint32_t lane_colorgrp_id = lane_id % grid.basis_dim;
    const uint32_t lane_colorgrp = lane_id / grid.basis_dim;

    if (ray.tmin > ray.tmax) {
        out[lane_colorgrp] = (grid.background_nlayers == 0) ? opt.background_brightness : 0.f;
        if (out_log_transmit != nullptr) {
            *out_log_transmit = 0.f;
        }
        return;
    }

    float t = ray.tmin;
    float outv = 0.f;

    float log_transmit = 0.f;
    // printf("tmin %f, tmax %f \n", ray.tmin, ray.tmax);

    while (t <= ray.tmax) {
#pragma unroll 3
        for (int j = 0; j < 3; ++j) {
            ray.pos[j] = fmaf(t, ray.dir[j], ray.origin[j]);
            ray.pos[j] = min(max(ray.pos[j], 0.f), grid.size[j] - 1.f);
            ray.l[j] = min(static_cast<int32_t>(ray.pos[j]), grid.size[j] - 2);
            ray.pos[j] -= static_cast<float>(ray.l[j]);
        }
        // if (lane_id == 0) {
        //     printf("[%d %d %d] [%f %f %f] %f\n",
        //             ray.l[0], ray.l[1], ray.l[2],
        //             ray.pos[0], ray.pos[1], ray.pos[2], log_transmit);
        // }

        const float skip = compute_skip_dist(ray,
                       grid.links, grid.stride_x,
                       grid.size[2], 0);

        if (skip >= opt.step_size) {
            // For consistency, we skip the by step size
            t += ceilf(skip / opt.step_size) * opt.step_size;
            continue;
        }
        float sigma = trilerp_cuvol_one(
                grid.links, grid.density_data,
                grid.stride_x,
                grid.size[2],
                1,
                ray.l, ray.pos,
                0);
        if (opt.last_sample_opaque && t + opt.step_size > ray.tmax) {
            sigma += 1e9;
        }
        if (opt.randomize && opt.random_sigma_std > 0.0) sigma += ray.rng.randn() * opt.random_sigma_std;
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
            const float weight = _EXP(log_transmit) * (1.f - _EXP(-pcnt));
            log_transmit -= pcnt;

            float lane_color_total = WarpReducef(temp_storage).HeadSegmentedSum(
                                           lane_color, lane_colorgrp_id == 0);
            outv += weight * fmaxf(lane_color_total + 0.5f, 0.f);  // Clamp to [+0, infty)
            if (_EXP(log_transmit) < opt.stop_thresh) {
                log_transmit = -1e3f;
                break;
            }
        }
        t += opt.step_size;
    }

    if (grid.background_nlayers == 0) {
        outv += _EXP(log_transmit) * opt.background_brightness;
    }
    if (lane_colorgrp_id == 0) {
        if (out_log_transmit != nullptr) {
            *out_log_transmit = log_transmit;
        }
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
        const float* __restrict__ sphfunc_val,
        float* __restrict__ grad_sphfunc_val,
        WarpReducef::TempStorage& __restrict__ temp_storage,
        float log_transmit_in,
        float beta_loss,
        float sparsity_loss,
        PackedGridOutputGrads& __restrict__ grads,
        float* __restrict__ accum_out,
        float* __restrict__ log_transmit_out
        ) {
    const uint32_t lane_colorgrp_id = lane_id % grid.basis_dim;
    const uint32_t lane_colorgrp = lane_id / grid.basis_dim;
    const uint32_t leader_mask = 1U | (1U << grid.basis_dim) | (1U << (2 * grid.basis_dim));

    float accum = fmaf(color_cache[0], grad_output[0],
                      fmaf(color_cache[1], grad_output[1],
                           color_cache[2] * grad_output[2]));

    if (beta_loss > 0.f) {
        const float transmit_in = _EXP(log_transmit_in);
        beta_loss *= (1 - transmit_in / (1 - transmit_in + 1e-3)); // d beta_loss / d log_transmit_in
        accum += beta_loss;
        // Interesting how this loss turns out, kinda nice?
    }

    if (ray.tmin > ray.tmax) {
        if (accum_out != nullptr) { *accum_out = accum; }
        if (log_transmit_out != nullptr) { *log_transmit_out = 0.f; }
        // printf("accum_end_fg_fast=%f\n", accum);
        return;
    }
    float t = ray.tmin;

    const float gout = grad_output[lane_colorgrp];

    float log_transmit = 0.f;

    // remat samples
    while (t <= ray.tmax) {
#pragma unroll 3
        for (int j = 0; j < 3; ++j) {
            ray.pos[j] = fmaf(t, ray.dir[j], ray.origin[j]);
            ray.pos[j] = min(max(ray.pos[j], 0.f), grid.size[j] - 1.f);
            ray.l[j] = min(static_cast<int32_t>(ray.pos[j]), grid.size[j] - 2);
            ray.pos[j] -= static_cast<float>(ray.l[j]);
        }
        const float skip = compute_skip_dist(ray,
                       grid.links, grid.stride_x,
                       grid.size[2], 0);
        if (skip >= opt.step_size) {
            // For consistency, we skip the by step size
            t += ceilf(skip / opt.step_size) * opt.step_size;
            continue;
        }

        float sigma = trilerp_cuvol_one(
                grid.links,
                grid.density_data,
                grid.stride_x,
                grid.size[2],
                1,
                ray.l, ray.pos,
                0);
        if (opt.last_sample_opaque && t + opt.step_size > ray.tmax) {
            sigma += 1e9;
        }
        if (opt.randomize && opt.random_sigma_std > 0.0) sigma += ray.rng.randn() * opt.random_sigma_std;
        if (sigma > opt.sigma_thresh) {
            float lane_color = trilerp_cuvol_one(
                            grid.links,
                            grid.sh_data,
                            grid.stride_x,
                            grid.size[2],
                            grid.sh_data_dim,
                            ray.l, ray.pos, lane_id);
            float weighted_lane_color = lane_color * sphfunc_val[lane_colorgrp_id];

            const float pcnt = ray.world_step * sigma;
            const float weight = _EXP(log_transmit) * (1.f - _EXP(-pcnt));
            log_transmit -= pcnt;

            const float lane_color_total = WarpReducef(temp_storage).HeadSegmentedSum(
                                           weighted_lane_color, lane_colorgrp_id == 0) + 0.5f;
            float total_color = fmaxf(lane_color_total, 0.f);
            float color_in_01 = total_color == lane_color_total;
            total_color *= gout; // Clamp to [+0, infty)

            float total_color_c1 = __shfl_sync(leader_mask, total_color, grid.basis_dim);
            total_color += __shfl_sync(leader_mask, total_color, 2 * grid.basis_dim);
            total_color += total_color_c1;

            color_in_01 = __shfl_sync((1U << grid.sh_data_dim) - 1, color_in_01, lane_colorgrp * grid.basis_dim);
            const float grad_common = weight * color_in_01 * gout;
            const float curr_grad_color = sphfunc_val[lane_colorgrp_id] * grad_common;

            if (grid.basis_type != BASIS_TYPE_SH) {
                float curr_grad_sphfunc = lane_color * grad_common;
                const float curr_grad_up2 = __shfl_down_sync((1U << grid.sh_data_dim) - 1,
                        curr_grad_sphfunc, 2 * grid.basis_dim);
                curr_grad_sphfunc += __shfl_down_sync((1U << grid.sh_data_dim) - 1,
                        curr_grad_sphfunc, grid.basis_dim);
                curr_grad_sphfunc += curr_grad_up2;
                if (lane_id < grid.basis_dim) {
                    grad_sphfunc_val[lane_id] += curr_grad_sphfunc;
                }
            }

            accum -= weight * total_color;
            float curr_grad_sigma = ray.world_step * (
                    total_color * _EXP(log_transmit) - accum);
            if (sparsity_loss > 0.f) {
                // Cauchy version (from SNeRG)
                curr_grad_sigma += sparsity_loss * (4 * sigma / (1 + 2 * (sigma * sigma)));

                // Alphs version (from PlenOctrees)
                // curr_grad_sigma += sparsity_loss * _EXP(-pcnt) * ray.world_step;
            }
            trilerp_backward_cuvol_one(grid.links, grads.grad_sh_out,
                    grid.stride_x,
                    grid.size[2],
                    grid.sh_data_dim,
                    ray.l, ray.pos,
                    curr_grad_color, lane_id);
            if (lane_id == 0) {
                trilerp_backward_cuvol_one_density(
                        grid.links,
                        grads.grad_density_out,
                        grads.mask_out,
                        grid.stride_x,
                        grid.size[2],
                        ray.l, ray.pos, curr_grad_sigma);
            }
            if (_EXP(log_transmit) < opt.stop_thresh) {
                break;
            }
        }
        t += opt.step_size;
    }
    if (lane_id == 0) {
        if (accum_out != nullptr) {
            // Cancel beta loss out in case of background
            accum -= beta_loss;
            *accum_out = accum;
        }
        if (log_transmit_out != nullptr) { *log_transmit_out = log_transmit; }
        // printf("accum_end_fg=%f\n", accum);
        // printf("log_transmit_fg=%f\n", log_transmit);
    }
}


__device__ __inline__ void render_background_forward(
            const PackedSparseGridSpec& __restrict__ grid,
            SingleRaySpec& __restrict__ ray,
            const RenderOptions& __restrict__ opt,
            float log_transmit,
            float* __restrict__ out
        ) {

    ConcentricSpheresIntersector csi(ray.origin, ray.dir);

    const float inner_radius = fmaxf(_dist_ray_to_origin(ray.origin, ray.dir) + 1e-3f, 1.f);
    float t, t_last;
    const int n_steps = int(grid.background_nlayers / opt.step_size) + 2;

    csi.intersect(inner_radius, &t_last);

    // printf("RAY o[%f,%f,%f] d[%f,%f,%f] rad=%f, t_last=%f ws=%f\n",
    //         ray.origin[0], ray.origin[1], ray.origin[2],
    //         ray.dir[0], ray.dir[1], ray.dir[2],
    //         inner_radius,
    //         t_last, ray.world_step);

    float outv[3] = {0.f, 0.f, 0.f};
    const int cubemap_step = 6 * grid.background_reso * grid.background_reso * /*n_channels*/ 4;
    const float layer_scale = (float(grid.background_nlayers - 1) / (n_steps + 1));
    for (int i = 0; i < n_steps; ++i) {
        float r = n_steps / (n_steps - i - 0.5);
        if (r < inner_radius || !csi.intersect(r, &t)) continue;
        const float t_mid = (t + t_last) * 0.5f;

#pragma unroll 3
        for (int j = 0; j < 3; ++j) {
            ray.pos[j] = fmaf(t_mid, ray.dir[j], ray.origin[j]);
        }
        const float normalized_inv_radius = fminf((i + 1.f) * layer_scale,
                                                  grid.background_nlayers - 1);
        int layerid = min((int)floorf(normalized_inv_radius), grid.background_nlayers - 2);
        const float interp_wt = normalized_inv_radius - layerid;
        const float* __restrict__ cubemap_data = grid.background_cubemap + cubemap_step * layerid;

        const CubemapCoord coord = dir_to_cubemap_coord(ray.pos,
                grid.background_reso,
                /* EAC */ true);
        const CubemapBilerpQuery query = cubemap_build_query(coord,
                grid.background_reso);

        float sigma = multi_cubemap_sample(
                cubemap_data,
                cubemap_data + cubemap_step,
                query,
                interp_wt,
                grid.background_reso,
                /*n_channels*/ 4,
                3);
        // printf("SAMP p[%f,%f,%f] invr=%f layerid=%d interp_wt=%f sigma=%f log_li=%f\n",
        //         ray.pos[0], ray.pos[1], ray.pos[2],
        //         normalized_inv_radius,
        //         layerid,
        //         interp_wt,
        //         sigma,
        //         log_transmit);
        if (opt.randomize && opt.random_sigma_std_background > 0.0)
            sigma += ray.rng.randn() * opt.random_sigma_std_background;
        if (sigma > opt.sigma_thresh) {
            const float pcnt = (t - t_last) * ray.world_step * sigma;
            const float weight = _EXP(log_transmit) * (1.f - _EXP(-pcnt));
            log_transmit -= pcnt;
#pragma unroll 3
            for (int i = 0; i < 3; ++i) {
                const float color = multi_cubemap_sample(
                        cubemap_data,
                        cubemap_data + cubemap_step,
                        query,
                        interp_wt,
                        grid.background_reso,
                        /*n_channels*/ 4,
                        i) * C0;
                // printf("%f %d: %f %f\n", r, i, sigma, fmaxf(color + 0.5f, 0.f));
                outv[i] += weight * fmaxf(color + 0.5f, 0.f);  // Clamp to [+0, infty)
            }
            if (_EXP(log_transmit) < opt.stop_thresh) {
                break;
            }
        }
        if (cubemap_data != nullptr)
            cubemap_data += cubemap_step;
        t_last = t;
    }
#pragma unroll 3
    for (int i = 0; i < 3; ++i) {
        out[i] += outv[i] + _EXP(log_transmit) * opt.background_brightness;
    }
}

__device__ __inline__ void render_background_backward(
            const PackedSparseGridSpec& __restrict__ grid,
            const float* __restrict__ grad_output,
            SingleRaySpec& __restrict__ ray,
            const RenderOptions& __restrict__ opt,
            float log_transmit,
            float accum,
            PackedGridOutputGrads& __restrict__ grads
        ) {
    // printf("accum_init=%f\n", accum);
    // printf("log_transmit_init=%f\n", log_transmit);
    ConcentricSpheresIntersector csi(ray.origin, ray.dir);

    float t, t_last;
    const int n_steps = int(grid.background_nlayers / opt.step_size) + 2;
    const int cubemap_step = 6 * grid.background_reso * grid.background_reso;

    const float inner_radius = fmaxf(_dist_ray_to_origin(ray.origin, ray.dir) + 1e-3f, 1.f);
    csi.intersect(inner_radius, &t_last);
    const float layer_scale = (float(grid.background_nlayers - 1) / (n_steps + 1));

    for (int i = 0; i < n_steps; ++i) {
        float r = n_steps / (n_steps - i - 0.5);

        if (r < inner_radius || !csi.intersect(r, &t)) continue;

        const float t_mid = (t + t_last) * 0.5f;
#pragma unroll 3
        for (int j = 0; j < 3; ++j) {
            ray.pos[j] = fmaf(t_mid, ray.dir[j], ray.origin[j]);
        }

        const float normalized_inv_radius = fminf((i + 1.f) * layer_scale,
                                                  grid.background_nlayers - 1);
        int layerid = min((int)floorf(normalized_inv_radius), grid.background_nlayers - 2);
        const float interp_wt = normalized_inv_radius - layerid;
        const float* __restrict__ cubemap_data = grid.background_cubemap + cubemap_step * 4 * layerid;
        float* __restrict__ grad_cubemap_data = grads.grad_background_out == nullptr ? nullptr :
                                   (grads.grad_background_out + cubemap_step * 4 * layerid);
        bool* __restrict__ mask_cubemap_ptr = grads.mask_background_out == nullptr ?  nullptr :
                                   (grads.mask_background_out + cubemap_step * layerid);

        const CubemapCoord coord = dir_to_cubemap_coord(ray.pos,
                grid.background_reso, /* EAC */ true);
        const CubemapBilerpQuery query = cubemap_build_query(coord,
                grid.background_reso);

        float sigma = multi_cubemap_sample(
                cubemap_data,
                cubemap_data + cubemap_step * 4,
                query,
                interp_wt,
                grid.background_reso,
                /*n_channels*/ 4,
                3);
        if (opt.randomize && opt.random_sigma_std_background > 0.0)
            sigma += ray.rng.randn() * opt.random_sigma_std_background;
        if (sigma > opt.sigma_thresh) {
            float total_color = 0.f;
            const float pcnt = ray.world_step * (t - t_last) * sigma;
            const float weight = _EXP(log_transmit) * (1.f - _EXP(-pcnt));
            log_transmit -= pcnt;

            for (int i = 0; i < 3; ++i) {
                const float color = multi_cubemap_sample(
                        cubemap_data,
                        cubemap_data + cubemap_step * 4,
                        query,
                        interp_wt,
                        grid.background_reso,
                        /*n_channels*/ 4,
                        i) * C0 + 0.5f;
                total_color += fmaxf(color, 0.f) * grad_output[i];
                if (color > 0.f) {
                    const float curr_grad_color = C0 * weight * grad_output[i];
                    multi_cubemap_sample_backward(
                            grad_cubemap_data,
                            grad_cubemap_data + cubemap_step * 4,
                            query,
                            interp_wt,
                            grid.background_reso,
                            4,
                            curr_grad_color,
                            i);
                }
            }

            accum -= weight * total_color;
            float curr_grad_sigma = ray.world_step * (t - t_last) * (
                    total_color * _EXP(log_transmit) - accum);

            multi_cubemap_sample_backward(
                    grad_cubemap_data,
                    grad_cubemap_data + cubemap_step * 4,
                    query,
                    interp_wt,
                    grid.background_reso,
                    4,
                    curr_grad_sigma,
                    3,
                    mask_cubemap_ptr,
                    mask_cubemap_ptr + cubemap_step);

            if (_EXP(log_transmit) < opt.stop_thresh) {
                break;
            }
        }
        t_last = t;
    }
}

// BEGIN KERNELS

__launch_bounds__(TRACE_RAY_CUDA_THREADS, MIN_BLOCKS_PER_SM)
__global__ void render_ray_kernel(
        PackedSparseGridSpec grid,
        PackedRaysSpec rays,
        RenderOptions opt,
        torch::PackedTensorAccessor32<float, 2, torch::RestrictPtrTraits> out,
        float* __restrict__ log_transmit_out = nullptr) {
    CUDA_GET_THREAD_ID(tid, int(rays.origins.size(0)) * WARP_SIZE);
    const int ray_id = tid >> 5;
    const int ray_blk_id = threadIdx.x >> 5;
    const int lane_id = threadIdx.x & 0x1F;

    if (lane_id >= grid.sh_data_dim)
        return;

    __shared__ float sphfunc_val[TRACE_RAY_CUDA_RAYS_PER_BLOCK][10];
    __shared__ SingleRaySpec ray_spec[TRACE_RAY_CUDA_RAYS_PER_BLOCK];
    __shared__ typename WarpReducef::TempStorage temp_storage[
        TRACE_RAY_CUDA_RAYS_PER_BLOCK];
    ray_spec[ray_blk_id].set(rays.origins[ray_id].data(),
            rays.dirs[ray_id].data());
    calc_sphfunc(grid, lane_id,
                 ray_id,
                 ray_spec[ray_blk_id].dir,
                 sphfunc_val[ray_blk_id]);
    ray_find_bounds(ray_spec[ray_blk_id], grid, opt, ray_id);
    __syncwarp((1U << grid.sh_data_dim) - 1);

    trace_ray_cuvol(
        grid,
        ray_spec[ray_blk_id],
        opt,
        lane_id,
        sphfunc_val[ray_blk_id],
        temp_storage[ray_blk_id],
        out[ray_id].data(),
        log_transmit_out == nullptr ? nullptr : log_transmit_out + ray_id);
}

__launch_bounds__(TRACE_RAY_BKWD_CUDA_THREADS, MIN_BLOCKS_PER_SM)
__global__ void render_ray_backward_kernel(
    PackedSparseGridSpec grid,
    const float* __restrict__ grad_output,
    const float* __restrict__ color_cache,
    PackedRaysSpec rays,
    RenderOptions opt,
    bool grad_out_is_rgb,
    const float* __restrict__ log_transmit_in,
    float beta_loss,
    float sparsity_loss,
    PackedGridOutputGrads grads,
    float* __restrict__ accum_out = nullptr,
    float* __restrict__ log_transmit_out = nullptr) {
    CUDA_GET_THREAD_ID(tid, int(rays.origins.size(0)) * WARP_SIZE);
    const int ray_id = tid >> 5;
    const int ray_blk_id = threadIdx.x >> 5;
    const int lane_id = threadIdx.x & 0x1F;

    if (lane_id >= grid.sh_data_dim)
        return;

    __shared__ float sphfunc_val[TRACE_RAY_BKWD_CUDA_RAYS_PER_BLOCK][10];
    __shared__ float grad_sphfunc_val[TRACE_RAY_CUDA_RAYS_PER_BLOCK][10];
    __shared__ SingleRaySpec ray_spec[TRACE_RAY_BKWD_CUDA_RAYS_PER_BLOCK];
    __shared__ typename WarpReducef::TempStorage temp_storage[
        TRACE_RAY_CUDA_RAYS_PER_BLOCK];
    ray_spec[ray_blk_id].set(rays.origins[ray_id].data(),
                             rays.dirs[ray_id].data());
    const float vdir[3] = {ray_spec[ray_blk_id].dir[0],
                     ray_spec[ray_blk_id].dir[1],
                     ray_spec[ray_blk_id].dir[2] };
    if (lane_id < grid.basis_dim) {
        grad_sphfunc_val[ray_blk_id][lane_id] = 0.f;
    }
    calc_sphfunc(grid, lane_id,
                 ray_id,
                 vdir, sphfunc_val[ray_blk_id]);
    if (lane_id == 0) {
        ray_find_bounds(ray_spec[ray_blk_id], grid, opt, ray_id);
    }

    float grad_out[3];
    if (grad_out_is_rgb) {
        const float norm_factor = 2.f / (3 * int(rays.origins.size(0)));
#pragma unroll 3
        for (int i = 0; i < 3; ++i) {
            const float resid = color_cache[ray_id * 3 + i] - grad_output[ray_id * 3 + i];
            grad_out[i] = resid * norm_factor;
        }
    } else {
#pragma unroll 3
        for (int i = 0; i < 3; ++i) {
            grad_out[i] = grad_output[ray_id * 3 + i];
        }
    }

    __syncwarp((1U << grid.sh_data_dim) - 1);
    trace_ray_cuvol_backward(
        grid,
        grad_out,
        color_cache + ray_id * 3,
        ray_spec[ray_blk_id],
        opt,
        lane_id,
        sphfunc_val[ray_blk_id],
        grad_sphfunc_val[ray_blk_id],
        temp_storage[ray_blk_id],
        log_transmit_in == nullptr ? 0.f : log_transmit_in[ray_id],
        beta_loss,
        sparsity_loss,
        grads,
        accum_out == nullptr ? nullptr : accum_out + ray_id,
        log_transmit_out == nullptr ? nullptr : log_transmit_out + ray_id);
    calc_sphfunc_backward(
                 grid, lane_id,
                 ray_id,
                 vdir,
                 sphfunc_val[ray_blk_id],
                 grad_sphfunc_val[ray_blk_id],
                 grads.grad_basis_out);
}

__launch_bounds__(TRACE_RAY_BG_CUDA_THREADS, MIN_BG_BLOCKS_PER_SM)
__global__ void render_background_kernel(
        PackedSparseGridSpec grid,
        PackedRaysSpec rays,
        RenderOptions opt,
        const float* __restrict__ log_transmit,
        // Outputs
        torch::PackedTensorAccessor32<float, 2, torch::RestrictPtrTraits> out) {
    CUDA_GET_THREAD_ID(ray_id, int(rays.origins.size(0)));
    if (log_transmit[ray_id] < -25.f) return;
    SingleRaySpec ray_spec(rays.origins[ray_id].data(), rays.dirs[ray_id].data());
    ray_find_bounds_bg(ray_spec, grid, opt, ray_id);
    render_background_forward(
        grid,
        ray_spec,
        opt,
        log_transmit[ray_id],
        out[ray_id].data());
}

__launch_bounds__(TRACE_RAY_BG_CUDA_THREADS, MIN_BG_BLOCKS_PER_SM)
__global__ void render_background_backward_kernel(
        PackedSparseGridSpec grid,
        const float* __restrict__ grad_output,
        const float* __restrict__ color_cache,
        PackedRaysSpec rays,
        RenderOptions opt,
        const float* __restrict__ log_transmit,
        const float* __restrict__ accum,
        bool grad_out_is_rgb,
        // Outputs
        PackedGridOutputGrads grads) {
    CUDA_GET_THREAD_ID(ray_id, int(rays.origins.size(0)));
    if (log_transmit[ray_id] < -25.f) return;
    SingleRaySpec ray_spec(rays.origins[ray_id].data(), rays.dirs[ray_id].data());
    ray_find_bounds_bg(ray_spec, grid, opt, ray_id);

    float grad_out[3];
    if (grad_out_is_rgb) {
        const float norm_factor = 2.f / (3 * int(rays.origins.size(0)));
#pragma unroll 3
        for (int i = 0; i < 3; ++i) {
            const float resid = color_cache[ray_id * 3 + i] - grad_output[ray_id * 3 + i];
            grad_out[i] = resid * norm_factor;
        }
    } else {
#pragma unroll 3
        for (int i = 0; i < 3; ++i) {
            grad_out[i] = grad_output[ray_id * 3 + i];
        }
    }

    render_background_backward(
        grid,
        grad_out,
        ray_spec,
        opt,
        log_transmit[ray_id],
        accum[ray_id],
        grads);
}

}  // namespace device

torch::Tensor _get_empty_1d(const torch::Tensor& origins) {
    auto options =
        torch::TensorOptions()
        .dtype(origins.dtype())
        .layout(torch::kStrided)
        .device(origins.device())
        .requires_grad(false);
    return torch::empty({origins.size(0)}, options);
}

}  // namespace

torch::Tensor volume_render_cuvol(SparseGridSpec& grid, RaysSpec& rays, RenderOptions& opt) {
    DEVICE_GUARD(grid.sh_data);
    grid.check();
    rays.check();


    const auto Q = rays.origins.size(0);

    torch::Tensor results = torch::empty_like(rays.origins);

    bool use_background = grid.background_cubemap.size(0) > 0;
    torch::Tensor log_transmit;
    if (use_background) {
        log_transmit = _get_empty_1d(rays.origins);
    }

    {
        const int cuda_n_threads = TRACE_RAY_CUDA_THREADS;
        const int blocks = CUDA_N_BLOCKS_NEEDED(Q * WARP_SIZE, cuda_n_threads);
        device::render_ray_kernel<<<blocks, cuda_n_threads>>>(
                grid, rays, opt,
                // Output
                results.packed_accessor32<float, 2, torch::RestrictPtrTraits>(),
                use_background ? log_transmit.data_ptr<float>() : nullptr);
    }

    if (use_background) {
        // printf("RENDER BG\n");
        const int blocks = CUDA_N_BLOCKS_NEEDED(Q, TRACE_RAY_BG_CUDA_THREADS);
        device::render_background_kernel<<<blocks, TRACE_RAY_BG_CUDA_THREADS>>>(
                grid,
                rays,
                opt,
                log_transmit.data_ptr<float>(),
                results.packed_accessor32<float, 2, torch::RestrictPtrTraits>());
    }

    CUDA_CHECK_ERRORS;
    return results;
}

void volume_render_cuvol_backward(
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
    const auto Q = rays.origins.size(0);

    bool use_background = grid.background_cubemap.size(0) > 0;
    torch::Tensor log_transmit, accum;
    if (use_background) {
        log_transmit = _get_empty_1d(rays.origins);
        accum = _get_empty_1d(rays.origins);
    }

    {
        const int cuda_n_threads_render_backward = TRACE_RAY_BKWD_CUDA_THREADS;
        const int blocks = CUDA_N_BLOCKS_NEEDED(Q * WARP_SIZE, cuda_n_threads_render_backward);
        device::render_ray_backward_kernel<<<blocks,
            cuda_n_threads_render_backward>>>(
                    grid,
                    grad_out.data_ptr<float>(),
                    color_cache.data_ptr<float>(),
                    rays, opt,
                    false,
                    nullptr,
                    0.f,
                    0.f,
                    // Output
                    grads,
                    use_background ? accum.data_ptr<float>() : nullptr,
                    use_background ? log_transmit.data_ptr<float>() : nullptr);
    }

    if (use_background) {
        const int blocks = CUDA_N_BLOCKS_NEEDED(Q, TRACE_RAY_BG_CUDA_THREADS);
        device::render_background_backward_kernel<<<blocks, TRACE_RAY_BG_CUDA_THREADS>>>(
                grid,
                grad_out.data_ptr<float>(),
                color_cache.data_ptr<float>(),
                rays,
                opt,
                log_transmit.data_ptr<float>(),
                accum.data_ptr<float>(),
                false,
                // Output
                grads);
    }

    CUDA_CHECK_ERRORS;
}

void volume_render_cuvol_fused(
        SparseGridSpec& grid,
        RaysSpec& rays,
        RenderOptions& opt,
        torch::Tensor rgb_gt,
        float beta_loss,
        float sparsity_loss,
        torch::Tensor rgb_out,
        GridOutputGrads& grads) {

    DEVICE_GUARD(grid.sh_data);
    CHECK_INPUT(rgb_gt);
    CHECK_INPUT(rgb_out);
    grid.check();
    rays.check();
    grads.check();
    const auto Q = rays.origins.size(0);

    bool use_background = grid.background_cubemap.size(0) > 0;
    bool need_log_transmit = use_background || beta_loss > 0.f;
    torch::Tensor log_transmit, accum;
    if (need_log_transmit) {
        log_transmit = _get_empty_1d(rays.origins);
    }
    if (use_background) {
        accum = _get_empty_1d(rays.origins);
    }

    {
        const int blocks = CUDA_N_BLOCKS_NEEDED(Q * WARP_SIZE, TRACE_RAY_CUDA_THREADS);
        device::render_ray_kernel<<<blocks, TRACE_RAY_CUDA_THREADS>>>(
                grid, rays, opt,
                // Output
                rgb_out.packed_accessor32<float, 2, torch::RestrictPtrTraits>(),
                need_log_transmit ? log_transmit.data_ptr<float>() : nullptr);
    }

    if (use_background) {
        const int blocks = CUDA_N_BLOCKS_NEEDED(Q, TRACE_RAY_BG_CUDA_THREADS);
        device::render_background_kernel<<<blocks, TRACE_RAY_BG_CUDA_THREADS>>>(
                grid,
                rays,
                opt,
                log_transmit.data_ptr<float>(),
                rgb_out.packed_accessor32<float, 2, torch::RestrictPtrTraits>());
    }

    {
        const int blocks = CUDA_N_BLOCKS_NEEDED(Q * WARP_SIZE, TRACE_RAY_BKWD_CUDA_THREADS);
        device::render_ray_backward_kernel<<<blocks, TRACE_RAY_BKWD_CUDA_THREADS>>>(
                grid,
                rgb_gt.data_ptr<float>(),
                rgb_out.data_ptr<float>(),
                rays, opt,
                true,
                beta_loss > 0.f ? log_transmit.data_ptr<float>() : nullptr,
                beta_loss / Q,
                sparsity_loss,
                // Output
                grads,
                use_background ? accum.data_ptr<float>() : nullptr,
                nullptr);
    }

    if (use_background) {
        const int blocks = CUDA_N_BLOCKS_NEEDED(Q, TRACE_RAY_BG_CUDA_THREADS);
        device::render_background_backward_kernel<<<blocks, TRACE_RAY_BG_CUDA_THREADS>>>(
                grid,
                rgb_gt.data_ptr<float>(),
                rgb_out.data_ptr<float>(),
                rays,
                opt,
                log_transmit.data_ptr<float>(),
                accum.data_ptr<float>(),
                true,
                // Output
                grads);
    }

    CUDA_CHECK_ERRORS;
}
