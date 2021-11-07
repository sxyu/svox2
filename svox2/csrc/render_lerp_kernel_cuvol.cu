// Copyright 2021 Alex Yu
#include <torch/extension.h>
#include "cuda_util.cuh"
#include "random_util.cuh"
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
        float* __restrict__ out) {
    const uint32_t lane_colorgrp_id = lane_id % grid.basis_dim;
    const uint32_t lane_colorgrp = lane_id / grid.basis_dim;

    if (ray.tmin > ray.tmax && grid.background_nlayers == 0) {
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
            outv += weight * fmaxf(lane_color_total + 0.5f, 0.f);  // Clamp to [+0, 1]
            if (_EXP(light_intensity) < opt.stop_thresh) {
                const float renorm_val = 1.f / (1.f - _EXP(light_intensity));
                if (lane_colorgrp_id == 0) {
                    out[lane_colorgrp] *= renorm_val;
                }
                light_intensity = -1e3f;
                break;
            }
        }
        t += opt.step_size;
    }

    if (grid.background_nlayers > 0 && light_intensity > -20.f) {
        // TODO WIP

        ConcentricSpheresIntersector csi(
                grid.size,
                ray.origin,
                ray.dir,
                ray.world_step / opt.step_size);

        // float t_last;
        // const float r_min = fmaxf(_dist_ray_to_origin(csi.origin, csi.dir), opt.background_msi_scale);
        // csi.intersect(r_min + 1e-4f, &t_last);
        // if (csi.dir[0] < 0.0) {
        //     printf("ray_ori=[%f, %f, %f] ray_dir=[%f, %f, %f] ray_w=[%f]\n",
        //             ray.origin[0], ray.origin[1], ray.origin[2],
        //             ray.dir[0], ray.dir[1], ray.dir[2],
        //             ray.world_step
        //           );
        //     printf("csi_ori=[%f, %f, %f] csi_dir=[%f, %f, %f] csi_wss=%f csi_q2a=%f qb=%f f=%f r_min=%f t_last=%f\n",
        //             csi.origin[0], csi.origin[1], csi.origin[2],
        //             csi.dir[0], csi.dir[1], csi.dir[2],
        //             csi.world_step_scale,
        //             csi.q2a,
        //             csi.qb,
        //             csi.f,
        //             r_min,
        //             t_last
        //           );
        // }

        const float* cubemap_data = grid.background_cubemap;
        const int cubemap_step = 6 * grid.background_reso * grid.background_reso * /*n_channels*/ 4;
        for (int i = 0; i < grid.background_nlayers; ++i) {
            const float radius = opt.background_msi_scale * float(grid.background_nlayers) /
                                (float(grid.background_nlayers - i - 0.5f));
            const float thickness = radius - opt.background_msi_scale * float(grid.background_nlayers) /
                                (float(grid.background_nlayers - i));
            float t_inter;
            // if (csi.dir[0] < 0.0) {
            //     printf("i=%d r=%f\n", i, radius);
            // }
            if (csi.intersect(radius, &t_inter)) {
#pragma unroll 3
                for (int j = 0; j < 3; ++j) {
                    ray.pos[j] = fmaf(t_inter, csi.dir[j], csi.origin[j]);
                }
                // if (csi.dir[0] < 0.0) {
                //     printf(" I! t_inter=%f pos=[%f, %f, %f]\n",
                //             t_inter,
                //             ray.pos[0], ray.pos[1], ray.pos[2]);
                // }

                const CubemapCoord coord = dir_to_cubemap_coord(ray.pos,
                                                                grid.background_reso, /* EAC */ true);
                const CubemapBilerpQuery query = cubemap_build_query(coord,
                                                                     grid.background_reso);


                const float sigma = cubemap_sample(cubemap_data,
                                           query,
                                           grid.background_reso,
                                           /*n_channels*/ 4,
                                           3);
                if (sigma > opt.sigma_thresh) {
                    const float group_color = cubemap_sample(cubemap_data,
                            query,
                            grid.background_reso,
                            /*n_channels*/ 4,
                            lane_colorgrp);

                    const float pcnt = csi.world_step_scale * thickness * sigma;
                    const float weight = _EXP(light_intensity) * (1.f - _EXP(-pcnt));
                    light_intensity -= pcnt;
                    // if (csi.dir[0] < 0.0) {
                    //     printf(" wsc=%f, t_inter=%f, t_last=%f, sigma=%f, weight=%f, li=%f\n",
                    //             csi.world_step_scale, t_inter, t_last, sigma, weight, light_intensity);
                    // }

                    outv += weight * fmaxf(group_color + 0.5f, 0.f);  // Clamp to [+0, infty)
                }
                // t_last = t_inter;
            }
            if (cubemap_data != nullptr)
                cubemap_data += cubemap_step;
        }
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
        const float* __restrict__ sphfunc_val,
        float* __restrict__ grad_sphfunc_val,
        WarpReducef::TempStorage& __restrict__ temp_storage,
        PackedGridOutputGrads& __restrict__ grads
        ) {
    const uint32_t lane_colorgrp_id = lane_id % grid.basis_dim;
    const uint32_t lane_colorgrp = lane_id / grid.basis_dim;
    const uint32_t leader_mask = 1U | (1U << grid.basis_dim) | (1U << (2 * grid.basis_dim));

    if (ray.tmin > ray.tmax && grid.background_nlayers == 0) return;
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
            const float weight = _EXP(light_intensity) * (1.f - _EXP(-pcnt));
            light_intensity -= pcnt;

            const float lane_color_total = WarpReducef(temp_storage).HeadSegmentedSum(
                                           weighted_lane_color, lane_colorgrp_id == 0) + 0.5f;
            float total_color = fmaxf(lane_color_total, 0.f);
            float color_in_01 = total_color == lane_color_total;
            total_color *= gout; // Clamp to [+0, 1]

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
                    total_color * _EXP(light_intensity) - accum);
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
            if (_EXP(light_intensity) < opt.stop_thresh) {
                break;
            }
        }
        t += opt.step_size;
    }

    if (grid.background_nlayers > 0 && light_intensity > -20.f &&
        grads.grad_background_out != nullptr) {
        // Performance SUCKS

        ConcentricSpheresIntersector csi(
                grid.size,
                ray.origin,
                ray.dir,
                ray.world_step / opt.step_size);

        // const float r_min = fmaxf(_dist_ray_to_origin(csi.origin, csi.dir), opt.background_msi_scale);
        // float t_last;
        // csi.intersect(r_min + 1e-4f, &t_last);

        const float* cubemap_data = grid.background_cubemap;
        float* grad_cubemap_data = grads.grad_background_out;
        bool* mask_cubemap_ptr = grads.mask_background_out;
        const int cubemap_step = 6 * grid.background_reso * grid.background_reso;
        for (int i = 0; i < grid.background_nlayers; ++i) {
            const float radius = opt.background_msi_scale * float(
                    grid.background_nlayers) / (float(grid.background_nlayers - i - 0.5f));
            const float thickness = radius - opt.background_msi_scale * float(grid.background_nlayers) /
                                (float(grid.background_nlayers - i));
            float t_inter;
            if (csi.intersect(radius, &t_inter)) {
#pragma unroll 3
                for (int j = 0; j < 3; ++j) {
                    ray.pos[j] = fmaf(t_inter, csi.dir[j], csi.origin[j]);
                }

                const CubemapCoord coord = dir_to_cubemap_coord(ray.pos,
                                                                grid.background_reso, /* EAC */ true);
                const CubemapBilerpQuery query = cubemap_build_query(coord,
                                                                     grid.background_reso);


                const float sigma = cubemap_sample(cubemap_data,
                                           query,
                                           grid.background_reso,
                                           /*n_channels*/ 4,
                                           3);
                if (sigma > opt.sigma_thresh) {
                    const float group_color = cubemap_sample(cubemap_data,
                            query,
                            grid.background_reso,
                            /*n_channels*/ 4,
                            lane_colorgrp) + 0.5f;

                    const float pcnt = csi.world_step_scale * thickness * sigma;
                    const float weight = _EXP(light_intensity) * (1.f - _EXP(-pcnt));
                    light_intensity -= pcnt;

                    float total_color = fmaxf(group_color, 0.f);
                    float color_in_01 = total_color == group_color;
                    total_color *= gout;

                    float total_color_c1 = __shfl_sync(leader_mask, total_color, grid.basis_dim);
                    total_color += __shfl_sync(leader_mask, total_color, 2 * grid.basis_dim);
                    total_color += total_color_c1;

                    const float curr_grad_color = weight * color_in_01 * gout;

                    accum -= weight * total_color;
                    float curr_grad_sigma = csi.world_step_scale * thickness * (
                            total_color * _EXP(light_intensity) - accum);

                    if (lane_colorgrp_id == 0) {
                        cubemap_sample_backward(
                                grad_cubemap_data,
                                query,
                                grid.background_reso,
                                4,
                                curr_grad_color,
                                lane_colorgrp);

                        if (lane_id == 0) {
                            cubemap_sample_backward(
                                    grad_cubemap_data,
                                    query,
                                    grid.background_reso,
                                    4,
                                    curr_grad_sigma,
                                    3,
                                    mask_cubemap_ptr);
                        }
                    }
                }

                // t_last = t_inter;
            }
            if (cubemap_data != nullptr)
                cubemap_data += cubemap_step * 4 /* n_channels */;
            if (grad_cubemap_data != nullptr)
                grad_cubemap_data += cubemap_step * 4/* n_channels */;
            if (mask_cubemap_ptr != nullptr)
                mask_cubemap_ptr += cubemap_step;
        }
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
    if (lane_id == 0) {
        ray_find_bounds(ray_spec[ray_blk_id], grid, opt);
    }
    __syncwarp((1U << grid.sh_data_dim) - 1);

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
    const float* __restrict__ grad_output,
    const float* __restrict__ color_cache,
    PackedRaysSpec rays,
    RenderOptions opt,
    PackedGridOutputGrads grads,
    bool grad_out_is_rgb = false) {
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
        ray_find_bounds(ray_spec[ray_blk_id], grid, opt);
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
        grads);
    calc_sphfunc_backward(
                 grid, lane_id,
                 ray_id,
                 vdir,
                 sphfunc_val[ray_blk_id],
                 grad_sphfunc_val[ray_blk_id],
                 grads.grad_basis_out);
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

    const int cuda_n_threads_render_backward = TRACE_RAY_BKWD_CUDA_THREADS;
    const int blocks = CUDA_N_BLOCKS_NEEDED(Q * WARP_SIZE, cuda_n_threads_render_backward);

    device::render_ray_backward_kernel<<<blocks,
           cuda_n_threads_render_backward>>>(
            grid,
            grad_out.data_ptr<float>(),
            color_cache.data_ptr<float>(),
            rays, opt,
            // Output
            grads);

    CUDA_CHECK_ERRORS;
}

void volume_render_cuvol_fused(
        SparseGridSpec& grid,
        RaysSpec& rays,
        RenderOptions& opt,
        torch::Tensor rgb_gt,
        torch::Tensor rgb_out,
        GridOutputGrads& grads) {

    DEVICE_GUARD(grid.sh_data);
    CHECK_INPUT(rgb_gt);
    CHECK_INPUT(rgb_out);
    grid.check();
    rays.check();
    grads.check();
    const auto Q = rays.origins.size(0);

    {
        const int blocks = CUDA_N_BLOCKS_NEEDED(Q * WARP_SIZE, TRACE_RAY_CUDA_THREADS);
        device::render_ray_kernel<<<blocks, TRACE_RAY_CUDA_THREADS>>>(
                grid, rays, opt,
                // Output
                rgb_out.packed_accessor32<float, 2, torch::RestrictPtrTraits>());
    }

    {
        const int blocks = CUDA_N_BLOCKS_NEEDED(Q * WARP_SIZE, TRACE_RAY_BKWD_CUDA_THREADS);
        device::render_ray_backward_kernel<<<blocks, TRACE_RAY_BKWD_CUDA_THREADS>>>(
                grid,
                rgb_gt.data_ptr<float>(),
                rgb_out.data_ptr<float>(),
                rays, opt,
                // Output
                grads,
                true);
    }

    CUDA_CHECK_ERRORS;
}
