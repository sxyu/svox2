// Surface rendering traversal version
#include <torch/extension.h>
#include "cuda_util.cuh"
#include "data_spec_packed.cuh"
#include "render_util.cuh"

#include <iostream>
#include <cstdint>
#include <tuple>
// #include <math.h>
#include <bits/stdc++.h>
#include <thrust/extrema.h>
#include <assert.h>
#include <thrust/execution_policy.h>

namespace {
const int WARP_SIZE = 32;

const int TRACE_RAY_CUDA_THREADS = 128;
const int TRACE_RAY_CUDA_RAYS_PER_BLOCK = TRACE_RAY_CUDA_THREADS / WARP_SIZE;

const int TRACE_RAY_BKWD_CUDA_THREADS = 128;
const int TRACE_RAY_BKWD_CUDA_RAYS_PER_BLOCK = TRACE_RAY_BKWD_CUDA_THREADS / WARP_SIZE;

const int MIN_BLOCKS_PER_SM = 8; // why?

const int TRACE_RAY_BG_CUDA_THREADS = 128;
const int MIN_BG_BLOCKS_PER_SM = 8;
typedef cub::WarpReduce<float> WarpReducef;

namespace device {


// * For ray rendering
__device__ __inline__ void trace_ray_surf_trav(
        const PackedSparseGridSpec& __restrict__ grid,
        SingleRaySpec& __restrict__ ray,
        // const PackedRayVoxIntersecSpec& __restrict__ ray_vox,
        const RenderOptions& __restrict__ opt,
        uint32_t lane_id,
        float* __restrict__ sphfunc_val,
        WarpReducef::TempStorage& __restrict__ temp_storage,
        float* __restrict__ out,
        float* __restrict__ out_log_transmit) {
    const uint32_t lane_colorgrp_id = lane_id % grid.basis_dim; // (9) every basis in SH has a lane
    const uint32_t lane_colorgrp = lane_id / grid.basis_dim;

    double const  ray_dir_d[] = {ray.dir[0], ray.dir[1], ray.dir[2]};
    double const  ray_origin_d[] = {ray.origin[0], ray.origin[1], ray.origin[2]};

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

    int32_t last_voxel[] = {-1,-1,-1};

    while (t <= ray.tmax) {
        int32_t voxel_l[3];
#pragma unroll 3
        for (int j = 0; j < 3; ++j) {
            voxel_l[j] = static_cast<int32_t>(fmaf(t, ray.dir[j], ray.origin[j])); // fmaf(x,y,z) = (x*y)+z
            voxel_l[j] = min(max(voxel_l[j], 0), grid.size[j] - 2);
        }

        // if ((voxel_l[0] == 27) && (voxel_l[1] == 13) && (voxel_l[2] == 46)){
        //     printf("found! \n");
        // }
        

        if ((voxel_l[0] == last_voxel[0]) && (voxel_l[1] == last_voxel[1]) && (voxel_l[2] == last_voxel[2])){
            // const float skip = compute_skip_dist(ray,
            //             grid.links, grid.stride_x,
            //             grid.size[2], 0);

            t += opt.step_size;
            continue;
        }

        int const offx = grid.stride_x, offy = grid.size[2];
        const int32_t* __restrict__ link_ptr = grid.links + (offx * voxel_l[0] + offy * voxel_l[1] + voxel_l[2]);

        // skip voxel if any of the vertices is turned off
        if ((voxel_l[0] + 1 >= grid.size[0]) || (voxel_l[1] + 1 >= grid.size[1]) || (voxel_l[2] + 1 >= grid.size[2]) \
            || (link_ptr[0] < 0) || (link_ptr[1] < 0) || (link_ptr[offy] < 0) || (link_ptr[offy+1] < 0) \
            || (link_ptr[offx] < 0) || (link_ptr[offx+1] < 0) || (link_ptr[offx+offy] < 0) || (link_ptr[offx+offy+1] < 0)
        ){
            // const float skip = compute_skip_dist(ray,
            //             grid.links, grid.stride_x,
            //             grid.size[2], 0);

            t += opt.step_size;
            continue;
        }

        last_voxel[0] = voxel_l[0];
        last_voxel[1] = voxel_l[1];
        last_voxel[2] = voxel_l[2];


        // check minimal of alpha raw
        if ((grid.density_data[link_ptr[0]] < opt.sigma_thresh) && \
            (grid.density_data[link_ptr[1]] < opt.sigma_thresh) && \
            (grid.density_data[link_ptr[offy]] < opt.sigma_thresh) && \
            (grid.density_data[link_ptr[offy+1]] < opt.sigma_thresh) && \
            (grid.density_data[link_ptr[offx]] < opt.sigma_thresh) && \
            (grid.density_data[link_ptr[offx+1]] < opt.sigma_thresh) && \
            (grid.density_data[link_ptr[offx+offy]] < opt.sigma_thresh) && \
            (grid.density_data[link_ptr[offx+offy+1]] < opt.sigma_thresh)){
                // const float skip = compute_skip_dist(ray,
                //             grid.links, grid.stride_x,
                //             grid.size[2], 0);

                t += opt.step_size;
                continue;
            }

        // find intersections
        double const surface[8] = {
            grid.surface_data[link_ptr[0]],
            grid.surface_data[link_ptr[1]],
            grid.surface_data[link_ptr[offy]],
            grid.surface_data[link_ptr[offy+1]],
            grid.surface_data[link_ptr[offx]],
            grid.surface_data[link_ptr[offx+1]],
            grid.surface_data[link_ptr[offx+offy]],
            grid.surface_data[link_ptr[offx+offy+1]],
        };

        double fs[4];
        surface_to_cubic_equation(surface, ray_origin_d, ray_dir_d, voxel_l, fs);

        // only supports single level set!
        const int level_set_num = 1;
        const auto mnmax = thrust::minmax_element(thrust::device, surface, surface+8);
        for (int i=0; i < level_set_num; ++i){
            double const lv_set = grid.level_set_data[i];
            if ((lv_set < *mnmax.first) || (lv_set > *mnmax.second)){
                continue;
            }
            // float const f0_lv = f0 - lv_set;

            // probably better ways to find roots
            // https://stackoverflow.com/questions/4906556/what-is-a-simple-way-to-find-real-roots-of-a-cubic-polynomial
            // https://www.sciencedirect.com/science/article/pii/B9780125434577500097
            // https://stackoverflow.com/questions/13328676/c-solving-cubic-equations


            ////////////// CUBIC ROOT SOLVING //////////////
            // float const eps = 1e-8;
            // float const eps_double = 1e-10;
            double st[3] = {-1, -1, -1}; // sample t

            cubic_equation_solver(
                fs[0] - lv_set, fs[1], fs[2], fs[3],
                1e-8, // float eps
                1e-10, // double eps
                st
                );

            bool has_sample = false;
            
            ////////////// TRILINEAR INTERPOLATE //////////////
            for (int j=0; j < 3; ++j){
                if (st[j] <= 0){
                    // ignore intersection at negative direction
                    continue;
                }

#pragma unroll 3
                for (int k=0; k < 3; ++k){
                    // assert(!isnan(st[j]));
                    ray.pos[k] = fmaf(static_cast<float>(st[j]), ray.dir[k], ray.origin[k]); // fmaf(x,y,z) = (x*y)+z
                    ray.l[k] = min(voxel_l[k], grid.size[k] - 2); // get l
                    ray.pos[k] -= static_cast<float>(ray.l[k]); // get trilinear interpolate distances
                }

                // float const volatile ray_pos[] = {ray.pos[0], ray.pos[1], ray.pos[2]};
                // float const volatile ray_dir[] = {ray.dir[0], ray.dir[1], ray.dir[2]};
                // float const volatile ray_origin[] = {ray.origin[0], ray.origin[1], ray.origin[2]};

                // float const volatile ray_pos0[] = {fmaf(st[j], ray.dir[0], ray.origin[0]), fmaf(st[j], ray.dir[1], ray.origin[1]), fmaf(st[j], ray.dir[2], ray.origin[2])};
                // int32_t const volatile ray_l[] = {ray.l[0], ray.l[1], ray.l[2]};

                // check if intersection is within grid
                if ((ray.pos[0] < 0) | (ray.pos[0] > 1) | (ray.pos[1] < 0) | (ray.pos[1] > 1) | (ray.pos[2] < 0) | (ray.pos[2] > 1)){
                    continue;
                }

                has_sample = true;
                float alpha = trilerp_cuvol_one(
                        grid.links, grid.density_data,
                        grid.stride_x,
                        grid.size[2],
                        1,
                        ray.l, ray.pos,
                        0);

                if (alpha > opt.sigma_thresh) {
                    alpha = _SIGMOID(alpha);
                    float lane_color = trilerp_cuvol_one(
                                    grid.links,
                                    grid.sh_data,
                                    grid.stride_x,
                                    grid.size[2],
                                    grid.sh_data_dim,
                                    ray.l, ray.pos, lane_id);
                    lane_color *= sphfunc_val[lane_colorgrp_id]; // bank conflict

                    // const float pcnt = ray.world_step * sigma;
                    const float pcnt = -1 * _LOG(1 - alpha);
                    const float weight = _EXP(log_transmit) * (1.f - _EXP(-pcnt));
                    log_transmit -= pcnt; // log_trans = sum(log(1-alpha)) = log(prod(1-alpha))
                    // log_transmit is now T_{i+1}

                    float lane_color_total = WarpReducef(temp_storage).HeadSegmentedSum(
                                                lane_color, lane_colorgrp_id == 0); // segment lanes into RGB channel, them sum
                    outv += weight * fmaxf(lane_color_total + 0.5f, 0.f);  // Clamp to [+0, infty)
                }
            }



            if ((!has_sample) && (opt.surf_fake_sample)){
                // there is no intersection between ray and surface
                // take fake sample if allowed


                // if (lane_id == 0){
                //     printf("==========\n");
                //     printf("taking fake sample for: [%d, %d, %d]\n", voxel_l[0], voxel_l[1], voxel_l[2]);
                // }

                // https://math.stackexchange.com/questions/967268/finding-the-closest-distance-between-a-point-a-curve
                

                // first find middle point of the ray in the voxel
                int32_t const close_plane[] = {
                    ray.dir[0] > 0.f ? voxel_l[0] : voxel_l[0]+1,
                    ray.dir[1] > 0.f ? voxel_l[1] : voxel_l[1]+1,
                    ray.dir[2] > 0.f ? voxel_l[2] : voxel_l[2]+1,
                };
                int32_t const far_plane[] = {
                    ray.dir[0] > 0.f ? voxel_l[0]+1 : voxel_l[0],
                    ray.dir[1] > 0.f ? voxel_l[1]+1 : voxel_l[1],
                    ray.dir[2] > 0.f ? voxel_l[2]+1 : voxel_l[2],
                };

                float const t_close = max(
                    max((static_cast<float>(close_plane[0])-ray.origin[0])/ray.dir[0], (static_cast<float>(close_plane[1])-ray.origin[1])/ray.dir[1]),
                    (static_cast<float>(close_plane[2])-ray.origin[2])/ray.dir[2]);
                float const t_far = min(
                    min((static_cast<float>(far_plane[0])-ray.origin[0])/ray.dir[0], (static_cast<float>(far_plane[1])-ray.origin[1])/ray.dir[1]),
                    (static_cast<float>(far_plane[2])-ray.origin[2])/ray.dir[2]);



                if ((t_far - t_close) > opt.surf_fake_sample_min_vox_len){
#pragma unroll 3
                    for (int k=0; k < 3; ++k){
                        // assert(!isnan(st[j]));
                        ray.pos[k] = fmaf((t_far + t_close) / 2, ray.dir[k], ray.origin[k]); // fmaf(x,y,z) = (x*y)+z
                        ray.l[k] = min(voxel_l[k], grid.size[k] - 2); // get l
                        ray.pos[k] -= static_cast<float>(ray.l[k]); // get trilinear interpolate distances

                        // if ((!(ray.pos[k] >= 0.f)) || (!(ray.pos[k] <= 1.f)) ){
                        //     printf("t_far: %f\n", t_far);
                        //     printf("t_close: %f\n", t_close);
                        //     printf("ray_l_k: %d\n", ray.l[k]);
                        //     printf("ray_pos_k: %f\n", ray.pos[k]);
                        // }
                        
                        // assert(ray.pos[k] <= 1.f);
                        // assert(ray.pos[k] >= 0.f);
                    }

                    float alpha = trilerp_cuvol_one(
                            grid.links, grid.density_data,
                            grid.stride_x,
                            grid.size[2],
                            1,
                            ray.l, ray.pos,
                            0);

                    if (alpha > opt.sigma_thresh) {
                        alpha = _SIGMOID(alpha);

                        // use distance to surface to re-weight alpha
                        // https://math.stackexchange.com/questions/1815397/distance-between-point-and-parametric-line
                        // we approximate the distance by normalizing the surface scalar values
                        // so the distance no longer relates to the scale of surface

                        float const surf_norm = sqrtf(
                            max(1e-9f, 
                            _SQR(surface[0]) + _SQR(surface[1]) + _SQR(surface[2]) + _SQR(surface[3]) + _SQR(surface[4]) + _SQR(surface[5]) + _SQR(surface[6]) + _SQR(surface[7])
                            )
                        );

                        // tri-lerp to get distance

                        #define _norm_surf(x) (static_cast<float>(surface[x]) / surf_norm)

                        const float ix0y0 = lerp(_norm_surf(0), _norm_surf(1), ray.pos[2]);
                        const float ix0y1 = lerp(_norm_surf(2), _norm_surf(3), ray.pos[2]);
                        const float ix0 = lerp(ix0y0, ix0y1, ray.pos[1]);
                        const float ix1y0 = lerp(_norm_surf(4), _norm_surf(5), ray.pos[2]);
                        const float ix1y1 = lerp(_norm_surf(6),
                                                _norm_surf(7), ray.pos[2]);
                        const float ix1 = lerp(ix1y0, ix1y1, ray.pos[1]);
                        const float fake_sample_dist = lerp(ix0, ix1, ray.pos[0]);

                        #undef _norm_surf

                        
                        // re-weight alpha using a simple gaussian
                        alpha = alpha * _EXP(-.5 * _SQR(fake_sample_dist/grid.fake_sample_std));


                        float lane_color = trilerp_cuvol_one(
                                        grid.links,
                                        grid.sh_data,
                                        grid.stride_x,
                                        grid.size[2],
                                        grid.sh_data_dim,
                                        ray.l, ray.pos, lane_id);
                        lane_color *= sphfunc_val[lane_colorgrp_id]; // bank conflict

                        // const float pcnt = ray.world_step * sigma;
                        const float pcnt = -1 * _LOG(1 - alpha);
                        const float weight = _EXP(log_transmit) * (1.f - _EXP(-pcnt));
                        log_transmit -= pcnt; // log_trans = sum(log(1-alpha)) = log(prod(1-alpha))
                        // log_transmit is now T_{i+1}

                        float lane_color_total = WarpReducef(temp_storage).HeadSegmentedSum(
                                                    lane_color, lane_colorgrp_id == 0); // segment lanes into RGB channel, them sum
                        outv += weight * fmaxf(lane_color_total + 0.5f, 0.f);  // Clamp to [+0, infty)
                    }

                }

                

            }

        }

        if (_EXP(log_transmit) < opt.stop_thresh) {
            log_transmit = -1e3f;
            break;
        }

        t += opt.step_size;

    }

    if (grid.background_nlayers == 0) {
        outv += _EXP(log_transmit) * opt.background_brightness; // 1-sum(weight)[i=0~N] = T_{N+1} !! 
    }
    if (lane_colorgrp_id == 0) {
        if (out_log_transmit != nullptr) {
            *out_log_transmit = log_transmit;
        }
        out[lane_colorgrp] = outv;
    }
}

__device__ __inline__ void trace_ray_expected_term(
        const PackedSparseGridSpec& __restrict__ grid,
        SingleRaySpec& __restrict__ ray,
        const RenderOptions& __restrict__ opt,
        float* __restrict__ out) {
    if (ray.tmin > ray.tmax) {
        *out = 0.f;
        return;
    }

    double const  ray_dir_d[] = {ray.dir[0], ray.dir[1], ray.dir[2]};
    double const  ray_origin_d[] = {ray.origin[0], ray.origin[1], ray.origin[2]};

    float t = ray.tmin;
    float outv = 0.f;

    float log_transmit = 0.f;
    // printf("tmin %f, tmax %f \n", ray.tmin, ray.tmax);

    int32_t last_voxel[] = {-1,-1,-1};

    while (t <= ray.tmax) {
        int32_t voxel_l[3];
#pragma unroll 3
        for (int j = 0; j < 3; ++j) {
            voxel_l[j] = static_cast<int32_t>(fmaf(t, ray.dir[j], ray.origin[j])); // fmaf(x,y,z) = (x*y)+z
            voxel_l[j] = min(max(voxel_l[j], 0), grid.size[j] - 2);
        }

        if ((voxel_l[0] == last_voxel[0]) && (voxel_l[1] == last_voxel[1]) && (voxel_l[2] == last_voxel[2])){
            // const float skip = compute_skip_dist(ray,
            //             grid.links, grid.stride_x,
            //             grid.size[2], 0);

            t += opt.step_size;
            continue;
        }

        int const offx = grid.stride_x, offy = grid.size[2];
        const int32_t* __restrict__ link_ptr = grid.links + (offx * voxel_l[0] + offy * voxel_l[1] + voxel_l[2]);

        // skip voxel if any of the vertices is turned off
        if ((voxel_l[0] + 1 >= grid.size[0]) || (voxel_l[1] + 1 >= grid.size[1]) || (voxel_l[2] + 1 >= grid.size[2]) \
            || (link_ptr[0] < 0) || (link_ptr[1] < 0) || (link_ptr[offy] < 0) || (link_ptr[offy+1] < 0) \
            || (link_ptr[offx] < 0) || (link_ptr[offx+1] < 0) || (link_ptr[offx+offy] < 0) || (link_ptr[offx+offy+1] < 0)
        ){
            // const float skip = compute_skip_dist(ray,
            //             grid.links, grid.stride_x,
            //             grid.size[2], 0);

            t += opt.step_size;
            continue;
        }

        last_voxel[0] = voxel_l[0];
        last_voxel[1] = voxel_l[1];
        last_voxel[2] = voxel_l[2];

        // find intersections
        double const surface[8] = {
            grid.surface_data[link_ptr[0]],
            grid.surface_data[link_ptr[1]],
            grid.surface_data[link_ptr[offy]],
            grid.surface_data[link_ptr[offy+1]],
            grid.surface_data[link_ptr[offx]],
            grid.surface_data[link_ptr[offx+1]],
            grid.surface_data[link_ptr[offx+offy]],
            grid.surface_data[link_ptr[offx+offy+1]],
        };

        double fs[4];
        surface_to_cubic_equation(surface, ray_origin_d, ray_dir_d, voxel_l, fs);

        // only supports single level set!
        const int level_set_num = 1;
        

        const auto mnmax = thrust::minmax_element(thrust::device, surface, surface+8); // TODO check if it works!
        for (int i=0; i < level_set_num; ++i){
            double const lv_set = grid.level_set_data[i];
            if ((lv_set < *mnmax.first) || (lv_set > *mnmax.second)){
                continue;
            }
            // float const f0_lv = f0 - lv_set;

            // probably better ways to find roots
            // https://stackoverflow.com/questions/4906556/what-is-a-simple-way-to-find-real-roots-of-a-cubic-polynomial
            // https://www.sciencedirect.com/science/article/pii/B9780125434577500097
            // https://stackoverflow.com/questions/13328676/c-solving-cubic-equations


            ////////////// CUBIC ROOT SOLVING //////////////
            double st[3] = {-1, -1, -1}; // sample t

            cubic_equation_solver(
                fs[0] - lv_set, fs[1], fs[2], fs[3],
                1e-8, // float eps
                1e-10, // double eps
                st
                );

            
            ////////////// TRILINEAR INTERPOLATE //////////////
            for (int j=0; j < 3; ++j){
                if (st[j] <= 0){
                    // ignore intersection at negative direction
                    continue;
                }

#pragma unroll 3
                for (int k=0; k < 3; ++k){
                    // assert(!isnan(st[j]));
                    ray.pos[k] = fmaf(static_cast<float>(st[j]), ray.dir[k], ray.origin[k]); // fmaf(x,y,z) = (x*y)+z
                    ray.l[k] = voxel_l[k]; // get l
                    ray.l[k] = min(voxel_l[k], grid.size[k] - 2); // get l
                    ray.pos[k] -= static_cast<float>(ray.l[k]); // get trilinear interpolate distances
                }

                // check if intersection is within grid
                if ((ray.pos[0] < 0) | (ray.pos[0] > 1) | (ray.pos[1] < 0) | (ray.pos[1] > 1) | (ray.pos[2] < 0) | (ray.pos[2] > 1)){
                    continue;
                }


                float alpha = _SIGMOID(trilerp_cuvol_one(
                        grid.links, grid.density_data,
                        grid.stride_x,
                        grid.size[2],
                        1,
                        ray.l, ray.pos,
                        0));

                // if (sigma > opt.sigma_thresh) {
                if (true) {
                    // const float pcnt = ray.world_step * sigma;
                    const float pcnt = -1 * _LOG(1 - alpha);
                    const float weight = _EXP(log_transmit) * (1.f - _EXP(-pcnt));
                    log_transmit -= pcnt; // log_trans = sum(log(1-alpha)) = log(prod(1-alpha))
                    // log_transmit is now T_{i+1}
                    outv += weight * st[j] / opt.step_size * ray.world_step;  
                }
            }

        }

        if (_EXP(log_transmit) < opt.stop_thresh) {
            log_transmit = -1e3f;
            break;
        }

        t += opt.step_size;

    }

    *out = outv;
}

// From Dex-NeRF
__device__ __inline__ void trace_ray_sigma_thresh(
        const PackedSparseGridSpec& __restrict__ grid,
        SingleRaySpec& __restrict__ ray,
        const RenderOptions& __restrict__ opt,
        float sigma_thresh,
        float* __restrict__ out) {
    if (ray.tmin > ray.tmax) {
        *out = 0.f;
        return;
    }

    double const  ray_dir_d[] = {ray.dir[0], ray.dir[1], ray.dir[2]};
    double const  ray_origin_d[] = {ray.origin[0], ray.origin[1], ray.origin[2]};

    float t = ray.tmin;

    int32_t last_voxel[] = {-1,-1,-1};

    while (t <= ray.tmax) {
        int32_t voxel_l[3];
#pragma unroll 3
        for (int j = 0; j < 3; ++j) {
            voxel_l[j] = static_cast<int32_t>(fmaf(t, ray.dir[j], ray.origin[j])); // fmaf(x,y,z) = (x*y)+z
            voxel_l[j] = min(max(voxel_l[j], 0), grid.size[j] - 2);
        }

        if ((voxel_l[0] == last_voxel[0]) && (voxel_l[1] == last_voxel[1]) && (voxel_l[2] == last_voxel[2])){
            // const float skip = compute_skip_dist(ray,
            //             grid.links, grid.stride_x,
            //             grid.size[2], 0);

            t += opt.step_size;
            continue;
        }

        int const offx = grid.stride_x, offy = grid.size[2];
        const int32_t* __restrict__ link_ptr = grid.links + (offx * voxel_l[0] + offy * voxel_l[1] + voxel_l[2]);

        // skip voxel if any of the vertices is turned off
        if ((voxel_l[0] + 1 >= grid.size[0]) || (voxel_l[1] + 1 >= grid.size[1]) || (voxel_l[2] + 1 >= grid.size[2]) \
            || (link_ptr[0] < 0) || (link_ptr[1] < 0) || (link_ptr[offy] < 0) || (link_ptr[offy+1] < 0) \
            || (link_ptr[offx] < 0) || (link_ptr[offx+1] < 0) || (link_ptr[offx+offy] < 0) || (link_ptr[offx+offy+1] < 0)
        ){
            // const float skip = compute_skip_dist(ray,
            //             grid.links, grid.stride_x,
            //             grid.size[2], 0);

            t += opt.step_size;
            continue;
        }

        last_voxel[0] = voxel_l[0];
        last_voxel[1] = voxel_l[1];
        last_voxel[2] = voxel_l[2];

        // find intersections
        double const surface[8] = {
            grid.surface_data[link_ptr[0]],
            grid.surface_data[link_ptr[1]],
            grid.surface_data[link_ptr[offy]],
            grid.surface_data[link_ptr[offy+1]],
            grid.surface_data[link_ptr[offx]],
            grid.surface_data[link_ptr[offx+1]],
            grid.surface_data[link_ptr[offx+offy]],
            grid.surface_data[link_ptr[offx+offy+1]],
        };

        double fs[4];
        surface_to_cubic_equation(surface, ray_origin_d, ray_dir_d, voxel_l, fs);

        // only supports single level set!
        const int level_set_num = 1;
        
        const auto mnmax = thrust::minmax_element(thrust::device, surface, surface+8); // TODO check if it works!
        for (int i=0; i < level_set_num; ++i){
            double const lv_set = grid.level_set_data[i];
            if ((lv_set < *mnmax.first) || (lv_set > *mnmax.second)){
                continue;
            }
            ////////////// CUBIC ROOT SOLVING //////////////
            double st[3] = {-1, -1, -1}; // sample t

            cubic_equation_solver(
                fs[0] - lv_set, fs[1], fs[2], fs[3],
                1e-8, // float eps
                1e-10, // double eps
                st
                );

            
            ////////////// TRILINEAR INTERPOLATE //////////////
            for (int j=0; j < 3; ++j){
                if (st[j] <= 0){
                    // ignore intersection at negative direction
                    continue;
                }

#pragma unroll 3
                for (int k=0; k < 3; ++k){
                    // assert(!isnan(st[j]));
                    ray.pos[k] = fmaf(static_cast<float>(st[j]), ray.dir[k], ray.origin[k]); // fmaf(x,y,z) = (x*y)+z
                    ray.l[k] = voxel_l[k]; // get l
                    ray.l[k] = min(voxel_l[k], grid.size[k] - 2); // get l
                    ray.pos[k] -= static_cast<float>(ray.l[k]); // get trilinear interpolate distances
                }

                // check if intersection is within grid
                if ((ray.pos[0] < 0) | (ray.pos[0] > 1) | (ray.pos[1] < 0) | (ray.pos[1] > 1) | (ray.pos[2] < 0) | (ray.pos[2] > 1)){
                    continue;
                }


                float const alpha = _SIGMOID(trilerp_cuvol_one(
                        grid.links, grid.density_data,
                        grid.stride_x,
                        grid.size[2],
                        1,
                        ray.l, ray.pos,
                        0));

                if (alpha > sigma_thresh) {
                    *out = (st[j] / opt.step_size) * ray.world_step;
                    return;
                }
            }

        }
        t += opt.step_size;
    }
    *out = 0.f;
}


__device__ __inline__ void trace_ray_extract_pt(
        const PackedSparseGridSpec& __restrict__ grid,
        SingleRaySpec& __restrict__ ray,
        const RenderOptions& __restrict__ opt,
        float alpha_thresh,
        float* __restrict__ out) {

    assert(false);

    if (ray.tmin > ray.tmax) {
        *out = 0.f;
        return;
    }

    double const  ray_dir_d[] = {ray.dir[0], ray.dir[1], ray.dir[2]};
    double const  ray_origin_d[] = {ray.origin[0], ray.origin[1], ray.origin[2]};

    float t = ray.tmin;

    int32_t last_voxel[] = {-1,-1,-1};

    while (t <= ray.tmax) {
        int32_t voxel_l[3];
#pragma unroll 3
        for (int j = 0; j < 3; ++j) {
            voxel_l[j] = static_cast<int32_t>(fmaf(t, ray.dir[j], ray.origin[j])); // fmaf(x,y,z) = (x*y)+z
            voxel_l[j] = min(max(voxel_l[j], 0), grid.size[j] - 2);
        }

        if ((voxel_l[0] == last_voxel[0]) && (voxel_l[1] == last_voxel[1]) && (voxel_l[2] == last_voxel[2])){
            // const float skip = compute_skip_dist(ray,
            //             grid.links, grid.stride_x,
            //             grid.size[2], 0);

            t += opt.step_size;
            continue;
        }

        int const offx = grid.stride_x, offy = grid.size[2];
        const int32_t* __restrict__ link_ptr = grid.links + (offx * voxel_l[0] + offy * voxel_l[1] + voxel_l[2]);

        // skip voxel if any of the vertices is turned off
        if ((voxel_l[0] + 1 >= grid.size[0]) || (voxel_l[1] + 1 >= grid.size[1]) || (voxel_l[2] + 1 >= grid.size[2]) \
            || (link_ptr[0] < 0) || (link_ptr[1] < 0) || (link_ptr[offy] < 0) || (link_ptr[offy+1] < 0) \
            || (link_ptr[offx] < 0) || (link_ptr[offx+1] < 0) || (link_ptr[offx+offy] < 0) || (link_ptr[offx+offy+1] < 0)
        ){
            // const float skip = compute_skip_dist(ray,
            //             grid.links, grid.stride_x,
            //             grid.size[2], 0);

            t += opt.step_size;
            continue;
        }

        last_voxel[0] = voxel_l[0];
        last_voxel[1] = voxel_l[1];
        last_voxel[2] = voxel_l[2];

        // find intersections
        double const surface[8] = {
            grid.surface_data[link_ptr[0]],
            grid.surface_data[link_ptr[1]],
            grid.surface_data[link_ptr[offy]],
            grid.surface_data[link_ptr[offy+1]],
            grid.surface_data[link_ptr[offx]],
            grid.surface_data[link_ptr[offx+1]],
            grid.surface_data[link_ptr[offx+offy]],
            grid.surface_data[link_ptr[offx+offy+1]],
        };

        double fs[4];
        surface_to_cubic_equation(surface, ray_origin_d, ray_dir_d, voxel_l, fs);

        // only supports single level set!
        const int level_set_num = 1;
        
        const auto mnmax = thrust::minmax_element(thrust::device, surface, surface+8); // TODO check if it works!
        for (int i=0; i < level_set_num; ++i){
            double const lv_set = grid.level_set_data[i];
            if ((lv_set < *mnmax.first) || (lv_set > *mnmax.second)){
                continue;
            }
            ////////////// CUBIC ROOT SOLVING //////////////
            double st[3] = {-1, -1, -1}; // sample t

            cubic_equation_solver(
                fs[0] - lv_set, fs[1], fs[2], fs[3],
                1e-8, // float eps
                1e-10, // double eps
                st
                );

            
            ////////////// TRILINEAR INTERPOLATE //////////////
            for (int j=0; j < 3; ++j){
                if (st[j] <= 0){
                    // ignore intersection at negative direction
                    continue;
                }

#pragma unroll 3
                for (int k=0; k < 3; ++k){
                    // assert(!isnan(st[j]));
                    ray.pos[k] = fmaf(static_cast<float>(st[j]), ray.dir[k], ray.origin[k]); // fmaf(x,y,z) = (x*y)+z
                    ray.l[k] = voxel_l[k]; // get l
                    ray.l[k] = min(voxel_l[k], grid.size[k] - 2); // get l
                    ray.pos[k] -= static_cast<float>(ray.l[k]); // get trilinear interpolate distances
                }

                // check if intersection is within grid
                if ((ray.pos[0] < 0) | (ray.pos[0] > 1) | (ray.pos[1] < 0) | (ray.pos[1] > 1) | (ray.pos[2] < 0) | (ray.pos[2] > 1)){
                    continue;
                }


                float const alpha = _SIGMOID(trilerp_cuvol_one(
                        grid.links, grid.density_data,
                        grid.stride_x,
                        grid.size[2],
                        1,
                        ray.l, ray.pos,
                        0));

                if (alpha > alpha_thresh) {
                    *out = (st[j] / opt.step_size) * ray.world_step;
                    return;
                }
            }

        }
        t += opt.step_size;
    }
    *out = 0.f;
}




__device__ __inline__ void trace_ray_surf_trav_backward(
        const PackedSparseGridSpec& __restrict__ grid,
        const float* __restrict__ grad_output, // array[3], MSE gradient wrt rgb channel
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
    const uint32_t lane_colorgrp_id = lane_id % grid.basis_dim; // basis id in each channel
    const uint32_t lane_colorgrp = lane_id / grid.basis_dim; // rgb channel id
    const uint32_t leader_mask = 1U | (1U << grid.basis_dim) | (1U << (2 * grid.basis_dim)); // mask for RGB channels of same basis
    double const  ray_dir_d[] = {ray.dir[0], ray.dir[1], ray.dir[2]};
    double const  ray_origin_d[] = {ray.origin[0], ray.origin[1], ray.origin[2]};

    float accum = fmaf(color_cache[0], grad_output[0],
                      fmaf(color_cache[1], grad_output[1],
                           color_cache[2] * grad_output[2])); // sum(d_mse/d_pred_rgb * pred_rgb)

    // if (beta_loss > 0.f) {
    //     const float transmit_in = _EXP(log_transmit_in);
    //     beta_loss *= (1 - transmit_in / (1 - transmit_in + 1e-3)); // d beta_loss / d log_transmit_in
    //     accum += beta_loss;
    //     // Interesting how this loss turns out, kinda nice?
    // }

    if (ray.tmin > ray.tmax) {
        if (accum_out != nullptr) { *accum_out = accum; }
        if (log_transmit_out != nullptr) { *log_transmit_out = 0.f; }
        // printf("accum_end_fg_fast=%f\n", accum);
        return;
    }
    float t = ray.tmin;

    const float gout = grad_output[lane_colorgrp]; // get gradient of corresponding RGB channel

    float log_transmit = 0.f;

    // remat samples. Needed because individual rgb/sigma are not stored during forward pass
    int32_t last_voxel[] = {-1,-1,-1};

    while (t <= ray.tmax) {
        int32_t voxel_l[3];
#pragma unroll 3
        for (int j = 0; j < 3; ++j) {
            voxel_l[j] = static_cast<int32_t>(fmaf(t, ray.dir[j], ray.origin[j])); // fmaf(x,y,z) = (x*y)+z
            voxel_l[j] = min(max(voxel_l[j], 0), grid.size[j] - 2);
        }

        if ((voxel_l[0] == last_voxel[0]) && (voxel_l[1] == last_voxel[1]) && (voxel_l[2] == last_voxel[2])){
            // const float skip = compute_skip_dist(ray,
            //             grid.links, grid.stride_x,
            //             grid.size[2], 0);

                t += opt.step_size;
                continue;
        }

        int const offx = grid.stride_x, offy = grid.size[2];
        const int32_t* __restrict__ link_ptr = grid.links + (offx * voxel_l[0] + offy * voxel_l[1] + voxel_l[2]);

        // skip voxel if any of the vertices is turned off
        if ((voxel_l[0] + 1 >= grid.size[0]) || (voxel_l[1] + 1 >= grid.size[1]) || (voxel_l[2] + 1 >= grid.size[2]) \
            || (link_ptr[0] < 0) || (link_ptr[1] < 0) || (link_ptr[offy] < 0) || (link_ptr[offy+1] < 0) \
            || (link_ptr[offx] < 0) || (link_ptr[offx+1] < 0) || (link_ptr[offx+offy] < 0) || (link_ptr[offx+offy+1] < 0)
        ){
            // const float skip = compute_skip_dist(ray,
            //             grid.links, grid.stride_x,
            //             grid.size[2], 0);


            t += opt.step_size;
            continue;
        }
        
        last_voxel[0] = voxel_l[0];
        last_voxel[1] = voxel_l[1];
        last_voxel[2] = voxel_l[2];

        // check minimal of alpha raw
        if ((grid.density_data[link_ptr[0]] < opt.sigma_thresh) && \
            (grid.density_data[link_ptr[1]] < opt.sigma_thresh) && \
            (grid.density_data[link_ptr[offy]] < opt.sigma_thresh) && \
            (grid.density_data[link_ptr[offy+1]] < opt.sigma_thresh) && \
            (grid.density_data[link_ptr[offx]] < opt.sigma_thresh) && \
            (grid.density_data[link_ptr[offx+1]] < opt.sigma_thresh) && \
            (grid.density_data[link_ptr[offx+offy]] < opt.sigma_thresh) && \
            (grid.density_data[link_ptr[offx+offy+1]] < opt.sigma_thresh)){
                // const float skip = compute_skip_dist(ray,
                //             grid.links, grid.stride_x,
                //             grid.size[2], 0);

                t += opt.step_size;
                continue;
            }

        // find intersections
        double const surface[8] = {
            grid.surface_data[link_ptr[0]],
            grid.surface_data[link_ptr[1]],
            grid.surface_data[link_ptr[offy]],
            grid.surface_data[link_ptr[offy+1]],
            grid.surface_data[link_ptr[offx]],
            grid.surface_data[link_ptr[offx+1]],
            grid.surface_data[link_ptr[offx+offy]],
            grid.surface_data[link_ptr[offx+offy+1]],
        };


        double fs[4];
        surface_to_cubic_equation(surface, ray_origin_d, ray_dir_d, voxel_l, fs);

        // only supports single level set!
        const int level_set_num = 1;

        
        const auto mnmax = thrust::minmax_element(thrust::device, surface, surface+8); 
        for (int i=0; i < level_set_num; ++i){
            double const lv_set = grid.level_set_data[i];
            if ((lv_set < *mnmax.first) || (lv_set > *mnmax.second)){
                continue;
            }

            fs[0] -= lv_set;

            ////////////// CUBIC ROOT SOLVING //////////////
            double st[3] = {-1, -1, -1}; // sample t

            enum BasisType const cubic_root_type = cubic_equation_solver(
                fs[0], fs[1], fs[2], fs[3],
                1e-8, // float eps
                1e-10, // double eps
                st
                );
            
            // sort intersections by depth
            // int st_ids[3] = {0,1,2};
            // sort index instead to keep track of root computation
            // thrust::sort(thrust::device, st, st + 3);
            // thrust::sort(thrust::device, st_ids, st_ids + 3, [&st](int i,int j){return st[i]<st[j];} );

            bool has_sample = false;

            ////////////// TRILINEAR INTERPOLATE //////////////
            for (int st_id=0; st_id < 3; ++st_id){
                // int const st_id = st_ids[j];
                if (st[st_id] <= 0){
                    // ignore intersection at negative direction
                    continue;
                }

#pragma unroll 3
                for (int k=0; k < 3; ++k){
                    // assert(!isnan(st[st_id]));
                    ray.pos[k] = fmaf(st[st_id], ray.dir[k], ray.origin[k]); // fmaf(x,y,z) = (x*y)+z
                    ray.l[k] = voxel_l[k]; // get l
                    ray.l[k] = min(voxel_l[k], grid.size[k] - 2); // get l
                    ray.pos[k] -= static_cast<float>(ray.l[k]); // get trilinear interpolate distances
                }

                // check if intersection is within grid
                if ((ray.pos[0] < 0) | (ray.pos[0] > 1) | (ray.pos[1] < 0) | (ray.pos[1] > 1) | (ray.pos[2] < 0) | (ray.pos[2] > 1)){
                    continue;
                }

                // int32_t volatile ray_l[] = {ray.l[0],ray.l[1],ray.l[2]};
                // float volatile ray_origin[] = {ray.origin[0],ray.origin[1],ray.origin[2]};
                // float volatile ray_dir[] = {ray.dir[0],ray.dir[1],ray.dir[2]};
                // float volatile ray_pos[] = {ray.pos[0],ray.pos[1],ray.pos[2]};

                has_sample = true;
                float const  raw_alpha = trilerp_cuvol_one(
                        grid.links, grid.density_data,
                        grid.stride_x,
                        grid.size[2],
                        1,
                        ray.l, ray.pos,
                        0);

                float const  alpha = _SIGMOID(raw_alpha);

                if (raw_alpha > opt.sigma_thresh) {
                    float lane_color = trilerp_cuvol_one(
                                    grid.links,
                                    grid.sh_data,
                                    grid.stride_x,
                                    grid.size[2],
                                    grid.sh_data_dim,
                                    ray.l, ray.pos, lane_id);

                    float weighted_lane_color = lane_color * sphfunc_val[lane_colorgrp_id];

                    const float  pcnt = -1 * _LOG(1 - alpha);
                    const float  weight = _EXP(log_transmit) * (1.f - _EXP(-pcnt));
                    

                    const float lane_color_total = WarpReducef(temp_storage).HeadSegmentedSum(
                                                weighted_lane_color, lane_colorgrp_id == 0) + 0.5f; // TODO: why +0.5f? -- because outv also has +0.5 before clamping
                    float total_color = fmaxf(lane_color_total, 0.f); // Clamp to [+0, infty), ci -- one channel of radiance of the sample
                    float color_in_01 = total_color == lane_color_total; // 1 if color >= 0.
                    // substract for background color? -- seems to have already been done somewhere
                    total_color *= gout; // d_mse/d_pred_c * ci

                    // https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#warp-description 
                    float total_color_c1 = __shfl_sync(leader_mask, total_color, grid.basis_dim); 
                    // taking d_mse/d_pred_c * ci of each RGB channels
                    total_color += __shfl_sync(leader_mask, total_color, 2 * grid.basis_dim); 
                    total_color += total_color_c1;

                    // get color_in_01 from 'parent' lane in the color group where the channel color is computed
                    color_in_01 = __shfl_sync((1U << grid.sh_data_dim) - 1, color_in_01, lane_colorgrp * grid.basis_dim); 
                    // d_mse/d_ci in the final computed color is within clamp range, compute proper gradient 
                    const float  grad_common = weight * color_in_01 * gout; 
                    // gradient wrt sh coefficient (d_mse/d_sh)
                    const float  curr_grad_color = sphfunc_val[lane_colorgrp_id] * grad_common; 

                    // if (grid.basis_type != BASIS_TYPE_SH) {
                    //     float curr_grad_sphfunc = lane_color * grad_common;
                    //     const float curr_grad_up2 = __shfl_down_sync((1U << grid.sh_data_dim) - 1,
                    //             curr_grad_sphfunc, 2 * grid.basis_dim);
                    //     curr_grad_sphfunc += __shfl_down_sync((1U << grid.sh_data_dim) - 1,
                    //             curr_grad_sphfunc, grid.basis_dim);
                    //     curr_grad_sphfunc += curr_grad_up2;
                    //     if (lane_id < grid.basis_dim) {
                    //         grad_sphfunc_val[lane_id] += curr_grad_sphfunc;
                    //     }
                    // }

                    // accum is now d_mse/d_pred_c * sum(wi * ci)[i=current+1~N]
                    accum -= weight * total_color;
                    // compute d_mse/d_alpha_i
                    float  curr_grad_alpha = accum / min(alpha-1.f, -1e-9f) + total_color * _EXP(log_transmit); 
                    log_transmit -= pcnt; // update log_transmit to log(T_{i+1})
                    // if (sparsity_loss > 0.f) {
                    //     // Cauchy version (from SNeRG)
                    //     // TODO: check if expected!
                    //     curr_grad_alpha += sparsity_loss * (4.f * alpha / (1.f + 2.f * (alpha * alpha)));

                    //     // Alphs version (from PlenOctrees)
                    //     // curr_grad_alpha += sparsity_loss * _EXP(-pcnt) * ray.world_step;
                    // }

                    // if (lane_id == 0){
                    //     // printf("weight: [%f]\n", weight);
                    //     // printf("grad_common: [%f]\n", grad_common);
                    //     // printf("reweight: [%f]\n", reweight);
                    //     printf("###########\n");
                    //     printf("taking real sample for: [%d, %d, %d]\n", voxel_l[0], voxel_l[1], voxel_l[2]);
                    //     printf("alpha: [%f]\n", alpha);
                    //     printf("curr_grad_alpha: [%f]\n", curr_grad_alpha);
                    //     printf("log_transmit: [%f]\n", log_transmit);
                    //     printf("accum: [%f]\n", accum);
                    //     printf("total_color: [%f]\n", total_color);
                    // }


                    trilerp_backward_cuvol_one(grid.links, grads.grad_sh_out,
                            grid.stride_x,
                            grid.size[2],
                            grid.sh_data_dim,
                            ray.l, ray.pos,
                            curr_grad_color, lane_id);

                    // compute gradient to surface via sh
                    float grad_xyz [3] = {0,0,0}; // d_mse/d_pos[x,y,z]
                    trilerp_backward_one_pos(
                                grid.links,
                                grid.sh_data,
                                grid.stride_x,
                                grid.size[2],
                                grid.sh_data_dim,
                                ray.l, ray.pos, lane_id,
                                curr_grad_color, grad_xyz);

                    // use WarpReducef to sum up gradient to surface via different sh basis
                    grad_xyz[0] = WarpReducef(temp_storage).Sum(grad_xyz[0]);
                    grad_xyz[1] = WarpReducef(temp_storage).Sum(grad_xyz[1]);
                    grad_xyz[2] = WarpReducef(temp_storage).Sum(grad_xyz[2]);

                    if (lane_id == 0) {
                        // compute gradient for sigmoid
                        float const  curr_grad_raw_alpha = curr_grad_alpha * _D_SIGMOID(raw_alpha);
                        ASSERT_NUM(curr_grad_raw_alpha);
                        trilerp_backward_cuvol_one_density(
                                grid.links,
                                grads.grad_density_out,
                                grads.mask_out,
                                grid.stride_x,
                                grid.size[2],
                                ray.l, ray.pos, curr_grad_raw_alpha);

                        // compute gradient to surface via density
                        trilerp_backward_one_pos(
                                    grid.links,
                                    grid.density_data,
                                    grid.stride_x,
                                    grid.size[2],
                                    1,
                                    ray.l, ray.pos, 0,
                                    curr_grad_raw_alpha, grad_xyz);

                        // grad_xyz is now d_mse/d_xyz
                        float const grad_st = grad_xyz[0]*ray.dir[0] + grad_xyz[1]*ray.dir[1] + grad_xyz[2]*ray.dir[2];
                        ASSERT_NUM(grad_st);
                        // grad_st is now d_mse/d_t

                        float grad_fs[4] = {grad_st, grad_st, grad_st, grad_st};
                        calc_cubic_root_grad(cubic_root_type, st_id, fs, grad_fs);
                        // grad_fs is now d_mse/d_f0123

                        float grad_surface[8];
                        calc_surface_grad(ray.origin, ray.dir, ray.l, grad_fs, grad_surface);

                        assign_surface_grad(
                            grid.links,
                            grads.grad_surface_out,
                            grads.mask_out,
                            grid.stride_x,
                            grid.size[2],
                            ray.l,
                            grad_surface
                        );
                    }


                }


            }
        
            if ((!has_sample) && (opt.surf_fake_sample)){
                // there is no intersection between ray and surface
                // take fake sample if allowed            

                // if (lane_id == 0){
                //     printf("==========\n");
                //     printf("taking fake sample for: [%d, %d, %d]\n", voxel_l[0], voxel_l[1], voxel_l[2]);
                // }    

                // first find middle point of the ray in the voxel
                int32_t const close_plane[] = {
                    ray.dir[0] > 0.f ? voxel_l[0] : voxel_l[0]+1,
                    ray.dir[1] > 0.f ? voxel_l[1] : voxel_l[1]+1,
                    ray.dir[2] > 0.f ? voxel_l[2] : voxel_l[2]+1,
                };
                int32_t const far_plane[] = {
                    ray.dir[0] > 0.f ? voxel_l[0]+1 : voxel_l[0],
                    ray.dir[1] > 0.f ? voxel_l[1]+1 : voxel_l[1],
                    ray.dir[2] > 0.f ? voxel_l[2]+1 : voxel_l[2],
                };

                float const t_close = max(
                    max((static_cast<float>(close_plane[0])-ray.origin[0])/ray.dir[0], (static_cast<float>(close_plane[1])-ray.origin[1])/ray.dir[1]),
                    (static_cast<float>(close_plane[2])-ray.origin[2])/ray.dir[2]);
                float const t_far = min(
                    min((static_cast<float>(far_plane[0])-ray.origin[0])/ray.dir[0], (static_cast<float>(far_plane[1])-ray.origin[1])/ray.dir[1]),
                    (static_cast<float>(far_plane[2])-ray.origin[2])/ray.dir[2]);



                if ((t_far - t_close) > opt.surf_fake_sample_min_vox_len){
#pragma unroll 3
                    for (int k=0; k < 3; ++k){
                        // assert(!isnan(st[j]));
                        ray.pos[k] = fmaf((t_far + t_close) / 2, ray.dir[k], ray.origin[k]); // fmaf(x,y,z) = (x*y)+z
                        ray.l[k] = min(voxel_l[k], grid.size[k] - 2); // get l
                        ray.pos[k] -= static_cast<float>(ray.l[k]); // get trilinear interpolate distances

                        // if ((!(ray.pos[k] >= 0.f)) || (!(ray.pos[k] <= 1.f)) ){
                        //     printf("t_far: %f\n", t_far);
                        //     printf("t_close: %f\n", t_close);
                        //     printf("ray_l_k: %d\n", ray.l[k]);
                        //     printf("ray_pos_k: %f\n", ray.pos[k]);
                        // }
                        
                        // assert(ray.pos[k] <= 1.f);
                        // assert(ray.pos[k] >= 0.f);
                    }

                    float const raw_alpha = trilerp_cuvol_one(
                            grid.links, grid.density_data,
                            grid.stride_x,
                            grid.size[2],
                            1,
                            ray.l, ray.pos,
                            0);

                    if (raw_alpha > opt.sigma_thresh) {
                        float const  alpha = _SIGMOID(raw_alpha);

                        // use distance to surface to re-weight alpha
                        // https://math.stackexchange.com/questions/1815397/distance-between-point-and-parametric-line
                        // we approximate the distance by normalizing the surface scalar values
                        // so the distance no longer relates to the scale of surface

                        float const surf_norm = sqrtf(
                            max(1e-9f, 
                            _SQR(surface[0]) + _SQR(surface[1]) + _SQR(surface[2]) + _SQR(surface[3]) + _SQR(surface[4]) + _SQR(surface[5]) + _SQR(surface[6]) + _SQR(surface[7])
                            )
                        );

                        // tri-lerp to get distance

                        #define _norm_surf(x) (static_cast<float>(surface[x]) / surf_norm)

                        const float ix0y0 = lerp(_norm_surf(0), _norm_surf(1), ray.pos[2]);
                        const float ix0y1 = lerp(_norm_surf(2), _norm_surf(3), ray.pos[2]);
                        const float ix0 = lerp(ix0y0, ix0y1, ray.pos[1]);
                        const float ix1y0 = lerp(_norm_surf(4), _norm_surf(5), ray.pos[2]);
                        const float ix1y1 = lerp(_norm_surf(6),
                                                _norm_surf(7), ray.pos[2]);
                        const float ix1 = lerp(ix1y0, ix1y1, ray.pos[1]);
                        const float  fake_sample_dist = lerp(ix0, ix1, ray.pos[0]);

                        #undef _norm_surf

                        
                        // re-weight alpha using a simple gaussian
                        float const  reweight = _EXP(-.5 * _SQR(fake_sample_dist/grid.fake_sample_std));
                        float const  rw_alpha = alpha * reweight;


                        float lane_color = trilerp_cuvol_one(
                                        grid.links,
                                        grid.sh_data,
                                        grid.stride_x,
                                        grid.size[2],
                                        grid.sh_data_dim,
                                        ray.l, ray.pos, lane_id);
                        // lane_color *= sphfunc_val[lane_colorgrp_id]; // bank conflict


                        // backward gradient computation
                        float weighted_lane_color = lane_color * sphfunc_val[lane_colorgrp_id];
                        const float  pcnt = -1 * _LOG(1 - rw_alpha);
                        const float  weight = _EXP(log_transmit) * (1.f - _EXP(-pcnt));
                        

                        const float lane_color_total = WarpReducef(temp_storage).HeadSegmentedSum(
                                                    weighted_lane_color, lane_colorgrp_id == 0) + 0.5f; // this is wrong!

                        // if (lane_colorgrp_id == 0){
                        //     printf("lane_color_total (%d): %f\n", lane_id, lane_color_total);
                        // }

                        float total_color_fs = fmaxf(lane_color_total, 0.f); // Clamp to [+0, infty), ci -- one channel of radiance of the sample
                        float color_in_01 = total_color_fs == lane_color_total; // 1 if color >= 0.
                        // substract for background color? -- seems to have already been done somewhere
                        total_color_fs *= gout; // d_mse/d_pred_c * ci

                        // https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#warp-description 
                        float total_color_fs_c1 = __shfl_sync(leader_mask, total_color_fs, grid.basis_dim); 
                        // taking d_mse/d_pred_c * ci of each RGB channels
                        total_color_fs += __shfl_sync(leader_mask, total_color_fs, 2 * grid.basis_dim); 
                        total_color_fs += total_color_fs_c1;

                        // get color_in_01 from 'parent' lane in the color group where the channel color is computed
                        color_in_01 = __shfl_sync((1U << grid.sh_data_dim) - 1, color_in_01, lane_colorgrp * grid.basis_dim); 
                        // d_mse/d_ci in the final computed color is within clamp range, compute proper gradient 
                        const float  grad_common = weight * color_in_01 * gout; 
                        // gradient wrt sh coefficient (d_mse/d_sh)
                        const float  curr_grad_color = sphfunc_val[lane_colorgrp_id] * grad_common; 

                        // if (grid.basis_type != BASIS_TYPE_SH) {
                        //     float curr_grad_sphfunc = lane_color * grad_common;
                        //     const float curr_grad_up2 = __shfl_down_sync((1U << grid.sh_data_dim) - 1,
                        //             curr_grad_sphfunc, 2 * grid.basis_dim);
                        //     curr_grad_sphfunc += __shfl_down_sync((1U << grid.sh_data_dim) - 1,
                        //             curr_grad_sphfunc, grid.basis_dim);
                        //     curr_grad_sphfunc += curr_grad_up2;
                        //     if (lane_id < grid.basis_dim) {
                        //         grad_sphfunc_val[lane_id] += curr_grad_sphfunc;
                        //     }
                        // }

                        // accum is now d_mse/d_pred_c * sum(wi * ci)[i=current+1~N]
                        accum -= weight * total_color_fs;
                        // compute d_mse/d_rwalpha_i (reweighted alpha)
                        float  curr_grad_rwalpha = accum / min(rw_alpha-1.f, -1e-9f) + total_color_fs * _EXP(log_transmit); 
                        log_transmit -= pcnt; // update log_transmit to log(T_{i+1})
                        // if (sparsity_loss > 0.f) {
                        //     // Cauchy version (from SNeRG)
                        //     // TODO: check if expected!
                        //     curr_grad_rwalpha += sparsity_loss * (4.f * alpha / (1.f + 2.f * (alpha * alpha)));

                        //     // Alphs version (from PlenOctrees)
                        //     // curr_grad_alpha += sparsity_loss * _EXP(-pcnt) * ray.world_step;
                        // }

                        // curr_grad_alpha is now d_mse/d_alpha_i
                        float const curr_grad_alpha = curr_grad_rwalpha * reweight;

                        // if (lane_id == 0){
                        //     printf("weight: [%f]\n", weight);
                        //     // printf("grad_common: [%f]\n", grad_common);
                        //     // printf("reweight: [%f]\n", reweight);
                        //     printf("alpha: [%f]\n", alpha);
                        //     printf("rw_alpha: [%f]\n", rw_alpha);
                        //     printf("curr_grad_rwalpha: [%f]\n", curr_grad_rwalpha);
                        //     printf("curr_grad_alpha: [%f]\n", curr_grad_alpha);
                        //     // printf("log_transmit: [%f]\n", log_transmit);
                        //     // printf("accum: [%f]\n", accum);
                        //     // printf("total_color_fs: [%f]\n", total_color_fs);
                        // }

                        trilerp_backward_cuvol_one(grid.links, grads.grad_sh_out,
                                grid.stride_x,
                                grid.size[2],
                                grid.sh_data_dim,
                                ray.l, ray.pos,
                                curr_grad_color, lane_id);


                        if (lane_id == 0) {
                            // compute gradient for sigmoid
                            float const  curr_grad_raw_alpha = curr_grad_alpha * _D_SIGMOID(raw_alpha);
                            ASSERT_NUM(curr_grad_raw_alpha);
                            trilerp_backward_cuvol_one_density(
                                    grid.links,
                                    grads.grad_density_out,
                                    grads.mask_out,
                                    grid.stride_x,
                                    grid.size[2],
                                    ray.l, ray.pos, curr_grad_raw_alpha);

                            
                            // gradient to surface via fake sample
                            // via curr_grad_rwalpha

                            float const grad_fake_dist = curr_grad_rwalpha * (-alpha) * fake_sample_dist * reweight / _SQR(grid.fake_sample_std);
                            
                            ASSERT_NUM(grad_fake_dist);

                            
                            float grad_ns[8]; // grad of normalized surface values

                            const float ay = 1.f - ray.pos[1], az = 1.f - ray.pos[2];
                            float xo = (1.0f - ray.pos[0]) * grad_fake_dist;

                            // printf("pos: [%f, %f, %f]\n", ray.pos[0], ray.pos[1], ray.pos[2]);

                            grad_ns[0] = ay * az * xo;
                            grad_ns[1] = ay * ray.pos[2] * xo;
                            grad_ns[2] = ray.pos[1] * az * xo;
                            grad_ns[3] = ray.pos[1] * ray.pos[2] * xo;

                            xo = ray.pos[0] * grad_fake_dist;
                            grad_ns[4] = ay * az * xo;
                            grad_ns[5] = ay * ray.pos[2] * xo;
                            grad_ns[6] = ray.pos[1] * az * xo;
                            grad_ns[7] = ray.pos[1] * ray.pos[2] * xo;


                            // for (int ks = 0; ks < 8; ++ks){
                            //     ASSERT_NUM(grad_ns[ks]);
                            // }



                            float grad_surface[8] = {0,0,0,0,0,0,0,0};
#pragma unroll 8
                            for (int ks = 0; ks < 8; ++ks){
#pragma unroll 8
                                for (int kn = 0; kn < 8; ++kn){
                                    if (ks == kn){
                                        grad_surface[ks] += grad_ns[kn] * (-_SQR(surface[ks]) / _CUBIC(surf_norm) + 1.f/surf_norm);
                                    } else {
                                        grad_surface[ks] +=  grad_ns[kn] * (-surface[ks]*surface[kn] / _CUBIC(surf_norm));
                                    }
                                }
                            }


                            // for (int ks = 0; ks < 8; ++ks){
                            //     ASSERT_NUM(grad_surface[ks]);
                            // }

                            assign_surface_grad(
                                grid.links,
                                grads.grad_surface_out,
                                grads.mask_out,
                                grid.stride_x,
                                grid.size[2],
                                ray.l,
                                grad_surface
                            );


                        }
                    }
                }
            }
            
        }

        if (_EXP(log_transmit) < opt.stop_thresh) {
            break;
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
    float t, invr_last = 1.f / inner_radius;
    const int n_steps = int(grid.background_nlayers / opt.step_size) + 2;

    // csi.intersect(inner_radius, &t_last);

    float outv[3] = {0.f, 0.f, 0.f};
    for (int i = 0; i < n_steps; ++i) {
        // Between 1 and infty
        float r = n_steps / (n_steps - i - 0.5);
        if (r < inner_radius || !csi.intersect(r, &t)) continue;

#pragma unroll 3
        for (int j = 0; j < 3; ++j) {
            ray.pos[j] = fmaf(t, ray.dir[j], ray.origin[j]);
        }
        const float invr_mid = _rnorm(ray.pos);
#pragma unroll 3
        for (int j = 0; j < 3; ++j) {
            ray.pos[j] *= invr_mid;
        }
        // NOTE: reusing ray.pos (ok if you check _unitvec2equirect)
        _unitvec2equirect(ray.pos, grid.background_reso, ray.pos);
        ray.pos[2] = fminf(fmaxf((1.f - invr_mid) * grid.background_nlayers - 0.5f, 0.f),
                       grid.background_nlayers - 1);
#pragma unroll 3
        for (int j = 0; j < 3; ++j) {
            ray.l[j] = (int) ray.pos[j];
        }
        ray.l[0] = min(ray.l[0], grid.background_reso * 2 - 1);
        ray.l[1] = min(ray.l[1], grid.background_reso - 1);
        ray.l[2] = min(ray.l[2], grid.background_nlayers - 2);
#pragma unroll 3
        for (int j = 0; j < 3; ++j) {
            ray.pos[j] -= ray.l[j];
        }

        float sigma = trilerp_bg_one(
                grid.background_links,
                grid.background_data,
                grid.background_reso,
                grid.background_nlayers,
                4,
                ray.l,
                ray.pos,
                3);

        // if (i == n_steps - 1) {
        //     ray.world_step = 1e9;
        // }
        // if (opt.randomize && opt.random_sigma_std_background > 0.0)
        //     sigma += ray.rng.randn() * opt.random_sigma_std_background;
        if (sigma > 0.f) {
            const float pcnt = (invr_last - invr_mid) * ray.world_step * sigma;
            const float weight = _EXP(log_transmit) * (1.f - _EXP(-pcnt));
            log_transmit -= pcnt;
#pragma unroll 3
            for (int i = 0; i < 3; ++i) {
                // Not efficient
                const float color = trilerp_bg_one(
                        grid.background_links,
                        grid.background_data,
                        grid.background_reso,
                        grid.background_nlayers,
                        4,
                        ray.l,
                        ray.pos,
                        i) * C0;  // Scale by SH DC factor to help normalize lrs
                outv[i] += weight * fmaxf(color + 0.5f, 0.f);  // Clamp to [+0, infty)
            }
            if (_EXP(log_transmit) < opt.stop_thresh) {
                break;
            }
        }
        invr_last = invr_mid;
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
            float sparsity_loss,
            PackedGridOutputGrads& __restrict__ grads
        ) {
    // printf("accum_init=%f\n", accum);
    // printf("log_transmit_init=%f\n", log_transmit);
    ConcentricSpheresIntersector csi(ray.origin, ray.dir);

    const int n_steps = int(grid.background_nlayers / opt.step_size) + 2;

    const float inner_radius = fmaxf(_dist_ray_to_origin(ray.origin, ray.dir) + 1e-3f, 1.f);
    float t, invr_last = 1.f / inner_radius;
    // csi.intersect(inner_radius, &t_last);
    for (int i = 0; i < n_steps; ++i) {
        float r = n_steps / (n_steps - i - 0.5);

        if (r < inner_radius || !csi.intersect(r, &t)) continue;

#pragma unroll 3
        for (int j = 0; j < 3; ++j) {
            ray.pos[j] = fmaf(t, ray.dir[j], ray.origin[j]);
        }

        const float invr_mid = _rnorm(ray.pos);
#pragma unroll 3
        for (int j = 0; j < 3; ++j) {
            ray.pos[j] *= invr_mid;
        }
        // NOTE: reusing ray.pos (ok if you check _unitvec2equirect)
        _unitvec2equirect(ray.pos, grid.background_reso, ray.pos);
        ray.pos[2] = fminf(fmaxf((1.f - invr_mid) * grid.background_nlayers - 0.5f, 0.f),
                       grid.background_nlayers - 1);
#pragma unroll 3
        for (int j = 0; j < 3; ++j) {
            ray.l[j] = (int) ray.pos[j];
        }
        ray.l[0] = min(ray.l[0], grid.background_reso * 2 - 1);
        ray.l[1] = min(ray.l[1], grid.background_reso - 1);
        ray.l[2] = min(ray.l[2], grid.background_nlayers - 2);
#pragma unroll 3
        for (int j = 0; j < 3; ++j) {
            ray.pos[j] -= ray.l[j];
        }


        float sigma = trilerp_bg_one(
                grid.background_links,
                grid.background_data,
                grid.background_reso,
                grid.background_nlayers,
                4,
                ray.l,
                ray.pos,
                3);
        // if (i == n_steps - 1) {
        //     ray.world_step = 1e9;
        // }

        // if (opt.randomize && opt.random_sigma_std_background > 0.0)
        //     sigma += ray.rng.randn() * opt.random_sigma_std_background;
        if (sigma > 0.f) {
            float total_color = 0.f;
            const float pcnt = ray.world_step * (invr_last - invr_mid) * sigma;
            const float weight = _EXP(log_transmit) * (1.f - _EXP(-pcnt));
            log_transmit -= pcnt;

            for (int i = 0; i < 3; ++i) {
                const float color = trilerp_bg_one(
                        grid.background_links,
                        grid.background_data,
                        grid.background_reso,
                        grid.background_nlayers,
                        4,
                        ray.l,
                        ray.pos,
                        i) * C0 + 0.5f;  // Scale by SH DC factor to help normalize lrs

                total_color += fmaxf(color, 0.f) * grad_output[i];
                if (color > 0.f) {
                    const float curr_grad_color = C0 * weight * grad_output[i];
                    trilerp_backward_bg_one(
                            grid.background_links,
                            grads.grad_background_out,
                            nullptr,
                            grid.background_reso,
                            grid.background_nlayers,
                            4,
                            ray.l,
                            ray.pos,
                            curr_grad_color,
                            i);
                }
            }

            accum -= weight * total_color;
            float curr_grad_sigma = ray.world_step * (invr_last - invr_mid) * (
                    total_color * _EXP(log_transmit) - accum);
            if (sparsity_loss > 0.f) {
                // Cauchy version (from SNeRG)
                curr_grad_sigma += sparsity_loss * (4 * sigma / (1 + 2 * (sigma * sigma)));

                // Alphs version (from PlenOctrees)
                // curr_grad_sigma += sparsity_loss * _EXP(-pcnt) * ray.world_step;
            }

            trilerp_backward_bg_one(
                    grid.background_links,
                    grads.grad_background_out,
                    grads.mask_background_out,
                    grid.background_reso,
                    grid.background_nlayers,
                    4,
                    ray.l,
                    ray.pos,
                    curr_grad_sigma,
                    3);

            if (_EXP(log_transmit) < opt.stop_thresh) {
                break;
            }
        }
        invr_last = invr_mid;
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
    const int ray_id = tid >> 5; // same as / 32, which is the WARP_SIZE
    const int ray_blk_id = threadIdx.x >> 5; // difference between tid and threadIdx.x? --> tid is the total id (batch/ray id)
    const int lane_id = threadIdx.x & 0x1F; // take only last 5 digits

    if (lane_id >= grid.sh_data_dim)  // Bad, but currently the best way due to coalesced memory access
        return;

    __shared__ float sphfunc_val[TRACE_RAY_CUDA_RAYS_PER_BLOCK][9]; // seems to be hard coded for 9 basis?
    __shared__ SingleRaySpec ray_spec[TRACE_RAY_CUDA_RAYS_PER_BLOCK];
    __shared__ typename WarpReducef::TempStorage temp_storage[
        TRACE_RAY_CUDA_RAYS_PER_BLOCK];
    ray_spec[ray_blk_id].set(rays.origins[ray_id].data(),
            rays.dirs[ray_id].data());
    calc_sphfunc(grid, lane_id,
                 ray_id,
                 ray_spec[ray_blk_id].dir,
                 sphfunc_val[ray_blk_id]); // calculate spherial harmonics function

    // this function also converts ray o/d into grid coordinate
    ray_find_bounds(ray_spec[ray_blk_id], grid, opt, ray_id);
    __syncwarp((1U << grid.sh_data_dim) - 1); // make sure all rays are loaded and sh computed?


    trace_ray_surf_trav(
        grid,
        ray_spec[ray_blk_id],
        // ray_vox,
        opt,
        lane_id,
        sphfunc_val[ray_blk_id],
        temp_storage[ray_blk_id],
        out[ray_id].data(),
        log_transmit_out == nullptr ? nullptr : log_transmit_out + ray_id);

}

__launch_bounds__(TRACE_RAY_CUDA_THREADS, MIN_BLOCKS_PER_SM)
__global__ void render_ray_image_kernel(
        PackedSparseGridSpec grid,
        PackedCameraSpec cam,
        RenderOptions opt,
        float* __restrict__ out,
        float* __restrict__ log_transmit_out = nullptr) {
    CUDA_GET_THREAD_ID(tid, cam.height * cam.width * WARP_SIZE);
    const int ray_id = tid >> 5;
    const int ray_blk_id = threadIdx.x >> 5;
    const int lane_id = threadIdx.x & 0x1F;

    if (lane_id >= grid.sh_data_dim)
        return;

    const int ix = ray_id % cam.width;
    const int iy = ray_id / cam.width;

    __shared__ float sphfunc_val[TRACE_RAY_CUDA_RAYS_PER_BLOCK][9];
    __shared__ SingleRaySpec ray_spec[TRACE_RAY_CUDA_RAYS_PER_BLOCK];
    __shared__ typename WarpReducef::TempStorage temp_storage[
        TRACE_RAY_CUDA_RAYS_PER_BLOCK];

    cam2world_ray(ix, iy, cam, ray_spec[ray_blk_id].dir, ray_spec[ray_blk_id].origin);
    calc_sphfunc(grid, lane_id,
                 ray_id,
                 ray_spec[ray_blk_id].dir,
                 sphfunc_val[ray_blk_id]);
    ray_find_bounds(ray_spec[ray_blk_id], grid, opt, ray_id);
    __syncwarp((1U << grid.sh_data_dim) - 1);

    trace_ray_surf_trav(
        grid,
        ray_spec[ray_blk_id],
        opt,
        lane_id,
        sphfunc_val[ray_blk_id],
        temp_storage[ray_blk_id],
        out + ray_id * 3,
        log_transmit_out == nullptr ? nullptr : log_transmit_out + ray_id);
}

__launch_bounds__(TRACE_RAY_BKWD_CUDA_THREADS, MIN_BLOCKS_PER_SM)
__global__ void render_ray_backward_kernel(
    PackedSparseGridSpec grid,
    const float* __restrict__ grad_output,
    const float* __restrict__ color_cache, // predict rgb
    PackedRaysSpec rays,
    RenderOptions opt,
    bool grad_out_is_rgb,
    const float* __restrict__ log_transmit_in,
    float beta_loss,
    float sparsity_loss,
    PackedGridOutputGrads grads,
    float* __restrict__ accum_out = nullptr, // left-over gradient for background?
    float* __restrict__ log_transmit_out = nullptr) {
    CUDA_GET_THREAD_ID(tid, int(rays.origins.size(0)) * WARP_SIZE);
    const int ray_id = tid >> 5;
    const int ray_blk_id = threadIdx.x >> 5;
    const int lane_id = threadIdx.x & 0x1F;

    if (lane_id >= grid.sh_data_dim)
        return;

    __shared__ float sphfunc_val[TRACE_RAY_BKWD_CUDA_RAYS_PER_BLOCK][9];
    __shared__ float grad_sphfunc_val[TRACE_RAY_CUDA_RAYS_PER_BLOCK][9];
    SingleRaySpec ray_spec[TRACE_RAY_BKWD_CUDA_RAYS_PER_BLOCK];
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
    // if (lane_id == 0) {
    //     ray_find_bounds(ray_spec[ray_blk_id], grid, opt, ray_id);
    // }
    ray_find_bounds(ray_spec[ray_blk_id], grid, opt, ray_id);

    float grad_out[3]; // computes gradient for current ray
    if (grad_out_is_rgb) { // true for fused function (grad_output is rgb_gt)
        const float norm_factor = 2.f / (3 * int(rays.origins.size(0))); // gradient of MSE
#pragma unroll 3
        for (int i = 0; i < 3; ++i) {
            const float resid = color_cache[ray_id * 3 + i] - grad_output[ray_id * 3 + i];
            grad_out[i] = resid * norm_factor;
        }
    } else { // for backward function (grad_output is grad for color)
#pragma unroll 3
        for (int i = 0; i < 3; ++i) {
            grad_out[i] = grad_output[ray_id * 3 + i];
        }
    }

    __syncwarp((1U << grid.sh_data_dim) - 1);
    trace_ray_surf_trav_backward(
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
__global__ void render_background_image_kernel(
        PackedSparseGridSpec grid,
        PackedCameraSpec cam,
        RenderOptions opt,
        const float* __restrict__ log_transmit,
        // Outputs
        float* __restrict__ out) {
    CUDA_GET_THREAD_ID(ray_id, cam.height * cam.width);
    if (log_transmit[ray_id] < -25.f) return;
    const int ix = ray_id % cam.width;
    const int iy = ray_id / cam.width;
    SingleRaySpec ray_spec;
    cam2world_ray(ix, iy, cam, ray_spec.dir, ray_spec.origin);
    ray_find_bounds_bg(ray_spec, grid, opt, ray_id);
    render_background_forward(
        grid,
        ray_spec,
        opt,
        log_transmit[ray_id],
        out + ray_id * 3);
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
        float sparsity_loss,
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
        sparsity_loss,
        grads);
}

__launch_bounds__(TRACE_RAY_CUDA_THREADS, MIN_BLOCKS_PER_SM)
__global__ void render_ray_expected_term_kernel(
        PackedSparseGridSpec grid,
        PackedRaysSpec rays,
        RenderOptions opt,
        float* __restrict__ out) {
        // const PackedSparseGridSpec& __restrict__ grid,
        // SingleRaySpec& __restrict__ ray,
        // const RenderOptions& __restrict__ opt,
        // float* __restrict__ out) {
    CUDA_GET_THREAD_ID(ray_id, rays.origins.size(0));
    SingleRaySpec ray_spec(rays.origins[ray_id].data(), rays.dirs[ray_id].data());
    ray_find_bounds(ray_spec, grid, opt, ray_id);
    trace_ray_expected_term(
        grid,
        ray_spec,
        opt,
        out + ray_id);
}

__launch_bounds__(TRACE_RAY_CUDA_THREADS, MIN_BLOCKS_PER_SM)
__global__ void render_ray_sigma_thresh_kernel(
        PackedSparseGridSpec grid,
        PackedRaysSpec rays,
        RenderOptions opt,
        float sigma_thresh,
        float* __restrict__ out) {
        // const PackedSparseGridSpec& __restrict__ grid,
        // SingleRaySpec& __restrict__ ray,
        // const RenderOptions& __restrict__ opt,
        // float* __restrict__ out) {
    CUDA_GET_THREAD_ID(ray_id, rays.origins.size(0));
    SingleRaySpec ray_spec(rays.origins[ray_id].data(), rays.dirs[ray_id].data());
    ray_find_bounds(ray_spec, grid, opt, ray_id);
    trace_ray_sigma_thresh(
        grid,
        ray_spec,
        opt,
        sigma_thresh,
        out + ray_id);
}

__launch_bounds__(TRACE_RAY_CUDA_THREADS, MIN_BLOCKS_PER_SM)
__global__ void extract_ray_pts_kernel(
        PackedSparseGridSpec grid,
        PackedRaysSpec rays,
        RenderOptions opt,
        float alpha_thresh,
        float* __restrict__ out) {
    CUDA_GET_THREAD_ID(ray_id, rays.origins.size(0));
    SingleRaySpec ray_spec(rays.origins[ray_id].data(), rays.dirs[ray_id].data());
    ray_find_bounds(ray_spec, grid, opt, ray_id);
    trace_ray_extract_pt(
        grid,
        ray_spec,
        opt,
        alpha_thresh,
        out + ray_id);
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

torch::Tensor volume_render_surf_trav(SparseGridSpec& grid, RaysSpec& rays, RenderOptions& opt) {
    DEVICE_GUARD(grid.sh_data);
    grid.check();
    rays.check();


    const auto Q = rays.origins.size(0);

    torch::Tensor results = torch::empty_like(rays.origins);

    bool use_background = grid.background_links.defined() &&
                          grid.background_links.size(0) > 0;
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

// torch::Tensor volume_render_surface_image(SparseGridSpec& grid, CameraSpec& cam, RayVoxIntersecSpec& ray_vox, RenderOptions& opt) {
//     DEVICE_GUARD(grid.sh_data);
//     grid.check();
//     cam.check();
//     ray_vox.check();


//     const auto Q = cam.height * cam.width;
//     auto options =
//         torch::TensorOptions()
//         .dtype(grid.sh_data.dtype())
//         .layout(torch::kStrided)
//         .device(grid.sh_data.device())
//         .requires_grad(false);

//     torch::Tensor results = torch::empty({cam.height, cam.width, 3}, options);

//     bool use_background = grid.background_links.defined() &&
//                           grid.background_links.size(0) > 0;
//     torch::Tensor log_transmit;
//     if (use_background) {
//         log_transmit = torch::empty({cam.height, cam.width}, options);
//     }

//     {
//         const int cuda_n_threads = TRACE_RAY_CUDA_THREADS;
//         const int blocks = CUDA_N_BLOCKS_NEEDED(Q * WARP_SIZE, cuda_n_threads);
//         device::render_ray_image_kernel<<<blocks, cuda_n_threads>>>(
//                 grid,
//                 cam,
//                 ray_vox,
//                 opt,
//                 // Output
//                 results.data_ptr<float>(),
//                 use_background ? log_transmit.data_ptr<float>() : nullptr);
//     }

//     if (use_background) {
//         // printf("RENDER BG\n");
//         const int blocks = CUDA_N_BLOCKS_NEEDED(Q, TRACE_RAY_BG_CUDA_THREADS);
//         device::render_background_image_kernel<<<blocks, TRACE_RAY_BG_CUDA_THREADS>>>(
//                 grid,
//                 cam,
//                 opt,
//                 log_transmit.data_ptr<float>(),
//                 results.data_ptr<float>());
//     }

//     CUDA_CHECK_ERRORS;
//     return results;
// }

void volume_render_surf_trav_backward(
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

    bool use_background = grid.background_links.defined() &&
                          grid.background_links.size(0) > 0;
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
                0.f,
                // Output
                grads);
    }

    CUDA_CHECK_ERRORS;
}

void volume_render_surf_trav_fused(
        SparseGridSpec& grid,
        RaysSpec& rays,
        RenderOptions& opt,
        torch::Tensor rgb_gt,
        float beta_loss, // beta loss and sparsity loss are just weights for those loss
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

    bool use_background = grid.background_links.defined() &&
                          grid.background_links.size(0) > 0;
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
                rgb_out.packed_accessor32<float, 2, torch::RestrictPtrTraits>(), // <dtype, dim, __restrict__>
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
                sparsity_loss,
                // Output
                grads);
    }

    CUDA_CHECK_ERRORS;
}

torch::Tensor volume_render_expected_term_surf_trav(SparseGridSpec& grid,
        RaysSpec& rays, RenderOptions& opt) {
    auto options =
        torch::TensorOptions()
        .dtype(rays.origins.dtype())
        .layout(torch::kStrided)
        .device(rays.origins.device())
        .requires_grad(false);
    torch::Tensor results = torch::empty({rays.origins.size(0)}, options);
    const auto Q = rays.origins.size(0);
    const int blocks = CUDA_N_BLOCKS_NEEDED(Q, TRACE_RAY_CUDA_THREADS);
    device::render_ray_expected_term_kernel<<<blocks, TRACE_RAY_CUDA_THREADS>>>(
            grid,
            rays,
            opt,
            results.data_ptr<float>()
        );
    return results;
}

torch::Tensor volume_render_sigma_thresh_surf_trav(SparseGridSpec& grid,
        RaysSpec& rays,
        RenderOptions& opt,
        float sigma_thresh) {
    auto options =
        torch::TensorOptions()
        .dtype(rays.origins.dtype())
        .layout(torch::kStrided)
        .device(rays.origins.device())
        .requires_grad(false);
    torch::Tensor results = torch::empty({rays.origins.size(0)}, options);
    const auto Q = rays.origins.size(0);
    const int blocks = CUDA_N_BLOCKS_NEEDED(Q, TRACE_RAY_CUDA_THREADS);
    device::render_ray_sigma_thresh_kernel<<<blocks, TRACE_RAY_CUDA_THREADS>>>(
            grid,
            rays,
            opt,
            sigma_thresh,
            results.data_ptr<float>()
        );
    return results;
}

torch::Tensor extract_pts_surf_trav(SparseGridSpec& grid,
        RaysSpec& rays,
        RenderOptions& opt,
        float sigma_thresh) {
    auto options =
        torch::TensorOptions()
        .dtype(rays.origins.dtype())
        .layout(torch::kStrided)
        .device(rays.origins.device())
        .requires_grad(false);
    torch::Tensor results = torch::empty({rays.origins.size(0)}, options);
    const auto Q = rays.origins.size(0);
    const int blocks = CUDA_N_BLOCKS_NEEDED(Q, TRACE_RAY_CUDA_THREADS);
    device::extract_ray_pts_kernel<<<blocks, TRACE_RAY_CUDA_THREADS>>>(
            grid,
            rays,
            opt,
            sigma_thresh,
            results.data_ptr<float>()
        );
    return results;
}
