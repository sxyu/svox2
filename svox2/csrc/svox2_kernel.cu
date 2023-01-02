// Copyright 2021 Alex Yu
#include <torch/extension.h>
#include <cstdint>
#include "cuda_util.cuh"
#include "data_spec_packed.cuh"
#include "render_util.cuh"

namespace {
namespace device {

__global__ void sample_grid_sh_kernel(
        PackedSparseGridSpec grid,
        const torch::PackedTensorAccessor32<float, 2, torch::RestrictPtrTraits> points,
        // Output
        torch::PackedTensorAccessor32<float, 2, torch::RestrictPtrTraits> out) {
    CUDA_GET_THREAD_ID(tid, points.size(0) * grid.sh_data_dim);
    const int idx = tid % grid.sh_data_dim;
    const int pid = tid / grid.sh_data_dim;

    float point[3] = {points[pid][0], points[pid][1], points[pid][2]};
    transform_coord(point, grid._scaling, grid._offset);

    int32_t l[3];
#pragma unroll 3
    for (int i = 0; i < 3; ++i) {
        point[i] = fminf(fmaxf(point[i], 0.f), grid.size[i] - 1.f);
        l[i] = min((int32_t)point[i], (int32_t)(grid.size[i] - 2));
        point[i] -= l[i];
    }

    const int offy = grid.size[2], offx = grid.size[1] * grid.size[2];
    const int32_t* __restrict__ link_ptr = &grid.links[l[0] * offx + l[1] * offy + l[2]];

#define MAYBE_READ_LINK(u) ((link_ptr[u] >= 0) ? grid.sh_data[ \
        link_ptr[u] * size_t(grid.sh_data_dim) + idx] : 0.f)

    const float ix0y0 = lerp(MAYBE_READ_LINK(0), MAYBE_READ_LINK(1), point[2]);
    const float ix0y1 = lerp(MAYBE_READ_LINK(offy), MAYBE_READ_LINK(offy + 1), point[2]);
    const float ix0 = lerp(ix0y0, ix0y1, point[1]);
    const float ix1y0 = lerp(MAYBE_READ_LINK(offx), MAYBE_READ_LINK(offx + 1), point[2]);
    const float ix1y1 = lerp(MAYBE_READ_LINK(offy + offx),
                             MAYBE_READ_LINK(offy + offx + 1), point[2]);
    const float ix1 = lerp(ix1y0, ix1y1, point[1]);
    out[pid][idx] = lerp(ix0, ix1, point[0]);
}
#undef MAYBE_READ_LINK

__global__ void sample_grid_density_kernel(
        PackedSparseGridSpec grid,
        const torch::PackedTensorAccessor32<float, 2, torch::RestrictPtrTraits> points,
        // Output
        torch::PackedTensorAccessor32<float, 2, torch::RestrictPtrTraits> out) {
    CUDA_GET_THREAD_ID(tid, points.size(0));

    float point[3] = {points[tid][0], points[tid][1], points[tid][2]};
    transform_coord(point, grid._scaling, grid._offset);

    int32_t l[3];
#pragma unroll 3
    for (int i = 0; i < 3; ++i) {
        point[i] = fminf(fmaxf(point[i], 0.f), grid.size[i] - 1.f);
        l[i] = min((int32_t)point[i], grid.size[i] - 2);
        point[i] -= l[i];
    }

    const int offy = grid.size[2], offx = grid.size[1] * grid.size[2];
    const int32_t* __restrict__ link_ptr = &grid.links[l[0] * offx + l[1] * offy + l[2]];

#define MAYBE_READ_LINK_D(u) ((link_ptr[u] >= 0) ? grid.density_data[link_ptr[u]] : 0.f)

    const float ix0y0 = lerp(MAYBE_READ_LINK_D(0), MAYBE_READ_LINK_D(1), point[2]);
    const float ix0y1 = lerp(MAYBE_READ_LINK_D(offy), MAYBE_READ_LINK_D(offy + 1), point[2]);
    const float ix0 = lerp(ix0y0, ix0y1, point[1]);
    const float ix1y0 = lerp(MAYBE_READ_LINK_D(offx), MAYBE_READ_LINK_D(offx + 1), point[2]);
    const float ix1y1 = lerp(MAYBE_READ_LINK_D(offy + offx),
                             MAYBE_READ_LINK_D(offy + offx + 1), point[2]);
    const float ix1 = lerp(ix1y0, ix1y1, point[1]);
    out[tid][0] = lerp(ix0, ix1, point[0]);
}
#undef MAYBE_READ_LINK_D
__global__ void sample_grid_alpha_kernel(
        PackedSparseGridSpec grid,
        const torch::PackedTensorAccessor32<float, 2, torch::RestrictPtrTraits> points,
        float empty_raw,
        // Output
        torch::PackedTensorAccessor32<float, 2, torch::RestrictPtrTraits> out) {
    CUDA_GET_THREAD_ID(tid, points.size(0));

    float point[3] = {points[tid][0], points[tid][1], points[tid][2]};
    transform_coord(point, grid._scaling, grid._offset);

    int32_t l[3];
#pragma unroll 3
    for (int i = 0; i < 3; ++i) {
        point[i] = fminf(fmaxf(point[i], 0.f), grid.size[i] - 1.f);
        l[i] = min((int32_t)point[i], grid.size[i] - 2);
        point[i] -= l[i];
    }

    const int offy = grid.size[2], offx = grid.size[1] * grid.size[2];
    const int32_t* __restrict__ link_ptr = &grid.links[l[0] * offx + l[1] * offy + l[2]];

#define MAYBE_READ_LINK_D(u) ((link_ptr[u] >= 0) ? grid.density_data[link_ptr[u]] : empty_raw)

    const float ix0y0 = lerp(MAYBE_READ_LINK_D(0), MAYBE_READ_LINK_D(1), point[2]);
    const float ix0y1 = lerp(MAYBE_READ_LINK_D(offy), MAYBE_READ_LINK_D(offy + 1), point[2]);
    const float ix0 = lerp(ix0y0, ix0y1, point[1]);
    const float ix1y0 = lerp(MAYBE_READ_LINK_D(offx), MAYBE_READ_LINK_D(offx + 1), point[2]);
    const float ix1y1 = lerp(MAYBE_READ_LINK_D(offy + offx),
                             MAYBE_READ_LINK_D(offy + offx + 1), point[2]);
    const float ix1 = lerp(ix1y0, ix1y1, point[1]);
    out[tid][0] = lerp(ix0, ix1, point[0]);
}
#undef MAYBE_READ_LINK_D

__global__ void sample_grid_surface_kernel(
        PackedSparseGridSpec grid,
        const torch::PackedTensorAccessor32<float, 2, torch::RestrictPtrTraits> points,
        float const default_surf,
        // Output
        torch::PackedTensorAccessor32<float, 2, torch::RestrictPtrTraits> out) {
    CUDA_GET_THREAD_ID(tid, points.size(0));

    float point[3] = {points[tid][0], points[tid][1], points[tid][2]};
    transform_coord(point, grid._scaling, grid._offset);

    int32_t l[3];
#pragma unroll 3
    for (int i = 0; i < 3; ++i) {
        point[i] = fminf(fmaxf(point[i], 0.f), grid.size[i] - 1.f);
        l[i] = min((int32_t)point[i], grid.size[i] - 2);
        point[i] -= l[i];
    }

    const int offy = grid.size[2], offx = grid.size[1] * grid.size[2];
    const int32_t* __restrict__ link_ptr = &grid.links[l[0] * offx + l[1] * offy + l[2]];

#define MAYBE_READ_LINK_D(u) ((link_ptr[u] >= 0) ? grid.surface_data[link_ptr[u]] : default_surf)

    const float ix0y0 = lerp(MAYBE_READ_LINK_D(0), MAYBE_READ_LINK_D(1), point[2]);
    const float ix0y1 = lerp(MAYBE_READ_LINK_D(offy), MAYBE_READ_LINK_D(offy + 1), point[2]);
    const float ix0 = lerp(ix0y0, ix0y1, point[1]);
    const float ix1y0 = lerp(MAYBE_READ_LINK_D(offx), MAYBE_READ_LINK_D(offx + 1), point[2]);
    const float ix1y1 = lerp(MAYBE_READ_LINK_D(offy + offx),
                             MAYBE_READ_LINK_D(offy + offx + 1), point[2]);
    const float ix1 = lerp(ix1y0, ix1y1, point[1]);
    out[tid][0] = lerp(ix0, ix1, point[0]);
}
#undef MAYBE_READ_LINK_D

__global__ void sample_grid_sh_backward_kernel(
        PackedSparseGridSpec grid,
        const torch::PackedTensorAccessor32<float, 2, torch::RestrictPtrTraits> points,
        const torch::PackedTensorAccessor32<float, 2, torch::RestrictPtrTraits> grad_out,
        // Output
        torch::PackedTensorAccessor64<float, 2, torch::RestrictPtrTraits> grad_data) {
    CUDA_GET_THREAD_ID(tid, points.size(0) * grid.sh_data_dim);
    const int idx = tid % grid.sh_data_dim;
    const int pid = tid / grid.sh_data_dim;

    float point[3] = {points[pid][0], points[pid][1], points[pid][2]};
    transform_coord(point, grid._scaling, grid._offset);

    int32_t l[3];
#pragma unroll 3
    for (int i = 0; i < 3; ++i) {
        point[i] = fminf(fmaxf(point[i], 0.f), grid.size[i] - 1.f);
        l[i] = min((int32_t)point[i], grid.size[i] - 2);
        point[i] -= l[i];
    }

    const int offy = grid.size[2], offx = grid.size[1] * grid.size[2];
    const int32_t* __restrict__ link_ptr = &grid.links[l[0] * offx + l[1] * offy + l[2]];

    const float go = grad_out[pid][idx];

    const float xb = point[0], yb = point[1], zb = point[2];
    const float xa = 1.f - point[0], ya = 1.f - point[1], za = 1.f - point[2];

#define MAYBE_ADD_GRAD_LINK_PTR(u, content) if (link_ptr[u] >= 0)  \
            atomicAdd(&grad_data[link_ptr[u]][idx], content)

    const float xago = xa * go;
    float tmp = ya * xago;
    MAYBE_ADD_GRAD_LINK_PTR(0, tmp * za);
    MAYBE_ADD_GRAD_LINK_PTR(1, tmp * zb);
    tmp = yb * xago;
    MAYBE_ADD_GRAD_LINK_PTR(offy, tmp * za);
    MAYBE_ADD_GRAD_LINK_PTR(offy + 1, tmp * zb);

    const float xbgo = xb * go;
    tmp = ya * xbgo;
    MAYBE_ADD_GRAD_LINK_PTR(offx, tmp * za);
    MAYBE_ADD_GRAD_LINK_PTR(offx + 1, tmp * zb);
    tmp = yb * xbgo;
    MAYBE_ADD_GRAD_LINK_PTR(offx + offy, tmp * za);
    MAYBE_ADD_GRAD_LINK_PTR(offx + offy + 1, tmp * zb);
}
#undef MAYBE_ADD_GRAD_LINK_PTR

__global__ void sample_grid_density_backward_kernel(
        PackedSparseGridSpec grid,
        const torch::PackedTensorAccessor32<float, 2, torch::RestrictPtrTraits> points,
        const torch::PackedTensorAccessor32<float, 2, torch::RestrictPtrTraits> grad_out,
        // Output
        torch::PackedTensorAccessor32<float, 2, torch::RestrictPtrTraits> grad_data) {
    CUDA_GET_THREAD_ID(tid, points.size(0));

    float point[3] = {points[tid][0], points[tid][1], points[tid][2]};
    transform_coord(point, grid._scaling, grid._offset);

    int32_t l[3];
#pragma unroll 3
    for (int i = 0; i < 3; ++i) {
        point[i] = fminf(fmaxf(point[i], 0.f), grid.size[i] - 1.f);
        l[i] = min((int32_t)point[i], grid.size[i] - 2);
        point[i] -= l[i];
    }

    const int offy = grid.size[2], offx = grid.size[1] * grid.size[2];
    const int32_t* __restrict__ link_ptr = &grid.links[l[0] * offx + l[1] * offy + l[2]];

    const float go = grad_out[tid][0];

    const float xb = point[0], yb = point[1], zb = point[2];
    const float xa = 1.f - point[0], ya = 1.f - point[1], za = 1.f - point[2];

#define MAYBE_ADD_GRAD_LINK_PTR_D(u, content) if (link_ptr[u] >= 0)  \
            atomicAdd(grad_data[link_ptr[u]].data(), content)

    const float xago = xa * go;
    float tmp = ya * xago;
    MAYBE_ADD_GRAD_LINK_PTR_D(0, tmp * za);
    MAYBE_ADD_GRAD_LINK_PTR_D(1, tmp * zb);
    tmp = yb * xago;
    MAYBE_ADD_GRAD_LINK_PTR_D(offy, tmp * za);
    MAYBE_ADD_GRAD_LINK_PTR_D(offy + 1, tmp * zb);

    const float xbgo = xb * go;
    tmp = ya * xbgo;
    MAYBE_ADD_GRAD_LINK_PTR_D(offx, tmp * za);
    MAYBE_ADD_GRAD_LINK_PTR_D(offx + 1, tmp * zb);
    tmp = yb * xbgo;
    MAYBE_ADD_GRAD_LINK_PTR_D(offx + offy, tmp * za);
    MAYBE_ADD_GRAD_LINK_PTR_D(offx + offy + 1, tmp * zb);
}

__global__ void cubic_extract_iso_pts_kernel(
        const torch::PackedTensorAccessor32<int32_t, 3, torch::RestrictPtrTraits> links,
        const torch::PackedTensorAccessor32<float, 2, torch::RestrictPtrTraits> level_data,
        const torch::PackedTensorAccessor32<float, 2, torch::RestrictPtrTraits> mask_data,
        const int32_t* __restrict__ cell_ids,
        size_t Q,
        const int n_sample,
        const float density_thresh,
        // Output
        torch::PackedTensorAccessor32<float, 3, torch::RestrictPtrTraits> out        
        ) {
    CUDA_GET_THREAD_ID_U64(tid, Q);
    const int xyz = cell_ids[tid];
    const int z = xyz % links.size(2);
    const int xy = xyz / links.size(2);
    const int y = xy % links.size(1);
    const int x = xy / links.size(1);

    
    // check if grid exist
    if ((x >= links.size(0) - 1) || (y >= links.size(1) - 1) || (z >= links.size(2) - 1) || \
        (links[x][y][z] < 0) || (links[x][y][z+1] < 0) || (links[x][y+1][z] < 0) || (links[x][y+1][z+1] < 0) || \
        (links[x+1][y][z] < 0) || (links[x+1][y][z+1] < 0) || (links[x+1][y+1][z] < 0) || (links[x+1][y+1][z+1] < 0)
    ){
        return;
    }

    // fetch surface
    const float* dptr = level_data.data();
    const float* dptr_mask = mask_data.data();

    #define __FETCH_LV_DATA(x,y,z) (dptr[links[x][y][z]])
    #define __FETCH_MASK_DATA(x,y,z) (dptr_mask[links[x][y][z]])

    double const surface[8] = {
        __FETCH_LV_DATA(x,y,z),
        __FETCH_LV_DATA(x,y,z+1),
        __FETCH_LV_DATA(x,y+1,z),
        __FETCH_LV_DATA(x,y+1,z+1),
        __FETCH_LV_DATA(x+1,y,z),
        __FETCH_LV_DATA(x+1,y,z+1),
        __FETCH_LV_DATA(x+1,y+1,z),
        __FETCH_LV_DATA(x+1,y+1,z+1)
    };

    const float step_size = 1.f / (n_sample - 1);

    for (int i = 0; i < n_sample; ++i){
        const float pos1 = i * step_size;
        for (int j = 0; j < n_sample; ++j){
            const float pos2 = j * step_size;
            for (int dir_id = 0; dir_id < 3; ++dir_id){
                double dirs[3] = {0., 0., 0.};
                double origin[3] = {0., 0., 0.};

                if (dir_id == 0){
                    // varying along y,z
                    dirs[0] = 1.;
                    origin[1] = pos1;
                    origin[2] = pos2;
                }else if (dir_id == 1){
                    // varying along x,z
                    dirs[1] = 1.;
                    origin[0] = pos1;
                    origin[2] = pos2;
                }else{
                    // varying along x,y
                    dirs[2] = 1.;
                    origin[0] = pos1;
                    origin[1] = pos2;
                }


                double fs[4];
                surface_to_cubic_equation_01(surface, origin, dirs, fs);

                double st[3] = {-1, -1, -1}; // sample t
                cubic_equation_solver_vieta(
                    fs[0], fs[1], fs[2], fs[3],
                    1e-8, // float eps
                    1e-10, // double eps
                    st
                    );

                for (int st_i=0; st_i<3; ++st_i){
                    if ((st[st_i]>= 0.) && (st[st_i] <= 1.)){
                        float const pt[] = {
                            origin[0] + dirs[0] * st[st_i],
                            origin[1] + dirs[1] * st[st_i],
                            origin[2] + dirs[2] * st[st_i]
                        };

                        
                        //check against mask
                        // float mask_val = trilerp_cuvol_one(
                        //         links, density_data,
                        //         stride_x,
                        //         links.size(2),
                        //         1,
                        //         ray.l, ray.pos,
                        //         0);

                        const float ix0y0 = lerp(__FETCH_MASK_DATA(x,y,z), __FETCH_MASK_DATA(x,y,z+1), pt[2]);            // stride is last dim of the data
                        const float ix0y1 = lerp(__FETCH_MASK_DATA(x,y+1,z), __FETCH_MASK_DATA(x,y+1,z+1), pt[2]);
                        const float ix0 = lerp(ix0y0, ix0y1, pt[1]);
                        const float ix1y0 = lerp(__FETCH_MASK_DATA(x+1,y,z), __FETCH_MASK_DATA(x+1,y,z+1), pt[2]);
                        const float ix1y1 = lerp(__FETCH_MASK_DATA(x+1,y+1,z),__FETCH_MASK_DATA(x+1,y+1,z+1), pt[2]);
                        const float ix1 = lerp(ix1y0, ix1y1, pt[1]);
                        float const mask_val = lerp(ix0, ix1, pt[0]);

                        if (mask_val >= density_thresh){
                            out[tid][i*n_sample*3 + j*3 + dir_id][0] = pt[0] + x;
                            out[tid][i*n_sample*3 + j*3 + dir_id][1] = pt[1] + y;
                            out[tid][i*n_sample*3 + j*3 + dir_id][2] = pt[2] + z;

                            break;
                        }

                    }
                }

            }
        }

    }



}



}  // namespace device
}  // namespace


std::tuple<torch::Tensor, torch::Tensor> sample_grid(SparseGridSpec& grid, torch::Tensor points,
                                                     bool want_colors) {
    DEVICE_GUARD(points);
    grid.check();
    CHECK_INPUT(points);
    TORCH_CHECK(points.ndimension() == 2);
    const auto Q = points.size(0) * grid.sh_data.size(1);
    const int cuda_n_threads = std::min<int>(Q, CUDA_MAX_THREADS);
    const int blocks = CUDA_N_BLOCKS_NEEDED(Q, cuda_n_threads);
    const int blocks_density = CUDA_N_BLOCKS_NEEDED(points.size(0), cuda_n_threads);
    torch::Tensor result_density = torch::empty({points.size(0),
                        grid.density_data.size(1)}, points.options());
    torch::Tensor result_sh = torch::empty({want_colors ? points.size(0) : 0,
                        grid.sh_data.size(1)}, points.options());

    cudaStream_t stream_1, stream_2;
    cudaStreamCreate(&stream_1);
    cudaStreamCreate(&stream_2);

    device::sample_grid_density_kernel<<<blocks_density, cuda_n_threads, 0, stream_1>>>(
            grid,
            points.packed_accessor32<float, 2, torch::RestrictPtrTraits>(),
            // Output
            result_density.packed_accessor32<float, 2, torch::RestrictPtrTraits>());
    if (want_colors) {
        device::sample_grid_sh_kernel<<<blocks, cuda_n_threads, 0, stream_2>>>(
                grid,
                points.packed_accessor32<float, 2, torch::RestrictPtrTraits>(),
                // Output
                result_sh.packed_accessor32<float, 2, torch::RestrictPtrTraits>());
    }

    cudaStreamSynchronize(stream_1);
    cudaStreamSynchronize(stream_2);
    CUDA_CHECK_ERRORS;
    return std::tuple<torch::Tensor, torch::Tensor>{result_density, result_sh};
}


std::tuple<torch::Tensor, torch::Tensor> sample_grid_sh_surf(SparseGridSpec& grid, torch::Tensor points,
                                                             bool want_colors, bool want_surfaces, float default_surf) {
    DEVICE_GUARD(points);
    grid.check();
    CHECK_INPUT(points);
    TORCH_CHECK(points.ndimension() == 2);
    const auto Q = points.size(0) * grid.sh_data.size(1);
    const int cuda_n_threads = std::min<int>(Q, CUDA_MAX_THREADS);
    const int blocks = CUDA_N_BLOCKS_NEEDED(Q, cuda_n_threads);
    const int blocks_surface = CUDA_N_BLOCKS_NEEDED(points.size(0), cuda_n_threads);
    torch::Tensor result_sh = torch::empty({want_colors ? points.size(0) : 0,
                        grid.sh_data.size(1)}, points.options());
    torch::Tensor result_surf = torch::empty({want_surfaces ? points.size(0) : 0,
                        grid.surface_data.size(1)}, points.options());

    cudaStream_t stream_1, stream_2;
    cudaStreamCreate(&stream_1);
    cudaStreamCreate(&stream_2);

    if (want_colors) {
        device::sample_grid_sh_kernel<<<blocks, cuda_n_threads, 0, stream_1>>>(
                grid,
                points.packed_accessor32<float, 2, torch::RestrictPtrTraits>(),
                // Output
                result_sh.packed_accessor32<float, 2, torch::RestrictPtrTraits>());
    }
    if (want_surfaces) {
        device::sample_grid_surface_kernel<<<blocks_surface, cuda_n_threads, 0, stream_2>>>(
                grid,
                points.packed_accessor32<float, 2, torch::RestrictPtrTraits>(),
                default_surf,
                // Output
                result_surf.packed_accessor32<float, 2, torch::RestrictPtrTraits>());
    }

    cudaStreamSynchronize(stream_1);
    cudaStreamSynchronize(stream_2);
    CUDA_CHECK_ERRORS;
    return std::tuple<torch::Tensor, torch::Tensor>{result_sh, result_surf};
}

torch::Tensor sample_grid_raw_alpha(SparseGridSpec& grid, torch::Tensor points, float empty_raw) {\
    /**
     * Sample raw alpha values from grid
     * Do not apply sigmoid activation, and empty voxels are considered as empty_raw
    */
    DEVICE_GUARD(points);
    grid.check();
    CHECK_INPUT(points);
    TORCH_CHECK(points.ndimension() == 2);
    const auto Q = points.size(0); // * grid.sh_data.size(1); TODO: check if correct!
    const int cuda_n_threads = std::min<int>(Q, CUDA_MAX_THREADS);
    const int blocks_alpha = CUDA_N_BLOCKS_NEEDED(points.size(0), cuda_n_threads);
    torch::Tensor result_alpha = torch::empty({points.size(0),
                        grid.density_data.size(1)}, points.options());


    device::sample_grid_alpha_kernel<<<blocks_alpha, cuda_n_threads, 0>>>(
            grid,
            points.packed_accessor32<float, 2, torch::RestrictPtrTraits>(),
            empty_raw,
            // Output
            result_alpha.packed_accessor32<float, 2, torch::RestrictPtrTraits>());

    CUDA_CHECK_ERRORS;
    return result_alpha;
}

void sample_grid_backward(
        SparseGridSpec& grid,
        torch::Tensor points,
        torch::Tensor grad_out_density,
        torch::Tensor grad_out_sh,
        torch::Tensor grad_density_out,
        torch::Tensor grad_sh_out,
        bool want_colors) {
    DEVICE_GUARD(points);
    grid.check();
    CHECK_INPUT(points);
    CHECK_INPUT(grad_out_density);
    CHECK_INPUT(grad_out_sh);
    CHECK_INPUT(grad_density_out);
    CHECK_INPUT(grad_sh_out);
    TORCH_CHECK(points.ndimension() == 2);
    TORCH_CHECK(grad_out_density.ndimension() == 2);
    TORCH_CHECK(grad_out_sh.ndimension() == 2);
    const auto Q = points.size(0) * grid.sh_data.size(1);

    const int cuda_n_threads = std::min<int>(Q, CUDA_MAX_THREADS);
    const int blocks = CUDA_N_BLOCKS_NEEDED(Q, cuda_n_threads);
    const int blocks_density = CUDA_N_BLOCKS_NEEDED(points.size(0), cuda_n_threads);

    cudaStream_t stream_1, stream_2;
    cudaStreamCreate(&stream_1);
    cudaStreamCreate(&stream_2);

    device::sample_grid_density_backward_kernel<<<blocks_density, cuda_n_threads, 0, stream_1>>>(
            grid,
            points.packed_accessor32<float, 2, torch::RestrictPtrTraits>(),
            grad_out_density.packed_accessor32<float, 2, torch::RestrictPtrTraits>(),
            // Output
            grad_density_out.packed_accessor32<float, 2, torch::RestrictPtrTraits>());

    if (want_colors) {
        device::sample_grid_sh_backward_kernel<<<blocks, cuda_n_threads, 0, stream_2>>>(
                grid,
                points.packed_accessor32<float, 2, torch::RestrictPtrTraits>(),
                grad_out_sh.packed_accessor32<float, 2, torch::RestrictPtrTraits>(),
                // Output
                grad_sh_out.packed_accessor64<float, 2, torch::RestrictPtrTraits>());
    }

    cudaStreamSynchronize(stream_1);
    cudaStreamSynchronize(stream_2);

    CUDA_CHECK_ERRORS;
}


torch::Tensor cubic_extract_iso_pts(
    torch::Tensor links,
    torch::Tensor level_data,
    torch::Tensor mask_data,
    torch::Tensor cell_ids,
    int n_sample,
    float density_thresh
    ) {
    DEVICE_GUARD(level_data);
    CHECK_INPUT(level_data);
    CHECK_INPUT(mask_data);
    CHECK_INPUT(links);
    CHECK_INPUT(cell_ids);


    auto options =
        torch::TensorOptions()
        .dtype(level_data.dtype())
        .layout(torch::kStrided)
        .device(level_data.device())
        .requires_grad(false);
    torch::Tensor out = torch::zeros({cell_ids.size(0), 3 * n_sample * n_sample, 3}, options);

    size_t Q = cell_ids.size(0);

    const int cuda_n_threads = 256;
    const int blocks = CUDA_N_BLOCKS_NEEDED(Q, cuda_n_threads);
    device::cubic_extract_iso_pts_kernel<<<blocks, cuda_n_threads>>>(
            links.packed_accessor32<int32_t, 3, torch::RestrictPtrTraits>(),
            level_data.packed_accessor32<float, 2, torch::RestrictPtrTraits>(), // check if 32 or 64
            mask_data.packed_accessor32<float, 2, torch::RestrictPtrTraits>(),
            cell_ids.data_ptr<int32_t>(),
            Q,
            n_sample,
            density_thresh,
            // Output
            out.packed_accessor32<float, 3, torch::RestrictPtrTraits>());
    CUDA_CHECK_ERRORS;

    return out;
}
