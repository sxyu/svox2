// Copyright 2021 Alex Yu
// Loss computation-related kernels

#include <torch/extension.h>
#include <cstdint>
#include <cstdio>
#include "cuda_util.cuh"
#include "render_util.cuh"
#include "data_spec_packed.cuh"

namespace {

const int WARP_SIZE = 32;
const int TV_GRAD_CUDA_THREADS = 256;
const int TV_GRAD_POINTS_PER_BLOCK = TV_GRAD_CUDA_THREADS / WARP_SIZE;
const int MIN_BLOCKS_PER_SM = 4;

typedef cub::WarpReduce<float> WarpReducef;

namespace device {

__device__ __inline__
void calculate_ray_scale(float ndc_coeffx,
                         float ndc_coeffy,
                         float z,
                         float maxx,
                         float maxy,
                         float maxz,
                         float* __restrict__ scale) {
    // if (ndc_coeffx > 0.f) {
    //     // FF NDC
    //     scale[0] = maxx * (1.f / 256.f);
    //     scale[1] = maxy * (1.f / 256.f);
    //     scale[2] = maxz * (1.f / 256.f);

        // The following shit does not work
        // // Normalized to [-1, 1] (with 0.5 padding)
        // // const float x_norm = (x + 0.5) / maxx * 2 - 1;
        // // const float y_norm = (y + 0.5) / maxy * 2 - 1;
        // const float z_norm = (z + 0.5) / maxz * 2 - 1;
        //
        // // NDC distances
        // const float disparity = (1 - z_norm) / 2.f; // in [0, 1]
        // scale[0] = (ndc_coeffx * disparity);
        // scale[1] = (ndc_coeffy * disparity);
        // scale[2] = -((z_norm - 1.f + 2.f / maxz) * disparity) / (maxz * 0.5f);
    // } else {
        scale[0] = maxx * (1.f / 256.f);
        scale[1] = maxy * (1.f / 256.f);
        scale[2] = maxz * (1.f / 256.f);
    // }
}


#define CALCULATE_RAY_SCALE(out_name, maxx, maxy, maxz) \
    calculate_ray_scale( \
            ndc_coeffx, ndc_coeffy, \
            z, \
            maxx, \
            maxy, \
            maxz, \
            out_name)

__launch_bounds__(TV_GRAD_CUDA_THREADS, MIN_BLOCKS_PER_SM)
__global__ void tv_kernel(
        torch::PackedTensorAccessor32<int32_t, 3, torch::RestrictPtrTraits> links,
        torch::PackedTensorAccessor64<float, 2, torch::RestrictPtrTraits> data,
        int start_dim, int end_dim,
        float scale,
        size_t Q,
        bool ignore_edge,
        float ndc_coeffx, float ndc_coeffy,
        // Output
        float* __restrict__ out) {
    CUDA_GET_THREAD_ID_U64(tid, Q);

    typedef cub::BlockReduce<float, 1024> BlockReduce;
    __shared__ typename BlockReduce::TempStorage temp_storage;

    const int idx = tid % (end_dim - start_dim) + start_dim;
    const int xyz = tid / (end_dim - start_dim);
    const int z = xyz % (links.size(2) - 1);
    const int xy = xyz / (links.size(2) - 1);
    const int y = xy % (links.size(1) - 1);
    const int x = xy / (links.size(1) - 1);

    if (ignore_edge && links[x][y][z] == 0) return;
    float scaling[3];
    CALCULATE_RAY_SCALE(scaling, links.size(0), links.size(1), links.size(2));

    const float val000 = (links[x][y][z] >= 0 ?
                          data[links[x][y][z]][idx] : 0.f);
    const float null_val = (ignore_edge ? val000 : 0.f);
    const float val100 = (links[x + 1][y][z] >= 0 ?
                          data[links[x + 1][y][z]][idx] : null_val);
    const float val010 = (links[x][y + 1][z] >= 0 ?
                          data[links[x][y + 1][z]][idx] : null_val);
    const float val001 = (links[x][y][z + 1] >= 0 ?
                          data[links[x][y][z + 1]][idx] : null_val);
    const float dx = (val100 - val000) * scaling[0];
    const float dy = (val010 - val000) * scaling[1];
    const float dz = (val001 - val000) * scaling[2];
    const float tresult = sqrtf(1e-5f + dx * dx + dy * dy + dz * dz);

    const float bresult = BlockReduce(temp_storage).Sum(tresult);
    if (threadIdx.x == 0) {
        atomicAdd(out, bresult * scale);
    }
}

__launch_bounds__(TV_GRAD_CUDA_THREADS, MIN_BLOCKS_PER_SM)
__global__ void tv_grad_kernel(
        const torch::PackedTensorAccessor32<int32_t, 3, torch::RestrictPtrTraits> links,
        const torch::PackedTensorAccessor64<float, 2, torch::RestrictPtrTraits> data,
        int start_dim, int end_dim,
        float scale,
        size_t Q,
        bool ignore_edge,
        float ndc_coeffx, float ndc_coeffy,
        // Output
        float* __restrict__ grad_data) {
    CUDA_GET_THREAD_ID_U64(tid, Q);
    float dummy;
    const int idx = tid % (end_dim - start_dim) + start_dim;
    const int xyz = tid / (end_dim - start_dim);
    const int z = xyz % (links.size(2) - 1);
    const int xy = xyz / (links.size(2) - 1);
    const int y = xy % (links.size(1) - 1);
    const int x = xy / (links.size(1) - 1);

    if (ignore_edge && links[x][y][z] == 0) return;

    float scaling[3];
    CALCULATE_RAY_SCALE(scaling, links.size(0), links.size(1), links.size(2));

    const float* dptr = data.data();
    const size_t ddim = data.size(1);
    float v000 = 0.f, v100 = 0.f, v010 = 0.f, v001 = 0.f;
    float* gptr000 = &dummy,
         * gptr100 = &dummy,
         * gptr010 = &dummy,
         * gptr001 = &dummy;

    if (links[x][y][z] >= 0) {
        const size_t lnk = links[x][y][z] * ddim + idx;
        v000 = dptr[lnk];
        gptr000 = grad_data + lnk;
    }
    if (links[x + 1][y][z] >= 0) {
        const size_t lnk = links[x + 1][y][z] * ddim + idx;
        v100 = dptr[lnk];
        gptr100 = grad_data + lnk;
    } else if (ignore_edge) v100 = v000;
    if (links[x][y + 1][z] >= 0) {
        const size_t lnk = links[x][y + 1][z] * ddim + idx;
        v010 = dptr[lnk];
        gptr010 = grad_data + lnk;
    } else if (ignore_edge) v010 = v000;
    if (links[x][y][z + 1] >= 0) {
        const size_t lnk = links[x][y][z + 1] * ddim + idx;
        v001 = dptr[lnk];
        gptr001 = grad_data + lnk;
    } else if (ignore_edge) v001 = v000;

    float dx = (v100 - v000);
    float dy = (v010 - v000);
    float dz = (v001 - v000);
    const float idelta = scale * rsqrtf(1e-9f + dx * dx + dy * dy + dz * dz);
    dx *= scaling[0];
    dy *= scaling[1];
    dz *= scaling[2];
    if (dx != 0.f) atomicAdd(gptr100, dx * idelta);
    if (dy != 0.f) atomicAdd(gptr010, dy * idelta);
    if (dz != 0.f) atomicAdd(gptr001, dz * idelta);
    atomicAdd(gptr000, -(dx + dy + dz) * idelta);
}

__launch_bounds__(TV_GRAD_CUDA_THREADS, MIN_BLOCKS_PER_SM)
__global__ void tv_grad_sparse_kernel(
        const torch::PackedTensorAccessor32<int32_t, 3, torch::RestrictPtrTraits> links,
        const torch::PackedTensorAccessor64<float, 2, torch::RestrictPtrTraits> data,
        const int32_t* __restrict__ rand_cells,
        int start_dim, int end_dim,
        float scale,
        size_t Q,
        bool ignore_edge,
        bool ignore_last_z,
        float ndc_coeffx, float ndc_coeffy,
        // Output
        bool* __restrict__ mask_out,
        float* __restrict__ grad_data) {
    CUDA_GET_THREAD_ID_U64(tid, Q);
    const int idx = tid % (end_dim - start_dim) + start_dim;
    const int xyz = rand_cells[tid / (end_dim - start_dim)];
    const int z = xyz % links.size(2);
    const int xy = xyz / links.size(2);
    const int y = xy % links.size(1);
    const int x = xy / links.size(1);

    const int32_t* __restrict__ links_ptr = &links[x][y][z];

    if (ignore_edge && *links_ptr == 0) return;

    float scaling[3];
    CALCULATE_RAY_SCALE(scaling, links.size(0), links.size(1), links.size(2));

    const int offx = links.stride(0), offy = links.stride(1);

    const auto lnk000 = links_ptr[0];
    const auto lnk001 = ((z + 1 < links.size(2)) &&
                         (!ignore_last_z || z != links.size(2) - 2)) ?
                        links_ptr[1] : 0;
    const auto lnk010 = y + 1 < links.size(1) ? links_ptr[offy] : 0;
    const auto lnk100 = x + 1 < links.size(0) ? links_ptr[offx] : 0;
    if (ignore_last_z && z == links.size(2) - 2) return;

    const float v000 = lnk000 >= 0 ? data[lnk000][idx] : 0.f;
    const float null_val = (ignore_edge ? v000 : 0.f);
    const float v001 = lnk001 >= 0 ? data[lnk001][idx] : null_val,
                v010 = lnk010 >= 0 ? data[lnk010][idx] : null_val,
                v100 = lnk100 >= 0 ? data[lnk100][idx] : null_val;

    float dx = (v100 - v000);
    float dy = (v010 - v000);
    float dz = (v001 - v000);
    const float idelta = scale * rsqrtf(1e-9f + dx * dx + dy * dy + dz * dz);

    dx *= scaling[0];
    dy *= scaling[1];
    dz *= scaling[2];

#define MAYBE_ADD_SET(lnk, val) if (lnk >= 0 && val != 0.f) { \
    atomicAdd(&grad_data[lnk * data.size(1) + idx], val * idelta); \
    if (mask_out != nullptr) { \
        mask_out[lnk] = true; \
    } \
} \

    const float sm = -(dx + dy + dz);
    MAYBE_ADD_SET(lnk000, sm);
    MAYBE_ADD_SET(lnk001, dz);
    MAYBE_ADD_SET(lnk010, dy);
    MAYBE_ADD_SET(lnk100, dx);

#undef MAYBE_ADD_SET
}

__launch_bounds__(TV_GRAD_CUDA_THREADS, MIN_BLOCKS_PER_SM)
__global__ void msi_tv_grad_sparse_kernel(
        // (reso * 2, reso)
        const torch::PackedTensorAccessor32<int32_t, 2, torch::RestrictPtrTraits> links,
        // (capacity, n_layers, n_channels)
        const torch::PackedTensorAccessor32<float, 3, torch::RestrictPtrTraits> msi,
        const int32_t* __restrict__ rand_cells,
        float scale,
        float scale_last,
        size_t Q,
        // Output
        torch::PackedTensorAccessor32<bool, 2, torch::RestrictPtrTraits> msi_mask,
        torch::PackedTensorAccessor32<float, 3, torch::RestrictPtrTraits> grad_msi) {
    CUDA_GET_THREAD_ID_U64(tid, Q);
    const int MSI_DATA_DIM = msi.size(2);
    const int channel_id = tid % MSI_DATA_DIM;
    const int msi_idx = rand_cells[tid / MSI_DATA_DIM];

    const int z = msi_idx % msi.size(1);
    int tmp = msi_idx / msi.size(1);

    const int y = tmp % links.size(1);
    const int x = tmp / links.size(1);

    const int nx = (x == links.size(0) - 1) ? 0 : x + 1;
    const int ny = (y == links.size(1) - 1) ? 0 : y + 1;

    const int lnk00 = links[x][y];
    const int lnk01 = links[x][ny];
    const int lnk10 = links[nx][y];

    const float v00 = lnk00 >= 0 ? msi[lnk00][z][channel_id] : 0.f;
    const float v_nxl = (lnk00 >= 0 && z + 1 < msi.size(1)) ? msi[lnk00][z + 1][channel_id] : ((channel_id == MSI_DATA_DIM - 1) ? 0.f : v00);
    const float v01 = lnk01 >= 0 ? msi[lnk01][z][channel_id] : 0.f;
    const float v10 = lnk10 >= 0 ? msi[lnk10][z][channel_id] : 0.f;

    if (channel_id == MSI_DATA_DIM - 1) {
        scale = scale_last;
    }

    float dx = (v10 - v00);
    float dy = (v01 - v00);
    float dz = (v_nxl - v00);
    const float idelta = scale * rsqrtf(1e-9f + dx * dx + dy * dy + dz * dz);
    // printf("x=%d y=%d z=%d nx=%d ny=%d dx=%f dy=%f dz=%f scale=%f\n", x, y, z,
    //        nx, ny, dx, dy, dz, scale);

    // const float msi_nlayers = msi.size(1);

    // const float radius = msi_nlayers / (msi_nlayers - z - 0.5f);
    // const float nxl_radius = msi_nlayers / (msi_nlayers - z - 1.5f);
    // const float invr = 1.f / radius;
    // float coord00[3], coord01[3], coord10[3];
    // _equirect2unitvec(x, y, links.size(1), coord00);
    // _equirect2unitvec(x, ny, links.size(1), coord01);
    // _equirect2unitvec(nx, y, links.size(1), coord10);
    // printf("r=%f nlr=%f coord00[%f %f %f] coord01[%f %f %f] coord10[%f %f %f]\n",
    //         radius, nxl_radius,
    //         coord00[0], coord00[1], coord00[2],
    //         coord01[0], coord01[1], coord01[2],
    //         coord10[0], coord10[1], coord10[2]);

    // xsuby3d(coord01, coord00);
    // xsuby3d(coord10, coord00);
    // dx *= _rnorm(coord10) * invr;
    // dy *= _rnorm(coord01) * invr;
    // dz *= 1.f / (nxl_radius - radius);
    dx *= links.size(0) * (1.f / 256.f);
    dy *= links.size(1) * (1.f / 256.f);
    dz *= msi.size(1) * (1.f / 256.f);

#define MAYBE_ADD_SET(link, zz, val) if (link >= 0 && val != 0.f) { \
    atomicAdd(&grad_msi[link][zz][channel_id], val * idelta); \
    if (msi_mask.size(0) > 0) \
        msi_mask[link][zz] = true; \
} \

    const float sm = -(dx + dy + dz);
    MAYBE_ADD_SET(lnk00, z, sm);
    if (z + 1 < msi.size(1)) {
        MAYBE_ADD_SET(lnk00, z + 1, dz);
    }
    MAYBE_ADD_SET(lnk01, z, dy);
    MAYBE_ADD_SET(lnk10, z, dx);
#undef MAYBE_ADD_SET
}

__launch_bounds__(TV_GRAD_CUDA_THREADS, MIN_BLOCKS_PER_SM)
__global__ void lumisphere_tv_grad_sparse_kernel(
        const PackedSparseGridSpec grid,
        const int32_t* __restrict__ rand_cells,
        const float* __restrict__ sphfunc_val,
        const float* __restrict__ sphfunc_val_u,
        float scale,
        size_t Q,
        float ndc_coeffx,
        float ndc_coeffy,
        float dir_factor,
        // Output
        PackedGridOutputGrads grads
        ) {
    CUDA_GET_THREAD_ID_U64(tid, Q);
    const int lane_id = tid & 0x1F;
    if (lane_id >= grid.sh_data_dim) return;
    const int point_id = tid >> 5;
    const int point_blk_id = threadIdx.x >> 5;

    const uint32_t lane_colorgrp_id = lane_id % grid.basis_dim;
    const uint32_t lane_colorgrp = lane_id / grid.basis_dim;

    const int idx = lane_id;

    const int xyz = rand_cells[point_id];
    const int z = xyz % (grid.size[2] - 1);
    const int xy = xyz / (grid.size[2] - 1);
    const int y = xy % (grid.size[1] - 1);
    const int x = xy / (grid.size[1] - 1);

    // __shared__ float grad_sphfunc_val[TV_GRAD_POINTS_PER_BLOCK][10];
    // __shared__ float grad_sphfunc_val_u[TV_GRAD_POINTS_PER_BLOCK][10];
    __shared__ typename WarpReducef::TempStorage temp_storage[TV_GRAD_POINTS_PER_BLOCK];

    uint32_t use_mask = (1U << grid.sh_data_dim) - 1;

    // Currently, will not work for MLP
    __syncwarp(use_mask);

    const int32_t* __restrict__ links_ptr = grid.links +
                         (x * grid.stride_x + y * grid.size[2] + z);

    if (*links_ptr == 0) return;

    float scaling[3];
    CALCULATE_RAY_SCALE(scaling, grid.size[0], grid.size[1], grid.size[2]);

    const int offx = grid.stride_x, offy = grid.size[2];

    const float v000 = links_ptr[0] >= 0 ? grid.sh_data[
                    links_ptr[0] * grid.sh_data_dim + idx] : 0.f;
    const float v001 = links_ptr[1] >= 0 ? grid.sh_data[
                    links_ptr[1] * grid.sh_data_dim + idx] : v000,
                v010 = links_ptr[offy] >= 0 ? grid.sh_data[
                    links_ptr[offy] * grid.sh_data_dim + idx] : v000,
                v100 = links_ptr[offx] >= 0 ? grid.sh_data[
                    links_ptr[offx] * grid.sh_data_dim + idx] : v000;

    const float sv = sphfunc_val[lane_colorgrp_id];
    const float v000a = v000 * sv,
                v001a = v001 * sv,
                v010a = v010 * sv,
                v100a = v100 * sv;
    const float v000u = v000 * sphfunc_val_u[lane_colorgrp_id];

    const bool is_leader = lane_colorgrp_id == 0;
    float v000a_sum = WarpReducef(temp_storage[point_blk_id]).HeadSegmentedSum(
                            v000a, is_leader);
    float v001a_sum = WarpReducef(temp_storage[point_blk_id]).HeadSegmentedSum(
                            v001a, is_leader);
    float v010a_sum = WarpReducef(temp_storage[point_blk_id]).HeadSegmentedSum(
                            v010a, is_leader);
    float v100a_sum = WarpReducef(temp_storage[point_blk_id]).HeadSegmentedSum(
                            v100a, is_leader);
    float v000u_sum = WarpReducef(temp_storage[point_blk_id]).HeadSegmentedSum(
                            v000u, is_leader);

    const float scale_u = dir_factor;

    float dx = (v100a_sum - v000a_sum) * scaling[0];
    float dy = (v010a_sum - v000a_sum) * scaling[1];
    float dz = (v001a_sum - v000a_sum) * scaling[2];
    float du = (v000u_sum - v000a_sum) * scale_u;

    int leader_id = lane_colorgrp * grid.basis_dim;
    dx = __shfl_sync(use_mask, dx, leader_id);
    dy = __shfl_sync(use_mask, dy, leader_id);
    dz = __shfl_sync(use_mask, dz, leader_id);
    du = __shfl_sync(use_mask, du, leader_id);

    const float idelta = scale * rsqrtf(1e-9f + dx * dx + dy * dy + dz * dz + du * du);

    dx *= scaling[0];
    dy *= scaling[1];
    dz *= scaling[2];
    du *= scale_u;

#define MAYBE_ADD_SET(gp, val) if (links_ptr[gp] >= 0 && val != 0.f) { \
    atomicAdd(&grads.grad_sh_out[links_ptr[gp] * grid.sh_data_dim + idx], val * idelta); \
    if (grads.mask_out != nullptr) { \
        grads.mask_out[links_ptr[gp]] = true; \
    } \
} \

    const float sm = -dx * sv - dy * sv - dz * sv +
                      du * (sphfunc_val_u[lane_colorgrp_id] - sv);
    MAYBE_ADD_SET(0, sm);
    MAYBE_ADD_SET(1, dz * sv);
    MAYBE_ADD_SET(offy, dy * sv);
    MAYBE_ADD_SET(offx, dx * sv);

#undef MAYBE_ADD_SET

    // TODO
    // __syncwarp(use_mask);
    // if (lane_id < grid.basis_dim) {
    //     calc_sphfunc_backward(
    //             grid,
    //             lane_id,
    //             point_id,
    //             dir,
    //             sphfunc_val[point_blk_id],
    //             grad_sphfunc_val_v[point_blk_id],
    //             grad_basis_out);
    //     calc_sphfunc_backward(
    //             grid,
    //             lane_id,
    //             point_id,
    //             dir_u,
    //             sphfunc_val_u[point_blk_id],
    //             grad_sphfunc_val[point_blk_id],
    //             grad_basis_out);
    //     calc_sphfunc_backward(
    //             grid,
    //             lane_id,
    //             point_id,
    //             dir_v,
    //             sphfunc_val_v[point_blk_id],
    //             grad_sphfunc_val_v[point_blk_id],
    //             grad_basis_out);
    // }
}

}  // namespace device
}  // namespace


torch::Tensor tv(torch::Tensor links, torch::Tensor data,
                 int start_dim, int end_dim,
                 bool use_logalpha,
                 float logalpha_delta,
                 bool ignore_edge,
                 float ndc_coeffx,
                 float ndc_coeffy) {
    DEVICE_GUARD(data);
    CHECK_INPUT(data);
    CHECK_INPUT(links);
    TORCH_CHECK(data.is_floating_point());
    TORCH_CHECK(!links.is_floating_point());
    TORCH_CHECK(data.ndimension() == 2);
    TORCH_CHECK(links.ndimension() == 3);

    int nl = (links.size(0) - 1) * (links.size(1) - 1) * (links.size(2) - 1);
    size_t Q = nl * size_t(end_dim - start_dim);

    const int blocks = CUDA_N_BLOCKS_NEEDED(Q, TV_GRAD_CUDA_THREADS);
    torch::Tensor result = torch::zeros({}, data.options());
    device::tv_kernel<<<blocks, TV_GRAD_CUDA_THREADS>>>(
            links.packed_accessor32<int32_t, 3, torch::RestrictPtrTraits>(),
            data.packed_accessor64<float, 2, torch::RestrictPtrTraits>(),
            start_dim,
            end_dim,
            1.f / nl,
            Q,
            ignore_edge,
            ndc_coeffx, ndc_coeffy,
            // Output
            result.data_ptr<float>());
    CUDA_CHECK_ERRORS;
    return result;
}

void tv_grad(torch::Tensor links,
             torch::Tensor data,
             int start_dim, int end_dim,
             float scale,
             bool use_logalpha,
             float logalpha_delta,
             bool ignore_edge,
             float ndc_coeffx,
             float ndc_coeffy,
             torch::Tensor grad_data) {
    DEVICE_GUARD(data);
    CHECK_INPUT(data);
    CHECK_INPUT(links);
    CHECK_INPUT(grad_data);
    TORCH_CHECK(data.is_floating_point());
    TORCH_CHECK(grad_data.is_floating_point());
    TORCH_CHECK(!links.is_floating_point());
    TORCH_CHECK(data.ndimension() == 2);
    TORCH_CHECK(links.ndimension() == 3);
    TORCH_CHECK(grad_data.ndimension() == 2);

    int nl = (links.size(0) - 1) * (links.size(1) - 1) * (links.size(2) - 1);
    size_t Q = nl * size_t(end_dim - start_dim);

    const int cuda_n_threads = TV_GRAD_CUDA_THREADS;
    const int blocks = CUDA_N_BLOCKS_NEEDED(Q, cuda_n_threads);
    device::tv_grad_kernel<<<blocks, cuda_n_threads>>>(
            links.packed_accessor32<int32_t, 3, torch::RestrictPtrTraits>(),
            data.packed_accessor64<float, 2, torch::RestrictPtrTraits>(),
            start_dim,
            end_dim,
            scale / nl,
            Q,
            ignore_edge,
            ndc_coeffx, ndc_coeffy,
            // Output
            grad_data.data_ptr<float>());
    CUDA_CHECK_ERRORS;
}

void tv_grad_sparse(torch::Tensor links,
             torch::Tensor data,
             torch::Tensor rand_cells,
             torch::Tensor mask_out,
             int start_dim, int end_dim,
             float scale,
             bool use_logalpha,
             float logalpha_delta,
             bool ignore_edge,
             bool ignore_last_z,
             float ndc_coeffx,
             float ndc_coeffy,
             torch::Tensor grad_data) {
    DEVICE_GUARD(data);
    CHECK_INPUT(data);
    CHECK_INPUT(links);
    CHECK_INPUT(grad_data);
    CHECK_INPUT(rand_cells);
    CHECK_INPUT(mask_out);
    TORCH_CHECK(data.is_floating_point());
    TORCH_CHECK(grad_data.is_floating_point());
    TORCH_CHECK(!links.is_floating_point());
    TORCH_CHECK(data.ndimension() == 2);
    TORCH_CHECK(links.ndimension() == 3);
    TORCH_CHECK(grad_data.ndimension() == 2);

    int nl = rand_cells.size(0);
    size_t Q = rand_cells.size(0) * size_t(end_dim - start_dim);

    const int cuda_n_threads = TV_GRAD_CUDA_THREADS;
    const int blocks = CUDA_N_BLOCKS_NEEDED(Q, cuda_n_threads);
    device::tv_grad_sparse_kernel<<<blocks, cuda_n_threads>>>(
            links.packed_accessor32<int32_t, 3, torch::RestrictPtrTraits>(),
            data.packed_accessor64<float, 2, torch::RestrictPtrTraits>(),
            rand_cells.data_ptr<int32_t>(),
            start_dim,
            end_dim,
            scale / nl,
            Q,
            ignore_edge,
            ignore_last_z,
            ndc_coeffx, ndc_coeffy,
            // Output
            (mask_out.dim() > 0) ? mask_out.data_ptr<bool>() : nullptr,
            grad_data.data_ptr<float>());
    CUDA_CHECK_ERRORS;
}

void msi_tv_grad_sparse(
             // (reso * 2, reso)
             torch::Tensor links,
             // (capacity, n_layers, n_channels)
             torch::Tensor msi,
             torch::Tensor rand_cells,
             torch::Tensor mask_out,
             float scale,
             float scale_last,
             torch::Tensor grad_msi) {
    DEVICE_GUARD(msi);
    CHECK_INPUT(links);
    CHECK_INPUT(msi);
    CHECK_INPUT(grad_msi);
    CHECK_INPUT(rand_cells);
    CHECK_INPUT(mask_out);
    TORCH_CHECK(msi.is_floating_point());
    TORCH_CHECK(grad_msi.is_floating_point());

    int nl = rand_cells.size(0);
    size_t Q = rand_cells.size(0) * msi.size(2);

    const int cuda_n_threads = TV_GRAD_CUDA_THREADS;
    const int blocks = CUDA_N_BLOCKS_NEEDED(Q, cuda_n_threads);
    device::msi_tv_grad_sparse_kernel<<<blocks, cuda_n_threads>>>(
            links.packed_accessor32<int32_t, 2, torch::RestrictPtrTraits>(),
            msi.packed_accessor32<float, 3, torch::RestrictPtrTraits>(),
            rand_cells.data_ptr<int32_t>(),
            scale / nl,
            scale_last / nl,
            Q,
            // Output
            mask_out.packed_accessor32<bool, 2, torch::RestrictPtrTraits>(),
            grad_msi.packed_accessor32<float, 3, torch::RestrictPtrTraits>());
    CUDA_CHECK_ERRORS;
}

void lumisphere_tv_grad_sparse(
             SparseGridSpec& grid,
             torch::Tensor rand_cells,
             torch::Tensor basis_fn,
             torch::Tensor basis_fn_u,
             float scale,
             float ndc_coeffx,
             float ndc_coeffy,
             float dir_factor,
             GridOutputGrads& grads) {
    DEVICE_GUARD(grid.sh_data);
    CHECK_INPUT(rand_cells);
    CHECK_INPUT(basis_fn);
    CHECK_INPUT(basis_fn_u);
    TORCH_CHECK(basis_fn.ndimension() == 1);
    grid.check();
    grads.check();

    int nl = rand_cells.size(0);
    size_t Q = rand_cells.size(0) * WARP_SIZE;

    const int cuda_n_threads = TV_GRAD_CUDA_THREADS;
    const int blocks = CUDA_N_BLOCKS_NEEDED(Q, cuda_n_threads);
    device::lumisphere_tv_grad_sparse_kernel<<<blocks, cuda_n_threads>>>(
            grid,
            rand_cells.data_ptr<int32_t>(),
            basis_fn.data_ptr<float>(),
            basis_fn_u.data_ptr<float>(),
            scale / nl,
            Q,
            ndc_coeffx, ndc_coeffy,
            dir_factor,
            // Output
            grads);
    CUDA_CHECK_ERRORS;
}
