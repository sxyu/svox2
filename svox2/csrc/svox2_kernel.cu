// Copyright 2021 Alex Yu
#include <torch/extension.h>
#include <cstdint>
#include "cuda_util.cuh"
#include "data_spec_packed.cuh"

namespace device {

__device__ __inline__ void trilerp(
        const torch::PackedTensorAccessor64<float, 2, torch::RestrictPtrTraits> data,
        const torch::PackedTensorAccessor32<int32_t, 3, torch::RestrictPtrTraits> links,
        float* __restrict__ pos,
        float* __restrict__ out) {
    int32_t l[3];

#pragma unroll 3
    for (int i = 0; i < 3; ++i) {
        pos[i] = min(max(pos[i], 0.f), links.size(i) - 1.f);
        l[i] = min((int32_t)pos[i], (int32_t)(links.size(i) - 2));
        pos[i] -= l[i];
    }
    const int32_t* __restrict__ link_ptr = &links[l[0]][l[1]][l[2]];
    const int offy = links.stride(1), offx = links.stride(0);
    const int data_dim = data.size(1);

    float buffer[8][49];
    const float* __restrict__ dptr = data.data();
    _COPY_8_NEIGHBORS(dptr, link_ptr, offx, offy, buffer, data_dim, 0, data_dim);

    _trilerp_buffered(buffer, out, pos, 0, data_dim);
}


__device__ __inline__ void trilerp_backward(
        torch::PackedTensorAccessor32<int32_t, 3, torch::RestrictPtrTraits> links,
        float* __restrict__ pos,
        const float* __restrict__ grad_out,
        torch::PackedTensorAccessor64<float, 2, torch::RestrictPtrTraits> grad_data) {
    int32_t l[3];
#pragma unroll 3
    for (int i = 0; i < 3; ++i) {
        pos[i] = min(max(pos[i], 0.f), links.size(i) - (1.f + 1e-3f));
        l[i] = min((int32_t)pos[i], (int32_t)(links.size(i) - 2));
        pos[i] -= l[i];
    }

    const int data_dim = grad_data.size(1);

    float buffer[8][49];
    float lerp_wt[8];
    _init_lerp_weight(lerp_wt, pos);

#pragma unroll
    for (int j = 0; j < data_dim; ++j) {
#pragma unroll 8
        for (int k = 0; k < 8; ++k)
            buffer[k][j] = lerp_wt[k] * grad_out[j];
    }

    // TODO: candidate for coalesced atomics, although this function is not critical
    const int32_t* __restrict__ link_ptr = &links[l[0]][l[1]][l[2]];
    float* __restrict__ gdptr = grad_data.data();
    const int offy = links.stride(1), offx = links.stride(0);
    _UPDATE_8_NEIGHBORS(gdptr, link_ptr, offx, offy, buffer, data_dim);
}

__global__ void sample_grid_kernel(
        PackedSparseGridSpec grid,
        const torch::PackedTensorAccessor32<float, 2, torch::RestrictPtrTraits> points,
        // Output
        torch::PackedTensorAccessor32<float, 2, torch::RestrictPtrTraits> out) {
    CUDA_GET_THREAD_ID(tid, points.size(0));

    float point[3] = {points[tid][0], points[tid][1], points[tid][2]};
    transform_coord(point, grid._scaling, grid._offset);
    // Destroys point
    trilerp(grid.data, grid.links, point, &out[tid][0]);
}

__global__ void sample_grid_backward_kernel(
        PackedSparseGridSpec grid,
        torch::PackedTensorAccessor32<float, 2, torch::RestrictPtrTraits> points,
        torch::PackedTensorAccessor32<float, 2, torch::RestrictPtrTraits> grad_out,
        // Output
        torch::PackedTensorAccessor64<float, 2, torch::RestrictPtrTraits> grad_data) {
    CUDA_GET_THREAD_ID(tid, points.size(0));
    float point[3] = {points[tid][0], points[tid][1], points[tid][2]};
    transform_coord(point, grid._scaling, grid._offset);
    // Destroys point
    trilerp_backward(grid.links, point, &grad_out[tid][0], grad_data);
}
}  // namespace device


torch::Tensor sample_grid(SparseGridSpec& grid, torch::Tensor points) {
    DEVICE_GUARD(points);
    grid.check();
    CHECK_INPUT(points);
    TORCH_CHECK(points.ndimension() == 2);
    const auto Q = points.size(0);
    const int cuda_n_threads = std::min<int>(Q, CUDA_MAX_THREADS);
    const int blocks = CUDA_N_BLOCKS_NEEDED(Q, cuda_n_threads);
    torch::Tensor result = torch::empty({Q, grid.data.size(1)}, points.options());

    device::sample_grid_kernel<<<blocks, cuda_n_threads>>>(
            grid,
            points.packed_accessor32<float, 2, torch::RestrictPtrTraits>(),
            // Output
            result.packed_accessor32<float, 2, torch::RestrictPtrTraits>());
    CUDA_CHECK_ERRORS;
    return result;
}

torch::Tensor sample_grid_backward(
        SparseGridSpec& grid,
        torch::Tensor points,
        torch::Tensor grad_out) {
    DEVICE_GUARD(points);
    grid.check();
    CHECK_INPUT(points);
    CHECK_INPUT(grad_out);
    TORCH_CHECK(points.ndimension() == 2);
    TORCH_CHECK(grad_out.ndimension() == 2);
    const auto Q = points.size(0);

    const int cuda_n_threads = std::min<int>(Q, CUDA_MAX_THREADS);
    const int blocks = CUDA_N_BLOCKS_NEEDED(Q, cuda_n_threads);

    torch::Tensor grad_data = torch::empty({grid.data.requires_grad() ? grid.data.size(0) : 0,
                                             grid.data.size(1)},
                                             points.options());

    device::sample_grid_backward_kernel<<<blocks, cuda_n_threads>>>(
            grid,
            points.packed_accessor32<float, 2, torch::RestrictPtrTraits>(),
            grad_out.packed_accessor32<float, 2, torch::RestrictPtrTraits>(),
            // Output
            grad_data.packed_accessor64<float, 2, torch::RestrictPtrTraits>());
    CUDA_CHECK_ERRORS;
    return grad_data;
}
