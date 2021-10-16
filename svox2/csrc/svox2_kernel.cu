// Copyright 2021 Alex Yu
#include <torch/extension.h>
#include <cstdint>
#include "cuda_util.cuh"
#include "data_spec_packed.cuh"

namespace device {

__global__ void sample_grid_kernel(
        PackedSparseGridSpec grid,
        const torch::PackedTensorAccessor32<float, 2, torch::RestrictPtrTraits> points,
        // Output
        torch::PackedTensorAccessor32<float, 2, torch::RestrictPtrTraits> out) {
    CUDA_GET_THREAD_ID(tid, points.size(0) * grid.data.size(1));
    const int idx = tid % grid.data.size(1);
    const int pid = tid / grid.data.size(1);

    float point[3] = {points[pid][0], points[pid][1], points[pid][2]};
    transform_coord(point, grid._scaling, grid._offset);

    int32_t l[3];
#pragma unroll 3
    for (int i = 0; i < 3; ++i) {
        point[i] = fminf(fmaxf(point[i], 0.f), grid.links.size(i) - 1.f);
        l[i] = min((int32_t)point[i], (int32_t)(grid.links.size(i) - 2));
        point[i] -= l[i];
    }

    const int32_t* __restrict__ link_ptr = &grid.links[l[0]][l[1]][l[2]];

    const int offy = grid.links.stride(1), offx = grid.links.stride(0);

#define MAYBE_READ_LINK(u) ((link_ptr[u] >= 0) ? grid.data[link_ptr[u]][idx] : 0.f)

    const float ix0y0 = lerp(MAYBE_READ_LINK(0), MAYBE_READ_LINK(1), point[2]);
    const float ix0y1 = lerp(MAYBE_READ_LINK(offy), MAYBE_READ_LINK(offy + 1), point[2]);
    const float ix0 = lerp(ix0y0, ix0y1, point[1]);
    const float ix1y0 = lerp(MAYBE_READ_LINK(offx), MAYBE_READ_LINK(offx + 1), point[2]);
    const float ix1y1 = lerp(MAYBE_READ_LINK(offy + offx),
                             MAYBE_READ_LINK(offy + offx + 1), point[2]);
    const float ix1 = lerp(ix1y0, ix1y1, point[1]);
    out[pid][idx] = lerp(ix0, ix1, point[0]);
}

__global__ void sample_grid_backward_kernel(
        PackedSparseGridSpec grid,
        const torch::PackedTensorAccessor32<float, 2, torch::RestrictPtrTraits> points,
        const torch::PackedTensorAccessor32<float, 2, torch::RestrictPtrTraits> grad_out,
        // Output
        torch::PackedTensorAccessor64<float, 2, torch::RestrictPtrTraits> grad_data) {
    CUDA_GET_THREAD_ID(tid, points.size(0) * grid.data.size(1));
    const int idx = tid % grid.data.size(1);
    const int pid = tid / grid.data.size(1);

    float point[3] = {points[pid][0], points[pid][1], points[pid][2]};
    transform_coord(point, grid._scaling, grid._offset);

    int32_t l[3];
#pragma unroll 3
    for (int i = 0; i < 3; ++i) {
        point[i] = fminf(fmaxf(point[i], 0.f), grid.links.size(i) - 1.f);
        l[i] = min((int32_t)point[i], (int32_t)(grid.links.size(i) - 2));
        point[i] -= l[i];
    }

    const int32_t* __restrict__ link_ptr = &grid.links[l[0]][l[1]][l[2]];

    const int offy = grid.links.stride(1), offx = grid.links.stride(0);
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
}  // namespace device


torch::Tensor sample_grid(SparseGridSpec& grid, torch::Tensor points) {
    DEVICE_GUARD(points);
    grid.check();
    CHECK_INPUT(points);
    TORCH_CHECK(points.ndimension() == 2);
    const auto Q = points.size(0) * grid.data.size(1);
    const int cuda_n_threads = std::min<int>(Q, CUDA_MAX_THREADS);
    const int blocks = CUDA_N_BLOCKS_NEEDED(Q, cuda_n_threads);
    torch::Tensor result = torch::empty({points.size(0),
                        grid.data.size(1)}, points.options());

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
    const auto Q = points.size(0) * grid.data.size(1);

    const int cuda_n_threads = std::min<int>(Q, CUDA_MAX_THREADS);
    const int blocks = CUDA_N_BLOCKS_NEEDED(Q, cuda_n_threads);

    torch::Tensor grad_data = torch::zeros({grid.data.requires_grad() ? grid.data.size(0) : 0,
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
