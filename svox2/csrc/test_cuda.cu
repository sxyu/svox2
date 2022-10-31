
#include <torch/extension.h>
#include "cuda_util.cuh"
#include "data_spec_packed.cuh"
#include "render_util.cuh"

#include <iostream>
#include <cstdint>
#include <tuple>
// #include <math.h>
#include <bits/stdc++.h>
#include <assert.h>

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

__launch_bounds__(TRACE_RAY_CUDA_THREADS, MIN_BLOCKS_PER_SM)
__global__ void test_cubic_root_grad_kernel(
    // float* __restrict__ fs,
    // const float grad_in,
    // torch::PackedTensorAccessor32<float, 2, torch::RestrictPtrTraits> grad_fs
    // const int st_id,
    // float* __restrict__ grad_fs
    torch::PackedTensorAccessor32<float, 2, torch::RestrictPtrTraits> T_fs,
    torch::PackedTensorAccessor32<int, 1, torch::RestrictPtrTraits> T_st_id,
    torch::PackedTensorAccessor32<float, 2, torch::RestrictPtrTraits> T_grad_fs
) {

    CUDA_GET_THREAD_ID(tid, int(T_fs.size(0)) * WARP_SIZE);
    const int ray_id = tid >> 5; // same as / 32, which is the WARP_SIZE
    const int ray_blk_id = threadIdx.x >> 5; // difference between tid and threadIdx.x? --> tid is the total id (batch/ray id)
    const int lane_id = threadIdx.x & 0x1F; // take only last 5 digits

    if (lane_id >= 1)  // Bad, but currently the best way due to coalesced memory access
        return;

    float* __restrict__ fs = T_fs[ray_id].data();
    const int st_id = T_st_id[ray_id];
    float* __restrict__ grad_fs = T_grad_fs[ray_id].data();

    float st[3] = {-1, -1, -1}; // sample t
    enum BasisType const cubic_root_type = cubic_equation_solver(
        fs[0], fs[1], fs[2], fs[3],
        1e-8, // float eps
        1e-10, // double eps
        st
    );

    if (cubic_root_type == CUBIC_TYPE_NO_ROOT){
        grad_fs[0] = 0;
        grad_fs[1] = 0;
        grad_fs[2] = 0;
        grad_fs[3] = 0;
        return;
    }
    int root_num = 1;
    if (cubic_root_type == CUBIC_TYPE_POLY){
        root_num = 2;
    } else if (cubic_root_type == CUBIC_TYPE_CUBIC_THREE_R){
        root_num = 3;
    }

    if (st_id >= root_num){
        grad_fs[0] = 0;
        grad_fs[1] = 0;
        grad_fs[2] = 0;
        grad_fs[3] = 0;
        return;
    }

    calc_cubic_root_grad(cubic_root_type, st_id, fs, grad_fs);
}

}

}  // namespace device

torch::Tensor test_cubic_root_grad(Tensor T_fs, Tensor T_st_ids, Tensor T_grad_fs) {
    const auto Q = T_fs.size(0);

        const int cuda_n_threads = TRACE_RAY_CUDA_THREADS;
        const int blocks = CUDA_N_BLOCKS_NEEDED(Q * WARP_SIZE, cuda_n_threads);
        device::test_cubic_root_grad_kernel<<<blocks, cuda_n_threads>>>(
                T_fs.packed_accessor32<float, 2, torch::RestrictPtrTraits>(),
                T_st_ids.packed_accessor32<int, 1, torch::RestrictPtrTraits>(),
                // Output
                T_grad_fs.packed_accessor32<float, 2, torch::RestrictPtrTraits>());


    return T_grad_fs;
}