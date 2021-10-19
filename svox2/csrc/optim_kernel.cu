// Copyright 2021 Alex Yu
// Optimizer-related kernels

#include <torch/extension.h>
#include "cuda_util.cuh"

namespace {

const int RMSPROP_STEP_CUDA_THREADS = 256;
const int MIN_BLOCKS_PER_SM = 4;

namespace device {

// RMSPROP
__inline__ __device__ void rmsprop_once(
        float* __restrict__ ptr_data,
        float* __restrict__ ptr_rms,
        float* __restrict__ ptr_grad, 
        const float beta, const float lr, const float epsilon) {
    float rms = lerp(_SQR(*ptr_grad), *ptr_rms, beta);
    *ptr_rms = rms;
    *ptr_data -= lr * (*ptr_grad) / (sqrtf(rms) + epsilon);
    *ptr_grad = 0.f;
}

__launch_bounds__(RMSPROP_STEP_CUDA_THREADS, MIN_BLOCKS_PER_SM)
__global__ void rmsprop_step_kernel(
        torch::PackedTensorAccessor64<float, 2, torch::RestrictPtrTraits> all_data,
        torch::PackedTensorAccessor64<float, 2, torch::RestrictPtrTraits> all_rms,
        torch::PackedTensorAccessor64<float, 2, torch::RestrictPtrTraits> all_grad,
        float beta,
        float lr,
        float epsilon) {
    CUDA_GET_THREAD_ID(tid, all_data.size(0) * all_data.size(1));
    rmsprop_once(all_data.data() + tid, all_rms.data() + tid,
                 all_grad.data() + tid,
                 beta,
                 lr,
                 epsilon);
}


__launch_bounds__(RMSPROP_STEP_CUDA_THREADS, MIN_BLOCKS_PER_SM)
__global__ void rmsprop_mask_step_kernel(
        torch::PackedTensorAccessor64<float, 2, torch::RestrictPtrTraits> all_data,
        torch::PackedTensorAccessor64<float, 2, torch::RestrictPtrTraits> all_rms,
        torch::PackedTensorAccessor64<float, 2, torch::RestrictPtrTraits> all_grad,
        const bool* __restrict__ mask,
        float beta,
        float lr,
        float epsilon) {
    CUDA_GET_THREAD_ID(tid, all_data.size(0) * all_data.size(1));
    if (mask != nullptr && mask[tid / all_data.size(1)] == false) return;
    rmsprop_once(all_data.data() + tid, all_rms.data() + tid,
                 all_grad.data() + tid,
                 beta,
                 lr,
                 epsilon);
}

__launch_bounds__(RMSPROP_STEP_CUDA_THREADS, MIN_BLOCKS_PER_SM)
__global__ void rmsprop_index_step_kernel(
        torch::PackedTensorAccessor64<float, 2, torch::RestrictPtrTraits> all_data,
        torch::PackedTensorAccessor64<float, 2, torch::RestrictPtrTraits> all_rms,
        torch::PackedTensorAccessor64<float, 2, torch::RestrictPtrTraits> all_grad,
        torch::PackedTensorAccessor32<int64_t, 1, torch::RestrictPtrTraits> indices,
        float beta,
        float lr,
        float epsilon) {
    CUDA_GET_THREAD_ID(tid, indices.size(0) * all_data.size(1));
    int32_t i = indices[tid / all_data.size(1)];
    int32_t j = tid % all_data.size(1);
    size_t off = i * all_data.size(1) + j;
    rmsprop_once(all_data.data() + off, all_rms.data() + off,
                 all_grad.data() + off,
                 beta,
                 lr,
                 epsilon);
}


// SGD
__inline__ __device__ void sgd_once(
        float* __restrict__ ptr_data,
        float* __restrict__ ptr_grad, 
        const float lr) {
    *ptr_data -= lr * (*ptr_grad);
    *ptr_grad = 0.f;
}

__launch_bounds__(RMSPROP_STEP_CUDA_THREADS, MIN_BLOCKS_PER_SM)
__global__ void sgd_step_kernel(
        torch::PackedTensorAccessor64<float, 2, torch::RestrictPtrTraits> all_data,
        torch::PackedTensorAccessor64<float, 2, torch::RestrictPtrTraits> all_grad,
        float lr) {
    CUDA_GET_THREAD_ID(tid, all_data.size(0) * all_data.size(1));
    sgd_once(all_data.data() + tid,
             all_grad.data() + tid,
             lr);
}

__launch_bounds__(RMSPROP_STEP_CUDA_THREADS, MIN_BLOCKS_PER_SM)
__global__ void sgd_mask_step_kernel(
        torch::PackedTensorAccessor64<float, 2, torch::RestrictPtrTraits> all_data,
        torch::PackedTensorAccessor64<float, 2, torch::RestrictPtrTraits> all_grad,
        const bool* __restrict__ mask,
        float lr) {
    CUDA_GET_THREAD_ID(tid, all_data.size(0) * all_data.size(1));
    if (mask != nullptr && mask[tid / all_data.size(1)] == false) return;
    sgd_once(all_data.data() + tid,
             all_grad.data() + tid,
             lr);
}

__launch_bounds__(RMSPROP_STEP_CUDA_THREADS, MIN_BLOCKS_PER_SM)
__global__ void sgd_index_step_kernel(
        torch::PackedTensorAccessor64<float, 2, torch::RestrictPtrTraits> all_data,
        torch::PackedTensorAccessor64<float, 2, torch::RestrictPtrTraits> all_grad,
        torch::PackedTensorAccessor32<int64_t, 1, torch::RestrictPtrTraits> indices,
        float lr) {
    CUDA_GET_THREAD_ID(tid, indices.size(0) * all_data.size(1));
    int32_t i = indices[tid / all_data.size(1)];
    int32_t j = tid % all_data.size(1);
    size_t off = i * all_data.size(1) + j;
    sgd_once(all_data.data() + off,
            all_grad.data() + off,
            lr);
}



}  // namespace device
}  // namespace

void rmsprop_step(
        torch::Tensor data,
        torch::Tensor rms,
        torch::Tensor grad,
        torch::Tensor indexer,
        float beta,
        float lr,
        float epsilon) {

    DEVICE_GUARD(data);
    CHECK_INPUT(data);
    CHECK_INPUT(rms);
    CHECK_INPUT(grad);
    CHECK_INPUT(indexer);

    const int cuda_n_threads = RMSPROP_STEP_CUDA_THREADS;

    if (indexer.dim() == 0) {
        const size_t Q = data.size(0) * data.size(1);
        const int blocks = CUDA_N_BLOCKS_NEEDED(Q, cuda_n_threads);
        device::rmsprop_step_kernel<<<blocks, cuda_n_threads>>>(
                data.packed_accessor64<float, 2, torch::RestrictPtrTraits>(),
                rms.packed_accessor64<float, 2, torch::RestrictPtrTraits>(),
                grad.packed_accessor64<float, 2, torch::RestrictPtrTraits>(),
                beta,
                lr,
                epsilon);
    } else if (indexer.size(0) == 0) {
        // Skip
    } else if (indexer.scalar_type() == at::ScalarType::Bool) {
        const size_t Q = data.size(0) * data.size(1);
        const int blocks = CUDA_N_BLOCKS_NEEDED(Q, cuda_n_threads);
        device::rmsprop_mask_step_kernel<<<blocks, cuda_n_threads>>>(
                data.packed_accessor64<float, 2, torch::RestrictPtrTraits>(),
                rms.packed_accessor64<float, 2, torch::RestrictPtrTraits>(),
                grad.packed_accessor64<float, 2, torch::RestrictPtrTraits>(),
                indexer.data_ptr<bool>(),
                beta,
                lr,
                epsilon);
    } else {
        const size_t Q = indexer.size(0) * data.size(1);
        const int blocks = CUDA_N_BLOCKS_NEEDED(Q, cuda_n_threads);
        device::rmsprop_index_step_kernel<<<blocks, cuda_n_threads>>>(
                data.packed_accessor64<float, 2, torch::RestrictPtrTraits>(),
                rms.packed_accessor64<float, 2, torch::RestrictPtrTraits>(),
                grad.packed_accessor64<float, 2, torch::RestrictPtrTraits>(),
                indexer.packed_accessor32<int64_t, 1, torch::RestrictPtrTraits>(),
                beta,
                lr,
                epsilon);
    }

    CUDA_CHECK_ERRORS;
}

void sgd_step(
        torch::Tensor data,
        torch::Tensor grad,
        torch::Tensor indexer,
        float lr) {

    DEVICE_GUARD(data);
    CHECK_INPUT(data);
    CHECK_INPUT(grad);
    CHECK_INPUT(indexer);

    const int cuda_n_threads = RMSPROP_STEP_CUDA_THREADS;

    if (indexer.dim() == 0) {
        const size_t Q = data.size(0) * data.size(1);
        const int blocks = CUDA_N_BLOCKS_NEEDED(Q, cuda_n_threads);
        device::sgd_step_kernel<<<blocks, cuda_n_threads>>>(
                data.packed_accessor64<float, 2, torch::RestrictPtrTraits>(),
                grad.packed_accessor64<float, 2, torch::RestrictPtrTraits>(),
                lr);
    } else if (indexer.size(0) == 0) {
        // Skip
    } else if (indexer.scalar_type() == at::ScalarType::Bool) {
        const size_t Q = data.size(0) * data.size(1);
        const int blocks = CUDA_N_BLOCKS_NEEDED(Q, cuda_n_threads);
        device::sgd_mask_step_kernel<<<blocks, cuda_n_threads>>>(
                data.packed_accessor64<float, 2, torch::RestrictPtrTraits>(),
                grad.packed_accessor64<float, 2, torch::RestrictPtrTraits>(),
                indexer.data_ptr<bool>(),
                lr);
    } else {
        const size_t Q = indexer.size(0) * data.size(1);
        const int blocks = CUDA_N_BLOCKS_NEEDED(Q, cuda_n_threads);
        device::sgd_index_step_kernel<<<blocks, cuda_n_threads>>>(
                data.packed_accessor64<float, 2, torch::RestrictPtrTraits>(),
                grad.packed_accessor64<float, 2, torch::RestrictPtrTraits>(),
                indexer.packed_accessor32<int64_t, 1, torch::RestrictPtrTraits>(),
                lr);
    }

    CUDA_CHECK_ERRORS;
}
