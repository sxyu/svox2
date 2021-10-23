// Copyright 2021 Alex Yu
#pragma once
#include <cuda_runtime.h>
#include <cub/cub.cuh>
#include <c10/cuda/CUDAGuard.h>
#include <ATen/cuda/CUDAContext.h>
#include "util.hpp"


#define DEVICE_GUARD(_ten) \
    const at::cuda::OptionalCUDAGuard device_guard(device_of(_ten));

#define CUDA_GET_THREAD_ID(tid, Q) const int tid = blockIdx.x * blockDim.x + threadIdx.x; \
                      if (tid >= Q) return
#define CUDA_GET_THREAD_ID_U64(tid, Q) const size_t tid = blockIdx.x * blockDim.x + threadIdx.x; \
                      if (tid >= Q) return
#define CUDA_N_BLOCKS_NEEDED(Q, CUDA_N_THREADS) ((Q - 1) / CUDA_N_THREADS + 1)
#define CUDA_CHECK_ERRORS \
    cudaError_t err = cudaGetLastError(); \
    if (err != cudaSuccess) \
            printf("Error in svox2.%s : %s\n", __FUNCTION__, cudaGetErrorString(err))

#define CUDA_MAX_THREADS at::cuda::getCurrentDeviceProperties()->maxThreadsPerBlock

#if !defined(__CUDA_ARCH__) || __CUDA_ARCH__ >= 600
#else
__device__ inline double atomicAdd(double* address, double val){
    unsigned long long int* address_as_ull = (unsigned long long int*)address;
    unsigned long long int old = *address_as_ull, assumed;
    do {
        assumed = old;
        old = atomicCAS(address_as_ull, assumed,
                __double_as_longlong(val + __longlong_as_double(assumed)));
    } while (assumed != old);
    return __longlong_as_double(old);
}
#endif

__device__ inline void atomicMax(float* result, float value){
    unsigned* result_as_u = (unsigned*)result;
    unsigned old = *result_as_u, assumed;
    do {
        assumed = old;
        old = atomicCAS(result_as_u, assumed,
                __float_as_int(fmaxf(value, __int_as_float(assumed))));
    } while (old != assumed);
    return;
}

__device__ inline void atomicMax(double* result, double value){
    unsigned long long int* result_as_ull = (unsigned long long int*)result;
    unsigned long long int old = *result_as_ull, assumed;
    do {
        assumed = old;
        old = atomicCAS(result_as_ull, assumed,
                __double_as_longlong(fmaxf(value, __longlong_as_double(assumed))));
    } while (old != assumed);
    return;
}

__device__ __inline__ void transform_coord(float* __restrict__ point,
                                           const float* __restrict__ scaling,
                                           const float* __restrict__ offset) {
    point[0] = fmaf(point[0], scaling[0], offset[0]); // a*b + c
    point[1] = fmaf(point[1], scaling[1], offset[1]); // a*b + c
    point[2] = fmaf(point[2], scaling[2], offset[2]); // a*b + c
}

// Linear interp
// Subtract and fused multiply-add
// (1-w) a + w b
template<class T>
__device__ __inline__ T lerp(T a, T b, T w) {
    return fmaf(w, b - a, a);
}
