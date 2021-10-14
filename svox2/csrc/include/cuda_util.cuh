// Copyright 2021 Alex Yu
#pragma once
#include <cuda_runtime.h>
#include <c10/cuda/CUDAGuard.h>
#include "util.hpp"


#undef _SIGMOID
// CUDA version of sigmoid
#define _SIGMOID(x) (1 / (1 + __expf(-(x))))

#define DEVICE_GUARD(_ten) \
    const at::cuda::OptionalCUDAGuard device_guard(device_of(_ten));

#define CUDA_GET_THREAD_ID(tid, Q) const int tid = blockIdx.x * blockDim.x + threadIdx.x; \
                      if (tid >= Q) return
#define CUDA_N_BLOCKS_NEEDED(Q, CUDA_N_THREADS) ((Q - 1) / CUDA_N_THREADS + 1)
#define CUDA_CHECK_ERRORS \
    cudaError_t err = cudaGetLastError(); \
    if (err != cudaSuccess) \
            printf("Error in svox.%s : %s\n", __FUNCTION__, cudaGetErrorString(err))

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

#define lerp(a, b, w) (a + w * (b - a))

#define _trilerp_buffered(buf, out, w, start_dim, end_dim) do { \
    float ix0, ix1; \
    for (int j = (start_dim); j < (end_dim); ++j) { \
        { \
            const float ix0y0 = lerp((buf)[0][j], \
                    (buf)[1][j], \
                    (w)[2]); \
            const float ix0y1 = lerp((buf)[2][j], \
                    (buf)[3][j], \
                    (w)[2]); \
            ix0 = lerp(ix0y0, ix0y1, (w)[1]); \
        } \
        { \
            const float ix1y0 = lerp((buf)[4][j], \
                    (buf)[5][j], \
                    (w)[2]); \
            const float ix1y1 = lerp((buf)[6][j],\
                    (buf)[7][j], \
                    (w)[2]); \
            ix1 = lerp(ix1y0, ix1y1, (w)[1]); \
        } \
        (out)[j] = lerp(ix0, ix1, (w)[0]); \
    } \
} while (false) \

__device__ __inline__ void _copy(
            const float* __restrict__ data,
            const int64_t index,
            float* __restrict__ output,
            const int stride,
            const int min_idx,
            const int max_idx) {
    if (index < 0) {
        for (int i = min_idx; i < max_idx; ++i)
            output[i] = 0.f;
        return;
    }
    const float* __restrict__ input = data + stride * index;
    for (int i = min_idx; i < max_idx; ++i) {
        output[i] = input[i];
    }
}

#define _COPY_8_NEIGHBORS(dptr, link_ptr, offx, offy, buffer_out, stride, min_idx, max_idx) do { \
    _copy(dptr, link_ptr[0], buffer_out[0], stride, min_idx, max_idx); \
    _copy(dptr, link_ptr[1], buffer_out[1], stride, min_idx, max_idx); \
    _copy(dptr, link_ptr[offy], buffer_out[2], stride, min_idx, max_idx); \
    _copy(dptr, link_ptr[offy + 1], buffer_out[3], stride, min_idx, max_idx); \
    _copy(dptr, link_ptr[offx], buffer_out[4], stride, min_idx, max_idx); \
    _copy(dptr, link_ptr[offx + 1], buffer_out[5], stride, min_idx, max_idx); \
    _copy(dptr, link_ptr[offx + offy], buffer_out[6], stride, min_idx, max_idx); \
    _copy(dptr, link_ptr[offx + offy + 1], buffer_out[7], stride, min_idx, max_idx); \
} while(0)


__device__ __inline__ void _ratomicadd(
            float* __restrict__ output_data,
            const int64_t index,
            const float* __restrict__ input,
            const int max_idx) {
    if (index < 0) return;
    float* __restrict__ output = output_data + max_idx * index;
    for (int i = 0; i < max_idx; ++i) {
        atomicAdd(&output[i], input[i]);
    }
}

#define _UPDATE_8_NEIGHBORS(gdptr_out, link_ptr, offx, offy, buffer, data_dim) do { \
    _ratomicadd(gdptr_out, link_ptr[0], buffer[0], data_dim); \
    _ratomicadd(gdptr_out, link_ptr[1], buffer[1], data_dim); \
    _ratomicadd(gdptr_out, link_ptr[offy], buffer[2], data_dim); \
    _ratomicadd(gdptr_out, link_ptr[offy + 1], buffer[3], data_dim); \
    _ratomicadd(gdptr_out, link_ptr[offx], buffer[4], data_dim); \
    _ratomicadd(gdptr_out, link_ptr[offx + 1], buffer[5], data_dim); \
    _ratomicadd(gdptr_out, link_ptr[offx + offy], buffer[6], data_dim); \
    _ratomicadd(gdptr_out, link_ptr[offx + offy + 1], buffer[7], data_dim); \
} while(0)

__device__ __inline__ void _init_lerp_weight(
            float* __restrict__ table,
            const float* __restrict__ b) {
    const float a[3] {1.f - b[0], 1.f - b[1], 1.f - b[2]};
    table[0] = a[0] * a[1] * a[2];
    table[1] = a[0] * a[1] * b[2];
    table[2] = a[0] * b[1] * a[2];
    table[3] = a[0] * b[1] * b[2];
    table[4] = b[0] * a[1] * a[2];
    table[5] = b[0] * a[1] * b[2];
    table[6] = b[0] * b[1] * a[2];
    table[7] = b[0] * b[1] * b[2];
}
