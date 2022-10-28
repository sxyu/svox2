// Copyright 2021 Alex Yu
#pragma once
#include <cuda_runtime.h>
#include <cub/cub.cuh>
#include <c10/cuda/CUDAGuard.h>
#include <ATen/cuda/CUDAContext.h>
#include "util.hpp"
#include "data_spec.hpp"

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
__host__ __device__ __inline__ T lerp(T a, T b, T w) {
    return fmaf(w, b - a, a);
}

__device__ __inline__ static float _norm(
                const float* __restrict__ dir) {
    // return sqrtf(dir[0] * dir[0] + dir[1] * dir[1] + dir[2] * dir[2]);
    return norm3df(dir[0], dir[1], dir[2]);
}

__device__ __inline__ static float _rnorm(
                const float* __restrict__ dir) {
    // return 1.f / _norm(dir);
    return rnorm3df(dir[0], dir[1], dir[2]);
}

__host__ __device__ __inline__ static void xsuby3d(
                float* __restrict__ x,
                const float* __restrict__ y) {
    x[0] -= y[0];
    x[1] -= y[1];
    x[2] -= y[2];
}

__host__ __device__ __inline__ static float _dot(
                const float* __restrict__ x,
                const float* __restrict__ y) {
    return x[0] * y[0] + x[1] * y[1] + x[2] * y[2];
}

__host__ __device__ __inline__ static void _cross(
                const float* __restrict__ a,
                const float* __restrict__ b,
                float* __restrict__ out) {
    out[0] = a[1] * b[2] - a[2] * b[1];
    out[1] = a[2] * b[0] - a[0] * b[2];
    out[2] = a[0] * b[1] - a[1] * b[0];
}

__device__ __inline__ static float _dist_ray_to_origin(
                const float* __restrict__ origin,
                const float* __restrict__ dir) {
    // dir must be unit vector
    float tmp[3];
    _cross(origin, dir, tmp);
    return _norm(tmp);
}

#define int_div2_ceil(x) ((((x) - 1) >> 1) + 1)

__host__ __inline__ cudaError_t cuda_assert(
        const cudaError_t code, const char* const file,
        const int line, const bool abort) {
    if (code != cudaSuccess) {
        fprintf(stderr, "cuda_assert: %s %s %s %d\n", cudaGetErrorName(code) ,cudaGetErrorString(code),
                file, line);

        if (abort) {
            cudaDeviceReset();
            exit(code);
        }
    }

    return code;
}

#define cuda(...) cuda_assert((cuda##__VA_ARGS__), __FILE__, __LINE__, true);



__device__ __inline__ void surface_to_cubic_equation(
    const float* __restrict__ surface,
    const float* __restrict__ origin,
    const float* __restrict__ dir,
    const int32_t* __restrict__ l,
    float* __restrict__ outs
){

    float const a00 = surface[0b000] * (1-origin[2]+l[2]) + surface[0b001] * (origin[2]-l[2]);
    float const a01 = surface[0b010] * (1-origin[2]+l[2]) + surface[0b011] * (origin[2]-l[2]);
    float const a10 = surface[0b100] * (1-origin[2]+l[2]) + surface[0b101] * (origin[2]-l[2]);
    float const a11 = surface[0b110] * (1-origin[2]+l[2]) + surface[0b111] * (origin[2]-l[2]);

    float const b00 = -surface[0b000] + surface[0b001];
    float const b01 = -surface[0b010] + surface[0b011];
    float const b10 = -surface[0b100] + surface[0b101];
    float const b11 = -surface[0b110] + surface[0b111];

    float const c0 = a00*(1-origin[1]+l[1]) + a01*(origin[1]-l[1]);
    float const c1 = a10*(1-origin[1]+l[1]) + a11*(origin[1]-l[1]);

    float const d0 = -(a00*dir[1] - dir[2]*b00*(1-origin[1]+l[1])) + (a01*dir[1] + dir[2]*b01*(origin[1]-l[1]));
    float const d1 = -(a10*dir[1] - dir[2]*b10*(1-origin[1]+l[1])) + (a11*dir[1] + dir[2]*b11*(origin[1]-l[1]));

    float const e0 = -dir[1]*dir[2]*b00 + dir[1]*dir[2]*b01;
    float const e1 = -dir[1]*dir[2]*b10 + dir[1]*dir[2]*b11;

    outs[3] = -e0*dir[0] + e1*dir[0];
    outs[2] = -d0*dir[0]+e0*(1-origin[0]+l[0]) + d1*dir[0]+e1*(origin[0]-l[0]);
    outs[1] = -c0*dir[0] + d0*(1-origin[0]+l[0]) + c1*dir[0] + d1*(origin[0]-l[0]);
    outs[0] = c0*(1-origin[0]+l[0]) + c1*(origin[0]-l[0]);
}


__device__ __inline__ enum BasisType cubic_equation_solver(
    float const f0,
    float const f1,
    float const f2,
    float const f3,
    float const eps,
    double const eps_double,
    float* __restrict__ outs
){
    if (_CLOSE_TO_ZERO(f3, eps)){
        if (_CLOSE_TO_ZERO(f2, eps)){
            if (_CLOSE_TO_ZERO(f1, eps)){
                // no solution
                return CUBIC_TYPE_NO_ROOT; 
            } else {
                // linear case
                outs[0] = -f0 / f1;
                assert(!isnan(outs[0]));
                return CUBIC_TYPE_LINEAR;
            }
        } else {
            // polynomial case
            // _b, _c, _d = f2[quad_mask], f1[quad_mask], f0[quad_mask]
            float const D = _SQR(f1) - 4.0 * f2 * f0;
            float const sqrt_D = sqrtf(D);
            if (D > 0){
                if (f2 > 0){
                    outs[0] = (-f1 - sqrt_D) / (2 * f2);
                    outs[1] = (-f1 + sqrt_D) / (2 * f2);
                }else{
                    outs[0] = (-f1 + sqrt_D) / (2 * f2);
                    outs[1] = (-f1 - sqrt_D) / (2 * f2);
                }

                // assert(!isnan(outs[0]));
                // assert(!isnan(outs[1]));
                if (_CLOSE_TO_ZERO(outs[0] - outs[1], eps)){
                    // if two roots are too similiar (D==0), then just take one
                    outs[1] = -1;
                    return CUBIC_TYPE_POLY_ONE_R;
                }
                return CUBIC_TYPE_POLY;
            }
            return CUBIC_TYPE_NO_ROOT;
        }
    } else {
        // cubic case
        double const eps_double = 1e-10;
        // double const norm_term = static_cast<double>(f3);
        // double const a = static_cast<double>(f3) / norm_term;
        // double const b = static_cast<double>(f2) / norm_term;
        // double const c = static_cast<double>(f1) / norm_term;
        // double const d = static_cast<double>(f0) / norm_term;

        // double const f = ((3*c/a) - (_SQR(b) / _SQR(a))) / 3;                      
        // double const g = (((2*_CUBIC(b)) / _CUBIC(a)) - ((9*b*c) / _SQR(a)) + (27*d/a)) / 27;                 
        // double const h = (_SQR(g) / 4 + _CUBIC(f) / 27);
        #define norm_term (static_cast<double>(f3))
        #define a (static_cast<double>(f3) / norm_term)
        #define b (static_cast<double>(f2) / norm_term)
        #define c (static_cast<double>(f1) / norm_term)
        #define d (static_cast<double>(f0) / norm_term)

        #define f (((3*c/a) - (_SQR(b) / _SQR(a))) / 3)
        #define g ((((2*_CUBIC(b)) / _CUBIC(a)) - ((9*b*c) / _SQR(a)) + (27*d/a)) / 27)
        #define h ((_SQR(g) / 4 + _CUBIC(f) / 27))
        // -inf + inf create nan!

        if ((_CLOSE_TO_ZERO(f, eps_double)) & (_CLOSE_TO_ZERO(g, eps_double)) & (_CLOSE_TO_ZERO(h, eps_double))){
            // all three roots are real and equal
            outs[0] = static_cast<float>(_COND_CBRT(d/a));
            // if ((isnan(outs[0])) | (!isfinite(outs[0]))){
            //     printf("a=%f\n", a);
            //     printf("b=%f\n", b);
            //     printf("c=%f\n", c);
            //     printf("d=%f\n", d);
            //     printf("g=%f\n", g);
            //     printf("h=%f\n", h);
            //     printf("f=%f\n", f);
            // }
            // assert(!isnan(outs[0]));

            return CUBIC_TYPE_CUBIC_ONE_R;

        } else if (h <= 0){
            // all three roots are real and distinct
            // note that if h==0, gradient cannot be computed
            if (h==0){
                return CUBIC_TYPE_NO_ROOT;
            }

            // double const _i = sqrt((_SQR(g) / 4.) - h);   
            // double const _j = cbrt(_i);
            // double const _k = acos(-(g / (2 * _i)));
            // double const _M = cos(_k / 3.);       
            // double const _N = sqrt(3) * sin(_k / 3.);
            // double const _P = (b / (3. * a)) * -1;         

            #define _i (sqrt((_SQR(g) / 4.) - h))
            #define _j (cbrt(_i))
            #define _k (acos(-(g / (2 * _i))))
            #define _M (cos(_k / 3.))
            #define _N (sqrt(3) * sin(_k / 3.))
            #define _P ((b / (3. * a)) * -1)

            outs[0] = static_cast<float>(-1 *_j * (_M + _N) + _P);
            outs[1] = static_cast<float>(-1 *_j * (_M - _N) + _P);
            outs[2] = static_cast<float>(2 * _j * _M + _P);
            // if (isnan(outs[0]) | isnan(outs[1]) | isnan(outs[2]) | (!isfinite(outs[0])) | (!isfinite(outs[1])) | (!isfinite(outs[2]))){
            //     printf("a=%f\n", a);
            //     printf("b=%f\n", b);
            //     printf("c=%f\n", c);
            //     printf("d=%f\n", d);
            //     printf("g=%f\n", g);
            //     printf("h=%f\n", h);
            //     printf("f=%f\n", f);

            //     printf("_i=%f\n", _i);
            //     printf("_j=%f\n", _j);
            //     printf("_k=%f\n", _k);
            //     printf("_M=%f\n", _M);
            //     printf("_N=%f\n", _N);
            //     printf("_P=%f\n", _P);
            // }
            // assert(!isnan(outs[0]));
            // assert(!isnan(outs[1]));
            // assert(!isnan(outs[2]));

            return CUBIC_TYPE_CUBIC_THREE_R;
        } else {
            // only one real root
            // double const _R = -(g / 2.) + sqrt(h);
            // double const _S = _COND_CBRT(_R);

            // double const _T = -(g / 2.) - sqrt(h);
            // double const _U = _COND_CBRT(_T);
            #define _R (-(g / 2.) + sqrt(h))
            #define _S (_COND_CBRT(_R))

            #define _T (-(g / 2.) - sqrt(h))
            #define _U (_COND_CBRT(_T))

            outs[0] = static_cast<float>((_S + _U) - (b / (3. * a)));

            // if ((isnan(outs[0])) | (!isfinite(outs[0]))){
            //     printf("a=%f\n", a);
            //     printf("b=%f\n", b);
            //     printf("c=%f\n", c);
            //     printf("d=%f\n", d);
            //     printf("g=%f\n", g);
            //     printf("h=%f\n", h);
            //     printf("f=%f\n", f);
            //     printf("_R=%f\n", _R);
            //     printf("_S=%f\n", _S);
            //     printf("_T=%f\n", _T);
            //     printf("_U=%f\n", _U);
            // }
            //     assert(!isnan(outs[0]));

            return CUBIC_TYPE_CUBIC_ONE_R_;
        }

    }
}



#undef norm_term 
#undef a  
#undef b  
#undef c  
#undef d  

#undef f  
#undef g  
#undef h

#undef _i
#undef _j
#undef _k
#undef _M
#undef _N
#undef _P

#undef _R
#undef _S
#undef _T
#undef _U