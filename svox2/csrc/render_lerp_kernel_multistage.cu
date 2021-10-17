// Copyright 2021 Alex Yu

#include <torch/extension.h>

#include <cuda_runtime.h>
#include <cstdint>
#include <tuple>
// FOR DEBUGGING
#include <vector>
#include <iostream>
// END FOR DEBUGGING

#include "cuda_util.cuh"
#include "render_util.cuh"
#include "data_spec_packed.cuh"

#define _RAY_ID(x) (x >> 16);
#define _SAMPLE_ID(x) (x & 0xFFFF);

namespace {

cudaError_t cuda_assert(const cudaError_t code, const char* const file,
        const int line, const bool abort) {
    if (code != cudaSuccess) {
        fprintf(stderr, "cuda_assert: %s %s %d\n", cudaGetErrorString(code),
                file, line);

        if (abort) {
            cudaDeviceReset();
            exit(code);
        }
    }

    return code;
}
#define cudaCheck(cmd) cuda_assert((cmd), __FILE__, __LINE__, true);

template<class T>
void print_cuarr(const char* desc, const T* cuarr, size_t size) {
    std::cerr << "\n" << desc << ": ";
    std::vector<T> vec(size);
    cudaCheck(cudaMemcpy(vec.data(), cuarr, size * sizeof(T),
                cudaMemcpyDeviceToHost));
    for (auto i : vec) {
        std::cerr << i << " ";
    }
    std::cerr << "\n";
}

struct GridTransform {
    float scaling[3];
    float offset[3];
    float size[3];
};

struct RaySig {
    float origin[3];
    float dir[3];
    float near, world_step;
};


namespace device {

__global__ void prep_rays(
            const PackedRaysSpec rays_in,
            int n_rays,
            int basis_dim,
            const GridTransform gridt,
            float step_size,
            RaySig* __restrict__ ray_sig,
            int32_t* __restrict__ ray_cnt,
            float* __restrict__ ray_basis
        ) {
    CUDA_GET_THREAD_ID(tid, n_rays);
    const float* __restrict__ origin_in = rays_in.origins[tid].data();
    const float* __restrict__ dir_in = rays_in.dirs[tid].data();

    RaySig sig;

#pragma unroll 3
    for (int i = 0; i < 3; ++i) {
        sig.origin[i] = origin_in[i];
        sig.dir[i] = dir_in[i];
    }

    {
        float* __restrict__ sphfunc_val = ray_basis + basis_dim * tid;
        calc_sh(basis_dim, sig.dir, sphfunc_val);
    }

    transform_coord(sig.origin, gridt.scaling, gridt.offset);
    sig.world_step = _get_delta_scale(gridt.scaling, sig.dir) * step_size;

    float far;
    {
        sig.near = 0.0f;
        far = 1e9f;
#pragma unroll 3
        for (int i = 0; i < 3; ++i) {
            if (sig.dir[i] == 0.f) continue;
            const float inv_dir = 1.f / sig.dir[i];
            const float t1 = (- sig.origin[i]) * inv_dir;
            const float t2 = (gridt.size[i] - 1.f  - sig.origin[i]) * inv_dir;
            sig.near = max(sig.near, min(t1, t2));
            far = min(far, max(t1, t2));
        }
    }

    ray_sig[tid] = sig;
    ray_cnt[tid] = int32_t(fmaxf((far - sig.near) / step_size, 0.f));
    // printf("ray %d, ori %f %f %f, dir %f %f %f, near %f, wstep %f, cnt %d\n", tid,
    //      sig.origin[0], sig.origin[1], sig.origin[2],
    //      sig.dir[0], sig.dir[1], sig.dir[2],
    //      sig.near,
    //      sig.world_step,
    //      ray_cnt[tid]);
}

__global__ void gen_sample_info(
            const int32_t* __restrict__ ray_cnt_sig,
            int n_rays,
            int32_t* __restrict__ out
        ) {
    CUDA_GET_THREAD_ID(tid, n_rays);
    int32_t* __restrict__ tout = out + ray_cnt_sig[tid];
    const int cnt = ray_cnt_sig[tid + 1] - ray_cnt_sig[tid];
    const int32_t ray_bits = tid << 16;
    for (int i = 0; i < cnt; ++i) {
        tout[i] = ray_bits | i;
    }
}

__global__ void trilerp_grid_density(
        const PackedSparseGridSpec grid,
        const RaySig* __restrict__ ray_sig,
        const int32_t* __restrict__ sample_info,
        int n_samples,
        float step_size,
        // Output
        float* __restrict__ out) {
    CUDA_GET_THREAD_ID(tid, n_samples);


    float point[3];

    {
        const int32_t tsample_info = sample_info[tid];
        const int32_t ray_id = _RAY_ID(tsample_info);
        const int32_t ray_sample_id = _SAMPLE_ID(tsample_info);
        const RaySig& __restrict__ sig = ray_sig[ray_id];
        const float t = sig.near + ray_sample_id * step_size;
#pragma unroll 3
        for (int i = 0; i < 3; ++i) {
            point[i] = sig.origin[i] + sig.dir[i] * t;
        }
    }

    int32_t l[3];
#pragma unroll 3
    for (int i = 0; i < 3; ++i) {
        l[i] = min((int32_t)point[i], (int32_t)(grid.links.size(i) - 2));
        point[i] -= l[i];
    }

    const int32_t* __restrict__ link_ptr = &grid.links[l[0]][l[1]][l[2]];

    const int offy = grid.links.stride(1), offx = grid.links.stride(0);

#define MAYBE_READ_LINK_D(u) ((link_ptr[u] >= 0) ? grid.density_data[link_ptr[u]][0] : 0.f)

    const float ix0y0 = lerp(MAYBE_READ_LINK_D(0), MAYBE_READ_LINK_D(1), point[2]);
    const float ix0y1 = lerp(MAYBE_READ_LINK_D(offy), MAYBE_READ_LINK_D(offy + 1), point[2]);
    const float ix0 = lerp(ix0y0, ix0y1, point[1]);
    const float ix1y0 = lerp(MAYBE_READ_LINK_D(offx), MAYBE_READ_LINK_D(offx + 1), point[2]);
    const float ix1y1 = lerp(MAYBE_READ_LINK_D(offy + offx),
                             MAYBE_READ_LINK_D(offy + offx + 1), point[2]);
    const float ix1 = lerp(ix1y0, ix1y1, point[1]);
    out[tid] = lerp(ix0, ix1, point[0]);
}
#undef MAYBE_READ_LINK_D

}  // namespace device
}  // namespace


torch::Tensor volume_render_multistage_fused(
        SparseGridSpec& grid,
        RaysSpec& rays, RenderOptions& opt,
        Tensor& grad_density_data_out,
        Tensor& grad_sh_data_out) {
    DEVICE_GUARD(grid.sh_data);
    grid.check();
    rays.check();
    CHECK_INPUT(grad_density_data_out);
    CHECK_INPUT(grad_sh_data_out);
    const size_t n_rays = rays.origins.size(0);
    const size_t cuda_max_threads = CUDA_MAX_THREADS;

    RaySig* ray_sig;
    int32_t* ray_cnt, * ray_cnt_scan;
    int32_t* sample_info;
    float* sample_density;
    float* ray_basis;
    cudaCheck(cudaMalloc(&ray_sig, n_rays * sizeof(RaySig)));
    cudaCheck(cudaMalloc(&ray_cnt, n_rays * sizeof(int32_t)));
    cudaCheck(cudaMalloc(&ray_cnt_scan, (n_rays + 1) * sizeof(int32_t)));
    cudaCheck(cudaMalloc(&ray_basis, n_rays * grid.basis_dim * sizeof(float)));
    cudaCheck(cudaMemsetAsync(ray_cnt_scan, 0, sizeof(int32_t)));

    float* scl = grid._scaling.data<float>();
    float* offs = grid._offset.data<float>();

    {
        const int cuda_n_threads = std::min(n_rays, cuda_max_threads);
        const int blocks = CUDA_N_BLOCKS_NEEDED(n_rays, cuda_n_threads);
        device::prep_rays<<<blocks, cuda_n_threads>>>(
                rays,
                n_rays,
                grid.basis_dim,
                GridTransform {{scl[0], scl[1], scl[2]},
                {offs[0], offs[1], offs[2]},
                {(float)grid.links.size(0), (float)grid.links.size(1),
                 (float)grid.links.size(2)}},
                opt.step_size,
                // Output
                ray_sig,
                ray_cnt,
                ray_basis);
    }

    // print_cuarr("cnt", ray_cnt, n_rays);

    // Determine temporary device storage requirements
    void     *d_temp_storage = NULL;
    size_t   temp_storage_bytes = 0;
    cub::DeviceScan::InclusiveSum(d_temp_storage, temp_storage_bytes, ray_cnt,
            ray_cnt_scan + 1, n_rays);
    // Allocate temporary storage
    cudaCheck(cudaMalloc(&d_temp_storage, temp_storage_bytes));
    // Run exclusive prefix sum
    cub::DeviceScan::InclusiveSum(d_temp_storage, temp_storage_bytes,
            ray_cnt, ray_cnt_scan + 1, n_rays);

    // print_cuarr("scan", ray_cnt_scan, n_rays + 1);

    cudaCheck(cudaFree(d_temp_storage));

    int32_t n_total_samples;
    cudaCheck(cudaMemcpyAsync(&n_total_samples, ray_cnt_scan + n_rays, sizeof(int32_t),
                              cudaMemcpyDeviceToHost));
    cudaCheck(cudaMalloc(&sample_info, n_total_samples * sizeof(int32_t)));
    cudaCheck(cudaMalloc(&sample_density, n_total_samples * sizeof(int32_t)));


    {
        const int cuda_n_threads = std::min(n_rays, cuda_max_threads);
        const int blocks = CUDA_N_BLOCKS_NEEDED(n_rays, cuda_n_threads);
        device::gen_sample_info<<<blocks, cuda_n_threads>>>(
                ray_cnt_scan,
                n_rays,
                // Output
                sample_info);
    }

    {
        const int cuda_n_threads = std::min<int>(n_total_samples, cuda_max_threads);
        const int blocks = CUDA_N_BLOCKS_NEEDED(n_total_samples, cuda_n_threads);
        device::trilerp_grid_density<<<blocks, cuda_n_threads>>>(
                grid,
                ray_sig,
                sample_info,
                n_total_samples,
                opt.step_size,
                sample_density);
    }

    // print_cuarr("sample_info", sample_info, n_total_samples);
    // std::cerr << "nts: " << n_total_samples << "\n";


    // const int cuda_n_threads = 768;
    // const int blocks = CUDA_N_BLOCKS_NEEDED(Q, cuda_n_threads);

    // device::render_ray_kernel<<<blocks, cuda_n_threads>>>(
    //         grid, rays, opt,
    //         // Output
    //         mse.packed_accessor32<float, 2, torch::RestrictPtrTraits>());
    // print_cuarr("sigma", sample_density, n_total_samples);

    cudaCheck(cudaFree(ray_sig));
    cudaCheck(cudaFree(ray_cnt));
    cudaCheck(cudaFree(ray_cnt_scan));
    cudaCheck(cudaFree(ray_basis));
    cudaCheck(cudaFree(sample_info));
    cudaCheck(cudaFree(sample_density));
    CUDA_CHECK_ERRORS;
    torch::Tensor mse = torch::zeros({rays.origins.size(0)}, grid.sh_data.options());

    float ntsf = n_total_samples;
    cudaCheck(cudaMemcpyAsync(mse.data<float>(), &ntsf, sizeof(float),
                cudaMemcpyHostToDevice)); // DEBUG
    return mse;
}
