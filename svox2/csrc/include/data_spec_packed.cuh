// Copyright 2021 Alex Yu
#pragma once
#include <torch/extension.h>
#include "data_spec.hpp"
#include "cuda_util.cuh"
#include "random_util.cuh"

namespace {
namespace device {

struct PackedSparseGridSpec {
    PackedSparseGridSpec(SparseGridSpec& spec)
        :
          density_data(spec.density_data.data_ptr<float>()),
          sh_data(spec.sh_data.data_ptr<float>()),
          links(spec.links.data_ptr<int32_t>()),
          basis_type(spec.basis_type),
          basis_data(spec.basis_data.defined() ? spec.basis_data.data_ptr<float>() : nullptr),
          background_links(spec.background_links.defined() ?
                           spec.background_links.data_ptr<int32_t>() :
                           nullptr),
          background_data(spec.background_data.defined() ?
                          spec.background_data.data_ptr<float>() :
                          nullptr),
          size{(int)spec.links.size(0),
               (int)spec.links.size(1),
               (int)spec.links.size(2)},
          stride_x{(int)spec.links.stride(0)},
          background_reso{
              spec.background_links.defined() ? (int)spec.background_links.size(1) : 0,
          },
          background_nlayers{
              spec.background_data.defined() ? (int)spec.background_data.size(1) : 0
          },
          basis_dim(spec.basis_dim),
          sh_data_dim((int)spec.sh_data.size(1)),
          basis_reso(spec.basis_data.defined() ? spec.basis_data.size(0) : 0),
          _offset{spec._offset.data_ptr<float>()[0],
                  spec._offset.data_ptr<float>()[1],
                  spec._offset.data_ptr<float>()[2]},
          _scaling{spec._scaling.data_ptr<float>()[0],
                   spec._scaling.data_ptr<float>()[1],
                   spec._scaling.data_ptr<float>()[2]} {
    }

    float* __restrict__ density_data;
    float* __restrict__ sh_data;
    const int32_t* __restrict__ links;

    const uint8_t basis_type;
    float* __restrict__ basis_data;

    const int32_t* __restrict__ background_links;
    float* __restrict__ background_data;

    const int size[3], stride_x;
    const int background_reso, background_nlayers;

    const int basis_dim, sh_data_dim, basis_reso;
    const float _offset[3];
    const float _scaling[3];
};

struct PackedGridOutputGrads {
    PackedGridOutputGrads(GridOutputGrads& grads) :
        grad_density_out(grads.grad_density_out.defined() ? grads.grad_density_out.data_ptr<float>() : nullptr),
        grad_sh_out(grads.grad_sh_out.defined() ? grads.grad_sh_out.data_ptr<float>() : nullptr),
        grad_basis_out(grads.grad_basis_out.defined() ? grads.grad_basis_out.data_ptr<float>() : nullptr),
        grad_background_out(grads.grad_background_out.defined() ? grads.grad_background_out.data_ptr<float>() : nullptr),
        mask_out((grads.mask_out.defined() && grads.mask_out.size(0) > 0) ? grads.mask_out.data_ptr<bool>() : nullptr),
        mask_background_out((grads.mask_background_out.defined() && grads.mask_background_out.size(0) > 0) ? grads.mask_background_out.data_ptr<bool>() : nullptr)
        {}
    float* __restrict__ grad_density_out;
    float* __restrict__ grad_sh_out;
    float* __restrict__ grad_basis_out;
    float* __restrict__ grad_background_out;

    bool* __restrict__ mask_out;
    bool* __restrict__ mask_background_out;
};

struct PackedCameraSpec {
    PackedCameraSpec(CameraSpec& cam) :
        c2w(cam.c2w.packed_accessor32<float, 2, torch::RestrictPtrTraits>()),
        fx(cam.fx), fy(cam.fy),
        cx(cam.cx), cy(cam.cy),
        width(cam.width), height(cam.height),
        ndc_coeffx(cam.ndc_coeffx), ndc_coeffy(cam.ndc_coeffy) {}
    const torch::PackedTensorAccessor32<float, 2, torch::RestrictPtrTraits>
        c2w;
    float fx;
    float fy;
    float cx;
    float cy;
    int width;
    int height;

    float ndc_coeffx;
    float ndc_coeffy;
};

struct PackedRaysSpec {
    const torch::PackedTensorAccessor32<float, 2, torch::RestrictPtrTraits> origins;
    const torch::PackedTensorAccessor32<float, 2, torch::RestrictPtrTraits> dirs;
    PackedRaysSpec(RaysSpec& spec) :
        origins(spec.origins.packed_accessor32<float, 2, torch::RestrictPtrTraits>()),
        dirs(spec.dirs.packed_accessor32<float, 2, torch::RestrictPtrTraits>())
    { }
};

struct SingleRaySpec {
    SingleRaySpec() = default;
    __device__ SingleRaySpec(const float* __restrict__ origin, const float* __restrict__ dir)
        : origin{origin[0], origin[1], origin[2]},
          dir{dir[0], dir[1], dir[2]} {}
    __device__ void set(const float* __restrict__ origin, const float* __restrict__ dir) {
#pragma unroll 3
        for (int i = 0; i < 3; ++i) {
            this->origin[i] = origin[i];
            this->dir[i] = dir[i];
        }
    }

    float origin[3];
    float dir[3];
    float tmin, tmax, world_step;

    float pos[3];
    int32_t l[3];
    RandomEngine32 rng;
};

}  // namespace device
}  // namespace
