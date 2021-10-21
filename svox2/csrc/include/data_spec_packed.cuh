// Copyright 2021 Alex Yu
#pragma once
#include <torch/extension.h>
#include "data_spec.hpp"
#include "cuda_util.cuh"

namespace {
namespace device {

struct PackedSparseGridSpec {
    PackedSparseGridSpec(SparseGridSpec& spec)
        :
          density_data(spec.density_data.data_ptr<float>()),
          sh_data(spec.sh_data.data_ptr<float>()),
          links(spec.links.data_ptr<int32_t>()),
          use_learned_basis(spec.use_learned_basis),
          basis_data(spec.basis_data.data_ptr<float>()),
          size{(int)spec.links.size(0),
               (int)spec.links.size(1),
               (int)spec.links.size(2)},
          stride_x{(int)spec.links.stride(0)},
          basis_dim(spec.basis_dim),
          sh_data_dim((int)spec.sh_data.size(1)),
          basis_reso(spec.basis_data.size(0)),
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
    const bool use_learned_basis;
    float* __restrict__ basis_data;

    const int size[3], stride_x;

    const int basis_dim, sh_data_dim, basis_reso;
    const float _offset[3];
    const float _scaling[3];
};

struct PackedCameraSpec {
    PackedCameraSpec(CameraSpec& cam) :
        c2w(cam.c2w.packed_accessor32<float, 2, torch::RestrictPtrTraits>()),
        fx(cam.fx), fy(cam.fy), width(cam.width), height(cam.height) {}
    const torch::PackedTensorAccessor32<float, 2, torch::RestrictPtrTraits>
        c2w;
    float fx;
    float fy;
    int width;
    int height;
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
};

}  // namespace device
}  // namespace
