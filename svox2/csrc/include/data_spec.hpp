// Copyright 2021 Alex Yu
#pragma once
#include "util.hpp"
#include <torch/extension.h>

using torch::Tensor;

enum BasisType {
  // For svox 1 compatibility
  // BASIS_TYPE_RGBA = 0
  BASIS_TYPE_SH = 1,
  // BASIS_TYPE_SG = 2
  // BASIS_TYPE_ASG = 3
  BASIS_TYPE_3D_TEXTURE = 4,
  BASIS_TYPE_MLP = 255,

  // for surface type
  SURFACE_TYPE_NONE = 100,
  SURFACE_TYPE_PLANE = 101,
  SURFACE_TYPE_SDF = 102,
  SURFACE_TYPE_UDF = 103,
  SURFACE_TYPE_UDF_ALPHA = 104,

  // for cubic solution
  CUBIC_TYPE_NO_ROOT = 200,
  CUBIC_TYPE_LINEAR = 201,
  CUBIC_TYPE_POLY_ONE_R = 202, // polynomial with a single distinct real root
  CUBIC_TYPE_POLY = 203, // polynomial with a two real roots
  CUBIC_TYPE_CUBIC_ONE_R = 204, // cubic with three real and equal roots
  CUBIC_TYPE_CUBIC_THREE_R = 205, // cubic with three real and distinct roots
  CUBIC_TYPE_CUBIC_ONE_R_ = 206, // cubic with a single real root
};

enum ActivationType {
  SIGMOID_FN = 0,
  EXP_FN = 1,
};

struct SparseGridSpec {
  Tensor density_data;
  Tensor surface_data;
  Tensor level_set_data;
  Tensor sh_data;
  Tensor links;
  Tensor _offset;
  Tensor _scaling;

  Tensor background_links;
  Tensor background_data;

  int basis_dim;
  uint8_t basis_type;
  uint8_t surface_type;
  Tensor basis_data;
  float fake_sample_std;

  inline void check() {
    CHECK_INPUT(density_data);
    CHECK_INPUT(sh_data);
    CHECK_INPUT(links);
    if (surface_type != SURFACE_TYPE_NONE){
      CHECK_INPUT(surface_data);
      CHECK_INPUT(level_set_data);
    }
    if (background_links.defined()) {
      CHECK_INPUT(background_links);
      CHECK_INPUT(background_data);
      TORCH_CHECK(background_links.ndimension() ==
                  2);                                 // (H, W) -> [N] \cup {-1}
      TORCH_CHECK(background_data.ndimension() == 3); // (N, D, C) -> R
    }
    if (basis_data.defined()) {
      CHECK_INPUT(basis_data);
    }
    CHECK_CPU_INPUT(_offset);
    CHECK_CPU_INPUT(_scaling);
    TORCH_CHECK(density_data.ndimension() == 2);
    TORCH_CHECK(sh_data.ndimension() == 2);
    TORCH_CHECK(links.ndimension() == 3);
  }
};

struct GridOutputGrads {
  torch::Tensor grad_density_out;
  torch::Tensor grad_surface_out;
  torch::Tensor grad_sh_out;
  torch::Tensor grad_basis_out;
  torch::Tensor grad_background_out;
  torch::Tensor grad_fake_sample_std_out;

  torch::Tensor mask_out;
  torch::Tensor mask_background_out;
  inline void check() {
    if (grad_density_out.defined()) {
      CHECK_INPUT(grad_density_out);
    }
    if (grad_sh_out.defined()) {
      CHECK_INPUT(grad_sh_out);
    }
    if (grad_basis_out.defined()) {
      CHECK_INPUT(grad_basis_out);
    }
    if (grad_background_out.defined()) {
      CHECK_INPUT(grad_background_out);
    }
    if (grad_surface_out.defined()) {
      CHECK_INPUT(grad_surface_out);
    }
    if (grad_fake_sample_std_out.defined()) {
      CHECK_INPUT(grad_fake_sample_std_out);
    }
    if (mask_out.defined() && mask_out.size(0) > 0) {
      CHECK_INPUT(mask_out);
    }
    if (mask_background_out.defined() && mask_background_out.size(0) > 0) {
      CHECK_INPUT(mask_background_out);
    }
  }
};

struct CameraSpec {
  torch::Tensor c2w;
  float fx;
  float fy;
  float cx;
  float cy;
  int width;
  int height;

  float ndc_coeffx;
  float ndc_coeffy;

  inline void check() {
    CHECK_INPUT(c2w);
    TORCH_CHECK(c2w.is_floating_point());
    TORCH_CHECK(c2w.ndimension() == 2);
    TORCH_CHECK(c2w.size(1) == 4);
  }
};

struct RaysSpec {
  Tensor origins;
  Tensor dirs;
  inline void check() {
    CHECK_INPUT(origins);
    CHECK_INPUT(dirs);
    TORCH_CHECK(origins.is_floating_point());
    TORCH_CHECK(dirs.is_floating_point());
  }
};

// struct to store interesctions between ray and voxels
struct RayVoxIntersecSpec {
  Tensor voxel_ls;
  Tensor vox_start_i;
  Tensor vox_num;
  inline void check() {
    CHECK_INPUT(voxel_ls);
    CHECK_INPUT(vox_start_i);
    CHECK_INPUT(vox_num);
  }
};

struct RenderOptions {
  float background_brightness;
  // float step_epsilon;
  float step_size;
  float sigma_thresh;
  float stop_thresh;

  float near_clip;
  bool use_spheric_clip;

  bool last_sample_opaque;

  bool surf_fake_sample;
  float surf_fake_sample_min_vox_len;
  bool limited_fake_sample;

  bool no_surf_grad_from_sh;
  // enum ActivationType alpha_activation_type;
  uint8_t alpha_activation_type;
  bool fake_sample_l_dist;
  bool fake_sample_normalize_surf;
  // bool randomize;
  // float random_sigma_std;
  // float random_sigma_std_background;
  // 32-bit RNG state masks
  // uint32_t _m1, _m2, _m3;

  // int msi_start_layer = 0;
  // int msi_end_layer = 66;
};
