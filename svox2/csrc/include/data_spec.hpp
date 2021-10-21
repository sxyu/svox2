// Copyright 2021 Alex Yu
#pragma once
#include "util.hpp"
#include <torch/extension.h>

using torch::Tensor;

struct SparseGridSpec {
  Tensor density_data;
  Tensor sh_data;
  Tensor links;
  Tensor _offset;
  Tensor _scaling;

  int basis_dim;
  bool use_learned_basis;
  Tensor basis_data;

  inline void check() {
    CHECK_INPUT(density_data);
    CHECK_INPUT(sh_data);
    CHECK_INPUT(links);
    CHECK_INPUT(basis_data);
    CHECK_CPU_INPUT(_offset);
    CHECK_CPU_INPUT(_scaling);
    TORCH_CHECK(density_data.is_floating_point());
    TORCH_CHECK(sh_data.is_floating_point());
    TORCH_CHECK(!links.is_floating_point());
    TORCH_CHECK(basis_data.is_floating_point());
    TORCH_CHECK(_offset.is_floating_point());
    TORCH_CHECK(_scaling.is_floating_point());
    TORCH_CHECK(density_data.ndimension() == 2);
    TORCH_CHECK(sh_data.ndimension() == 2);
    TORCH_CHECK(links.ndimension() == 3);
    TORCH_CHECK(basis_data.ndimension() == 4);
  }
};

struct CameraSpec {
  torch::Tensor c2w;
  float fx;
  float fy;
  int width;
  int height;

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

struct RenderOptions {
  float background_brightness;
  // float step_epsilon;
  float step_size;
  float sigma_thresh;
  float stop_thresh;

  // bool randomize;
  // uint32_t _m1, _m2, _m3;
};
