// Copyright 2021 Alex Yu
#pragma once
#include "util.hpp"
#include <torch/extension.h>

using torch::Tensor;

struct SparseGridSpec {
    Tensor data;
    Tensor links;
    Tensor _offset;
    Tensor _scaling;
    int basis_dim;
    inline void check() {
        CHECK_INPUT(data);
        CHECK_INPUT(links);
        CHECK_CPU_INPUT(_offset);
        CHECK_CPU_INPUT(_scaling);
        TORCH_CHECK(data.is_floating_point());
        TORCH_CHECK(!links.is_floating_point());
        TORCH_CHECK(_offset.is_floating_point());
        TORCH_CHECK(_scaling.is_floating_point());
        TORCH_CHECK(data.ndimension() == 2);
        TORCH_CHECK(links.ndimension() == 3);
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
    float step_epsilon;
    float step_size;
    float sigma_thresh;
    float stop_thresh;
};
