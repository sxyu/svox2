// Copyright 2021 Alex Yu

// This file contains only Python bindings
#include "data_spec.hpp"
#include <cstdint>
#include <torch/extension.h>
#include <tuple>

using torch::Tensor;

std::tuple<torch::Tensor, torch::Tensor> sample_grid(SparseGridSpec &, Tensor);
void sample_grid_backward(SparseGridSpec &, Tensor, Tensor, Tensor, Tensor,
                          Tensor);

Tensor volume_render_cuvol(SparseGridSpec &, RaysSpec &, RenderOptions &);

Tensor volume_render_cuvol_backward(SparseGridSpec &, RaysSpec &,
                                    RenderOptions &, Tensor, Tensor, Tensor,
                                    Tensor);

Tensor volume_render_cuvol_fused(SparseGridSpec &, RaysSpec &, RenderOptions &,
                                 Tensor, Tensor, Tensor, Tensor);

Tensor volume_render_cuvol_image(SparseGridSpec &, CameraSpec &,
                                 RenderOptions &);

Tensor volume_render_cuvol_image_backward(SparseGridSpec &, CameraSpec &,
                                          RenderOptions &, Tensor, Tensor,
                                          Tensor, Tensor);

// Returns MSE

Tensor dilate(Tensor);
Tensor tv(Tensor, Tensor, int, int);
void tv_grad(Tensor, Tensor, int, int, float, Tensor);
void grid_weight_render(Tensor, CameraSpec &, float, Tensor, Tensor, Tensor);

void rmsprop_step(Tensor, Tensor, Tensor, Tensor, float, float, float);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
#define _REG_FUNC(funname) m.def(#funname, &funname)
  _REG_FUNC(sample_grid);
  _REG_FUNC(sample_grid_backward);
  _REG_FUNC(volume_render_cuvol);
  _REG_FUNC(volume_render_cuvol_backward);
  _REG_FUNC(volume_render_cuvol_fused);
  _REG_FUNC(volume_render_cuvol_image);
  _REG_FUNC(volume_render_cuvol_image_backward);
  _REG_FUNC(dilate);
  _REG_FUNC(tv);
  _REG_FUNC(tv_grad);
  _REG_FUNC(grid_weight_render);
  _REG_FUNC(rmsprop_step);
#undef _REG_FUNC

  py::class_<SparseGridSpec>(m, "SparseGridSpec")
      .def(py::init<>())
      .def_readwrite("density_data", &SparseGridSpec::density_data)
      .def_readwrite("sh_data", &SparseGridSpec::sh_data)
      .def_readwrite("links", &SparseGridSpec::links)
      .def_readwrite("_offset", &SparseGridSpec::_offset)
      .def_readwrite("_scaling", &SparseGridSpec::_scaling)
      .def_readwrite("basis_dim", &SparseGridSpec::basis_dim);

  py::class_<CameraSpec>(m, "CameraSpec")
      .def(py::init<>())
      .def_readwrite("c2w", &CameraSpec::c2w)
      .def_readwrite("fx", &CameraSpec::fx)
      .def_readwrite("fy", &CameraSpec::fy)
      .def_readwrite("width", &CameraSpec::width)
      .def_readwrite("height", &CameraSpec::height);

  py::class_<RaysSpec>(m, "RaysSpec")
      .def(py::init<>())
      .def_readwrite("origins", &RaysSpec::origins)
      .def_readwrite("dirs", &RaysSpec::dirs);

  py::class_<RenderOptions>(m, "RenderOptions")
      .def(py::init<>())
      .def_readwrite("background_brightness",
                     &RenderOptions::background_brightness)
      // .def_readwrite("step_epsilon", &RenderOptions::step_epsilon)
      .def_readwrite("step_size", &RenderOptions::step_size)
      .def_readwrite("sigma_thresh", &RenderOptions::sigma_thresh)
      .def_readwrite("stop_thresh", &RenderOptions::stop_thresh);
  // .def_readwrite("randomize", &RenderOptions::randomize)
  // .def_readwrite("_m1", &RenderOptions::_m1)
  // .def_readwrite("_m2", &RenderOptions::_m2)
  // .def_readwrite("_m3", &RenderOptions::_m3);
}
