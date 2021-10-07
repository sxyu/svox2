// Copyright 2021 Alex Yu

// This file contains only Python bindings
#include "data_spec.hpp"
#include <cstdint>
#include <torch/extension.h>
#include <tuple>

using torch::Tensor;

Tensor sample_grid(SparseGridSpec &, Tensor);
Tensor sample_grid_backward(SparseGridSpec &, Tensor, Tensor);

Tensor volume_render_cuvol(SparseGridSpec &, RaysSpec &, RenderOptions &);
std::tuple<Tensor, Tensor> volume_render_cuvol_backward(SparseGridSpec &,
                                                        RaysSpec &,
                                                        RenderOptions &, Tensor,
                                                        Tensor);

Tensor dilate(Tensor);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
#define _REG_FUNC(funname) m.def(#funname, &funname)
  _REG_FUNC(sample_grid);
  _REG_FUNC(sample_grid_backward);
  _REG_FUNC(volume_render_cuvol);
  _REG_FUNC(volume_render_cuvol_backward);
  _REG_FUNC(dilate);
#undef _REG_FUNC

  py::class_<SparseGridSpec>(m, "SparseGridSpec")
      .def(py::init<>())
      .def_readwrite("data", &SparseGridSpec::data)
      .def_readwrite("links", &SparseGridSpec::links)
      .def_readwrite("cubemap", &SparseGridSpec::cubemap)
      .def_readwrite("_offset", &SparseGridSpec::_offset)
      .def_readwrite("_scaling", &SparseGridSpec::_scaling)
      .def_readwrite("basis_dim", &SparseGridSpec::basis_dim);

  py::class_<RaysSpec>(m, "RaysSpec")
      .def(py::init<>())
      .def_readwrite("origins", &RaysSpec::origins)
      .def_readwrite("dirs", &RaysSpec::dirs);

  py::class_<RenderOptions>(m, "RenderOptions")
      .def(py::init<>())
      .def_readwrite("background_brightness",
                     &RenderOptions::background_brightness)
      .def_readwrite("step_epsilon", &RenderOptions::step_epsilon)
      .def_readwrite("step_size", &RenderOptions::step_size)
      .def_readwrite("sigma_thresh", &RenderOptions::sigma_thresh)
      .def_readwrite("stop_thresh", &RenderOptions::stop_thresh);
}
