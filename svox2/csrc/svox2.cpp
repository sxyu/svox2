// Copyright 2021 Alex Yu

// This file contains only Python bindings
#include "data_spec.hpp"
#include <cstdint>
#include <torch/extension.h>
#include <tuple>

using torch::Tensor;

std::tuple<torch::Tensor, torch::Tensor> sample_grid(SparseGridSpec &, Tensor,
                                                     bool);
void sample_grid_backward(SparseGridSpec &, Tensor, Tensor, Tensor, Tensor,
                          Tensor, bool);

// ** NeRF rendering formula (trilerp)
Tensor volume_render_cuvol(SparseGridSpec &, RaysSpec &, RenderOptions &);
Tensor volume_render_cuvol_image(SparseGridSpec &, CameraSpec &,
                                 RenderOptions &);
void volume_render_cuvol_backward(SparseGridSpec &, RaysSpec &, RenderOptions &,
                                  Tensor, Tensor, GridOutputGrads &);
void volume_render_cuvol_fused(SparseGridSpec &, RaysSpec &, RenderOptions &,
                               Tensor, float, float, Tensor, GridOutputGrads &);
// Expected termination (depth) rendering
torch::Tensor volume_render_expected_term(SparseGridSpec &, RaysSpec &,
                                          RenderOptions &);
// Depth rendering based on sigma-threshold as in Dex-NeRF
torch::Tensor volume_render_sigma_thresh(SparseGridSpec &, RaysSpec &,
                                         RenderOptions &, float);

// ** NV rendering formula (trilerp)
Tensor volume_render_nvol(SparseGridSpec &, RaysSpec &, RenderOptions &);
void volume_render_nvol_backward(SparseGridSpec &, RaysSpec &, RenderOptions &,
                                 Tensor, Tensor, GridOutputGrads &);
void volume_render_nvol_fused(SparseGridSpec &, RaysSpec &, RenderOptions &,
                              Tensor, float, float, Tensor, GridOutputGrads &);

// ** NeRF rendering formula (nearest-neighbor, infinitely many steps)
Tensor volume_render_svox1(SparseGridSpec &, RaysSpec &, RenderOptions &);
void volume_render_svox1_backward(SparseGridSpec &, RaysSpec &, RenderOptions &,
                                  Tensor, Tensor, GridOutputGrads &);
void volume_render_svox1_fused(SparseGridSpec &, RaysSpec &, RenderOptions &,
                               Tensor, float, float, Tensor, GridOutputGrads &);

// Tensor volume_render_cuvol_image(SparseGridSpec &, CameraSpec &,
//                                  RenderOptions &);
//
// void volume_render_cuvol_image_backward(SparseGridSpec &, CameraSpec &,
//                                         RenderOptions &, Tensor, Tensor,
//                                         GridOutputGrads &);

// Misc
Tensor dilate(Tensor);
void accel_dist_prop(Tensor);
void grid_weight_render(Tensor, CameraSpec &, float, float, bool, Tensor,
                        Tensor, Tensor);
// void sample_cubemap(Tensor, Tensor, bool, Tensor);

// Loss
Tensor tv(Tensor, Tensor, int, int, bool, float, bool, float, float);
void tv_grad(Tensor, Tensor, int, int, float, bool, float, bool, float, float,
             Tensor);
void tv_grad_sparse(Tensor, Tensor, Tensor, Tensor, int, int, float, bool,
                    float, bool, bool, float, float, Tensor);
void msi_tv_grad_sparse(Tensor, Tensor, Tensor, Tensor, float, float, Tensor);
void lumisphere_tv_grad_sparse(SparseGridSpec &, Tensor, Tensor, Tensor, float,
                               float, float, float, GridOutputGrads &);

// Optim
void rmsprop_step(Tensor, Tensor, Tensor, Tensor, float, float, float, float,
                  float);
void sgd_step(Tensor, Tensor, Tensor, float, float);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
#define _REG_FUNC(funname) m.def(#funname, &funname)
  _REG_FUNC(sample_grid);
  _REG_FUNC(sample_grid_backward);
  _REG_FUNC(volume_render_cuvol);
  _REG_FUNC(volume_render_cuvol_image);
  _REG_FUNC(volume_render_cuvol_backward);
  _REG_FUNC(volume_render_cuvol_fused);
  _REG_FUNC(volume_render_expected_term);
  _REG_FUNC(volume_render_sigma_thresh);

  _REG_FUNC(volume_render_nvol);
  _REG_FUNC(volume_render_nvol_backward);
  _REG_FUNC(volume_render_nvol_fused);

  _REG_FUNC(volume_render_svox1);
  _REG_FUNC(volume_render_svox1_backward);
  _REG_FUNC(volume_render_svox1_fused);

  // _REG_FUNC(volume_render_cuvol_image);
  // _REG_FUNC(volume_render_cuvol_image_backward);

  // Loss
  _REG_FUNC(tv);
  _REG_FUNC(tv_grad);
  _REG_FUNC(tv_grad_sparse);
  _REG_FUNC(msi_tv_grad_sparse);
  _REG_FUNC(lumisphere_tv_grad_sparse);

  // Misc
  _REG_FUNC(dilate);
  _REG_FUNC(accel_dist_prop);
  _REG_FUNC(grid_weight_render);
  // _REG_FUNC(sample_cubemap);

  // Optimizer
  _REG_FUNC(rmsprop_step);
  _REG_FUNC(sgd_step);
#undef _REG_FUNC

  py::class_<SparseGridSpec>(m, "SparseGridSpec")
      .def(py::init<>())
      .def_readwrite("density_data", &SparseGridSpec::density_data)
      .def_readwrite("sh_data", &SparseGridSpec::sh_data)
      .def_readwrite("links", &SparseGridSpec::links)
      .def_readwrite("_offset", &SparseGridSpec::_offset)
      .def_readwrite("_scaling", &SparseGridSpec::_scaling)
      .def_readwrite("basis_dim", &SparseGridSpec::basis_dim)
      .def_readwrite("basis_type", &SparseGridSpec::basis_type)
      .def_readwrite("basis_data", &SparseGridSpec::basis_data)
      .def_readwrite("background_links", &SparseGridSpec::background_links)
      .def_readwrite("background_data", &SparseGridSpec::background_data);

  py::class_<CameraSpec>(m, "CameraSpec")
      .def(py::init<>())
      .def_readwrite("c2w", &CameraSpec::c2w)
      .def_readwrite("fx", &CameraSpec::fx)
      .def_readwrite("fy", &CameraSpec::fy)
      .def_readwrite("cx", &CameraSpec::cx)
      .def_readwrite("cy", &CameraSpec::cy)
      .def_readwrite("width", &CameraSpec::width)
      .def_readwrite("height", &CameraSpec::height)
      .def_readwrite("ndc_coeffx", &CameraSpec::ndc_coeffx)
      .def_readwrite("ndc_coeffy", &CameraSpec::ndc_coeffy);

  py::class_<RaysSpec>(m, "RaysSpec")
      .def(py::init<>())
      .def_readwrite("origins", &RaysSpec::origins)
      .def_readwrite("dirs", &RaysSpec::dirs);

  py::class_<RenderOptions>(m, "RenderOptions")
      .def(py::init<>())
      .def_readwrite("background_brightness",
                     &RenderOptions::background_brightness)
      .def_readwrite("step_size", &RenderOptions::step_size)
      .def_readwrite("sigma_thresh", &RenderOptions::sigma_thresh)
      .def_readwrite("stop_thresh", &RenderOptions::stop_thresh)
      .def_readwrite("near_clip", &RenderOptions::near_clip)
      .def_readwrite("use_spheric_clip", &RenderOptions::use_spheric_clip)
      .def_readwrite("last_sample_opaque", &RenderOptions::last_sample_opaque);
  // .def_readwrite("randomize", &RenderOptions::randomize)
  // .def_readwrite("random_sigma_std", &RenderOptions::random_sigma_std)
  // .def_readwrite("random_sigma_std_background",
  //                &RenderOptions::random_sigma_std_background)
  // .def_readwrite("_m1", &RenderOptions::_m1)
  // .def_readwrite("_m2", &RenderOptions::_m2)
  // .def_readwrite("_m3", &RenderOptions::_m3);

  py::class_<GridOutputGrads>(m, "GridOutputGrads")
      .def(py::init<>())
      .def_readwrite("grad_density_out", &GridOutputGrads::grad_density_out)
      .def_readwrite("grad_sh_out", &GridOutputGrads::grad_sh_out)
      .def_readwrite("grad_basis_out", &GridOutputGrads::grad_basis_out)
      .def_readwrite("grad_background_out",
                     &GridOutputGrads::grad_background_out)
      .def_readwrite("mask_out", &GridOutputGrads::mask_out)
      .def_readwrite("mask_background_out",
                     &GridOutputGrads::mask_background_out);
}
