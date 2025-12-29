// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//
// C++ Data Dispatcher for OpenVINO Python Bindings
// This provides a fast alternative to the Python singledispatch-based data_dispatcher.py

#pragma once

#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "openvino/runtime/tensor.hpp"
#include "openvino/runtime/infer_request.hpp"
#include "openvino/core/type/element_type.hpp"
#include "pyopenvino/core/infer_request.hpp"

namespace py = pybind11;

namespace Common {
namespace data_dispatch {

/**
 * @brief Register data dispatcher functions with pybind11 module
 */
void regmodule_data_dispatch(py::module m);

}  // namespace data_dispatch
}  // namespace Common
