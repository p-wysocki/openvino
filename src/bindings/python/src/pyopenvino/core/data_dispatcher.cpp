// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//
// C++ Data Dispatcher Implementation
// Replaces Python data_dispatcher.py for faster inference dispatch

#include "data_dispatcher.hpp"
#include "common.hpp"

namespace py = pybind11;

namespace Common {
namespace data_dispatch {

namespace {

// Get tensor from request by key type
ov::Tensor get_request_tensor(InferRequestWrapper& wrapper, const py::object& key) {
    if (key.is_none()) {
        return wrapper.m_request.get_input_tensor();
    } else if (py::isinstance<py::int_>(key)) {
        return wrapper.m_request.get_input_tensor(key.cast<size_t>());
    } else if (py::isinstance<py::str>(key)) {
        return wrapper.m_request.get_tensor(key.cast<std::string>());
    } else if (py::isinstance<ov::Output<const ov::Node>>(key)) {
        return wrapper.m_request.get_tensor(key.cast<ov::Output<const ov::Node>>());
    }
    throw py::type_error("Unsupported key type for tensor lookup");
}

// Ensure array is C-contiguous
py::array ensure_contiguous(const py::array& array) {
    if ((array.flags() & py::array::c_style) != 0) {
        return array;
    }
    return py::array::ensure(array, py::array::c_style | py::array::forcecast);
}

// Check if list contains only simple types
bool is_simple_type_list(const py::list& input_list) {
    for (const auto& item : input_list) {
        if (py::isinstance<py::list>(item)) {
            for (const auto& elem : item.cast<py::list>()) {
                if (!py::isinstance<py::str>(elem) && 
                    !py::isinstance<py::bytes>(elem) &&
                    !py::isinstance<py::int_>(elem) && 
                    !py::isinstance<py::float_>(elem)) {
                    return false;
                }
            }
        } else {
            if (!py::isinstance<py::str>(item) && 
                !py::isinstance<py::bytes>(item) &&
                !py::isinstance<py::int_>(item) && 
                !py::isinstance<py::float_>(item)) {
                return false;
            }
        }
    }
    return true;
}

// Convert numpy array to tensor
ov::Tensor array_to_tensor(
    const py::array& input_array,
    InferRequestWrapper& wrapper,
    bool is_shared,
    const py::object& key
) {
    auto tensor = get_request_tensor(wrapper, key);
    auto tensor_type = tensor.get_element_type();
    
    // String type: always use Python Tensor constructor
    if (tensor_type == ov::element::string) {
        py::object ov_tensor_cls = py::module_::import("openvino").attr("Tensor");
        return ov_tensor_cls(input_array, false).cast<ov::Tensor>();
    }
    
    // Get array properties
    auto array = ensure_contiguous(input_array);
    auto arr_dtype = array.dtype();
    auto tensor_dtype = type_helpers::get_dtype(tensor_type);
    
    // Scalar edge-case (0-dim array)
    if (array.ndim() == 0) {
        py::object ov_tensor_cls = py::module_::import("openvino").attr("Tensor");
        auto tensor_shape = tensor.get_shape();
        
        if (arr_dtype.is(tensor_dtype) && tensor_shape.empty()) {
            return ov_tensor_cls(array, is_shared).cast<ov::Tensor>();
        } else if (tensor.get_size() == 0) {
            // Dynamic input first inference - reshape to (1,)
            auto converted = array.attr("astype")(tensor_dtype).attr("reshape")(py::make_tuple(1));
            return ov_tensor_cls(converted, false).cast<ov::Tensor>();
        } else {
            auto converted = array.attr("astype")(tensor_dtype).attr("reshape")(py::tuple(py::cast(tensor_shape)));
            return ov_tensor_cls(converted, false).cast<ov::Tensor>();
        }
    }
    
    // BF16 workaround: always copy with view conversion
    if (tensor_type == ov::element::bf16) {
        py::object ov_tensor_cls = py::module_::import("openvino").attr("Tensor");
        py::object ov_type_cls = py::module_::import("openvino").attr("Type");
        auto new_tensor = ov_tensor_cls(ov_type_cls.attr("bf16"), array.attr("shape"));
        new_tensor.attr("data")[py::slice(py::none(), py::none(), py::none())] = 
            array.attr("view")(tensor_dtype);
        return new_tensor.cast<ov::Tensor>();
    }
    
    // Non-writeable array: always copy
    if (!array.writeable()) {
        py::object ov_tensor_cls = py::module_::import("openvino").attr("Tensor");
        py::object ov_type_cls = py::module_::import("openvino").attr("Type");
        py::array src_array = arr_dtype.is(tensor_dtype) ? array : array.attr("astype")(tensor_dtype).cast<py::array>();
        auto new_tensor = ov_tensor_cls(ov_type_cls.attr(tensor_type.get_type_name().c_str()), array.attr("shape"));
        new_tensor.attr("data")[py::slice(py::none(), py::none(), py::none())] = src_array;
        return new_tensor.cast<ov::Tensor>();
    }
    
    // Type mismatch: convert and copy
    if (!arr_dtype.is(tensor_dtype)) {
        py::object ov_tensor_cls = py::module_::import("openvino").attr("Tensor");
        auto converted = array.attr("astype")(tensor_dtype);
        return ov_tensor_cls(converted, false).cast<ov::Tensor>();
    }
    
    // Happy path: use specified sharing mode
    py::object ov_tensor_cls = py::module_::import("openvino").attr("Tensor");
    return ov_tensor_cls(array, is_shared).cast<ov::Tensor>();
}

// Process a single value and return tensor
ov::Tensor value_to_tensor(
    const py::object& value,
    InferRequestWrapper& wrapper,
    bool is_shared,
    const py::object& key
) {
    // Fast path: already a Tensor
    if (py::isinstance<ov::Tensor>(value)) {
        return value.cast<ov::Tensor>();
    }
    
    // numpy array - most common case
    if (py::isinstance<py::array>(value)) {
        return array_to_tensor(value.cast<py::array>(), wrapper, is_shared, key);
    }
    
    // Object with __array__ protocol (torch.Tensor, etc.)
    if (py::hasattr(value, "__array__")) {
        py::array arr;
        if (is_shared) {
            // Try to get array without copy
            arr = py::array::ensure(value.attr("__array__")());
        } else {
            // Force copy
            arr = value.attr("__array__")(py::none(), true).cast<py::array>();
        }
        return array_to_tensor(arr, wrapper, is_shared, key);
    }
    
    // Python list - create tensor directly
    if (py::isinstance<py::list>(value)) {
        py::object ov_tensor_cls = py::module_::import("openvino").attr("Tensor");
        return ov_tensor_cls(value).cast<ov::Tensor>();
    }
    
    // Scalars: int, float, str, bytes, numpy scalar
    if (py::isinstance<py::int_>(value) || py::isinstance<py::float_>(value) ||
        py::isinstance<py::str>(value) || py::isinstance<py::bytes>(value)) {
        
        auto tensor = get_request_tensor(wrapper, key);
        auto tensor_type = tensor.get_element_type();
        auto tensor_dtype = type_helpers::get_dtype(tensor_type);
        
        // Create 0-d array from scalar
        py::module_ np = py::module_::import("numpy");
        py::array tmp = np.attr("array")(value);
        
        if (tensor_type != ov::element::string) {
            auto tmp_dtype = tmp.dtype();
            if (!tmp_dtype.is(tensor_dtype)) {
                tmp = tmp.attr("astype")(tensor_dtype).cast<py::array>();
            }
        }
        py::object ov_tensor_cls = py::module_::import("openvino").attr("Tensor");
        return ov_tensor_cls(tmp, false).cast<ov::Tensor>();
    }
    
    // numpy scalar types
    py::module_ np = py::module_::import("numpy");
    if (py::isinstance(value, np.attr("number"))) {
        auto tensor = get_request_tensor(wrapper, key);
        auto tensor_dtype = type_helpers::get_dtype(tensor.get_element_type());
        py::array tmp = np.attr("array")(value);
        if (!tmp.dtype().is(tensor_dtype)) {
            tmp = tmp.attr("astype")(tensor_dtype).cast<py::array>();
        }
        py::object ov_tensor_cls = py::module_::import("openvino").attr("Tensor");
        return ov_tensor_cls(tmp, false).cast<ov::Tensor>();
    }
    
    throw py::type_error("Incompatible input type: " + std::string(py::str(value.get_type())));
}

// Update tensor in-place (for copy mode)
void update_tensor_inplace(
    const py::object& value,
    InferRequestWrapper& wrapper,
    const py::object& key
) {
    auto tensor = get_request_tensor(wrapper, key);
    
    py::array array;
    if (py::isinstance<py::array>(value)) {
        array = ensure_contiguous(value.cast<py::array>());
    } else if (py::hasattr(value, "__array__")) {
        array = ensure_contiguous(value.attr("__array__")().cast<py::array>());
    } else {
        // For non-array types, fall back to creating new tensor
        return;
    }
    
    // Handle scalars separately
    if (array.ndim() == 0) {
        return;
    }
    
    auto tensor_type = tensor.get_element_type();
    
    // String tensors use different path
    if (tensor_type == ov::element::string) {
        py::object py_tensor = py::cast(tensor);
        py_tensor.attr("bytes_data") = array;
        return;
    }
    
    // Get shape as vector
    std::vector<size_t> new_shape;
    for (py::ssize_t i = 0; i < array.ndim(); ++i) {
        new_shape.push_back(array.shape(i));
    }
    
    // Update shape if needed
    if (tensor.get_shape() != ov::Shape(new_shape)) {
        tensor.set_shape(ov::Shape(new_shape));
    }
    
    // Copy data
    py::object py_tensor = py::cast(tensor);
    auto tensor_dtype = type_helpers::get_dtype(tensor_type);
    auto arr_dtype = array.dtype();
    
    if (arr_dtype.is(tensor_dtype)) {
        py_tensor.attr("data")[py::slice(py::none(), py::none(), py::none())] = array;
    } else {
        py_tensor.attr("data")[py::slice(py::none(), py::none(), py::none())] = 
            array.attr("astype")(tensor_dtype);
    }
}

}  // anonymous namespace

// Main dispatch function for shared mode
py::object create_shared_dispatch(
    InferRequestWrapper& wrapper,
    const py::object& inputs
) {
    // None or empty
    if (inputs.is_none()) {
        return py::dict();
    }
    
    // Already a Tensor - pass through
    if (py::isinstance<ov::Tensor>(inputs)) {
        return inputs;
    }
    
    // Dict input
    if (py::isinstance<py::dict>(inputs)) {
        py::dict result;
        py::dict input_dict = inputs.cast<py::dict>();
        
        for (auto item : input_dict) {
            py::object key = py::reinterpret_borrow<py::object>(item.first);
            py::object value = py::reinterpret_borrow<py::object>(item.second);
            result[key] = py::cast(value_to_tensor(value, wrapper, true, key));
        }
        return result;
    }
    
    // Tuple input
    if (py::isinstance<py::tuple>(inputs)) {
        py::dict result;
        py::tuple input_tuple = inputs.cast<py::tuple>();
        
        for (size_t i = 0; i < input_tuple.size(); ++i) {
            py::object key = py::cast(i);
            py::object value = py::reinterpret_borrow<py::object>(input_tuple[i]);
            result[key] = py::cast(value_to_tensor(value, wrapper, true, key));
        }
        return result;
    }
    
    // List input
    if (py::isinstance<py::list>(inputs)) {
        py::list input_list = inputs.cast<py::list>();
        
        // Check if single-input model with simple type list
        size_t num_inputs = wrapper.m_inputs.size();
        if (num_inputs == 1 && is_simple_type_list(input_list)) {
            return py::cast(value_to_tensor(inputs, wrapper, true, py::none()));
        }
        
        py::dict result;
        for (size_t i = 0; i < input_list.size(); ++i) {
            py::object key = py::cast(i);
            py::object value = py::reinterpret_borrow<py::object>(input_list[i]);
            result[key] = py::cast(value_to_tensor(value, wrapper, true, key));
        }
        return result;
    }
    
    // Single array or value with __array__
    if (py::isinstance<py::array>(inputs) || py::hasattr(inputs, "__array__")) {
        return py::cast(value_to_tensor(inputs, wrapper, true, py::none()));
    }
    
    // Scalars
    if (py::isinstance<py::int_>(inputs) || py::isinstance<py::float_>(inputs) ||
        py::isinstance<py::str>(inputs) || py::isinstance<py::bytes>(inputs)) {
        return py::cast(value_to_tensor(inputs, wrapper, true, py::none()));
    }
    
    // numpy scalar
    py::module_ np = py::module_::import("numpy");
    if (py::isinstance(inputs, np.attr("number"))) {
        return py::cast(value_to_tensor(inputs, wrapper, true, py::none()));
    }
    
    throw py::type_error("Incompatible inputs type: " + std::string(py::str(inputs.get_type())));
}

// Main dispatch function for copy mode
py::object create_copied_dispatch(
    InferRequestWrapper& wrapper,
    const py::object& inputs
) {
    // None or empty
    if (inputs.is_none()) {
        return py::dict();
    }
    
    // Already a Tensor - pass through
    if (py::isinstance<ov::Tensor>(inputs)) {
        return inputs;
    }
    
    // Dict input
    if (py::isinstance<py::dict>(inputs)) {
        py::dict result;
        py::dict input_dict = inputs.cast<py::dict>();
        
        for (auto item : input_dict) {
            py::object key = py::reinterpret_borrow<py::object>(item.first);
            py::object value = py::reinterpret_borrow<py::object>(item.second);
            
            if (py::isinstance<ov::Tensor>(value)) {
                result[key] = value;
            } else if (py::isinstance<py::list>(value)) {
                py::object ov_tensor_cls = py::module_::import("openvino").attr("Tensor");
                result[key] = ov_tensor_cls(value);
            } else if (py::isinstance<py::array>(value) || py::hasattr(value, "__array__")) {
                update_tensor_inplace(value, wrapper, key);
            } else {
                result[key] = py::cast(value_to_tensor(value, wrapper, false, key));
            }
        }
        return result;
    }
    
    // Tuple input
    if (py::isinstance<py::tuple>(inputs)) {
        py::dict result;
        py::tuple input_tuple = inputs.cast<py::tuple>();
        
        for (size_t i = 0; i < input_tuple.size(); ++i) {
            py::object key = py::cast(i);
            py::object value = py::reinterpret_borrow<py::object>(input_tuple[i]);
            
            if (py::isinstance<ov::Tensor>(value)) {
                result[key] = value;
            } else if (py::isinstance<py::list>(value)) {
                py::object ov_tensor_cls = py::module_::import("openvino").attr("Tensor");
                result[key] = ov_tensor_cls(value);
            } else if (py::isinstance<py::array>(value) || py::hasattr(value, "__array__")) {
                update_tensor_inplace(value, wrapper, key);
            } else {
                result[key] = py::cast(value_to_tensor(value, wrapper, false, key));
            }
        }
        return result;
    }
    
    // List input
    if (py::isinstance<py::list>(inputs)) {
        py::list input_list = inputs.cast<py::list>();
        
        size_t num_inputs = wrapper.m_inputs.size();
        if (num_inputs == 1 && is_simple_type_list(input_list)) {
            py::object ov_tensor_cls = py::module_::import("openvino").attr("Tensor");
            return ov_tensor_cls(inputs);
        }
        
        py::dict result;
        for (size_t i = 0; i < input_list.size(); ++i) {
            py::object key = py::cast(i);
            py::object value = py::reinterpret_borrow<py::object>(input_list[i]);
            
            if (py::isinstance<ov::Tensor>(value)) {
                result[key] = value;
            } else if (py::isinstance<py::list>(value)) {
                py::object ov_tensor_cls = py::module_::import("openvino").attr("Tensor");
                result[key] = ov_tensor_cls(value);
            } else if (py::isinstance<py::array>(value) || py::hasattr(value, "__array__")) {
                update_tensor_inplace(value, wrapper, key);
            } else {
                result[key] = py::cast(value_to_tensor(value, wrapper, false, key));
            }
        }
        return result;
    }
    
    // Single array
    if (py::isinstance<py::array>(inputs) || py::hasattr(inputs, "__array__")) {
        update_tensor_inplace(inputs, wrapper, py::none());
        return py::dict();
    }
    
    // Scalars
    if (py::isinstance<ov::Tensor>(inputs) || 
        py::isinstance<py::int_>(inputs) || py::isinstance<py::float_>(inputs) ||
        py::isinstance<py::str>(inputs) || py::isinstance<py::bytes>(inputs)) {
        return py::cast(value_to_tensor(inputs, wrapper, false, py::none()));
    }
    
    // numpy scalar
    py::module_ np = py::module_::import("numpy");
    if (py::isinstance(inputs, np.attr("number"))) {
        return py::cast(value_to_tensor(inputs, wrapper, false, py::none()));
    }
    
    throw py::type_error("Incompatible inputs type: " + std::string(py::str(inputs.get_type())));
}

void regmodule_data_dispatch(py::module m) {
    auto dispatch_module = m.def_submodule("_data_dispatch", "Fast C++ data dispatcher");
    
    dispatch_module.def(
        "_data_dispatch_cpp",
        [](InferRequestWrapper& wrapper, const py::object& inputs, bool is_shared) {
            if (is_shared) {
                return create_shared_dispatch(wrapper, inputs);
            } else {
                return create_copied_dispatch(wrapper, inputs);
            }
        },
        py::arg("request"),
        py::arg("inputs"),
        py::arg("is_shared") = false,
        R"(
            Fast C++ implementation of data dispatch for inference inputs.
            
            :param request: InferRequest wrapper object
            :param inputs: Input data (dict, list, tuple, array, or Tensor)
            :param is_shared: If True, attempts zero-copy sharing
            :return: Dict of tensors or single tensor
        )"
    );
}

}  // namespace data_dispatch
}  // namespace Common
