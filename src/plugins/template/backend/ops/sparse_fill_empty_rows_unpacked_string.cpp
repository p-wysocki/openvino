// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "openvino/reference/sparse_fill_empty_rows.hpp"

#include "element_visitor.hpp"
#include "evaluate_node.hpp"
#include "sparse_fill_empty_rows_unpacked_string_shape_inference.hpp"


template <ov::element::Type_t ET_idx>
bool evaluate_sparse_fill_empty_rows_unpacked_string(
    const std::shared_ptr<ov::op::v16::SparseFillEmptyRowsUnpackedString>& op,
    ov::TensorVector& outputs,
    const ov::TensorVector& inputs) {
    
    using T_idx = typename ov::element_type_traits<ET_idx>::value_type;

    auto input_shapes = std::vector<ov::PartialShape>{
        op->get_input_shape(0),  // begins
        op->get_input_shape(1),  // ends
        op->get_input_shape(2),  // symbols
        op->get_input_shape(3),  // indices
        op->get_input_shape(4),  // dense_shape
        op->get_input_shape(5)   // default_value
    };

    const auto output_shapes = ov::op::v16::shape_infer(op.get(), input_shapes, make_tensor_accessor(inputs));
    outputs[0].set_shape(output_shapes[0].to_shape());  // output_begins
    outputs[1].set_shape(output_shapes[1].to_shape());  // output_ends
    outputs[2].set_shape(output_shapes[2].to_shape());  // output_symbols
    outputs[3].set_shape(output_shapes[3].to_shape());  // output_indices
    outputs[4].set_shape(output_shapes[4].to_shape());  // empty_row_indicator

    std::cout << "output_begins shape: " << outputs[0].get_shape() << std::endl;
    std::cout << "output_ends shape: " << outputs[1].get_shape() << std::endl;
    std::cout << "output_symbols shape: " << outputs[2].get_shape() << std::endl;
    std::cout << "output_indices shape: " << outputs[3].get_shape() << std::endl;
    std::cout << "empty_row_indicator shape: " << outputs[4].get_shape() << std::endl;

    const T_idx* begins = inputs[0].data<const T_idx>();
    const T_idx* ends = inputs[1].data<const T_idx>();
    const uint8_t* symbols = inputs[2].data<const uint8_t>();
    const T_idx* indices = inputs[3].data<const T_idx>();
    const T_idx* dense_shape = inputs[4].data<const T_idx>();
    const uint8_t* default_value = inputs[5].data<const uint8_t>();

    T_idx* output_begins = outputs[0].data<T_idx>();
    T_idx* output_ends = outputs[1].data<T_idx>();
    uint8_t* output_symbols = outputs[2].data<uint8_t>();
    T_idx* output_indices = outputs[3].data<T_idx>();
    bool* empty_row_indicator = outputs[4].data<bool>();

    const size_t num_values = inputs[0].get_shape()[0];
    const size_t default_value_size = inputs[5].get_shape()[0];
    const size_t symbols_size = inputs[2].get_shape()[0];

    ov::reference::sparse_fill_empty_rows_unpacked_string(
        begins,
        ends,
        symbols,
        indices,
        dense_shape,
        default_value,
        default_value_size,
        num_values,
        symbols_size,
        output_begins,
        output_ends,
        output_indices,
        output_symbols,
        empty_row_indicator);

    return true;
}

template <>
bool evaluate_node<ov::op::v16::SparseFillEmptyRowsUnpackedString>(
    std::shared_ptr<ov::Node> node,
    ov::TensorVector& outputs,
    const ov::TensorVector& inputs) {
    
    using ov::op::v16::SparseFillEmptyRowsUnpackedString;
    using namespace ov::element;

    auto op = ov::as_type_ptr<SparseFillEmptyRowsUnpackedString>(node);
    const auto& index_type = op->get_input_element_type(0);

    switch (index_type) {
    case i32:
        return evaluate_sparse_fill_empty_rows_unpacked_string<i32>(op, outputs, inputs);
    case i64:
        return evaluate_sparse_fill_empty_rows_unpacked_string<i64>(op, outputs, inputs);
    default:
        OPENVINO_THROW("Unhandled index type ", index_type, 
                       " in evaluate_node() for SparseFillEmptyRowsUnpackedString");
    }
}
