// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "megakernel_inst.h"

#include "intel_gpu/op/megakernel.hpp"
#include "json_object.h"
#include "primitive_type_base.h"

#include <sstream>

namespace cldnn {

GPU_DEFINE_PRIMITIVE_TYPE_ID(megakernel)

// ---------------------------------------------------------------------------
// Shape inference
// ---------------------------------------------------------------------------
// Outputs:
//   [0]  hidden_states_out  [B, S, hidden_size]
//   [1]  present_key        [num_layers, B, num_kv_heads, S_past + S, head_dim]
//   [2]  present_val        [num_layers, B, num_kv_heads, S_past + S, head_dim]

template <typename ShapeType>
std::vector<layout> megakernel_inst::calc_output_layouts(megakernel_node const& /*node*/,
                                                                 const kernel_impl_params& impl_param) {
    auto desc = impl_param.typed_desc<megakernel>();

    const auto& hs_layout   = impl_param.get_input_layout(0);  // hidden_states

    const auto out0_dt = data_types::f32;   // hidden_states_out
    const auto fmt    = format::bfyx;   // 4D format for 3D hidden_states_out

    // Derive B, S from hidden_states
    const auto& hs_ps = hs_layout.get<ShapeType>();
    ShapeType B_dim   = hs_ps.rank().is_static() ? ShapeType{hs_ps[0]} : ShapeType{ov::Dimension::dynamic()};
    ShapeType S_dim   = hs_ps.rank().is_static() ? ShapeType{hs_ps[1]} : ShapeType{ov::Dimension::dynamic()};

    const int64_t H  = desc->hidden_size;

    // Single output: hidden_states_out [B, S, H]. KV cache is internal to the impl.
    std::vector<layout> outs;
    outs.emplace_back(ShapeType{B_dim[0], S_dim[0], ov::Dimension(H)}, out0_dt, fmt);
    return outs;
}

layout megakernel_inst::calc_output_layout(megakernel_node const& node,
                                                   kernel_impl_params const& impl_param) {
    return calc_output_layouts<ov::PartialShape>(node, impl_param)[0];
}

template std::vector<layout> megakernel_inst::calc_output_layouts<ov::PartialShape>(
    megakernel_node const& node,
    const kernel_impl_params& impl_param);

// ---------------------------------------------------------------------------
// to_string / constructor
// ---------------------------------------------------------------------------
std::string megakernel_inst::to_string(megakernel_node const& node) {
    auto desc = node.get_primitive();
    auto node_info = node.desc_to_json();
    std::stringstream ss;
    json_composite info;
    info.add("num_layers",   desc->num_layers);
    info.add("hidden_size",  desc->hidden_size);
    info.add("num_kv_heads", desc->num_kv_heads);
    info.add("head_dim",     desc->head_dim);
    node_info->add("megakernel_info", info);
    node_info->dump(ss);
    return ss.str();
}

megakernel_inst::typed_primitive_inst(network& network, megakernel_node const& node)
    : parent(network, node) {}

}  // namespace cldnn
