// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "megakernel_decode_inst.h"

#include "intel_gpu/op/megakernel_decode.hpp"
#include "json_object.h"
#include "primitive_type_base.h"

#include <sstream>

namespace cldnn {

GPU_DEFINE_PRIMITIVE_TYPE_ID(megakernel_decode)

// ---------------------------------------------------------------------------
// Shape inference
// ---------------------------------------------------------------------------
// Outputs:
//   [0]  hidden_states_out  [B, S, hidden_size]
//   [1]  present_key        [num_layers, B, num_kv_heads, S_past + S, head_dim]
//   [2]  present_val        [num_layers, B, num_kv_heads, S_past + S, head_dim]

template <typename ShapeType>
std::vector<layout> megakernel_decode_inst::calc_output_layouts(megakernel_decode_node const& /*node*/,
                                                                 const kernel_impl_params& impl_param) {
    auto desc = impl_param.typed_desc<megakernel_decode>();

    const auto& hs_layout   = impl_param.get_input_layout(0);  // hidden_states
    const auto& past_layout = impl_param.get_input_layout(3);  // past_key [L,B,Kh,S_past,Hd]

    const auto out0_dt = data_types::f32;   // hidden_states_out
    const auto out_kv_dt = data_types::f16; // present_key / present_val — match KV variable dtype
    const auto fmt    = format::bfyx;   // 4D format for 3D hidden_states_out
    const auto fmt5d  = format::bfzyx;  // 5D format for present_key / present_val

    // Derive B, S from hidden_states
    const auto& hs_ps = hs_layout.get<ShapeType>();
    ShapeType B_dim   = hs_ps.rank().is_static() ? ShapeType{hs_ps[0]} : ShapeType{ov::Dimension::dynamic()};
    ShapeType S_dim   = hs_ps.rank().is_static() ? ShapeType{hs_ps[1]} : ShapeType{ov::Dimension::dynamic()};

    // Derive S_past from past_key dim 3
    const auto& pk_ps = past_layout.get<ShapeType>();
    ShapeType S_past  = (pk_ps.rank().is_static() && pk_ps.rank().get_length() >= 4)
                            ? ShapeType{pk_ps[3]}
                            : ShapeType{ov::Dimension::dynamic()};

    const int64_t L  = desc->num_layers;
    const int64_t H  = desc->hidden_size;
    const int64_t Kh = desc->num_kv_heads;
    const int64_t Hd = desc->head_dim;

    layout out0{ShapeType{B_dim[0], S_dim[0], ov::Dimension(H)},          out0_dt, fmt};
    layout out1{ShapeType{ov::Dimension(L), B_dim[0], ov::Dimension(Kh), S_past[0] + S_dim[0], ov::Dimension(Hd)}, out_kv_dt, fmt5d};
    layout out2 = out1;

    return {out0, out1, out2};
}

layout megakernel_decode_inst::calc_output_layout(megakernel_decode_node const& node,
                                                   kernel_impl_params const& impl_param) {
    return calc_output_layouts<ov::PartialShape>(node, impl_param)[0];
}

template std::vector<layout> megakernel_decode_inst::calc_output_layouts<ov::PartialShape>(
    megakernel_decode_node const& node,
    const kernel_impl_params& impl_param);

// ---------------------------------------------------------------------------
// to_string / constructor
// ---------------------------------------------------------------------------
std::string megakernel_decode_inst::to_string(megakernel_decode_node const& node) {
    auto desc = node.get_primitive();
    auto node_info = node.desc_to_json();
    std::stringstream ss;
    json_composite info;
    info.add("num_layers",   desc->num_layers);
    info.add("hidden_size",  desc->hidden_size);
    info.add("num_kv_heads", desc->num_kv_heads);
    info.add("head_dim",     desc->head_dim);
    node_info->add("megakernel_decode_info", info);
    node_info->dump(ss);
    return ss.str();
}

megakernel_decode_inst::typed_primitive_inst(network& network, megakernel_decode_node const& node)
    : parent(network, node) {}

}  // namespace cldnn
