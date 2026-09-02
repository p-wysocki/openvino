// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "intel_gpu/op/megakernel.hpp"

namespace ov::intel_gpu::op {

MegaKernel::MegaKernel(const ov::OutputVector& inputs, const MegaKernelAttrs& attrs,
                       MegaKernelKvVariables kv_variables)
    : Op(inputs),
      m_attrs(attrs),
      m_kv_variables(std::move(kv_variables)) {
    validate_and_infer_types();
}

bool MegaKernel::visit_attributes(ov::AttributeVisitor& visitor) {
    visitor.on_attribute("num_layers",          m_attrs.num_layers);
    visitor.on_attribute("hidden_size",         m_attrs.hidden_size);
    visitor.on_attribute("num_attention_heads", m_attrs.num_attention_heads);
    visitor.on_attribute("num_kv_heads",        m_attrs.num_kv_heads);
    visitor.on_attribute("head_dim",            m_attrs.head_dim);
    visitor.on_attribute("intermediate_size",   m_attrs.intermediate_size);
    visitor.on_attribute("rms_norm_eps",        m_attrs.rms_norm_eps);
    return true;
}

void MegaKernel::validate_and_infer_types() {
    // Port 0: hidden_states [B, S, hidden_size]
    const auto& hs_ps   = get_input_partial_shape(0);

    const int64_t H = m_attrs.hidden_size;

    ov::Dimension B = ov::Dimension::dynamic();
    ov::Dimension S = ov::Dimension::dynamic();

    if (hs_ps.rank().is_static()) {
        if (hs_ps.rank().get_length() >= 1) B = hs_ps[0];
        if (hs_ps.rank().get_length() >= 2) S = hs_ps[1];
    }

    const auto out_et = ov::element::f32;
    // Single output: hidden_states_out [B, S, hidden_size] (f32, matches model.norm).
    // The KV cache is kept entirely inside the plugin impl (MegaKernelFastImpl), so
    // the op exposes no KV outputs and the pass drops the KV-cache Assign sinks.
    set_output_type(0, out_et, ov::PartialShape{B, S, H});
}

std::shared_ptr<ov::Node> MegaKernel::clone_with_new_inputs(const ov::OutputVector& new_args) const {
    check_new_args_count(this, new_args);
    return std::make_shared<MegaKernel>(new_args, m_attrs, m_kv_variables);
}

}  // namespace ov::intel_gpu::op
