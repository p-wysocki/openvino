// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "openvino/op/op.hpp"

#include <string>
#include <vector>

namespace ov::intel_gpu::op {

struct MegaKernelAttrs {
    int64_t num_layers           = 28;
    int64_t hidden_size          = 1024;
    int64_t num_attention_heads  = 16;
    int64_t num_kv_heads         = 8;
    int64_t head_dim             = 128;
    int64_t intermediate_size    = 3072;
    float   rms_norm_eps         = 1e-6f;
};

// Input port map, shared by InsertMegaKernel (which builds the node) and the
// cldnn impl (which reads the memories back).
namespace mk_port {
constexpr size_t HIDDEN       = 0;
constexpr size_t POSITION_IDS = 1;
constexpr size_t BEAM_IDX     = 2;
constexpr size_t WEIGHTS      = 3;                        // 12 stacked weight tensors
constexpr size_t NUM_WEIGHTS  = 12;
constexpr size_t COUNT        = WEIGHTS + NUM_WEIGHTS;
}  // namespace mk_port

// KV-cache variables the MegaKernel takes over from the model's ReadValue/Assign pairs.
// They are no longer part of the graph — keeping 56 ReadValues alive costs ~0.9 ms per
// decode step — but they stay registered as OpenVINO variable states so that the prefill
// model's cache can be handed over through set_state(). The impl reads them straight from
// the network by id and imports them into the MegaKernel's own cache.
struct MegaKernelKvVariables {
    std::vector<std::string> ids;  // num_layers key ids followed by num_layers value ids
    ov::PartialShape shape;
    ov::element::Type type;
    ov::element::Type user_type;  // element type the application sees through get_state()
};

class MegaKernel : public ov::op::Op {
public:
    OPENVINO_OP("MegaKernel", "gpu_opset");

    MegaKernel() = default;

    MegaKernel(const ov::OutputVector& inputs, const MegaKernelAttrs& attrs,
               MegaKernelKvVariables kv_variables = {});

    bool visit_attributes(ov::AttributeVisitor& visitor) override;
    void validate_and_infer_types() override;
    std::shared_ptr<Node> clone_with_new_inputs(const ov::OutputVector& new_args) const override;

    const MegaKernelAttrs& get_attrs() const { return m_attrs; }
    const MegaKernelKvVariables& get_kv_variables() const { return m_kv_variables; }

private:
    MegaKernelAttrs m_attrs;
    MegaKernelKvVariables m_kv_variables;
};

}  // namespace ov::intel_gpu::op
