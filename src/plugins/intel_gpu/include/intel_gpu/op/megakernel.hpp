// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "openvino/op/op.hpp"

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

class MegaKernel : public ov::op::Op {
public:
    OPENVINO_OP("MegaKernel", "gpu_opset");

    MegaKernel() = default;

    MegaKernel(const ov::OutputVector& inputs, const MegaKernelAttrs& attrs);

    bool visit_attributes(ov::AttributeVisitor& visitor) override;
    void validate_and_infer_types() override;
    std::shared_ptr<Node> clone_with_new_inputs(const ov::OutputVector& new_args) const override;

    const MegaKernelAttrs& get_attrs() const { return m_attrs; }

private:
    MegaKernelAttrs m_attrs;
};

}  // namespace ov::intel_gpu::op
