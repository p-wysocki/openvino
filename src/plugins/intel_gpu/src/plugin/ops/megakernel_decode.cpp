// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "intel_gpu/op/megakernel_decode.hpp"
#include "intel_gpu/plugin/common_utils.hpp"
#include "intel_gpu/plugin/program_builder.hpp"
#include "intel_gpu/primitives/megakernel_decode.hpp"

// Alias into the ov::op::internal namespace so REGISTER_FACTORY_IMPL can find it.
namespace ov {
namespace op {
namespace internal {
using MegaKernelDecode = ov::intel_gpu::op::MegaKernelDecode;
}  // namespace internal
}  // namespace op
}  // namespace ov

namespace ov::intel_gpu {

static void CreateMegaKernelDecodeOp(ProgramBuilder& p,
                                      const std::shared_ptr<ov::intel_gpu::op::MegaKernelDecode>& op) {
    std::cerr << "[CreateMegaKernelDecodeOp] start, inputs=" << op->get_input_size() << "\n";
    validate_inputs_count(op, {17});
    std::cerr << "[CreateMegaKernelDecodeOp] validated input count\n";
    auto inputs = p.GetInputInfo(op);
    std::cerr << "[CreateMegaKernelDecodeOp] got " << inputs.size() << " InputInfo objects\n";
    const auto& attrs = op->get_attrs();

    std::vector<cldnn::input_info> prim_inputs;
    prim_inputs.reserve(inputs.size());
    for (auto& ii : inputs)
        prim_inputs.push_back(cldnn::input_info(ii));
    std::cerr << "[CreateMegaKernelDecodeOp] built prim_inputs\n";

    auto prim = cldnn::megakernel_decode(
        layer_type_name_ID(op),
        prim_inputs,
        attrs.num_layers,
        attrs.hidden_size,
        attrs.num_kv_heads,
        attrs.head_dim);
    std::cerr << "[CreateMegaKernelDecodeOp] prim created\n";

    prim.output_data_types = get_output_data_types(op);
    std::cerr << "[CreateMegaKernelDecodeOp] output_data_types set, calling add_primitive\n";
    p.add_primitive(*op, prim);
    std::cerr << "[CreateMegaKernelDecodeOp] done\n";
}

REGISTER_FACTORY_IMPL(internal, MegaKernelDecode);

}  // namespace ov::intel_gpu
