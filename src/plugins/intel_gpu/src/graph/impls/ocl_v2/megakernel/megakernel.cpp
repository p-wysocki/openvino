// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "megakernel.hpp"

#include "../common_utils/dispatch_utils.hpp"
#include "../common_utils/jitter.hpp"
#include "intel_gpu/primitives/megakernel.hpp"
#include "megakernel_inst.h"
#include "../primitive_ocl_base.hpp"
#include "../utils/kernel_generator.hpp"
#include "ocl_v2/utils/jitter.hpp"
#include "intel_gpu/runtime/memory.hpp"
#include "intel_gpu/graph/network.hpp"

namespace ov::intel_gpu::ocl {
namespace {

// ---------------------------------------------------------------------------
// KernelGenerator
// ---------------------------------------------------------------------------
// Kernel "megakernel_zero" is dispatched 2D:
//   global[0] = hidden_states_out element count   (output 0)
//   global[1] = present_key       element count   (output 1 == output 2)
// ---------------------------------------------------------------------------
class MegaKernelZeroGenerator : public KernelGenerator {
public:
    MegaKernelZeroGenerator() : KernelGenerator("megakernel", "zero") {}

protected:
    [[nodiscard]] Arguments get_arguments_desc(const RuntimeParams& params) const override {
        Arguments args;
        if (params.is_dynamic()) {
            args.push_back({ArgumentDescriptor::Types::SHAPE_INFO, 0});
        }
        // All 17 inputs (hidden_states … rope_inv_freq)
        for (uint32_t i = 0; i < 17; i++) {
            args.push_back({ArgumentDescriptor::Types::INPUT, i});
        }
        // 3 outputs (hidden_states_out, present_key, present_val)
        args.push_back({ArgumentDescriptor::Types::OUTPUT, 0});
        args.push_back({ArgumentDescriptor::Types::OUTPUT, 1});
        args.push_back({ArgumentDescriptor::Types::OUTPUT, 2});
        return args;
    }

    [[nodiscard]] JitConstants get_jit_constants(const RuntimeParams& params) const override {
        // Base defines: KERNEL(), IS_DYNAMIC, OPTIONAL_SHAPE_INFO_ARG, etc.
        auto jit = KernelGenerator::make_base_jit_constants(params);

        const auto& in_offsets_map  = params.in_port_to_shape_info_offset;
        const auto& out_offsets_map = params.out_port_to_shape_info_offset;

        // INPUT0..INPUT16 — all 17 inputs (hidden_states … rope_inv_freq)
        for (size_t i = 0; i < params.input_layouts.size(); i++) {
            jit.add(make_layout_jit_constants("INPUT" + to_code_string(i),
                                              params.input_layouts[i],
                                              in_offsets_map.at(i)));
        }

        // OUTPUT / OUTPUT1 / OUTPUT2 — hidden_states_out, present_key, present_val
        jit.add(make_layout_jit_constants("OUTPUT",
                                          params.output_layouts[0],
                                          out_offsets_map.at(0)));
        for (size_t i = 1; i < params.output_layouts.size(); i++) {
            jit.add(make_layout_jit_constants("OUTPUT" + to_code_string(i),
                                              params.output_layouts[i],
                                              out_offsets_map.at(i)));
        }

        // MegaKernel-specific compile-time constants available in the kernel.
        const auto& desc = params.typed_desc<cldnn::megakernel>();
        jit.make("MEGAKERNEL_NUM_LAYERS",    desc->num_layers);
        jit.make("MEGAKERNEL_HIDDEN_SIZE",   desc->hidden_size);
        jit.make("MEGAKERNEL_NUM_KV_HEADS",  desc->num_kv_heads);
        jit.make("MEGAKERNEL_HEAD_DIM",      desc->head_dim);
        jit.make("MEGAKERNEL_NUM_HEADS",     desc->num_heads);
        jit.make("MEGAKERNEL_INTERMEDIATE_SIZE", desc->intermediate_size);
        jit.make("MEGAKERNEL_RMS_EPS",       desc->rms_norm_eps);

        return jit;
    }

    [[nodiscard]] DispatchDataFunc get_dispatch_data_func() const override {
        return DispatchDataFunc{[](const RuntimeParams& params, KernelData& kd, ImplRuntimeParams*) {
            // The fused kernel runs as a SINGLE work-group; all 256 work-items
            // cooperate over the one decode token via __local memory + barriers.
            auto& wgs = kd.params.workGroups;
            constexpr size_t LWS = 256;
            wgs.global = {LWS, 1, 1};
            wgs.local  = {LWS, 1, 1};
        }};
    }
};

// ---------------------------------------------------------------------------
// PrimitiveImplOCL wrapper
// ---------------------------------------------------------------------------
class MegaKernelZeroImpl : public PrimitiveImplOCL {
public:
    DECLARE_OBJECT_TYPE_SERIALIZATION(ov::intel_gpu::ocl::MegaKernelZeroImpl)

    Stage::Ptr zero_stage = make_stage<MegaKernelZeroGenerator>();

    MegaKernelZeroImpl() : PrimitiveImplOCL(MegaKernelImpl::get_type_info_static()) {}

    explicit MegaKernelZeroImpl(const program_node& node, const RuntimeParams& params)
        : MegaKernelZeroImpl() {
        add_stage(zero_stage, params);
    }

    [[nodiscard]] std::unique_ptr<primitive_impl> clone() const override {
        return make_deep_copy<MegaKernelZeroImpl>(this);
    }

    // Override get_arguments to handle null past_key/past_val inputs (inputs 3 and 4).
    // During prefill the KV-cache variable buffers have zero sequence length and the
    // ReadValue node may not have allocated memory.  We substitute the corresponding
    // OUTPUT buffer (present_key / present_val) so the kernel argument slot is never
    // null.  The placeholder zero-fill kernel does not read past KV data anyway.
    [[nodiscard]] cldnn::kernel_arguments_data get_arguments(const cldnn::primitive_inst& instance) const override {
        cldnn::kernel_arguments_data args = PrimitiveImplOCL::get_arguments(instance);

        // Input 3 = past_key, Input 4 = past_val — substitute output 1/2 if null.
        if (args.inputs.size() > 4) {
            if (!args.inputs[3] && args.outputs.size() > 1 && args.outputs[1])
                args.inputs[3] = args.outputs[1];
            if (!args.inputs[4] && args.outputs.size() > 2 && args.outputs[2])
                args.inputs[4] = args.outputs[2];
        }

        return args;
    }
};

}  // namespace

// ---------------------------------------------------------------------------
// Factory entry point
// ---------------------------------------------------------------------------
std::unique_ptr<primitive_impl> MegaKernelImpl::create_impl(const program_node& node,
                                                                    const RuntimeParams& params) const {
    OPENVINO_ASSERT(node.is_type<megakernel>());
    return std::make_unique<MegaKernelZeroImpl>(node, params);
}

}  // namespace ov::intel_gpu::ocl

BIND_BINARY_BUFFER_WITH_TYPE(cldnn::megakernel)
BIND_BINARY_BUFFER_WITH_TYPE(ov::intel_gpu::ocl::MegaKernelZeroImpl)
