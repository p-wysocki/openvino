// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//
// MegaKernel plugin implementation — task-system-scheduled decoder.
// The entire Qwen3 decoder (all layers) runs in ONE kernel launch per token, but
// instead of a persistent grid + grid-wide software barrier, the work is driven
// by the fine-tuned GPU task system (MEGAKERNEL_POC/research/preloading_gemv):
// a pool of persistent worker work-groups pulls topologically-sorted tasks FIFO
// from a shared work queue; each layer stage is a set of per-workgroup tiles, and
// inter-stage ordering (the old grid barrier) is expressed as global atomic
// sync-flag dependencies resolved by the tasks themselves. Decode is one launch;
// prefill loops it per token. Key techniques: intel_sub_group_block_read, fused
// RMSNorm, fused RoPE, workgroup-cooperative flash-decoding attention, split-K GEMV.

#include "megakernel.hpp"

#include <CL/cl.h>
#include <CL/cl_ext.h>

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <mutex>
#include <vector>

#include "../primitive_ocl_base.hpp"
#include "intel_gpu/graph/network.hpp"
#include "intel_gpu/op/megakernel.hpp"
#include "intel_gpu/primitives/megakernel.hpp"
#include "intel_gpu/runtime/memory.hpp"
#include "megakernelImpl.h"
#include "megakernel_inst.h"
#include "ocl/ocl_engine.hpp"
#include "ocl/ocl_event.hpp"
#include "ocl/ocl_stream.hpp"

namespace ov::intel_gpu::ocl {

using cldnn::ocl::ocl_engine;
using cldnn::ocl::ocl_event;
using cldnn::ocl::ocl_stream;

namespace {
// ---------------------------------------------------------------------------
// MegaKernelFastImpl
// ---------------------------------------------------------------------------

class MegaKernelFastImpl : public cldnn::primitive_impl {
public:
    DECLARE_OBJECT_TYPE_SERIALIZATION(ov::intel_gpu::ocl::MegaKernelFastImpl)

    MegaKernelFastImpl() = default;
    explicit MegaKernelFastImpl(const cldnn::program_node&, const RuntimeParams&) {}
    // Copy constructor: copy the primitive_impl base subobject so metadata such
    // as m_manager and the dynamic flag are preserved (required by the impl
    // caches in ImplementationsFactory). The OpenCL/runtime members below keep
    // their default null/zero initializers so device state is re-created lazily.
    MegaKernelFastImpl(const MegaKernelFastImpl& other) : cldnn::primitive_impl(other) {}

    [[nodiscard]] std::unique_ptr<cldnn::primitive_impl> clone() const override {
        OPENVINO_ASSERT(megakernelRuntime_ == nullptr,
                        "[GPU] MegaKernelFastImpl::clone() should not be called if megakernel runtime is initialized; use create_impl() instead.");
        return std::make_unique<MegaKernelFastImpl>(*this);
    }
    bool is_cpu() const override {
        return false;
    }
    void save(BinaryOutputBuffer&) const override {}
    void load(BinaryInputBuffer&) override {}
    void init_kernels(const cldnn::kernels_cache&, const cldnn::kernel_impl_params&) override {}
    void set_arguments(cldnn::primitive_inst&) override {}
    void set_arguments(cldnn::primitive_inst&, cldnn::kernel_arguments_data&) override {}
    std::vector<cldnn::BufferDescriptor> get_internal_buffer_descs(const cldnn::kernel_impl_params&) const override {
        return {};
    }

    void ensure_ready(cldnn::primitive_inst& instance) {
        std::lock_guard<std::mutex> g(mu_);
        if (megakernelRuntime_ != nullptr) {
            return;
        }

        megakernelRuntime_ = CreateMegaKernelPOCRuntime();

        auto& eng = cldnn::downcast<cldnn::ocl::ocl_engine>(instance.get_network().get_engine());
        cl_context ctx = eng.get_cl_context().get();
        cl_device_id dl_device = eng.get_cl_device().get();
        cl_command_queue queue = cldnn::downcast<cldnn::ocl::ocl_stream>(instance.get_network().get_stream()).get_cl_queue().get();

        // Resolve the raw USM device pointers for every model input/output. The
        // task-system tasks dereference these directly out of the context struct,
        // which requires genuine USM device allocations (asserted below).
        auto usm_raw = [](cldnn::memory& m, const char* name) -> void* {
            auto at = m.get_allocation_type();
            bool usm = at == cldnn::allocation_type::usm_device || at == cldnn::allocation_type::usm_host || at == cldnn::allocation_type::usm_shared;
            OPENVINO_ASSERT(usm, "[MegaKernel] input/output '", name, "' must be a USM allocation for the task-system path");
            return m.buffer_ptr();
        };

        // TODO: Specific to Qwen06BPOC, but will have to be generalized to any megakernel runtime with a known constant parameter type.
        mk::ConstantParamsImpl weights{};
        weights.q_proj_w = usm_raw(instance.input_memory(op::mk_port::WEIGHTS + 0), "q_proj_w");
        weights.k_proj_w = usm_raw(instance.input_memory(op::mk_port::WEIGHTS + 1), "k_proj_w");
        weights.v_proj_w = usm_raw(instance.input_memory(op::mk_port::WEIGHTS + 2), "v_proj_w");
        weights.o_proj_w = usm_raw(instance.input_memory(op::mk_port::WEIGHTS + 3), "o_proj_w");
        weights.gate_proj_w = usm_raw(instance.input_memory(op::mk_port::WEIGHTS + 4), "gate_proj_w");
        weights.up_proj_w = usm_raw(instance.input_memory(op::mk_port::WEIGHTS + 5), "up_proj_w");
        weights.down_proj_w = usm_raw(instance.input_memory(op::mk_port::WEIGHTS + 6), "down_proj_w");
        weights.input_ln_w = usm_raw(instance.input_memory(op::mk_port::WEIGHTS + 7), "input_ln_w");
        weights.post_attn_ln_w = usm_raw(instance.input_memory(op::mk_port::WEIGHTS + 8), "post_attn_ln_w");
        weights.q_norm_w = usm_raw(instance.input_memory(op::mk_port::WEIGHTS + 9), "q_norm_w");
        weights.k_norm_w = usm_raw(instance.input_memory(op::mk_port::WEIGHTS + 10), "k_norm_w");
        weights.rope_inv_freq = usm_raw(instance.input_memory(op::mk_port::WEIGHTS + 11), "rope_inv_freq");
        // ---

        // Specific to OpenCL platform, but will have to be generalized to any plaform supported by megakernel runtime.
        mk::PlatformParamsImpl platformParams{};
        platformParams.context = ctx;
        platformParams.deviceId = dl_device;
        platformParams.stream = queue;
        megakernelRuntime_->Init(&weights, &platformParams);
    }

    cldnn::event::ptr execute(const std::vector<cldnn::event::ptr>& events, cldnn::primitive_inst& instance) override {
        ensure_ready(instance);

        auto& strm = instance.get_network().get_stream();
        auto& ocls = downcast<ocl_stream>(strm);
        cl_command_queue q = ocls.get_cl_queue().get();

        for (auto& e : events)
            strm.wait_for_events({e});  // inputs ready before we read them

        OPENVINO_ASSERT(instance.input_memory(1).get_layout().data_type == cldnn::data_types::i64,
                        "[MegaKernel] supports only i64 position_ids (input 1) for the task-system path");

        const int new_tokens = (int)instance.input_memory(0).get_layout().get<ov::PartialShape>()[1].get_length();
        const bool import_past = collect_past_kv(instance);

        // TODO: Specific to Qwen06BPOC, but will have to be generalized to any megakernel runtime with a known runtime parameter type.
        mk::RuntimeParamsImpl io{};
        io.hidden_states = instance.input_memory(0).buffer_ptr();
        io.position_ids = instance.input_memory(1).buffer_ptr();
        io.hidden_states_out = instance.output_memory(0).buffer_ptr();
        io.newTokens = new_tokens;
        if (import_past) {
            io.past_key = past_key_.data();
            io.past_value = past_value_.data();
            io.past_key_stride = past_key_stride_.data();
            io.past_value_stride = past_value_stride_.data();
            io.import_past_len = past_len_;
        }
        // ---

        megakernelRuntime_->Execute(&io);

        cl_event marker;
        clEnqueueMarkerWithWaitList(q, 0, nullptr, &marker);
        return std::make_shared<ocl_event>(cl::Event(marker, false), 0ULL);
    }

    ~MegaKernelFastImpl() override {
        if (megakernelRuntime_)
            DestroyMegaKernelPOCRuntime(megakernelRuntime_);
        megakernelRuntime_ = nullptr;
    }

private:
    // Two-model PoC: hand the prefill KV cache over to the MegaKernel's internal cache.
    //
    // The decode model has no KV ReadValue/Assign nodes, so nothing but the host ever
    // writes its variable states: they are set exactly once per sequence, when the prefill
    // request's cache is copied over. Consuming that "is set" flag therefore triggers the
    // import on the first decode step of every sequence and on no other step. Afterwards
    // the MegaKernel keeps growing its own cache and the states stay untouched.
    bool collect_past_kv(cldnn::primitive_inst& instance) {
        const auto& ids = instance.get_typed_desc<cldnn::megakernel>()->kv_variable_ids;
        const size_t num_layers = ids.size() / 2;
        auto& network = instance.get_network();

        auto& key0 = network.get_variable(ids[0]);
        if (!key0.is_set() || key0.get_memory() == nullptr)
            return false;
        past_len_ = static_cast<int>(key0.get_layout().get_dims()[2]);
        if (past_len_ == 0)
            return false;

        past_key_.resize(num_layers);
        past_value_.resize(num_layers);
        past_key_stride_.resize(num_layers);
        past_value_stride_.resize(num_layers);
        for (size_t l = 0; l < num_layers; ++l) {
            const auto collect = [&](const std::string& id, const void*& ptr, int& stride) {
                auto& variable = network.get_variable(id);
                const auto& layout = variable.get_layout();
                OPENVINO_ASSERT(layout.data_type == cldnn::data_types::f16,
                                "[MegaKernel] past KV cache must be f16, got ", layout.data_type);
                ptr = variable.get_memory()->buffer_ptr();
                // The sequence dimension may be over-allocated; the tokens of one head are
                // laid out contiguously with this stride.
                stride = static_cast<int>(layout.get_padded_dims()[2]);
                variable.unset();
            };
            collect(ids[l], past_key_[l], past_key_stride_[l]);
            collect(ids[num_layers + l], past_value_[l], past_value_stride_[l]);
        }
        return true;
    }

    std::mutex mu_;
    mk::IMegakernelRuntime* megakernelRuntime_ = nullptr;
    std::vector<const void*> past_key_, past_value_;
    std::vector<int> past_key_stride_, past_value_stride_;
    int past_len_ = 0;
};

}  // namespace

// ---------------------------------------------------------------------------
// Factory
// ---------------------------------------------------------------------------
std::unique_ptr<cldnn::primitive_impl> MegaKernelImpl::create_impl(const cldnn::program_node& node, const RuntimeParams& params) const {
    OPENVINO_ASSERT(node.is_type<cldnn::megakernel>());
    return std::make_unique<MegaKernelFastImpl>(node, params);
}

}  // namespace ov::intel_gpu::ocl

BIND_BINARY_BUFFER_WITH_TYPE(cldnn::megakernel)
BIND_BINARY_BUFFER_WITH_TYPE(ov::intel_gpu::ocl::MegaKernelFastImpl)