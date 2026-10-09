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

// Bridge cldnn's primitive_inst to the model-agnostic mk::MegakernelIo accessor,
// so that the port -> parameter mapping can live with the megakernel itself.
void* usm_raw(cldnn::memory& m, int port) {
    auto at = m.get_allocation_type();
    bool usm = at == cldnn::allocation_type::usm_device || at == cldnn::allocation_type::usm_host || at == cldnn::allocation_type::usm_shared;
    OPENVINO_ASSERT(usm, "[MegaKernel] port ", port, " must be a USM allocation for the task-system path");
    return m.buffer_ptr();
}

// Accessor context: the instance plus USM copies of inputs that arrived in a
// non-USM allocation (e.g. a cl_mem tensor handed over by GenAI).
struct IoCtx {
    cldnn::primitive_inst* inst;
    const std::vector<cldnn::memory::ptr>* staging;
};

mk::MegakernelIo make_io(IoCtx& c) {
    auto& instance = *c.inst;
    mk::MegakernelIo io{};
    io.ctx = &c;
    io.input_ptr = [](void* ctx, int port) -> void* {
        auto& ic = *static_cast<IoCtx*>(ctx);
        if (ic.staging && static_cast<size_t>(port) < ic.staging->size() && (*ic.staging)[port])
            return (*ic.staging)[port]->buffer_ptr();
        return usm_raw(ic.inst->input_memory(port), port);
    };
    io.output_ptr = [](void* ctx, int port) -> void* {
        auto& inst = *static_cast<IoCtx*>(ctx)->inst;
        return inst.output_memory(port).buffer_ptr();
    };
    io.input_dim = [](void* ctx, int port, int axis) -> int64_t {
        auto& inst = *static_cast<IoCtx*>(ctx)->inst;
        return inst.input_memory(port).get_layout().get<ov::PartialShape>()[axis].get_length();
    };
    io.kind = instance.get_impl_params()->typed_desc<cldnn::megakernel>()->kind;
    io.num_inputs = static_cast<int64_t>(instance.inputs_memory_count());
    return io;
}

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
        IoCtx ic{&instance, nullptr};
        mk::MegakernelIo io = make_io(ic);

        mk::ConstantParamsImpl weights{};
        FillMegaKernelConstantParams(&weights, &io);

        // Specific to OpenCL platform, but will have to be generalized to any plaform supported by megakernel runtime.
        mk::PlatformParamsImpl platformParams{};
        platformParams.context = ctx;
        platformParams.deviceId = dl_device;
        platformParams.stream = queue;
        const mk::TErrorcode rc = megakernelRuntime_->Init(&weights, &platformParams);
        OPENVINO_ASSERT(rc == 0, "[MegaKernel] runtime Init failed: ", rc);
    }

    cldnn::event::ptr execute(const std::vector<cldnn::event::ptr>& events, cldnn::primitive_inst& instance) override {
        ensure_ready(instance);

        auto& strm = instance.get_network().get_stream();
        auto& ocls = downcast<ocl_stream>(strm);
        cl_command_queue q = ocls.get_cl_queue().get();

        // An in-order queue already serialises us behind our producers; waiting
        // on the host here would drain the queue once per megakernel.
        if (strm.get_queue_type() != QueueTypes::in_order) {
            for (auto& e : events)
                strm.wait_for_events({e});
        }

        // Runtime inputs that are not USM (cl_mem) are copied into USM staging
        // buffers on the same queue; constants were validated as USM in Init.
        const size_t n_in = instance.inputs_memory_count();
        staging_.resize(n_in);
        for (size_t p = 0; p < n_in; ++p) {
            auto& m = instance.input_memory(p);
            if (cldnn::memory_capabilities::is_usm_type(m.get_allocation_type())) {
                staging_[p] = nullptr;
                continue;
            }
            if (!staging_[p] || staging_[p]->size() < m.size()) {
                cldnn::layout l({static_cast<int64_t>(m.size())}, ov::element::u8, cldnn::format::bfyx);
                staging_[p] = instance.get_network().get_engine().allocate_memory(l, cldnn::allocation_type::usm_device, false);
            }
            staging_[p]->copy_from(strm, m, 0, 0, m.size(), strm.get_queue_type() != QueueTypes::in_order);
        }

        IoCtx ic{&instance, &staging_};
        mk::MegakernelIo io_acc = make_io(ic);
        mk::RuntimeParamsImpl io{};
        FillMegaKernelRuntimeParams(&io, &io_acc);

        const mk::TErrorcode rc = megakernelRuntime_->Execute(&io);
        OPENVINO_ASSERT(rc == 0, "[MegaKernel] runtime Execute failed: ", rc);

        if (strm.get_queue_type() == QueueTypes::in_order && !instance.needs_completion_event() && !instance.is_output())
            return ocls.create_base_event();
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
    std::mutex mu_;
    mk::IMegakernelRuntime* megakernelRuntime_ = nullptr;
    std::vector<cldnn::memory::ptr> staging_;
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