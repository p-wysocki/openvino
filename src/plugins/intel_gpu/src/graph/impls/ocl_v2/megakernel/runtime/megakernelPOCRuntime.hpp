#pragma once

#include <CL/cl.h>
#include <CL/cl_ext.h>

#include "taskSystem/host/taskManagerHost.h"

namespace cldnn {
class primitive_inst;
};

namespace mk {

// Simplified runtime for MEGAKERNEL POC.
// For POC we want to have detailed control over the runtime, so
// we implement a custom runtime instead of using the standard runtime.
class MegaKernelPOCRuntime {
public:
    MegaKernelPOCRuntime() = default;
    ~MegaKernelPOCRuntime() = default;

    void Init(cldnn::primitive_inst& instance);
    void Execute(cldnn::primitive_inst& instance);
    void Deinit();

private:
    cl_context ctx_ = nullptr;
    cl_device_id dev_ = nullptr;
    cl_program prog_ = nullptr;
    cl_kernel kTask_ = nullptr;  // task-system worker kernel (whole model)
    // Intel USM extension entry points (resolved in Init).
    clDeviceMemAllocINTEL_fn usmAlloc_ = nullptr;
    clMemFreeINTEL_fn usmFree_ = nullptr;
    clEnqueueMemcpyINTEL_fn usmMemcpy_ = nullptr;
    clEnqueueMemFillINTEL_fn usmMemFill_ = nullptr;
    // USM device allocations: per-token scratch, KV cache, sync counters, context.
    void* mQb_ = nullptr;
    void* mKb_ = nullptr;
    void* mVb_ = nullptr;
    void* mGb_ = nullptr;
    void* mXn_ = nullptr;
    void* mH_ = nullptr;
    void* mNb_ = nullptr;
    void* mKC_ = nullptr;
    void* mVC_ = nullptr;
    void* mSync_ = nullptr;
    void* mCtx_ = nullptr;
    TaskManager taskManager_{};
    cl_mem mTaskMgr_ = nullptr;
};

}  // namespace mk