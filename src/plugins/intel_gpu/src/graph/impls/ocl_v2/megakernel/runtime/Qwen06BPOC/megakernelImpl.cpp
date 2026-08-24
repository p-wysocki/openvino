#include "megakernelImpl.h"

#include "impl/qwen06BPOCRuntime.h"

namespace mk {
/////////////////////////////////////////////////////////////////////////
IMegakernelRuntime* CreateMegaKernelPOCRuntime() {
    return new mk::Qwen06BPOCRuntime();
}

/////////////////////////////////////////////////////////////////////////
void DestroyMegaKernelPOCRuntime(IMegakernelRuntime* runtime) {
    if (runtime)
        delete runtime;
}
};  // namespace mk
