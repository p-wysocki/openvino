#include "qwen06BMegakernel.h"
#include "impl/megakernelPOCRuntime.hpp"

namespace mk {
/////////////////////////////////////////////////////////////////////////
IMegakernelRuntime* CreateMegaKernelPOCRuntime() {
    return new mk::MegaKernelPOCRuntime();
}

/////////////////////////////////////////////////////////////////////////
void DestroyMegaKernelPOCRuntime(IMegakernelRuntime* runtime) {
    if (runtime)
        delete runtime;
}
};  // namespace mk
