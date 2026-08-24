#pragma once
#include "iMegakernelRuntime.h"
#include "qwen06BMegakernel.h"

namespace mk {
// Factory functions to create and destroy a MegaKernelPOCRuntime instance.
IMegakernelRuntime* CreateMegaKernelPOCRuntime();
void DestroyMegaKernelPOCRuntime(IMegakernelRuntime* runtime);

}  // namespace mk