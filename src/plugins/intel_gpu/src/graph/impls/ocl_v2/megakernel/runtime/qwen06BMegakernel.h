#pragma once
#include "iMegakernelRuntime.h"
#include "megakernelPOCParams.h"

namespace mk {
// Create a MegaKernelPOCRuntime instance.
IMegakernelRuntime* CreateMegaKernelPOCRuntime();

// Destroy a MegaKernelPOCRuntime instance.
void DestroyMegaKernelPOCRuntime(IMegakernelRuntime* runtime);

}  // namespace mk