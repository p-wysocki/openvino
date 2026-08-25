#pragma once
#include "../iMegakernelRuntime.h"
#include "qwen06BPOCParams.h"
#include "qwen06BPOCExport.h"

// Create a MegaKernelPOCRuntime instance.
extern "C" QWEN06BPOC_API mk::IMegakernelRuntime* CreateMegaKernelPOCRuntime();

// Destroy a MegaKernelPOCRuntime instance.
extern "C" QWEN06BPOC_API void DestroyMegaKernelPOCRuntime(mk::IMegakernelRuntime* runtime);
