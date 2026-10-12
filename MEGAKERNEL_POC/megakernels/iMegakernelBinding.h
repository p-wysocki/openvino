#pragma once

#include <cstdint>

#include "iMegakernelRuntime.h"

namespace mk {

// Lazy accessors to the MegaKernel op's graph ports (raw USM pointers and
// shapes), so a megakernel can bind them without depending on cldnn.
struct MegakernelIo {
  void* ctx;
  void* (*input_ptr)(void* ctx, int port);
  void* (*output_ptr)(void* ctx, int port);
  int64_t (*input_dim)(void* ctx, int port, int axis);
};

}  // namespace mk

// The port order is defined by the megakernel's own transformation, so mapping
// ports back onto its params belongs to the megakernel too. Every megakernel
// exports these from megakernelImpl.h:
//
//   extern "C" EXPORT_API void FillMegaKernelConstantParams(
//       mk::IConstantParams* params, const mk::MegakernelIo* io);
//   extern "C" EXPORT_API void FillMegaKernelRuntimeParams(
//       mk::IRuntimeParams* params, const mk::MegakernelIo* io);
