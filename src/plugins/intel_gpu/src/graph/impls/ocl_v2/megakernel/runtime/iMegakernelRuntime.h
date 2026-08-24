#pragma once

// Basic interface for a MegaKernel runtime.
namespace mk {
// Constant parameters (weights) for the runtime.
class IConstantParams {};

// Runtime parameters (inputs/outputs) for the runtime.
class IRuntimeParams {};

// Platform-specific parameters for the runtime(e.g., OpenCL context, device, etc.).
class IPlatformParams {};

// Interface for a MegaKernel runtime.
class IMegakernelRuntime {
public:
    // Initialize the runtime with constant parameters and platform-specific parameters.
    // This is where e.g. megakernel compilation and memory allocation would happen.
    virtual void Init(const IConstantParams* weights, const IPlatformParams* platform) = 0;

    // Execute(run inference) the runtime with the given input/output parameters.
    virtual void Execute(const IRuntimeParams* io) = 0;

    // Destroy the runtime and release any allocated resources.
    virtual void Destroy() = 0;

    virtual ~IMegakernelRuntime() = default;
};

}  // namespace mk