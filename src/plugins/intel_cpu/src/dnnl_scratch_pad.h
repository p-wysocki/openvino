// Copyright (C) 2018-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <memory>

#include "cpu_memory.h"
#include "utils/general_utils.h"

namespace ov {
namespace intel_cpu {

class DnnlScratchPad {
    MemoryMngrPtr mgrPtr;
    dnnl::engine eng;

public:
    DnnlScratchPad(dnnl::engine eng, int numa_node = -1) : eng(eng) {
        mgrPtr = std::make_shared<DnnlMemoryMngr>(make_unique<MemoryMngrWithReuse>(numa_node));
    }

    MemoryPtr createScratchPadMem(const MemoryDescPtr& md) {
        return std::make_shared<Memory>(eng, md, mgrPtr);
    }
};

using DnnlScratchPadPtr = std::shared_ptr<DnnlScratchPad>;

}  // namespace intel_cpu
}  // namespace ov
