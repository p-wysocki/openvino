// Copyright (C) 2018-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "openvino/runtime/threading/cpu_streams_executor.hpp"
#include "cache/multi_cache.h"
#include "config.h"
#include "dnnl_scratch_pad.h"
#include "weights_cache.hpp"

namespace ov {
namespace intel_cpu {

class GraphContext {
public:
    typedef std::shared_ptr<GraphContext> Ptr;
    typedef std::shared_ptr<const GraphContext> CPtr;

    GraphContext(const Config& config,
                 WeightsSharing::Ptr w_cache,
                 bool isGraphQuantized,
                 ov::threading::IStreamsExecutor::Ptr streamExecutor = nullptr)
        : config(config),
          weightsCache(w_cache),
          isGraphQuantizedFlag(isGraphQuantized),
          streamExecutor(streamExecutor) {
        rtParamsCache = std::make_shared<MultiCache>(config.rtCacheCapacity);
        // primitive/executors can be shared across sub-stream
        // but scratch pad cannot be shared.
        numNumaNodes = 1;
        if (streamExecutor) {
            cpuStreamExecutor = std::dynamic_pointer_cast<ov::threading::CPUStreamsExecutor>(streamExecutor);
            auto nNumaNodes = get_num_numa_nodes();
            if (numNumaNodes < nNumaNodes)
                numNumaNodes = nNumaNodes;
        }
        for (int i = 0; i < numNumaNodes; i++) {
            rtScratchPads.push_back(std::make_shared<DnnlScratchPad>(getEngine(), i));
        }
    }

    const Config& getConfig() const {
        return config;
    }

    WeightsSharing::Ptr getWeightsCache() const {
        return weightsCache;
    }


    MultiCachePtr getParamsCache() const {
        return rtParamsCache;
    }

    DnnlScratchPadPtr getScratchPad(int subStreamID = 0) const {
        if (subStreamID < 0)
            subStreamID = 0;
        if (subStreamID >= numNumaNodes - 1)
            subStreamID = numNumaNodes - 1;
        return rtScratchPads[subStreamID];
    }

    const std::vector<DnnlScratchPadPtr>& getScratchPads() const {
        return rtScratchPads;
    }

    static const dnnl::engine& getEngine();

    bool isGraphQuantized() const {
        return isGraphQuantizedFlag;
    }

    ov::threading::CPUStreamsExecutor::Ptr getCPUStreamExecutor() const {
        return cpuStreamExecutor;
    }

    int getNumNumaNodes() const {
        return numNumaNodes;
    }

private:
    Config config;  // network-level config

    WeightsSharing::Ptr weightsCache;         // per NUMA node caches for sharing weights data

    MultiCachePtr rtParamsCache;     // primitive cache
    DnnlScratchPadPtr rtScratchPad;  // scratch pad

    bool isGraphQuantizedFlag = false;

    std::vector<DnnlScratchPadPtr> rtScratchPads;  // scratch pad (each sub-stream has its own copy)

    ov::threading::IStreamsExecutor::Ptr streamExecutor;   // stream executor for current graph

    ov::threading::CPUStreamsExecutor::Ptr cpuStreamExecutor;   // cpu stream executor for current graph

    int numNumaNodes = 1;
};

}  // namespace intel_cpu
}  // namespace ov
