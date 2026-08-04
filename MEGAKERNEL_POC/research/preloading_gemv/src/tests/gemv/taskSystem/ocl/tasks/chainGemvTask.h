#pragma once

#include "taskSystem/shared/hostDeviceCompilation.h"

#ifdef DEVICE_COMPILATION
typedef half ChainGemvTaskElement;
#else
#include <CL/cl_half.h>
typedef cl_half ChainGemvTaskElement;
#endif

typedef struct ChainGemvTask {
  GLOBAL_DEVICE_PTR const ChainGemvTaskElement* matrix;
  GLOBAL_DEVICE_PTR const ChainGemvTaskElement* vector;
  GLOBAL_DEVICE_PTR ChainGemvTaskElement* output;
  GLOBAL_DEVICE_PTR int* inputReady;
  GLOBAL_DEVICE_PTR int* outputReady;
  int inputTaskCount;
  int tileId;
} ChainGemvTask;

#ifdef DEVICE_COMPILATION
#include "common/semaphore.hcl"
inline void WaitForGemvInput_block(const ChainGemvTask* task) {
  WaitForSemaphore_block(0, (volatile __global atomic_int*)task->inputReady,
                         task->inputTaskCount);
}

inline void SignalGemvOutput_block(const ChainGemvTask* task) {
  SignalSemaphore_block(0, (volatile __global atomic_int*)task->outputReady);
}
#endif