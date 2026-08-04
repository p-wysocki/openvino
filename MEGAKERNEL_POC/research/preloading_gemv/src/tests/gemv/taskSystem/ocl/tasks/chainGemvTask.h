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
inline void WaitForGemvInput_block(const ChainGemvTask* task) {
  if (get_local_id(0) == 0 && task->inputReady != NULL) {
    volatile __global atomic_int* inputReady =
        (volatile __global atomic_int*)task->inputReady;
    while (atomic_load_explicit(inputReady, memory_order_acquire,
                                memory_scope_device) != task->inputTaskCount) {
    }
  }
  barrier(CLK_GLOBAL_MEM_FENCE);
}

inline void SignalGemvOutput_block(const ChainGemvTask* task) {
  barrier(CLK_GLOBAL_MEM_FENCE);
  if (get_local_id(0) == 0) {
    atomic_fetch_add_explicit((volatile __global atomic_int*)task->outputReady,
                              1, memory_order_release, memory_scope_device);
  }
}
#endif