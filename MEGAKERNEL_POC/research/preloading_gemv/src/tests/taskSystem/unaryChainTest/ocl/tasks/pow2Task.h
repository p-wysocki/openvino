#pragma once
#include "taskSystem/shared/hostDeviceCompilation.h"

typedef struct Pow2Task {
  GLOBAL_DEVICE_PTR float* input;
  GLOBAL_DEVICE_PTR float* output;
  GLOBAL_DEVICE_PTR int* outputReady;
  int size;
} Pow2Task;

#ifdef DEVICE_COMPILATION
#include "common/semaphore.hcl"
inline void ExecuteTestTask(const Pow2Task* task, __local char* slmBuffer) {
  __local float* input_local = (__local float*)slmBuffer;
  for (int i = get_local_id(0); i < task->size; i += get_local_size(0)) {
    input_local[i] = task->input[i];
  }

  barrier(CLK_LOCAL_MEM_FENCE);
  for (int i = get_local_id(0); i < task->size; i += get_local_size(0)) {
    task->output[i] = input_local[i] * input_local[i];
  }

  SignalSemaphore_block(0, (volatile __global atomic_int*)task->outputReady);
}
#endif