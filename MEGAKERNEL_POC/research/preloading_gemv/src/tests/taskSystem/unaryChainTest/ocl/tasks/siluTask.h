#pragma once
#include "taskSystem/shared/hostDeviceCompilation.h"

typedef struct SiluTask {
  GLOBAL_DEVICE_PTR float* input;
  GLOBAL_DEVICE_PTR float* output;
  GLOBAL_DEVICE_PTR int* inputReady;
  int syncValue;
  int size;
} SiluTask;

#ifdef DEVICE_COMPILATION
#include "common/semaphore.hcl"
inline void ExecuteSiluTask(const SiluTask* task, __local char* slmBuffer) {
  WaitForSemaphore_block(0, (volatile __global atomic_int*)task->inputReady,
                         task->syncValue);

  for (int i = get_local_id(0); i < task->size; i += get_local_size(0)) {
    const float value = task->input[i];
    task->output[i] = value / (1.0f + exp(-value));
  }
}
#endif