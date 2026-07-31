#pragma once
#include "taskSystem/shared/hostDeviceCompilation.h"

typedef struct SiluTask {
  GLOBAL_DEVICE_PTR float* input;
  GLOBAL_DEVICE_PTR float* output;
  int size;
} SiluTask;

#ifdef DEVICE_COMPILATION
inline void ExecuteSiluTask(const SiluTask* task, __local char* slmBuffer) {
  volatile __global atomic_int* inputReady =
      (volatile __global atomic_int*)(task->input + task->size);
  while (atomic_load_explicit(inputReady, memory_order_acquire,
                              memory_scope_device) == 0) {
  }

  for (int i = get_local_id(0); i < task->size; i += get_local_size(0)) {
    const float value = task->input[i];
    task->output[i] = value / (1.0f + exp(-value));
  }
}
#endif