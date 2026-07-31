#pragma once
#include "taskSystem/shared/hostDeviceCompilation.h"

typedef struct Pow2Task {
  GLOBAL_DEVICE_PTR float* input;
  GLOBAL_DEVICE_PTR float* output;
  int size;
} Pow2Task;

#ifdef DEVICE_COMPILATION
inline void ExecuteTestTask(const Pow2Task* task, __local char* slmBuffer) {
  __local float* input_local = (__local float*)slmBuffer;
  for (int i = get_local_id(0); i < task->size; i += get_local_size(0)) {
    input_local[i] = task->input[i];
  }

  barrier(CLK_LOCAL_MEM_FENCE);
  for (int i = get_local_id(0); i < task->size; i += get_local_size(0)) {
    task->output[i] = input_local[i] * input_local[i];
  }

  barrier(CLK_GLOBAL_MEM_FENCE);
  if (get_local_id(0) == 0) {
    atomic_store_explicit(
        (volatile __global atomic_int*)(task->output + task->size), 1,
        memory_order_release, memory_scope_device);
  }
  barrier(CLK_LOCAL_MEM_FENCE);
}
#endif