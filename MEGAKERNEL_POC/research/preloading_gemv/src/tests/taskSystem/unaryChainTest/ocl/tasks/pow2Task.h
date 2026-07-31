#pragma once
#include "taskSystem/shared/hostDeviceCompilation.h"

typedef struct Pow2Task {
  GLOBAL_DEVICE_PTR int* input;
  GLOBAL_DEVICE_PTR int* output;
  int size;
} Pow2Task;

#ifdef DEVICE_COMPILATION
inline void ExecuteTestTask(const Pow2Task* task, __local char* slmBuffer) {
  __local int* input_local = (__local int*)slmBuffer;
  for (int i = get_local_id(0); i < task->size; i += get_local_size(0)) {
    input_local[i] = task->input[i];
  }

  barrier(CLK_LOCAL_MEM_FENCE);
  for (int i = get_local_id(0); i < task->size; i += get_local_size(0)) {
    task->output[i] = input_local[i] * input_local[i];
  }
}
#endif