#pragma once
#include "hostDeviceCompilation.h"

typedef enum TaskType { GEMV } TaskType;

// TODO __alignas(16) ?
typedef struct TaskDesc {
  GLOBAL_DEVICE_PTR const void* __restrict__ weights;
  GLOBAL_DEVICE_PTR const void* __restrict__ input;
  GLOBAL_DEVICE_PTR void* __restrict__ output;
  int id;
  TaskType taskType;
} TaskDesc;