#pragma once

#include "taskSystem/shared/hostDeviceCompilation.h"

#ifdef DEVICE_COMPILATION
typedef half GemvTaskElement;
#else
#include <CL/cl_half.h>
typedef cl_half GemvTaskElement;
#endif

typedef struct GemvTask {
  GLOBAL_DEVICE_PTR const GemvTaskElement* matrix;
  GLOBAL_DEVICE_PTR const GemvTaskElement* vector;
  GLOBAL_DEVICE_PTR GemvTaskElement* output;
  int tileId;
} GemvTask;