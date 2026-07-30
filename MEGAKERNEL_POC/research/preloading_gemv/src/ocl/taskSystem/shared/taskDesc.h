#pragma once
#include "hostDeviceCompilation.h"

typedef struct TestTask {
  int id;
  GLOBAL_DEVICE_PTR void* output;
} TestTask;

// TODO __alignas(16) ?
typedef struct TaskDesc {
  int type;
  char payload[16 - sizeof(int)];
} TaskDesc;