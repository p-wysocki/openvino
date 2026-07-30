#pragma once
#include "hostDeviceCompilation.h"

typedef struct TestTask {
  int id;
  GLOBAL_DEVICE_PTR int* output;
} TestTask;

// TODO __alignas(16) ?
#define PAYLOAD_SIZE (32 - sizeof(int))
typedef struct TaskDesc {
  int type;
  char payload[PAYLOAD_SIZE];
} TaskDesc;