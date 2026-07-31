#pragma once
#include "hostDeviceCompilation.h"

// TODO __alignas(16) ?
#define PAYLOAD_SIZE (64 - sizeof(int))
typedef struct TaskDesc {
  int type;
  char payload[PAYLOAD_SIZE];
} TaskDesc;