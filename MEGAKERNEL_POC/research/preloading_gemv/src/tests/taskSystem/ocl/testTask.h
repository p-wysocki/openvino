#pragma once
#include "taskSystem/shared/hostDeviceCompilation.h"

typedef struct TestTask {
  int id;
  GLOBAL_DEVICE_PTR int* output;
} TestTask;