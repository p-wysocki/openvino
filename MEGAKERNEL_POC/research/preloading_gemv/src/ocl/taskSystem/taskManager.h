#pragma once

#include "taskDesc.h"

typedef struct TaskManager {
  __global const TaskDesc* array;
  volatile __global int* atomicSlotId;
} TaskManager;

static inline __global const TaskDesc* GetNextTask(
    __constant const TaskManager* taskManager) {
  const int slotId = atomic_inc(taskManager->atomicSlotId);
  return taskManager->array + slotId;
}