#include "taskManager.h"

__kernel void taskManagerTest(__constant const TaskManager* taskManager,
                              __global int* taskIds) {
  if (get_local_id(0) == 0) {
    const __global TaskDesc* task = GetNextTask(taskManager);
    taskIds[get_group_id(0)] = task->id;
  }
}