#include "taskManager.h"

inline void global_barrier_atomic(__constant const TaskManager* taskManager) {
  barrier(CLK_LOCAL_MEM_FENCE);

  bool firstThreadPerWg = (get_local_id(0) == 0) && (get_local_id(1) == 0) &&
                          (get_local_id(2) == 0);
  size_t numGroups = get_num_groups(0) * get_num_groups(1) * get_num_groups(2);

  __global volatile int* syncBuffer =
      (__global volatile int*)taskManager->syncBarrierBuffer;
  __global volatile int* offsetVar = syncBuffer + 2;

  if (firstThreadPerWg) {
    int offset = atomic_or(offsetVar, 0);
    __global volatile int* syncVar = syncBuffer + offset;

    if (get_global_linear_id() == 0) {
      atomic_sub(syncVar, numGroups - 1);
    } else {
      atomic_inc(syncVar);
    }

    while (atomic_or(syncVar, 0) != 0) {
    }

    if (offset) {
      atomic_and(offsetVar, 0);
    } else {
      atomic_or(offsetVar, 1);
    }
  }

  barrier(CLK_LOCAL_MEM_FENCE);
}

__kernel void taskManagerTest(__constant const TaskManager* taskManager,
                              __global int* taskIds) {
  if (get_local_id(0) == 0) {
    const __global TaskDesc* task = GetNextTask(taskManager);

    while (task != NULL) {
      // printf("Task with id <%d> is executed by work-group <%lu>\n", task->id,
      //        get_group_id(0));
      const int executedID = taskIds[task->id];

      if (executedID == -1) {
        taskIds[task->id] = get_group_id(0);
      } else {
        taskIds[task->id] = -2;
      }

      task = GetNextTask(taskManager);
    }
  }

  global_barrier_atomic(taskManager);

  if (get_local_id(0) == 0) {
    ClearTaskManagerState(taskManager);
  }
}