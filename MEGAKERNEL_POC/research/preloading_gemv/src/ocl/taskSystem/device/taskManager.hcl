#pragma once

#include "shared/taskManager.h"

inline __global const TaskDesc* GetNextTask_thread(
    __constant const TaskManager* taskManager) {
  const int slotId = atomic_inc(taskManager->processedTaskCount);
  if (slotId >= taskManager->workQueueSize) {
    return NULL;
  }
  return taskManager->workQueue + slotId;
}

inline void ClearTaskManagerState_thread(
    __constant const TaskManager* taskManager) {
  atomic_xchg(taskManager->processedTaskCount, 0);
}

inline void GlobalBarrier_block(__constant const TaskManager* taskManager) {
  barrier(CLK_LOCAL_MEM_FENCE);

  const bool firstThreadPerWg = (get_local_id(0) == 0) &&
                                (get_local_id(1) == 0) &&
                                (get_local_id(2) == 0);
  const size_t numGroups =
      get_num_groups(0) * get_num_groups(1) * get_num_groups(2);

  __global volatile int* syncBuffer =
      (__global volatile int*)taskManager->syncBarrierBuffer;

  if (firstThreadPerWg) {
    if (get_global_linear_id() == 0) {
      atomic_sub(syncBuffer, numGroups - 1);
    } else {
      atomic_inc(syncBuffer);
    }

    while (atomic_or(syncBuffer, 0) != 0) {
    }
  }
  barrier(CLK_LOCAL_MEM_FENCE);
}