#pragma once

#include "../shared/taskManager.h"

// GetNext task to execute.
// Returns invalid task(NULL) if no more tasks are available.
__global const TaskDesc* GetNextTask_block(
    __constant const TaskManager* taskManager, __local char* slmBuffer);

// Clear the state of the task manager.
void ClearTaskManagerState_thread(__constant const TaskManager* taskManager);

///////////////////////////////////////////////////////////////
//
// INLINES:
//
/////////////////////////////////////////////////////////////

/////////////////////////////////////////////////////////////
inline __global const TaskDesc* GetNextTask_thread(
    __constant const TaskManager* taskManager) {
  const int slotId = atomic_inc(taskManager->processedTaskCount);
  if (slotId >= taskManager->workQueueSize) {
    return NULL;
  }
  return taskManager->workQueue + slotId;
}

/////////////////////////////////////////////////////////////
inline __global const TaskDesc* GetNextTask_block(
    __constant const TaskManager* taskManager, __local char* slmBuffer) {
  __global const TaskDesc* task = NULL;
  __local ulong* taskAddress = (__local ulong*)slmBuffer;

  // Broadcast the task pointer to all threads in the block, without using
  // work_group_broadcast, which uses SLM indirectly and decreseas occupancy in
  // case where block uses all SLM for its own purposes.

  if (get_local_id(0) == 0) {
    task = GetNextTask_thread(taskManager);
    *taskAddress = (ulong)task;
  }

  barrier(CLK_LOCAL_MEM_FENCE);
  task = (__global const TaskDesc*)(*taskAddress);
  return task;
}

/////////////////////////////////////////////////////////////
inline void ClearTaskManagerState_thread(
    __constant const TaskManager* taskManager) {
  atomic_xchg(taskManager->processedTaskCount, 0);
}

/////////////////////////////////////////////////////////////
inline void LastWorkerClearTaskManagerState_block(
    __constant const TaskManager* taskManager) {
  barrier(CLK_LOCAL_MEM_FENCE);

  if (get_local_id(0) == 0) {
    volatile __global atomic_int* syncBuffer =
        (volatile __global atomic_int*)(taskManager->processedTaskCount);
    const int processed = atomic_load_explicit(syncBuffer, memory_order_acquire,
                                               memory_scope_device);

    const int workers =
        get_num_groups(0) * get_num_groups(1) * get_num_groups(2);

    // Last executing worker clears the task manager state.
    if (processed == workers + taskManager->workQueueSize) {
      ClearTaskManagerState_thread(taskManager);
    }
  }
}