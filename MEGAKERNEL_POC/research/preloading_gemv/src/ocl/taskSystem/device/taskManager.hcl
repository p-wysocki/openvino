#pragma once

#include "../shared/taskManager.h"

// GetNext task to execute.
// Returns invalid task(type = -1) if no more tasks are available.
__global const TaskDesc* GetNextTask_block(
    __constant const TaskManager* taskManager, __local char* slmBuffer);

// Global barrier for all work-groups in the kernel.
// All threads in all blocks has to call this function to synchronize.
void GlobalBarrier_block(__constant const TaskManager* taskManager);

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
  atomic_xchg(taskManager->syncBarrierBuffer, 0);
}

/////////////////////////////////////////////////////////////
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