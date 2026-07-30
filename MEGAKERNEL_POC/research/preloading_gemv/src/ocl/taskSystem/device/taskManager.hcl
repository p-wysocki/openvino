#pragma once

#include "shared/taskManager.h"

// GetNext task to execute.
// Returns invalid task(type = -1) if no more tasks are available.
void GetNextTask_block(__constant const TaskManager* taskManager,
                       __local char* slmBuffer, __private TaskDesc* taskPtr);

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
inline void GetNextTask_block(__constant const TaskManager* taskManager,
                              __local char* slmBuffer,
                              __private TaskDesc* taskPtr) {
  __global const TaskDesc* task = NULL;
  __local TaskDesc* taskLocal = (__local TaskDesc*)slmBuffer;

  // Thread 0 of warp 0 gets task from global.
  if (get_sub_group_local_id() == 0 && get_sub_group_id() == 0) {
    task = GetNextTask_thread(taskManager);
  }

  // Thread 0 of warp 0 broadcasts task pointer to all threads in the warp.
  if (get_sub_group_id() == 0) {
    ulong ptr_int = (ulong)task;
    ptr_int = sub_group_broadcast(ptr_int, 0);
    const char* taskChar = (const __global char*)ptr_int;

    TaskDesc invalidTask;
    invalidTask.type = -1;

    if (taskChar == NULL) {
      taskChar = (const char*)&invalidTask;
    }

    for (int i = get_sub_group_local_id(); i < sizeof(TaskDesc);
         i += get_sub_group_size()) {
      slmBuffer[i] = taskChar[i];
    };
  }

  barrier(CLK_LOCAL_MEM_FENCE);

  // All threads read task from SLM.
  *taskPtr = *taskLocal;
}

/////////////////////////////////////////////////////////////
inline void ClearTaskManagerState_thread(
    __constant const TaskManager* taskManager) {
  atomic_xchg(taskManager->processedTaskCount, 0);
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