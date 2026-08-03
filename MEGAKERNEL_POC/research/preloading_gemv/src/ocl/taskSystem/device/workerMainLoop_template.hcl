#include "common/template.hcl"
#include "taskManager.hcl"

#ifndef WorkerMainLoop_block_SUFFIX
#define WorkerMainLoop_block_SUFFIX
#endif

// Template function.
// Main worker loo for each block.

// Requires template parameters:
// #define WorkerMainLoop_block_EXEC_FUN -> policy function: void FUNC(TaskDesc
// task, __local char* slmBuffer)

// Optional parameter to give unique name of template instantiation.
// #define WorkerMainLoop_block_SUFFIX
inline void TEMPLATE(WorkerMainLoop_block, WorkerMainLoop_block_SUFFIX)(
    __constant const TaskManager* taskManager, __local char* slmBuffer);

////////////////////////////////////////////////////////////////
//
// IMPLEMENTATION
//
////////////////////////////////////////////////////////////////

#ifndef WorkerMainLoop_block_EXEC_FUN
#error "WorkerMainLoop_block_EXEC_FUN is not defined"
#endif

inline void TEMPLATE(WorkerMainLoop_block, WorkerMainLoop_block_SUFFIX)(
    __constant const TaskManager* taskManager, __local char* slmBuffer) {
  __global const TaskDesc* taskPtr = NULL;
  taskPtr = GetNextTask_block(taskManager, slmBuffer);

  while (taskPtr != NULL) {
    TaskDesc task = *taskPtr;
    WorkerMainLoop_block_EXEC_FUN(task, slmBuffer);
    taskPtr = GetNextTask_block(taskManager, slmBuffer);
  }

  barrier(CLK_LOCAL_MEM_FENCE);

  if (get_local_id(0) == 0) {
    __global volatile int* syncBuffer =
        (__global volatile int*)taskManager->syncBarrierBuffer;

    int prev = atomic_inc(syncBuffer);
    const int workers =
        get_num_groups(0) * get_num_groups(1) * get_num_groups(2);

    // Last executing worker clears the task manager state.
    if (prev == (workers - 1)) {
      ClearTaskManagerState_thread(taskManager);
    }
  }
}

#undef WorkerMainLoop_block_EXEC_FUN
#undef WorkerMainLoop_block_SUFFIX