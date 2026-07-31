#include "../../gemv_opt/detail/template.hcl"
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
  TaskDesc task;
  GetNextTask_block(taskManager, slmBuffer, &task);

  while (task.type != -1) {
    WorkerMainLoop_block_EXEC_FUN(task, slmBuffer);
    GetNextTask_block(taskManager, slmBuffer, &task);
  }

  // This is needed to clear state of task manager for next iteration.
  GlobalBarrier_block(taskManager);

  if (get_global_id(0) == 0) {
    ClearTaskManagerState_thread(taskManager);
  }
}

#undef WorkerMainLoop_block_EXEC_FUN
#undef WorkerMainLoop_block_SUFFIX