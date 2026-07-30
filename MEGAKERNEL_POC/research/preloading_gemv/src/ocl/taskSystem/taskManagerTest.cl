#include "device/taskManager.hcl"

inline void ExecuteTask(__global const TaskDesc* task) {
  switch (task->type) {
    case 0: {
      const __global TestTask* testTask =
          (const __global TestTask*)task->payload;
      __global int* output = testTask->output;
      *output = get_group_id(0);
      break;
    }
    default:
      break;
  }
}

__attribute__((intel_reqd_sub_group_size(32))) __kernel void taskManagerTest(
    __constant const TaskManager* taskManager) {
  if (get_local_id(0) == 0) {
    const __global TaskDesc* task = GetNextTask_thread(taskManager);

    while (task != NULL) {
      ExecuteTask(task);
      task = GetNextTask_thread(taskManager);
    }
  }

  // This is needed to clear state of task manager for next iteration.
  GlobalBarrier_block(taskManager);

  if (get_global_id(0) == 0) {
    ClearTaskManagerState_thread(taskManager);
  }
}