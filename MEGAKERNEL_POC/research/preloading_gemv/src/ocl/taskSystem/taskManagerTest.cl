#include "device/taskManager.hcl"

__attribute__((intel_reqd_sub_group_size(32))) __kernel void taskManagerTest(
    __constant const TaskManager* taskManager, __global int* taskIds) {
  if (get_local_id(0) == 0) {
    const __global TaskDesc* task = GetNextTask(taskManager);

    while (task != NULL) {
      taskIds[task->id] = get_group_id(0);
      task = GetNextTask(taskManager);
    }
  }

  // This is needed to clear state of task manager for next iteration.
  GlobalBarrier(taskManager);

  if (get_global_id(0) == 0) {
    ClearTaskManagerState(taskManager);
  }
}