#include "device/taskManager.hcl"

void inline ExecuteTask(__global const TaskDesc* task, __global int* taskIds) {
  switch (task->type) {
    case 0: {
      const __global TestTask* testTask =
          (const __global TestTask*)task->payload;
      __global int* output = testTask->output;
      if (output != taskIds) {
        printf(
            "Error: output pointer mismatch for task %d, output=%p, "
            "taskIds=%p\n",
            testTask->id, output, taskIds);
      }
      output[testTask->id] = get_group_id(0);
      break;
    }
    default:
      break;
  }
}

__attribute__((intel_reqd_sub_group_size(32))) __kernel void taskManagerTest(
    __constant const TaskManager* taskManager, __global int* taskIds) {
  if (get_local_id(0) == 0) {
    const __global TaskDesc* task = GetNextTask(taskManager);

    while (task != NULL) {
      ExecuteTask(task, taskIds);
      task = GetNextTask(taskManager);
    }
  }

  // This is needed to clear state of task manager for next iteration.
  GlobalBarrier(taskManager);

  if (get_global_id(0) == 0) {
    ClearTaskManagerState(taskManager);
  }
}