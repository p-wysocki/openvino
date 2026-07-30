#include "device/taskManager.hcl"

inline void ExecuteTask(TaskDesc task, __local char* slmBuffer) {
  switch (task.type) {
    case 0: {
      if (get_local_id(0) == 0) {
        const TestTask* testTask = (const TestTask*)task.payload;
        __global int* output = testTask->output;
        *output = get_group_id(0);
      }
      break;
    }
    default:
      break;
  }
}

inline bool GetNextTask_block(__constant const TaskManager* taskManager,
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
  return true;
}

__attribute__((intel_reqd_sub_group_size(32))) __kernel void taskManagerTest(
    __constant const TaskManager* taskManager) {
  __local char slmBuffer[32 * 1024];
  TaskDesc task;
  GetNextTask_block(taskManager, slmBuffer, &task);

  while (task.type != -1) {
    ExecuteTask(task, slmBuffer);
    GetNextTask_block(taskManager, slmBuffer, &task);
  }

  // This is needed to clear state of task manager for next iteration.
  GlobalBarrier_block(taskManager);

  if (get_global_id(0) == 0) {
    ClearTaskManagerState_thread(taskManager);
  }
}