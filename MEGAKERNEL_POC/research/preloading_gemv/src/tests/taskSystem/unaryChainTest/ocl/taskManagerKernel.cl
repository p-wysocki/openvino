
#include "taskSystem/shared/taskDesc.h"
#include "tasks/pow2Task.h"
#include "tasks/siluTask.h"

inline void ExecuteTask(TaskDesc task, __local char* slmBuffer) {
  switch (task.type) {
    case 0: {
      const Pow2Task* pow2Task = (const Pow2Task*)task.payload;
      ExecuteTestTask(pow2Task, slmBuffer);
      break;
    }
    case 1: {
      const SiluTask* siluTask = (const SiluTask*)task.payload;
      ExecuteSiluTask(siluTask, slmBuffer);
      break;
    }
    default:
      break;
  }
}

#define WorkerMainLoop_block_EXEC_FUN ExecuteTask
#include "taskSystem/device/workerMainLoop_template.hcl"

__attribute__((intel_reqd_sub_group_size(32))) __kernel void taskManagerKernel(
    __constant const TaskManager* taskManager) {
  __local char slmBuffer[32 * 1024];
  WorkerMainLoop_block(taskManager, slmBuffer);
}