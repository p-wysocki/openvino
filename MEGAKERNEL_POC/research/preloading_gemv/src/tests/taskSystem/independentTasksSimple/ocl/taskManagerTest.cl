
#include "taskSystem/shared/taskDesc.h"
#include "testTask.h"

inline void ExecuteTask(TaskDesc task, __local char* slmBuffer) {
  switch (task.type) {
    case 0: {
      if (get_local_id(0) == 0) {
        const TestTask* testTask = (const TestTask*)task.payload;
        const int id = testTask->id;
        __global int* output = testTask->output;
        *output = id * id;
      }
      break;
    }
    default:
      break;
  }
}

#define WorkerMainLoop_block_EXEC_FUN ExecuteTask
#include "taskSystem/device/workerMainLoop_template.hcl"

__attribute__((intel_reqd_sub_group_size(32))) __kernel void taskManagerTest(
    __constant const TaskManager* taskManager) {
  __local char slmBuffer[32 * 1024];
  WorkerMainLoop_block(taskManager, slmBuffer);
}