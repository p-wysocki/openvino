#include "taskSystem/shared/taskDesc.h"
#include "tasks/gemvTask.h"

inline void ExecuteTask(TaskDesc task, __local char* slmBuffer) {
  switch (task.type) {
    case 0: {
      const GemvTask* gemvTask = (const GemvTask*)task.payload;
      ExecuteGemvTask(gemvTask, slmBuffer);
      break;
    }
    default:
      break;
  }
}

#define WorkerMainLoop_block_EXEC_FUN ExecuteTask
#include "taskSystem/device/workerMainLoop_template.hcl"

__attribute__((reqd_work_group_size(512, 1, 1)))
__attribute__((intel_reqd_sub_group_size(32))) __kernel void
taskSystemGemvKernel(__constant const TaskManager* taskManager) {
  _Static_assert(GemvBlockSLMNeededSizeInBytes <= 32 * 1024,
                 "SLM size exceeds 32KB limit");
  __local char slmBuffer[GemvBlockSLMNeededSizeInBytes];
  WorkerMainLoop_block(taskManager, slmBuffer);
}