#include "common/inkernelProfile.hcl"
#include "taskSystem/shared/taskDesc.h"
#include "tasks/gemv1024x2048Task.h"
#include "tasks/gemv1024x3072Task.h"
#include "tasks/gemv3072x1024Task.h"

inline void ExecuteTasks(TaskDesc task, __local char* slmBuffer) {
  switch (task.type) {
    case 3: {
      const Gemv1024x2048Task gemv1024x2048Task =
          *(const Gemv1024x2048Task*)task.payload;
      IN_KERNEL_PROFILE_BLOCK(
          ExecuteGemv1024x2048Task(gemv1024x2048Task, slmBuffer),
          "ExecuteGemv1024x2048Task");
      break;
    }
    case 4: {
      const Gemv3072x1024Task gemv3072x1024Task =
          *(const Gemv3072x1024Task*)task.payload;
      IN_KERNEL_PROFILE_BLOCK(
          ExecuteGemv3072x1024Task(gemv3072x1024Task, slmBuffer),
          "ExecuteGemv3072x1024Task");
      break;
    }
    case 5: {
      const Gemv1024x3072Task gemv1024x3072Task =
          *(const Gemv1024x3072Task*)task.payload;
      IN_KERNEL_PROFILE_BLOCK(
          ExecuteGemv1024x3072Task(gemv1024x3072Task, slmBuffer),
          "ExecuteGemv1024x3072Task");
      break;
    }
    default:
      break;
  }
}

#define WorkerMainLoop_block_EXEC_FUN ExecuteTasks
#include "taskSystem/device/workerMainLoop_template.hcl"

__attribute__((reqd_work_group_size(512, 1, 1)))
__attribute__((intel_reqd_sub_group_size(32))) __kernel void
chainTaskSystemGemvKernel(__constant const TaskManager* taskManager) {
  __local char slmBuffer[64 * 1024];
  WorkerMainLoop_block(taskManager, slmBuffer);
}