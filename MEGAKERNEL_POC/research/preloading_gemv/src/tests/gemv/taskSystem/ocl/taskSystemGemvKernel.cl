#include "taskSystem/shared/taskDesc.h"
#include "tasks/gemvTask.h"

#define GemvBlock_MATRIX_ROWS MATRIX_ROWS
#define GemvBlock_MATRIX_COLUMNS MATRIX_COLUMNS
#define GemvBlock_BLOCK_TILE_ROWS BLOCK_TILE_ROWS
#define GemvBlock_PHASE_TILE_ROWS GEMV_PHASE_TILE_ROWS
#define GemvBlock_COMPUTE_WARPS GEMV_COMPUTE_WARPS
#include "gemvOpt/gemvBlock.hcl"

inline void ExecuteTask(TaskDesc task, __local char* slmBuffer) {
  const GemvTask* gemvTask = (const GemvTask*)task.payload;
  GemvBlock(gemvTask->tileId, gemvTask->matrix, gemvTask->vector,
            gemvTask->output, slmBuffer);
}

#define WorkerMainLoop_block_EXEC_FUN ExecuteTask
#include "taskSystem/device/workerMainLoop_template.hcl"

__attribute__((reqd_work_group_size(512, 1, 1)))
__attribute__((intel_reqd_sub_group_size(32)))
__kernel void taskSystemGemvKernel(__constant const TaskManager* taskManager) {
  _Static_assert(GemvBlockSLMNeededSizeInBytes <= 32 * 1024,
                 "SLM size exceeds 32KB limit");
  __local char slmBuffer[GemvBlockSLMNeededSizeInBytes];
  WorkerMainLoop_block(taskManager, slmBuffer);
}