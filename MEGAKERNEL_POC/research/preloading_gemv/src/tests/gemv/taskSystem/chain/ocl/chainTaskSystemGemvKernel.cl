#include "taskSystem/shared/taskDesc.h"
#include "tasks/chainGemvTask.h"

#define GemvBlock_MATRIX_ROWS 1024
#define GemvBlock_MATRIX_COLUMNS 2048
#define GemvBlock_BLOCK_TILE_ROWS 32
#define GemvBlock_PHASE_TILE_ROWS 4
#define GemvBlock_COMPUTE_WARPS 4
#define GemvBlock_SUFFIX _layer0
#include "gemvOpt/gemvBlock.hcl"

#define GemvBlock_MATRIX_ROWS 3072
#define GemvBlock_MATRIX_COLUMNS 1024
#define GemvBlock_BLOCK_TILE_ROWS 32
#define GemvBlock_PHASE_TILE_ROWS 4
#define GemvBlock_COMPUTE_WARPS 4
#define GemvBlock_SUFFIX _layer1
#include "gemvOpt/gemvBlock.hcl"

#define GemvBlock_MATRIX_ROWS 1024
#define GemvBlock_MATRIX_COLUMNS 3072
#define GemvBlock_BLOCK_TILE_ROWS 32
#define GemvBlock_PHASE_TILE_ROWS 2
#define GemvBlock_COMPUTE_WARPS 2
#define GemvBlock_SUFFIX _layer2
#include "gemvOpt/gemvBlock.hcl"

inline void ExecuteTask(TaskDesc task, __local char* slmBuffer) {
  const ChainGemvTask* gemvTask = (const ChainGemvTask*)task.payload;
  WaitForGemvInput_block(gemvTask);
  switch (task.type) {
    case 0:
      GemvBlock_layer0(gemvTask->tileId, gemvTask->matrix, gemvTask->vector,
                       gemvTask->output, slmBuffer);
      break;
    case 1:
      GemvBlock_layer1(gemvTask->tileId, gemvTask->matrix, gemvTask->vector,
                       gemvTask->output, slmBuffer);
      break;
    case 2:
      GemvBlock_layer2(gemvTask->tileId, gemvTask->matrix, gemvTask->vector,
                       gemvTask->output, slmBuffer);
      break;
  }
  SignalGemvOutput_block(gemvTask);
}

#define WorkerMainLoop_block_EXEC_FUN ExecuteTask
#include "taskSystem/device/workerMainLoop_template.hcl"

__attribute__((reqd_work_group_size(512, 1, 1)))
__attribute__((intel_reqd_sub_group_size(32))) __kernel void
chainTaskSystemGemvKernel(__constant const TaskManager* taskManager) {
  __local char slmBuffer[32 * 1024];
  WorkerMainLoop_block(taskManager, slmBuffer);
}