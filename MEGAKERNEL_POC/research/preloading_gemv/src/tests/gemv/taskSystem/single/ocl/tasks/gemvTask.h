#pragma once

#include "taskSystem/shared/hostDeviceCompilation.h"

#ifdef DEVICE_COMPILATION
typedef half GemvTaskElement;
#else
#include <CL/cl_half.h>
typedef cl_half GemvTaskElement;
#endif

typedef struct GemvTask {
  GLOBAL_DEVICE_PTR const GemvTaskElement* matrix;
  GLOBAL_DEVICE_PTR const GemvTaskElement* vector;
  GLOBAL_DEVICE_PTR GemvTaskElement* output;
  int tileId;
} GemvTask;

#ifdef DEVICE_COMPILATION

#define GemvBlock_MATRIX_ROWS MATRIX_ROWS
#define GemvBlock_MATRIX_COLUMNS MATRIX_COLUMNS
#define GemvBlock_BLOCK_TILE_ROWS BLOCK_TILE_ROWS
#define GemvBlock_PHASE_TILE_ROWS GEMV_PHASE_TILE_ROWS
#define GemvBlock_COMPUTE_WARPS GEMV_COMPUTE_WARPS
#include "gemvOpt/gemvBlock.hcl"

inline void ExecuteGemvTask(const GemvTask* task, __local char* slmBuffer) {
  GemvBlock(task->tileId, task->matrix, task->vector, task->output, slmBuffer);
}
#endif