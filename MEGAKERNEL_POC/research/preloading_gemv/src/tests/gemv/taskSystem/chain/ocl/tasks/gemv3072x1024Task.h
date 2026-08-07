#pragma once
#include "common.h"

typedef struct Gemv3072x1024Task {
  GLOBAL_DEVICE_PTR const GemvTaskElement* matrix;
  GLOBAL_DEVICE_PTR const GemvTaskElement* vector;
  GLOBAL_DEVICE_PTR GemvTaskElement* output;
  GLOBAL_DEVICE_PTR int* inputSemaphore;
  GLOBAL_DEVICE_PTR int* outputSemaphore;
  int wantedInputSyncValue;
  int tileId;
} Gemv3072x1024Task;

#ifdef DEVICE_COMPILATION

#define GemvBlock_MATRIX_ROWS 3072
#define GemvBlock_MATRIX_COLUMNS 1024
#define GemvBlock_BLOCK_TILE_ROWS BLOCK_TILE_ROWS_3072x1024
#define GemvBlock_PHASE_TILE_ROWS PHASE_TILE_ROWS_3072x1024
#define GemvBlock_COMPUTE_WARPS COMPUTE_WARPS_3072x1024
#define GemvBlock_SUFFIX _3072x1024
#include "gemvOpt/gemvBlock.hcl"

inline void ExecuteGemv3072x1024Task(const Gemv3072x1024Task task,
                                     __local char* slmBuffer) {
  GemvBlock_3072x1024(task.tileId, task.matrix, task.vector, task.output,
                      slmBuffer,
                      (volatile __global atomic_int*)task.inputSemaphore,
                      task.wantedInputSyncValue);
  SignalSemaphore_block(0,
                        (volatile __global atomic_int*)task.outputSemaphore);
}
#endif