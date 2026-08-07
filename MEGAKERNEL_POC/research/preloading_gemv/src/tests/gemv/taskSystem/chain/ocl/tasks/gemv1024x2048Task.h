#pragma once
#include "common.h"

typedef struct Gemv1024x2048Task {
  GLOBAL_DEVICE_PTR const GemvTaskElement* __restrict__ matrix;
  GLOBAL_DEVICE_PTR const GemvTaskElement* __restrict__ vector;
  GLOBAL_DEVICE_PTR GemvTaskElement* __restrict__ output;
  GLOBAL_DEVICE_PTR int* __restrict__ inputSemaphore;
  GLOBAL_DEVICE_PTR int* __restrict__ outputSemaphore;
  int wantedInputSyncValue;
  int tileId;
} Gemv1024x2048Task;

#ifdef DEVICE_COMPILATION

#define GemvBlock_MATRIX_ROWS 1024
#define GemvBlock_MATRIX_COLUMNS 2048
#define GemvBlock_BLOCK_TILE_ROWS BLOCK_TILE_ROWS_1024x2048
#define GemvBlock_PHASE_TILE_ROWS PHASE_TILE_ROWS_1024x2048
#define GemvBlock_COMPUTE_WARPS COMPUTE_WARPS_1024x2048
#define GemvBlock_SUFFIX _1024x2048
#include "gemvOpt/gemvBlock.hcl"

inline void ExecuteGemv1024x2048Task(const Gemv1024x2048Task task,
                                     __local char* slmBuffer) {
  GemvBlock_1024x2048(task.tileId, task.matrix, task.vector, task.output,
                      slmBuffer,
                      (volatile __global atomic_int*)task.inputSemaphore,
                      task.wantedInputSyncValue);
  SignalSemaphore_block(0, (volatile __global atomic_int*)task.outputSemaphore);
}
#endif