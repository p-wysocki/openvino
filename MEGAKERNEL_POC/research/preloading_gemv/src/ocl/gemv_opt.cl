
// Needed params for gemv kernel:
// #define MATRIX_ROWS
// #define MATRIX_COLUMNS
// #define BLOCK_TILE_ROWS

#define PHASE_TILE_ROWS 4
#define COMPUTE_WARPS 4

#define GemvBlock_MATRIX_ROWS MATRIX_ROWS
#define GemvBlock_MATRIX_COLUMNS MATRIX_COLUMNS
#define GemvBlock_BLOCK_TILE_ROWS BLOCK_TILE_ROWS
#define GemvBlock_PHASE_TILE_ROWS PHASE_TILE_ROWS
#define GemvBlock_COMPUTE_WARPS COMPUTE_WARPS
#include "detail/gemvBlock.hcl"

// Each block handles BLOCK_TILE_ROWS rows.
__attribute__((reqd_work_group_size(TOTAL_WARPS * WARP_SIZE, 1, 1)))
__attribute__((intel_reqd_sub_group_size(WARP_SIZE))) __kernel void
gemv(__global const half* restrict matrix, __global const half* restrict vector,
     __global half* restrict output) {
  __local char slmBuffer[GemvBlockSLMNeededSizeInBytes];

  GemvBlock(matrix, vector, output, slmBuffer);
}