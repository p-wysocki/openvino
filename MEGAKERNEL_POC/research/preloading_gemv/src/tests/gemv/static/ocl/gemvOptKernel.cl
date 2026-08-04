
// Needed params for gemv kernel:
// #define MATRIX_ROWS
// #define MATRIX_COLUMNS
// #define BLOCK_TILE_ROWS
// #define GEMV_PHASE_TILE_ROWS
// #define GEMV_COMPUTE_WARPS

// If this is not true, the kernel will fail with out-of-bounds memory access
// for now.
_Static_assert(MATRIX_ROWS % BLOCK_TILE_ROWS == 0,
               "MATRIX_ROWS must be divisible by BLOCK_TILE_ROWS");

#define GemvBlock_MATRIX_ROWS MATRIX_ROWS
#define GemvBlock_MATRIX_COLUMNS MATRIX_COLUMNS
#define GemvBlock_BLOCK_TILE_ROWS BLOCK_TILE_ROWS
#define GemvBlock_PHASE_TILE_ROWS GEMV_PHASE_TILE_ROWS
#define GemvBlock_COMPUTE_WARPS GEMV_COMPUTE_WARPS
#include "gemvOpt/gemvBlock.hcl"

// Each block handles BLOCK_TILE_ROWS rows.
__attribute__((reqd_work_group_size(TOTAL_WARPS * WARP_SIZE, 1, 1)))
__attribute__((intel_reqd_sub_group_size(WARP_SIZE))) __kernel void
gemvOptKernel(__global const half* restrict matrix,
              __global const half* restrict vector,
              __global half* restrict output) {
  _Static_assert(GemvBlockSLMNeededSizeInBytes <= 64 * 1024,
                 "SLM size exceeds 64KB limit");
  __local char slmBuffer[GemvBlockSLMNeededSizeInBytes];

  GemvBlock(get_group_id(0), matrix, vector, output, slmBuffer);
}