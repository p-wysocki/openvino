#include "detail/commonConstants.hcl"
#include "detail/inkernelProfile.hcl"

#define MATRIX_ROWS 2048
#define MATRIX_COLUMNS 1024
#define BLOCK_TILE_ROWS 28
#define PHASE_TILE_ROWS 4

#define COMPUTE_WARPS 4
#define LOAD_WARPS (TOTAL_WARPS - COMPUTE_WARPS)
#define PHASES_PER_BLOCK (BLOCK_TILE_ROWS / PHASE_TILE_ROWS)
#define PHASE_TILE_SIZE (PHASE_TILE_ROWS * MATRIX_COLUMNS)

// Define templates:
#define ComputeGemvTile_TILE_ROWS PHASE_TILE_ROWS
#define ComputeGemvTile_TILE_COLUMNS MATRIX_COLUMNS
#define ComputeGemvTile_COMPUTE_WARPS COMPUTE_WARPS
#include "detail/computeGemvTile_template.hcl"

#define LoadDataTile_LOAD_DATA_TILE_SIZE PHASE_TILE_SIZE
#define LoadDataTile_LOAD_WARPS TOTAL_WARPS
#define LoadDataTile_FIRST_LOAD_WARP_ID 0
#define SUFFIX _allWarps
#include "detail/loadDataTile_template.hcl"

#define LoadDataTile_LOAD_DATA_TILE_SIZE PHASE_TILE_SIZE
#define LoadDataTile_LOAD_WARPS LOAD_WARPS
#define LoadDataTile_FIRST_LOAD_WARP_ID COMPUTE_WARPS
#define SUFFIX _loadWarps
#include "detail/loadDataTile_template.hcl"

///////////////////////////////////////////////////////////////
inline void SwapPtr(__local half* restrict __private* a,
                    __local half* restrict __private* b) {
  __local half* temp = *a;
  *a = *b;
  *b = temp;
}

// Each block handles PHASE_TILE_ROWS rows, and each subgroup handles
// ROWS_FOR_COMPUTE_WARP rows. All compute subgroups cooperate to compute the
// dot products for their assigned rows.
__attribute__((reqd_work_group_size(TOTAL_WARPS * WARP_SIZE, 1, 1)))
__attribute__((intel_reqd_sub_group_size(WARP_SIZE))) __kernel void
gemv(__global const half* restrict matrix, __global const half* restrict vector,
     __global half* restrict output, const uint rowCount,
     const uint columnCount) {
  __local half matrixPhaseTileBuff1_local[PHASE_TILE_SIZE];
  __local half matrixPhaseTileBuff2_local[PHASE_TILE_SIZE];

  __global half* restrict outputBlockTilePtr_global =
      output + get_group_id(0) * BLOCK_TILE_ROWS;

  __global const half* restrict matrixBlockTilePtr_global =
      matrix + get_group_id(0) * BLOCK_TILE_ROWS * MATRIX_COLUMNS;

  __local half* restrict computeBufferPtr_local =
      (__local half* restrict)matrixPhaseTileBuff2_local;
  __local half* restrict loadBufferPtr_local =
      (__local half* restrict)matrixPhaseTileBuff1_local;

  // ---------------------------------------------------
  // Preload vector data into registers for reuse across dot products.
  half4 cachedVector_thisWarp[ComputeGemvTile_CACHE_SIZE];
  //---------------------------------------------------------

  IN_KERNEL_PROFILE(
      LoadDataTile_allWarps(loadBufferPtr_local,
                            matrixBlockTilePtr_global + 0 * PHASE_TILE_SIZE),
      "INITIAL LoadDataTile_allWarps");

  IN_KERNEL_PROFILE(barrier(CLK_LOCAL_MEM_FENCE), "Initial Barrier");

  if (get_sub_group_id() < COMPUTE_WARPS) {
    IN_KERNEL_PROFILE(PreloadVectorData(cachedVector_thisWarp, vector),
                      "PreloadVectorData");
  }

#pragma unroll
  for (int phase = 0; phase < PHASES_PER_BLOCK - 1; ++phase) {
    SwapPtr(&computeBufferPtr_local, &loadBufferPtr_local);

    if (get_sub_group_id() < COMPUTE_WARPS) {
      IN_KERNEL_PROFILE(
          ComputeGemvTile((__local half4* restrict)computeBufferPtr_local,
                          cachedVector_thisWarp,
                          outputBlockTilePtr_global + phase * PHASE_TILE_ROWS),
          "ComputeGemvTile");
    } else {
      IN_KERNEL_PROFILE(
          LoadDataTile_loadWarps(
              loadBufferPtr_local,
              matrixBlockTilePtr_global + (phase + 1) * PHASE_TILE_SIZE),
          "LoadDataTile_loadWarps");
    }

    IN_KERNEL_PROFILE(barrier(CLK_LOCAL_MEM_FENCE), "Barrier");
  }

  SwapPtr(&computeBufferPtr_local, &loadBufferPtr_local);
  if (get_sub_group_id() < COMPUTE_WARPS) {
    IN_KERNEL_PROFILE(
        ComputeGemvTile((__local half4* restrict)computeBufferPtr_local,
                        cachedVector_thisWarp,
                        outputBlockTilePtr_global +
                            (PHASES_PER_BLOCK - 1) * PHASE_TILE_ROWS),
        "Last ComputeGemvTile");
  }
}