#pragma once
#include "detail/commonConstants.hcl"
#include "detail/inkernelProfile.hcl"
#include "detail/template.hcl"

#ifndef GemvBlock_SUFFIX
#define GemvBlock_SUFFIX
#endif

// Template function.
// Computes gemv for given block of rows. Each block computes
// GemvBlock_BLOCK_TILE_ROWS rows.
// This function expects SLM buffer of size
// TEMPLATE(GemvBlockSLMNeededSizeInBytes, GemvBlock_SUFFIX) to be passed as
// buff_local.

// Requires template parameters:
// #define GemvBlock_MATRIX_ROWS
// #define GemvBlock_MATRIX_COLUMNS
// #define GemvBlock_BLOCK_TILE_ROWS
// #define GemvBlock_PHASE_TILE_ROWS
// #define GemvBlock_COMPUTE_WARPS

// Optional parameter to give unique name of template instantiation.
// #define GemvBlock_SUFFIX
inline void TEMPLATE(GemvBlock,
                     GemvBlock_SUFFIX)(__global const half* restrict matrix,
                                       __global const half* restrict vector,
                                       __global half* restrict output,
                                       __local char* restrict buff_local);

////////////////////////////////////////////////////////////////
//
// IMPLEMENTATION:
//
////////////////////////////////////////////////////////////////

#ifndef GemvBlock_MATRIX_ROWS
#error "GemvBlock_MATRIX_ROWS is not defined"
#endif

#ifndef GemvBlock_MATRIX_COLUMNS
#error "GemvBlock_MATRIX_COLUMNS is not defined"
#endif

#ifndef GemvBlock_BLOCK_TILE_ROWS
#error "GemvBlock_BLOCK_TILE_ROWS is not defined"
#endif

#ifndef GemvBlock_PHASE_TILE_ROWS
#error "GemvBlock_PHASE_TILE_ROWS is not defined"
#endif

#ifndef GemvBlock_COMPUTE_WARPS
#error "GemvBlock_COMPUTE_WARPS is not defined"
#endif

_Static_assert(
    GemvBlock_BLOCK_TILE_ROWS % GemvBlock_PHASE_TILE_ROWS == 0,
    "GemvBlock_BLOCK_TILE_ROWS must be divisible by GemvBlock_PHASE_TILE_ROWS");

enum {
  TEMPLATE(GemvBlockSLMNeededSizeInBytes, GemvBlock_SUFFIX) =
      GemvBlock_PHASE_TILE_ROWS * GemvBlock_MATRIX_COLUMNS * sizeof(half) * 2
};

#define LOAD_WARPS (TOTAL_WARPS - GemvBlock_COMPUTE_WARPS)
#define PHASES_PER_BLOCK (GemvBlock_BLOCK_TILE_ROWS / GemvBlock_PHASE_TILE_ROWS)
#define PHASE_TILE_SIZE (GemvBlock_PHASE_TILE_ROWS * GemvBlock_MATRIX_COLUMNS)

// Define templates:
#define ComputeGemvTile_TILE_ROWS GemvBlock_PHASE_TILE_ROWS
#define ComputeGemvTile_TILE_COLUMNS GemvBlock_MATRIX_COLUMNS
#define ComputeGemvTile_COMPUTE_WARPS GemvBlock_COMPUTE_WARPS
#define SUFFIX
#include "detail/computeGemvTile_template.hcl"

#define LoadDataTile_LOAD_DATA_TILE_SIZE PHASE_TILE_SIZE
#define LoadDataTile_LOAD_WARPS TOTAL_WARPS
#define LoadDataTile_FIRST_LOAD_WARP_ID 0
#define LoadDataTile_NON_TEMPORAL_LOAD 1
#define LoadDataTile_SUFFIX _allWarps
#include "detail/loadDataTile_template.hcl"

#define LoadDataTile_LOAD_DATA_TILE_SIZE PHASE_TILE_SIZE
#define LoadDataTile_LOAD_WARPS LOAD_WARPS
#define LoadDataTile_FIRST_LOAD_WARP_ID GemvBlock_COMPUTE_WARPS
#define LoadDataTile_NON_TEMPORAL_LOAD 1
#define LoadDataTile_SUFFIX _loadWarps
#include "detail/loadDataTile_template.hcl"

#define LoadDataTile_LOAD_DATA_TILE_SIZE GemvBlock_MATRIX_COLUMNS
#define LoadDataTile_LOAD_WARPS TOTAL_WARPS
#define LoadDataTile_FIRST_LOAD_WARP_ID 0
#define LoadDataTile_NON_TEMPORAL_LOAD 0
#define LoadDataTile_SUFFIX _allWarpsCached
#include "detail/loadDataTile_template.hcl"

///////////////////////////////////////////////////////////////
inline void SwapPtr(__local half* restrict __private* a,
                    __local half* restrict __private* b) {
  __local half* temp = *a;
  *a = *b;
  *b = temp;
}

////////////////////////////////////////////////////////////////
inline void TEMPLATE(GemvBlock,
                     GemvBlock_SUFFIX)(__global const half* restrict matrix,
                                       __global const half* restrict vector,
                                       __global half* restrict output,
                                       __local char* restrict buff_local) {
  __global half* restrict outputBlockTilePtr_global =
      output + get_group_id(0) * GemvBlock_BLOCK_TILE_ROWS;

  __global const half* restrict matrixBlockTilePtr_global =
      matrix +
      get_group_id(0) * GemvBlock_BLOCK_TILE_ROWS * GemvBlock_MATRIX_COLUMNS;

  __local half* restrict computeBufferPtr_local =
      (__local half* restrict)buff_local;
  __local half* restrict loadBufferPtr_local =
      (__local half* restrict)(buff_local + GemvBlock_PHASE_TILE_ROWS *
                                                GemvBlock_MATRIX_COLUMNS *
                                                sizeof(half));

  // --------------------------------------------------------
  // Preload vector data into registers for reuse across dot products.
  half4 cachedVector_thisWarp[ComputeGemvTile_CACHE_SIZE];
  //---------------------------------------------------------

  IN_KERNEL_PROFILE(
      LoadDataTile_allWarps(loadBufferPtr_local,
                            matrixBlockTilePtr_global + 0 * PHASE_TILE_SIZE),
      "INITIAL LoadDataTile_allWarps");

  // The way that preloading of vector dara is implemented should be
  // parametrized by 'policy' design, which is hard to achive wihouth templates.
  // Basically that policy will depend on the size of vector.
  // For now, I just hardcoded this policy as it is the best in avg case.
  IN_KERNEL_PROFILE(LoadDataTile_allWarpsCached(computeBufferPtr_local, vector),
                    "INITIAL VECTOR LOAD LoadDataTile_allWarps");

  IN_KERNEL_PROFILE(barrier(CLK_LOCAL_MEM_FENCE), "Initial Barrier");

  if (get_sub_group_id() < GemvBlock_COMPUTE_WARPS) {
    IN_KERNEL_PROFILE(
        PreloadVectorData(cachedVector_thisWarp, computeBufferPtr_local),
        "PreloadVectorData");
  }
  IN_KERNEL_PROFILE(barrier(CLK_LOCAL_MEM_FENCE),
                    "Barrier After PreloadVectorData");

#pragma unroll
  for (int phase = 0; phase < PHASES_PER_BLOCK - 1; ++phase) {
    SwapPtr(&computeBufferPtr_local, &loadBufferPtr_local);

    if (get_sub_group_id() < GemvBlock_COMPUTE_WARPS) {
      IN_KERNEL_PROFILE(
          ComputeGemvTile(
              (__local half4* restrict)computeBufferPtr_local,
              cachedVector_thisWarp,
              outputBlockTilePtr_global + phase * GemvBlock_PHASE_TILE_ROWS),
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
  if (get_sub_group_id() < GemvBlock_COMPUTE_WARPS) {
    IN_KERNEL_PROFILE(
        ComputeGemvTile((__local half4* restrict)computeBufferPtr_local,
                        cachedVector_thisWarp,
                        outputBlockTilePtr_global +
                            (PHASES_PER_BLOCK - 1) * GemvBlock_PHASE_TILE_ROWS),
        "Last ComputeGemvTile");
  }
}

#undef GemvBlock_MATRIX_ROWS
#undef GemvBlock_MATRIX_COLUMNS
#undef GemvBlock_BLOCK_TILE_ROWS
#undef GemvBlock_PHASE_TILE_ROWS
#undef GemvBlock_COMPUTE_WARPS
#undef SUFFIX