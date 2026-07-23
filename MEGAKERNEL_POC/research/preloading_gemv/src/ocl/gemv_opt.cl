// Tiled GEMV specialized for 32-wide subgroups.
// Each subgroup computes ROWS_FOR_COMPUTE_WARP rows so vector loads are reused
// across dot products.
//
// Launch configuration (host side):
//   global = (ceil(rowCount / TOTAL_ROWS_FOR_BLOCK) * WG_SIZE)
//   local  = (WG_SIZE)
#pragma OPENCL EXTENSION cl_khr_fp16 : enable
#include "detail/inkernelProfile.hcl"

#define TOTAL_ROWS_FOR_BLOCK 28
#define TOTAL_WARPS 16
#define COMPUTE_WARPS 4
#define ROWS_FOR_COMPUTE_WARP 1
#define WARP_SIZE 32
#define COMPUTE_WG_SIZE (COMPUTE_WARPS * WARP_SIZE)
#define LOAD_DATA_WG_SIZE ((TOTAL_WARPS - COMPUTE_WARPS) * WARP_SIZE)
#define ROWS_FOR_BLOCK_FOR_PHASE (COMPUTE_WARPS * ROWS_FOR_COMPUTE_WARP)
#define PHASES_PER_BLOCK (TOTAL_ROWS_FOR_BLOCK / ROWS_FOR_BLOCK_FOR_PHASE)

// Computes gemv for give tile.
// Each warp compuutes ROWS_FOR_COMPUTE_WARP rows.
// Warps compute whole dot product for their assigned rows.
#define COMPUTE_GEMV_BLOCK_ROWS ROWS_FOR_BLOCK_FOR_PHASE
#define COMPUTE_GEMV_BLOCK_COLUMS 1024
inline void ComputeGemvTileActiveWarps(
    __local const half4* restrict matrixTile_local,
    __private const half4* restrict cachedVector,
    __global half* restrict result) {
#define VECTOR_WIDTH 4
#define VECTOR_ITEMS_FOR_WARP (WARP_SIZE * VECTOR_WIDTH)

  const int laneLid = get_sub_group_local_id();
  const int startingRowIdxForThisWarp =
      get_sub_group_id() * ROWS_FOR_COMPUTE_WARP;
  bool rowIsValid[ROWS_FOR_COMPUTE_WARP];
  float acc[ROWS_FOR_COMPUTE_WARP];

#pragma unroll ROWS_FOR_COMPUTE_WARP
  for (int rowIdx = 0; rowIdx < ROWS_FOR_COMPUTE_WARP; ++rowIdx) {
    rowIsValid[rowIdx] =
        (startingRowIdxForThisWarp + rowIdx) < COMPUTE_GEMV_BLOCK_ROWS;
    acc[rowIdx] = 0.0f;
  }

  // Compute dot products for assigned rows.
#pragma unroll
  for (int colIdx = laneLid; colIdx < COMPUTE_GEMV_BLOCK_COLUMS / VECTOR_WIDTH;
       colIdx += WARP_SIZE) {
    const float4 vectorData = convert_float4(cachedVector[colIdx / WARP_SIZE]);
#pragma unroll
    for (int rowIdx = 0; rowIdx < ROWS_FOR_COMPUTE_WARP; ++rowIdx) {
      if (rowIsValid[rowIdx]) {
        const int rowOffset = (startingRowIdxForThisWarp + rowIdx) *
                              (COMPUTE_GEMV_BLOCK_COLUMS / VECTOR_WIDTH);
        const float4 matrixData =
            convert_float4(matrixTile_local[rowOffset + colIdx]);
        acc[rowIdx] += dot(matrixData, vectorData);
      }
    }
  }

  float reduced[ROWS_FOR_COMPUTE_WARP];
#pragma unroll ROWS_FOR_COMPUTE_WARP
  for (int rowIdx = 0; rowIdx < ROWS_FOR_COMPUTE_WARP; ++rowIdx) {
    const float rowAcc = acc[rowIdx];
    reduced[rowIdx] = sub_group_reduce_add(rowAcc);
  }

  // Save the results.
  if (laneLid == 0) {
#pragma unroll ROWS_FOR_COMPUTE_WARP
    for (int rowIdx = 0; rowIdx < ROWS_FOR_COMPUTE_WARP; ++rowIdx) {
      if (rowIsValid[rowIdx]) {
        const int outputIdx = startingRowIdxForThisWarp + rowIdx;
        result[outputIdx] = convert_half_rte(reduced[rowIdx]);
      }
    }
  }
}

#define LOAD_DATA_BLOCK_SIZE ROWS_FOR_BLOCK_FOR_PHASE* COMPUTE_GEMV_BLOCK_COLUMS

#define LoadDataTile_LOAD_DATA_BLOCK_SIZE LOAD_DATA_BLOCK_SIZE
#define LoadDataTile_LOAD_WG_SIZE TOTAL_WARPS* WARP_SIZE
#define LoadDataTile_COMPUTE_WG_SIZE 0
#define SUFFIX _allWarps
#include "detail/loadDataTile_block_template.hcl"

#define LoadDataTile_LOAD_DATA_BLOCK_SIZE LOAD_DATA_BLOCK_SIZE
#define LoadDataTile_LOAD_WG_SIZE LOAD_DATA_WG_SIZE
#define LoadDataTile_COMPUTE_WG_SIZE COMPUTE_WG_SIZE
#define SUFFIX _loadWarps
#include "detail/loadDataTile_block_template.hcl"

/////////////////////////////////////////////////////////////////////
inline void PreloadVectorData(__private half4* restrict cachedVector,
                              __global const half* restrict vector) {
  const int laneLid = get_sub_group_local_id();
#pragma unroll
  for (int colIdx = laneLid; colIdx < COMPUTE_GEMV_BLOCK_COLUMS / VECTOR_WIDTH;
       colIdx += WARP_SIZE) {
    cachedVector[colIdx / WARP_SIZE] =
        vload4(0, vector + colIdx * VECTOR_WIDTH);
  }
}

///////////////////////////////////////////////////////////////
inline void SwapPtr(__local half* restrict __private* a,
                    __local half* restrict __private* b) {
  __local half* temp = *a;
  *a = *b;
  *b = temp;
}

// Each block handles ROWS_FOR_BLOCK_FOR_PHASE rows, and each subgroup handles
// ROWS_FOR_COMPUTE_WARP rows. All compute subgroups cooperate to compute the
// dot products for their assigned rows.
__attribute__((reqd_work_group_size(TOTAL_WARPS * WARP_SIZE, 1, 1)))
__attribute__((intel_reqd_sub_group_size(WARP_SIZE))) __kernel void
gemv(__global const half* restrict matrix, __global const half* restrict vector,
     __global half* restrict result, const uint rowCount,
     const uint columnCount) {
  __local half matrixBlockBuff1_local[LOAD_DATA_BLOCK_SIZE];
  __local half matrixBlockBuff2_local[LOAD_DATA_BLOCK_SIZE];

  __global half* restrict result_block =
      result + get_group_id(0) * TOTAL_ROWS_FOR_BLOCK;

  __global const half* restrict matrixBlock_global =
      matrix +
      get_group_id(0) * TOTAL_ROWS_FOR_BLOCK * COMPUTE_GEMV_BLOCK_COLUMS;

  __local half* restrict computeBuffer =
      (__local half* restrict)matrixBlockBuff2_local;
  __local half* restrict loadBuffer =
      (__local half* restrict)matrixBlockBuff1_local;

  // ---------------------------------------------------
  // Preload vector data into registers for reuse across dot products.
  half4 cachedVector_thisWarp[COMPUTE_GEMV_BLOCK_COLUMS / VECTOR_WIDTH /
                              WARP_SIZE];
  //---------------------------------------------------------

  IN_KERNEL_PROFILE(
      LoadDataTile_allWarps(loadBuffer,
                            matrixBlock_global + 0 * LOAD_DATA_BLOCK_SIZE),
      "INITIAL LoadDataTile_allWarps");

  IN_KERNEL_PROFILE(barrier(CLK_LOCAL_MEM_FENCE), "Initial Barrier");

  if (get_sub_group_id() < COMPUTE_WARPS) {
    // Preload vector data into registers for reuse across dot products.
    IN_KERNEL_PROFILE(PreloadVectorData(cachedVector_thisWarp, vector),
                      "PreloadVectorData");
  }

#pragma unroll
  for (int phase = 0; phase < PHASES_PER_BLOCK - 1; ++phase) {
    SwapPtr(&computeBuffer, &loadBuffer);

    if (get_sub_group_id() < COMPUTE_WARPS) {
      IN_KERNEL_PROFILE(
          ComputeGemvTileActiveWarps(
              (__local half4* restrict)computeBuffer, cachedVector_thisWarp,
              result_block + phase * ROWS_FOR_BLOCK_FOR_PHASE),
          "ComputeGemvTileActiveWarps");
    } else {
      IN_KERNEL_PROFILE(LoadDataTile_loadWarps(
                            loadBuffer, matrixBlock_global +
                                            (phase + 1) * LOAD_DATA_BLOCK_SIZE),
                        "LoadDataTile_loadWarps");
    }

    IN_KERNEL_PROFILE(barrier(CLK_LOCAL_MEM_FENCE), "Barrier");
  }

  SwapPtr(&computeBuffer, &loadBuffer);
  if (get_sub_group_id() < COMPUTE_WARPS) {
    IN_KERNEL_PROFILE(
        ComputeGemvTileActiveWarps(
            (__local half4* restrict)computeBuffer, cachedVector_thisWarp,
            result_block + (PHASES_PER_BLOCK - 1) * ROWS_FOR_BLOCK_FOR_PHASE),
        "Last ComputeGemvTileActiveWarps");
  }
}