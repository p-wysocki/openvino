// Tiled GEMV specialized for 32-wide subgroups.
// Each subgroup computes ROWS_FOR_COMPUTE_WARP rows so vector loads are reused
// across dot products.
//
// Launch configuration (host side):
//   global = (ceil(rowCount / TOTAL_ROWS_FOR_BLOCK) * WG_SIZE)
//   local  = (WG_SIZE)
#pragma OPENCL EXTENSION cl_khr_fp16 : enable
// #pragma OPENCL EXTENSION cl_khr_subgroup_named_barrier : enable

#define TOTAL_ROWS_FOR_BLOCK 16
#define TOTAL_WARPS 8
#define COMPUTE_WARPS 4
#define ROWS_FOR_COMPUTE_WARP 1
#define COL_BLOCKS_PER_LOOP 4
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
inline void ComputeGemvTile_block(
    __local const half4* restrict matrixTile_local,
    __private const float4 (*restrict cachedVector)[COL_BLOCKS_PER_LOOP],
    __global half* restrict result) {
#define VECTOR_WIDTH 4
#define VECTOR_ITEMS_FOR_WARP (WARP_SIZE * VECTOR_WIDTH)
#define COL_ITEMS_PER_LOOP (COL_BLOCKS_PER_LOOP * VECTOR_ITEMS_FOR_WARP)

  const int laneLid = get_sub_group_local_id();
  const int rowBase = get_sub_group_id() * ROWS_FOR_COMPUTE_WARP;
  bool rowIsValid[ROWS_FOR_COMPUTE_WARP];
  float acc[ROWS_FOR_COMPUTE_WARP];

#pragma unroll ROWS_FOR_COMPUTE_WARP
  for (int rowIdx = 0; rowIdx < ROWS_FOR_COMPUTE_WARP; ++rowIdx) {
    rowIsValid[rowIdx] = (rowBase + rowIdx) < COMPUTE_GEMV_BLOCK_ROWS;
    acc[rowIdx] = 0.0f;
  }

  // Compute dot products for assigned rows.
#pragma unroll
  for (int col = laneLid * VECTOR_WIDTH; col < COMPUTE_GEMV_BLOCK_COLUMS;
       col += COL_ITEMS_PER_LOOP) {
#pragma unroll COL_BLOCKS_PER_LOOP
    for (uint blockIdx = 0; blockIdx < COL_BLOCKS_PER_LOOP; ++blockIdx) {
      const float4 vectorData =
          cachedVector[col / COL_ITEMS_PER_LOOP][blockIdx];
#pragma unroll ROWS_FOR_COMPUTE_WARP
      for (int rowIdx = 0; rowIdx < ROWS_FOR_COMPUTE_WARP; ++rowIdx) {
        if (rowIsValid[rowIdx]) {
          const int rowOffset = (rowBase + rowIdx) * COMPUTE_GEMV_BLOCK_COLUMS;
          const int colOffset = col + blockIdx * VECTOR_ITEMS_FOR_WARP;
          const float4 matrixData =
              convert_float4(matrixTile_local[(rowOffset + colOffset) / 4]);
          acc[rowIdx] += dot(matrixData, vectorData);
        }
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
        const int outputIdx = rowBase + rowIdx;
        result[outputIdx] = convert_half_rte(reduced[rowIdx]);
      }
    }
  }
}

#define LOAD_DATA_BLOCK_SIZE ROWS_FOR_BLOCK_FOR_PHASE* COMPUTE_GEMV_BLOCK_COLUMS
inline void LoadDataTile_block(__local half* restrict matrixBlock_local,
                               __global const half* restrict matrixBlock_global,
                               int computeWGSize, int loadDataWGSize) {
  __local half8* restrict matrixBlock_local8 =
      (__local half8* restrict)matrixBlock_local;
  __global half8* restrict matrixBlock_global8 =
      (__global half8* restrict)matrixBlock_global;

#pragma unroll
  for (int i = get_local_id(0) - computeWGSize; i < LOAD_DATA_BLOCK_SIZE / 8;
       i += loadDataWGSize) {
    matrixBlock_local8[i] = matrixBlock_global8[i];
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
  float4 cachedVector_thisWarp[COMPUTE_GEMV_BLOCK_COLUMS / COL_ITEMS_PER_LOOP]
                              [COL_BLOCKS_PER_LOOP];
  //---------------------------------------------------------

  LoadDataTile_block(loadBuffer, matrixBlock_global + 0 * LOAD_DATA_BLOCK_SIZE,
                     0, TOTAL_WARPS * WARP_SIZE);

  __asm__ volatile("barrier");

  if (get_sub_group_id() < COMPUTE_WARPS) {
    // Preload vector data into registers for reuse across dot products.
    const int laneLid = get_sub_group_local_id();
#pragma unroll
    for (int col = laneLid * VECTOR_WIDTH; col < COMPUTE_GEMV_BLOCK_COLUMS;
         col += COL_ITEMS_PER_LOOP) {
#pragma unroll COL_BLOCKS_PER_LOOP
      for (uint blockIdx = 0; blockIdx < COL_BLOCKS_PER_LOOP; ++blockIdx) {
        const int colOffset = col + blockIdx * VECTOR_ITEMS_FOR_WARP;
        cachedVector_thisWarp[col / COL_ITEMS_PER_LOOP][blockIdx] =
            convert_float4(vload4(0, vector + colOffset));
      }
    }
    // ----------------------------------------------------------------
  }

  for (int phase = 0; phase < PHASES_PER_BLOCK - 1; ++phase) {
    SwapPtr(&computeBuffer, &loadBuffer);

    if (get_sub_group_id() < COMPUTE_WARPS) {
      ComputeGemvTile_block((__local half4* restrict)computeBuffer,
                            cachedVector_thisWarp,
                            result_block + phase * ROWS_FOR_BLOCK_FOR_PHASE);
    } else {
      LoadDataTile_block(
          loadBuffer, matrixBlock_global + (phase + 1) * LOAD_DATA_BLOCK_SIZE,
          COMPUTE_WG_SIZE, LOAD_DATA_WG_SIZE);
    }

    __asm__ volatile("barrier");
  }

  SwapPtr(&computeBuffer, &loadBuffer);
  if (get_sub_group_id() < COMPUTE_WARPS) {
    ComputeGemvTile_block((__local half4* restrict)computeBuffer,
                          cachedVector_thisWarp,
                          result_block + 3 * ROWS_FOR_BLOCK_FOR_PHASE);
  }
}