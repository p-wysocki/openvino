// Tiled GEMV specialized for 32-wide subgroups.
// Each subgroup computes ROWS_PER_SUBGROUP rows so vector loads are reused
// across dot products.
//
// Launch configuration (host side):
//   global = (ceil(rowCount / ROWS_PER_GROUP) * WG_SIZE)
//   local  = (WG_SIZE)
#pragma OPENCL EXTENSION cl_khr_fp16 : enable
// #pragma OPENCL EXTENSION cl_khr_subgroup_named_barrier : enable

#define TOTAL_WARPS 8
#define COMPUTE_WARPS 4
#define ROWS_PER_SUBGROUP 1
#define COL_BLOCKS_PER_LOOP 4
#define COMPUTE_WG_SIZE (COMPUTE_WARPS * SUBGROUP_SIZE)
#define LOAD_DATA_WG_SIZE ((TOTAL_WARPS - COMPUTE_WARPS) * SUBGROUP_SIZE)
#define SUBGROUP_SIZE 32
#define SUBGROUPS_PER_GROUP (COMPUTE_WG_SIZE / SUBGROUP_SIZE)
#define ROWS_PER_GROUP (SUBGROUPS_PER_GROUP * ROWS_PER_SUBGROUP)
#define COMPUTE_GEMV_BLOCK_ROWS ROWS_PER_GROUP
#define COMPUTE_GEMV_BLOCK_COLUMS 1024

inline void computeGemv_block(__local const half* restrict matrix,
                              __global const half* restrict vector,
                              __global half* restrict result) {
#define VECTOR_WIDTH 4u
#define VECTOR_ITEMS_FOR_SUBGROUP (SUBGROUP_SIZE * VECTOR_WIDTH)
#define COL_ITEMS_PER_LOOP (COL_BLOCKS_PER_LOOP * VECTOR_ITEMS_FOR_SUBGROUP)
#define LAST_COL_BLOCK_OFFSET \
  ((COL_BLOCKS_PER_LOOP - 1) * VECTOR_ITEMS_FOR_SUBGROUP)

  const uint laneLid = get_sub_group_local_id();
  const uint rowBase = get_sub_group_id() * ROWS_PER_SUBGROUP;
  uint rows[ROWS_PER_SUBGROUP];
  bool rowIsValid[ROWS_PER_SUBGROUP];
  uint rowOffsets[ROWS_PER_SUBGROUP];
  float acc[ROWS_PER_SUBGROUP][COL_BLOCKS_PER_LOOP];

#pragma unroll ROWS_PER_SUBGROUP
  for (uint rowIdx = 0; rowIdx < ROWS_PER_SUBGROUP; ++rowIdx) {
    rows[rowIdx] = rowBase + rowIdx;
    rowIsValid[rowIdx] = rows[rowIdx] < COMPUTE_GEMV_BLOCK_ROWS;
    rowOffsets[rowIdx] = rows[rowIdx] * COMPUTE_GEMV_BLOCK_COLUMS;
#pragma unroll COL_BLOCKS_PER_LOOP
    for (uint blockIdx = 0; blockIdx < COL_BLOCKS_PER_LOOP; ++blockIdx) {
      acc[rowIdx][blockIdx] = 0.0f;
    }
  }

  const uint vecLimit = COMPUTE_GEMV_BLOCK_COLUMS & ~3u;
  uint col = laneLid << 2;

#pragma unroll
  for (; col + LAST_COL_BLOCK_OFFSET < vecLimit; col += COL_ITEMS_PER_LOOP) {
    uint blockCols[COL_BLOCKS_PER_LOOP];
    float4 blockVectors[COL_BLOCKS_PER_LOOP];
#pragma unroll COL_BLOCKS_PER_LOOP
    for (uint blockIdx = 0; blockIdx < COL_BLOCKS_PER_LOOP; ++blockIdx) {
      blockCols[blockIdx] = col + blockIdx * VECTOR_ITEMS_FOR_SUBGROUP;
      blockVectors[blockIdx] =
          convert_float4(vload4(0, vector + blockCols[blockIdx]));
    }

#pragma unroll ROWS_PER_SUBGROUP
    for (uint rowIdx = 0; rowIdx < ROWS_PER_SUBGROUP; ++rowIdx) {
      if (rowIsValid[rowIdx]) {
#pragma unroll COL_BLOCKS_PER_LOOP
        for (uint blockIdx = 0; blockIdx < COL_BLOCKS_PER_LOOP; ++blockIdx) {
          acc[rowIdx][blockIdx] +=
              dot(convert_float4(vload4(
                      0, matrix + rowOffsets[rowIdx] + blockCols[blockIdx])),
                  blockVectors[blockIdx]);
        }
      }
    }
  }

  float reduced[ROWS_PER_SUBGROUP];
#pragma unroll ROWS_PER_SUBGROUP
  for (uint rowIdx = 0; rowIdx < ROWS_PER_SUBGROUP; ++rowIdx) {
    float rowAcc = acc[rowIdx][1];
#pragma unroll COL_BLOCKS_PER_LOOP
    for (uint blockIdx = 2; blockIdx < COL_BLOCKS_PER_LOOP; ++blockIdx) {
      rowAcc += acc[rowIdx][blockIdx];
    }
    rowAcc = acc[rowIdx][0] + rowAcc;
    reduced[rowIdx] = sub_group_reduce_add(rowAcc);
  }

  // Save the results.
  if (laneLid == 0) {
#pragma unroll ROWS_PER_SUBGROUP
    for (uint rowIdx = 0; rowIdx < ROWS_PER_SUBGROUP; ++rowIdx) {
      if (rowIsValid[rowIdx]) {
        result[rows[rowIdx]] = convert_half_rte(reduced[rowIdx]);
      }
    }
  }
}

#define LOAD_DATA_BLOCK_SIZE ROWS_PER_GROUP* COMPUTE_GEMV_BLOCK_COLUMS
inline void LoadData_block(__local half* restrict matrixBlock_local,
                           __global const half* restrict matrixBlock_global,
                           int computeWGSize, int loadDataWGSize) {
  __local half16* restrict matrixBlock_local16 =
      (__local half16* restrict)matrixBlock_local;
  __global half16* restrict matrixBlock_global16 =
      (__global half16* restrict)matrixBlock_global;

#pragma unroll
  for (int i = get_local_id(0) - computeWGSize; i < LOAD_DATA_BLOCK_SIZE / 16;
       i += loadDataWGSize) {
    matrixBlock_local16[i] = matrixBlock_global16[i];
  }
}

// Each block handles ROWS_PER_GROUP rows, and each subgroup handles
// ROWS_PER_SUBGROUP rows. All compute subgroups cooperate to compute the dot
// products for their assigned rows.
__attribute__((reqd_work_group_size(TOTAL_WARPS * SUBGROUP_SIZE, 1, 1)))
__attribute__((intel_reqd_sub_group_size(32)))
__attribute__((vec_type_hint(float4))) __kernel void
gemv(__global const half* restrict matrix, __global const half* restrict vector,
     __global half* restrict result, const uint rowCount,
     const uint columnCount) {
  __local half
      matrixBlockBuff1_local[ROWS_PER_GROUP * COMPUTE_GEMV_BLOCK_COLUMS];

  __local half
      matrixBlockBuff2_local[ROWS_PER_GROUP * COMPUTE_GEMV_BLOCK_COLUMS];

  __global half* restrict result_block =
      result + get_group_id(0) * 4 * ROWS_PER_GROUP;

  __global const half* restrict matrixBlock_global =
      matrix + get_group_id(0) * 4 * ROWS_PER_GROUP * COMPUTE_GEMV_BLOCK_COLUMS;

  // named_barrier_t barrier_team_A = sub_group_barrier_init(0,
  // get_sub_group_size());

  // ---------------------------------------------------

  LoadData_block(
      matrixBlockBuff1_local,
      matrixBlock_global + 0 * ROWS_PER_GROUP * COMPUTE_GEMV_BLOCK_COLUMS, 0,
      TOTAL_WARPS * SUBGROUP_SIZE);

  //barrier(CLK_LOCAL_MEM_FENCE);
  __asm__ volatile("barrier");

  if (get_sub_group_id() < COMPUTE_WARPS) {
    computeGemv_block(matrixBlockBuff1_local, vector,
                      result_block + 0 * ROWS_PER_GROUP);
  } else {
    LoadData_block(
        matrixBlockBuff2_local,
        matrixBlock_global + 1 * ROWS_PER_GROUP * COMPUTE_GEMV_BLOCK_COLUMS,
        COMPUTE_WG_SIZE, LOAD_DATA_WG_SIZE);
  }

  //barrier(CLK_LOCAL_MEM_FENCE);
  __asm__ volatile("barrier");

  if (get_sub_group_id() < COMPUTE_WARPS) {
    computeGemv_block(matrixBlockBuff2_local, vector,
                      result_block + 1 * ROWS_PER_GROUP);
  } else {
    LoadData_block(
        matrixBlockBuff1_local,
        matrixBlock_global + 2 * ROWS_PER_GROUP * COMPUTE_GEMV_BLOCK_COLUMS,
        COMPUTE_WG_SIZE, LOAD_DATA_WG_SIZE);
  }

  //barrier(CLK_LOCAL_MEM_FENCE);
  __asm__ volatile("barrier");

  if (get_sub_group_id() < COMPUTE_WARPS) {
    computeGemv_block(matrixBlockBuff1_local, vector,
                      result_block + 2 * ROWS_PER_GROUP);
  } else {
    LoadData_block(
        matrixBlockBuff2_local,
        matrixBlock_global + 3 * ROWS_PER_GROUP * COMPUTE_GEMV_BLOCK_COLUMS,
        COMPUTE_WG_SIZE, LOAD_DATA_WG_SIZE);
  }

  //barrier(CLK_LOCAL_MEM_FENCE);
  __asm__ volatile("barrier");

  if (get_sub_group_id() < COMPUTE_WARPS) {
    computeGemv_block(matrixBlockBuff2_local, vector,
                      result_block + 3 * ROWS_PER_GROUP);
  }
}