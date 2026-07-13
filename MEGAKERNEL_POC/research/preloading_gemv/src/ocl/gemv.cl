// Tiled GEMV specialized for 32-wide subgroups.
// Each subgroup computes ROWS_PER_SUBGROUP rows so vector loads are reused
// across dot products.
//
// Launch configuration (host side):
//   global = (ceil(rowCount / ROWS_PER_GROUP) * WG_SIZE)
//   local  = (WG_SIZE)
#pragma OPENCL EXTENSION cl_khr_fp16 : enable

#define ROWS_PER_SUBGROUP 4
#define COL_BLOCKS_PER_LOOP 4
#define WG_SIZE 128u
#define SUBGROUP_SIZE 32

inline void computeGemv_block(__global const half* restrict matrix,
                              __global const half* restrict vector,
                              __global half* restrict result,
                              const uint rowCount, const uint columnCount) {
#define SUBGROUPS_PER_GROUP (WG_SIZE / SUBGROUP_SIZE)
#define ROWS_PER_GROUP (SUBGROUPS_PER_GROUP * ROWS_PER_SUBGROUP)
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
    rowIsValid[rowIdx] = rows[rowIdx] < rowCount;
    rowOffsets[rowIdx] = rows[rowIdx] * columnCount;
#pragma unroll COL_BLOCKS_PER_LOOP
    for (uint blockIdx = 0; blockIdx < COL_BLOCKS_PER_LOOP; ++blockIdx) {
      acc[rowIdx][blockIdx] = 0.0f;
    }
  }

  const uint vecLimit = columnCount & ~3u;
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

// Each block handles ROWS_PER_GROUP rows, and each subgroup handles
// ROWS_PER_SUBGROUP rows. All compute subgroups cooperate to compute the dot
// products for their assigned rows.
__attribute__((reqd_work_group_size(WG_SIZE, 1, 1)))
__attribute__((intel_reqd_sub_group_size(32)))
__attribute__((vec_type_hint(float4))) __kernel void
gemv(__global const half* restrict matrix, __global const half* restrict vector,
     __global half* restrict result, const uint rowCount,
     const uint columnCount) {
  __global const half* restrict matrix_block = matrix + get_group_id(0) * ROWS_PER_GROUP * columnCount;
  __global half* restrict result_block = result + get_group_id(0) * ROWS_PER_GROUP;
  computeGemv_block(matrix_block, vector, result_block, ROWS_PER_GROUP, columnCount);
}