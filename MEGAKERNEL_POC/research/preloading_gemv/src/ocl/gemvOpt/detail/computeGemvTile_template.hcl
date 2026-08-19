#include "common/template.hcl"

#ifndef ComputeGemvTile_SUFFIX
#define ComputeGemvTile_SUFFIX
#endif

// Template function.
// Computes gemv for given tile. Each warp computes multiple rows of the tile.
// cachedVector is assumed to be Preloaded into private memory with
// PreloadVectorData.
// Result is stored directly in global memory.

// Requires template parameters:
// #define ComputeGemvTile_TILE_ROWS
// #define ComputeGemvTile_TILE_COLUMNS
// #define ComputeGemvTile_COMPUTE_WARPS

// Optional parameter to give unique name of template instantiation.
// #define ComputeGemvTile_SUFFIX
void TEMPLATE(ComputeGemvTile, ComputeGemvTile_SUFFIX)(
    __local const half* restrict matrixTile_local,
    __global const half* restrict cachedVector, __global half* restrict result);

////////////////////////////////////////////////////////////////
//
// IMPLEMENTATION
//
////////////////////////////////////////////////////////////////

#ifndef ComputeGemvTile_TILE_ROWS
#error "ComputeGemvTile_TILE_ROWS is not defined"
#endif

#ifndef ComputeGemvTile_TILE_COLUMNS
#error "ComputeGemvTile_TILE_COLUMNS is not defined"
#endif

#ifndef ComputeGemvTile_COMPUTE_WARPS
#error "ComputeGemvTile_COMPUTE_WARPS is not defined"
#endif

#define ComputeGemvTile_ROWS_FOR_COMPUTE_WARP \
  (ComputeGemvTile_TILE_ROWS / ComputeGemvTile_COMPUTE_WARPS)

_Static_assert(ComputeGemvTile_TILE_ROWS % ComputeGemvTile_COMPUTE_WARPS == 0,
               "ComputeGemvTile_TILE_ROWS must be divisible by "
               "ComputeGemvTile_COMPUTE_WARPS");

////////////////////////////////////////////////////////////////
inline void TEMPLATE(ComputeGemvTile, ComputeGemvTile_SUFFIX)(
    __local const half* restrict matrixTile_local,
    __global const half* restrict vector, __global half* restrict result) {
  const int laneLid = get_sub_group_local_id();
  const int startingRowIdxForThisWarp =
      get_sub_group_id() * ComputeGemvTile_ROWS_FOR_COMPUTE_WARP;
  __local const half* restrict matrixTileForThisWarp_local =
      matrixTile_local +
      startingRowIdxForThisWarp * ComputeGemvTile_TILE_COLUMNS;
  float acc[ComputeGemvTile_ROWS_FOR_COMPUTE_WARP];

#pragma unroll ComputeGemvTile_ROWS_FOR_COMPUTE_WARP
  for (int rowIdx = 0; rowIdx < ComputeGemvTile_ROWS_FOR_COMPUTE_WARP;
       ++rowIdx) {
    acc[rowIdx] = 0.0f;
  }

  // Compute dot products for assigned rows.
#pragma unroll
  for (int thisWarpOffset = 0; thisWarpOffset < ComputeGemvTile_TILE_COLUMNS;
       thisWarpOffset += WARP_SIZE * 8) {
    const __global ushort* vector_us =
        (const __global ushort*)(vector + thisWarpOffset);
    // NOTE: I have tried a version where vector is cached in registers, but
    // due to lack of GRF, compiler started to move them to private memory
    // which was basically slower in some cases then loading vector multiple
    // times, since vector should be in cache.
    const float8 vectorData =
        convert_float8(as_half8(intel_sub_group_block_read_us8(vector_us)));

#pragma unroll
    for (int rowIdx = 0; rowIdx < ComputeGemvTile_ROWS_FOR_COMPUTE_WARP;
         ++rowIdx) {
      const int rowOffset =
          rowIdx * ComputeGemvTile_TILE_COLUMNS + thisWarpOffset;
      const __local ushort* matrix_us =
          (const __local ushort*)(matrixTileForThisWarp_local + rowOffset);
      const float8 matrixData =
          convert_float8(as_half8(intel_sub_group_block_read_us8(matrix_us)));
      const float8 mul = matrixData * vectorData;
      acc[rowIdx] +=
          mul.s0 + mul.s1 + mul.s2 + mul.s3 + mul.s4 + mul.s5 + mul.s6 + mul.s7;
    }
  }

  float reduced[ComputeGemvTile_ROWS_FOR_COMPUTE_WARP];
#pragma unroll ComputeGemvTile_ROWS_FOR_COMPUTE_WARP
  for (int rowIdx = 0; rowIdx < ComputeGemvTile_ROWS_FOR_COMPUTE_WARP;
       ++rowIdx) {
    const float rowAcc = acc[rowIdx];
    reduced[rowIdx] = sub_group_reduce_add(rowAcc);
  }

  // Save the results.
  if (laneLid == 0) {
#pragma unroll ComputeGemvTile_ROWS_FOR_COMPUTE_WARP
    for (int rowIdx = 0; rowIdx < ComputeGemvTile_ROWS_FOR_COMPUTE_WARP;
         ++rowIdx) {
      const int outputIdx = startingRowIdxForThisWarp + rowIdx;
      result[outputIdx] = convert_half_rte(reduced[rowIdx]);
    }
  }
}

#undef ComputeGemvTile_TILE_ROWS
#undef ComputeGemvTile_TILE_COLUMNS
#undef ComputeGemvTile_COMPUTE_WARPS
#undef ComputeGemvTile_ROWS_FOR_COMPUTE_WARP
#undef ComputeGemvTile_SUFFIX