#include "detail/template.hcl"

#ifndef ComputeGemvTile_SUFFIX
#define ComputeGemvTile_SUFFIX
#endif

// Preloads vector data into private memory.
// cached vector is assumed to have size defined by
// TEMPLATE(ComputeGemvTile_CACHE_SIZE, ComputeGemvTile_SUFFIX).

// Requires template parameters:
// #define ComputeGemvTile_TILE_COLUMNS
void TEMPLATE(PreloadVectorData,
              ComputeGemvTile_SUFFIX)(__private half4* restrict cachedVector,
                                      __local const half* restrict vector);

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
    __local const half4* restrict matrixTile_local,
    __private const half4* restrict cachedVector,
    __global half* restrict result);

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

#define ComputeGemvTile_DATA_WIDTH 4

enum {
  TEMPLATE(ComputeGemvTile_CACHE_SIZE, ComputeGemvTile_SUFFIX) =
      ComputeGemvTile_TILE_COLUMNS / ComputeGemvTile_DATA_WIDTH / WARP_SIZE
};

////////////////////////////////////////////////////////////////
inline void TEMPLATE(ComputeGemvTile, ComputeGemvTile_SUFFIX)(
    __local const half4* restrict matrixTile_local,
    __private const half4* restrict cachedVector,
    __global half* restrict result) {
  const int laneLid = get_sub_group_local_id();
  const int vectorizedNumColumns =
      ComputeGemvTile_TILE_COLUMNS / ComputeGemvTile_DATA_WIDTH;
  const int startingRowIdxForThisWarp =
      get_sub_group_id() * ComputeGemvTile_ROWS_FOR_COMPUTE_WARP;
  __local const half4* restrict matrixTileForThisWarp_local =
      matrixTile_local + startingRowIdxForThisWarp * vectorizedNumColumns;
  float acc[ComputeGemvTile_ROWS_FOR_COMPUTE_WARP];

#pragma unroll ComputeGemvTile_ROWS_FOR_COMPUTE_WARP
  for (int rowIdx = 0; rowIdx < ComputeGemvTile_ROWS_FOR_COMPUTE_WARP;
       ++rowIdx) {
    acc[rowIdx] = 0.0f;
  }

  // Compute dot products for assigned rows.
#pragma unroll
  for (int colIdx = laneLid; colIdx < vectorizedNumColumns;
       colIdx += WARP_SIZE) {
    const float4 vectorData = convert_float4(cachedVector[colIdx / WARP_SIZE]);
#pragma unroll
    for (int rowIdx = 0; rowIdx < ComputeGemvTile_ROWS_FOR_COMPUTE_WARP;
         ++rowIdx) {
      const int rowOffset = rowIdx * vectorizedNumColumns;
      const float4 matrixData =
          convert_float4(matrixTileForThisWarp_local[rowOffset + colIdx]);
      acc[rowIdx] += dot(matrixData, vectorData);
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

/////////////////////////////////////////////////////////////////////
inline void TEMPLATE(PreloadVectorData,
                     SUFFIX)(__private half4* restrict cachedVector,
                             __local const half* restrict vector) {
  const int laneLid = get_sub_group_local_id();
  __local const half4* restrict vector4 = (__local const half4* restrict)vector;
#pragma unroll
  for (int colIdx = laneLid;
       colIdx < ComputeGemvTile_TILE_COLUMNS / ComputeGemvTile_DATA_WIDTH;
       colIdx += WARP_SIZE) {
    cachedVector[colIdx / WARP_SIZE] = vector4[colIdx];
  }
}

#undef ComputeGemvTile_TILE_ROWS
#undef ComputeGemvTile_TILE_COLUMNS
#undef ComputeGemvTile_COMPUTE_WARPS
#undef ComputeGemvTile_ROWS_FOR_COMPUTE_WARP
#undef ComputeGemvTile_DATA_WIDTH
#undef ComputeGemvTile_SUFFIX