#include "detail/template.hcl"

// Preloads vector data into private memory.
// cached vector is assumed to have size defined by
// TEMPLATE(ComputeGemvTile_CACHE_SIZE, SUFFIX).

// Requires template parameters:
// #define ComputeGemvTile_TILE_COLUMNS
void TEMPLATE(PreloadVectorData, SUFFIX)(__private half4* restrict cachedVector,
                                         __global const half* restrict vector);

// Template function.
// Computes gemv for given tile. Each warp computes multiple rows of the tile.
// cachedVector is assumed to be Preloaded into private memory with
// PreloadVectorData.
// Result is stored directly in global memory.

// Requires template parameters:
// #define ComputeGemvTile_TILE_ROWS
// #define ComputeGemvTile_TILE_COLUMNS
// #define ComputeGemvTile_COMPUTE_WARPS
void TEMPLATE(ComputeGemvTile,
              SUFFIX)(__local const half4* restrict matrixTile_local,
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
#define ComputeGemvTile_DATA_WIDTH 4

enum {
  TEMPLATE(ComputeGemvTile_CACHE_SIZE, SUFFIX) =
      ComputeGemvTile_TILE_COLUMNS / ComputeGemvTile_DATA_WIDTH / WARP_SIZE
};

////////////////////////////////////////////////////////////////
inline void TEMPLATE(ComputeGemvTile,
                     SUFFIX)(__local const half4* restrict matrixTile_local,
                             __private const half4* restrict cachedVector,
                             __global half* restrict result) {
  const int laneLid = get_sub_group_local_id();
  const int startingRowIdxForThisWarp =
      get_sub_group_id() * ComputeGemvTile_ROWS_FOR_COMPUTE_WARP;
  bool rowIsValid[ComputeGemvTile_ROWS_FOR_COMPUTE_WARP];
  float acc[ComputeGemvTile_ROWS_FOR_COMPUTE_WARP];

#pragma unroll ComputeGemvTile_ROWS_FOR_COMPUTE_WARP
  for (int rowIdx = 0; rowIdx < ComputeGemvTile_ROWS_FOR_COMPUTE_WARP;
       ++rowIdx) {
    rowIsValid[rowIdx] =
        (startingRowIdxForThisWarp + rowIdx) < ComputeGemvTile_TILE_ROWS;
    acc[rowIdx] = 0.0f;
  }

  // Compute dot products for assigned rows.
#pragma unroll
  for (int colIdx = laneLid;
       colIdx < ComputeGemvTile_TILE_COLUMNS / ComputeGemvTile_DATA_WIDTH;
       colIdx += WARP_SIZE) {
    const float4 vectorData = convert_float4(cachedVector[colIdx / WARP_SIZE]);
#pragma unroll
    for (int rowIdx = 0; rowIdx < ComputeGemvTile_ROWS_FOR_COMPUTE_WARP;
         ++rowIdx) {
      if (rowIsValid[rowIdx]) {
        const int rowOffset =
            (startingRowIdxForThisWarp + rowIdx) *
            (ComputeGemvTile_TILE_COLUMNS / ComputeGemvTile_DATA_WIDTH);
        const float4 matrixData =
            convert_float4(matrixTile_local[rowOffset + colIdx]);
        acc[rowIdx] += dot(matrixData, vectorData);
      }
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
      if (rowIsValid[rowIdx]) {
        const int outputIdx = startingRowIdxForThisWarp + rowIdx;
        result[outputIdx] = convert_half_rte(reduced[rowIdx]);
      }
    }
  }
}

/////////////////////////////////////////////////////////////////////
inline void TEMPLATE(PreloadVectorData,
                     SUFFIX)(__private half4* restrict cachedVector,
                             __global const half* restrict vector) {
  const int laneLid = get_sub_group_local_id();
#pragma unroll
  for (int colIdx = laneLid;
       colIdx < ComputeGemvTile_TILE_COLUMNS / ComputeGemvTile_DATA_WIDTH;
       colIdx += WARP_SIZE) {
    cachedVector[colIdx / WARP_SIZE] =
        vload4(0, vector + colIdx * ComputeGemvTile_DATA_WIDTH);
  }
}

#undef ComputeGemvTile_TILE_ROWS
#undef ComputeGemvTile_TILE_COLUMNS
#undef ComputeGemvTile_COMPUTE_WARPS
#undef ComputeGemvTile_ROWS_FOR_COMPUTE_WARP
#undef ComputeGemvTile_DATA_WIDTH
#undef SUFFIX