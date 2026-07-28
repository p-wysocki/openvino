#include "detail/nonTemporalLoads.hcl"
#include "detail/template.hcl"

#ifndef LoadDataTile_SUFFIX
#define LoadDataTile_SUFFIX
#endif

// Template function.
// Loads tile of data from global memory to local memory. Data is assumed to be
// continous in global memory.
// Loading will be performed by LoadDataTile_LOAD_WARPS warps,
// with first loading warp defined by LoadDataTile_FIRST_LOAD_WARP_ID warp.

// Requires template parameters:
// #define LoadDataTile_LOAD_DATA_TILE_SIZE -> int
// #define LoadDataTile_LOAD_WARPS          -> int
// #define LoadDataTile_FIRST_LOAD_WARP_ID  -> int
// #define LoadDataTile_NON_TEMPORAL_LOAD   -> bool

// Optional parameter to give unique name of template instantiation.
// #define LoadDataTile_SUFFIX
inline void TEMPLATE(LoadDataTile, LoadDataTile_SUFFIX)(
    __local half* restrict dataTile_local,
    __global const half* restrict dataBlock_global);

////////////////////////////////////////////////////////////////
//
// IMPLEMENTATION
//
////////////////////////////////////////////////////////////////

#ifndef LoadDataTile_LOAD_DATA_TILE_SIZE
#error "LoadDataTile_LOAD_DATA_TILE_SIZE is not defined"
#endif

#ifndef LoadDataTile_LOAD_WARPS
#error "LoadDataTile_LOAD_WARPS is not defined"
#endif

#ifndef LoadDataTile_FIRST_LOAD_WARP_ID
#error "LoadDataTile_FIRST_LOAD_WARP_ID is not defined"
#endif

#ifndef LoadDataTile_NON_TEMPORAL_LOAD
#error "LoadDataTile_NON_TEMPORAL_LOAD is not defined"
#endif

inline void TEMPLATE(LoadDataTile, LoadDataTile_SUFFIX)(
    __local half* restrict dataTile_local,
    __global const half* restrict dataBlock_global) {
  __local half8* restrict dataTile_local8 =
      (__local half8* restrict)dataTile_local;
  __global half8* restrict dataBlock_global8 =
      (__global half8* restrict)dataBlock_global;

#pragma unroll
  for (int i = get_local_id(0) - LoadDataTile_FIRST_LOAD_WARP_ID * WARP_SIZE;
       i < LoadDataTile_LOAD_DATA_TILE_SIZE / 8;
       i += LoadDataTile_LOAD_WARPS * WARP_SIZE) {
#if LoadDataTile_NON_TEMPORAL_LOAD == 0
    dataTile_local8[i] = dataBlock_global8[i];
#else
    dataTile_local8[i] = NontemporalLoad(dataBlock_global8 + i);
#endif
  }
}

#undef LoadDataTile_LOAD_DATA_TILE_SIZE
#undef LoadDataTile_LOAD_WARPS
#undef LoadDataTile_FIRST_LOAD_WARP_ID
#undef LoadDataTile_NON_TEMPORAL_LOAD
#undef LoadDataTile_SUFFIX