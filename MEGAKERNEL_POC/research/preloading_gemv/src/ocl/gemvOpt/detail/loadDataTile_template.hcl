#include "common/nonTemporalLoads.hcl"
#include "common/template.hcl"

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

  const int DATA_SIZE8 = LoadDataTile_LOAD_DATA_TILE_SIZE / 8;
  const int LEFTOWERS_SIZE = DATA_SIZE8 % (LoadDataTile_LOAD_WARPS * WARP_SIZE);
  const int PROCESSED_SIZE = DATA_SIZE8 - LEFTOWERS_SIZE;
  if (PROCESSED_SIZE > 0) {
#pragma unroll
    for (int i = 0; i < PROCESSED_SIZE;
       i += LoadDataTile_LOAD_WARPS * WARP_SIZE) {
      const int thisThread_i =
        i + get_local_id(0) - LoadDataTile_FIRST_LOAD_WARP_ID * WARP_SIZE;
#if LoadDataTile_NON_TEMPORAL_LOAD == 0
      dataTile_local8[thisThread_i] = dataBlock_global8[thisThread_i];
#else
      dataTile_local8[thisThread_i] =
        NontemporalLoad(dataBlock_global8 + thisThread_i);
#endif
      }
  }

  const int threadIdx = get_local_id(0) -
          LoadDataTile_FIRST_LOAD_WARP_ID * WARP_SIZE +
          PROCESSED_SIZE;
  if (threadIdx < DATA_SIZE8) {
#if LoadDataTile_NON_TEMPORAL_LOAD == 0
    dataTile_local8[threadIdx] = dataBlock_global8[threadIdx];
#else
    dataTile_local8[threadIdx] = NontemporalLoad(dataBlock_global8 + threadIdx);
#endif
  }
}

#undef LoadDataTile_LOAD_DATA_TILE_SIZE
#undef LoadDataTile_LOAD_WARPS
#undef LoadDataTile_FIRST_LOAD_WARP_ID
#undef LoadDataTile_NON_TEMPORAL_LOAD
#undef LoadDataTile_SUFFIX