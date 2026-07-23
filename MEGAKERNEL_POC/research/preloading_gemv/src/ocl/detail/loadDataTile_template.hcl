#include "detail/nonTemporalLoads.hcl"
#include "detail/template.hcl"

#ifndef LoadDataTile_LOAD_DATA_BLOCK_SIZE
#error "LoadDataTile_LOAD_DATA_BLOCK_SIZE is not defined"
#endif

#ifndef LoadDataTile_LOAD_WG_SIZE
#error "LoadDataTile_LOAD_WG_SIZE is not defined"
#endif

#ifndef LoadDataTile_COMPUTE_WG_SIZE
#error "LoadDataTile_COMPUTE_WG_SIZE is not defined"
#endif

inline void TEMPLATE(LoadDataTile,
                     SUFFIX)(__local half* restrict matrixBlock_local,
                             __global const half* restrict matrixBlock_global) {
  __local half8* restrict matrixBlock_local8 =
      (__local half8* restrict)matrixBlock_local;
  __global half8* restrict matrixBlock_global8 =
      (__global half8* restrict)matrixBlock_global;

#pragma unroll
  for (int i = get_local_id(0) - LoadDataTile_COMPUTE_WG_SIZE;
       i < LoadDataTile_LOAD_DATA_BLOCK_SIZE / 8;
       i += LoadDataTile_LOAD_WG_SIZE) {
    matrixBlock_local8[i] = NontemporalLoad(matrixBlock_global8 + i);
  }
}

#undef LoadDataTile_LOAD_DATA_BLOCK_SIZE
#undef LoadDataTile_LOAD_WG_SIZE
#undef LoadDataTile_COMPUTE_WG_SIZE
#undef SUFFIX