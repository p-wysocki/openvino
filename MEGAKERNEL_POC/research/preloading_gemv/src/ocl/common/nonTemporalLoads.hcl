#pragma once

// Intel-specific extension for LSC load/store
// https://github.com/intel/intel-graphics-compiler/blob/a7ef0163286db1b56e9acfd8565c5462ee6aaea0/IGC/BiFModule/Implementation/IGCBiF_Intrinsics_Lsc.cl
enum LSC_LDCC {
  LSC_LDCC_DEFAULT = 0,
  LSC_LDCC_L1UC_L3UC = 1,  // Override to L1 uncached and L3 uncached
  LSC_LDCC_L1UC_L3C = 2,   // Override to L1 uncached and L3 cached
  LSC_LDCC_L1C_L3UC = 3,   // Override to L1 cached and L3 uncached
  LSC_LDCC_L1C_L3C = 4,    // Override to L1 cached and L3 cached
  LSC_LDCC_L1S_L3UC = 5,   // Override to L1 streaming load and L3 uncached
  LSC_LDCC_L1S_L3C = 6,    // Override to L1 streaming load and L3 cached
  LSC_LDCC_L1IAR_L3C =
      7,  // Override to L1 invalidate-after-read, and L3 cached
  LSC_LDCC_L1_L2_L3_DEF = 16,

  LSC_LDCC_L1UC_L2UC_L3UC =
      18,  // Override to L1 uncached, L2 uncached, L3 uncached
  LSC_LDCC_L1UC_L2UC_L3C =
      19,  // Override to L1 uncached, L2 uncached, L3 cached
  LSC_LDCC_L1UC_L2C_L3UC =
      20,  // Override to L1 uncached, L2 cached, L3 uncached
  LSC_LDCC_L1UC_L2C_L3C = 21,  // Override to L1 uncached, L2 cached, L3 cached

  LSC_LDCC_L1C_L2UC_L3UC =
      22,  // Override to L1 cached, L2 uncached, L3 uncached
  LSC_LDCC_L1C_L2UC_L3C = 23,  // Override to L1 cached, L2 uncached, L3 cached
  LSC_LDCC_L1C_L2C_L3UC = 24,  // Override to L1 cached, L2 cached, L3 uncached
  LSC_LDCC_L1C_L2C_L3C = 25,   // Override to L1 cached, L2 cached, L3 cached

  LSC_LDCC_L1S_L2UC_L3UC =
      26,  // Override to L1 streaming load, L2 uncached, L3 uncached
  LSC_LDCC_L1S_L2UC_L3C =
      27,  // Override to L1 streaming load, L2 uncached, L3 cached
  LSC_LDCC_L1S_L2C_L3UC =
      28,  // Override to L1 streaming load, L2 cached, L3 uncached
  LSC_LDCC_L1S_L2C_L3C =
      29,  // Override to L1 streaming load, L2 cached, L3 cached

  LSC_LDCC_L1IAR_L2IAR_L3IAR =
      30,  // Override to L1, L2, L3 invalidate-after-read
};

extern uint4 __builtin_IB_lsc_load_global_uint4(const __global uint4* base,
                                                int immElemOff,
                                                enum LSC_LDCC cacheControl);

//////////////////////////////////////////////////////////////////
inline half8 NontemporalLoad(__global const half8* ptr) {
  const uint4 value = __builtin_IB_lsc_load_global_uint4(
      (const __global uint4*)ptr, 0, LSC_LDCC_L1C_L3UC);
  return as_half8(value);
}