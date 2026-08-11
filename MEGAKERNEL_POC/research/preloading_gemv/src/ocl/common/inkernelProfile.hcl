#pragma once

extern ulong __builtin_IB_read_cycle_counter(void);
// --------------------------------------
// #define PROFILE_IN_KERNEL
#ifdef PROFILE_IN_KERNEL
#define IN_KERNEL_PROFILE(FUNC, TXT)                             \
  {                                                              \
    const ulong start = __builtin_IB_read_cycle_counter();       \
    FUNC;                                                        \
    const ulong end = __builtin_IB_read_cycle_counter();         \
    if (get_sub_group_local_id() == 0 && get_group_id(0) == 6) { \
      printf(TXT " took %lu cycles for warp %d\n", end - start,  \
             get_sub_group_id());                                \
    }                                                            \
  }
#else
#define IN_KERNEL_PROFILE(FUNC, TXT) FUNC
#endif

//#define PROFILE_IN_KERNEL_BLOCK
#ifdef PROFILE_IN_KERNEL_BLOCK
#define IN_KERNEL_PROFILE_BLOCK(FUNC, TXT)                         \
  {                                                                \
    const ulong start = __builtin_IB_read_cycle_counter();         \
    FUNC;                                                          \
    const ulong end = __builtin_IB_read_cycle_counter();           \
    if (get_local_id(0) == 100) {                                  \
      printf(TXT " took %lu cycles for WORKER %lu\n", end - start, \
             get_group_id(0));                                     \
    }                                                              \
  }
#else
#define IN_KERNEL_PROFILE_BLOCK(FUNC, TXT) FUNC
#endif