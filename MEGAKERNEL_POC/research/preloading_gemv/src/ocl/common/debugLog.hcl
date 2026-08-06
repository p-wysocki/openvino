#pragma once

#define DEBUG_LOG(WARP_ID, WARP_THREAD_ID, TXT, ...)           \
  if (get_sub_group_id() == WARP_ID &&                         \
      get_sub_group_local_id() == WARP_THREAD_ID) {            \
    printf("WORKER: %lu, " TXT, get_group_id(0), __VA_ARGS__); \
  }
