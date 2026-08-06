#pragma once

// Whole block waits for the semaphore to reach the wanted value.
// Sync is done by thread 0 of selected warp.
inline void WaitForSemaphore_block(int warpID,
                                   volatile __global atomic_int* syncMemory,
                                   int wantedSyncVal);

// Whole block signals the semaphore by incrementing it by 1.
// Sync is done by thread 0 of selected warp.
inline void SignalSemaphore_block(int warpID,
                                  volatile __global atomic_int* syncMemory);

//////////////////////////////////////////////////////////////////////////////////////
//
// INLINES:
//
//////////////////////////////////////////////////////////////////////////////////////

//////////////////////////////////////////////////////////////////////////////////////
inline void WaitForSemaphore_block(int warpID,
                                   volatile __global atomic_int* syncMemory,
                                   int wantedSyncVal) {
  if (syncMemory != NULL && get_sub_group_id() == warpID &&
      get_sub_group_local_id() == 0) {
    // TODO: memory_order_relaxed should be sufficient here.
    int val = 0;
    do {
      val = atomic_load_explicit(syncMemory, memory_order_acquire,
                                 memory_scope_device);
    } while (val == 0 || (val % wantedSyncVal) != 0);
  }
  barrier(CLK_GLOBAL_MEM_FENCE);
}

//////////////////////////////////////////////////////////////////////////////////////
inline void SignalSemaphore_block(int warpID,
                                  volatile __global atomic_int* syncMemory) {
  barrier(CLK_GLOBAL_MEM_FENCE);
  if (syncMemory != NULL && get_sub_group_id() == warpID &&
      get_sub_group_local_id() == 0) {
    // TODO: memory_order_relaxed should be sufficient here.
    atomic_fetch_add_explicit(syncMemory, 1, memory_order_release,
                              memory_scope_device);
  }
}