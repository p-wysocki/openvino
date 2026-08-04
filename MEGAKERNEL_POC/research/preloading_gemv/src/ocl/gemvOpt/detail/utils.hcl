#pragma once

// Utility functions for GEMV kernel.
void SwapPtr(__local half* restrict __private* a,
             __local half* restrict __private* b);

//////////////////////////////////////////////////////////////
//
// INLINES:
//
/////////////////////////////////////////////////////////////

inline void SwapPtr(__local half* restrict __private* a,
                    __local half* restrict __private* b) {
  __local half* temp = *a;
  *a = *b;
  *b = temp;
}