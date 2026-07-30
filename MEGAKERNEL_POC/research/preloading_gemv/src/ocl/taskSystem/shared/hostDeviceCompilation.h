#pragma once


#ifdef __OPENCL_VERSION__
#define DEVICE_COMPILATION
#else
#define HOST_COMPILATION
#endif

#ifdef DEVICE_COMPILATION
#define GLOBAL_DEVICE_PTR __global
#else
#define GLOBAL_DEVICE_PTR
#endif
