#pragma once
#include <iostream>
#include <string>
#include <vector>

#include "ocltestCommon.h"

namespace ocltest {
struct ProfileResult {
  double averageUs = 0.0;
  size_t iterations = 0;
  std::vector<float> result;

  // Prints the profiling result to the console with a label.
  void print(const std::string& label) const;
};

// Profiles execution of kernels submitted by submitFunc and returns the average
// execution time in microseconds.
// If CLEAR_CACHE is true, the L3 cache will be saturated before each kernel
// submission to minimize the effect of caching on the profiling results.
template <bool CLEAR_CACHE, typename TSUBMIT_FUNC>
ProfileResult ProfileOpenCL(const TSUBMIT_FUNC& submitFunc,
                            cl_command_queue queue, size_t warmupIterations,
                            size_t benchmarkIterations);

/////////////////////////////////////////////////////////////////////
//
// IMPLEMENTATION
//
////////////////////////////////////////////////////////////////////

////////////////////////////////////////////////////////////////////
inline constexpr char saturateL3CacheKernelSource[] = R"(
__kernel void saturate_l3_cache(__global uint* data, uint element_count) {
  const size_t index = get_global_id(0);
  if (index < element_count) {
    data[index] = data[index] * 1664525u + 1013904223u;
  }
}
)";

////////////////////////////////////////////////////////////////////
inline cl_int EnqueueSaturateL3Cache(cl_command_queue queue, cl_kernel kernel,
                                     cl_mem buffer, size_t bufferSizeBytes) {
  const cl_uint elementCount =
      static_cast<cl_uint>(bufferSizeBytes / sizeof(cl_uint));
  if (elementCount == 0) {
    return CL_INVALID_BUFFER_SIZE;
  }

  cl_int status = clSetKernelArg(kernel, 0, sizeof(buffer), &buffer);
  if (status != CL_SUCCESS) {
    return status;
  }
  status = clSetKernelArg(kernel, 1, sizeof(elementCount), &elementCount);
  if (status != CL_SUCCESS) {
    return status;
  }

  constexpr size_t localWorkSize = 256;
  const size_t globalWorkSize =
      (elementCount + localWorkSize - 1) / localWorkSize * localWorkSize;
  return clEnqueueNDRangeKernel(queue, kernel, 1, nullptr, &globalWorkSize,
                                &localWorkSize, 0, nullptr, nullptr);
}

////////////////////////////////////////////////////////////////////
struct L3CacheSaturator {
  cl_program program = nullptr;
  cl_kernel kernel = nullptr;
  cl_mem buffer = nullptr;
  size_t bufferSizeBytes = 0;
};

////////////////////////////////////////////////////////////////////
inline L3CacheSaturator CreateL3CacheSaturator(cl_command_queue queue) {
  cl_context context = nullptr;
  cl_device_id device = nullptr;
  ASSERT_OCL_SUCCESS(clGetCommandQueueInfo(queue, CL_QUEUE_CONTEXT,
                                           sizeof(context), &context, nullptr));
  ASSERT_OCL_SUCCESS(clGetCommandQueueInfo(queue, CL_QUEUE_DEVICE,
                                           sizeof(device), &device, nullptr));

  cl_ulong cacheSizeBytes = 0;
  ASSERT_OCL_SUCCESS(clGetDeviceInfo(device, CL_DEVICE_GLOBAL_MEM_CACHE_SIZE,
                                     sizeof(cacheSizeBytes), &cacheSizeBytes,
                                     nullptr));
  constexpr size_t fallbackCacheSizeBytes = 16 * 1024 * 1024;
  const size_t bufferSizeBytes =
      4 * (cacheSizeBytes == 0 ? fallbackCacheSizeBytes
                               : static_cast<size_t>(cacheSizeBytes));

  cl_int status = CL_SUCCESS;
  const char* source = saturateL3CacheKernelSource;
  const size_t sourceSize = sizeof(saturateL3CacheKernelSource) - 1;
  cl_program program =
      clCreateProgramWithSource(context, 1, &source, &sourceSize, &status);
  ASSERT_OCL_SUCCESS(status);
  ASSERT_OCL_SUCCESS(
      clBuildProgram(program, 1, &device, "-Werror", nullptr, nullptr));
  cl_kernel kernel = clCreateKernel(program, "saturate_l3_cache", &status);
  ASSERT_OCL_SUCCESS(status);
  cl_mem buffer = clCreateBuffer(context, CL_MEM_READ_WRITE, bufferSizeBytes,
                                 nullptr, &status);
  ASSERT_OCL_SUCCESS(status);

  return {program, kernel, buffer, bufferSizeBytes};
}

////////////////////////////////////////////////////////////////////
inline void ReleaseL3CacheSaturator(const L3CacheSaturator& saturator) {
  ASSERT_OCL_SUCCESS(clReleaseMemObject(saturator.buffer));
  ASSERT_OCL_SUCCESS(clReleaseKernel(saturator.kernel));
  ASSERT_OCL_SUCCESS(clReleaseProgram(saturator.program));
}

////////////////////////////////////////////////////////////////////
inline void ProfileResult::print(const std::string& label) const {
  std::cout << label << " latency over " << iterations
            << " iterations: avg=" << averageUs << " us\n";
}

////////////////////////////////////////////////////////////////////
template <typename T>
ProfileResult ProfileOpenCL_Impl(const T& submitFunc, cl_command_queue queue,
                                 size_t warmupIterations,
                                 size_t benchmarkIterations) {
  for (size_t iteration = 0; iteration < warmupIterations; ++iteration) {
    submitFunc();
  }
  ASSERT_OCL_SUCCESS(clFinish(queue));

  cl_event startEvent = nullptr;
  cl_event endEvent = nullptr;
  ASSERT_OCL_SUCCESS(
      clEnqueueMarkerWithWaitList(queue, 0, nullptr, &startEvent));

  for (size_t iteration = 0; iteration < benchmarkIterations; ++iteration) {
    submitFunc();
  }

  ASSERT_OCL_SUCCESS(clEnqueueMarkerWithWaitList(queue, 0, nullptr, &endEvent));
  ASSERT_OCL_SUCCESS(clWaitForEvents(1, &endEvent));

  cl_ulong startNs = 0;
  cl_ulong endNs = 0;
  ASSERT_OCL_SUCCESS(
      clGetEventProfilingInfo(startEvent, CL_PROFILING_COMMAND_END,
                              sizeof(startNs), &startNs, nullptr));
  ASSERT_OCL_SUCCESS(clGetEventProfilingInfo(
      endEvent, CL_PROFILING_COMMAND_START, sizeof(endNs), &endNs, nullptr));
  ASSERT_OCL_SUCCESS(clReleaseEvent(endEvent));
  ASSERT_OCL_SUCCESS(clReleaseEvent(startEvent));

  ocltest::ProfileResult stats;
  stats.averageUs = (static_cast<double>(endNs - startNs) / 1000.0) /
                    static_cast<double>(benchmarkIterations);
  stats.iterations = benchmarkIterations;
  return stats;
}

////////////////////////////////////////////////////////////////////
template <bool CLEAR_CACHE, typename TSUBMIT_FUNC>
ProfileResult ProfileOpenCL(const TSUBMIT_FUNC& submitFunc,
                            cl_command_queue queue, size_t warmupIterations,
                            size_t benchmarkIterations) {
  if (!CLEAR_CACHE) {
    return ProfileOpenCL_Impl(submitFunc, queue, warmupIterations,
                              benchmarkIterations);
  }

  const L3CacheSaturator cacheSaturator = CreateL3CacheSaturator(queue);
  ProfileResult clearCacheStats;
  const auto saturateL3Cache = [&]() {
    ASSERT_OCL_SUCCESS(EnqueueSaturateL3Cache(queue, cacheSaturator.kernel,
                                              cacheSaturator.buffer,
                                              cacheSaturator.bufferSizeBytes));
  };

  clearCacheStats = ProfileOpenCL_Impl(saturateL3Cache, queue, warmupIterations,
                                       benchmarkIterations);

  const auto profileFunc = [&]() {
    saturateL3Cache();
    submitFunc();
  };

  ocltest::ProfileResult finalStats = ProfileOpenCL_Impl(
      profileFunc, queue, warmupIterations, benchmarkIterations);

  finalStats.averageUs -= clearCacheStats.averageUs;
  ReleaseL3CacheSaturator(cacheSaturator);

  return finalStats;
}

}  // namespace ocltest