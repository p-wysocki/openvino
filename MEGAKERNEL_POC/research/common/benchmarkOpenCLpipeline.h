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
template <typename T>
ProfileResult ProfileOpenCL(const T& submitFunc, cl_command_queue queue,
                            size_t warmupIterations,
                            size_t benchmarkIterations);

/////////////////////////////////////////////////////////////////////
//
// IMPLEMENTATION
//
////////////////////////////////////////////////////////////////////

////////////////////////////////////////////////////////////////////
inline void ProfileResult::print(const std::string& label) const {
  std::cout << label << " latency over " << iterations
            << " iterations: avg=" << averageUs << " us\n";
}

////////////////////////////////////////////////////////////////////
template <typename T>
ProfileResult ProfileOpenCL(const T& submitFunc, cl_command_queue queue,
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

}  // namespace ocltest