#pragma once

#include <CL/cl.h>
#include <gtest/gtest.h>

namespace ocltest {

// Prints OpenCL error status as a string.
const char* printOclErrorStr(cl_int status);

// Define true to print error and exit on first OCL failure
#define EXIT_ON_FIRST_OCL_ERROR false  

// Gtest OCL assert.
#define ASSERT_OCL_SUCCESS(status)                                           \
  {                                                                          \
    EXPECT_EQ(status, CL_SUCCESS)                                            \
        << " error: " << ocltest::printOclErrorStr(status) << " (" << status \
        << ")";                                                              \
    if (status != CL_SUCCESS && EXIT_ON_FIRST_OCL_ERROR)                     \
      std::exit(EXIT_FAILURE);                                               \
  }

////////////////////////////////////////////////////////////////////
//
// INLINES:
//
////////////////////////////////////////////////////////////////////

////////////////////////////////////////////////////////////////////
inline const char* printOclErrorStr(cl_int status) {
  switch (status) {
    case CL_SUCCESS:
      return "CL_SUCCESS";
    case CL_DEVICE_NOT_FOUND:
      return "CL_DEVICE_NOT_FOUND";
    case CL_DEVICE_NOT_AVAILABLE:
      return "CL_DEVICE_NOT_AVAILABLE";
    case CL_COMPILER_NOT_AVAILABLE:
      return "CL_COMPILER_NOT_AVAILABLE";
    case CL_MEM_OBJECT_ALLOCATION_FAILURE:
      return "CL_MEM_OBJECT_ALLOCATION_FAILURE";
    case CL_OUT_OF_RESOURCES:
      return "CL_OUT_OF_RESOURCES";
    case CL_OUT_OF_HOST_MEMORY:
      return "CL_OUT_OF_HOST_MEMORY";
    case CL_BUILD_PROGRAM_FAILURE:
      return "CL_BUILD_PROGRAM_FAILURE";
    case CL_INVALID_VALUE:
      return "CL_INVALID_VALUE";
    case CL_INVALID_DEVICE:
      return "CL_INVALID_DEVICE";
    case CL_INVALID_BINARY:
      return "CL_INVALID_BINARY";
    case CL_INVALID_BUILD_OPTIONS:
      return "CL_INVALID_BUILD_OPTIONS";
    case CL_INVALID_PROGRAM:
      return "CL_INVALID_PROGRAM";
    case CL_INVALID_PROGRAM_EXECUTABLE:
      return "CL_INVALID_PROGRAM_EXECUTABLE";
    case CL_INVALID_KERNEL_NAME:
      return "CL_INVALID_KERNEL_NAME";
    case CL_INVALID_KERNEL:
      return "CL_INVALID_KERNEL";
    case CL_INVALID_KERNEL_ARGS:
      return "CL_INVALID_KERNEL_ARGS";
    case CL_INVALID_WORK_DIMENSION:
      return "CL_INVALID_WORK_DIMENSION";
    case CL_INVALID_WORK_GROUP_SIZE:
      return "CL_INVALID_WORK_GROUP_SIZE";
    case CL_INVALID_COMMAND_QUEUE:
      return "CL_INVALID_COMMAND_QUEUE";
    default:
      return "UNKNOWN_OPENCL_STATUS - CHECK printOclErrorStr if handled "
             "properly!";
  }
}
}  // namespace ocltest