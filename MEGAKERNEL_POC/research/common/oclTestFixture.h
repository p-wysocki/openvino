#include <CL/cl.h>
#include <gtest/gtest.h>

namespace ocltest {

// Prints OpenCL error status as a string.
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
    default:
      return "UNKNOWN_OPENCL_STATUS";
  }
}

// Gtest OCL assert.
#define ASSERT_OCL_SUCCESS(status)                                         \
  EXPECT_EQ(status, CL_SUCCESS)                                            \
      << " error: " << ocltest::printOclErrorStr(status) << " (" << status \
      << ")"

// Main OCL test fixture class.
class OclTestFixture : public testing::Test {
 protected:
  struct OCLBinary {
    cl_program program = nullptr;
    cl_kernel kernel = nullptr;
  };

  // GTest common overrides.
  void SetUp() override;
  void TearDown() override;
  // --

  // Compiles kernel from source file and creates kernel object.
  OCLBinary createProgramAndKernel(const std::string& sourcePath,
                                   const std::string& kernelName,
                                   const std::string& buildOptions = "") const;

  // Releases kernel and program objects.
  void releaseOCLBinary(OCLBinary binary);

  // Returns the OpenCL context.
  cl_context context() const { return oclContext; }

  // Returns the OpenCL command queue.
  cl_command_queue queue() const { return commandQueue; }

  // Returns the selected OpenCL device.
  cl_device_id deviceId() const { return device; }

 private:
  cl_device_id device = nullptr;
  cl_context oclContext = nullptr;
  cl_command_queue commandQueue = nullptr;
};
}  // namespace ocltest