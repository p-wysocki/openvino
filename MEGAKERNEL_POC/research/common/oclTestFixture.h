#include "ocltestCommon.h"

namespace ocltest {

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