#include "../../common/oclTestFixture.h"
#include "../../common/utils.h"

namespace {

const std::string kernelSourcePath = OPENCL_KERNEL_SOURCE_PATH;

std::vector<float> computeGemvReference(const std::vector<float>& matrix,
                                        const std::vector<float>& vector,
                                        size_t rowCount, size_t columnCount) {
  std::vector<float> reference(rowCount, 0.0f);
  for (size_t row = 0; row < rowCount; ++row) {
    float accumulator = 0.0f;
    const size_t rowOffset = row * columnCount;
    for (size_t column = 0; column < columnCount; ++column) {
      accumulator += matrix[rowOffset + column] * vector[column];
    }
    reference[row] = accumulator;
  }
  return reference;
}

class PreloadingTest : public ocltest::OclTestFixture {
 public:
  void SetUp() override {
    ocltest::OclTestFixture::SetUp();
    _oclBinary = createProgramAndKernel(kernelSourcePath, "gemv");
  }

  void TearDown() override {
    releaseOCLBinary(_oclBinary);
    ocltest::OclTestFixture::TearDown();
  }

  cl_kernel kernel() const { return _oclBinary.kernel; }

 private:
  OCLBinary _oclBinary;
};

TEST_F(PreloadingTest, GemvKernelProducesReferenceResults) {
  cl_int status = CL_SUCCESS;
  constexpr size_t rowCount = 64;
  constexpr size_t columnCount = 128;

  std::vector<float> matrix =
      utils::createRandomBuffer(rowCount * columnCount, 0);
  std::vector<float> vector = utils::createRandomBuffer(columnCount, 1);
  std::vector<float> result(rowCount, 0.0f);

  cl_mem matrixBuffer =
      clCreateBuffer(context(), CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                     matrix.size() * sizeof(float), matrix.data(), &status);
  ASSERT_OCL_SUCCESS(status);
  cl_mem vectorBuffer =
      clCreateBuffer(context(), CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                     vector.size() * sizeof(float), vector.data(), &status);
  ASSERT_OCL_SUCCESS(status);
  cl_mem resultBuffer =
      clCreateBuffer(context(), CL_MEM_WRITE_ONLY,
                     result.size() * sizeof(float), nullptr, &status);
  ASSERT_OCL_SUCCESS(status);

  const cl_uint clRowCount = static_cast<cl_uint>(rowCount);
  const cl_uint clColumnCount = static_cast<cl_uint>(columnCount);
  ASSERT_OCL_SUCCESS(
      clSetKernelArg(kernel(), 0, sizeof(cl_mem), &matrixBuffer));
  ASSERT_OCL_SUCCESS(
      clSetKernelArg(kernel(), 1, sizeof(cl_mem), &vectorBuffer));
  ASSERT_OCL_SUCCESS(
      clSetKernelArg(kernel(), 2, sizeof(cl_mem), &resultBuffer));
  ASSERT_OCL_SUCCESS(
      clSetKernelArg(kernel(), 3, sizeof(clRowCount), &clRowCount));
  ASSERT_OCL_SUCCESS(
      clSetKernelArg(kernel(), 4, sizeof(clColumnCount), &clColumnCount));

  const size_t globalWorkSize = rowCount;
  ASSERT_OCL_SUCCESS(clEnqueueNDRangeKernel(queue(), kernel(), 1, nullptr,
                                            &globalWorkSize, nullptr, 0,
                                            nullptr, nullptr));
  ASSERT_OCL_SUCCESS(clEnqueueReadBuffer(queue(), resultBuffer, CL_TRUE, 0,
                                         result.size() * sizeof(float),
                                         result.data(), 0, nullptr, nullptr));

  const std::vector<float> reference =
      computeGemvReference(matrix, vector, rowCount, columnCount);
  constexpr float tolerance = 1e-4f;

  for (size_t row = 0; row < rowCount; ++row) {
    ASSERT_NEAR(result[row], reference[row], tolerance)
        << "GEMV result mismatch at row " << row;
  }

  clReleaseMemObject(resultBuffer);
  clReleaseMemObject(vectorBuffer);
  clReleaseMemObject(matrixBuffer);

  std::cout << "GEMV OpenCL kernel executed successfully.\n";
}

}  // namespace