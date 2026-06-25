#include <dnnl.hpp>
#include <dnnl_ocl.hpp>
#include <stdexcept>

#include "../../common/oclTestFixture.h"
#include "../../common/utils.h"

namespace {

const std::string kernelSourcePath = OPENCL_KERNEL_SOURCE_PATH;

std::vector<float> computeGemvReference(const std::vector<float>& matrix,
                                        const std::vector<float>& vector,
                                        size_t rowCount, size_t columnCount,
                                        cl_device_id device, cl_context context,
                                        cl_command_queue queue) {
  std::vector<float> reference(rowCount, 0.0f);

  dnnl::engine engine = dnnl::ocl_interop::make_engine(device, context);
  dnnl::stream stream = dnnl::ocl_interop::make_stream(engine, queue);

  const dnnl::memory::dims matrixDims = {
      static_cast<dnnl::memory::dim>(rowCount),
      static_cast<dnnl::memory::dim>(columnCount)};
  const dnnl::memory::dims vectorDims = {
      static_cast<dnnl::memory::dim>(columnCount), 1};
  const dnnl::memory::dims resultDims = {
      static_cast<dnnl::memory::dim>(rowCount), 1};

  const auto matrixDesc = dnnl::memory::desc(
      matrixDims, dnnl::memory::data_type::f32, dnnl::memory::format_tag::ab);
  const auto vectorDesc = dnnl::memory::desc(
      vectorDims, dnnl::memory::data_type::f32, dnnl::memory::format_tag::ab);
  const auto resultDesc = dnnl::memory::desc(
      resultDims, dnnl::memory::data_type::f32, dnnl::memory::format_tag::ab);

  cl_int status = CL_SUCCESS;
  cl_mem matrixBuffer =
      clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                     matrix.size() * sizeof(float),
                     const_cast<float*>(matrix.data()), &status);
  ASSERT_OCL_SUCCESS(status);
  cl_mem vectorBuffer =
      clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                     vector.size() * sizeof(float),
                     const_cast<float*>(vector.data()), &status);
  ASSERT_OCL_SUCCESS(status);
  cl_mem referenceBuffer =
      clCreateBuffer(context, CL_MEM_WRITE_ONLY,
                     reference.size() * sizeof(float), nullptr, &status);
  ASSERT_OCL_SUCCESS(status);

  auto matrixMemory =
      dnnl::ocl_interop::make_memory(matrixDesc, engine, matrixBuffer);
  auto vectorMemory =
      dnnl::ocl_interop::make_memory(vectorDesc, engine, vectorBuffer);
  auto resultMemory =
      dnnl::ocl_interop::make_memory(resultDesc, engine, referenceBuffer);

  const auto gemv = dnnl::matmul(
      dnnl::matmul::primitive_desc(engine, matrixDesc, vectorDesc, resultDesc));
  gemv.execute(stream, {{DNNL_ARG_SRC, matrixMemory},
                        {DNNL_ARG_WEIGHTS, vectorMemory},
                        {DNNL_ARG_DST, resultMemory}});
  stream.wait();

  ASSERT_OCL_SUCCESS(clEnqueueReadBuffer(queue, referenceBuffer, CL_TRUE, 0,
                                    reference.size() * sizeof(float),
                                    reference.data(), 0, nullptr, nullptr));

  ASSERT_OCL_SUCCESS(clReleaseMemObject(referenceBuffer));
  ASSERT_OCL_SUCCESS(clReleaseMemObject(vectorBuffer));
  ASSERT_OCL_SUCCESS(clReleaseMemObject(matrixBuffer));

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

  const std::vector<float> reference = computeGemvReference(
      matrix, vector, rowCount, columnCount, deviceId(), context(), queue());
  constexpr float tolerance = 1e-4f;

  for (size_t row = 0; row < rowCount; ++row) {
    ASSERT_NEAR(result[row], reference[row], tolerance)
        << "GEMV result mismatch at row " << row;
  }

  ASSERT_OCL_SUCCESS(clReleaseMemObject(resultBuffer));
  ASSERT_OCL_SUCCESS(clReleaseMemObject(vectorBuffer));
  ASSERT_OCL_SUCCESS(clReleaseMemObject(matrixBuffer));

  std::cout << "GEMV OpenCL kernel executed successfully.\n";
}

}  // namespace