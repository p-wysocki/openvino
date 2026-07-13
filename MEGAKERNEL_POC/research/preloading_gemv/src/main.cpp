#include <CL/cl_half.h>

#include <cmath>
#include <dnnl.hpp>
#include <dnnl_ocl.hpp>
#include <unordered_map>

#include "../../common/benchmarkOpenCLpipeline.h"
#include "../../common/oclTestFixture.h"
#include "../../common/utils.h"

namespace {

static const std::string kernelSourcePath = OPENCL_KERNEL_SOURCE_PATH;
static constexpr size_t warmupIterations = 1000;
static constexpr size_t benchmarkIterations = 10000;
static constexpr size_t rowCount = 2048;
static constexpr size_t columnCount = 1024;
static constexpr float ABS_ERROR = 1e-3f;

static constexpr size_t WG_SIZE = 128;
static constexpr size_t ROWS_PER_GROUP = 16;

static_assert(WG_SIZE % 32 == 0,
              "WG_SIZE must contain whole 32-thread subgroups");
static_assert(ROWS_PER_GROUP % (WG_SIZE / 32) == 0,
              "Each subgroup must compute the same number of rows");

struct BenchmarkResult {
  ocltest::ProfileResult profileResult;
  std::vector<float> output;
};

std::vector<cl_half> convertToHalf(const std::vector<float>& input) {
  std::vector<cl_half> output(input.size());
  for (size_t i = 0; i < input.size(); ++i) {
    output[i] = cl_half_from_float(input[i], CL_HALF_RTE);
  }
  return output;
}

std::vector<float> convertToFloat(const std::vector<cl_half>& input) {
  std::vector<float> output(input.size());
  for (size_t i = 0; i < input.size(); ++i) {
    output[i] = cl_half_to_float(input[i]);
  }
  return output;
}

cl_int EnqueueGemvKernel(cl_mem vectorBuffer, cl_mem matrixBuffer,
                         cl_mem resultBuffer, size_t rowCount,
                         size_t columnCount, cl_kernel kernel,
                         cl_command_queue queue) {
  const cl_uint clRowCount = static_cast<cl_uint>(rowCount);
  const cl_uint clColumnCount = static_cast<cl_uint>(columnCount);
  ASSERT_OCL_SUCCESS(clSetKernelArg(kernel, 0, sizeof(cl_mem), &matrixBuffer));
  ASSERT_OCL_SUCCESS(clSetKernelArg(kernel, 1, sizeof(cl_mem), &vectorBuffer));
  ASSERT_OCL_SUCCESS(clSetKernelArg(kernel, 2, sizeof(cl_mem), &resultBuffer));
  ASSERT_OCL_SUCCESS(
      clSetKernelArg(kernel, 3, sizeof(clRowCount), &clRowCount));
  ASSERT_OCL_SUCCESS(
      clSetKernelArg(kernel, 4, sizeof(clColumnCount), &clColumnCount));

  const size_t localWorkSize = WG_SIZE;
  const size_t workGroupCount =
      (rowCount + ROWS_PER_GROUP - 1) / ROWS_PER_GROUP;
  const size_t globalWorkSize = workGroupCount * WG_SIZE;
  return clEnqueueNDRangeKernel(queue, kernel, 1, nullptr, &globalWorkSize,
                                &localWorkSize, 0, nullptr, nullptr);
}

BenchmarkResult benchmarkGemvKernelLatency(
    cl_kernel kernel, const std::vector<float>& matrix,
    const std::vector<float>& vector, size_t rowCount, size_t columnCount,
    cl_device_id device, cl_context context, cl_command_queue queue,
    size_t warmupIterations, size_t benchmarkIterations) {
  const std::vector<cl_half> matrixHalf = convertToHalf(matrix);
  const std::vector<cl_half> vectorHalf = convertToHalf(vector);
  std::vector<cl_half> resultHalf(rowCount,
                                  cl_half_from_float(0.0f, CL_HALF_RTE));

  cl_int status = CL_SUCCESS;
  cl_mem matrixBuffer = clCreateBuffer(
      context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
      matrixHalf.size() * sizeof(cl_half), (void*)matrixHalf.data(), &status);
  ASSERT_OCL_SUCCESS(status);
  cl_mem vectorBuffer = clCreateBuffer(
      context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
      vectorHalf.size() * sizeof(cl_half), (void*)vectorHalf.data(), &status);
  ASSERT_OCL_SUCCESS(status);
  cl_mem resultBuffer =
      clCreateBuffer(context, CL_MEM_WRITE_ONLY,
                     resultHalf.size() * sizeof(cl_half), nullptr, &status);
  ASSERT_OCL_SUCCESS(status);

  ocltest::ProfileResult stats = ocltest::ProfileOpenCL(
      [&](void) {
        ASSERT_OCL_SUCCESS(EnqueueGemvKernel(vectorBuffer, matrixBuffer,
                                             resultBuffer, rowCount,
                                             columnCount, kernel, queue));
      },
      queue, warmupIterations, benchmarkIterations);

  ASSERT_OCL_SUCCESS(clEnqueueReadBuffer(
      queue, resultBuffer, CL_TRUE, 0, resultHalf.size() * sizeof(cl_half),
      resultHalf.data(), 0, nullptr, nullptr));

  ASSERT_OCL_SUCCESS(clReleaseMemObject(resultBuffer));
  ASSERT_OCL_SUCCESS(clReleaseMemObject(vectorBuffer));
  ASSERT_OCL_SUCCESS(clReleaseMemObject(matrixBuffer));

  return {stats, convertToFloat(resultHalf)};
}

BenchmarkResult benchmarkDnnlGemvLatency(
    const std::vector<float>& matrix, const std::vector<float>& vector,
    size_t rowCount, size_t columnCount, cl_device_id device,
    cl_context context, cl_command_queue queue, size_t warmupIterations,
    size_t benchmarkIterations) {
  dnnl::engine engine = dnnl::ocl_interop::make_engine(device, context);
  dnnl::stream stream = dnnl::ocl_interop::make_stream(engine, queue);

  const dnnl::memory::dims vectorDims = {
      1, static_cast<dnnl::memory::dim>(columnCount)};
  const dnnl::memory::dims matrixDims = {
      static_cast<dnnl::memory::dim>(columnCount),
      static_cast<dnnl::memory::dim>(rowCount)};
  const dnnl::memory::dims resultDims = {
      1, static_cast<dnnl::memory::dim>(rowCount)};

  const auto vectorDesc = dnnl::memory::desc(
      vectorDims, dnnl::memory::data_type::f16, dnnl::memory::format_tag::ab);
  const auto matrixDesc = dnnl::memory::desc(
      matrixDims, dnnl::memory::data_type::f16, dnnl::memory::format_tag::ba);
  const auto resultDesc = dnnl::memory::desc(
      resultDims, dnnl::memory::data_type::f16, dnnl::memory::format_tag::ab);

  const std::vector<cl_half> matrixHalf = convertToHalf(matrix);
  const std::vector<cl_half> vectorHalf = convertToHalf(vector);

  cl_int status = CL_SUCCESS;
  cl_mem vectorBuffer =
      clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                     vectorHalf.size() * sizeof(cl_half),
                     const_cast<cl_half*>(vectorHalf.data()), &status);
  ASSERT_OCL_SUCCESS(status);
  cl_mem matrixBuffer =
      clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                     matrixHalf.size() * sizeof(cl_half),
                     const_cast<cl_half*>(matrixHalf.data()), &status);
  ASSERT_OCL_SUCCESS(status);
  cl_mem resultBuffer = clCreateBuffer(
      context, CL_MEM_WRITE_ONLY, rowCount * sizeof(cl_half), nullptr, &status);
  ASSERT_OCL_SUCCESS(status);

  auto vectorMemory =
      dnnl::ocl_interop::make_memory(vectorDesc, engine, vectorBuffer);
  auto matrixMemory =
      dnnl::ocl_interop::make_memory(matrixDesc, engine, matrixBuffer);
  auto resultMemory =
      dnnl::ocl_interop::make_memory(resultDesc, engine, resultBuffer);

  dnnl::primitive_attr attr;
  attr.set_accumulation_mode(dnnl::accumulation_mode::f32);
  const auto gemv = dnnl::matmul(dnnl::matmul::primitive_desc(
      engine, vectorDesc, matrixDesc, resultDesc, attr));

  ocltest::ProfileResult stats = ocltest::ProfileOpenCL(
      [&](void) {
        const std::unordered_map<int, dnnl::memory> args = {
            {DNNL_ARG_SRC, vectorMemory},
            {DNNL_ARG_WEIGHTS, matrixMemory},
            {DNNL_ARG_DST, resultMemory}};
        gemv.execute(stream, args);
      },
      queue, warmupIterations, benchmarkIterations);

  std::vector<cl_half> resultHalf(rowCount,
                                  cl_half_from_float(0.0f, CL_HALF_RTE));
  ASSERT_OCL_SUCCESS(clEnqueueReadBuffer(
      queue, resultBuffer, CL_TRUE, 0, resultHalf.size() * sizeof(cl_half),
      resultHalf.data(), 0, nullptr, nullptr));

  ASSERT_OCL_SUCCESS(clReleaseMemObject(resultBuffer));
  ASSERT_OCL_SUCCESS(clReleaseMemObject(vectorBuffer));
  ASSERT_OCL_SUCCESS(clReleaseMemObject(matrixBuffer));

  return {stats, convertToFloat(resultHalf)};
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

  std::vector<float> matrix =
      utils::createRandomBuffer(rowCount * columnCount, 0);
  std::vector<float> vector = utils::createRandomBuffer(columnCount, 1);

  const BenchmarkResult gemvLatency = benchmarkGemvKernelLatency(
      kernel(), matrix, vector, rowCount, columnCount, deviceId(), context(),
      queue(), warmupIterations, benchmarkIterations);

  const BenchmarkResult dnnlLatency = benchmarkDnnlGemvLatency(
      matrix, vector, rowCount, columnCount, deviceId(), context(), queue(),
      warmupIterations, benchmarkIterations);

  gemvLatency.profileResult.print("GEMV OpenCL kernel");
  dnnlLatency.profileResult.print("GEMV oneDNN kernel");

  for (size_t i = 0; i < rowCount; ++i) {
    ASSERT_NEAR(gemvLatency.output[i], dnnlLatency.output[i], ABS_ERROR)
        << "GEMV result mismatch at idx " << i;
  }
}

}  // namespace