#include <algorithm>
#include <dnnl.hpp>
#include <dnnl_ocl.hpp>
#include <numeric>
#include <stdexcept>
#include <unordered_map>

#include "../../common/oclTestFixture.h"
#include "../../common/utils.h"

namespace {

static const std::string kernelSourcePath = OPENCL_KERNEL_SOURCE_PATH;
static constexpr size_t warmupIterations = 100;
static constexpr size_t benchmarkIterations = 1000;
static constexpr size_t rowCount = 2048;
static constexpr size_t columnCount = 1024;
static constexpr size_t WG_SIZE = 32;
static constexpr size_t ROWS_PER_GROUP = 1;

static_assert(WG_SIZE % ROWS_PER_GROUP == 0,
              "WG_SIZE must be divisible by ROWS_PER_GROUP");

struct BenchmarkResult {
  double averageUs = 0.0;
  double minUs = 0.0;
  double maxUs = 0.0;
  size_t iterations = 0;
  std::vector<float> result;

  void print(const std::string& label) const {
    std::cout << label << " latency over " << iterations
              << " iterations: avg=" << averageUs << " us, min=" << minUs
              << " us, max=" << maxUs << " us\n";
  }
};

BenchmarkResult calculateLatencyStats(const std::vector<double>& latenciesUs,
                                      const std::vector<float>& result) {
  const auto minMax =
      std::minmax_element(latenciesUs.begin(), latenciesUs.end());
  BenchmarkResult stats;
  stats.averageUs =
      std::accumulate(latenciesUs.begin(), latenciesUs.end(), 0.0) /
      static_cast<double>(latenciesUs.size());
  stats.minUs = *minMax.first;
  stats.maxUs = *minMax.second;
  stats.iterations = latenciesUs.size();
  stats.result = result;
  return stats;
}

BenchmarkResult benchmarkGemvKernelLatency(
    cl_kernel kernel, const std::vector<float>& matrix,
    const std::vector<float>& vector, size_t rowCount, size_t columnCount,
    cl_device_id device, cl_context context, cl_command_queue queue,
    size_t warmupIterations, size_t benchmarkIterations) {
  std::vector<float> result(rowCount, 0.0f);

  cl_int status = CL_SUCCESS;
  cl_mem matrixBuffer = clCreateBuffer(
      context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
      matrix.size() * sizeof(float), (void*)matrix.data(), &status);
  ASSERT_OCL_SUCCESS(status);
  cl_mem vectorBuffer = clCreateBuffer(
      context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
      vector.size() * sizeof(float), (void*)vector.data(), &status);
  ASSERT_OCL_SUCCESS(status);
  cl_mem resultBuffer =
      clCreateBuffer(context, CL_MEM_WRITE_ONLY, result.size() * sizeof(float),
                     nullptr, &status);
  ASSERT_OCL_SUCCESS(status);

  const cl_uint clRowCount = static_cast<cl_uint>(rowCount);
  const cl_uint clColumnCount = static_cast<cl_uint>(columnCount);
  ASSERT_OCL_SUCCESS(clSetKernelArg(kernel, 0, sizeof(cl_mem), &matrixBuffer));
  ASSERT_OCL_SUCCESS(clSetKernelArg(kernel, 1, sizeof(cl_mem), &vectorBuffer));
  ASSERT_OCL_SUCCESS(clSetKernelArg(kernel, 2, sizeof(cl_mem), &resultBuffer));
  ASSERT_OCL_SUCCESS(
      clSetKernelArg(kernel, 3, sizeof(clRowCount), &clRowCount));
  ASSERT_OCL_SUCCESS(
      clSetKernelArg(kernel, 4, sizeof(clColumnCount), &clColumnCount));
  // Arg 5: local reduction scratch buffer – one float per work-item.
  ASSERT_OCL_SUCCESS(
      clSetKernelArg(kernel, 5, WG_SIZE * sizeof(float), nullptr));

  const size_t localWorkSize = WG_SIZE;
  const size_t workGroupCount =
      (rowCount + ROWS_PER_GROUP - 1) / ROWS_PER_GROUP;
  const size_t globalWorkSize = workGroupCount * WG_SIZE;
  for (size_t iteration = 0; iteration < warmupIterations; ++iteration) {
    ASSERT_OCL_SUCCESS(clEnqueueNDRangeKernel(queue, kernel, 1, nullptr,
                                              &globalWorkSize, &localWorkSize,
                                              0, nullptr, nullptr));
  }
  ASSERT_OCL_SUCCESS(clFinish(queue));

  std::vector<double> latenciesUs;
  latenciesUs.reserve(benchmarkIterations);

  for (size_t iteration = 0; iteration < benchmarkIterations; ++iteration) {
    cl_event event = nullptr;
    ASSERT_OCL_SUCCESS(clEnqueueNDRangeKernel(queue, kernel, 1, nullptr,
                                              &globalWorkSize, &localWorkSize,
                                              0, nullptr, &event));
    ASSERT_OCL_SUCCESS(clWaitForEvents(1, &event));

    cl_ulong startNs = 0;
    cl_ulong endNs = 0;
    ASSERT_OCL_SUCCESS(clGetEventProfilingInfo(
        event, CL_PROFILING_COMMAND_START, sizeof(startNs), &startNs, nullptr));
    ASSERT_OCL_SUCCESS(clGetEventProfilingInfo(event, CL_PROFILING_COMMAND_END,
                                               sizeof(endNs), &endNs, nullptr));
    ASSERT_OCL_SUCCESS(clReleaseEvent(event));

    latenciesUs.push_back(static_cast<double>(endNs - startNs) / 1000.0);
  }

  ASSERT_OCL_SUCCESS(clEnqueueReadBuffer(queue, resultBuffer, CL_TRUE, 0,
                                         result.size() * sizeof(float),
                                         result.data(), 0, nullptr, nullptr));

  ASSERT_OCL_SUCCESS(clReleaseMemObject(resultBuffer));
  ASSERT_OCL_SUCCESS(clReleaseMemObject(vectorBuffer));
  ASSERT_OCL_SUCCESS(clReleaseMemObject(matrixBuffer));

  return calculateLatencyStats(latenciesUs, result);
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
      vectorDims, dnnl::memory::data_type::f32, dnnl::memory::format_tag::ab);
  const auto matrixDesc = dnnl::memory::desc(
      matrixDims, dnnl::memory::data_type::f32, dnnl::memory::format_tag::ba);
  const auto resultDesc = dnnl::memory::desc(
      resultDims, dnnl::memory::data_type::f32, dnnl::memory::format_tag::ab);

  cl_int status = CL_SUCCESS;
  cl_mem vectorBuffer =
      clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                     vector.size() * sizeof(float),
                     const_cast<float*>(vector.data()), &status);
  ASSERT_OCL_SUCCESS(status);
  cl_mem matrixBuffer =
      clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                     matrix.size() * sizeof(float),
                     const_cast<float*>(matrix.data()), &status);
  ASSERT_OCL_SUCCESS(status);
  cl_mem resultBuffer = clCreateBuffer(
      context, CL_MEM_WRITE_ONLY, rowCount * sizeof(float), nullptr, &status);
  ASSERT_OCL_SUCCESS(status);

  auto vectorMemory =
      dnnl::ocl_interop::make_memory(vectorDesc, engine, vectorBuffer);
  auto matrixMemory =
      dnnl::ocl_interop::make_memory(matrixDesc, engine, matrixBuffer);
  auto resultMemory =
      dnnl::ocl_interop::make_memory(resultDesc, engine, resultBuffer);

  const auto gemv = dnnl::matmul(
      dnnl::matmul::primitive_desc(engine, vectorDesc, matrixDesc, resultDesc));

  const std::unordered_map<int, dnnl::memory> args = {
      {DNNL_ARG_SRC, vectorMemory},
      {DNNL_ARG_WEIGHTS, matrixMemory},
      {DNNL_ARG_DST, resultMemory}};

  for (size_t iteration = 0; iteration < warmupIterations; ++iteration) {
    gemv.execute(stream, args);
  }
  stream.wait();

  std::vector<double> latenciesUs;
  latenciesUs.reserve(benchmarkIterations);

  for (size_t iteration = 0; iteration < benchmarkIterations; ++iteration) {
    cl_event startEvent = nullptr;
    cl_event endEvent = nullptr;
    ASSERT_OCL_SUCCESS(
        clEnqueueMarkerWithWaitList(queue, 0, nullptr, &startEvent));
    gemv.execute(stream, args);
    ASSERT_OCL_SUCCESS(
        clEnqueueMarkerWithWaitList(queue, 0, nullptr, &endEvent));
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

    latenciesUs.push_back(static_cast<double>(endNs - startNs) / 1000.0);
  }

  std::vector<float> result(rowCount, 0.0f);
  ASSERT_OCL_SUCCESS(clEnqueueReadBuffer(queue, resultBuffer, CL_TRUE, 0,
                                         result.size() * sizeof(float),
                                         result.data(), 0, nullptr, nullptr));

  ASSERT_OCL_SUCCESS(clReleaseMemObject(resultBuffer));
  ASSERT_OCL_SUCCESS(clReleaseMemObject(vectorBuffer));
  ASSERT_OCL_SUCCESS(clReleaseMemObject(matrixBuffer));

  return calculateLatencyStats(latenciesUs, result);
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

  const BenchmarkResult latency = benchmarkGemvKernelLatency(
      kernel(), matrix, vector, rowCount, columnCount, deviceId(), context(),
      queue(), warmupIterations, benchmarkIterations);

  const BenchmarkResult dnnlLatency = benchmarkDnnlGemvLatency(
      matrix, vector, rowCount, columnCount, deviceId(), context(), queue(),
      warmupIterations, benchmarkIterations);

  constexpr float tolerance = 1e-4f;

  for (size_t row = 0; row < rowCount; ++row) {
    ASSERT_NEAR(latency.result[row], dnnlLatency.result[row], tolerance)
        << "GEMV result mismatch at row " << row;
  }

  latency.print("GEMV OpenCL kernel");
  dnnlLatency.print("GEMV oneDNN kernel");
}

}  // namespace