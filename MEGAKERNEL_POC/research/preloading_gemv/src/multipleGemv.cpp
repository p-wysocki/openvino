#include <CL/cl_half.h>

#include <array>
#include <cmath>
#include <dnnl.hpp>
#include <dnnl_ocl.hpp>
#include <unordered_map>

#include "../../common/benchmarkOpenCLpipeline.h"
#include "../../common/oclTestFixture.h"
#include "../../common/utils.h"

namespace {

static const std::string kernelSourcePath = OPENCL_KERNEL_SOURCE_PATH;
static constexpr size_t warmupIterations = 100;
static constexpr size_t benchmarkIterations = 1000;
static constexpr float ABS_ERROR = 2e-2f;
static constexpr size_t WG_SIZE = 512;
static constexpr const char* kernelSourceFileName = "gemv_opt.cl";

struct GemvShape {
  size_t rowCount;
  size_t columnCount;
  size_t rowsPerBlock;
};

static constexpr std::array<GemvShape, 3> gemvShapes = {
    GemvShape{1024, 2048, 32},
    GemvShape{3072, 1024, 20},
    GemvShape{1024, 3072, 32},
};

static_assert(WG_SIZE % 32 == 0,
              "WG_SIZE must contain whole 32-thread subgroups");

struct BenchmarkResult {
  ocltest::ProfileResult profileResult;
  std::vector<float> output;
};

std::vector<cl_half> convertToHalf(const std::vector<float>& input) {
  std::vector<cl_half> output(input.size());
  for (size_t index = 0; index < input.size(); ++index) {
    output[index] = cl_half_from_float(input[index], CL_HALF_RTE);
  }
  return output;
}

std::vector<float> convertToFloat(const std::vector<cl_half>& input) {
  std::vector<float> output(input.size());
  for (size_t index = 0; index < input.size(); ++index) {
    output[index] = cl_half_to_float(input[index]);
  }
  return output;
}

cl_int enqueueGemvKernel(cl_mem inputBuffer, cl_mem matrixBuffer,
                         cl_mem outputBuffer, const GemvShape& shape,
                         cl_kernel kernel, cl_command_queue queue) {
  ASSERT_OCL_SUCCESS(clSetKernelArg(kernel, 0, sizeof(cl_mem), &matrixBuffer));
  ASSERT_OCL_SUCCESS(clSetKernelArg(kernel, 1, sizeof(cl_mem), &inputBuffer));
  ASSERT_OCL_SUCCESS(clSetKernelArg(kernel, 2, sizeof(cl_mem), &outputBuffer));

  const size_t localWorkSize = WG_SIZE;
  const size_t workGroupCount =
      (shape.rowCount + shape.rowsPerBlock - 1) / shape.rowsPerBlock;
  const size_t globalWorkSize = workGroupCount * localWorkSize;
  return clEnqueueNDRangeKernel(queue, kernel, 1, nullptr, &globalWorkSize,
                                &localWorkSize, 0, nullptr, nullptr);
}

BenchmarkResult benchmarkDnnlGemvChain(
    const std::array<std::vector<float>, 3>& matrices,
    const std::vector<float>& input, cl_device_id device, cl_context context,
    cl_command_queue queue) {
  dnnl::engine engine = dnnl::ocl_interop::make_engine(device, context);
  dnnl::stream stream = dnnl::ocl_interop::make_stream(engine, queue);

  std::array<std::vector<cl_half>, 3> matricesHalf;
  for (size_t layer = 0; layer < gemvShapes.size(); ++layer) {
    matricesHalf[layer] = convertToHalf(matrices[layer]);
  }
  const std::vector<cl_half> inputHalf = convertToHalf(input);

  cl_int status = CL_SUCCESS;
  cl_mem inputBuffer =
      clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                     inputHalf.size() * sizeof(cl_half),
                     const_cast<cl_half*>(inputHalf.data()), &status);
  ASSERT_OCL_SUCCESS(status);

  std::array<cl_mem, 3> matrixBuffers{};
  std::array<cl_mem, 3> outputBuffers{};
  std::array<dnnl::memory, 3> inputMemories;
  std::array<dnnl::memory, 3> matrixMemories;
  std::array<dnnl::memory, 3> outputMemories;
  std::array<dnnl::matmul, 3> gemvs;

  dnnl::primitive_attr attr;
  attr.set_accumulation_mode(dnnl::accumulation_mode::f32);

  for (size_t layer = 0; layer < gemvShapes.size(); ++layer) {
    const GemvShape& shape = gemvShapes[layer];
    const auto inputDesc = dnnl::memory::desc(
        {1, 1, static_cast<dnnl::memory::dim>(shape.columnCount)},
        dnnl::memory::data_type::f16, dnnl::memory::format_tag::abc);
    const auto matrixDesc = dnnl::memory::desc(
        {1, static_cast<dnnl::memory::dim>(shape.columnCount),
         static_cast<dnnl::memory::dim>(shape.rowCount)},
        dnnl::memory::data_type::f16, dnnl::memory::format_tag::acb);
    const auto outputDesc = dnnl::memory::desc(
        {1, 1, static_cast<dnnl::memory::dim>(shape.rowCount)},
        dnnl::memory::data_type::f16, dnnl::memory::format_tag::abc);

    matrixBuffers[layer] =
        clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                       matricesHalf[layer].size() * sizeof(cl_half),
                       matricesHalf[layer].data(), &status);
    ASSERT_OCL_SUCCESS(status);
    outputBuffers[layer] =
        clCreateBuffer(context, CL_MEM_READ_WRITE,
                       shape.rowCount * sizeof(cl_half), nullptr, &status);
    ASSERT_OCL_SUCCESS(status);

    const cl_mem layerInput =
        layer == 0 ? inputBuffer : outputBuffers[layer - 1];
    inputMemories[layer] =
        dnnl::ocl_interop::make_memory(inputDesc, engine, layerInput);
    matrixMemories[layer] = dnnl::ocl_interop::make_memory(
        matrixDesc, engine, matrixBuffers[layer]);
    outputMemories[layer] = dnnl::ocl_interop::make_memory(
        outputDesc, engine, outputBuffers[layer]);
    gemvs[layer] = dnnl::matmul(dnnl::matmul::primitive_desc(
        engine, inputDesc, matrixDesc, outputDesc, attr));
  }

  const ocltest::ProfileResult profileResult = ocltest::ProfileOpenCL(
      [&]() {
        for (size_t layer = 0; layer < gemvShapes.size(); ++layer) {
          const std::unordered_map<int, dnnl::memory> arguments = {
              {DNNL_ARG_SRC, inputMemories[layer]},
              {DNNL_ARG_WEIGHTS, matrixMemories[layer]},
              {DNNL_ARG_DST, outputMemories[layer]}};
          gemvs[layer].execute(stream, arguments);
        }
      },
      queue, warmupIterations, benchmarkIterations);

  std::vector<cl_half> outputHalf(gemvShapes.back().rowCount);
  ASSERT_OCL_SUCCESS(clEnqueueReadBuffer(queue, outputBuffers.back(), CL_TRUE,
                                         0, outputHalf.size() * sizeof(cl_half),
                                         outputHalf.data(), 0, nullptr,
                                         nullptr));

  for (size_t layer = 0; layer < gemvShapes.size(); ++layer) {
    ASSERT_OCL_SUCCESS(clReleaseMemObject(outputBuffers[layer]));
    ASSERT_OCL_SUCCESS(clReleaseMemObject(matrixBuffers[layer]));
  }
  ASSERT_OCL_SUCCESS(clReleaseMemObject(inputBuffer));

  return {profileResult, convertToFloat(outputHalf)};
}

class MultipleGemvTest : public ocltest::OclTestFixture {
 public:
  BenchmarkResult benchmarkOpenClGemvChain(
      const std::array<std::vector<float>, 3>& matrices,
      const std::vector<float>& input) {
    std::array<ocltest::OclTestFixture::OCLBinary, 3> oclBinaries;
    std::array<std::vector<cl_half>, 3> matricesHalf;

    for (size_t layer = 0; layer < gemvShapes.size(); ++layer) {
      const GemvShape& shape = gemvShapes[layer];
      oclBinaries[layer] = createProgramAndKernel(
          kernelSourcePath + kernelSourceFileName, "gemv",
          "-cl-std=CL3.0 -I " + kernelSourcePath +
              " -DMATRIX_ROWS=" + std::to_string(shape.rowCount) +
              " -DMATRIX_COLUMNS=" + std::to_string(shape.columnCount) +
              " -DBLOCK_TILE_ROWS=" + std::to_string(shape.rowsPerBlock));
      matricesHalf[layer] = convertToHalf(matrices[layer]);
    }

    const std::vector<cl_half> inputHalf = convertToHalf(input);
    cl_int status = CL_SUCCESS;
    cl_mem inputBuffer =
        clCreateBuffer(context(), CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                       inputHalf.size() * sizeof(cl_half),
                       const_cast<cl_half*>(inputHalf.data()), &status);
    ASSERT_OCL_SUCCESS(status);

    std::array<cl_mem, 3> matrixBuffers{};
    std::array<cl_mem, 3> outputBuffers{};
    for (size_t layer = 0; layer < gemvShapes.size(); ++layer) {
      matrixBuffers[layer] =
          clCreateBuffer(context(), CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                         matricesHalf[layer].size() * sizeof(cl_half),
                         matricesHalf[layer].data(), &status);
      ASSERT_OCL_SUCCESS(status);
      outputBuffers[layer] = clCreateBuffer(
          context(), CL_MEM_READ_WRITE,
          gemvShapes[layer].rowCount * sizeof(cl_half), nullptr, &status);
      ASSERT_OCL_SUCCESS(status);
    }

    const ocltest::ProfileResult profileResult = ocltest::ProfileOpenCL(
        [&]() {
          for (size_t layer = 0; layer < gemvShapes.size(); ++layer) {
            const cl_mem layerInput =
                layer == 0 ? inputBuffer : outputBuffers[layer - 1];
            ASSERT_OCL_SUCCESS(enqueueGemvKernel(
                layerInput, matrixBuffers[layer], outputBuffers[layer],
                gemvShapes[layer], oclBinaries[layer].kernel, queue()));
          }
        },
        queue(), warmupIterations, benchmarkIterations);

    std::vector<cl_half> outputHalf(gemvShapes.back().rowCount);
    ASSERT_OCL_SUCCESS(
        clEnqueueReadBuffer(queue(), outputBuffers.back(), CL_TRUE, 0,
                            outputHalf.size() * sizeof(cl_half),
                            outputHalf.data(), 0, nullptr, nullptr));

    for (size_t layer = 0; layer < gemvShapes.size(); ++layer) {
      ASSERT_OCL_SUCCESS(clReleaseMemObject(outputBuffers[layer]));
      ASSERT_OCL_SUCCESS(clReleaseMemObject(matrixBuffers[layer]));
      releaseOCLBinary(oclBinaries[layer]);
    }
    ASSERT_OCL_SUCCESS(clReleaseMemObject(inputBuffer));

    return {profileResult, convertToFloat(outputHalf)};
  }
};

TEST_F(MultipleGemvTest, ThreeGemvChain) {
  std::array<std::vector<float>, 3> matrices;
  for (size_t layer = 0; layer < gemvShapes.size(); ++layer) {
    matrices[layer] = utils::createRandomBuffer(
        gemvShapes[layer].rowCount * gemvShapes[layer].columnCount, layer);
    const float scale =
        1.0f / std::sqrt(static_cast<float>(gemvShapes[layer].columnCount));
    for (float& value : matrices[layer]) {
      value *= scale;
    }
  }
  const std::vector<float> input =
      utils::createRandomBuffer(gemvShapes.front().columnCount, 3);

  const BenchmarkResult openClResult =
      benchmarkOpenClGemvChain(matrices, input);
  const BenchmarkResult dnnlResult =
      benchmarkDnnlGemvChain(matrices, input, deviceId(), context(), queue());

  openClResult.profileResult.print("3-GEMV OpenCL chain");
  dnnlResult.profileResult.print("3-GEMV oneDNN chain");
  std::cout << "Speedup: "
            << dnnlResult.profileResult.averageUs /
                   openClResult.profileResult.averageUs
            << "x\n";

  ASSERT_EQ(openClResult.output.size(), dnnlResult.output.size())
      << "Output size mismatch between OpenCL and oneDNN GEMV chains";
  for (size_t index = 0; index < openClResult.output.size(); ++index) {
    ASSERT_NEAR(openClResult.output[index], dnnlResult.output[index], ABS_ERROR)
        << "GEMV chain result mismatch at index " << index;
  }
}

}  // namespace