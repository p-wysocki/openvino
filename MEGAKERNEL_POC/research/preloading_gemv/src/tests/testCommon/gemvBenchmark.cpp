#include "gemvBenchmark.h"

#include <CL/cl_half.h>

#include <dnnl.hpp>
#include <dnnl_ocl.hpp>
#include <unordered_map>

namespace ocltest {
namespace {

constexpr size_t workGroupSize = 512;
static_assert(workGroupSize % 32 == 0,
              "workGroupSize must contain whole 32-thread subgroups");

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

  const size_t workGroupCount =
      (shape.rowCount + shape.rowsPerBlock - 1) / shape.rowsPerBlock;
  const size_t globalWorkSize = workGroupCount * workGroupSize;
  return clEnqueueNDRangeKernel(queue, kernel, 1, nullptr, &globalWorkSize,
                                &workGroupSize, 0, nullptr, nullptr);
}

}  // namespace

GemvBenchmarkResult GemvTestFixture::benchmarkOpenClGemvChain(
    const std::vector<std::vector<float>>& matrices,
    const std::vector<float>& input, const std::vector<GemvShape>& shapes,
    const std::string& kernelSourcePath,
    const std::string& kernelSourceFileName, size_t warmupIterations,
    size_t benchmarkIterations) {
  std::vector<OCLBinary> oclBinaries(shapes.size());
  std::vector<std::vector<cl_half>> matricesHalf(shapes.size());

  for (size_t layer = 0; layer < shapes.size(); ++layer) {
    const GemvShape& shape = shapes[layer];
    oclBinaries[layer] = createProgramAndKernel(
        kernelSourcePath + kernelSourceFileName, "gemvOptKernel",
        "-cl-std=CL3.0 -I " + kernelSourcePath + " -I " + std::string(OPENCL_KERNEL_SOURCE_PATH) +
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

  std::vector<cl_mem> matrixBuffers(shapes.size());
  std::vector<cl_mem> outputBuffers(shapes.size());
  for (size_t layer = 0; layer < shapes.size(); ++layer) {
    matrixBuffers[layer] =
        clCreateBuffer(context(), CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                       matricesHalf[layer].size() * sizeof(cl_half),
                       matricesHalf[layer].data(), &status);
    ASSERT_OCL_SUCCESS(status);
    outputBuffers[layer] = clCreateBuffer(
        context(), CL_MEM_READ_WRITE, shapes[layer].rowCount * sizeof(cl_half),
        nullptr, &status);
    ASSERT_OCL_SUCCESS(status);
  }

  const ProfileResult profileResult =
      ProfileOpenCL<CLEAR_CACHE_BEFORE_BENCHMARK>(
          [&]() {
            for (size_t layer = 0; layer < shapes.size(); ++layer) {
              const cl_mem layerInput =
                  layer == 0 ? inputBuffer : outputBuffers[layer - 1];
              ASSERT_OCL_SUCCESS(enqueueGemvKernel(
                  layerInput, matrixBuffers[layer], outputBuffers[layer],
                  shapes[layer], oclBinaries[layer].kernel, queue()));
            }
          },
          queue(), warmupIterations, benchmarkIterations);

  std::vector<cl_half> outputHalf(shapes.back().rowCount);
  ASSERT_OCL_SUCCESS(clEnqueueReadBuffer(queue(), outputBuffers.back(), CL_TRUE,
                                         0, outputHalf.size() * sizeof(cl_half),
                                         outputHalf.data(), 0, nullptr,
                                         nullptr));

  for (size_t layer = 0; layer < shapes.size(); ++layer) {
    ASSERT_OCL_SUCCESS(clReleaseMemObject(outputBuffers[layer]));
    ASSERT_OCL_SUCCESS(clReleaseMemObject(matrixBuffers[layer]));
    releaseOCLBinary(oclBinaries[layer]);
  }
  ASSERT_OCL_SUCCESS(clReleaseMemObject(inputBuffer));

  return {profileResult, convertToFloat(outputHalf)};
}

GemvBenchmarkResult GemvTestFixture::benchmarkDnnlGemvChain(
    const std::vector<std::vector<float>>& matrices,
    const std::vector<float>& input, const std::vector<GemvShape>& shapes,
    size_t warmupIterations, size_t benchmarkIterations) {
  dnnl::engine engine = dnnl::ocl_interop::make_engine(deviceId(), context());
  dnnl::stream stream = dnnl::ocl_interop::make_stream(engine, queue());

  std::vector<std::vector<cl_half>> matricesHalf(shapes.size());
  for (size_t layer = 0; layer < shapes.size(); ++layer) {
    matricesHalf[layer] = convertToHalf(matrices[layer]);
  }
  const std::vector<cl_half> inputHalf = convertToHalf(input);

  cl_int status = CL_SUCCESS;
  cl_mem inputBuffer =
      clCreateBuffer(context(), CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                     inputHalf.size() * sizeof(cl_half),
                     const_cast<cl_half*>(inputHalf.data()), &status);
  ASSERT_OCL_SUCCESS(status);

  std::vector<cl_mem> matrixBuffers(shapes.size());
  std::vector<cl_mem> outputBuffers(shapes.size());
  std::vector<dnnl::memory> inputMemories(shapes.size());
  std::vector<dnnl::memory> matrixMemories(shapes.size());
  std::vector<dnnl::memory> outputMemories(shapes.size());
  std::vector<dnnl::matmul> gemvs(shapes.size());

  dnnl::primitive_attr attr;
  attr.set_accumulation_mode(dnnl::accumulation_mode::f32);

  for (size_t layer = 0; layer < shapes.size(); ++layer) {
    const GemvShape& shape = shapes[layer];
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
        clCreateBuffer(context(), CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                       matricesHalf[layer].size() * sizeof(cl_half),
                       matricesHalf[layer].data(), &status);
    ASSERT_OCL_SUCCESS(status);
    outputBuffers[layer] =
        clCreateBuffer(context(), CL_MEM_READ_WRITE,
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

  const ProfileResult profileResult =
      ProfileOpenCL<CLEAR_CACHE_BEFORE_BENCHMARK>(
          [&]() {
            for (size_t layer = 0; layer < shapes.size(); ++layer) {
              const std::unordered_map<int, dnnl::memory> arguments = {
                  {DNNL_ARG_SRC, inputMemories[layer]},
                  {DNNL_ARG_WEIGHTS, matrixMemories[layer]},
                  {DNNL_ARG_DST, outputMemories[layer]}};
              gemvs[layer].execute(stream, arguments);
            }
          },
          queue(), warmupIterations, benchmarkIterations);

  std::vector<cl_half> outputHalf(shapes.back().rowCount);
  ASSERT_OCL_SUCCESS(clEnqueueReadBuffer(queue(), outputBuffers.back(), CL_TRUE,
                                         0, outputHalf.size() * sizeof(cl_half),
                                         outputHalf.data(), 0, nullptr,
                                         nullptr));

  for (size_t layer = 0; layer < shapes.size(); ++layer) {
    ASSERT_OCL_SUCCESS(clReleaseMemObject(outputBuffers[layer]));
    ASSERT_OCL_SUCCESS(clReleaseMemObject(matrixBuffers[layer]));
  }
  ASSERT_OCL_SUCCESS(clReleaseMemObject(inputBuffer));

  return {profileResult, convertToFloat(outputHalf)};
}

}  // namespace ocltest