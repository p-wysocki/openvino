#include <CL/cl_ext.h>
#include <CL/cl_half.h>

#include <algorithm>
#include <cmath>
#include <cstring>
#include <string>
#include <vector>

#include "../../../../../../common/utils.h"
#include "../../../../ocl/taskSystem/host/taskManagerHost.h"
#include "../../testCommon/gemvBenchmark.h"
#include "ocl/tasks/gemv1024x2048Task.h"
#include "ocl/tasks/gemv1024x3072Task.h"
#include "ocl/tasks/gemv3072x1024Task.h"

namespace {

constexpr size_t WORKERS = 80;
constexpr size_t WORK_GROUP_SIZE = 512;

const std::string GEMV_KERNEL_PATH =
    std::string(OPENCL_KERNEL_SOURCE_PATH) + "../tests/gemv/static/ocl/";
const std::string TASK_SYSTEM_GEMV_KERNEL_PATH =
    std::string(OPENCL_KERNEL_SOURCE_PATH) +
    "../tests/gemv/taskSystem/chain/ocl/";

const std::vector<ocltest::GemvParams> GEMV_PARAMS = {
    {1024, 2048, 32},
    {3072, 1024, 32},
    {1024, 3072, 32, 2, 2},
};

std::vector<cl_half> ConvertToHalf(const std::vector<float>& input) {
  std::vector<cl_half> output(input.size());
  for (size_t index = 0; index < input.size(); ++index) {
    output[index] = cl_half_from_float(input[index], CL_HALF_RTE);
  }
  return output;
}

std::vector<float> ConvertToFloat(const std::vector<cl_half>& input) {
  std::vector<float> output(input.size());
  for (size_t index = 0; index < input.size(); ++index) {
    output[index] = cl_half_to_float(input[index]);
  }
  return output;
}

void PrintTaskQueue(const std::vector<TaskDesc>& tasks) {
  // Print tasks for debugging:
  for (size_t taskIndex = 0; taskIndex < tasks.size(); ++taskIndex) {
    const TaskDesc& task = tasks[taskIndex];
    std::cout << "Task " << taskIndex << ": type=" << task.type
              << ", payloadSize=" << sizeof(task.payload) << "\n";
    // Print payload casted to the appropriate GEMV task struct based on the
    // task type
    switch (task.type) {
      case 3: {
        const Gemv1024x2048Task* gemvTask =
            reinterpret_cast<const Gemv1024x2048Task*>(task.payload);
        std::cout << "  Gemv1024x2048Task: tileId=" << gemvTask->tileId
                  << ", wantedInputSyncValue=" << gemvTask->wantedInputSyncValue
                  << "\n";
        break;
      }
      case 4: {
        const Gemv3072x1024Task* gemvTask =
            reinterpret_cast<const Gemv3072x1024Task*>(task.payload);
        std::cout << "  Gemv3072x1024Task: tileId=" << gemvTask->tileId
                  << ", wantedInputSyncValue=" << gemvTask->wantedInputSyncValue
                  << "\n";
        break;
      }
      case 5: {
        const Gemv1024x3072Task* gemvTask =
            reinterpret_cast<const Gemv1024x3072Task*>(task.payload);
        std::cout << "  Gemv1024x3072Task: tileId=" << gemvTask->tileId
                  << ", wantedInputSyncValue=" << gemvTask->wantedInputSyncValue
                  << "\n";
        break;
      }
      default:
        std::cerr << "Unknown task type: " << task.type << "\n";
        break;
    }
  }
}

template <typename GemvTask>
TaskDesc CreateGemvTaskDesc(int type, const cl_half* matrix,
                            const cl_half* vector, cl_half* output,
                            int* inputSemaphore, int* outputSemaphore,
                            int wantedInputSyncValue, int tileId) {
  TaskDesc taskDesc{};
  taskDesc.type = type;
  const GemvTask task = {matrix,         vector,          output,
                         inputSemaphore, outputSemaphore, wantedInputSyncValue,
                         tileId};
  static_assert(sizeof(task) <= PAYLOAD_SIZE,
                "GEMV task size exceeds task payload size");
  std::memcpy(taskDesc.payload, &task, sizeof(task));
  return taskDesc;
}

class ChainTaskSystemGemvTest : public ocltest::GemvTestFixture {};

TEST_F(ChainTaskSystemGemvTest, ThreeGemvChain) {
  std::vector<std::vector<float>> matrices(GEMV_PARAMS.size());
  std::vector<std::vector<cl_half>> matricesHalf(GEMV_PARAMS.size());
  for (size_t layer = 0; layer < GEMV_PARAMS.size(); ++layer) {
    matrices[layer] = utils::createRandomBuffer(
        GEMV_PARAMS[layer].rowCount * GEMV_PARAMS[layer].columnCount, layer);
    const float scale =
        1.0f / std::sqrt(static_cast<float>(GEMV_PARAMS[layer].columnCount));
    for (float& value : matrices[layer]) {
      value *= scale;
    }
    matricesHalf[layer] = ConvertToHalf(matrices[layer]);
  }
  const std::vector<float> input =
      utils::createRandomBuffer(GEMV_PARAMS.front().columnCount, 3);
  const std::vector<cl_half> inputHalf = ConvertToHalf(input);

  const OCLBinary binary = createProgramAndKernel(
      TASK_SYSTEM_GEMV_KERNEL_PATH + "chainTaskSystemGemvKernel.cl",
      "chainTaskSystemGemvKernel",
      "-I " + std::string(OPENCL_KERNEL_SOURCE_PATH) + " -I " +
          TASK_SYSTEM_GEMV_KERNEL_PATH +
          " -igc_opts 'VISAOptions=-hybridRAWithSpill -fastCompileRA'");

  cl_int status = CL_SUCCESS;
  cl_platform_id platform = nullptr;
  ASSERT_OCL_SUCCESS(clGetDeviceInfo(deviceId(), CL_DEVICE_PLATFORM,
                                     sizeof(platform), &platform, nullptr));
  const auto deviceMemAlloc = reinterpret_cast<clDeviceMemAllocINTEL_fn>(
      clGetExtensionFunctionAddressForPlatform(platform,
                                               "clDeviceMemAllocINTEL"));
  const auto enqueueMemcpy = reinterpret_cast<clEnqueueMemcpyINTEL_fn>(
      clGetExtensionFunctionAddressForPlatform(platform,
                                               "clEnqueueMemcpyINTEL"));
  const auto memFree = reinterpret_cast<clMemFreeINTEL_fn>(
      clGetExtensionFunctionAddressForPlatform(platform, "clMemFreeINTEL"));
  ASSERT_NE(deviceMemAlloc, nullptr);
  ASSERT_NE(enqueueMemcpy, nullptr);
  ASSERT_NE(memFree, nullptr);

  std::vector<cl_half*> matrixGpu(GEMV_PARAMS.size());
  for (size_t layer = 0; layer < GEMV_PARAMS.size(); ++layer) {
    matrixGpu[layer] = static_cast<cl_half*>(
        deviceMemAlloc(context(), deviceId(), nullptr,
                       matricesHalf[layer].size() * sizeof(cl_half),
                       alignof(cl_half), &status));
    ASSERT_OCL_SUCCESS(status);
    ASSERT_OCL_SUCCESS(enqueueMemcpy(
        queue(), CL_TRUE, matrixGpu[layer], matricesHalf[layer].data(),
        matricesHalf[layer].size() * sizeof(cl_half), 0, nullptr, nullptr));
  }

  std::vector<cl_half*> vectorsGpu(GEMV_PARAMS.size() + 1);
  vectorsGpu.front() = static_cast<cl_half*>(deviceMemAlloc(
      context(), deviceId(), nullptr, inputHalf.size() * sizeof(cl_half),
      alignof(cl_half), &status));
  ASSERT_OCL_SUCCESS(status);
  ASSERT_OCL_SUCCESS(
      enqueueMemcpy(queue(), CL_TRUE, vectorsGpu.front(), inputHalf.data(),
                    inputHalf.size() * sizeof(cl_half), 0, nullptr, nullptr));
  for (size_t layer = 0; layer < GEMV_PARAMS.size(); ++layer) {
    vectorsGpu[layer + 1] = static_cast<cl_half*>(
        deviceMemAlloc(context(), deviceId(), nullptr,
                       GEMV_PARAMS[layer].rowCount * sizeof(cl_half),
                       alignof(cl_half), &status));
    ASSERT_OCL_SUCCESS(status);
  }

  const std::vector<int> clearedCompletionCounts(GEMV_PARAMS.size(), 0);
  int* completionCountsGpu = static_cast<int*>(deviceMemAlloc(
      context(), deviceId(), nullptr,
      clearedCompletionCounts.size() * sizeof(int), alignof(int), &status));
  ASSERT_OCL_SUCCESS(status);
  ASSERT_OCL_SUCCESS(enqueueMemcpy(
      queue(), CL_TRUE, completionCountsGpu, clearedCompletionCounts.data(),
      clearedCompletionCounts.size() * sizeof(int), 0, nullptr, nullptr));

  size_t totalTaskCount = 0;
  for (const ocltest::GemvParams& params : GEMV_PARAMS) {
    totalTaskCount += params.rowCount / params.rowsPerBlock;
  }
  std::vector<TaskDesc> tasks;
  tasks.reserve(totalTaskCount);
  size_t prevLayerOutputTiles = 0;
  for (size_t layer = 0; layer < GEMV_PARAMS.size(); ++layer) {
    const ocltest::GemvParams& params = GEMV_PARAMS[layer];
    const size_t taskCount = params.rowCount / params.rowsPerBlock;
    std::cout << "Layer " << layer << ": " << taskCount << " tasks\n";
    for (size_t tileId = 0; tileId < taskCount; ++tileId) {
      switch (layer) {
        case 0:
          tasks.push_back(CreateGemvTaskDesc<Gemv1024x2048Task>(
              3, matrixGpu[layer], vectorsGpu[layer], vectorsGpu[layer + 1],
              nullptr, completionCountsGpu, 0, static_cast<int>(tileId)));
          break;
        case 1:
          tasks.push_back(CreateGemvTaskDesc<Gemv3072x1024Task>(
              4, matrixGpu[layer], vectorsGpu[layer], vectorsGpu[layer + 1],
              &completionCountsGpu[0], &completionCountsGpu[1],
              prevLayerOutputTiles, static_cast<int>(tileId)));
          break;
        case 2:
          tasks.push_back(CreateGemvTaskDesc<Gemv1024x3072Task>(
              5, matrixGpu[layer], vectorsGpu[layer], vectorsGpu[layer + 1],
              &completionCountsGpu[1], nullptr, prevLayerOutputTiles,
              static_cast<int>(tileId)));
          break;
      }
    }

    prevLayerOutputTiles = taskCount;
  }

  TaskManager taskManager;
  ASSERT_OCL_SUCCESS(HostInitalizeTaskSystem(taskManager, tasks, deviceId(),
                                             context(), queue()));
  cl_mem taskManagerBuffer =
      clCreateBuffer(context(), CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                     sizeof(taskManager), &taskManager, &status);
  ASSERT_OCL_SUCCESS(status);
  ASSERT_OCL_SUCCESS(
      clSetKernelArg(binary.kernel, 0, sizeof(cl_mem), &taskManagerBuffer));
  std::vector<void*> indirectPointers;
  indirectPointers.reserve(matrixGpu.size() + vectorsGpu.size() + 1);
  indirectPointers.insert(indirectPointers.end(), matrixGpu.begin(),
                          matrixGpu.end());
  indirectPointers.insert(indirectPointers.end(), vectorsGpu.begin(),
                          vectorsGpu.end());
  indirectPointers.push_back(completionCountsGpu);
  ASSERT_OCL_SUCCESS(clSetKernelExecInfo(
      binary.kernel, CL_KERNEL_EXEC_INFO_USM_PTRS_INTEL,
      indirectPointers.size() * sizeof(void*), indirectPointers.data()));
  const auto selectedWorkers = std::min(WORKERS, totalTaskCount);
  const size_t globalWorkSize = selectedWorkers * WORK_GROUP_SIZE;

  std::cout << "Total task count: " << totalTaskCount
            << ", workers: " << selectedWorkers << "\n";

  // Get initial output to check correctness after benchmarking:
  ASSERT_OCL_SUCCESS(clEnqueueNDRangeKernel(queue(), binary.kernel, 1, nullptr,
                                            &globalWorkSize, &WORK_GROUP_SIZE,
                                            0, nullptr, nullptr));

  std::vector<cl_half> outputHalf(GEMV_PARAMS.back().rowCount);
  ASSERT_OCL_SUCCESS(
      enqueueMemcpy(queue(), CL_TRUE, outputHalf.data(), vectorsGpu.back(),
                    outputHalf.size() * sizeof(cl_half), 0, nullptr, nullptr));

  std::cout << "Benchmarking three-GEMV task-system chain...\n";
  const ocltest::ProfileResult taskSystemProfile =
      ocltest::ProfileOpenCL<ocltest::CLEAR_CACHE_BEFORE_BENCHMARK>(

          [&]() {
            ASSERT_OCL_SUCCESS(clEnqueueNDRangeKernel(
                queue(), binary.kernel, 1, nullptr, &globalWorkSize,
                &WORK_GROUP_SIZE, 0, nullptr, nullptr));
          },
          queue(), ocltest::WARMUP_ITERATIONS, ocltest::BENCHMARK_ITERATIONS);

  const ocltest::GemvBenchmarkResult taskSystemResult = {
      taskSystemProfile, ConvertToFloat(outputHalf)};

  ASSERT_OCL_SUCCESS(clReleaseMemObject(taskManagerBuffer));
  ASSERT_OCL_SUCCESS(HostReleaseTaskSystem(taskManager, deviceId(), context()));
  ASSERT_OCL_SUCCESS(memFree(context(), completionCountsGpu));
  for (cl_half* vectorGpu : vectorsGpu) {
    ASSERT_OCL_SUCCESS(memFree(context(), vectorGpu));
  }
  for (cl_half* matrix : matrixGpu) {
    ASSERT_OCL_SUCCESS(memFree(context(), matrix));
  }
  releaseOCLBinary(binary);

  const ocltest::GemvBenchmarkResult openClResult =
      benchmarkOpenClGemvChain(matrices, input, GEMV_PARAMS, GEMV_KERNEL_PATH);
  const ocltest::GemvBenchmarkResult dnnlResult =
      benchmarkDnnlGemvChain(matrices, input, GEMV_PARAMS);

  taskSystemResult.profileResult.print("3-GEMV task-system chain");
  openClResult.profileResult.print("3-GEMV OpenCL chain");
  dnnlResult.profileResult.print("3-GEMV oneDNN chain");
  std::cout << "OpenCL / task-system speedup: "
            << openClResult.profileResult.averageUs /
                   taskSystemResult.profileResult.averageUs
            << "x\n";
  std::cout << "oneDNN / task-system speedup: "
            << dnnlResult.profileResult.averageUs /
                   taskSystemResult.profileResult.averageUs
            << "x\n";

  ASSERT_EQ(taskSystemResult.output.size(), dnnlResult.output.size());
  ASSERT_EQ(taskSystemResult.output.size(), openClResult.output.size());
  for (size_t index = 0; index < taskSystemResult.output.size(); ++index) {
    ASSERT_NEAR(taskSystemResult.output[index], openClResult.output[index],
                ocltest::ABS_ERROR)
        << "Task-system GEMV chain differs from OpenCL at index " << index;
    ASSERT_NEAR(taskSystemResult.output[index], dnnlResult.output[index],
                ocltest::ABS_ERROR)
        << "Task-system GEMV chain differs from oneDNN at index " << index;
  }
}

}  // namespace