#include <CL/cl_ext.h>

#include <algorithm>
#include <cstring>
#include <numeric>
#include <vector>

#include "../../../../../common/oclTestFixture.h"
#include "../../../ocl/taskSystem/host/taskManagerHost.h"
#include "ocl/tasks/pow2Task.h"

namespace {

constexpr size_t WORKERS = 80;
constexpr size_t THREADS = 512;

class TaskSystemTests : public ocltest::OclTestFixture {};

const std::string TASK_SYSTEM_KERNEL_PATH =
    std::string(OPENCL_KERNEL_SOURCE_PATH) +
    "../tests/taskSystem/unaryChainTest/ocl/";

inline int Pow2(int x) { return x * x; }

TEST_F(TaskSystemTests, UnaryChainTest) {
  constexpr size_t taskCount = 100;

  const OCLBinary binary = createProgramAndKernel(
      TASK_SYSTEM_KERNEL_PATH + "taskManagerKernel.cl", "taskManagerKernel",
      "-I " + std::string(OPENCL_KERNEL_SOURCE_PATH) + " -I " +
          std::string(TASK_SYSTEM_KERNEL_PATH));

  // Create buffers:
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
  const auto setKernelArgMemPointer =
      reinterpret_cast<clSetKernelArgMemPointerINTEL_fn>(
          clGetExtensionFunctionAddressForPlatform(
              platform, "clSetKernelArgMemPointerINTEL"));
  const auto memFree = reinterpret_cast<clMemFreeINTEL_fn>(
      clGetExtensionFunctionAddressForPlatform(platform, "clMemFreeINTEL"));

  std::vector<int> inputHost(taskCount * THREADS);
  for (int i = 0; i < inputHost.size(); ++i) {
    inputHost[i] = i;
  }

  int* inputGPU = static_cast<int*>(
      deviceMemAlloc(context(), deviceId(), nullptr,
                     inputHost.size() * sizeof(int), alignof(int), &status));
  ASSERT_OCL_SUCCESS(status);
  ASSERT_OCL_SUCCESS(enqueueMemcpy(queue(), CL_TRUE, inputGPU, inputHost.data(),
                                   inputHost.size() * sizeof(int), 0, nullptr,
                                   nullptr));

  int* outputGPU = static_cast<int*>(
      deviceMemAlloc(context(), deviceId(), nullptr,
                     inputHost.size() * sizeof(int), alignof(int), &status));
  ASSERT_OCL_SUCCESS(status);

  std::vector<int> outputClearHOST(inputHost.size(), 0);
  std::vector<int> outputHOST(inputHost.size(), 0);
  // -------------------------------------------------

  // Create task queue on the host and submit it:
  std::vector<TaskDesc> topologicallySortedTaskQueue(taskCount);
  for (size_t index = 0; index < taskCount; ++index) {
    topologicallySortedTaskQueue[index].type = 0;
    Pow2Task task;
    task.size = THREADS;
    task.input = inputGPU + index * THREADS;
    task.output = outputGPU + index * THREADS;
    static_assert(sizeof(task) <= PAYLOAD_SIZE,
                  "Pow2Task size exceeds payload size");
    std::memcpy(topologicallySortedTaskQueue[index].payload, &task,
                sizeof(task));
  }

  TaskManager taskManager;
  ASSERT_OCL_SUCCESS(HostInitalizeTaskSystem(taskManager,
                                             topologicallySortedTaskQueue,
                                             deviceId(), context(), queue()));
  cl_mem taskManagerBuffer =
      clCreateBuffer(context(), CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                     sizeof(taskManager), &taskManager, &status);
  ASSERT_OCL_SUCCESS(status);
  // -------------------------------------------------

  size_t workers = WORKERS;
  for (int i = 0; i < 100; ++i) {
    ASSERT_OCL_SUCCESS(enqueueMemcpy(
        queue(), CL_TRUE, outputGPU, outputClearHOST.data(),
        outputClearHOST.size() * sizeof(int), 0, nullptr, nullptr));

    ASSERT_OCL_SUCCESS(
        clSetKernelArg(binary.kernel, 0, sizeof(cl_mem), &taskManagerBuffer));

    const size_t globalWorkSize = workers * THREADS;
    ASSERT_OCL_SUCCESS(clEnqueueNDRangeKernel(queue(), binary.kernel, 1,
                                              nullptr, &globalWorkSize,
                                              &THREADS, 0, nullptr, nullptr));

    ASSERT_OCL_SUCCESS(enqueueMemcpy(queue(), CL_TRUE, outputHOST.data(),
                                     outputGPU, outputHOST.size() * sizeof(int),
                                     0, nullptr, nullptr));

    for (int i = 0; i < outputHOST.size(); ++i) {
      ASSERT_EQ(outputHOST[i], Pow2(inputHost[i]))
          << "Task " << i << " was not executed correctly";
    }

    workers = (workers + 53) % WORKERS +
              1;  // Change number of workers for next iteration
  }

  ASSERT_OCL_SUCCESS(clReleaseMemObject(taskManagerBuffer));
  ASSERT_OCL_SUCCESS(memFree(context(), inputGPU));
  ASSERT_OCL_SUCCESS(memFree(context(), outputGPU));
  ASSERT_OCL_SUCCESS(HostReleaseTaskSystem(taskManager, deviceId(), context()));
  releaseOCLBinary(binary);
}

}  // namespace