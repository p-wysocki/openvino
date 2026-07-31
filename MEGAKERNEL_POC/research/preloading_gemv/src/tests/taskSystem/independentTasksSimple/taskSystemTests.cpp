#include <CL/cl_ext.h>

#include <algorithm>
#include <cstring>
#include <numeric>
#include <vector>

#include "../../../../../common/oclTestFixture.h"
#include "../../../ocl/taskSystem/host/taskManagerHost.h"
#include "ocl/testTask.h"

namespace {

constexpr size_t WORKERS = 80;
constexpr size_t THREADS = 512;

class TaskSystemTests : public ocltest::OclTestFixture {};

const std::string TASK_SYSTEM_KERNEL_PATH =
    std::string(OPENCL_KERNEL_SOURCE_PATH) +
    "../tests/taskSystem/independentTasksSimple/ocl/";

TEST_F(TaskSystemTests, ClaimsOneHundredTasks) {
  constexpr size_t taskCount = 100;

  const OCLBinary binary = createProgramAndKernel(
      TASK_SYSTEM_KERNEL_PATH + "taskManagerTest.cl", "taskManagerTest",
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

  std::vector<int> taskExecutedHost_ClearBuffer(taskCount, -1);
  std::vector<int> taskExecutedHost(taskCount, -1);
  int* taskExecutedGPU = static_cast<int*>(
      deviceMemAlloc(context(), deviceId(), nullptr, taskCount * sizeof(int),
                     alignof(int), &status));
  ASSERT_OCL_SUCCESS(status);
  // -------------------------------------------------

  // Create task queue on the host and submit it:
  std::vector<TaskDesc> hostTasks(taskCount);
  for (size_t index = 0; index < taskCount; ++index) {
    hostTasks[index].type = 0;
    TestTask task;
    task.id = static_cast<int>(index);
    task.output = taskExecutedGPU + index;
    static_assert(sizeof(task) <= PAYLOAD_SIZE,
                  "TestTask size exceeds payload size");
    std::memcpy(hostTasks[index].payload, &task, sizeof(task));
  }

  TaskManager taskManager;
  ASSERT_OCL_SUCCESS(HostInitalizeTaskSystem(taskManager, hostTasks, deviceId(),
                                             context(), queue()));
  cl_mem taskManagerBuffer =
      clCreateBuffer(context(), CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                     sizeof(taskManager), &taskManager, &status);
  ASSERT_OCL_SUCCESS(status);
  ASSERT_OCL_SUCCESS(
      clSetKernelExecInfo(binary.kernel, CL_KERNEL_EXEC_INFO_USM_PTRS_INTEL,
                          sizeof(taskExecutedGPU), &taskExecutedGPU));
  // -------------------------------------------------

  size_t workers = WORKERS;
  for (int i = 0; i < 100; ++i) {
    ASSERT_OCL_SUCCESS(enqueueMemcpy(
        queue(), CL_TRUE, taskExecutedGPU, taskExecutedHost_ClearBuffer.data(),
        taskCount * sizeof(int), 0, nullptr, nullptr));

    ASSERT_OCL_SUCCESS(
        clSetKernelArg(binary.kernel, 0, sizeof(cl_mem), &taskManagerBuffer));

    const size_t globalWorkSize = workers * THREADS;
    ASSERT_OCL_SUCCESS(clEnqueueNDRangeKernel(queue(), binary.kernel, 1,
                                              nullptr, &globalWorkSize,
                                              &THREADS, 0, nullptr, nullptr));

    ASSERT_OCL_SUCCESS(enqueueMemcpy(queue(), CL_TRUE, taskExecutedHost.data(),
                                     taskExecutedGPU, taskCount * sizeof(int),
                                     0, nullptr, nullptr));

    for (int i = 0; i < taskExecutedHost.size(); ++i) {
      ASSERT_GE(taskExecutedHost[i], i * i)
          << "Task " << i << " was not executed correctly";
    }

    workers = (workers + 53) % WORKERS +
              1;  // Change number of workers for next iteration
  }

  ASSERT_OCL_SUCCESS(clReleaseMemObject(taskManagerBuffer));
  ASSERT_OCL_SUCCESS(memFree(context(), taskExecutedGPU));
  ASSERT_OCL_SUCCESS(HostReleaseTaskSystem(taskManager, deviceId(), context()));
  releaseOCLBinary(binary);
}

}  // namespace