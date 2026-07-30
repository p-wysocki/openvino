#include <CL/cl_ext.h>

#include <algorithm>
#include <numeric>
#include <vector>

#include "../../common/oclTestFixture.h"
#include "ocl/taskSystem/shared/taskManager.h"

namespace {

constexpr size_t WORKERS = 80;
constexpr size_t THREADS = 512;

#define CHECK_OCL_SUCCESS(stmt) \
  {                             \
    cl_int status = (stmt);     \
    if (status != CL_SUCCESS) { \
      return status;            \
    }                           \
  }

/////////////////////////////////////////////////////////
static cl_int HostInitalizeTaskSystem(TaskManager& taskManager,
                                      std::vector<TaskDesc>& tasksQueue,
                                      cl_device_id deviceId, cl_context context,
                                      cl_command_queue queue) {
  const int ZERO = 0;

  cl_platform_id platform = nullptr;
  CHECK_OCL_SUCCESS(clGetDeviceInfo(deviceId, CL_DEVICE_PLATFORM,
                                    sizeof(platform), &platform, nullptr));
  const auto deviceMemAlloc = reinterpret_cast<clDeviceMemAllocINTEL_fn>(
      clGetExtensionFunctionAddressForPlatform(platform,
                                               "clDeviceMemAllocINTEL"));
  const auto enqueueMemcpy = reinterpret_cast<clEnqueueMemcpyINTEL_fn>(
      clGetExtensionFunctionAddressForPlatform(platform,
                                               "clEnqueueMemcpyINTEL"));
  cl_int status = CL_SUCCESS;
  TaskDesc* taskQueueGPU = static_cast<TaskDesc*>(deviceMemAlloc(
      context, deviceId, nullptr, tasksQueue.size() * sizeof(TaskDesc),
      alignof(TaskDesc), &status));
  CHECK_OCL_SUCCESS(status);
  int* nextTaskIDGPU = static_cast<int*>(deviceMemAlloc(
      context, deviceId, nullptr, sizeof(int), alignof(int), &status));
  CHECK_OCL_SUCCESS(status);
  int* syncBarrierBufferGPU = static_cast<int*>(deviceMemAlloc(
      context, deviceId, nullptr, sizeof(int), alignof(int), &status));
  CHECK_OCL_SUCCESS(status);

  CHECK_OCL_SUCCESS(
      enqueueMemcpy(queue, CL_TRUE, taskQueueGPU, tasksQueue.data(),
                    tasksQueue.size() * sizeof(TaskDesc), 0, nullptr, nullptr));
  CHECK_OCL_SUCCESS(enqueueMemcpy(queue, CL_TRUE, nextTaskIDGPU, &ZERO,
                                  sizeof(ZERO), 0, nullptr, nullptr));
  CHECK_OCL_SUCCESS(enqueueMemcpy(queue, CL_TRUE, syncBarrierBufferGPU, &ZERO,
                                  sizeof(ZERO), 0, nullptr, nullptr));

  taskManager.workQueue = taskQueueGPU;
  taskManager.workQueueSize = static_cast<int>(tasksQueue.size());
  taskManager.processedTaskCount = nextTaskIDGPU;
  taskManager.syncBarrierBuffer = syncBarrierBufferGPU;

  return CL_SUCCESS;
}

/////////////////////////////////////////////////////////
static cl_int HostReleaseTaskSystem(TaskManager& taskManager,
                                    cl_device_id deviceId, cl_context context) {
  cl_platform_id platform = nullptr;
  CHECK_OCL_SUCCESS(clGetDeviceInfo(deviceId, CL_DEVICE_PLATFORM,
                                    sizeof(platform), &platform, nullptr));
  const auto memFree = reinterpret_cast<clMemFreeINTEL_fn>(
      clGetExtensionFunctionAddressForPlatform(platform, "clMemFreeINTEL"));
  CHECK_OCL_SUCCESS(
      memFree(context, const_cast<TaskDesc*>(taskManager.workQueue)));
  CHECK_OCL_SUCCESS(memFree(context, taskManager.processedTaskCount));
  CHECK_OCL_SUCCESS(memFree(context, taskManager.syncBarrierBuffer));

  return CL_SUCCESS;
}

class TaskSystemTests : public ocltest::OclTestFixture {};

const std::string TASK_SYSTEM_KERNEL_PATH =
    std::string(OPENCL_KERNEL_SOURCE_PATH) + "/taskSystem/";

TEST_F(TaskSystemTests, ClaimsOneHundredTasks) {
  constexpr size_t taskCount = 100;

  const OCLBinary binary = createProgramAndKernel(
      TASK_SYSTEM_KERNEL_PATH + "taskManagerTest.cl", "taskManagerTest",
      "-I " + TASK_SYSTEM_KERNEL_PATH);

  std::vector<TaskDesc> hostTasks(taskCount);
  for (size_t index = 0; index < taskCount; ++index) {
    hostTasks[index].input = nullptr;
    hostTasks[index].output = nullptr;
    hostTasks[index].weights = nullptr;
    hostTasks[index].id = static_cast<int>(index);
    hostTasks[index].taskType = GEMV;
  }

  cl_int status = CL_SUCCESS;
  TaskManager taskManager;
  ASSERT_OCL_SUCCESS(HostInitalizeTaskSystem(taskManager, hostTasks, deviceId(),
                                             context(), queue()));
  cl_mem taskManagerBuffer =
      clCreateBuffer(context(), CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                     sizeof(taskManager), &taskManager, &status);
  ASSERT_OCL_SUCCESS(status);

  std::vector<int> taskExecutedHost_ClearBuffer(taskCount, -1);
  std::vector<int> taskExecutedHost(taskCount, -1);
  cl_mem taskExecutedGPU = clCreateBuffer(
      context(), CL_MEM_READ_WRITE, taskCount * sizeof(int), nullptr, &status);
  ASSERT_OCL_SUCCESS(status);

  size_t workers = WORKERS;
  for (int i = 0; i < 100; ++i) {
    ASSERT_OCL_SUCCESS(clEnqueueWriteBuffer(
        queue(), taskExecutedGPU, CL_TRUE, 0, taskCount * sizeof(int),
        taskExecutedHost_ClearBuffer.data(), 0, nullptr, nullptr));

    ASSERT_OCL_SUCCESS(
        clSetKernelArg(binary.kernel, 0, sizeof(cl_mem), &taskManagerBuffer));
    ASSERT_OCL_SUCCESS(
        clSetKernelArg(binary.kernel, 1, sizeof(cl_mem), &taskExecutedGPU));

    const size_t globalWorkSize = workers * THREADS;
    ASSERT_OCL_SUCCESS(clEnqueueNDRangeKernel(queue(), binary.kernel, 1,
                                              nullptr, &globalWorkSize,
                                              &THREADS, 0, nullptr, nullptr));

    ASSERT_OCL_SUCCESS(clEnqueueReadBuffer(
        queue(), taskExecutedGPU, CL_TRUE, 0, taskCount * sizeof(int),
        taskExecutedHost.data(), 0, nullptr, nullptr));

    for (int i = 0; i < taskExecutedHost.size(); ++i) {
      std::cout << "Task with id <" << i << "> was executed by "
                << taskExecutedHost[i] << std::endl;
      ASSERT_GE(taskExecutedHost[i], 0);
    }

    workers = (workers + 17) % WORKERS +
              1;  // Change number of workers for next iteration
  }

  ASSERT_OCL_SUCCESS(clReleaseMemObject(taskManagerBuffer));
  ASSERT_OCL_SUCCESS(clReleaseMemObject(taskExecutedGPU));
  ASSERT_OCL_SUCCESS(HostReleaseTaskSystem(taskManager, deviceId(), context()));
  releaseOCLBinary(binary);
}

}  // namespace