#include <CL/cl_ext.h>

#include <algorithm>
#include <numeric>
#include <vector>

#include "../../common/oclTestFixture.h"
#include "ocl/taskSystem/host/taskManagerHost.h"

namespace {

constexpr size_t WORKERS = 80;
constexpr size_t THREADS = 512;

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