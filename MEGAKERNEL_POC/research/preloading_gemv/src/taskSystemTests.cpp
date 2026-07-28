#include <CL/cl_ext.h>

#include <algorithm>
#include <numeric>
#include <vector>

#include "../../common/oclTestFixture.h"

namespace {

class TaskSystemTests : public ocltest::OclTestFixture {};

struct HostTaskDesc {
  const void* weights;
  const void* input;
  void* output;
  int id;
  int taskType;
};

struct HostTaskManager {
  const HostTaskDesc* array;
  int* atomicSlotId;
};

const std::string TASK_SYSTEM_KERNEL_PATH =
    std::string(OPENCL_KERNEL_SOURCE_PATH) + "/taskSystem/";

TEST_F(TaskSystemTests, ClaimsOneHundredTasks) {
  constexpr size_t taskCount = 100;
  constexpr size_t workGroupSize = 8;

  const OCLBinary binary = createProgramAndKernel(
      TASK_SYSTEM_KERNEL_PATH + "taskManagerTest.cl", "taskManagerTest",
      "-I " + TASK_SYSTEM_KERNEL_PATH);

  cl_platform_id platform = nullptr;
  ASSERT_OCL_SUCCESS(clGetDeviceInfo(deviceId(), CL_DEVICE_PLATFORM,
                                     sizeof(platform), &platform, nullptr));
  const auto sharedMemAlloc = reinterpret_cast<clSharedMemAllocINTEL_fn>(
      clGetExtensionFunctionAddressForPlatform(platform,
                                               "clSharedMemAllocINTEL"));
  const auto memFree = reinterpret_cast<clMemFreeINTEL_fn>(
      clGetExtensionFunctionAddressForPlatform(platform, "clMemFreeINTEL"));
  const auto setKernelArgMemPointer =
      reinterpret_cast<clSetKernelArgMemPointerINTEL_fn>(
          clGetExtensionFunctionAddressForPlatform(
              platform, "clSetKernelArgMemPointerINTEL"));
  ASSERT_NE(sharedMemAlloc, nullptr);
  ASSERT_NE(memFree, nullptr);
  ASSERT_NE(setKernelArgMemPointer, nullptr);

  cl_int status = CL_SUCCESS;
  auto* tasks = static_cast<HostTaskDesc*>(sharedMemAlloc(
      context(), deviceId(), nullptr, taskCount * sizeof(HostTaskDesc),
      alignof(HostTaskDesc), &status));
  ASSERT_OCL_SUCCESS(status);
  auto* atomicSlotId = static_cast<int*>(sharedMemAlloc(
      context(), deviceId(), nullptr, sizeof(int), alignof(int), &status));
  ASSERT_OCL_SUCCESS(status);
  auto* claimedTaskIds = static_cast<int*>(
      sharedMemAlloc(context(), deviceId(), nullptr, taskCount * sizeof(int),
                     alignof(int), &status));
  ASSERT_OCL_SUCCESS(status);

  for (size_t index = 0; index < taskCount; ++index) {
    tasks[index] = {nullptr, nullptr, nullptr, static_cast<int>(index), 0};
  }
  *atomicSlotId = 0;
  std::fill(claimedTaskIds, claimedTaskIds + taskCount, -1);
  HostTaskManager taskManager = {tasks, atomicSlotId};
  cl_mem taskManagerBuffer =
      clCreateBuffer(context(), CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                     sizeof(taskManager), &taskManager, &status);
  ASSERT_OCL_SUCCESS(status);

  ASSERT_OCL_SUCCESS(
      clSetKernelArg(binary.kernel, 0, sizeof(cl_mem), &taskManagerBuffer));
  ASSERT_OCL_SUCCESS(setKernelArgMemPointer(binary.kernel, 1, claimedTaskIds));
  const void* indirectUsmPointers[] = {tasks, atomicSlotId};
  ASSERT_OCL_SUCCESS(
      clSetKernelExecInfo(binary.kernel, CL_KERNEL_EXEC_INFO_USM_PTRS_INTEL,
                          sizeof(indirectUsmPointers), indirectUsmPointers));
  const size_t globalWorkSize = taskCount * workGroupSize;
  ASSERT_OCL_SUCCESS(clEnqueueNDRangeKernel(queue(), binary.kernel, 1, nullptr,
                                            &globalWorkSize, &workGroupSize, 0,
                                            nullptr, nullptr));
  ASSERT_OCL_SUCCESS(clFinish(queue()));

  const int claimedTaskCount = *atomicSlotId;
  std::vector<int> actualTaskIds(claimedTaskIds, claimedTaskIds + taskCount);
  std::sort(actualTaskIds.begin(), actualTaskIds.end());
  std::vector<int> expectedTaskIds(taskCount);
  std::iota(expectedTaskIds.begin(), expectedTaskIds.end(), 0);

  ASSERT_OCL_SUCCESS(clReleaseMemObject(taskManagerBuffer));
  ASSERT_OCL_SUCCESS(memFree(context(), claimedTaskIds));
  ASSERT_OCL_SUCCESS(memFree(context(), atomicSlotId));
  ASSERT_OCL_SUCCESS(memFree(context(), tasks));
  releaseOCLBinary(binary);

  EXPECT_EQ(claimedTaskCount, taskCount);
  EXPECT_EQ(actualTaskIds, expectedTaskIds);
}

}  // namespace