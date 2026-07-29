#include <algorithm>
#include <cctype>
#include <cmath>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

#include "oclTestFixture.h"

namespace ocltest {
namespace utility {

// Reads the contents of a text file and returns it as a string.
static std::string readTextFile(const std::string& path) {
  std::ifstream stream(path);
  if (!stream) {
    throw std::runtime_error("Failed to open OpenCL source file: " + path);
  }

  std::ostringstream buffer;
  buffer << stream.rdbuf();
  return buffer.str();
}

// Returns the OpenCL program build status.
static std::string lowerCopy(std::string value) {
  std::transform(
      value.begin(), value.end(), value.begin(),
      [](unsigned char ch) { return static_cast<char>(std::tolower(ch)); });
  return value;
}

// Counts the number of occurrences of a substring in a string.
static size_t countSubstring(const std::string& text,
                             const std::string& token) {
  size_t count = 0;
  size_t offset = text.find(token);
  while (offset != std::string::npos) {
    ++count;
    offset = text.find(token, offset + token.size());
  }
  return count;
}
}  // namespace utility

// Prints OpenCL error status as a string.
static const char* PrintBuildStatusName(cl_build_status status) {
  switch (status) {
    case CL_BUILD_SUCCESS:
      return "CL_BUILD_SUCCESS";
    case CL_BUILD_NONE:
      return "CL_BUILD_NONE";
    case CL_BUILD_ERROR:
      return "CL_BUILD_ERROR";
    case CL_BUILD_IN_PROGRESS:
      return "CL_BUILD_IN_PROGRESS";
    default:
      return "UNKNOWN_OPENCL_BUILD_STATUS";
  }
}

// Gets OCL devices.
static std::vector<cl_device_id> getDevices(cl_platform_id platform,
                                            cl_device_type type) {
  cl_uint count = 0;
  cl_int status = clGetDeviceIDs(platform, type, 0, nullptr, &count);
  if (status == CL_DEVICE_NOT_FOUND) {
    return {};
  }
  ASSERT_OCL_SUCCESS(status);

  std::vector<cl_device_id> devices(count);
  ASSERT_OCL_SUCCESS(
      clGetDeviceIDs(platform, type, count, devices.data(), nullptr));
  return devices;
}

// Prints OpenCL platform information as a string.
static std::string platformInfo(cl_platform_id platform,
                                cl_platform_info info) {
  size_t size = 0;
  ASSERT_OCL_SUCCESS(clGetPlatformInfo(platform, info, 0, nullptr, &size));
  std::string value(size, '\0');
  ASSERT_OCL_SUCCESS(
      clGetPlatformInfo(platform, info, size, value.data(), nullptr));
  if (!value.empty() && value.back() == '\0') {
    value.pop_back();
  }
  return value;
}

// Prints OpenCL device information as a string.
static std::string deviceInfo(cl_device_id device, cl_device_info info) {
  size_t size = 0;
  ASSERT_OCL_SUCCESS(clGetDeviceInfo(device, info, 0, nullptr, &size));
  std::string value(size, '\0');
  ASSERT_OCL_SUCCESS(
      clGetDeviceInfo(device, info, size, value.data(), nullptr));
  if (!value.empty() && value.back() == '\0') {
    value.pop_back();
  }
  return value;
}

// Returns the OpenCL program build log.
static std::string programBuildLog(cl_program program, cl_device_id device) {
  size_t logSize = 0;
  ASSERT_OCL_SUCCESS(clGetProgramBuildInfo(
      program, device, CL_PROGRAM_BUILD_LOG, 0, nullptr, &logSize));

  std::string log(logSize, '\0');
  if (logSize > 0) {
    ASSERT_OCL_SUCCESS(clGetProgramBuildInfo(program, device,
                                             CL_PROGRAM_BUILD_LOG, log.size(),
                                             log.data(), nullptr));
  }
  while (!log.empty() && log.back() == '\0') {
    log.pop_back();
  }
  return log;
}

// Returns the OpenCL program build status.
static cl_build_status programBuildStatus(cl_program program,
                                          cl_device_id device) {
  cl_build_status buildStatus = CL_BUILD_NONE;
  ASSERT_OCL_SUCCESS(
      clGetProgramBuildInfo(program, device, CL_PROGRAM_BUILD_STATUS,
                            sizeof(buildStatus), &buildStatus, nullptr));
  return buildStatus;
}

// Prints OpenCL build diagnostics, including build log, warnings, and errors.
static void printBuildDiagnostics(cl_program program, cl_device_id device,
                                  cl_int buildResult) {
  const cl_build_status buildStatus = programBuildStatus(program, device);
  const std::string buildLog = programBuildLog(program, device);
  const std::string normalizedLog = utility::lowerCopy(buildLog);
  const size_t warningCount =
      utility::countSubstring(normalizedLog, "warning:");
  const size_t errorCount = utility::countSubstring(normalizedLog, "error:");

  if (buildResult != CL_SUCCESS) {
    std::cerr << "OpenCL compiler error: " << printOclErrorStr(buildResult)
              << " (" << buildResult << ")\n";
    std::cerr << "OpenCL build status: " << PrintBuildStatusName(buildStatus)
              << " (" << buildStatus << ")\n";
  }

  if (warningCount > 0) {
    std::cerr << "OpenCL compiler warnings: " << warningCount << '\n';
  }
  if (errorCount > 0) {
    std::cerr << "OpenCL compiler log errors: " << errorCount << '\n';
  }

  if (!buildLog.empty()) {
    std::cerr << "OpenCL compiler diagnostics:\n" << buildLog << '\n';
  } else if (buildResult != CL_SUCCESS) {
    std::cerr << "OpenCL compiler diagnostics: <empty>\n";
  }
}

// Selects an OpenCL device and prints its platform and device information.
static cl_device_id selectAndPrintOclDeviceInfo(size_t deviceIndex = 0) {
  cl_uint count = 0;
  ASSERT_OCL_SUCCESS(clGetPlatformIDs(0, nullptr, &count));
  std::vector<cl_platform_id> platforms(count);
  ASSERT_OCL_SUCCESS(clGetPlatformIDs(count, platforms.data(), nullptr));
  if (platforms.empty()) {
    throw std::runtime_error(
        "No OpenCL platforms found. Check that an OpenCL runtime/ICD is "
        "installed.");
  }

  const size_t platformIndex = 0;
  const cl_platform_id platform = platforms[platformIndex];

  auto devices = getDevices(platform, CL_DEVICE_TYPE_GPU);
  if (devices.empty()) {
    devices = getDevices(platform, CL_DEVICE_TYPE_ALL);
  }
  if (devices.empty()) {
    throw std::runtime_error("No OpenCL devices found on selected platform");
  }

  EXPECT_LT(deviceIndex, devices.size())
      << "OPENCL_DEVICE_INDEX is out of range";
  const cl_device_id device = devices[deviceIndex];

  std::cout << "Platform: " << platformInfo(platform, CL_PLATFORM_NAME) << '\n';
  std::cout << "Device: " << deviceInfo(device, CL_DEVICE_NAME) << '\n';
  return device;
}

//////////////////////////////////////////////////////////////////
//
// OclTestFixture
//
//////////////////////////////////////////////////////////////////

//////////////////////////////////////////////////////////////////
void OclTestFixture::SetUp() {
  device = selectAndPrintOclDeviceInfo();

  cl_int status = CL_SUCCESS;
  oclContext = clCreateContext(nullptr, 1, &device, nullptr, nullptr, &status);
  ASSERT_OCL_SUCCESS(status);

  const cl_queue_properties props[] = {
      CL_QUEUE_PROPERTIES, CL_QUEUE_PROFILING_ENABLE,
      0  // terminator
  };

  commandQueue =
      clCreateCommandQueueWithProperties(oclContext, device, props, &status);
  ASSERT_OCL_SUCCESS(status);
}

//////////////////////////////////////////////////////////////////
void OclTestFixture::TearDown() {
  ASSERT_OCL_SUCCESS(clReleaseCommandQueue(commandQueue));
  ASSERT_OCL_SUCCESS(clReleaseContext(oclContext));
}

//////////////////////////////////////////////////////////////////
OclTestFixture::OCLBinary OclTestFixture::createProgramAndKernel(
    const std::string& sourcePath, const std::string& kernelName,
    const std::string& buildOptions) const {
  cl_int status = CL_SUCCESS;
  std::string source = utility::readTextFile(sourcePath);
  const char* sourceData = source.c_str();
  const size_t sourceSize = source.size();
  cl_program program = clCreateProgramWithSource(context(), 1, &sourceData,
                                                 &sourceSize, &status);
  ASSERT_OCL_SUCCESS(status);

  const std::string defaultBuildOptions = "-Werror -cl-std=CL3.0";
  const std::string finalBuildOptions =
      defaultBuildOptions + " " + buildOptions;
  status = clBuildProgram(program, 1, &device, finalBuildOptions.c_str(),
                          nullptr, nullptr);
  printBuildDiagnostics(program, device, status);
  ASSERT_OCL_SUCCESS(status);

  cl_kernel kernel = clCreateKernel(program, kernelName.c_str(), &status);
  ASSERT_OCL_SUCCESS(status);

  return {program, kernel};
}

//////////////////////////////////////////////////////////////////
void OclTestFixture::releaseOCLBinary(OCLBinary binary) {
  ASSERT_OCL_SUCCESS(clReleaseKernel(binary.kernel));
  ASSERT_OCL_SUCCESS(clReleaseProgram(binary.program));
}

}  // namespace ocltest