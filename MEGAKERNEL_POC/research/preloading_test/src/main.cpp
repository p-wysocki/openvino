#include <CL/cl.h>
#include <gtest/gtest.h>

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

#include "utils.h"

namespace {

const char* kernel_source_path = OPENCL_KERNEL_SOURCE_PATH;

const char* status_name(cl_int status) {
  switch (status) {
    case CL_SUCCESS:
      return "CL_SUCCESS";
    case CL_DEVICE_NOT_FOUND:
      return "CL_DEVICE_NOT_FOUND";
    case CL_DEVICE_NOT_AVAILABLE:
      return "CL_DEVICE_NOT_AVAILABLE";
    case CL_COMPILER_NOT_AVAILABLE:
      return "CL_COMPILER_NOT_AVAILABLE";
    case CL_MEM_OBJECT_ALLOCATION_FAILURE:
      return "CL_MEM_OBJECT_ALLOCATION_FAILURE";
    case CL_OUT_OF_RESOURCES:
      return "CL_OUT_OF_RESOURCES";
    case CL_OUT_OF_HOST_MEMORY:
      return "CL_OUT_OF_HOST_MEMORY";
    case CL_BUILD_PROGRAM_FAILURE:
      return "CL_BUILD_PROGRAM_FAILURE";
    case CL_INVALID_VALUE:
      return "CL_INVALID_VALUE";
    case CL_INVALID_DEVICE:
      return "CL_INVALID_DEVICE";
    case CL_INVALID_BINARY:
      return "CL_INVALID_BINARY";
    case CL_INVALID_BUILD_OPTIONS:
      return "CL_INVALID_BUILD_OPTIONS";
    case CL_INVALID_PROGRAM:
      return "CL_INVALID_PROGRAM";
    case CL_INVALID_PROGRAM_EXECUTABLE:
      return "CL_INVALID_PROGRAM_EXECUTABLE";
    case CL_INVALID_KERNEL_NAME:
      return "CL_INVALID_KERNEL_NAME";
    case CL_INVALID_KERNEL:
      return "CL_INVALID_KERNEL";
    case CL_INVALID_KERNEL_ARGS:
      return "CL_INVALID_KERNEL_ARGS";
    case CL_INVALID_WORK_DIMENSION:
      return "CL_INVALID_WORK_DIMENSION";
    case CL_INVALID_WORK_GROUP_SIZE:
      return "CL_INVALID_WORK_GROUP_SIZE";
    default:
      return "UNKNOWN_OPENCL_STATUS";
  }
}

const char* build_status_name(cl_build_status status) {
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

void check(cl_int status, const char* operation) {
  if (status != CL_SUCCESS) {
    throw std::runtime_error(std::string(operation) +
                             " failed: " + status_name(status) + " (" +
                             std::to_string(status) + ")");
  }
}

template <typename T>
std::vector<T> get_ids(cl_int (*getter)(cl_uint, T*, cl_uint*)) {
  cl_uint count = 0;
  check(getter(0, nullptr, &count), "query id count");
  std::vector<T> ids(count);
  check(getter(count, ids.data(), nullptr), "query ids");
  return ids;
}

std::vector<cl_device_id> get_devices(cl_platform_id platform,
                                      cl_device_type type) {
  cl_uint count = 0;
  cl_int status = clGetDeviceIDs(platform, type, 0, nullptr, &count);
  if (status == CL_DEVICE_NOT_FOUND) {
    return {};
  }
  check(status, "query device count");

  std::vector<cl_device_id> devices(count);
  check(clGetDeviceIDs(platform, type, count, devices.data(), nullptr),
        "query devices");
  return devices;
}

std::string platform_info(cl_platform_id platform, cl_platform_info info) {
  size_t size = 0;
  check(clGetPlatformInfo(platform, info, 0, nullptr, &size),
        "query platform info size");
  std::string value(size, '\0');
  check(clGetPlatformInfo(platform, info, size, value.data(), nullptr),
        "query platform info");
  if (!value.empty() && value.back() == '\0') {
    value.pop_back();
  }
  return value;
}

std::string device_info(cl_device_id device, cl_device_info info) {
  size_t size = 0;
  check(clGetDeviceInfo(device, info, 0, nullptr, &size),
        "query device info size");
  std::string value(size, '\0');
  check(clGetDeviceInfo(device, info, size, value.data(), nullptr),
        "query device info");
  if (!value.empty() && value.back() == '\0') {
    value.pop_back();
  }
  return value;
}

size_t env_index(const char* name, size_t fallback) {
  const char* value = std::getenv(name);
  if (value == nullptr || value[0] == '\0') {
    return fallback;
  }

  char* end = nullptr;
  const unsigned long parsed = std::strtoul(value, &end, 10);
  if (end == value || *end != '\0') {
    throw std::runtime_error(std::string(name) +
                             " must be a non-negative integer");
  }
  return static_cast<size_t>(parsed);
}

std::string read_text_file(const std::string& path) {
  std::ifstream stream(path);
  if (!stream) {
    throw std::runtime_error("Failed to open OpenCL source file: " + path);
  }

  std::ostringstream buffer;
  buffer << stream.rdbuf();
  return buffer.str();
}

std::string program_build_log(cl_program program, cl_device_id device) {
  size_t log_size = 0;
  check(clGetProgramBuildInfo(program, device, CL_PROGRAM_BUILD_LOG, 0, nullptr,
                              &log_size),
        "query OpenCL build log size");

  std::string log(log_size, '\0');
  if (log_size > 0) {
    check(clGetProgramBuildInfo(program, device, CL_PROGRAM_BUILD_LOG,
                                log.size(), log.data(), nullptr),
          "query OpenCL build log");
  }
  while (!log.empty() && log.back() == '\0') {
    log.pop_back();
  }
  return log;
}

std::string lower_copy(std::string value) {
  std::transform(
      value.begin(), value.end(), value.begin(),
      [](unsigned char ch) { return static_cast<char>(std::tolower(ch)); });
  return value;
}

size_t count_substring(const std::string& text, const std::string& token) {
  size_t count = 0;
  size_t offset = text.find(token);
  while (offset != std::string::npos) {
    ++count;
    offset = text.find(token, offset + token.size());
  }
  return count;
}

cl_build_status program_build_status(cl_program program, cl_device_id device) {
  cl_build_status build_status = CL_BUILD_NONE;
  check(clGetProgramBuildInfo(program, device, CL_PROGRAM_BUILD_STATUS,
                              sizeof(build_status), &build_status, nullptr),
        "query OpenCL build status");
  return build_status;
}

void print_build_diagnostics(cl_program program, cl_device_id device,
                             cl_int build_result) {
  const cl_build_status build_status = program_build_status(program, device);
  const std::string build_log = program_build_log(program, device);
  const std::string normalized_log = lower_copy(build_log);
  const size_t warning_count = count_substring(normalized_log, "warning:");
  const size_t error_count = count_substring(normalized_log, "error:");

  if (build_result != CL_SUCCESS) {
    std::cerr << "OpenCL compiler error: " << status_name(build_result) << " ("
              << build_result << ")\n";
    std::cerr << "OpenCL build status: " << build_status_name(build_status)
              << " (" << build_status << ")\n";
  }

  if (warning_count > 0) {
    std::cerr << "OpenCL compiler warnings: " << warning_count << '\n';
  }
  if (error_count > 0) {
    std::cerr << "OpenCL compiler log errors: " << error_count << '\n';
  }

  if (!build_log.empty()) {
    std::cerr << "OpenCL compiler diagnostics:\n" << build_log << '\n';
  } else if (build_result != CL_SUCCESS) {
    std::cerr << "OpenCL compiler diagnostics: <empty>\n";
  }
}

void release_context(cl_context context) {
  if (context != nullptr) {
    clReleaseContext(context);
  }
}

void release_queue(cl_command_queue queue) {
  if (queue != nullptr) {
    clReleaseCommandQueue(queue);
  }
}

void release_program(cl_program program) {
  if (program != nullptr) {
    clReleaseProgram(program);
  }
}

void release_kernel(cl_kernel kernel) {
  if (kernel != nullptr) {
    clReleaseKernel(kernel);
  }
}

void release_mem_object(cl_mem memory) {
  if (memory != nullptr) {
    clReleaseMemObject(memory);
  }
}

std::vector<float> compute_gemv_reference(const std::vector<float>& matrix,
                                          const std::vector<float>& vector,
                                          size_t row_count,
                                          size_t column_count) {
  std::vector<float> reference(row_count, 0.0f);
  for (size_t row = 0; row < row_count; ++row) {
    float accumulator = 0.0f;
    const size_t row_offset = row * column_count;
    for (size_t column = 0; column < column_count; ++column) {
      accumulator += matrix[row_offset + column] * vector[column];
    }
    reference[row] = accumulator;
  }
  return reference;
}

cl_device_id GetAndPrintOCLDeviceInfo() {
  cl_uint count = 0;
  check(clGetPlatformIDs(0, nullptr, &count), "query id count");
  std::vector<cl_platform_id> platforms(count);
  check(clGetPlatformIDs(count, platforms.data(), nullptr), "query ids");
  if (platforms.empty()) {
    throw std::runtime_error(
        "No OpenCL platforms found. Check that an OpenCL runtime/ICD is "
        "installed.");
  }

  const size_t platform_index = env_index("OPENCL_PLATFORM_INDEX", 0);
  if (platform_index >= platforms.size()) {
    throw std::runtime_error("OPENCL_PLATFORM_INDEX is out of range");
  }
  const cl_platform_id platform = platforms[platform_index];

  auto devices = get_devices(platform, CL_DEVICE_TYPE_GPU);
  if (devices.empty()) {
    devices = get_devices(platform, CL_DEVICE_TYPE_ALL);
  }
  if (devices.empty()) {
    throw std::runtime_error("No OpenCL devices found on selected platform");
  }

  const size_t device_index = env_index("OPENCL_DEVICE_INDEX", 0);
  if (device_index >= devices.size()) {
    throw std::runtime_error("OPENCL_DEVICE_INDEX is out of range");
  }
  const cl_device_id device = devices[device_index];

  std::cout << "Platform: " << platform_info(platform, CL_PLATFORM_NAME)
            << '\n';
  std::cout << "Device: " << device_info(device, CL_DEVICE_NAME) << '\n';
  return device;
}

void RunGemvKernel() {
  const std::string source = read_text_file(kernel_source_path);

  cl_device_id device = GetAndPrintOCLDeviceInfo();

  cl_int status = CL_SUCCESS;
  cl_context context =
      clCreateContext(nullptr, 1, &device, nullptr, nullptr, &status);
  check(status, "clCreateContext");

  cl_command_queue queue = clCreateCommandQueue(context, device, 0, &status);
  check(status, "clCreateCommandQueue");

  const char* source_data = source.c_str();
  const size_t source_size = source.size();
  cl_program program = clCreateProgramWithSource(context, 1, &source_data,
                                                 &source_size, &status);
  check(status, "clCreateProgramWithSource");

  const char* build_options = "-Werror";
  status = clBuildProgram(program, 1, &device, build_options, nullptr, nullptr);
  print_build_diagnostics(program, device, status);
  if (status != CL_SUCCESS) {
    check(status, "clBuildProgram");
  }

  cl_kernel kernel = clCreateKernel(program, "gemv", &status);
  check(status, "clCreateKernel");

  constexpr size_t row_count = 64;
  constexpr size_t column_count = 128;

  std::vector<float> matrix = create_random_buffer(row_count * column_count, 0);
  std::vector<float> vector = create_random_buffer(column_count, 1);
  std::vector<float> result(row_count, 0.0f);

  cl_mem matrix_buffer =
      clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                     matrix.size() * sizeof(float), matrix.data(), &status);
  check(status, "clCreateBuffer matrix");
  cl_mem vector_buffer =
      clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                     vector.size() * sizeof(float), vector.data(), &status);
  check(status, "clCreateBuffer vector");
  cl_mem result_buffer =
      clCreateBuffer(context, CL_MEM_WRITE_ONLY, result.size() * sizeof(float),
                     nullptr, &status);
  check(status, "clCreateBuffer result");

  const cl_uint cl_row_count = static_cast<cl_uint>(row_count);
  const cl_uint cl_column_count = static_cast<cl_uint>(column_count);
  check(clSetKernelArg(kernel, 0, sizeof(cl_mem), &matrix_buffer),
        "clSetKernelArg matrix");
  check(clSetKernelArg(kernel, 1, sizeof(cl_mem), &vector_buffer),
        "clSetKernelArg vector");
  check(clSetKernelArg(kernel, 2, sizeof(cl_mem), &result_buffer),
        "clSetKernelArg result");
  check(clSetKernelArg(kernel, 3, sizeof(cl_row_count), &cl_row_count),
        "clSetKernelArg row count");
  check(clSetKernelArg(kernel, 4, sizeof(cl_column_count), &cl_column_count),
        "clSetKernelArg column count");

  const size_t global_work_size = row_count;
  check(clEnqueueNDRangeKernel(queue, kernel, 1, nullptr, &global_work_size,
                               nullptr, 0, nullptr, nullptr),
        "clEnqueueNDRangeKernel");
  check(clEnqueueReadBuffer(queue, result_buffer, CL_TRUE, 0,
                            result.size() * sizeof(float), result.data(), 0,
                            nullptr, nullptr),
        "clEnqueueReadBuffer result");

  const std::vector<float> reference =
      compute_gemv_reference(matrix, vector, row_count, column_count);
  constexpr float tolerance = 1e-4f;
  for (size_t row = 0; row < row_count; ++row) {
    ASSERT_NEAR(result[row], reference[row], tolerance)
        << "GEMV result mismatch at row " << row;
  }

  std::cout << "First 10 GEMV results:\n";
  for (size_t row = 0; row < std::min<size_t>(10, row_count); ++row) {
    std::cout << row << ": " << result[row] << " (reference: " << reference[row]
              << ")\n";
  }

  release_mem_object(result_buffer);
  release_mem_object(vector_buffer);
  release_mem_object(matrix_buffer);
  release_kernel(kernel);
  release_program(program);
  release_queue(queue);
  release_context(context);

  std::cout << "GEMV OpenCL kernel executed successfully.\n";
}

}  // namespace

TEST(PreloadingTest, GemvKernelProducesReferenceResults) {
  ASSERT_NO_THROW(RunGemvKernel());
}