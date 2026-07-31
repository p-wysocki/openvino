#pragma once

#include <cstddef>
#include <string>
#include <vector>

#include "../../../common/benchmarkOpenCLpipeline.h"
#include "../../../common/oclTestFixture.h"

namespace ocltest {

static const std::string KERNEL_SOURCE_PATH =
    std::string(OPENCL_KERNEL_SOURCE_PATH) + "/gemv_opt/";
static constexpr const char* KERNEL_SOURCE_FILE_NAME = "gemvOptKernel.cl";
static constexpr size_t WARMUP_ITERATIONS = 100;
static constexpr size_t BENCHMARK_ITERATIONS = 1000;
static constexpr float ABS_ERROR = 2e-2f;
static constexpr bool CLEAR_CACHE_BEFORE_BENCHMARK = true;

struct GemvShape {
  size_t rowCount;
  size_t columnCount;
  size_t rowsPerBlock;
};

struct GemvBenchmarkResult {
  ProfileResult profileResult;
  std::vector<float> output;
};

class GemvTestFixture : public OclTestFixture {
 protected:
  GemvBenchmarkResult benchmarkOpenClGemvChain(
      const std::vector<std::vector<float>>& matrices,
      const std::vector<float>& input, const std::vector<GemvShape>& shapes,
      const std::string& kernelSourcePath = KERNEL_SOURCE_PATH,
      const std::string& kernelSourceFileName = KERNEL_SOURCE_FILE_NAME,
      size_t warmupIterations = WARMUP_ITERATIONS,
      size_t benchmarkIterations = BENCHMARK_ITERATIONS);

  GemvBenchmarkResult benchmarkDnnlGemvChain(
      const std::vector<std::vector<float>>& matrices,
      const std::vector<float>& input, const std::vector<GemvShape>& shapes,
      size_t warmupIterations = WARMUP_ITERATIONS,
      size_t benchmarkIterations = BENCHMARK_ITERATIONS);
};

}  // namespace ocltest