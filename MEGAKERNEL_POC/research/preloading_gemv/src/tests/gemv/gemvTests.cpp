#include "../../../../common/utils.h"
#include "../testCommon/gemvBenchmark.h"

namespace {

const std::string KERNEL_PATH =
    std::string(OPENCL_KERNEL_SOURCE_PATH) +
    "../tests/gemv/ocl/";

class PreloadingTest : public ocltest::GemvTestFixture {
 public:
  void RunGemvBenchmark(int rows, int columns, int rowsPerBlock) {
    std::vector<float> matrix = utils::createRandomBuffer(rows * columns, 0);
    std::vector<float> vector = utils::createRandomBuffer(columns, 1);
    const std::vector<ocltest::GemvShape> shapes = {
        {static_cast<size_t>(rows), static_cast<size_t>(columns),
         static_cast<size_t>(rowsPerBlock)}};

    const ocltest::GemvBenchmarkResult gemvLatency =
        benchmarkOpenClGemvChain({matrix}, vector, shapes, KERNEL_PATH);

    const ocltest::GemvBenchmarkResult dnnlLatency =
        benchmarkDnnlGemvChain({matrix}, vector, shapes);

    gemvLatency.profileResult.print("GEMV OpenCL kernel");
    dnnlLatency.profileResult.print("GEMV oneDNN kernel");

    std::cout << "Speedup: "
              << dnnlLatency.profileResult.averageUs /
                     gemvLatency.profileResult.averageUs
              << "x\n";

    ASSERT_EQ(gemvLatency.output.size(), dnnlLatency.output.size())
        << "Output size mismatch between GEMV and oneDNN results";
    for (size_t i = 0; i < rows; ++i) {
      ASSERT_NEAR(gemvLatency.output[i], dnnlLatency.output[i],
                  ocltest::ABS_ERROR)
          << "GEMV result mismatch at idx " << i;
    }
  }
};

// NOTE: rowsPerBlock is a meta-parameter of opt gemv kernel,
// that should be tuned for each matrix size and hw - e.g. with auto-tuning.
#define RUN_GEMV_BENCHMARK(rows, columns, rowsPerBlock) \
  TEST_F(PreloadingTest, Gemv##rows##x##columns) {      \
    RunGemvBenchmark(rows, columns, rowsPerBlock);      \
  }

RUN_GEMV_BENCHMARK(2048, 1024, 32)
RUN_GEMV_BENCHMARK(1024, 2048, 32)
RUN_GEMV_BENCHMARK(1024, 3072, 32)
RUN_GEMV_BENCHMARK(3072, 1024, 32)

}  // namespace