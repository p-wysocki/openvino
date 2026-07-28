#include <cmath>
#include <vector>

#include "../../common/utils.h"
#include "testCommon/gemvBenchmark.h"

namespace {

static const std::vector<ocltest::GemvShape> gemvShapes = {
    {1024, 2048, 32},
    {3072, 1024, 32},
    {1024, 3072, 32},
};

class ChainGemvTests : public ocltest::GemvTestFixture {};

TEST_F(ChainGemvTests, ThreeGemvChain) {
  std::vector<std::vector<float>> matrices(gemvShapes.size());
  for (size_t layer = 0; layer < gemvShapes.size(); ++layer) {
    matrices[layer] = utils::createRandomBuffer(
        gemvShapes[layer].rowCount * gemvShapes[layer].columnCount, layer);
    const float scale =
        1.0f / std::sqrt(static_cast<float>(gemvShapes[layer].columnCount));
    for (float& value : matrices[layer]) {
      value *= scale;
    }
  }
  const std::vector<float> input =
      utils::createRandomBuffer(gemvShapes.front().columnCount, 3);

  const ocltest::GemvBenchmarkResult openClResult =
      benchmarkOpenClGemvChain(matrices, input, gemvShapes);
  const ocltest::GemvBenchmarkResult dnnlResult =
      benchmarkDnnlGemvChain(matrices, input, gemvShapes);

  openClResult.profileResult.print("3-GEMV OpenCL chain");
  dnnlResult.profileResult.print("3-GEMV oneDNN chain");
  std::cout << "Speedup: "
            << dnnlResult.profileResult.averageUs /
                   openClResult.profileResult.averageUs
            << "x\n";

  ASSERT_EQ(openClResult.output.size(), dnnlResult.output.size())
      << "Output size mismatch between OpenCL and oneDNN GEMV chains";
  for (size_t index = 0; index < openClResult.output.size(); ++index) {
    ASSERT_NEAR(openClResult.output[index], dnnlResult.output[index],
                ocltest::ABS_ERROR)
        << "GEMV chain result mismatch at index " << index;
  }
}

}  // namespace