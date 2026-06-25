#pragma once

#include <cstddef>
#include <random>
#include <vector>

namespace utils {
// Creates random buffer of floats.
inline std::vector<float> createRandomBuffer(size_t elementCount,
                                             unsigned int seed) {
  std::mt19937 generator(seed);
  std::uniform_real_distribution<float> distribution(-1.0f, 1.0f);

  std::vector<float> buffer(elementCount);
  for (float& value : buffer) {
    value = distribution(generator);
  }
  return buffer;
}

}  // namespace utils