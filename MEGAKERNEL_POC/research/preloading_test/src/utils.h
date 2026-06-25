#pragma once

#include <cstddef>
#include <random>
#include <vector>

inline std::vector<float> create_random_buffer(size_t element_count, unsigned int seed) {
    std::mt19937 generator(seed);
    std::uniform_real_distribution<float> distribution(-1.0f, 1.0f);

    std::vector<float> buffer(element_count);
    for (float& value : buffer) {
        value = distribution(generator);
    }
    return buffer;
}