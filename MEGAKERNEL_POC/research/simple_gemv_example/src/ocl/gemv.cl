
__kernel void gemv(__global const float* matrix,
                   __global const float* vector,
                   __global float* result,
                   const uint rowCount,
                   const uint columnCount) {
    const uint row = get_global_id(0);
    if (row >= rowCount) {
        return;
    }

    float accumulator = 0.0f;
    const uint rowOffset = row * columnCount;
    for (uint column = 0; column < columnCount; ++column) {
        accumulator += matrix[rowOffset + column] * vector[column];
    }

    result[row] = accumulator;
}