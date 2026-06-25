__kernel void gemv(__global const float* matrix,
                   __global const float* vector,
                   __global float* result,
                   const uint row_count,
                   const uint column_count) {
    const uint row = get_global_id(0);
    if (row >= row_count) {
        return;
    }

    float accumulator = 0.0f;
    const uint row_offset = row * column_count;
    for (uint column = 0; column < column_count; ++column) {
        accumulator += matrix[row_offset + column] * vector[column];
    }

    result[row] = accumulator;
}