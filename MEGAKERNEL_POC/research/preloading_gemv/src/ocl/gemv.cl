// Tiled GEMV: one work-group per output row.
// All work-items in the group collaboratively tile across the row's columns,
// then perform an in-group tree reduction to produce the single output value.
//
// Launch configuration (host side):
//   global = (rowCount * WG_SIZE)
//   local  = (WG_SIZE)              -- must be a power of two
//   arg 5  = __local float[WG_SIZE] -- scratch reduction buffer
__kernel void gemv(__global const float* matrix,
                   __global const float* vector,
                   __global float* result,
                   const uint rowCount,
                   const uint columnCount,
                   __local float* localSums) {
    const uint row   = get_group_id(0);
    const uint lid   = get_local_id(0);
    const uint lsize = get_local_size(0);

    if (row >= rowCount) {
        return;
    }

    // Phase 1 – tiled accumulation: each work-item strides across columns
    // by lsize so the full row is covered without overlap.
    float acc = 0.0f;
    const uint rowOffset = row * columnCount;
    for (uint col = lid; col < columnCount; col += lsize) {
        acc += matrix[rowOffset + col] * vector[col];
    }
    localSums[lid] = acc;
    barrier(CLK_LOCAL_MEM_FENCE);

    // Phase 2 – tree reduction inside the work-group.
    for (uint stride = lsize >> 1; stride > 0; stride >>= 1) {
        if (lid < stride) {
            localSums[lid] += localSums[lid + stride];
        }
        barrier(CLK_LOCAL_MEM_FENCE);
    }

    // Work-item 0 writes the final dot-product for this row.
    if (lid == 0) {
        result[row] = localSums[0];
    }
}