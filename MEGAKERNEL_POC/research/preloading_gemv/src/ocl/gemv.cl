// Tiled GEMV: each work-group is partitioned into several independent row lanes.
// Work-items in the same lane collaboratively tile across one row's columns,
// then perform a lane-local tree reduction to produce that row's output value.
//
// Launch configuration (host side):
//   global = (ceil(rowCount / ROWS_PER_GROUP) * WG_SIZE)
//   local  = (WG_SIZE) -- must be divisible by ROWS_PER_GROUP
//   arg 5  = __local float[WG_SIZE] -- scratch reduction buffer
#define ROWS_PER_GROUP 4u

__kernel void gemv(__global const float* matrix,
                   __global const float* vector,
                   __global float* result,
                   const uint rowCount,
                   const uint columnCount,
                   __local float* localSums) {
    const uint lid = get_local_id(0);
    const uint lsize = get_local_size(0);
    const uint laneSize = lsize / ROWS_PER_GROUP;
    const uint rowInGroup = lid / laneSize;
    const uint laneLid = lid % laneSize;
    const uint row = get_group_id(0) * ROWS_PER_GROUP + rowInGroup;
    const bool rowIsValid = row < rowCount;

    // Phase 1 – tiled accumulation: each lane strides across columns by laneSize
    // so a full row is covered without overlap inside the lane.
    float acc = 0.0f;
    if (rowIsValid) {
        const uint rowOffset = row * columnCount;
        for (uint col = laneLid; col < columnCount; col += laneSize) {
            acc += matrix[rowOffset + col] * vector[col];
        }
    }
    localSums[lid] = acc;
    barrier(CLK_LOCAL_MEM_FENCE);

    // Phase 2 – tree reduction inside each row lane.
    for (uint stride = laneSize >> 1; stride > 0; stride >>= 1) {
        if (laneLid < stride) {
            localSums[lid] += localSums[lid + stride];
        }
        barrier(CLK_LOCAL_MEM_FENCE);
    }

    // The first work-item in each lane writes the final dot-product.
    if (rowIsValid && laneLid == 0) {
        result[row] = localSums[lid];
    }
}