// Tiled GEMV: each work-group is partitioned into several independent row lanes.
// Work-items in the same lane collaboratively tile across one row's columns,
// then perform a lane-local tree reduction to produce that row's output value.
//
// Launch configuration (host side):
//   global = (ceil(rowCount / ROWS_PER_GROUP) * WG_SIZE)
//   local  = (WG_SIZE) -- must be divisible by ROWS_PER_GROUP
//   arg 5  = __local float[WG_SIZE] -- scratch reduction buffer
#define ROWS_PER_GROUP 1u

__attribute__((intel_reqd_sub_group_size(32)))
__kernel void gemv(__global const float* restrict matrix,
                   __global const float* restrict vector,
                   __global float* restrict result,
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
        const uint vecLimit = columnCount & ~3u;
        uint col = laneLid << 2;
        float acc1 = 0.0f;

        for (; col + (laneSize << 2) < vecLimit; col += laneSize << 3) {
            const float4 m = vload4(0, matrix + rowOffset + col);
            const float4 v = vload4(0, vector + col);
            const uint col1 = col + (laneSize << 2);
            const float4 m1 = vload4(0, matrix + rowOffset + col1);
            const float4 v1 = vload4(0, vector + col1);
            acc += dot(m, v);
            acc1 += dot(m1, v1);
        }

        acc += acc1;

        for (; col < vecLimit; col += laneSize << 2) {
            const float4 m = vload4(0, matrix + rowOffset + col);
            const float4 v = vload4(0, vector + col);
            acc += dot(m, v);
        }

        for (uint tail = vecLimit + laneLid; tail < columnCount; tail += laneSize) {
            acc += matrix[rowOffset + tail] * vector[tail];
        }
    }
    const float reduced = sub_group_reduce_add(acc);

    // The first work-item in each lane writes the final dot-product.
    if (rowIsValid && laneLid == 0) {
        result[row] = reduced;
    }
}