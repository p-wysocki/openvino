// Tiled GEMV specialized for one 32-wide subgroup per output row.
// Each work-group computes one row and uses subgroup reduction for the final sum.
//
// Launch configuration (host side):
//   global = (ceil(rowCount / ROWS_PER_GROUP) * WG_SIZE)
//   local  = (WG_SIZE)
#pragma OPENCL EXTENSION cl_khr_fp16 : enable

#define ROWS_PER_GROUP 1u

__attribute__((reqd_work_group_size(32, 1, 1)))
__attribute__((intel_reqd_sub_group_size(32)))
__attribute__((vec_type_hint(float4)))
__kernel void gemv(__global const half* restrict matrix,
                   __global const half* restrict vector,
                   __global half* restrict result,
                   const uint rowCount,
                   const uint columnCount) {
    const uint laneLid = get_sub_group_local_id();
    const uint row = get_group_id(0);
    const bool rowIsValid = row < rowCount;

    float acc = 0.0f;
    if (rowIsValid) {
        const uint rowOffset = row * columnCount;
        const uint vecLimit = columnCount & ~3u;
        uint col = laneLid << 2;
        float acc1 = 0.0f;
        float acc2 = 0.0f;
        float acc3 = 0.0f;

        #pragma unroll
        for (; col + 384u < vecLimit; col += 512u) {
            const float4 m = convert_float4(vload4(0, matrix + rowOffset + col));
            const float4 v = convert_float4(vload4(0, vector + col));
            const uint col1 = col + 128u;
            const float4 m1 = convert_float4(vload4(0, matrix + rowOffset + col1));
            const float4 v1 = convert_float4(vload4(0, vector + col1));
            const uint col2 = col + 256u;
            const float4 m2 = convert_float4(vload4(0, matrix + rowOffset + col2));
            const float4 v2 = convert_float4(vload4(0, vector + col2));
            const uint col3 = col + 384u;
            const float4 m3 = convert_float4(vload4(0, matrix + rowOffset + col3));
            const float4 v3 = convert_float4(vload4(0, vector + col3));
            acc += dot(m, v);
            acc1 += dot(m1, v1);
            acc2 += dot(m2, v2);
            acc3 += dot(m3, v3);
        }

        acc += acc1 + acc2 + acc3;

        #pragma unroll
        for (; col < vecLimit; col += 128u) {
            const float4 m = convert_float4(vload4(0, matrix + rowOffset + col));
            const float4 v = convert_float4(vload4(0, vector + col));
            acc += dot(m, v);
        }

        for (uint tail = vecLimit + laneLid; tail < columnCount; tail += 32u) {
            acc += (float)matrix[rowOffset + tail] * (float)vector[tail];
        }
    }
    const float reduced = sub_group_reduce_add(acc);

    if (rowIsValid && laneLid == 0) {
        result[row] = convert_half_rte(reduced);
    }
}