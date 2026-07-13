// Tiled GEMV specialized for 32-wide subgroups.
// Each subgroup computes four rows so vector loads are reused across dot products.
//
// Launch configuration (host side):
//   global = (ceil(rowCount / ROWS_PER_GROUP) * WG_SIZE)
//   local  = (WG_SIZE)
#pragma OPENCL EXTENSION cl_khr_fp16 : enable

#define ROWS_PER_SUBGROUP 4u
#define ROWS_PER_GROUP 16u
#define WG_SIZE 128u
#define VECTOR_WIDTH 4u

// Each block handles ROWS_PER_GROUP rows, and each subgroup handles ROWS_PER_SUBGROUP rows. 
// Each subgroup has WG_SIZE / 4 threads, and each thread handles 4 columns of the matrix.
__attribute__((reqd_work_group_size(WG_SIZE, 1, 1)))
__attribute__((intel_reqd_sub_group_size(32)))
__attribute__((vec_type_hint(float4)))
__kernel void gemv(__global const half* restrict matrix,
                   __global const half* restrict vector,
                   __global half* restrict result,
                   const uint rowCount,
                   const uint columnCount) {
    const uint laneLid = get_sub_group_local_id();
    const uint rowBase = mad24(get_group_id(0), ROWS_PER_GROUP,
                               get_sub_group_id() * ROWS_PER_SUBGROUP);
    const uint row0 = rowBase;
    const uint row1 = rowBase + 1u;
    const uint row2 = rowBase + 2u;
    const uint row3 = rowBase + 3u;
    const bool row0IsValid = row0 < rowCount;
    const bool row1IsValid = row1 < rowCount;
    const bool row2IsValid = row2 < rowCount;
    const bool row3IsValid = row3 < rowCount;

    const uint row0Offset = row0 * columnCount;
    const uint row1Offset = row1 * columnCount;
    const uint row2Offset = row2 * columnCount;
    const uint row3Offset = row3 * columnCount;

    float acc0 = 0.0f;
    float acc1 = 0.0f;
    float acc2 = 0.0f;
    float acc3 = 0.0f;
    float acc0_1 = 0.0f;
    float acc1_1 = 0.0f;
    float acc2_1 = 0.0f;
    float acc3_1 = 0.0f;
    float acc0_2 = 0.0f;
    float acc1_2 = 0.0f;
    float acc2_2 = 0.0f;
    float acc3_2 = 0.0f;
    float acc0_3 = 0.0f;
    float acc1_3 = 0.0f;
    float acc2_3 = 0.0f;
    float acc3_3 = 0.0f;

    const uint vecLimit = columnCount & ~3u;
    uint col = laneLid << 2;

    #pragma unroll
    for (; col + 384u < vecLimit; col += 512u) {
        const float4 v = convert_float4(vload4(0, vector + col));
        const uint col1 = col + 128u;
        const float4 v1 = convert_float4(vload4(0, vector + col1));
        const uint col2 = col + 256u;
        const float4 v2 = convert_float4(vload4(0, vector + col2));
        const uint col3 = col + 384u;
        const float4 v3 = convert_float4(vload4(0, vector + col3));

        if (row0IsValid) {
            acc0 += dot(convert_float4(vload4(0, matrix + row0Offset + col)), v);
            acc0_1 += dot(convert_float4(vload4(0, matrix + row0Offset + col1)), v1);
            acc0_2 += dot(convert_float4(vload4(0, matrix + row0Offset + col2)), v2);
            acc0_3 += dot(convert_float4(vload4(0, matrix + row0Offset + col3)), v3);
        }
        if (row1IsValid) {
            acc1 += dot(convert_float4(vload4(0, matrix + row1Offset + col)), v);
            acc1_1 += dot(convert_float4(vload4(0, matrix + row1Offset + col1)), v1);
            acc1_2 += dot(convert_float4(vload4(0, matrix + row1Offset + col2)), v2);
            acc1_3 += dot(convert_float4(vload4(0, matrix + row1Offset + col3)), v3);
        }
        if (row2IsValid) {
            acc2 += dot(convert_float4(vload4(0, matrix + row2Offset + col)), v);
            acc2_1 += dot(convert_float4(vload4(0, matrix + row2Offset + col1)), v1);
            acc2_2 += dot(convert_float4(vload4(0, matrix + row2Offset + col2)), v2);
            acc2_3 += dot(convert_float4(vload4(0, matrix + row2Offset + col3)), v3);
        }
        if (row3IsValid) {
            acc3 += dot(convert_float4(vload4(0, matrix + row3Offset + col)), v);
            acc3_1 += dot(convert_float4(vload4(0, matrix + row3Offset + col1)), v1);
            acc3_2 += dot(convert_float4(vload4(0, matrix + row3Offset + col2)), v2);
            acc3_3 += dot(convert_float4(vload4(0, matrix + row3Offset + col3)), v3);
        }
    }

    acc0 += acc0_1 + acc0_2 + acc0_3;
    acc1 += acc1_1 + acc1_2 + acc1_3;
    acc2 += acc2_1 + acc2_2 + acc2_3;
    acc3 += acc3_1 + acc3_2 + acc3_3;

    const float reduced0 = sub_group_reduce_add(acc0);
    const float reduced1 = sub_group_reduce_add(acc1);
    const float reduced2 = sub_group_reduce_add(acc2);
    const float reduced3 = sub_group_reduce_add(acc3);

    if (laneLid == 0) {
        if (row0IsValid) {
            result[row0] = convert_half_rte(reduced0);
        }
        if (row1IsValid) {
            result[row1] = convert_half_rte(reduced1);
        }
        if (row2IsValid) {
            result[row2] = convert_half_rte(reduced2);
        }
        if (row3IsValid) {
            result[row3] = convert_half_rte(reduced3);
        }
    }
}