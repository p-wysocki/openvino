// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//
// Placeholder MegaKernel decode kernel.
// Fills all three output buffers (hidden_states_out, present_key, present_val)
// with zeros. Replace this file with the actual fused-layer implementation.
//
// Inputs (17):
//   0  hidden_states    [B, S, hidden_size]          f32
//   1  position_ids     [B, S]                       i64
//   2  beam_idx         [B]                          i32
//   3  past_key         [L, B, num_kv_heads, S, Hd]  f32
//   4  past_val         [L, B, num_kv_heads, S, Hd]  f32
//   5  q_proj_w         [L, 2*H, H]                  f16
//   6  k_proj_w         [L, H, H]                    f16
//   7  v_proj_w         [L, H, H]                    f16
//   8  o_proj_w         [L, H, 2*H]                  f16
//   9  gate_proj_w      [L, I, H]                    f16
//  10  up_proj_w        [L, I, H]                    f16
//  11  down_proj_w      [L, H, I]                    f16
//  12  input_ln_w       [L, H]                       f16
//  13  post_attn_ln_w   [L, H]                       f16
//  14  q_norm_w         [L, Hd]                      f16
//  15  k_norm_w         [L, Hd]                      f16
//  16  rope_inv_freq    [64]                          f16
//
// Outputs (3):
//   0  hidden_states_out  [B, S, H]                  f32
//   1  present_key        [L, B, num_kv_heads, S+1, Hd] f32
//   2  present_val        same shape as present_key   f32
//
// Dispatch: 2-D
//   get_global_id(0) — linear index into hidden_states_out
//   get_global_id(1) — linear index into present_key / present_val

KERNEL(megakernel_decode_zero)(
    OPTIONAL_SHAPE_INFO_ARG
    // --- inputs ---
    const __global float*  restrict hidden_states,    // input  0
    const __global long*   restrict position_ids,     // input  1
    const __global int*    restrict beam_idx,         // input  2
    const __global float*  restrict past_key,         // input  3
    const __global float*  restrict past_val,         // input  4
    const __global half*   restrict q_proj_w,         // input  5
    const __global half*   restrict k_proj_w,         // input  6
    const __global half*   restrict v_proj_w,         // input  7
    const __global half*   restrict o_proj_w,         // input  8
    const __global half*   restrict gate_proj_w,      // input  9
    const __global half*   restrict up_proj_w,        // input 10
    const __global half*   restrict down_proj_w,      // input 11
    const __global half*   restrict input_ln_w,       // input 12
    const __global half*   restrict post_attn_ln_w,   // input 13
    const __global half*   restrict q_norm_w,         // input 14
    const __global half*   restrict k_norm_w,         // input 15
    const __global half*   restrict rope_inv_freq,    // input 16
    // --- outputs ---
    __global float* restrict out_hidden,              // output 0 : hidden_states_out (f32)
    __global half*  restrict out_key,                 // output 1 : present_key  (f16)
    __global half*  restrict out_val                  // output 2 : present_val  (f16)
)
{
    // Dim 0 zeros hidden_states_out
    const uint gid0 = get_global_id(0);
    out_hidden[gid0] = 0.0f;

    // Dim 1 zeros both present_key and present_val (same size)
    const uint gid1 = get_global_id(1);
    out_key[gid1] = (half)0.0f;
    out_val[gid1] = (half)0.0f;

    // Debug: confirm the kernel is actually dispatched (prints once per launch).
    if (gid0 == 0 && gid1 == 0) {
        printf("[MegaKernelDecode OCL] kernel executing — global_size=(%u, %u)\n",
               (uint)get_global_size(0), (uint)get_global_size(1));
    }
}
