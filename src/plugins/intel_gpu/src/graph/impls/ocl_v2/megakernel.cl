// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//
// Fused MegaKernel decode kernel for Qwen3-0.6B.
//
// This single OpenCL kernel runs the ENTIRE decoder forward pass for ONE token
// (the decode step) across all transformer layers.  It fuses the kernels that
// OpenVINO normally launches per layer (RMSNorm, FullyConnected q/k/v/o/gate/up/
// down, RoPE, KV-cache append and Scaled-Dot-Product-Attention) into one launch.
//
// It is dispatched as a SINGLE work-group (global == local == MEGA_LWS).  All
// intermediate tensors for the single token live in __local memory; barriers
// synchronise the cooperating work-items between dependent stages.
//
// Inputs (17):
//   0  hidden_states    [B, S, H=1024]                 f16   (token embedding / residual)
//   1  position_ids     [B, S]                          i64
//   2  beam_idx         [B]                             i32   (unused for B==1 greedy)
//   3  past_key         [L, B, Kh=8, S_past, Hd=128]    f16
//   4  past_val         [L, B, Kh=8, S_past, Hd=128]    f16
//   5  q_proj_w         [L, 2048, 1024]                 f16
//   6  k_proj_w         [L, 1024, 1024]                 f16
//   7  v_proj_w         [L, 1024, 1024]                 f16
//   8  o_proj_w         [L, 1024, 2048]                 f16
//   9  gate_proj_w      [L, 3072, 1024]                 f16
//  10  up_proj_w        [L, 3072, 1024]                 f16
//  11  down_proj_w      [L, 1024, 3072]                 f16
//  12  input_ln_w       [L, 1024]                       f16
//  13  post_attn_ln_w   [L, 1024]                       f16
//  14  q_norm_w         [L, 128]                        f16
//  15  k_norm_w         [L, 128]                        f16
//  16  rope_inv_freq    [1, 64, 1]                      f16
//
// Outputs (3):
//   0  hidden_states_out  [B, S, H]                     f32
//   1  present_key        [L, B, Kh, S_past+1, Hd]      f16
//   2  present_val        [L, B, Kh, S_past+1, Hd]      f16
//
// Dispatch: 1-D, single work-group of MEGA_LWS work-items.

#include "include/batch_headers/fetch_data.cl"

#define MEGA_LWS            256
#define MEGA_L              (MEGAKERNEL_NUM_LAYERS)         // 28
#define MEGA_H              (MEGAKERNEL_HIDDEN_SIZE)        // 1024
#define MEGA_KVH            (MEGAKERNEL_NUM_KV_HEADS)       // 8
#define MEGA_HD             (MEGAKERNEL_HEAD_DIM)           // 128
#define MEGA_NH             (MEGAKERNEL_NUM_HEADS)          // 16
#define MEGA_IM             (MEGAKERNEL_INTERMEDIATE_SIZE)  // 3072
#define MEGA_EPS            (MEGAKERNEL_RMS_EPS)            // 1e-6
#define MEGA_QDIM           (MEGA_NH * MEGA_HD)             // 2048
#define MEGA_KVDIM          (MEGA_KVH * MEGA_HD)            // 1024
#define MEGA_HALF_HD        (MEGA_HD / 2)                   // 64
#define MEGA_GQA_GROUP      (MEGA_NH / MEGA_KVH)            // 2

// ---------------------------------------------------------------------------
// Local-memory reduction: returns the sum of partial[0..MEGA_LWS) to ALL lanes.
// Uses scratch[] as workspace.  Caller must barrier before reusing scratch.
// ---------------------------------------------------------------------------
inline float mega_block_reduce_sum(float partial, __local float* scratch, uint lid) {
    scratch[lid] = partial;
    barrier(CLK_LOCAL_MEM_FENCE);
    for (uint stride = MEGA_LWS >> 1; stride > 0; stride >>= 1) {
        if (lid < stride)
            scratch[lid] += scratch[lid + stride];
        barrier(CLK_LOCAL_MEM_FENCE);
    }
    float total = scratch[0];
    barrier(CLK_LOCAL_MEM_FENCE);
    return total;
}

// ---------------------------------------------------------------------------
// RMSNorm over a vector of length N (all lanes cooperate).
//   out[i] = in[i] * rsqrt(mean(in^2)+eps) * w[i]
// in and out may alias.  weight is f16.
// ---------------------------------------------------------------------------
inline void mega_rmsnorm(const __local float* in,
                         __local float* out,
                         const __global half* w,
                         uint N,
                         __local float* scratch,
                         uint lid) {
    float ss = 0.0f;
    for (uint i = lid; i < N; i += MEGA_LWS) {
        float v = in[i];
        ss += v * v;
    }
    float total = mega_block_reduce_sum(ss, scratch, lid);
    float inv = rsqrt(total / (float)N + MEGA_EPS);
    for (uint i = lid; i < N; i += MEGA_LWS) {
        out[i] = in[i] * inv * convert_float(w[i]);
    }
    barrier(CLK_LOCAL_MEM_FENCE);
}

// ---------------------------------------------------------------------------
// Dense FullyConnected:  out[n] = sum_k in[k] * W[n*IN + k]
// W is row-major [OUT, IN] (f16), base already points to the layer slice.
// ---------------------------------------------------------------------------
inline void mega_matmul(const __local float* in,
                        __local float* out,
                        const __global half* W,
                        uint OUT, uint IN,
                        uint lid) {
    for (uint n = lid; n < OUT; n += MEGA_LWS) {
        const __global half* wrow = W + (ulong)n * IN;
        float acc = 0.0f;
        for (uint k = 0; k < IN; ++k)
            acc += in[k] * convert_float(wrow[k]);
        out[n] = acc;
    }
    barrier(CLK_LOCAL_MEM_FENCE);
}

KERNEL(megakernel_zero)(
    OPTIONAL_SHAPE_INFO_ARG
    // --- inputs ---
    const __global half*   restrict hidden_states,    // input  0
    const __global long*   restrict position_ids,     // input  1
    const __global int*    restrict beam_idx,         // input  2
    const __global half*   restrict past_key,         // input  3
    const __global half*   restrict past_val,         // input  4
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
    const uint lid = get_local_id(0);

    // KV-cache sequence lengths (runtime, from shape-info).
    const uint S_past  = (uint)INPUT3_SIZE_Y;     // tokens already in the cache
    const uint S_total = (uint)OUTPUT1_SIZE_Y;    // past + current
    const uint S_new   = S_total - S_past;        // tokens processed by THIS call
                                                  // (1 for decode, >1 for prefill)

    // --- local working buffers (a SINGLE token, independent of S_new) ---
    __local float h[MEGA_H];        // residual stream
    __local float xn[MEGA_H];       // normed input / temp
    __local float q[MEGA_QDIM];     // query (later reused for attention output)
    __local float kbuf[MEGA_KVDIM]; // key  / generic 1024-wide temp
    __local float vbuf[MEGA_KVDIM]; // value
    __local float gate[MEGA_IM];    // gate (later reused for MLP activation)
    __local float scratch[MEGA_LWS];

    // Initialise the per-token residual stream in the f32 out_hidden buffer from
    // the token embeddings (input 0).  out_hidden has logical shape [B, S_new, H];
    // for B==1 token t / channel c maps to the flat offset t*H + c regardless of
    // whether the bfyx layout puts H in the x or y dimension (no padding here).
    for (uint i = lid; i < S_new * MEGA_H; i += MEGA_LWS)
        out_hidden[i] = convert_float(hidden_states[i]);
    barrier(CLK_LOCAL_MEM_FENCE | CLK_GLOBAL_MEM_FENCE);

    const float attn_scale = rsqrt((float)MEGA_HD);

    for (uint L = 0; L < MEGA_L; ++L) {
        const ulong qb   = (ulong)L * MEGA_QDIM  * MEGA_H;    // q_proj   [2048,1024]
        const ulong kvb  = (ulong)L * MEGA_KVDIM * MEGA_H;    // k/v_proj [1024,1024]
        const ulong ob   = (ulong)L * MEGA_H     * MEGA_QDIM; // o_proj   [1024,2048]
        const ulong gub  = (ulong)L * MEGA_IM    * MEGA_H;    // gate/up  [3072,1024]
        const ulong db   = (ulong)L * MEGA_H     * MEGA_IM;   // down     [1024,3072]
        const __global half* iln = input_ln_w     + (ulong)L * MEGA_H;
        const __global half* pln = post_attn_ln_w + (ulong)L * MEGA_H;
        const __global half* qnw = q_norm_w        + (ulong)L * MEGA_HD;
        const __global half* knw = k_norm_w        + (ulong)L * MEGA_HD;

        // Copy the past KV cache [0, S_past) into the present cache ONCE per layer.
        // Each new token writes its own slot (S_past + t) inside the token loop.
        for (uint idx = lid; idx < (uint)(MEGA_KVH * S_past * MEGA_HD); idx += MEGA_LWS) {
            uint d  = idx % MEGA_HD;
            uint s  = (idx / MEGA_HD) % S_past;
            uint kh = idx / (MEGA_HD * S_past);
            out_key[OUTPUT1_GET_INDEX(L, 0, kh, s, d)] = past_key[INPUT3_GET_INDEX(L, 0, kh, s, d)];
            out_val[OUTPUT2_GET_INDEX(L, 0, kh, s, d)] = past_val[INPUT4_GET_INDEX(L, 0, kh, s, d)];
        }
        barrier(CLK_LOCAL_MEM_FENCE | CLK_GLOBAL_MEM_FENCE);

        // Process the new tokens sequentially.  Because token t writes its KV to
        // the cache before token t+1 attends, causal attention is satisfied
        // without any extra scratch.
        for (uint t = 0; t < S_new; ++t) {
            const uint cache_idx = S_past + t;          // this token's cache slot
            const uint S_attend  = cache_idx + 1;       // causal window [0, cache_idx]
            const int  pos       = (int)position_ids[t];

            // load this token's residual from the global stream
            for (uint i = lid; i < MEGA_H; i += MEGA_LWS)
                h[i] = out_hidden[(ulong)t * MEGA_H + i];
            barrier(CLK_LOCAL_MEM_FENCE);

            // 1) input layernorm: xn = RMSNorm(h)
            mega_rmsnorm(h, xn, iln, MEGA_H, scratch, lid);

            // 2) q/k/v projections
            mega_matmul(xn, q,    q_proj_w + qb,  MEGA_QDIM,  MEGA_H, lid);
            mega_matmul(xn, kbuf, k_proj_w + kvb, MEGA_KVDIM, MEGA_H, lid);
            mega_matmul(xn, vbuf, v_proj_w + kvb, MEGA_KVDIM, MEGA_H, lid);

            // 3) per-head q_norm / k_norm (RMSNorm over head_dim), one lane per head
            if (lid < MEGA_NH) {
                __local float* qh = q + (uint)lid * MEGA_HD;
                float ss = 0.0f;
                for (uint d = 0; d < MEGA_HD; ++d) ss += qh[d] * qh[d];
                float inv = rsqrt(ss / (float)MEGA_HD + MEGA_EPS);
                for (uint d = 0; d < MEGA_HD; ++d) qh[d] = qh[d] * inv * convert_float(qnw[d]);
            }
            if (lid < MEGA_KVH) {
                __local float* kh = kbuf + (uint)lid * MEGA_HD;
                float ss = 0.0f;
                for (uint d = 0; d < MEGA_HD; ++d) ss += kh[d] * kh[d];
                float inv = rsqrt(ss / (float)MEGA_HD + MEGA_EPS);
                for (uint d = 0; d < MEGA_HD; ++d) kh[d] = kh[d] * inv * convert_float(knw[d]);
            }
            barrier(CLK_LOCAL_MEM_FENCE);

            // 4) RoPE (rotate-half, non-interleaved) on q (16 heads) and k (8 heads)
            if (lid < MEGA_NH) {
                __local float* qh = q + (uint)lid * MEGA_HD;
                for (uint j = 0; j < MEGA_HALF_HD; ++j) {
                    float angle = (float)pos * convert_float(rope_inv_freq[j]);
                    float c = native_cos(angle), s = native_sin(angle);
                    float x0 = qh[j], x1 = qh[j + MEGA_HALF_HD];
                    qh[j]                = x0 * c - x1 * s;
                    qh[j + MEGA_HALF_HD] = x1 * c + x0 * s;
                }
            }
            if (lid < MEGA_KVH) {
                __local float* kh = kbuf + (uint)lid * MEGA_HD;
                for (uint j = 0; j < MEGA_HALF_HD; ++j) {
                    float angle = (float)pos * convert_float(rope_inv_freq[j]);
                    float c = native_cos(angle), s = native_sin(angle);
                    float x0 = kh[j], x1 = kh[j + MEGA_HALF_HD];
                    kh[j]                = x0 * c - x1 * s;
                    kh[j + MEGA_HALF_HD] = x1 * c + x0 * s;
                }
            }
            barrier(CLK_LOCAL_MEM_FENCE);

            // 5) write this token's KV into its cache slot
            for (uint idx = lid; idx < (uint)(MEGA_KVH * MEGA_HD); idx += MEGA_LWS) {
                uint d  = idx % MEGA_HD;
                uint kh = idx / MEGA_HD;
                out_key[OUTPUT1_GET_INDEX(L, 0, kh, cache_idx, d)] = convert_half(kbuf[kh * MEGA_HD + d]);
                out_val[OUTPUT2_GET_INDEX(L, 0, kh, cache_idx, d)] = convert_half(vbuf[kh * MEGA_HD + d]);
            }
            barrier(CLK_LOCAL_MEM_FENCE | CLK_GLOBAL_MEM_FENCE);

            // 6) causal SDPA per query head (one lane per head), online softmax.
            //    Attends over [0, cache_idx]; preceding new tokens have already
            //    written their KV slots above.
            if (lid < MEGA_NH) {
                const uint hh  = lid;
                const uint kvh = hh / MEGA_GQA_GROUP;
                __local float* qh = q + hh * MEGA_HD;
                float acc[MEGA_HD];
                for (uint d = 0; d < MEGA_HD; ++d) acc[d] = 0.0f;
                float m = -INFINITY, lsum = 0.0f;
                for (uint s = 0; s < S_attend; ++s) {
                    float sc = 0.0f;
                    for (uint d = 0; d < MEGA_HD; ++d)
                        sc += qh[d] * convert_float(out_key[OUTPUT1_GET_INDEX(L, 0, kvh, s, d)]);
                    sc *= attn_scale;
                    float m_new = fmax(m, sc);
                    float corr  = native_exp(m - m_new);
                    float p     = native_exp(sc - m_new);
                    lsum = lsum * corr + p;
                    for (uint d = 0; d < MEGA_HD; ++d)
                        acc[d] = acc[d] * corr + p * convert_float(out_val[OUTPUT2_GET_INDEX(L, 0, kvh, s, d)]);
                    m = m_new;
                }
                float inv_l = 1.0f / lsum;
                for (uint d = 0; d < MEGA_HD; ++d)
                    qh[d] = acc[d] * inv_l;
            }
            barrier(CLK_LOCAL_MEM_FENCE);

            // 7) o_proj and residual add: h += o_proj(attn)
            mega_matmul(q, kbuf, o_proj_w + ob, MEGA_H, MEGA_QDIM, lid);
            for (uint i = lid; i < MEGA_H; i += MEGA_LWS)
                h[i] += kbuf[i];
            barrier(CLK_LOCAL_MEM_FENCE);

            // 8) post-attention layernorm
            mega_rmsnorm(h, xn, pln, MEGA_H, scratch, lid);

            // 9) MLP: down( silu(gate(xn)) * up(xn) )
            mega_matmul(xn, gate, gate_proj_w + gub, MEGA_IM, MEGA_H, lid);  // gate raw
            for (uint n = lid; n < MEGA_IM; n += MEGA_LWS) {
                const __global half* urow = up_proj_w + gub + (ulong)n * MEGA_H;
                float up = 0.0f;
                for (uint k = 0; k < MEGA_H; ++k)
                    up += xn[k] * convert_float(urow[k]);
                float g = gate[n];
                float silu = g / (1.0f + native_exp(-g));
                gate[n] = silu * up;   // MLP activation, overwrites gate
            }
            barrier(CLK_LOCAL_MEM_FENCE);

            mega_matmul(gate, kbuf, down_proj_w + db, MEGA_H, MEGA_IM, lid);
            for (uint i = lid; i < MEGA_H; i += MEGA_LWS)
                h[i] += kbuf[i];
            barrier(CLK_LOCAL_MEM_FENCE);

            // store this token's updated residual back to the global stream
            for (uint i = lid; i < MEGA_H; i += MEGA_LWS)
                out_hidden[(ulong)t * MEGA_H + i] = h[i];
            barrier(CLK_LOCAL_MEM_FENCE | CLK_GLOBAL_MEM_FENCE);
        }
    }
    // out_hidden already holds the final hidden state for every token.
}
