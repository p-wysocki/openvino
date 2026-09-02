#pragma once
#include <CL/cl.h>

#include "../iMegakernelRuntime.h"

namespace mk {
class Qwen06BConstantParams : public IConstantParams {
public:
    void* q_proj_w;
    void* k_proj_w;
    void* v_proj_w;
    void* o_proj_w;
    void* gate_proj_w;
    void* up_proj_w;
    void* down_proj_w;
    void* input_ln_w;
    void* post_attn_ln_w;
    void* q_norm_w;
    void* k_norm_w;
    void* rope_inv_freq;
};

class Qwen06BRuntimeParams : public IRuntimeParams {
public:
    void* hidden_states;
    void* position_ids;
    void* hidden_states_out;
    int newTokens;

    // Two-model PoC: past KV cache handed over from the separate prefill model.
    // Ignored unless import_past_len > 0, in which case the runtime overwrites the
    // first import_past_len tokens of its internal cache before decoding. Each
    // past_key/past_value entry points at one layer's f16 cache laid out as
    // [num_kv_heads, stride, head_dim]; only the leading import_past_len tokens of
    // every head are valid.
    const void* const* past_key = nullptr;
    const void* const* past_value = nullptr;
    const int* past_key_stride = nullptr;
    const int* past_value_stride = nullptr;
    int import_past_len = 0;
};

class Qwen06BPlatformParams : public IPlatformParams {
public:
    cl_device_id deviceId;
    cl_context context;
    cl_command_queue stream;
};

using ConstantParamsImpl = Qwen06BConstantParams;
using RuntimeParamsImpl = Qwen06BRuntimeParams;
using PlatformParamsImpl = Qwen06BPlatformParams;

}  // namespace mk