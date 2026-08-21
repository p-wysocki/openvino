#pragma once

namespace mk {
struct Qwen06BWeights {
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

struct Qwen06BInputsOutputs {
    void* hidden_states;
    void* position_ids;
    void* hidden_states_out;
    int newTokens;
};
}  // namespace mk