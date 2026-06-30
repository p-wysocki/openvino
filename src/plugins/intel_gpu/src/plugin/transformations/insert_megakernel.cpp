// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//
// The pass (MegaKernel-only — handles BOTH prefill and decode):
//  1. Detects the model by counting 56 ReadValue/Assign "past_key_values" pairs.
//  2. Collects per-layer weight Constants (q/k/v/o proj, gate/up/down proj,
//     input_ln, post_attn_ln, q_norm, k_norm) and stacks them along a new dim-0.
//  3. Stacks the 56 ReadValue outputs into two [28,B,8,S,128] tensors via
//     Unsqueeze+Concat (the ReadValue nodes are kept; only wiring changes).
//  4. Creates MegaKernel.
//  5. Rewires model.norm to consume the MegaKernel hidden-state output directly
//     (a single Convert to f16 to match the downstream precision).
//  6. Splits present_key / present_val (outputs 1/2) back to 28 slices and wires
//     each directly to the matching Assign.
//
// The original 28-layer SDPA sub-graph is left with no consumers and therefore
// becomes unreachable from Results/Sinks — it is never compiled, so there is no
// duplicate weight memory and no runtime branching (Select) overhead.  The
// MegaKernel itself decides prefill vs decode at runtime from the token count.

#include "insert_megakernel.hpp"

#include "intel_gpu/op/megakernel.hpp"

#include "openvino/core/graph_util.hpp"
#include "openvino/core/rt_info.hpp"
#include "openvino/op/assign.hpp"
#include "openvino/op/concat.hpp"
#include "openvino/op/constant.hpp"
#include "openvino/op/convert.hpp"
#include "openvino/op/parameter.hpp"
#include "openvino/op/read_value.hpp"
#include "openvino/op/squeeze.hpp"
#include "openvino/op/split.hpp"
#include "openvino/op/unsqueeze.hpp"
#include "openvino/op/gather.hpp"
#include "openvino/op/equal.hpp"
#include "openvino/op/select.hpp"
#include "openvino/op/shape_of.hpp"
#include "openvino/pass/constant_folding.hpp"

#include <algorithm>
#include <cstdlib>
#include <map>
#include <string>
#include <vector>

namespace ov::intel_gpu {

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------
namespace {

constexpr int NUM_LAYERS = 28;

// Return the node with the given friendly name or nullptr.
static std::shared_ptr<ov::Node> find_node(const ov::NodeVector& ops, const std::string& name) {
    for (auto& op : ops) {
        if (op->get_friendly_name() == name)
            return op;
    }
    return nullptr;
}

// Walk through Convert wrappers to the underlying Constant and return its data
// as a flat float16 (f16) buffer re-packaged into a new Constant of the given
// shape.  If the node is already a Constant of dtype f16 it is returned as-is.
static std::shared_ptr<ov::op::v0::Constant> get_f16_constant(std::shared_ptr<ov::Node> node) {
    // Strip Convert wrappers (decompressor pattern) to reach the underlying Constant.
    for (int depth = 0; depth < 10; ++depth) {
        if (ov::is_type<ov::op::v0::Constant>(node))
            break;
        if (!ov::is_type<ov::op::v0::Convert>(node))
            break;
        node = node->get_input_node_shared_ptr(0);
    }
    auto c = ov::as_type_ptr<ov::op::v0::Constant>(node);
    OPENVINO_ASSERT(c != nullptr,
                    "[MegaKernel] expected Constant, got '", node->get_type_name(),
                    "' (name='", node->get_friendly_name(), "')");
    // If already f16, return directly
    if (c->get_element_type() == ov::element::f16)
        return c;
    // Otherwise cast to f16
    auto data_f32 = c->cast_vector<ov::float16>();
    return ov::op::v0::Constant::create(ov::element::f16, c->get_shape(), data_f32);
}

// Stack a vector of per-layer Constants along a new leading dimension.
// Each constant must have the same shape.  Returns new stacked Constant.
static std::shared_ptr<ov::op::v0::Constant> stack_constants(
        const std::vector<std::shared_ptr<ov::op::v0::Constant>>& per_layer,
        const std::string& debug_name) {

    OPENVINO_ASSERT(!per_layer.empty());
    const ov::Shape elem_shape = per_layer[0]->get_shape();
    const ov::element::Type dtype = per_layer[0]->get_element_type();

    // Flatten each layer's data and concatenate
    size_t elem_size = dtype.bitwidth() / 8;
    size_t layer_bytes = ov::shape_size(elem_shape) * elem_size;
    std::vector<uint8_t> buf(per_layer.size() * layer_bytes);

    for (size_t i = 0; i < per_layer.size(); ++i) {
        OPENVINO_ASSERT(per_layer[i]->get_element_type() == dtype,
                        "[MegaKernel] dtype mismatch while stacking ", debug_name);
        const auto* src = static_cast<const uint8_t*>(per_layer[i]->get_data_ptr());
        std::copy(src, src + layer_bytes, buf.data() + i * layer_bytes);
    }

    ov::Shape stacked_shape;
    stacked_shape.push_back(per_layer.size());
    for (auto d : elem_shape)
        stacked_shape.push_back(d);

    auto stacked = ov::op::v0::Constant::create(dtype, stacked_shape, buf.data());
    stacked->set_friendly_name(debug_name);
    return stacked;
}

// Squeeze all size-1 leading dims from a constant (e.g. [1,1,1024]->[1024]).
static std::shared_ptr<ov::op::v0::Constant> squeeze_leading_ones(
        std::shared_ptr<ov::op::v0::Constant> c) {
    ov::Shape s = c->get_shape();
    size_t first_nontrivial = 0;
    while (first_nontrivial < s.size() && s[first_nontrivial] == 1)
        ++first_nontrivial;
    if (first_nontrivial == 0)
        return c;
    ov::Shape new_shape(s.begin() + first_nontrivial, s.end());
    if (new_shape.empty())
        new_shape = {1};
    size_t elem_size = c->get_element_type().bitwidth() / 8;
    size_t nbytes = ov::shape_size(new_shape) * elem_size;
    std::vector<uint8_t> buf(nbytes);
    std::copy(static_cast<const uint8_t*>(c->get_data_ptr()),
              static_cast<const uint8_t*>(c->get_data_ptr()) + nbytes,
              buf.data());
    return ov::op::v0::Constant::create(c->get_element_type(), new_shape, buf.data());
}

// Find the weight constant for a per-layer matmul weight.
// The graph has pattern:  Constant (f16 compressed) -> Convert -> MatMul
// The constant's friendly name is "self.model.layers.<l>.<proj_suffix>"
static std::shared_ptr<ov::op::v0::Constant> get_proj_weight(
        const ov::NodeVector& ops, int layer, const std::string& weight_name_suffix) {
    const std::string expected = "self.model.layers." + std::to_string(layer) + "." + weight_name_suffix;
    auto node = find_node(ops, expected);
    OPENVINO_ASSERT(node != nullptr, "[MegaKernel] Cannot find weight node: ", expected);
    auto c = ov::as_type_ptr<ov::op::v0::Constant>(node);
    OPENVINO_ASSERT(c != nullptr);
    return c;  // already f16 compressed
}

// Find norm weight constant for a given layer and norm name fragment.
// Strategy: find the last Multiply_1 in the RMSNorm sequence whose name
// contains "layers.<l>.<norm_fragment>", then scan both inputs to find the
// one that traces back to a Constant through Convert wrappers (the weight).
// The other input is the normalised activation path.
static std::shared_ptr<ov::op::v0::Constant> get_norm_weight(
        const ov::NodeVector& ops, int layer, const std::string& norm_fragment) {
    const std::string target_frag = "layers." + std::to_string(layer) + "." + norm_fragment;
    for (auto& op : ops) {
        const std::string& n = op->get_friendly_name();
        if (n.find(target_frag) == std::string::npos) continue;
        if (n.find("Multiply_1") == std::string::npos) continue;
        // Try both input ports — weight may be at port 0 or 1 depending on export order.
        for (size_t port = 0; port < op->get_input_size(); ++port) {
            auto candidate = op->get_input_node_shared_ptr(port);
            // Trace through Convert wrappers only to identify a Constant source.
            auto trace = candidate;
            for (int d = 0; d < 10 && !ov::is_type<ov::op::v0::Constant>(trace); ++d) {
                if (!ov::is_type<ov::op::v0::Convert>(trace)) { trace = nullptr; break; }
                trace = trace->get_input_node_shared_ptr(0);
            }
            if (trace && ov::is_type<ov::op::v0::Constant>(trace))
                return get_f16_constant(candidate);
        }
    }
    OPENVINO_THROW("[MegaKernel] Cannot find norm weight for layer ", layer, " fragment '", norm_fragment, "'");
}

// Collect sorted ReadValue ops for key and value caches.
// var_id format: "past_key_values.<l>.key_states" / "...value_states"
// Note: we must check the SUFFIX after the layer number, not naive string search,
// because every var_id contains "key_values" (so both "key" and "value" are present).
static void collect_sorted_rv(const ov::NodeVector& ops,
                               std::vector<std::shared_ptr<ov::op::v6::ReadValue>>& key_rvs,
                               std::vector<std::shared_ptr<ov::op::v6::ReadValue>>& val_rvs) {
    std::map<int, std::shared_ptr<ov::op::v6::ReadValue>> key_map, val_map;
    for (auto& op : ops) {
        auto rv = ov::as_type_ptr<ov::op::v6::ReadValue>(op);
        if (!rv) continue;
        const std::string& vid = rv->get_variable_id();
        if (vid.find("past_key_values") == std::string::npos) continue;
        // parse layer index: "past_key_values.<l>.<type>..."
        auto dot1 = vid.find('.');
        auto dot2 = vid.find('.', dot1 + 1);
        if (dot1 == std::string::npos || dot2 == std::string::npos) continue;
        int layer = std::stoi(vid.substr(dot1 + 1, dot2 - dot1 - 1));
        // suffix is everything after the layer number dot, e.g. "key_states" or "value_states"
        const std::string suffix = vid.substr(dot2 + 1);
        if (suffix.rfind("key", 0) == 0)   // starts_with("key")
            key_map[layer] = rv;
        else
            val_map[layer] = rv;
    }
    key_rvs.clear(); val_rvs.clear();
    for (int l = 0; l < NUM_LAYERS; ++l) {
        OPENVINO_ASSERT(key_map.count(l), "[MegaKernel] missing key ReadValue for layer ", l);
        OPENVINO_ASSERT(val_map.count(l), "[MegaKernel] missing val ReadValue for layer ", l);
        key_rvs.push_back(key_map.at(l));
        val_rvs.push_back(val_map.at(l));
    }
}

// Collect sorted Assign ops for key and value caches.
static void collect_sorted_assign(const ov::NodeVector& ops,
                                   std::vector<std::shared_ptr<ov::op::v6::Assign>>& key_as,
                                   std::vector<std::shared_ptr<ov::op::v6::Assign>>& val_as) {
    std::map<int, std::shared_ptr<ov::op::v6::Assign>> key_map, val_map;
    for (auto& op : ops) {
        auto as = ov::as_type_ptr<ov::op::v6::Assign>(op);
        if (!as) continue;
        const std::string& vid = as->get_variable_id();
        if (vid.find("past_key_values") == std::string::npos) continue;
        auto dot1 = vid.find('.');
        auto dot2 = vid.find('.', dot1 + 1);
        if (dot1 == std::string::npos || dot2 == std::string::npos) continue;
        int layer = std::stoi(vid.substr(dot1 + 1, dot2 - dot1 - 1));
        // suffix is everything after the layer number dot, e.g. "key_states" or "value_states"
        const std::string suffix = vid.substr(dot2 + 1);
        if (suffix.rfind("key", 0) == 0)   // starts_with("key")
            key_map[layer] = as;
        else
            val_map[layer] = as;
    }
    key_as.clear(); val_as.clear();
    for (int l = 0; l < NUM_LAYERS; ++l) {
        key_as.push_back(key_map.at(l));
        val_as.push_back(val_map.at(l));
    }
}

// Build [28, B, num_kv_heads, S_past, head_dim] by stacking 28 ReadValue outputs
// via Unsqueeze(axis=0) + Concat(axis=0).
static ov::Output<ov::Node> build_stacked_kv(
        const std::vector<std::shared_ptr<ov::op::v6::ReadValue>>& rvs) {
    auto axis_const = ov::op::v0::Constant::create(ov::element::i64, ov::Shape{1}, {0});
    ov::OutputVector unsqueezed;
    unsqueezed.reserve(rvs.size());
    for (auto& rv : rvs) {
        auto us = std::make_shared<ov::op::v0::Unsqueeze>(rv->output(0), axis_const);
        unsqueezed.push_back(us->output(0));
    }
    auto cat = std::make_shared<ov::op::v0::Concat>(unsqueezed, 0);
    return cat->output(0);
}

}  // namespace

// ---------------------------------------------------------------------------
// Pass entry point
// ---------------------------------------------------------------------------
bool InsertMegaKernel::run_on_model(const std::shared_ptr<ov::Model>& m) {
    // --- Runtime toggle: skip the fusion when OV_MEGAKERNEL_DISABLE=1 ----------
    // Lets the same plugin binary run the stock baseline (pass off) or the fused
    // megakernel (pass on) without rebuilding, so both can be benchmarked fairly.
    if (const char* off = std::getenv("OV_MEGAKERNEL_DISABLE"); off && off[0] == '1')
        return false;

    const ov::NodeVector ops = m->get_ordered_ops();

    // --- Guard: only apply to Qwen3-0.6B (56 kv-cache ReadValue ops) ----------
    int rv_kv_count = 0;
    for (auto& op : ops) {
        auto rv = ov::as_type_ptr<ov::op::v6::ReadValue>(op);
        if (rv && rv->get_variable_id().find("past_key_values") != std::string::npos)
            ++rv_kv_count;
    }
    if (rv_kv_count != 2 * NUM_LAYERS)
        return false;

    // --- 1. Find boundary nodes --------------------------------------------------
    auto embed_gather = find_node(ops, "__module.model.embed_tokens/ov_ext::embedding/Gather");
    OPENVINO_ASSERT(embed_gather, "[MegaKernel] embed_tokens Gather not found");

    auto last_add = find_node(ops, "__module.model.layers.27/aten::add/Add_1");
    OPENVINO_ASSERT(last_add, "[MegaKernel] layer 27 final Add not found");

    // Parameter nodes
    std::shared_ptr<ov::Node> pos_ids_node = nullptr, beam_idx_node = nullptr;
    for (auto& op : ops) {
        if (!ov::is_type<ov::op::v0::Parameter>(op)) continue;
        const auto& name = op->get_friendly_name();
        if (name == "position_ids")  pos_ids_node  = op;
        if (name == "beam_idx")      beam_idx_node  = op;
    }
    OPENVINO_ASSERT(pos_ids_node  && beam_idx_node, "[MegaKernel] Missing Parameter nodes");

    // --- 2. Stacked projection weights ------------------------------------------
    // Stack per-layer weights into [28, ...] constants used as MegaKernel inputs.
    // We do NOT redirect the original SDPA path's weight consumers: once the SDPA
    // path is disconnected (steps 7/8) it becomes unreachable and is never
    // compiled, so its weight constants are dropped automatically → no duplicate
    // device memory.
    static const std::vector<std::pair<std::string, std::string>> PROJ_WEIGHTS = {
        {"q_proj",    "self_attn.q_proj.weight"},
        {"k_proj",    "self_attn.k_proj.weight"},
        {"v_proj",    "self_attn.v_proj.weight"},
        {"o_proj",    "self_attn.o_proj.weight"},
        {"gate_proj", "mlp.gate_proj.weight"},
        {"up_proj",   "mlp.up_proj.weight"},
        {"down_proj", "mlp.down_proj.weight"},
    };

    std::map<std::string, std::shared_ptr<ov::op::v0::Constant>> stacked_proj;
    for (auto& [tag, suffix] : PROJ_WEIGHTS) {
        std::vector<std::shared_ptr<ov::op::v0::Constant>> per_layer;
        per_layer.reserve(NUM_LAYERS);
        for (int l = 0; l < NUM_LAYERS; ++l)
            per_layer.push_back(get_proj_weight(ops, l, suffix));
        stacked_proj[tag] = stack_constants(per_layer, "megakernel_stacked_" + tag);
    }

    // --- 3. Stacked norm weights -------------------------------------------------
    // Norm weights are tiny (~7MB total) so no need to eliminate originals.
    static const std::vector<std::pair<std::string, std::string>> NORM_WEIGHTS = {
        {"input_ln",      "input_layernorm"},
        {"post_attn_ln",  "post_attention_layernorm"},
        {"q_norm",        "self_attn.q_norm"},
        {"k_norm",        "self_attn.k_norm"},
    };

    std::map<std::string, std::shared_ptr<ov::op::v0::Constant>> stacked_norm;
    for (auto& [tag, fragment] : NORM_WEIGHTS) {
        std::vector<std::shared_ptr<ov::op::v0::Constant>> per_layer;
        per_layer.reserve(NUM_LAYERS);
        for (int l = 0; l < NUM_LAYERS; ++l) {
            auto c = get_norm_weight(ops, l, fragment);
            per_layer.push_back(squeeze_leading_ones(c));
        }
        stacked_norm[tag] = stack_constants(per_layer, "megakernel_stacked_" + tag);
    }

    // --- 4. rope_inv_freq --------------------------------------------------------
    // Find the [1,64,1] inv_freq Constant (64 elements = head_dim/2) that is
    // consumed by the rotary embedding node.
    ov::Output<ov::Node> rope_inv_freq;
    {
        bool found = false;
        for (auto& op : ops) {
            auto c = ov::as_type_ptr<ov::op::v0::Constant>(op);
            if (!c || ov::shape_size(c->get_shape()) != 64) continue;
            for (auto& out : c->outputs()) {
                for (auto& inp : out.get_target_inputs()) {
                    if (inp.get_node()->get_friendly_name().find("rotary") != std::string::npos) {
                        rope_inv_freq = c->output(0);
                        found = true;
                        break;
                    }
                }
                if (found) break;
            }
            if (found) break;
        }
        OPENVINO_ASSERT(found, "[MegaKernel] rope inv_freq not found");
    }

    // Collect sorted Assign ops (their inputs are rewired to the MegaKernel KV).
    std::vector<std::shared_ptr<ov::op::v6::Assign>> key_as, val_as;
    collect_sorted_assign(ops, key_as, val_as);

    // --- 5. Stack KV caches ------------------------------------------------------
    std::vector<std::shared_ptr<ov::op::v6::ReadValue>> key_rvs, val_rvs;
    collect_sorted_rv(ops, key_rvs, val_rvs);
    auto past_key = build_stacked_kv(key_rvs);
    auto past_val = build_stacked_kv(val_rvs);

    // --- 6. Build MegaKernel -----------------------------------------------
    op::MegaKernelAttrs attrs;
    // attrs already has the correct Qwen3-0.6B defaults

    ov::OutputVector mk_inputs = {
        embed_gather->output(0),               // 0  hidden_states
        pos_ids_node->output(0),               // 1  position_ids
        beam_idx_node->output(0),              // 2  beam_idx
        past_key,                              // 3  past_key  [28,B,8,S,128]
        past_val,                              // 4  past_val
        stacked_proj["q_proj"]->output(0),     // 5
        stacked_proj["k_proj"]->output(0),     // 6
        stacked_proj["v_proj"]->output(0),     // 7
        stacked_proj["o_proj"]->output(0),     // 8
        stacked_proj["gate_proj"]->output(0),  // 9
        stacked_proj["up_proj"]->output(0),    // 10
        stacked_proj["down_proj"]->output(0),  // 11
        stacked_norm["input_ln"]->output(0),   // 12
        stacked_norm["post_attn_ln"]->output(0),// 13
        stacked_norm["q_norm"]->output(0),     // 14
        stacked_norm["k_norm"]->output(0),     // 15
        rope_inv_freq,                         // 16
    };

    auto mk_node = std::make_shared<op::MegaKernel>(mk_inputs, attrs);
    mk_node->set_friendly_name("MegaKernel");
    ov::copy_runtime_info(last_add, mk_node);

    // --- 7. Rewire model.norm ← MegaKernel hidden-state output -------------------
    // The downstream model.norm path is lowered to f16 by the GPU precision passes.
    // MegaKernel output 0 is declared f32; insert a single Convert to f16 so the
    // type matches, then redirect every consumer of last_add to it.  This makes the
    // entire 28-layer SDPA path unreachable (and thus never compiled).
    auto mk_hidden_f16 = std::make_shared<ov::op::v0::Convert>(mk_node->output(0), ov::element::f16);
    mk_hidden_f16->set_friendly_name("mk_hidden_f16");
    auto last_add_consumers = last_add->output(0).get_target_inputs();
    for (auto& inp : last_add_consumers)
        inp.replace_source_output(mk_hidden_f16->output(0));

    // --- 8. Rewire Assign ops ← MegaKernel KV outputs ---------------------------
    // Split MegaKernel KV outputs (f16) into 28 per-layer slices, squeeze the
    // leading layer dim, convert to the Assign's element type if needed, and wire
    // directly to each Assign (no runtime branching).
    auto axis_const_split = ov::op::v0::Constant::create(ov::element::i64, ov::Shape{}, {0});
    auto axis_const_sq    = ov::op::v0::Constant::create(ov::element::i64, ov::Shape{1}, {0});

    auto rewire_assigns = [&](ov::Output<ov::Node> stacked_kv,
                               std::vector<std::shared_ptr<ov::op::v6::Assign>>& assigns) {
        const ov::element::Type assign_et = assigns[0]->input_value(0).get_element_type();
        auto split = std::make_shared<ov::op::v1::Split>(stacked_kv, axis_const_split, NUM_LAYERS);
        for (int l = 0; l < NUM_LAYERS; ++l) {
            auto sq = std::make_shared<ov::op::v0::Squeeze>(split->output(l), axis_const_sq);
            ov::Output<ov::Node> kv_branch = sq->output(0);
            if (sq->output(0).get_element_type() != assign_et) {
                auto cvt = std::make_shared<ov::op::v0::Convert>(sq->output(0), assign_et);
                kv_branch = cvt->output(0);
            }
            assigns[l]->input(0).replace_source_output(kv_branch);
        }
    };

    rewire_assigns(mk_node->output(1), key_as);
    rewire_assigns(mk_node->output(2), val_as);

    m->validate_nodes_and_infer_types();

    return true;
}

}  // namespace ov::intel_gpu
