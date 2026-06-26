// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "primitive.hpp"

namespace cldnn {

struct megakernel_decode : public primitive_base<megakernel_decode> {
    CLDNN_DECLARE_PRIMITIVE(megakernel_decode)

    megakernel_decode() : primitive_base("", {}) {}

    megakernel_decode(const primitive_id& id,
                      const std::vector<input_info>& inputs,
                      int64_t num_layers,
                      int64_t hidden_size,
                      int64_t num_kv_heads,
                      int64_t head_dim)
        : primitive_base(id, inputs, 3 /*num_outputs*/),
          num_layers(num_layers),
          hidden_size(hidden_size),
          num_kv_heads(num_kv_heads),
          head_dim(head_dim) {}

    int64_t num_layers  = 28;
    int64_t hidden_size = 1024;
    int64_t num_kv_heads = 8;
    int64_t head_dim    = 128;

    size_t hash() const override {
        size_t seed = primitive::hash();
        seed = hash_combine(seed, num_layers);
        seed = hash_combine(seed, hidden_size);
        seed = hash_combine(seed, num_kv_heads);
        seed = hash_combine(seed, head_dim);
        return seed;
    }

    bool operator==(const primitive& rhs) const override {
        if (!compare_common_params(rhs))
            return false;
        auto rhs_c = downcast<const megakernel_decode>(rhs);
        return num_layers   == rhs_c.num_layers &&
               hidden_size  == rhs_c.hidden_size &&
               num_kv_heads == rhs_c.num_kv_heads &&
               head_dim     == rhs_c.head_dim;
    }

    void save(BinaryOutputBuffer& ob) const override {
        primitive_base<megakernel_decode>::save(ob);
        ob << num_layers << hidden_size << num_kv_heads << head_dim;
    }

    void load(BinaryInputBuffer& ib) override {
        primitive_base<megakernel_decode>::load(ib);
        ib >> num_layers >> hidden_size >> num_kv_heads >> head_dim;
    }
};

}  // namespace cldnn
