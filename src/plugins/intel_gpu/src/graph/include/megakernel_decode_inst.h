// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "intel_gpu/primitives/megakernel_decode.hpp"
#include "primitive_inst.h"

namespace cldnn {

template <>
struct typed_program_node<megakernel_decode> : public typed_program_node_base<megakernel_decode> {
    using parent = typed_program_node_base<megakernel_decode>;
    using parent::parent;

    program_node& input(size_t idx = 0) const { return get_dependency(idx); }

    std::vector<size_t> get_shape_infer_dependencies() const override { return {}; }
};

using megakernel_decode_node = typed_program_node<megakernel_decode>;

template <>
class typed_primitive_inst<megakernel_decode> : public typed_primitive_inst_base<megakernel_decode> {
    using parent = typed_primitive_inst_base<megakernel_decode>;
    using parent::parent;

public:
    template <typename ShapeType>
    static std::vector<layout> calc_output_layouts(megakernel_decode_node const& node,
                                                   const kernel_impl_params& impl_param);
    static layout calc_output_layout(megakernel_decode_node const& node, kernel_impl_params const& impl_param);
    static std::string to_string(megakernel_decode_node const& node);

    typed_primitive_inst(network& network, megakernel_decode_node const& node);
};

using megakernel_decode_inst = typed_primitive_inst<megakernel_decode>;

}  // namespace cldnn
