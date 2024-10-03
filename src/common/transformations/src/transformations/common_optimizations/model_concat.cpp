// Copyright (C) 2018-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "transformations/common_optimizations/model_concat.hpp"

#include <memory>
#include <vector>

#include "itt.hpp"
#include "openvino/core/rt_info.hpp"
#include "openvino/op/constant.hpp"
#include "openvino/op/detection_output.hpp"
#include "openvino/op/squeeze.hpp"
#include "openvino/op/result.hpp"
#include "openvino/pass/pattern/op/wrap_type.hpp"
#include "transformations/utils/utils.hpp"
#include "openvino/pass/visualize_tree.hpp"
#include "openvino/pass/manager.hpp"
#include "openvino/runtime/core.hpp"

using namespace ov::op;

ov::pass::ModelConcat::ModelConcat() {
    MATCHER_SCOPE(ModelConcat);
    auto result_input_pattern = pattern::any_input();
    auto result_pattern = pattern::wrap_type<v0::Result>({result_input_pattern});

    ov::matcher_pass_callback callback = [=](pattern::Matcher& m) {
        ov::pass::Manager manager(get_pass_config());
        const auto& pattern_map = m.get_pattern_map();
        const auto& result = pattern_map.at(result_pattern);
        const auto& last_node = result->get_input_node_ptr(0);

        ov::Core core;
        auto second_model = core.read_model("/home/pwysocki/models/bark_fine_lm_0.xml");
        const auto second_model_input = second_model->input().get_node_shared_ptr();
        last_node->output(0).replace(second_model_input);
        manager.register_pass<VisualizeTree>("/home/pwysocki/models/model_after_merge.svg");
        
        //res_node->set_friendly_name(adaptive_pool->get_friendly_name());
        //copy_runtime_info(adaptive_pool, res_node);
        return true;
    };

    auto m = std::make_shared<ov::pass::pattern::Matcher>(result_pattern, matcher_name);
    this->register_matcher(m, callback);
}
