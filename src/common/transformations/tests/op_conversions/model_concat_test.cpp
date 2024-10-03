// Copyright (C) 2018-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "transformations/common_optimizations/model_concat.hpp"

#include <gtest/gtest.h>

#include <memory>
#include <string>
#include <openvino/pass/serialize.hpp>

#include "common_test_utils/ov_test_utils.hpp"
#include "openvino/core/model.hpp"

#include "openvino/opsets/opset9.hpp"
#include "transformations/init_node_info.hpp"
#include "openvino/runtime/core.hpp"
using namespace ov;
using namespace testing;

TEST_F(TransformationTestsF, ModelConcat) {
    {
        ov::Core core;
        auto model = core.read_model("/home/pwysocki/models/bark_fine_feature_extractor.xml");

        manager.register_pass<ov::pass::ModelConcat>();
        manager.run_passes(model);
        std::string xml = "/home/pwysocki/models/concatenated.xml";
        std::string bin = "/home/pwysocki/models/concatenated.bin";
        ov::serialize(model, xml, bin);
    }
    {
        ov::Core core;
        auto model_ref = core.read_model("/home/pwysocki/models/bark_fine_feature_extractor.xml");
    }
}
