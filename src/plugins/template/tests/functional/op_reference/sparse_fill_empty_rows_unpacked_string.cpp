// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "openvino/op/sparse_fill_empty_rows_unpacked_string.hpp"

#include <gtest/gtest.h>

#include "base_reference_test.hpp"
#include "openvino/core/type/element_type_traits.hpp"
#include "openvino/op/constant.hpp"

namespace {
struct SparseFillEmptyRowsUnpackedStringParams {
    SparseFillEmptyRowsUnpackedStringParams(
        const reference_tests::Tensor& beginsTensor,
        const reference_tests::Tensor& endsTensor,
        const reference_tests::Tensor& symbolsTensor,
        const reference_tests::Tensor& indicesTensor,
        const reference_tests::Tensor& denseShapeTensor,
        const reference_tests::Tensor& defaultValueTensor,
        const reference_tests::Tensor& expectedBeginsTensor,
        const reference_tests::Tensor& expectedEndsTensor,
        const reference_tests::Tensor& expectedSymbolsTensor,
        const reference_tests::Tensor& expectedIndicesTensor,
        const reference_tests::Tensor& expectedEmptyRowIndicatorTensor)
        : beginsTensor(beginsTensor),
          endsTensor(endsTensor),
          symbolsTensor(symbolsTensor),
          indicesTensor(indicesTensor),
          denseShapeTensor(denseShapeTensor),
          defaultValueTensor(defaultValueTensor),
          expectedBeginsTensor(expectedBeginsTensor),
          expectedEndsTensor(expectedEndsTensor),
          expectedSymbolsTensor(expectedSymbolsTensor),
          expectedIndicesTensor(expectedIndicesTensor),
          expectedEmptyRowIndicatorTensor(expectedEmptyRowIndicatorTensor) {}

    reference_tests::Tensor beginsTensor;
    reference_tests::Tensor endsTensor;
    reference_tests::Tensor symbolsTensor;
    reference_tests::Tensor indicesTensor;
    reference_tests::Tensor denseShapeTensor;
    reference_tests::Tensor defaultValueTensor;
    reference_tests::Tensor expectedBeginsTensor;
    reference_tests::Tensor expectedEndsTensor;
    reference_tests::Tensor expectedSymbolsTensor;
    reference_tests::Tensor expectedIndicesTensor;
    reference_tests::Tensor expectedEmptyRowIndicatorTensor;
};

class ReferenceSparseFillEmptyRowsUnpackedStringV16LayerTest : public testing::TestWithParam<SparseFillEmptyRowsUnpackedStringParams>,
                                                 public reference_tests::CommonReferenceTest {
protected:
    void SetUp() override {
        const auto& params = GetParam();
        function = CreateFunction(params);
        inputData = {params.beginsTensor.data,
                     params.endsTensor.data,
                     params.symbolsTensor.data,
                     params.indicesTensor.data,
                     params.denseShapeTensor.data,
                     params.defaultValueTensor.data};
        refOutData = {params.expectedBeginsTensor.data,
                      params.expectedEndsTensor.data,
                      params.expectedSymbolsTensor.data,
                      params.expectedIndicesTensor.data,
                      params.expectedEmptyRowIndicatorTensor.data};
    }

public:
    static std::string getTestCaseName(const testing::TestParamInfo<SparseFillEmptyRowsUnpackedStringParams>& obj) {
        auto param = obj.param;
        std::ostringstream result;
        result << "beginsType=" << param.beginsTensor.type;
        result << "_beginsShape=" << param.beginsTensor.shape;
        result << "_denseShapeType=" << param.denseShapeTensor.type;
        result << "_denseShapeValues=" << testing::PrintToString(param.denseShapeTensor.data);
        result << "_indicesType=" << param.indicesTensor.type;
        result << "_indicesShape=" << param.indicesTensor.shape;
        result << "_symbolsShape=" << param.symbolsTensor.shape;
        return result.str();
    }

private:
    static std::shared_ptr<ov::Model> CreateFunction(const SparseFillEmptyRowsUnpackedStringParams& params) {
        using ov::op::v0::Parameter;

        const auto begins = std::make_shared<Parameter>(params.beginsTensor.type, params.beginsTensor.shape);
        const auto ends = std::make_shared<Parameter>(params.endsTensor.type, params.endsTensor.shape);
        const auto symbols = std::make_shared<Parameter>(params.symbolsTensor.type, params.symbolsTensor.shape);
        const auto indices = std::make_shared<Parameter>(params.indicesTensor.type, params.indicesTensor.shape);
        const auto dense_shape = 
            std::make_shared<Parameter>(params.denseShapeTensor.type, params.denseShapeTensor.shape);
        const auto default_value =
            std::make_shared<Parameter>(params.defaultValueTensor.type, params.defaultValueTensor.shape);
 
        const auto sparseFillEmptyRowsUnpackedString =
            std::make_shared<ov::op::v16::SparseFillEmptyRowsUnpackedString>(
                begins, ends, symbols, indices, dense_shape, default_value);

        return std::make_shared<ov::Model>(
            ov::OutputVector{sparseFillEmptyRowsUnpackedString->output(0),
                     sparseFillEmptyRowsUnpackedString->output(1),
                     sparseFillEmptyRowsUnpackedString->output(2),
                     sparseFillEmptyRowsUnpackedString->output(3),
                     sparseFillEmptyRowsUnpackedString->output(4)},
            ov::ParameterVector{begins, ends, symbols, indices, dense_shape, default_value});
    }
};

TEST_P(ReferenceSparseFillEmptyRowsUnpackedStringV16LayerTest, CompareWithRefs) {
    Exec();
}

template <ov::element::Type_t T_idx>
std::vector<SparseFillEmptyRowsUnpackedStringParams> generateSparseFillEmptyRowsUnpackedStringParams() {
    using T_I = typename ov::element_type_traits<T_idx>::value_type;
    using reference_tests::Tensor;

    std::vector<SparseFillEmptyRowsUnpackedStringParams> params{
        // No empty rows - All indices are filled
        //SparseFillEmptyRowsUnpackedStringParams(
        //    // begins
        //    Tensor({3}, T_idx, std::vector<T_I>{0, 5, 10}),
        //    // ends
        //    Tensor({3}, T_idx, std::vector<T_I>{5, 10, 15}),
        //    // symbols - "HelloWorldTensor" encoded as bytes
        //    Tensor({15}, ov::element::u8, std::vector<uint8_t>{'H','e','l','l','o','W','o','r','l','d','T','e','n','s','r'}),
        //    // indices
        //    Tensor({3, 2}, T_idx, std::vector<T_I>{0, 0, 1, 0, 2, 0}),
        //    // dense_shape
        //    Tensor({2}, T_idx, std::vector<T_I>{3, 1}),
        //    // default_value - "Empty" encoded as bytes
        //    Tensor({5}, ov::element::u8, std::vector<uint8_t>{'E','m','p','t','y'}),
        //    // expected_begins
        //    Tensor({3}, T_idx, std::vector<T_I>{0, 5, 10}),
        //    // expected_ends
        //    Tensor({3}, T_idx, std::vector<T_I>{5, 10, 15}),
        //    // expected_indices
        //    Tensor({3, 2}, T_idx, std::vector<T_I>{0, 0, 1, 0, 2, 0}),
        //    // expected_symbols - same as input, no empty rows to fill
        //    Tensor({15}, ov::element::u8, std::vector<uint8_t>{'H','e','l','l','o','W','o','r','l','d','T','e','n','s','r'}),
        //    // expected_empty_row_indicator
        //    Tensor({3}, ov::element::boolean, std::vector<uint8_t>{0, 0, 0})
        //),

        // One empty row in the middle
        SparseFillEmptyRowsUnpackedStringParams(
            // begins
            Tensor({3}, T_idx, std::vector<T_I>{0, 5, 10}),
            // ends
            Tensor({3}, T_idx, std::vector<T_I>{5, 10, 15}),
            // symbols - "HelloWorldTensor" encoded as bytes
            Tensor({15}, ov::element::u8, std::vector<uint8_t>{'H','e','l','l','o','W','o','r','l','d','T','e','n','s','r'}),
            // indices
            Tensor({3, 2}, T_idx, std::vector<T_I>{0, 0, 1, 0, 3, 0}),
            // dense_shape
            Tensor({2}, T_idx, std::vector<T_I>{4, 1}),
            // default_value - "Empty" encoded as bytes
            Tensor({5}, ov::element::u8, std::vector<uint8_t>{'E','m','p','t','y'}),
            // expected_begins
            Tensor({4}, T_idx, std::vector<T_I>{0, 5, 15, 10}),
            // expected_ends
            Tensor({4}, T_idx, std::vector<T_I>{5, 10, 20, 15}),
            // expected_symbols - "HelloWorldEmptyTensor" encoded as bytes
            Tensor({20}, ov::element::u8, std::vector<uint8_t>{'H','e','l','l','o','W','o','r','l','d','T','e','n','s','r','E','m','p','t','y'}),
            // expected_indices
            Tensor({4, 2}, T_idx, std::vector<T_I>{0, 0, 1, 0, 2, 0, 3, 0}),
            // expected_empty_row_indicator
            Tensor({4}, ov::element::boolean, std::vector<uint8_t>{0, 0, 1, 0})
        ),

        // Multiple empty rows
        //SparseFillEmptyRowsUnpackedStringParams(
        //    // begins
        //    Tensor({2}, T_idx, std::vector<T_I>{0, 5}),
        //    // ends
        //    Tensor({2}, T_idx, std::vector<T_I>{5, 10}),
        //    // symbols - "HelloWorld" encoded as bytes
        //    Tensor({10}, ov::element::u8, std::vector<uint8_t>{'H','e','l','l','o','W','o','r','l','d'}),
        //    // indices
        //    Tensor({2, 2}, T_idx, std::vector<T_I>{0, 0, 4, 0}),
        //    // dense_shape
        //    Tensor({2}, T_idx, std::vector<T_I>{5, 1}),
        //    // default_value - "Empty" encoded as bytes
        //    Tensor({5}, ov::element::u8, std::vector<uint8_t>{'E','m','p','t','y'}),
        //    // expected_begins
        //    Tensor({5}, T_idx, std::vector<T_I>{0, 10, 10, 10, 5}),
        //    // expected_ends
        //    Tensor({5}, T_idx, std::vector<T_I>{5, 15, 15, 15, 10}),
        //    // expected_indices
        //    Tensor({5, 2}, T_idx, std::vector<T_I>{0, 0, 1, 0, 2, 0, 3, 0, 4, 0}),
        //    // expected_symbols - "HelloWorldEmpty" encoded as bytes
        //    Tensor({15}, ov::element::u8, std::vector<uint8_t>{'H','e','l','l','o','W','o','r','l','d','E','m','p','t','y'}),
        //    // expected_empty_row_indicator
        //    Tensor({5}, ov::element::boolean, std::vector<uint8_t>{0, 1, 1, 1, 0})
        //),

        // All rows empty
        //SparseFillEmptyRowsUnpackedStringParams(
        //    // begins - empty
        //    Tensor({0}, T_idx, std::vector<T_I>{}),
        //    // ends - empty
        //    Tensor({0}, T_idx, std::vector<T_I>{}),
        //    // symbols - empty
        //    Tensor({0}, ov::element::u8, std::vector<uint8_t>{}),
        //    // indices - empty
        //    Tensor({0, 2}, T_idx, std::vector<T_I>{}),
        //    // dense_shape
        //    Tensor({2}, T_idx, std::vector<T_I>{3, 1}),
        //    // default_value - "Empty" encoded as bytes
        //    Tensor({5}, ov::element::u8, std::vector<uint8_t>{'E','m','p','t','y'}),
        //    // expected_begins
        //    Tensor({3}, T_idx, std::vector<T_I>{0, 0, 0}),
        //    // expected_ends
        //    Tensor({3}, T_idx, std::vector<T_I>{5, 5, 5}),
        //    // expected_indices
        //    Tensor({3, 2}, T_idx, std::vector<T_I>{0, 0, 1, 0, 2, 0}),
        //    // expected_symbols - "Empty" encoded as bytes
        //    Tensor({5}, ov::element::u8, std::vector<uint8_t>{'E','m','p','t','y'}),
        //    // expected_empty_row_indicator
        //    Tensor({3}, ov::element::boolean, std::vector<uint8_t>{1, 1, 1})
        //),

        // Example from the spec
        //SparseFillEmptyRowsUnpackedStringParams(
        //    // begins
        //    Tensor({6}, T_idx, std::vector<T_I>{0, 5, 15, 20, 25, 30}),
        //    // ends
        //    Tensor({6}, T_idx, std::vector<T_I>{5, 10, 20, 25, 30, 35}),
        //    // symbols - "HelloWorldOpenVINOTensorProcessing" encoded as bytes
        //    Tensor({35}, ov::element::u8, 
        //        std::vector<uint8_t>{'H','e','l','l','o','W','o','r','l','d','O','p','e','n','V','I','N','O','T','e','n','s','o','r','P','r','o','c','e','s','s','i','n','g'}),
        //    // indices
        //    Tensor({6, 2}, T_idx, std::vector<T_I>{0, 0, 0, 1, 2, 0, 2, 1, 3, 0, 3, 1}),
        //    // dense_shape
        //    Tensor({2}, T_idx, std::vector<T_I>{5, 2}),
        //    // default_value - "Empty" encoded as bytes
        //    Tensor({5}, ov::element::u8, std::vector<uint8_t>{'E','m','p','t','y'}),
        //    // expected_begins
        //    Tensor({8}, T_idx, std::vector<T_I>{0, 5, 35, 15, 20, 25, 30, 35}),
        //    // expected_ends
        //    Tensor({8}, T_idx, std::vector<T_I>{5, 10, 40, 20, 25, 30, 35, 40}),
        //    // expected_indices
        //    Tensor({8, 2}, T_idx, std::vector<T_I>{0, 0, 0, 1, 1, 0, 2, 0, 2, 1, 3, 0, 3, 1, 4, 0}),
        //    // expected_symbols - "HelloWorldOpenVINOTensorProcessingEmpty" encoded as bytes
        //    Tensor({40}, ov::element::u8, 
        //        std::vector<uint8_t>{'H','e','l','l','o','W','o','r','l','d','O','p','e','n','V','I','N','O','T','e','n','s','o','r','P','r','o','c','e','s','s','i','n','g','E','m','p','t','y'}),
        //    // expected_empty_row_indicator
        //    Tensor({5}, ov::element::boolean, std::vector<uint8_t>{0, 1, 0, 0, 1})
        //)
    };

    return params;
}

std::vector<SparseFillEmptyRowsUnpackedStringParams> generateSparseFillEmptyRowsUnpackedStringV16CombinedParams() {
    using ov::element::Type_t;
    const std::vector<std::vector<SparseFillEmptyRowsUnpackedStringParams>> typeParams{
        generateSparseFillEmptyRowsUnpackedStringParams<Type_t::i32>(),
        generateSparseFillEmptyRowsUnpackedStringParams<Type_t::i64>()
    };

    std::vector<SparseFillEmptyRowsUnpackedStringParams> combinedParams;
    for (const auto& params : typeParams) {
        combinedParams.insert(combinedParams.end(), params.begin(), params.end());
    }
    return combinedParams;
}

INSTANTIATE_TEST_SUITE_P(smoke_SparseFillEmptyRowsUnpackedString_With_Hardcoded_Refs,
                         ReferenceSparseFillEmptyRowsUnpackedStringV16LayerTest,
                         testing::ValuesIn(generateSparseFillEmptyRowsUnpackedStringV16CombinedParams()),
                         ReferenceSparseFillEmptyRowsUnpackedStringV16LayerTest::getTestCaseName);
}  // namespace
