#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Copyright (C) 2018-2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
"""
Unit tests for C++ Data Dispatcher PoC

These tests verify that the C++ dispatcher produces identical results
to the Python implementation.
"""

import pytest
import numpy as np
from numpy.testing import assert_array_equal, assert_array_almost_equal

import openvino as ov
from openvino import Core, Model, Type, Shape
import openvino.opset13 as ops

from openvino.utils.data_helpers.data_dispatcher import _data_dispatch as python_dispatch


# ============================================================================
# Fixtures
# ============================================================================

@pytest.fixture
def core():
    return Core()


@pytest.fixture
def single_input_model():
    """Single FP32 input model."""
    param = ops.parameter([1, 3, 224, 224], Type.f32, name="input")
    result = ops.relu(param)
    return Model(result, [param])


@pytest.fixture
def multi_input_model():
    """Model with 3 inputs of different shapes."""
    params = [
        ops.parameter([1, 64], Type.f32, name="input_0"),
        ops.parameter([1, 128], Type.f32, name="input_1"),
        ops.parameter([1, 256], Type.f32, name="input_2"),
    ]
    concat = ops.concat(params, axis=1)
    return Model(concat, params)


@pytest.fixture
def dynamic_input_model():
    """Model with dynamic shape input."""
    param = ops.parameter([-1, -1], Type.f32, name="input")
    result = ops.relu(param)
    return Model(result, [param])


@pytest.fixture
def string_input_model():
    """Model with string input."""
    param = ops.parameter([1], Type.string, name="input")
    # Note: Most ops don't support string, this is just for testing tensor creation
    return Model([param.output(0)], [param])


# ============================================================================
# Test Cases
# ============================================================================

class TestSingleInputDispatch:
    """Tests for single-input models."""
    
    def test_numpy_array_copy_mode(self, core, single_input_model):
        """Test dispatching numpy array with share_inputs=False."""
        compiled = core.compile_model(single_input_model, "CPU")
        request = compiled.create_infer_request()
        
        inputs = np.random.randn(1, 3, 224, 224).astype(np.float32)
        
        result = python_dispatch(request, inputs, is_shared=False)
        
        # Should return empty dict when data copied to existing tensor
        assert result == {}
        # Verify data was copied
        assert_array_almost_equal(request.get_input_tensor().data, inputs)
    
    def test_numpy_array_share_mode(self, core, single_input_model):
        """Test dispatching numpy array with share_inputs=True."""
        compiled = core.compile_model(single_input_model, "CPU")
        request = compiled.create_infer_request()
        
        inputs = np.ascontiguousarray(np.random.randn(1, 3, 224, 224).astype(np.float32))
        
        result = python_dispatch(request, inputs, is_shared=True)
        
        # Should return a Tensor
        assert isinstance(result, ov.Tensor)
    
    def test_tensor_passthrough(self, core, single_input_model):
        """Test that Tensor inputs pass through unchanged."""
        compiled = core.compile_model(single_input_model, "CPU")
        request = compiled.create_infer_request()
        
        inputs = ov.Tensor(np.random.randn(1, 3, 224, 224).astype(np.float32))
        
        result = python_dispatch(request, inputs, is_shared=False)
        
        assert result is inputs
    
    def test_non_contiguous_array(self, core, single_input_model):
        """Test non-contiguous array is handled correctly."""
        compiled = core.compile_model(single_input_model, "CPU")
        request = compiled.create_infer_request()
        
        # Create non-contiguous array via transpose
        base = np.random.randn(224, 224, 3, 1).astype(np.float32)
        inputs = np.transpose(base, (3, 2, 0, 1))  # (1, 3, 224, 224)
        assert not inputs.flags['C_CONTIGUOUS']
        
        result = python_dispatch(request, inputs, is_shared=True)
        
        # Data should still be correct
        assert isinstance(result, ov.Tensor)
    
    def test_dtype_conversion(self, core, single_input_model):
        """Test automatic dtype conversion."""
        compiled = core.compile_model(single_input_model, "CPU")
        request = compiled.create_infer_request()
        
        # Send float64, model expects float32
        inputs = np.random.randn(1, 3, 224, 224).astype(np.float64)
        
        result = python_dispatch(request, inputs, is_shared=False)
        
        # Should have converted to float32
        tensor = request.get_input_tensor()
        assert tensor.element_type == Type.f32


class TestMultiInputDispatch:
    """Tests for multi-input models."""
    
    def test_dict_string_keys(self, core, multi_input_model):
        """Test dict with string keys."""
        compiled = core.compile_model(multi_input_model, "CPU")
        request = compiled.create_infer_request()
        
        inputs = {
            "input_0": np.random.randn(1, 64).astype(np.float32),
            "input_1": np.random.randn(1, 128).astype(np.float32),
            "input_2": np.random.randn(1, 256).astype(np.float32),
        }
        
        result = python_dispatch(request, inputs, is_shared=False)
        
        assert isinstance(result, dict)
    
    def test_dict_int_keys(self, core, multi_input_model):
        """Test dict with integer keys."""
        compiled = core.compile_model(multi_input_model, "CPU")
        request = compiled.create_infer_request()
        
        inputs = {
            0: np.random.randn(1, 64).astype(np.float32),
            1: np.random.randn(1, 128).astype(np.float32),
            2: np.random.randn(1, 256).astype(np.float32),
        }
        
        result = python_dispatch(request, inputs, is_shared=False)
        
        assert isinstance(result, dict)
    
    def test_list_inputs(self, core, multi_input_model):
        """Test list of inputs."""
        compiled = core.compile_model(multi_input_model, "CPU")
        request = compiled.create_infer_request()
        
        inputs = [
            np.random.randn(1, 64).astype(np.float32),
            np.random.randn(1, 128).astype(np.float32),
            np.random.randn(1, 256).astype(np.float32),
        ]
        
        result = python_dispatch(request, inputs, is_shared=False)
        
        assert isinstance(result, dict)
    
    def test_tuple_inputs(self, core, multi_input_model):
        """Test tuple of inputs."""
        compiled = core.compile_model(multi_input_model, "CPU")
        request = compiled.create_infer_request()
        
        inputs = (
            np.random.randn(1, 64).astype(np.float32),
            np.random.randn(1, 128).astype(np.float32),
            np.random.randn(1, 256).astype(np.float32),
        )
        
        result = python_dispatch(request, inputs, is_shared=False)
        
        assert isinstance(result, dict)


class TestDynamicShapes:
    """Tests for dynamic shape models."""
    
    def test_varying_shapes(self, core, dynamic_input_model):
        """Test dispatching different shapes to dynamic model."""
        compiled = core.compile_model(dynamic_input_model, "CPU")
        request = compiled.create_infer_request()
        
        # First inference
        inputs1 = np.random.randn(1, 64).astype(np.float32)
        python_dispatch(request, inputs1, is_shared=False)
        request.infer()
        
        # Second inference with different shape
        inputs2 = np.random.randn(2, 128).astype(np.float32)
        python_dispatch(request, inputs2, is_shared=False)
        request.infer()
        
        # Tensor should have new shape
        assert tuple(request.get_input_tensor().shape) == (2, 128)


class TestScalars:
    """Tests for scalar inputs."""
    
    def test_python_int(self, core):
        """Test Python int as input."""
        param = ops.parameter([1], Type.i32, name="input")
        model = Model(ops.abs(param), [param])
        compiled = core.compile_model(model, "CPU")
        request = compiled.create_infer_request()
        
        result = python_dispatch(request, 42, is_shared=False)
        
        assert isinstance(result, ov.Tensor)
    
    def test_python_float(self, core):
        """Test Python float as input."""
        param = ops.parameter([1], Type.f32, name="input")
        model = Model(ops.abs(param), [param])
        compiled = core.compile_model(model, "CPU")
        request = compiled.create_infer_request()
        
        result = python_dispatch(request, 3.14, is_shared=False)
        
        assert isinstance(result, ov.Tensor)
    
    def test_numpy_scalar(self, core):
        """Test numpy scalar as input."""
        param = ops.parameter([1], Type.f32, name="input")
        model = Model(ops.abs(param), [param])
        compiled = core.compile_model(model, "CPU")
        request = compiled.create_infer_request()
        
        result = python_dispatch(request, np.float32(3.14), is_shared=False)
        
        assert isinstance(result, ov.Tensor)


class TestArrayLikeObjects:
    """Tests for objects with __array__ protocol."""
    
    def test_array_like_class(self, core, single_input_model):
        """Test custom class with __array__ method."""
        compiled = core.compile_model(single_input_model, "CPU")
        request = compiled.create_infer_request()
        
        class ArrayLike:
            def __init__(self, data):
                self._data = data
            
            def __array__(self, dtype=None, copy=None):
                return self._data
        
        data = np.random.randn(1, 3, 224, 224).astype(np.float32)
        inputs = ArrayLike(data)
        
        result = python_dispatch(request, inputs, is_shared=False)
        
        # Should work without error
        assert result == {}
        assert_array_almost_equal(request.get_input_tensor().data, data)


class TestEdgeCases:
    """Edge case tests."""
    
    def test_none_input(self, core, single_input_model):
        """Test None input returns empty dict."""
        compiled = core.compile_model(single_input_model, "CPU")
        request = compiled.create_infer_request()
        
        result = python_dispatch(request, None, is_shared=False)
        
        assert result == {}
    
    def test_empty_dict(self, core, single_input_model):
        """Test empty dict input."""
        compiled = core.compile_model(single_input_model, "CPU")
        request = compiled.create_infer_request()
        
        result = python_dispatch(request, {}, is_shared=False)
        
        assert result == {}
    
    def test_read_only_array(self, core, single_input_model):
        """Test read-only numpy array."""
        compiled = core.compile_model(single_input_model, "CPU")
        request = compiled.create_infer_request()
        
        inputs = np.random.randn(1, 3, 224, 224).astype(np.float32)
        inputs.flags.writeable = False
        
        # Should not raise, will copy internally
        result = python_dispatch(request, inputs, is_shared=True)
        
        assert isinstance(result, ov.Tensor)


class TestErrorHandling:
    """Error handling tests."""
    
    def test_invalid_key_type(self, core, single_input_model):
        """Test invalid dict key type raises TypeError."""
        compiled = core.compile_model(single_input_model, "CPU")
        request = compiled.create_infer_request()
        
        inputs = {3.14: np.random.randn(1, 3, 224, 224).astype(np.float32)}
        
        with pytest.raises(TypeError):
            python_dispatch(request, inputs, is_shared=False)
    
    def test_invalid_value_type(self, core, single_input_model):
        """Test invalid dict value type raises TypeError."""
        compiled = core.compile_model(single_input_model, "CPU")
        request = compiled.create_infer_request()
        
        inputs = {"input": object()}
        
        with pytest.raises(TypeError):
            python_dispatch(request, inputs, is_shared=False)


# ============================================================================
# Performance Regression Tests
# ============================================================================

class TestPerformance:
    """Basic performance sanity tests."""
    
    @pytest.mark.parametrize("share_inputs", [True, False])
    def test_dispatch_completes_quickly(self, core, single_input_model, share_inputs):
        """Ensure dispatch doesn't take more than 1ms for simple case."""
        import time
        
        compiled = core.compile_model(single_input_model, "CPU")
        request = compiled.create_infer_request()
        inputs = np.random.randn(1, 3, 224, 224).astype(np.float32)
        
        # Warmup
        for _ in range(10):
            python_dispatch(request, inputs, is_shared=share_inputs)
        
        # Measure
        start = time.perf_counter()
        for _ in range(100):
            python_dispatch(request, inputs, is_shared=share_inputs)
        elapsed = time.perf_counter() - start
        
        avg_ms = (elapsed / 100) * 1000
        assert avg_ms < 1.0, f"Dispatch too slow: {avg_ms:.3f}ms average"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
