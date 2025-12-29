#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Copyright (C) 2018-2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
"""
Benchmark: Python vs C++ Data Dispatcher Performance

This script compares the overhead of:
1. Current Python singledispatch-based data_dispatcher.py
2. New C++ implementation (_data_dispatch_cpp)

The benchmark measures ONLY the dispatch overhead, not inference time.
"""

import time
import statistics
import numpy as np
from typing import Callable, Any

import openvino as ov
from openvino import Core, Model, Type, Shape
import openvino.opset13 as ops


# ============================================================================
# Benchmark Utilities
# ============================================================================

def create_test_model(input_shapes: list[tuple], dtypes: list[Type] = None) -> Model:
    """Create a simple model with specified input shapes."""
    if dtypes is None:
        dtypes = [Type.f32] * len(input_shapes)
    
    params = []
    for i, (shape, dtype) in enumerate(zip(input_shapes, dtypes)):
        params.append(ops.parameter(shape, dtype, name=f"input_{i}"))
    
    if len(params) == 1:
        result = ops.relu(params[0])
        return Model(result, params)
    else:
        # Multi-output model for different shaped inputs
        return Model([ops.relu(p) for p in params], params)


def benchmark(func: Callable, warmup: int = 100, iterations: int = 10000) -> dict:
    """Run benchmark and return statistics."""
    # Warmup
    for _ in range(warmup):
        func()
    
    # Timed runs
    times = []
    for _ in range(iterations):
        start = time.perf_counter_ns()
        func()
        end = time.perf_counter_ns()
        times.append(end - start)
    
    return {
        "mean_ns": statistics.mean(times),
        "median_ns": statistics.median(times),
        "stdev_ns": statistics.stdev(times) if len(times) > 1 else 0,
        "min_ns": min(times),
        "max_ns": max(times),
        "p95_ns": sorted(times)[int(len(times) * 0.95)],
        "p99_ns": sorted(times)[int(len(times) * 0.99)],
        "iterations": iterations,
    }


def format_time(ns: float) -> str:
    """Format nanoseconds to human-readable string."""
    if ns < 1000:
        return f"{ns:.1f} ns"
    elif ns < 1_000_000:
        return f"{ns/1000:.2f} µs"
    else:
        return f"{ns/1_000_000:.2f} ms"


# ============================================================================
# Python Dispatcher (Current Implementation)
# ============================================================================

from openvino.utils.data_helpers.data_dispatcher import _data_dispatch as python_dispatch


def benchmark_python_dispatcher(
    request: ov.InferRequest,
    inputs: Any,
    is_shared: bool = False,
    **kwargs
) -> dict:
    """Benchmark the current Python dispatcher."""
    def dispatch_fn():
        python_dispatch(request, inputs, is_shared=is_shared)
    
    return benchmark(dispatch_fn, **kwargs)


# ============================================================================
# C++ Dispatcher (New Implementation)
# ============================================================================

# Try to import the C++ dispatcher - will fail if not compiled yet
try:
    from openvino._pyopenvino._data_dispatch import _data_dispatch_cpp as cpp_dispatch
    CPP_DISPATCHER_AVAILABLE = True
    print("✓ C++ dispatcher loaded successfully\n")
except ImportError:
    CPP_DISPATCHER_AVAILABLE = False
    print("⚠ WARNING: C++ dispatcher not available. Rebuild OpenVINO to enable it.")
    print("           Using Python fallback simulation instead.\n")


def cpp_dispatch_fallback(
    request: ov.InferRequest,
    inputs: Any,
    is_shared: bool = False,
) -> dict:
    """
    Fallback Python simulation when C++ dispatcher is not compiled.
    This simulates optimal Python code without singledispatch overhead.
    """
    if inputs is None:
        return {}
    
    if isinstance(inputs, ov.Tensor):
        return inputs
    
    if isinstance(inputs, np.ndarray):
        tensor = request.get_input_tensor()
        if is_shared and inputs.flags['C_CONTIGUOUS'] and inputs.flags['WRITEABLE']:
            return ov.Tensor(inputs, shared_memory=True)
        else:
            tensor.shape = inputs.shape
            np.copyto(tensor.data, inputs)
            return {}
    
    if isinstance(inputs, dict):
        result = {}
        for key, value in inputs.items():
            if isinstance(value, ov.Tensor):
                result[key] = value
            elif isinstance(value, np.ndarray):
                if is_shared and value.flags['C_CONTIGUOUS'] and value.flags['WRITEABLE']:
                    result[key] = ov.Tensor(value, shared_memory=True)
                else:
                    tensor = request.get_tensor(key) if isinstance(key, str) else request.get_input_tensor(key)
                    if tuple(tensor.shape) != value.shape:
                        tensor.shape = value.shape
                    np.copyto(tensor.data, value)
            else:
                result[key] = ov.Tensor(np.asarray(value), shared_memory=False)
        return result
    
    raise TypeError(f"Unsupported type: {type(inputs)}")


def get_cpp_dispatch_fn():
    """Get the C++ dispatcher function (real or fallback)."""
    if CPP_DISPATCHER_AVAILABLE:
        return cpp_dispatch
    return cpp_dispatch_fallback


def benchmark_cpp_dispatcher(
    request: ov.InferRequest,
    inputs: Any,
    is_shared: bool = False,
    **kwargs
) -> dict:
    """Benchmark the C++ dispatcher (or fallback)."""
    dispatch_fn_impl = get_cpp_dispatch_fn()
    
    def dispatch_fn():
        dispatch_fn_impl(request, inputs, is_shared)
    
    return benchmark(dispatch_fn, **kwargs)


# ============================================================================
# Main Benchmark Suite
# ============================================================================

def run_benchmarks():
    """Run all benchmarks and print results."""
    core = Core()
    
    cpp_status = "REAL C++" if CPP_DISPATCHER_AVAILABLE else "SIMULATED (Python fallback)"
    
    print("=" * 80)
    print("OpenVINO Data Dispatcher Benchmark")
    print(f"Python singledispatch vs C++ implementation [{cpp_status}]")
    print("=" * 80)
    print()
    
    test_cases = [
        # (name, input_shapes, input_generator)
        (
            "Single small input (224x224 FP32)",
            [(1, 3, 224, 224)],
            lambda shapes: np.random.randn(*shapes[0]).astype(np.float32)
        ),
        (
            "Single large input (1024x1024 FP32)", 
            [(1, 3, 1024, 1024)],
            lambda shapes: np.random.randn(*shapes[0]).astype(np.float32)
        ),
        (
            "Dict with 3 inputs",
            [(1, 64), (1, 128), (1, 256)],
            lambda shapes: {f"input_{i}": np.random.randn(*s).astype(np.float32) for i, s in enumerate(shapes)}
        ),
        (
            "Dict with 10 inputs",
            [(1, 64)] * 10,
            lambda shapes: {f"input_{i}": np.random.randn(*s).astype(np.float32) for i, s in enumerate(shapes)}
        ),
        (
            "Small batch (1x64 FP32)",
            [(1, 64)],
            lambda shapes: np.random.randn(*shapes[0]).astype(np.float32)
        ),
        (
            "Tuple of 3 inputs",
            [(1, 32), (1, 32), (1, 32)],
            lambda shapes: tuple(np.random.randn(*s).astype(np.float32) for s in shapes)
        ),
    ]
    
    iterations = 5000
    warmup = 500
    
    for name, input_shapes, input_gen in test_cases:
        print(f"\n{'─' * 70}")
        print(f"Test: {name}")
        print(f"{'─' * 70}")
        
        # Create model and compile
        model = create_test_model(input_shapes)
        compiled = core.compile_model(model, "CPU")
        request = compiled.create_infer_request()
        
        # Generate test inputs
        inputs = input_gen(input_shapes)
        
        for mode_name, is_shared in [("share_inputs=False", False), ("share_inputs=True", True)]:
            print(f"\n  Mode: {mode_name}")
            
            # Benchmark Python dispatcher
            py_stats = benchmark_python_dispatcher(
                request, inputs, is_shared=is_shared,
                warmup=warmup, iterations=iterations
            )
            
            # Benchmark C++ dispatcher
            cpp_stats = benchmark_cpp_dispatcher(
                request, inputs, is_shared=is_shared,
                warmup=warmup, iterations=iterations
            )
            
            # Calculate speedup
            speedup = py_stats["mean_ns"] / cpp_stats["mean_ns"] if cpp_stats["mean_ns"] > 0 else float('inf')
            
            print(f"    Python dispatch:  {format_time(py_stats['mean_ns']):>12} (median: {format_time(py_stats['median_ns'])})")
            print(f"    C++ dispatch:     {format_time(cpp_stats['mean_ns']):>12} (median: {format_time(cpp_stats['median_ns'])})")
            print(f"    Speedup:          {speedup:.1f}x")
            print(f"    Python P99:       {format_time(py_stats['p99_ns'])}")
            print(f"    C++ P99:          {format_time(cpp_stats['p99_ns'])}")

    print()
    print("=" * 80)
    if not CPP_DISPATCHER_AVAILABLE:
        print("NOTE: C++ dispatcher was NOT compiled. Showing Python fallback comparison.")
        print("      Rebuild OpenVINO with the new data_dispatcher.cpp to see real gains.")
    else:
        print("NOTE: Using real C++ dispatcher. These are actual performance numbers.")
    print("=" * 80)


def run_end_to_end_comparison():
    """Compare full inference including dispatch overhead."""
    print("\n" + "=" * 80)
    print("End-to-End Inference Timing (includes inference time)")
    print("=" * 80)
    
    core = Core()
    
    # Create a small, fast model to highlight dispatch overhead
    model = create_test_model([(1, 64)])
    compiled = core.compile_model(model, "CPU")
    request = compiled.create_infer_request()
    
    inputs = np.random.randn(1, 64).astype(np.float32)
    iterations = 10000
    
    # Measure with Python dispatch (current)
    times_python = []
    for _ in range(iterations):
        start = time.perf_counter_ns()
        result = request.infer({"input_0": inputs}, share_inputs=False)
        times_python.append(time.perf_counter_ns() - start)
    
    # Measure with pre-created tensor (bypasses dispatch)
    tensor = ov.Tensor(inputs)
    times_tensor = []
    for _ in range(iterations):
        start = time.perf_counter_ns()
        result = request.infer({"input_0": tensor}, share_inputs=False)
        times_tensor.append(time.perf_counter_ns() - start)
    
    # Measure with share_inputs=True
    times_shared = []
    for _ in range(iterations):
        start = time.perf_counter_ns()
        result = request.infer({"input_0": inputs}, share_inputs=True)
        times_shared.append(time.perf_counter_ns() - start)
    
    py_mean = statistics.mean(times_python)
    tensor_mean = statistics.mean(times_tensor)
    shared_mean = statistics.mean(times_shared)
    
    print(f"\nSmall model (1x64 FP32 ReLU) - {iterations} iterations:")
    print(f"  numpy array (share=False): {format_time(py_mean)}")
    print(f"  numpy array (share=True):  {format_time(shared_mean)}")  
    print(f"  Pre-created Tensor:        {format_time(tensor_mean)}")
    print(f"\nDispatch overhead estimate: {format_time(py_mean - tensor_mean)}")
    print(f"This overhead can be reduced with C++ dispatcher.")


if __name__ == "__main__":
    run_benchmarks()
    run_end_to_end_comparison()
