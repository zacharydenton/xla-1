#!/usr/bin/env python3
"""JAX integration test for the XDNA NPU PJRT plugin.

Runs progressive tests from device discovery through element-wise ops.
All tests use f32[4] arrays to match what the XDNA lowering supports.

Usage:
  # Build the plugin first:
  bazel build //xla/pjrt/plugin/xdna:pjrt_c_api_xdna_plugin.so

  # Run tests:
  python xla/pjrt/plugin/xdna/test_jax_xdna.py

  # Or with explicit plugin path:
  XDNA_PLUGIN_PATH=/path/to/pjrt_c_api_xdna_plugin.so python xla/pjrt/plugin/xdna/test_jax_xdna.py
"""

import sys
import os
import traceback

# Register the XDNA plugin before importing jax
# (JAX initializes backends on first import of jax.numpy, etc.)
sys.path.insert(0, os.path.dirname(__file__))
from jax_xdna import initialize
initialize()

import jax
import jax.numpy as jnp
import numpy as np

# Force XDNA as the only platform
jax.config.update("jax_platforms", "xdna")


def run_test(name, fn):
    """Run a test function, print pass/fail."""
    try:
        fn()
        print(f"  PASS: {name}")
        return True
    except Exception as e:
        print(f"  FAIL: {name}")
        traceback.print_exc()
        print()
        return False


def test_device_discovery():
    """Test 1: Plugin loads and device is visible."""
    devices = jax.devices()
    assert len(devices) > 0, f"No devices found: {devices}"
    dev = devices[0]
    assert dev.platform == "xdna", f"Expected platform 'xdna', got '{dev.platform}'"
    print(f"         Device: {dev}")


def test_add():
    """Test 2: Binary add — matches C++ CompileAndExecuteAdd test."""
    # Use jax.jit with numpy inputs to avoid eager convert_element_type.
    # JAX will transfer numpy arrays to device as part of the jit call.
    a = np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float32)
    b = np.array([10.0, 20.0, 30.0, 40.0], dtype=np.float32)
    result = jax.jit(lambda x, y: x + y)(a, b)
    expected = np.array([11.0, 22.0, 33.0, 44.0], dtype=np.float32)
    np.testing.assert_allclose(np.array(result), expected, rtol=1e-5)


def test_multiply():
    """Test 3: Binary multiply."""
    a = np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float32)
    b = np.array([10.0, 20.0, 30.0, 40.0], dtype=np.float32)
    result = jax.jit(lambda x, y: x * y)(a, b)
    expected = np.array([10.0, 40.0, 90.0, 160.0], dtype=np.float32)
    np.testing.assert_allclose(np.array(result), expected, rtol=1e-5)


def test_negate():
    """Test 4: Unary negate."""
    a = np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float32)
    result = jax.jit(lambda x: -x)(a)
    expected = np.array([-1.0, -2.0, -3.0, -4.0], dtype=np.float32)
    np.testing.assert_allclose(np.array(result), expected, rtol=1e-5)


def test_subtract():
    """Test 5: Binary subtract."""
    a = np.array([10.0, 20.0, 30.0, 40.0], dtype=np.float32)
    b = np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float32)
    result = jax.jit(lambda x, y: x - y)(a, b)
    expected = np.array([9.0, 18.0, 27.0, 36.0], dtype=np.float32)
    np.testing.assert_allclose(np.array(result), expected, rtol=1e-5)


def main():
    tests = [
        ("Device discovery", test_device_discovery),
        ("jit(x + y)", test_add),
        ("jit(x * y)", test_multiply),
        ("jit(-x)", test_negate),
        ("jit(x - y)", test_subtract),
    ]

    print(f"XDNA JAX Integration Tests")
    print(f"=" * 40)

    passed = 0
    failed = 0
    for name, fn in tests:
        if run_test(name, fn):
            passed += 1
        else:
            failed += 1

    print(f"=" * 40)
    print(f"Results: {passed} passed, {failed} failed, {len(tests)} total")
    sys.exit(0 if failed == 0 else 1)


if __name__ == "__main__":
    main()
