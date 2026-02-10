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
import signal
import faulthandler
import traceback
import threading

# Enable faulthandler for SIGSEGV/SIGABRT stack traces.
faulthandler.enable()

# Dump all thread stacks after 45 seconds if we're still alive (hang detection).
faulthandler.dump_traceback_later(45, exit=True)


def _alarm_handler(signum, frame):
    """On SIGALRM, dump all thread stacks and abort to get a core dump."""
    sys.stderr.write("\n=== HANG DETECTED (SIGALRM) ===\n")
    sys.stderr.write(f"Main thread: {threading.current_thread()}\n")
    sys.stderr.write(f"Active threads: {threading.enumerate()}\n\n")
    faulthandler.dump_traceback(file=sys.stderr, all_threads=True)
    sys.stderr.write("\nSending SIGABRT for core dump...\n")
    sys.stderr.flush()
    os.kill(os.getpid(), signal.SIGABRT)


signal.signal(signal.SIGALRM, _alarm_handler)
signal.alarm(40)  # 40 seconds before alarm fires

print("[test] Starting XDNA JAX integration test", flush=True)
print(f"[test] PID: {os.getpid()}", flush=True)

# Register the XDNA plugin before importing jax
sys.path.insert(0, os.path.dirname(__file__))
print("[test] Importing jax_xdna...", flush=True)
from jax_xdna import initialize
print("[test] Calling initialize()...", flush=True)
initialize()
print("[test] Plugin registered.", flush=True)

print("[test] Importing jax...", flush=True)
import jax
import jax.numpy as jnp
import numpy as np
print("[test] JAX imported.", flush=True)

# Force XDNA as the only platform
print("[test] Setting jax_platforms=xdna...", flush=True)
jax.config.update("jax_platforms", "xdna")
print("[test] Platform set.", flush=True)

print("[test] Checking devices...", flush=True)
devs = jax.devices()
print(f"[test] Devices: {devs}", flush=True)


def run_test(name, fn):
    """Run a test function, print pass/fail."""
    print(f"  [test] Starting: {name}", flush=True)
    try:
        fn()
        print(f"  PASS: {name}", flush=True)
        return True
    except Exception as e:
        print(f"  FAIL: {name}", flush=True)
        traceback.print_exc()
        print(flush=True)
        return False


def test_device_discovery():
    """Test 1: Plugin loads and device is visible."""
    devices = jax.devices()
    assert len(devices) > 0, f"No devices found: {devices}"
    dev = devices[0]
    assert dev.platform == "xdna", f"Expected platform 'xdna', got '{dev.platform}'"
    print(f"         Device: {dev}", flush=True)


def test_add():
    """Test 2: Binary add — matches C++ CompileAndExecuteAdd test."""
    a = np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float32)
    b = np.array([10.0, 20.0, 30.0, 40.0], dtype=np.float32)
    print("  [test] Calling jax.jit(x + y)...", flush=True)
    result = jax.jit(lambda x, y: x + y)(a, b)
    print("  [test] jit returned, converting result...", flush=True)
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

    print(f"XDNA JAX Integration Tests", flush=True)
    print(f"=" * 40, flush=True)

    passed = 0
    failed = 0
    for name, fn in tests:
        if run_test(name, fn):
            passed += 1
        else:
            failed += 1

    print(f"=" * 40, flush=True)
    print(f"Results: {passed} passed, {failed} failed, {len(tests)} total", flush=True)

    # Cancel alarm — we finished
    signal.alarm(0)

    sys.exit(0 if failed == 0 else 1)


if __name__ == "__main__":
    main()
