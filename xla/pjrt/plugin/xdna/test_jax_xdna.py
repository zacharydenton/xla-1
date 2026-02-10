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

# Register SIGUSR1 to dump all threads on demand: kill -USR1 <pid>
faulthandler.register(signal.SIGUSR1, all_threads=True, chain=False)


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


def test_bf16_add():
    """Test 6: bf16 add — native AIE2p type."""
    import ml_dtypes
    a = np.array([1.0, 2.0, 3.0, 4.0], dtype=ml_dtypes.bfloat16)
    b = np.array([10.0, 20.0, 30.0, 40.0], dtype=ml_dtypes.bfloat16)
    result = jax.jit(lambda x, y: x + y)(a, b)
    expected = np.array([11.0, 22.0, 33.0, 44.0], dtype=np.float32)
    np.testing.assert_allclose(np.array(result, dtype=np.float32), expected, rtol=1e-2)


def test_bf16_multiply():
    """Test 7: bf16 multiply — native vmul.f hardware path."""
    import ml_dtypes
    a = np.array([1.0, 2.0, 3.0, 4.0], dtype=ml_dtypes.bfloat16)
    b = np.array([10.0, 20.0, 30.0, 40.0], dtype=ml_dtypes.bfloat16)
    result = jax.jit(lambda x, y: x * y)(a, b)
    expected = np.array([10.0, 40.0, 90.0, 160.0], dtype=np.float32)
    np.testing.assert_allclose(np.array(result, dtype=np.float32), expected, rtol=1e-2)


def test_add_f32_256():
    """Test 8: f32[256] add — scalar path (no f32 vector add in Peano)."""
    a = np.random.randn(256).astype(np.float32)
    b = np.random.randn(256).astype(np.float32)
    result = jax.jit(lambda x, y: x + y)(a, b)
    np.testing.assert_allclose(np.array(result), a + b, rtol=1e-5)


def test_negate_f32_256():
    """Test 9: f32[256] negate — vectorized via XOR sign-bit flip on <16 x i32>."""
    a = np.random.randn(256).astype(np.float32)
    result = jax.jit(lambda x: -x)(a)
    np.testing.assert_allclose(np.array(result), -a, rtol=1e-5)


def test_multiply_f32_256():
    """Test 10: f32[256] multiply — scalar bf16 workaround (no f32 vector mul)."""
    a = np.array([float(i) for i in range(1, 257)], dtype=np.float32)
    b = np.full(256, 2.0, dtype=np.float32)
    result = jax.jit(lambda x, y: x * y)(a, b)
    # bf16 truncation loses precision, so use relaxed tolerance
    expected = a * b
    np.testing.assert_allclose(np.array(result), expected, rtol=1e-2)


def test_bf16_add_256():
    """Test 11: bf16[256] add — scalar path (no bf16 vector add in Peano)."""
    import ml_dtypes
    a = np.random.randn(256).astype(ml_dtypes.bfloat16)
    b = np.random.randn(256).astype(ml_dtypes.bfloat16)
    result = jax.jit(lambda x, y: x + y)(a, b)
    expected = (a.astype(np.float32) + b.astype(np.float32))
    np.testing.assert_allclose(np.array(result, dtype=np.float32), expected, rtol=1e-2)


def test_bf16_multiply_256():
    """Test 12: bf16[256] multiply — vectorized fmul <32 x bfloat> (native)."""
    import ml_dtypes
    a = np.array([float(i) for i in range(1, 257)], dtype=ml_dtypes.bfloat16)
    b = np.full(256, 2.0, dtype=ml_dtypes.bfloat16)
    result = jax.jit(lambda x, y: x * y)(a, b)
    expected = (a.astype(np.float32) * b.astype(np.float32))
    np.testing.assert_allclose(np.array(result, dtype=np.float32), expected, rtol=1e-2)


def test_bf16_negate_256():
    """Test 13: bf16[256] negate — vectorized via XOR sign-bit flip on <32 x i16>."""
    import ml_dtypes
    a = np.random.randn(256).astype(ml_dtypes.bfloat16)
    result = jax.jit(lambda x: -x)(a)
    expected = -(a.astype(np.float32))
    np.testing.assert_allclose(np.array(result, dtype=np.float32), expected, rtol=1e-2)


def test_cache_hit():
    """Test 8: Second compilation of same HLO hits XDNA cache.

    test_add already compiled f32[4] add and cached the result.
    A new jax.jit instance with the same computation triggers a fresh
    compilation request from JAX, but should hit our XDNA in-memory cache.
    We redirect C-level stderr to verify the cache hit trace message.
    """
    # Redirect C-level stderr (fd 2) to a pipe so we can capture XDNA_TRACE output.
    read_fd, write_fd = os.pipe()
    saved_stderr = os.dup(2)
    os.dup2(write_fd, 2)

    a = np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float32)
    b = np.array([5.0, 6.0, 7.0, 8.0], dtype=np.float32)
    result = jax.jit(lambda x, y: x + y)(a, b)

    # Restore stderr and read captured output.
    os.dup2(saved_stderr, 2)
    os.close(saved_stderr)
    os.close(write_fd)
    captured = os.read(read_fd, 65536).decode('utf-8', errors='replace')
    os.close(read_fd)

    expected = np.array([6.0, 8.0, 10.0, 12.0], dtype=np.float32)
    np.testing.assert_allclose(np.array(result), expected, rtol=1e-5)

    assert "Compilation cache hit" in captured, \
        f"Expected 'Compilation cache hit' in stderr. Got:\n{captured}"
    print("         Cache hit confirmed via stderr trace.", flush=True)


def main():
    tests = [
        ("Device discovery", test_device_discovery),
        ("jit(x + y)", test_add),
        ("jit(x * y)", test_multiply),
        ("jit(-x)", test_negate),
        ("jit(x - y)", test_subtract),
        ("jit(x + y) bf16", test_bf16_add),
        ("jit(x * y) bf16", test_bf16_multiply),
        ("jit(x + y) f32[256]", test_add_f32_256),
        ("jit(-x) f32[256]", test_negate_f32_256),
        ("jit(x * y) f32[256]", test_multiply_f32_256),
        ("jit(x + y) bf16[256]", test_bf16_add_256),
        ("jit(x * y) bf16[256]", test_bf16_multiply_256),
        ("jit(-x) bf16[256]", test_bf16_negate_256),
        ("cache hit", test_cache_hit),
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
