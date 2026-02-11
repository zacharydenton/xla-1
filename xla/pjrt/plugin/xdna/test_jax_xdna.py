#!/usr/bin/env python3
"""JAX integration test for the XDNA NPU PJRT plugin.

Runs progressive tests from device discovery through element-wise ops.

Note: The XDNA lowering supports multi-op linalg.generic bodies (via
GenerateMultiOpScalarLoop), but XLA/StableHLO-to-linalg produces separate
linalg ops for each StableHLO op. The only case that produces a multi-op
generic body is f16 ops (extf->addf->truncf), which is blocked by a Peano
bug (fpext half<->float uses bf16 conversion instructions). Multi-op tests
will be added when Peano fixes f16 support.

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
faulthandler.dump_traceback_later(310, exit=True)

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
signal.alarm(300)  # 5 minutes before alarm fires

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


def test_i32_add_256():
    """Test 14: i32[256] add — vectorized add <16 x i32> (native)."""
    a = np.arange(1, 257, dtype=np.int32)
    b = np.full(256, 10, dtype=np.int32)
    result = jax.jit(lambda x, y: x + y)(a, b)
    np.testing.assert_array_equal(np.array(result), a + b)


def test_i16_add_256():
    """Test 15: i16[256] add — vectorized add <32 x i16> (native)."""
    a = np.arange(1, 257, dtype=np.int16)
    b = np.full(256, 10, dtype=np.int16)
    result = jax.jit(lambda x, y: x + y)(a, b)
    np.testing.assert_array_equal(np.array(result), a + b)


def test_i8_add_256():
    """Test 16: i8[256] add — vectorized add <64 x i8> (native)."""
    a = np.tile(np.arange(1, 65, dtype=np.int8), 4)  # 256 elements in i8 range
    b = np.full(256, 2, dtype=np.int8)
    result = jax.jit(lambda x, y: x + y)(a, b)
    np.testing.assert_array_equal(np.array(result), (a + b).astype(np.int8))


def _make_op_test(op_fn, a, b, expected, rtol):
    """Create a test closure for a single (op, dtype, size) combo."""
    def test():
        if b is not None:
            result = jax.jit(op_fn)(a, b)
        else:
            result = jax.jit(op_fn)(a)
        np.testing.assert_allclose(
            np.array(result, dtype=np.float32), expected,
            rtol=rtol, atol=1e-5)
    return test


def _make_tiling_tests():
    """Generate tiling tests for N=1024,4096,65536 × f32/bf16 × add/sub/mul/neg."""
    import ml_dtypes
    tests = []
    for n in [1024, 4096, 65536]:
        for dtype, dname, rtol in [
            (np.float32, "f32", 1e-5),
            (ml_dtypes.bfloat16, "bf16", 1e-2),
        ]:
            rng = np.random.RandomState(42 + n)
            a = rng.randn(n).astype(dtype)
            b = rng.randn(n).astype(dtype)
            a32, b32 = a.astype(np.float32), b.astype(np.float32)
            # f32 mul uses bf16 workaround → relaxed tolerance.
            # Random data with large values causes ~2% relative error
            # due to bf16 truncation of mantissa bits.
            mul_rtol = 2e-2
            ops = [
                ("add", lambda x, y: x + y, b, a32 + b32, rtol),
                ("sub", lambda x, y: x - y, b, a32 - b32, rtol),
                ("mul", lambda x, y: x * y, b, a32 * b32, mul_rtol),
                ("neg", lambda x: -x, None, -a32, rtol),
            ]
            for op_name, op_fn, b_arg, expected, op_rtol in ops:
                tests.append((
                    f"jit({op_name}) {dname}[{n}]",
                    _make_op_test(op_fn, a, b_arg, expected, op_rtol),
                ))
    return tests


def test_multicore_auto_reduce():
    """Test: 1006 elements not divisible by 4 or 3, auto-reduces to 2 cores.

    f32[1006]: 1006 % 4 = 2, 1006 % 3 = 2, 1006 % 2 = 0 → 2 cores.
    We verify the computation is correct with no tail-drop.
    """
    a = np.arange(1006, dtype=np.float32)
    b = np.ones(1006, dtype=np.float32)
    result = jax.jit(lambda x, y: x + y)(a, b)
    np.testing.assert_allclose(np.array(result), a + b, rtol=1e-5)


def test_multicore_single_fallback():
    """Test: Prime size (7) forces single-core fallback.

    f32[7]: 7 % 4 != 0, 7 % 3 != 0, 7 % 2 != 0, so falls back to 1 core.
    """
    a = np.arange(7, dtype=np.float32)
    b = np.ones(7, dtype=np.float32)
    result = jax.jit(lambda x, y: x + y)(a, b)
    np.testing.assert_allclose(np.array(result), a + b, rtol=1e-5)


def test_multicore_override_1():
    """Test: XDNA_NUM_CORES=1 forces single-core execution.

    Runs a subprocess with XDNA_NUM_CORES=1 to verify the override is
    respected. The subprocess compiles and runs f32[256] add (which would
    normally use 4 cores) and checks correctness.
    """
    import subprocess
    script = (
        "import os; os.environ['XDNA_NUM_CORES']='1'; "
        "os.environ['JAX_PLATFORMS']='xdna'; "
        "import sys; sys.path.insert(0,'.'); "
        "from jax_xdna import initialize; initialize(); "
        "import jax; import numpy as np; "
        "a=np.arange(256,dtype=np.float32); b=np.ones(256,dtype=np.float32); "
        "r=np.array(jax.jit(lambda x,y:x+y)(a,b)); "
        "np.testing.assert_allclose(r,a+b,rtol=1e-5); "
        "print('override OK')"
    )
    result = subprocess.run(
        [sys.executable, "-c", script],
        capture_output=True, text=True, timeout=120,
        cwd=os.path.dirname(os.path.abspath(__file__)),
    )
    assert result.returncode == 0, (
        f"Override subprocess failed:\nstdout: {result.stdout}\n"
        f"stderr: {result.stderr[-500:]}"
    )
    assert 'using 1 core(s)' in result.stderr, (
        f"Expected 'using 1 core(s)' in compiler log but got:\n"
        f"{result.stderr[-500:]}"
    )
    assert "override OK" in result.stdout


def test_matmul_f32_identity():
    """Matmul: f32 [4,4] @ [4,4] identity matrix passthrough."""
    a = np.array([[1, 2, 3, 4],
                  [5, 6, 7, 8],
                  [9, 10, 11, 12],
                  [13, 14, 15, 16]], dtype=np.float32)
    b = np.eye(4, dtype=np.float32)
    result = jax.jit(jnp.dot)(a, b)
    np.testing.assert_allclose(np.array(result), a @ b, atol=0.5)


def test_matmul_f32_16x16():
    """Matmul: f32 [16,16] @ [16,16] — single tile."""
    rng = np.random.RandomState(42)
    a = rng.randn(16, 16).astype(np.float32)
    b = rng.randn(16, 16).astype(np.float32)
    result = jax.jit(jnp.dot)(a, b)
    np.testing.assert_allclose(np.array(result), a @ b, atol=0.5)


def test_matmul_f32_32x32():
    """Matmul: f32 [32,32] @ [32,32]."""
    rng = np.random.RandomState(43)
    a = rng.randn(32, 32).astype(np.float32)
    b = rng.randn(32, 32).astype(np.float32)
    result = jax.jit(jnp.dot)(a, b)
    np.testing.assert_allclose(np.array(result), a @ b, atol=0.5)


def test_matmul_f32_rect():
    """Matmul: f32 [8,16] @ [16,4] — rectangular M!=K!=N."""
    rng = np.random.RandomState(44)
    a = rng.randn(8, 16).astype(np.float32)
    b = rng.randn(16, 4).astype(np.float32)
    result = jax.jit(jnp.dot)(a, b)
    np.testing.assert_allclose(np.array(result), a @ b, atol=0.5)


def test_matmul_f32_64x64():
    """Matmul: f32 [64,64] @ [64,64] — multi-tile with K-accumulation."""
    rng = np.random.RandomState(45)
    a = rng.randn(64, 64).astype(np.float32)
    b = rng.randn(64, 64).astype(np.float32)
    result = jax.jit(jnp.dot)(a, b)
    np.testing.assert_allclose(np.array(result), a @ b, atol=0.5)


def test_matmul_bf16_32x32():
    """Matmul: bf16 [32,32] @ [32,32] — native multiply, f32 accumulation."""
    import ml_dtypes
    rng = np.random.RandomState(46)
    a = rng.randn(32, 32).astype(ml_dtypes.bfloat16)
    b = rng.randn(32, 32).astype(ml_dtypes.bfloat16)
    result = jax.jit(jnp.dot)(a, b)
    expected = a.astype(np.float32) @ b.astype(np.float32)
    np.testing.assert_allclose(np.array(result, dtype=np.float32), expected, atol=0.5)


def test_matmul_bf16_64x64():
    """Matmul: bf16 [64,64] @ [64,64] — multi-tile bf16."""
    import ml_dtypes
    rng = np.random.RandomState(47)
    a = rng.randn(64, 64).astype(ml_dtypes.bfloat16)
    b = rng.randn(64, 64).astype(ml_dtypes.bfloat16)
    result = jax.jit(jnp.dot)(a, b)
    expected = a.astype(np.float32) @ b.astype(np.float32)
    np.testing.assert_allclose(np.array(result, dtype=np.float32), expected, atol=0.5)


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
        ("jit(x + y) i32[256]", test_i32_add_256),
        ("jit(x + y) i16[256]", test_i16_add_256),
        ("jit(x + y) i8[256]", test_i8_add_256),
    ] + _make_tiling_tests() + [
        ("matmul f32 identity [4,4]@[4,4]", test_matmul_f32_identity),
        ("matmul f32 [16,16]@[16,16]", test_matmul_f32_16x16),
        ("matmul f32 [32,32]@[32,32]", test_matmul_f32_32x32),
        ("matmul f32 rect [8,16]@[16,4]", test_matmul_f32_rect),
        ("matmul f32 [64,64]@[64,64]", test_matmul_f32_64x64),
        ("matmul bf16 [32,32]@[32,32]", test_matmul_bf16_32x32),
        ("matmul bf16 [64,64]@[64,64]", test_matmul_bf16_64x64),
    ] + [
        ("multicore auto-reduce f32[1006]", test_multicore_auto_reduce),
        ("multicore single fallback f32[7]", test_multicore_single_fallback),
        ("multicore override f32[256]", test_multicore_override_1),
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
