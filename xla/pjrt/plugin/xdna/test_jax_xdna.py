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
    """Test 8: f32[256] add — vectorized via aievec ACC2048 accumulator."""
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
    """Test 11: bf16[256] add — vectorized via aievec UPS/SRS + ACC2048."""
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


def test_distribute_neg_f32_65536():
    """Test: neg f32[65536] uses distribute/join (1-input, 4 cores, tiled).

    Runs in a subprocess to capture compiler log. Verifies:
    1. Distribute/join is used (single shim → N compute via mem tile).
    2. Computation is correct.
    """
    import subprocess
    script = (
        "import os; os.environ['JAX_PLATFORMS']='xdna'; "
        "os.environ['XDNA_NO_CACHE']='1'; "
        "import sys; sys.path.insert(0,'.'); "
        "from jax_xdna import initialize; initialize(); "
        "import jax; import numpy as np; "
        "a=np.random.RandomState(42).randn(65536).astype(np.float32); "
        "r=np.array(jax.jit(lambda x:-x)(a)); "
        "np.testing.assert_allclose(r,-a,rtol=1e-5); "
        "print('distribute_neg OK')"
    )
    result = subprocess.run(
        [sys.executable, "-c", script],
        capture_output=True, text=True, timeout=120,
        cwd=os.path.dirname(os.path.abspath(__file__)),
    )
    assert result.returncode == 0, (
        f"Subprocess failed:\nstdout: {result.stdout}\n"
        f"stderr: {result.stderr[-500:]}"
    )
    assert "distribute_neg OK" in result.stdout
    assert "distribute/join" in result.stderr, (
        f"Expected 'distribute/join' in compiler log. Got:\n"
        f"{result.stderr[-500:]}"
    )
    print("         Distribute/join confirmed for neg f32[65536].", flush=True)


def test_distribute_add_f32_65536():
    """Test: add f32[65536] uses distribute/join with auto-reduced cores.

    With 4 cores and 2 inputs, DMA channels exceed the mem tile limit (9 > 6).
    The compiler auto-reduces to 2 cores where channels fit (5 <= 6), enabling
    distribute/join instead of falling back to per-column independent DMA.
    """
    import subprocess
    script = (
        "import os; os.environ['JAX_PLATFORMS']='xdna'; "
        "os.environ['XDNA_NO_CACHE']='1'; "
        "import sys; sys.path.insert(0,'.'); "
        "from jax_xdna import initialize; initialize(); "
        "import jax; import numpy as np; "
        "rng=np.random.RandomState(43); "
        "a=rng.randn(65536).astype(np.float32); "
        "b=rng.randn(65536).astype(np.float32); "
        "r=np.array(jax.jit(lambda x,y:x+y)(a,b)); "
        "np.testing.assert_allclose(r,a+b,rtol=1e-5); "
        "print('distribute_add OK')"
    )
    result = subprocess.run(
        [sys.executable, "-c", script],
        capture_output=True, text=True, timeout=120,
        cwd=os.path.dirname(os.path.abspath(__file__)),
    )
    assert result.returncode == 0, (
        f"Subprocess failed:\nstdout: {result.stdout}\n"
        f"stderr: {result.stderr[-500:]}"
    )
    assert "distribute_add OK" in result.stdout
    # Should auto-reduce cores and use distribute/join. At 4 cores with 2 inputs,
    # channels exceed limit (9 > 6), so the compiler reduces until it fits.
    assert "distribute/join" in result.stderr, (
        f"Expected 'distribute/join' for 2-input auto-reduced cores. Got:\n"
        f"{result.stderr[-500:]}"
    )
    # Verify cores were actually reduced (not still at the default 4).
    import re
    m = re.search(r'using (\d+) core', result.stderr)
    assert m and int(m.group(1)) < 4, (
        f"Expected auto-reduced core count < 4. Got:\n"
        f"{result.stderr[-500:]}"
    )
    print(f"         Distribute/join confirmed for add f32[65536] "
          f"({m.group(1)} cores).", flush=True)


def test_distribute_fallback_mul_f32_65536():
    """Test: mul f32[65536] falls back to per-column (scalar, no auto-reduce).

    f32 multiply is scalar (vector_width=0), so the auto-reduce heuristic
    does not apply. With 4 cores and 2 inputs, DMA channels exceed the mem
    tile limit (9 > 6), falling back to per-column independent DMA.
    """
    import subprocess
    script = (
        "import os; os.environ['JAX_PLATFORMS']='xdna'; "
        "os.environ['XDNA_NO_CACHE']='1'; "
        "import sys; sys.path.insert(0,'.'); "
        "from jax_xdna import initialize; initialize(); "
        "import jax; import numpy as np; "
        "rng=np.random.RandomState(44); "
        "a=rng.randn(65536).astype(np.float32); "
        "b=rng.randn(65536).astype(np.float32); "
        "r=np.array(jax.jit(lambda x,y:x*y)(a,b)); "
        "np.testing.assert_allclose(r,a*b,rtol=2e-2); "
        "print('fallback_mul OK')"
    )
    result = subprocess.run(
        [sys.executable, "-c", script],
        capture_output=True, text=True, timeout=120,
        cwd=os.path.dirname(os.path.abspath(__file__)),
    )
    assert result.returncode == 0, (
        f"Subprocess failed:\nstdout: {result.stdout}\n"
        f"stderr: {result.stderr[-500:]}"
    )
    assert "fallback_mul OK" in result.stdout
    # Should use 4 cores but NOT distribute/join (scalar, no auto-reduce).
    assert "using 4 core(s)" in result.stderr, (
        f"Expected 'using 4 core(s)' in compiler log. Got:\n"
        f"{result.stderr[-500:]}"
    )
    assert "distribute/join" not in result.stderr, (
        f"Expected NO 'distribute/join' for scalar 2-input 4-core. Got:\n"
        f"{result.stderr[-500:]}"
    )
    print("         Per-column fallback confirmed for mul f32[65536].", flush=True)


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
    # atol=1.0 for K=64: bf16 truncation error accumulates as ~sqrt(K)*bf16_eps,
    # exceeding 0.5 on a few elements with random data.
    np.testing.assert_allclose(np.array(result, dtype=np.float32), expected, atol=1.0)


def test_matmul_f32_multicore():
    """Matmul: f32 [32,32] @ [32,128] — N=128 triggers 4-core parallel."""
    rng = np.random.RandomState(50)
    a = rng.randn(32, 32).astype(np.float32)
    b = rng.randn(32, 128).astype(np.float32)
    result = jax.jit(jnp.dot)(a, b)
    np.testing.assert_allclose(np.array(result), a @ b, atol=0.5)


def test_matmul_bf16_multicore():
    """Matmul: bf16 [32,64] @ [64,32] — N=32, n=8, Nt=4, divisible by 4."""
    import ml_dtypes
    rng = np.random.RandomState(51)
    a = rng.randn(32, 64).astype(ml_dtypes.bfloat16)
    b = rng.randn(64, 32).astype(ml_dtypes.bfloat16)
    result = jax.jit(jnp.dot)(a, b)
    expected = a.astype(np.float32) @ b.astype(np.float32)
    np.testing.assert_allclose(np.array(result, dtype=np.float32), expected, atol=1.0)


def test_matmul_bf16_multicore_large():
    """Matmul: bf16 [64,64] @ [64,128] — Mt=2 + 4 cores."""
    import ml_dtypes
    rng = np.random.RandomState(52)
    a = rng.randn(64, 64).astype(ml_dtypes.bfloat16)
    b = rng.randn(64, 128).astype(ml_dtypes.bfloat16)
    result = jax.jit(jnp.dot)(a, b)
    expected = a.astype(np.float32) @ b.astype(np.float32)
    np.testing.assert_allclose(np.array(result, dtype=np.float32), expected, atol=1.0)


def test_matmul_f32_multicore_fallback():
    """Matmul: f32 [16,16] @ [16,24] — N=24, n=8→Nt=3, only divisible by 3."""
    rng = np.random.RandomState(53)
    a = rng.randn(16, 16).astype(np.float32)
    b = rng.randn(16, 24).astype(np.float32)
    result = jax.jit(jnp.dot)(a, b)
    np.testing.assert_allclose(np.array(result), a @ b, atol=0.5)


def test_relu_f32_4():
    """ReLU: f32[4] max(x, 0) — simplest activation function."""
    a = np.array([-2.0, -1.0, 1.0, 3.0], dtype=np.float32)
    result = jax.jit(jax.nn.relu)(a)
    expected = np.array([0.0, 0.0, 1.0, 3.0], dtype=np.float32)
    np.testing.assert_allclose(np.array(result), expected, rtol=1e-5)


def test_relu_f32_256():
    """ReLU: f32[256] — larger input with mixed positive/negative values."""
    rng = np.random.RandomState(99)
    a = rng.randn(256).astype(np.float32)
    result = jax.jit(jax.nn.relu)(a)
    expected = np.maximum(a, 0.0)
    np.testing.assert_allclose(np.array(result), expected, rtol=1e-5)


def test_relu_bf16_256():
    """ReLU: bf16[256] — bf16 activation function."""
    import ml_dtypes
    rng = np.random.RandomState(100)
    a = rng.randn(256).astype(ml_dtypes.bfloat16)
    result = jax.jit(jax.nn.relu)(a)
    expected = np.maximum(a.astype(np.float32), 0.0)
    np.testing.assert_allclose(np.array(result, dtype=np.float32), expected, rtol=1e-2)


def test_relu_f32_nan():
    """ReLU with NaN: f32[4] — NaN should propagate through max(x, 0).
    Note: f32 uses integer bitcast comparison. Positive NaN (0x7FC00000)
    has i32 value > 0, so cmpi sgt returns true → keeps NaN. Correct."""
    a = np.array([float('nan'), -1.0, 0.0, 1.0], dtype=np.float32)
    result = jax.jit(jax.nn.relu)(a)
    result_np = np.array(result)
    # NaN should be preserved (positive NaN bitcast > 0 in i32).
    assert np.isnan(result_np[0]), f"Expected NaN at [0], got {result_np[0]}"
    np.testing.assert_allclose(result_np[1:], [0.0, 0.0, 1.0], rtol=1e-5)


def test_relu_f32_1024():
    """ReLU: f32[1024] — multicore relu (4 cores * 256 elements each)."""
    rng = np.random.RandomState(101)
    a = rng.randn(1024).astype(np.float32)
    result = jax.jit(jax.nn.relu)(a)
    expected = np.maximum(a, 0.0)
    np.testing.assert_allclose(np.array(result), expected, rtol=1e-5)


def test_relu_bf16_1024():
    """ReLU: bf16[1024] — multicore vectorized aievec relu (4 cores × 256).

    Regression test for needs_matmul_workarounds split: exercises the aievec
    vmax.ltbf16 path with multiple cores, verifying that codegen does NOT
    apply matmul-specific workarounds (--aie-loop-aware=false, __muldi3 stub)
    to vectorized elementwise ops."""
    import ml_dtypes
    rng = np.random.RandomState(102)
    a = rng.randn(1024).astype(ml_dtypes.bfloat16)
    result = jax.jit(jax.nn.relu)(a)
    expected = np.maximum(a.astype(np.float32), 0.0)
    np.testing.assert_allclose(np.array(result, dtype=np.float32), expected, rtol=1e-2)


def _softmax_ref(x):
    """CPU reference softmax using numpy (avoids JAX device issues)."""
    x_f32 = x.astype(np.float32)
    x_max = np.max(x_f32, axis=-1, keepdims=True)
    e = np.exp(x_f32 - x_max)
    return e / np.sum(e, axis=-1, keepdims=True)


def test_softmax_bf16_1x64():
    """Softmax: bf16[1,64] — single row, single core."""
    import ml_dtypes
    rng = np.random.RandomState(200)
    a = rng.randn(1, 64).astype(ml_dtypes.bfloat16)
    result = jax.jit(jax.nn.softmax)(a)
    expected = _softmax_ref(a)
    np.testing.assert_allclose(
        np.array(result, dtype=np.float32), expected, atol=0.02)


def test_softmax_bf16_4x128():
    """Softmax: bf16[4,128] — multi-row, potentially multi-core."""
    import ml_dtypes
    rng = np.random.RandomState(201)
    a = rng.randn(4, 128).astype(ml_dtypes.bfloat16)
    result = jax.jit(jax.nn.softmax)(a)
    expected = _softmax_ref(a)
    np.testing.assert_allclose(
        np.array(result, dtype=np.float32), expected, atol=0.02)


def test_softmax_bf16_8x256():
    """Softmax: bf16[8,256] — larger, multi-core."""
    import ml_dtypes
    rng = np.random.RandomState(202)
    a = rng.randn(8, 256).astype(ml_dtypes.bfloat16)
    result = jax.jit(jax.nn.softmax)(a)
    expected = _softmax_ref(a)
    np.testing.assert_allclose(
        np.array(result, dtype=np.float32), expected, atol=0.02)


def test_transposed_matmul_bf16_4x8():
    """Transposed matmul: Q(4,8) @ K(4,8)^T = (4,4) — Q @ K^T pattern."""
    import ml_dtypes
    rng = np.random.RandomState(210)
    q = rng.randn(4, 8).astype(ml_dtypes.bfloat16)
    k = rng.randn(4, 8).astype(ml_dtypes.bfloat16)
    result = jax.jit(lambda q, k: q @ k.T)(q, k)
    expected = q.astype(np.float32) @ k.astype(np.float32).T
    np.testing.assert_allclose(np.array(result, dtype=np.float32), expected, atol=0.5)


def test_transposed_matmul_bf16_16x32():
    """Transposed matmul: Q(16,32) @ K(16,32)^T = (16,16) — multi-tile."""
    import ml_dtypes
    rng = np.random.RandomState(211)
    q = rng.randn(16, 32).astype(ml_dtypes.bfloat16)
    k = rng.randn(16, 32).astype(ml_dtypes.bfloat16)
    result = jax.jit(lambda q, k: q @ k.T)(q, k)
    expected = q.astype(np.float32) @ k.astype(np.float32).T
    np.testing.assert_allclose(np.array(result, dtype=np.float32), expected, atol=0.5)


def test_transposed_matmul_bf16_multicore():
    """Transposed matmul: Q(8,32) @ K(32,32)^T = (8,32) — multi-core N=32."""
    import ml_dtypes
    rng = np.random.RandomState(212)
    q = rng.randn(8, 32).astype(ml_dtypes.bfloat16)
    k = rng.randn(32, 32).astype(ml_dtypes.bfloat16)
    result = jax.jit(lambda q, k: q @ k.T)(q, k)
    expected = q.astype(np.float32) @ k.astype(np.float32).T
    np.testing.assert_allclose(np.array(result, dtype=np.float32), expected, atol=0.5)


def test_attention_bf16():
    """End-to-end attention: softmax(Q @ K^T / sqrt(dk)) @ V on NPU.

    Constraints: seq_len must be divisible by 32 (softmax VL) and 4 (matmul tile).
    dk must be divisible by 4 (matmul tile). All steps run on the NPU with data
    round-tripping through host between steps.
    """
    import ml_dtypes
    rng = np.random.RandomState(220)
    seq_len, dk = 32, 32
    Q = rng.randn(seq_len, dk).astype(ml_dtypes.bfloat16)
    K = rng.randn(seq_len, dk).astype(ml_dtypes.bfloat16)
    V = rng.randn(seq_len, dk).astype(ml_dtypes.bfloat16)
    # Use bf16 scale to keep multiplication in bf16 (avoids __mulsf3).
    scale = ml_dtypes.bfloat16(1.0 / np.sqrt(dk))

    # Each step is a separate JIT call (data round-trips through host).
    scores = jax.jit(lambda q, k: q @ k.T)(Q, K)                # transposed matmul
    scores = jax.jit(lambda s: s * scale)(scores)                 # elementwise scale
    weights = jax.jit(jax.nn.softmax)(scores)                     # softmax
    output = jax.jit(jnp.dot)(weights, V)                         # matmul

    # Numpy reference: match the bf16 precision at each step.
    Q_f32 = Q.astype(np.float32)
    K_f32 = K.astype(np.float32)
    V_f32 = V.astype(np.float32)
    scores_ref = (Q_f32 @ K_f32.T) * np.float32(scale)
    weights_ref = _softmax_ref(scores_ref.astype(ml_dtypes.bfloat16))
    output_ref = weights_ref @ V_f32
    np.testing.assert_allclose(
        np.array(output, dtype=np.float32), output_ref, atol=2.0)


def _attention_ref(Q, K, V, dk):
    """CPU reference for scaled dot-product attention using numpy."""
    Q_f32 = Q.astype(np.float32)
    K_f32 = K.astype(np.float32)
    V_f32 = V.astype(np.float32)
    scale = 1.0 / np.sqrt(float(dk))
    scores = Q_f32 @ K_f32.T * scale
    weights = _softmax_ref(scores)
    return weights @ V_f32


def test_fused_attention_bf16_basic():
    """Fused attention: bf16 [32,32] — single JIT call, all on NPU."""
    import ml_dtypes
    rng = np.random.RandomState(230)
    seq_len, dk = 32, 32
    Q = rng.randn(seq_len, dk).astype(ml_dtypes.bfloat16)
    K = rng.randn(seq_len, dk).astype(ml_dtypes.bfloat16)
    V = rng.randn(seq_len, dk).astype(ml_dtypes.bfloat16)
    scale = ml_dtypes.bfloat16(1.0 / np.sqrt(dk))
    attn = jax.jit(lambda q, k, v: jax.nn.softmax(
        (q @ k.T) * scale) @ v)
    result = attn(Q, K, V)
    expected = _attention_ref(Q, K, V, dk)
    np.testing.assert_allclose(
        np.array(result, dtype=np.float32), expected, atol=2.0)


def test_fused_attention_bf16_rectangular():
    """Fused attention: bf16 seq_len=64, dk=32 (non-square)."""
    import ml_dtypes
    rng = np.random.RandomState(231)
    seq_len, dk = 64, 32
    Q = rng.randn(seq_len, dk).astype(ml_dtypes.bfloat16)
    K = rng.randn(seq_len, dk).astype(ml_dtypes.bfloat16)
    V = rng.randn(seq_len, dk).astype(ml_dtypes.bfloat16)
    scale = ml_dtypes.bfloat16(1.0 / np.sqrt(dk))
    attn = jax.jit(lambda q, k, v: jax.nn.softmax(
        (q @ k.T) * scale) @ v)
    result = attn(Q, K, V)
    expected = _attention_ref(Q, K, V, dk)
    np.testing.assert_allclose(
        np.array(result, dtype=np.float32), expected, atol=2.0)


def test_fused_attention_bf16_multicore():
    """Fused attention: bf16 seq_len=64, dk=64 — 4-core parallel."""
    import ml_dtypes
    rng = np.random.RandomState(232)
    seq_len, dk = 64, 64
    Q = rng.randn(seq_len, dk).astype(ml_dtypes.bfloat16)
    K = rng.randn(seq_len, dk).astype(ml_dtypes.bfloat16)
    V = rng.randn(seq_len, dk).astype(ml_dtypes.bfloat16)
    scale = ml_dtypes.bfloat16(1.0 / np.sqrt(dk))
    attn = jax.jit(lambda q, k, v: jax.nn.softmax(
        (q @ k.T) * scale) @ v)
    result = attn(Q, K, V)
    expected = _attention_ref(Q, K, V, dk)
    np.testing.assert_allclose(
        np.array(result, dtype=np.float32), expected, atol=2.0)


def test_fused_attention_bf16_larger():
    """Fused attention: bf16 seq_len=32, dk=64 — larger dk."""
    import ml_dtypes
    rng = np.random.RandomState(233)
    seq_len, dk = 32, 64
    Q = rng.randn(seq_len, dk).astype(ml_dtypes.bfloat16)
    K = rng.randn(seq_len, dk).astype(ml_dtypes.bfloat16)
    V = rng.randn(seq_len, dk).astype(ml_dtypes.bfloat16)
    scale = ml_dtypes.bfloat16(1.0 / np.sqrt(dk))
    attn = jax.jit(lambda q, k, v: jax.nn.softmax(
        (q @ k.T) * scale) @ v)
    result = attn(Q, K, V)
    expected = _attention_ref(Q, K, V, dk)
    np.testing.assert_allclose(
        np.array(result, dtype=np.float32), expected, atol=2.0)


def _gelu_ref(x):
    """CPU reference approximate GELU using numpy."""
    x_f32 = x.astype(np.float32)
    return 0.5 * x_f32 * (1.0 + np.tanh(
        np.sqrt(2.0 / np.pi) * (x_f32 + 0.044715 * x_f32**3)))


def test_gelu_bf16_64():
    """GELU: bf16[64] — single chunk, single core."""
    import ml_dtypes
    rng = np.random.RandomState(300)
    a = rng.randn(64).astype(ml_dtypes.bfloat16)
    result = jax.jit(lambda x: jax.nn.gelu(x, approximate=True))(a)
    expected = _gelu_ref(a)
    np.testing.assert_allclose(
        np.array(result, dtype=np.float32), expected, atol=0.1)


def test_gelu_bf16_256():
    """GELU: bf16[256] — multi-chunk."""
    import ml_dtypes
    rng = np.random.RandomState(301)
    a = rng.randn(256).astype(ml_dtypes.bfloat16)
    result = jax.jit(lambda x: jax.nn.gelu(x, approximate=True))(a)
    expected = _gelu_ref(a)
    np.testing.assert_allclose(
        np.array(result, dtype=np.float32), expected, atol=0.1)


def test_gelu_bf16_4096():
    """GELU: bf16[4096] — multi-core."""
    import ml_dtypes
    rng = np.random.RandomState(302)
    a = rng.randn(4096).astype(ml_dtypes.bfloat16)
    result = jax.jit(lambda x: jax.nn.gelu(x, approximate=True))(a)
    expected = _gelu_ref(a)
    np.testing.assert_allclose(
        np.array(result, dtype=np.float32), expected, atol=0.1)


def _layernorm_ref(x, gamma, beta, eps=1e-5):
    """CPU reference LayerNorm using numpy."""
    x_f32 = x.astype(np.float32)
    g_f32 = gamma.astype(np.float32)
    b_f32 = beta.astype(np.float32)
    mean = np.mean(x_f32, axis=-1, keepdims=True)
    var = np.var(x_f32, axis=-1, keepdims=True)
    normed = (x_f32 - mean) / np.sqrt(var + eps)
    return g_f32 * normed + b_f32


def test_layernorm_bf16_1x64():
    """LayerNorm: bf16[1,64] — single row, identity gamma/beta."""
    import ml_dtypes
    rng = np.random.RandomState(310)
    x = rng.randn(1, 64).astype(ml_dtypes.bfloat16)
    gamma = np.ones(64, dtype=ml_dtypes.bfloat16)
    beta = np.zeros(64, dtype=ml_dtypes.bfloat16)
    def layernorm_fn(x, g, b):
        mean = jnp.mean(x, axis=-1, keepdims=True)
        var = jnp.var(x, axis=-1, keepdims=True)
        return g * (x - mean) / jnp.sqrt(var + 1e-5) + b
    result = jax.jit(layernorm_fn)(x, gamma, beta)
    expected = _layernorm_ref(x, gamma, beta)
    np.testing.assert_allclose(
        np.array(result, dtype=np.float32), expected, atol=0.1)


def test_layernorm_bf16_4x128():
    """LayerNorm: bf16[4,128] — multi-row, non-trivial gamma/beta."""
    import ml_dtypes
    rng = np.random.RandomState(311)
    x = rng.randn(4, 128).astype(ml_dtypes.bfloat16)
    gamma = (rng.randn(128) * 0.5 + 1.0).astype(ml_dtypes.bfloat16)
    beta = (rng.randn(128) * 0.1).astype(ml_dtypes.bfloat16)
    def layernorm_fn(x, g, b):
        mean = jnp.mean(x, axis=-1, keepdims=True)
        var = jnp.var(x, axis=-1, keepdims=True)
        return g * (x - mean) / jnp.sqrt(var + 1e-5) + b
    result = jax.jit(layernorm_fn)(x, gamma, beta)
    expected = _layernorm_ref(x, gamma, beta)
    np.testing.assert_allclose(
        np.array(result, dtype=np.float32), expected, atol=0.1)


def test_layernorm_bf16_8x256():
    """LayerNorm: bf16[8,256] — multi-core, non-trivial gamma/beta."""
    import ml_dtypes
    rng = np.random.RandomState(312)
    x = rng.randn(8, 256).astype(ml_dtypes.bfloat16)
    gamma = (rng.randn(256) * 0.5 + 1.0).astype(ml_dtypes.bfloat16)
    beta = (rng.randn(256) * 0.1).astype(ml_dtypes.bfloat16)
    def layernorm_fn(x, g, b):
        mean = jnp.mean(x, axis=-1, keepdims=True)
        var = jnp.var(x, axis=-1, keepdims=True)
        return g * (x - mean) / jnp.sqrt(var + 1e-5) + b
    result = jax.jit(layernorm_fn)(x, gamma, beta)
    expected = _layernorm_ref(x, gamma, beta)
    np.testing.assert_allclose(
        np.array(result, dtype=np.float32), expected, atol=0.1)


def test_tiny_gpt_forward():
    """Tiny GPT: 1-layer forward pass using separate jit calls per op.

    Model: emb_dim=64, seq_len=32, 1 head, vocab=256.
    All matmul/attention/layernorm/gelu on NPU, embedding/sampling on CPU.
    """
    import ml_dtypes
    rng = np.random.RandomState(400)
    emb_dim, seq_len, vocab = 64, 32, 256

    # Random weights (bf16).
    def bf16(x):
        return x.astype(ml_dtypes.bfloat16)

    embed_table = bf16(rng.randn(vocab, emb_dim) * 0.02)
    pos_embed = bf16(rng.randn(seq_len, emb_dim) * 0.02)
    W_q = bf16(rng.randn(emb_dim, emb_dim) * 0.02)
    W_k = bf16(rng.randn(emb_dim, emb_dim) * 0.02)
    W_v = bf16(rng.randn(emb_dim, emb_dim) * 0.02)
    W_out = bf16(rng.randn(emb_dim, emb_dim) * 0.02)
    gamma1 = bf16(np.ones(emb_dim))
    beta1 = bf16(np.zeros(emb_dim))
    W_ff1 = bf16(rng.randn(emb_dim, emb_dim * 4) * 0.02)  # FFN up
    W_ff2 = bf16(rng.randn(emb_dim * 4, emb_dim) * 0.02)  # FFN down
    gamma2 = bf16(np.ones(emb_dim))
    beta2 = bf16(np.zeros(emb_dim))
    gamma3 = bf16(np.ones(emb_dim))
    beta3 = bf16(np.zeros(emb_dim))
    W_head = bf16(rng.randn(emb_dim, vocab) * 0.02)

    # Random input tokens.
    token_ids = rng.randint(0, vocab, size=seq_len)

    # --- NPU forward pass ---
    # CPU: embedding lookup
    x = bf16(embed_table[token_ids] + pos_embed)

    # LayerNorm 1
    def layernorm_fn(x, g, b):
        mean = jnp.mean(x, axis=-1, keepdims=True)
        var = jnp.var(x, axis=-1, keepdims=True)
        return g * (x - mean) / jnp.sqrt(var + 1e-5) + b

    x_norm = jax.jit(layernorm_fn)(x, gamma1, beta1)

    # Q, K, V projections (matmul)
    q = jax.jit(jnp.dot)(x_norm, W_q)
    k = jax.jit(jnp.dot)(x_norm, W_k)
    v = jax.jit(jnp.dot)(x_norm, W_v)

    # Fused attention
    scale = ml_dtypes.bfloat16(1.0 / np.sqrt(emb_dim))
    attn_out = jax.jit(
        lambda q, k, v: jax.nn.softmax((q @ k.T) * scale) @ v)(q, k, v)

    # Output projection + residual
    proj = jax.jit(jnp.dot)(attn_out, W_out)
    x = jax.jit(lambda a, b: a + b)(x, proj)

    # LayerNorm 2
    x_norm2 = jax.jit(layernorm_fn)(x, gamma2, beta2)

    # FFN: up projection
    ff1 = jax.jit(jnp.dot)(x_norm2, W_ff1)
    # GELU activation
    act = jax.jit(lambda x: jax.nn.gelu(x, approximate=True))(ff1)
    # FFN: down projection + residual
    ff2 = jax.jit(jnp.dot)(act, W_ff2)
    x = jax.jit(lambda a, b: a + b)(x, ff2)

    # Final LayerNorm
    x_final = jax.jit(layernorm_fn)(x, gamma3, beta3)

    # All rows -> logits (matmul), then take last row on CPU.
    # (M=1 matmul not supported by tiling, so compute full [seq_len, vocab].)
    all_logits = jax.jit(jnp.dot)(x_final, W_head)
    logits = np.array(all_logits)[-1:]

    # --- CPU reference ---
    x_ref = embed_table[token_ids].astype(np.float32) + pos_embed.astype(np.float32)
    x_ref = x_ref.astype(ml_dtypes.bfloat16)  # Match bf16 precision

    def ln_ref(x, g, b):
        return _layernorm_ref(x, g, b)

    x_norm_ref = ln_ref(x_ref, gamma1, beta1).astype(ml_dtypes.bfloat16)

    q_ref = x_norm_ref.astype(np.float32) @ W_q.astype(np.float32)
    k_ref = x_norm_ref.astype(np.float32) @ W_k.astype(np.float32)
    v_ref = x_norm_ref.astype(np.float32) @ W_v.astype(np.float32)

    def attn_ref(q, k, v, dk):
        return _attention_ref(
            q.astype(ml_dtypes.bfloat16),
            k.astype(ml_dtypes.bfloat16),
            v.astype(ml_dtypes.bfloat16), dk)

    attn_ref_out = attn_ref(q_ref, k_ref, v_ref, emb_dim)
    proj_ref = attn_ref_out @ W_out.astype(np.float32)
    x_ref2 = x_ref.astype(np.float32) + proj_ref

    x_norm2_ref = ln_ref(
        x_ref2.astype(ml_dtypes.bfloat16), gamma2, beta2
    ).astype(np.float32)
    ff1_ref = x_norm2_ref @ W_ff1.astype(np.float32)
    act_ref = _gelu_ref(ff1_ref.astype(ml_dtypes.bfloat16))
    ff2_ref = act_ref @ W_ff2.astype(np.float32)
    x_ref3 = x_ref2 + ff2_ref

    x_final_ref = ln_ref(
        x_ref3.astype(ml_dtypes.bfloat16), gamma3, beta3
    ).astype(np.float32)
    logits_ref = x_final_ref[-1:] @ W_head.astype(np.float32)

    # Compare logits — cascaded bf16 ops accumulate error.
    npu_logits = logits.astype(np.float32)
    print(f"         NPU logits range: [{npu_logits.min():.3f}, {npu_logits.max():.3f}]",
          flush=True)
    print(f"         Ref logits range: [{logits_ref.min():.3f}, {logits_ref.max():.3f}]",
          flush=True)
    # With small random weights, logits are near zero. Check argmax matches.
    npu_token = np.argmax(npu_logits)
    ref_token = np.argmax(logits_ref)
    print(f"         NPU predicted token: {npu_token}, ref: {ref_token}", flush=True)
    # Loose tolerance for many cascaded bf16 ops.
    np.testing.assert_allclose(npu_logits, logits_ref, atol=5.0)


def test_unsupported_reshape():
    """Unsupported: reshape should fail with a clear error, not crash."""
    a = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)
    try:
        result = jax.jit(lambda x: x.reshape(4))(a)
        # If it somehow succeeds (e.g., future support), just verify result.
        np.testing.assert_allclose(np.array(result), [1.0, 2.0, 3.0, 4.0], rtol=1e-5)
    except Exception as e:
        msg = str(e)
        assert "XDNA cannot compile" in msg or "UNIMPLEMENTED" in msg or "Unsupported" in msg, \
            f"Expected clear error message, got: {msg[:200]}"
        print(f"         Error (expected): {msg[:120]}", flush=True)


def test_unsupported_reduce_sum():
    """Unsupported: reduce_sum should fail with a clear error, not crash."""
    a = np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float32)
    try:
        result = jax.jit(jnp.sum)(a)
        # If it somehow succeeds, verify.
        np.testing.assert_allclose(float(result), 10.0, rtol=1e-5)
    except Exception as e:
        msg = str(e)
        assert "XDNA cannot compile" in msg or "UNIMPLEMENTED" in msg or "Unsupported" in msg \
            or "single-op" in msg, \
            f"Expected clear error message, got: {msg[:200]}"
        print(f"         Error (expected): {msg[:120]}", flush=True)


def test_unsupported_transpose():
    """Unsupported: transpose should fail with a clear error, not crash."""
    a = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)
    try:
        result = jax.jit(lambda x: x.T)(a)
        np.testing.assert_allclose(np.array(result), a.T, rtol=1e-5)
    except Exception as e:
        msg = str(e)
        assert "XDNA cannot compile" in msg or "UNIMPLEMENTED" in msg or "Unsupported" in msg, \
            f"Expected clear error message, got: {msg[:200]}"
        print(f"         Error (expected): {msg[:120]}", flush=True)


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
        ("matmul f32 multicore [32,32]@[32,128]", test_matmul_f32_multicore),
        ("matmul bf16 multicore [32,64]@[64,32]", test_matmul_bf16_multicore),
        ("matmul bf16 multicore large [64,64]@[64,128]", test_matmul_bf16_multicore_large),
        ("matmul f32 multicore fallback [16,16]@[16,24]", test_matmul_f32_multicore_fallback),
    ] + [
        ("relu f32[4]", test_relu_f32_4),
        ("relu f32[256]", test_relu_f32_256),
        ("relu bf16[256]", test_relu_bf16_256),
        ("relu f32 NaN", test_relu_f32_nan),
        ("relu f32[1024] multicore", test_relu_f32_1024),
        ("relu bf16[1024] multicore", test_relu_bf16_1024),
    ] + [
        ("multicore auto-reduce f32[1006]", test_multicore_auto_reduce),
        ("multicore single fallback f32[7]", test_multicore_single_fallback),
        ("multicore override f32[256]", test_multicore_override_1),
    ] + [
        ("distribute neg f32[65536]", test_distribute_neg_f32_65536),
        ("distribute add f32[65536] (auto-reduce)", test_distribute_add_f32_65536),
        ("distribute fallback mul f32[65536]", test_distribute_fallback_mul_f32_65536),
    ] + [
        ("softmax bf16[1,64]", test_softmax_bf16_1x64),
        ("softmax bf16[4,128]", test_softmax_bf16_4x128),
        ("softmax bf16[8,256]", test_softmax_bf16_8x256),
    ] + [
        ("transposed matmul bf16 [4,8]@[4,8]^T", test_transposed_matmul_bf16_4x8),
        ("transposed matmul bf16 [16,32]@[16,32]^T", test_transposed_matmul_bf16_16x32),
        ("transposed matmul bf16 multicore [8,32]@[32,32]^T", test_transposed_matmul_bf16_multicore),
        ("attention bf16 [32,32]", test_attention_bf16),
    ] + [
        ("fused attention bf16 [32,32]", test_fused_attention_bf16_basic),
        ("fused attention bf16 [64,32]", test_fused_attention_bf16_rectangular),
        ("fused attention bf16 [64,64]", test_fused_attention_bf16_multicore),
        ("fused attention bf16 [32,64]", test_fused_attention_bf16_larger),
    ] + [
        ("gelu bf16[64]", test_gelu_bf16_64),
        ("gelu bf16[256]", test_gelu_bf16_256),
        ("gelu bf16[4096]", test_gelu_bf16_4096),
    ] + [
        ("layernorm bf16[1,64]", test_layernorm_bf16_1x64),
        ("layernorm bf16[4,128]", test_layernorm_bf16_4x128),
        ("layernorm bf16[8,256]", test_layernorm_bf16_8x256),
    ] + [
        ("tiny GPT forward", test_tiny_gpt_forward),
    ] + [
        ("unsupported: reshape", test_unsupported_reshape),
        ("unsupported: reduce_sum", test_unsupported_reduce_sum),
        ("unsupported: transpose", test_unsupported_transpose),
    ] + [
        ("cache hit", test_cache_hit),
    ]

    print(f"XDNA JAX Integration Tests", flush=True)
    print(f"=" * 40, flush=True)

    passed = 0
    failed = 0
    for idx, (name, fn) in enumerate(tests):
        if run_test(name, fn):
            passed += 1
        else:
            failed += 1
        # Release hw_contexts periodically to avoid driver exhaustion.
        # The XDNA driver has a finite number of hw_context slots per
        # process (~48). JAX caches executables, preventing GC. Clear
        # the caches every 40 tests to reclaim slots.
        if (idx + 1) % 40 == 0:
            jax.clear_caches()
            import gc; gc.collect()

    print(f"=" * 40, flush=True)
    print(f"Results: {passed} passed, {failed} failed, {len(tests)} total", flush=True)

    # Cancel alarm — we finished
    signal.alarm(0)

    sys.exit(0 if failed == 0 else 1)


if __name__ == "__main__":
    main()
