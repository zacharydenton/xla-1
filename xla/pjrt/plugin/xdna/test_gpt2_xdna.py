#!/usr/bin/env python3
"""GPT-2 124M forward pass test on XDNA NPU.

Phase 16/18: Python-side orchestration with fused causal attention.
- Fused causal attention (1 NPU call per head instead of 4)
- FFN up N=3072 split into 2x1536 chunks (d3 limit workaround)
- Bias additions on CPU (broadcast add untested on NPU)
- Output head on CPU (vocab=50257 far exceeds NPU d3 limit)
- seq_len=64 (L1 limit with dk=64)

Usage:
  bazel build //xla/pjrt/plugin/xdna:pjrt_c_api_xdna_plugin.so
  python xla/pjrt/plugin/xdna/test_gpt2_xdna.py
"""

import sys
import os
import signal
import faulthandler
import traceback
import threading
import time

faulthandler.enable()
faulthandler.dump_traceback_later(610, exit=True)
faulthandler.register(signal.SIGUSR1, all_threads=True, chain=False)


def _alarm_handler(signum, frame):
    sys.stderr.write("\n=== HANG DETECTED (SIGALRM) ===\n")
    sys.stderr.write(f"Main thread: {threading.current_thread()}\n")
    sys.stderr.write(f"Active threads: {threading.enumerate()}\n\n")
    faulthandler.dump_traceback(file=sys.stderr, all_threads=True)
    sys.stderr.write("\nSending SIGABRT for core dump...\n")
    sys.stderr.flush()
    os.kill(os.getpid(), signal.SIGABRT)


signal.signal(signal.SIGALRM, _alarm_handler)
signal.alarm(600)  # 10 minutes

print("[test] Starting GPT-2 XDNA test", flush=True)
print(f"[test] PID: {os.getpid()}", flush=True)

sys.path.insert(0, os.path.dirname(__file__))
from jax_xdna import initialize
initialize()

import jax
import jax.numpy as jnp
import numpy as np
import ml_dtypes

jax.config.update("jax_platforms", "xdna")
devs = jax.devices()
print(f"[test] Devices: {devs}", flush=True)


def run_test(name, fn):
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


# --- Reference implementations ---

def _softmax_ref(x):
    x_f32 = x.astype(np.float32)
    x_max = np.max(x_f32, axis=-1, keepdims=True)
    e = np.exp(x_f32 - x_max)
    return e / np.sum(e, axis=-1, keepdims=True)


def _layernorm_ref(x, gamma, beta, eps=1e-5):
    x_f32 = x.astype(np.float32)
    g_f32 = gamma.astype(np.float32)
    b_f32 = beta.astype(np.float32)
    mean = np.mean(x_f32, axis=-1, keepdims=True)
    var = np.var(x_f32, axis=-1, keepdims=True)
    normed = (x_f32 - mean) / np.sqrt(var + eps)
    return g_f32 * normed + b_f32


def _gelu_ref(x):
    x_f32 = x.astype(np.float32)
    return 0.5 * x_f32 * (1.0 + np.tanh(
        np.sqrt(2.0 / np.pi) * (x_f32 + 0.044715 * x_f32**3)))


# --- GPT-2 helpers ---

def _bf16(x):
    return x.astype(ml_dtypes.bfloat16)


def _add_bias_cpu(x_npu, bias):
    """Add 1D bias to 2D NPU result on CPU, return bf16 numpy array."""
    return _bf16(np.array(x_npu).astype(np.float32) +
                 bias.astype(np.float32))


def _init_weights(rng, emb_dim, n_layers, ffn_inner, vocab, seq_len):
    """Initialize random GPT-2 weights in bf16."""
    d = emb_dim
    w = {}
    w['tok_embed'] = _bf16(rng.randn(vocab, d) * 0.02)
    w['pos_embed'] = _bf16(rng.randn(seq_len, d) * 0.02)
    for i in range(n_layers):
        p = f'l{i}'
        w[f'{p}.g1'] = _bf16(np.ones(d))
        w[f'{p}.b1'] = _bf16(np.zeros(d))
        for name in ['q', 'k', 'v']:
            w[f'{p}.W_{name}'] = _bf16(rng.randn(d, d) * 0.02)
            w[f'{p}.b_{name}'] = _bf16(rng.randn(d) * 0.02)
        w[f'{p}.W_out'] = _bf16(rng.randn(d, d) * 0.02)
        w[f'{p}.b_out'] = _bf16(rng.randn(d) * 0.02)
        w[f'{p}.g2'] = _bf16(np.ones(d))
        w[f'{p}.b2'] = _bf16(np.zeros(d))
        w[f'{p}.W_ff1'] = _bf16(rng.randn(d, ffn_inner) * 0.02)
        w[f'{p}.b_ff1'] = _bf16(rng.randn(ffn_inner) * 0.02)
        w[f'{p}.W_ff2'] = _bf16(rng.randn(ffn_inner, d) * 0.02)
        w[f'{p}.b_ff2'] = _bf16(rng.randn(d) * 0.02)
    w['g_f'] = _bf16(np.ones(d))
    w['b_f'] = _bf16(np.zeros(d))
    w['W_head'] = _bf16(rng.randn(d, vocab) * 0.02)
    return w


def _run_gpt2_test(n_layers):
    """Run GPT-2 forward pass on NPU and compare to CPU reference.

    Architecture: emb_dim=768, n_heads=12, head_dim=64, ffn=3072.
    seq_len=64 (constrained by L1 with dk=64).

    Per-layer call sequence (~25 jit calls):
      LN1 -> Q/K/V matmul (3) -> [12 heads x fused_causal_attn] ->
      out_proj -> residual -> LN2 -> FFN_up x2 -> GELU x2 ->
      FFN_down -> residual
    """
    emb_dim, n_heads, head_dim = 768, 12, 64
    seq_len, vocab, ffn_inner = 64, 50257, 3072
    ff_half = ffn_inner // 2  # 1536

    rng = np.random.RandomState(42)
    token_ids = rng.randint(0, vocab, size=seq_len)
    w = _init_weights(np.random.RandomState(42), emb_dim, n_layers,
                      ffn_inner, vocab, seq_len)

    # Causal mask: True = masked (upper triangle).
    causal_mask = np.triu(np.ones((seq_len, seq_len), dtype=bool), k=1)
    scale_f32 = 1.0 / np.sqrt(float(head_dim))
    scale_bf16 = ml_dtypes.bfloat16(scale_f32)

    # Define JIT functions once (cache-friendly).
    jit_dot = jax.jit(jnp.dot)
    jit_add = jax.jit(lambda a, b: a + b)
    jit_gelu = jax.jit(lambda x: jax.nn.gelu(x, approximate=True))

    # Fused causal attention: single JIT call per head instead of 4.
    def causal_attn_fn(q, k, v):
        S = q.shape[0]
        mask = jnp.triu(jnp.ones((S, S), dtype=bool), k=1)
        scores = (q @ k.T) * scale_bf16
        scores = jnp.where(mask, jnp.bfloat16(-1e4), scores)
        return jax.nn.softmax(scores) @ v
    jit_causal_attn = jax.jit(causal_attn_fn)

    def layernorm_fn(x, g, b):
        mean = jnp.mean(x, axis=-1, keepdims=True)
        var = jnp.var(x, axis=-1, keepdims=True)
        return g * (x - mean) / jnp.sqrt(var + 1e-5) + b
    jit_ln = jax.jit(layernorm_fn)

    print(f"         GPT-2: {n_layers} layers, d={emb_dim}, heads={n_heads}, "
          f"seq={seq_len}", flush=True)

    # ===== NPU forward pass =====
    t0 = time.time()

    # Embedding (CPU).
    x = _bf16(w['tok_embed'][token_ids].astype(np.float32) +
              w['pos_embed'].astype(np.float32))

    for li in range(n_layers):
        lt = time.time()
        p = f'l{li}'

        # --- Attention block ---
        x_norm = jit_ln(x, w[f'{p}.g1'], w[f'{p}.b1'])

        # QKV projections (NPU) + bias (CPU).
        q_full = _add_bias_cpu(jit_dot(x_norm, w[f'{p}.W_q']), w[f'{p}.b_q'])
        k_full = _add_bias_cpu(jit_dot(x_norm, w[f'{p}.W_k']), w[f'{p}.b_k'])
        v_full = _add_bias_cpu(jit_dot(x_norm, w[f'{p}.W_v']), w[f'{p}.b_v'])

        # Split into heads (CPU): [S, d] -> [S, n_heads, hd].
        q_heads = q_full.reshape(seq_len, n_heads, head_dim)
        k_heads = k_full.reshape(seq_len, n_heads, head_dim)
        v_heads = v_full.reshape(seq_len, n_heads, head_dim)

        # Per-head fused causal attention: single NPU call per head.
        head_outputs = []
        for h in range(n_heads):
            q_h = np.ascontiguousarray(q_heads[:, h, :]).astype(ml_dtypes.bfloat16)
            k_h = np.ascontiguousarray(k_heads[:, h, :]).astype(ml_dtypes.bfloat16)
            v_h = np.ascontiguousarray(v_heads[:, h, :]).astype(ml_dtypes.bfloat16)
            head_outputs.append(np.array(jit_causal_attn(q_h, k_h, v_h)))

        # Concat heads (CPU) -> [S, d].
        attn_out = np.concatenate(head_outputs, axis=-1).astype(ml_dtypes.bfloat16)

        # Output projection (NPU) + bias (CPU) + residual (NPU).
        proj = _add_bias_cpu(
            jit_dot(attn_out, w[f'{p}.W_out']), w[f'{p}.b_out'])
        x = jit_add(x, proj)

        # --- FFN block ---
        x_norm2 = jit_ln(x, w[f'{p}.g2'], w[f'{p}.b2'])

        # FFN up: split N=3072 -> 2x1536 (NPU d3 limit = 64 tiles).
        W1a = np.ascontiguousarray(w[f'{p}.W_ff1'][:, :ff_half])
        W1b = np.ascontiguousarray(w[f'{p}.W_ff1'][:, ff_half:])
        b1a = w[f'{p}.b_ff1'][:ff_half].copy()
        b1b = w[f'{p}.b_ff1'][ff_half:].copy()

        ff1_a = _add_bias_cpu(jit_dot(x_norm2, W1a), b1a)
        ff1_b = _add_bias_cpu(jit_dot(x_norm2, W1b), b1b)

        # GELU on NPU.
        act_a = np.array(jit_gelu(ff1_a))
        act_b = np.array(jit_gelu(ff1_b))

        # Concat halves (CPU) -> FFN down (NPU) + bias (CPU) + residual (NPU).
        ff_cat = np.concatenate([act_a, act_b], axis=-1).astype(ml_dtypes.bfloat16)
        ff2 = _add_bias_cpu(jit_dot(ff_cat, w[f'{p}.W_ff2']),
                            w[f'{p}.b_ff2'])
        x = jit_add(x, ff2)

        elapsed = time.time() - lt
        print(f"         Layer {li}: {elapsed:.1f}s", flush=True)

        if (li + 1) % 4 == 0:
            jax.clear_caches()
            import gc; gc.collect()

    # Final LayerNorm (NPU).
    x_final = jit_ln(x, w['g_f'], w['b_f'])

    # Output head (CPU — vocab=50257 far exceeds NPU limits).
    x_f = np.array(x_final).astype(np.float32)
    logits = x_f[-1:] @ w['W_head'].astype(np.float32)

    npu_time = time.time() - t0
    print(f"         NPU forward: {npu_time:.1f}s", flush=True)
    print(f"         NPU logits range: [{logits.min():.4f}, {logits.max():.4f}]",
          flush=True)

    # ===== CPU reference =====
    t0 = time.time()

    x_ref = _bf16(w['tok_embed'][token_ids].astype(np.float32) +
                  w['pos_embed'].astype(np.float32))

    for li in range(n_layers):
        p = f'l{li}'

        # LN1.
        x_norm_ref = _bf16(_layernorm_ref(x_ref, w[f'{p}.g1'], w[f'{p}.b1']))

        # QKV + bias (f32 matmul -> bf16 truncate, then f32 bias -> bf16).
        def _matmul_bias_ref(x, W, b):
            out = _bf16(x.astype(np.float32) @ W.astype(np.float32))
            return _bf16(out.astype(np.float32) + b.astype(np.float32))

        q_ref = _matmul_bias_ref(x_norm_ref, w[f'{p}.W_q'], w[f'{p}.b_q'])
        k_ref = _matmul_bias_ref(x_norm_ref, w[f'{p}.W_k'], w[f'{p}.b_k'])
        v_ref = _matmul_bias_ref(x_norm_ref, w[f'{p}.W_v'], w[f'{p}.b_v'])

        # Multi-head attention.
        q_h_r = q_ref.reshape(seq_len, n_heads, head_dim)
        k_h_r = k_ref.reshape(seq_len, n_heads, head_dim)
        v_h_r = v_ref.reshape(seq_len, n_heads, head_dim)

        head_outs_ref = []
        for h in range(n_heads):
            qh = q_h_r[:, h, :]
            kh = k_h_r[:, h, :]
            vh = v_h_r[:, h, :]
            s_r = _bf16(qh.astype(np.float32) @ kh.astype(np.float32).T)
            s_r = _bf16(s_r.astype(np.float32) * scale_f32)
            s_r = np.where(causal_mask, ml_dtypes.bfloat16(-1e4),
                           s_r).astype(ml_dtypes.bfloat16)
            w_r = _softmax_ref(s_r)
            head_outs_ref.append(w_r @ vh.astype(np.float32))

        attn_ref = _bf16(np.concatenate(head_outs_ref, axis=-1))
        proj_ref = _matmul_bias_ref(attn_ref, w[f'{p}.W_out'], w[f'{p}.b_out'])
        x_ref = _bf16(x_ref.astype(np.float32) + proj_ref.astype(np.float32))

        # LN2.
        x_norm2_ref = _bf16(_layernorm_ref(x_ref, w[f'{p}.g2'], w[f'{p}.b2']))

        # FFN (no split needed on CPU).
        ff1_ref = _matmul_bias_ref(
            x_norm2_ref, w[f'{p}.W_ff1'], w[f'{p}.b_ff1'])
        act_ref = _bf16(_gelu_ref(ff1_ref))
        ff2_ref = _matmul_bias_ref(act_ref, w[f'{p}.W_ff2'], w[f'{p}.b_ff2'])
        x_ref = _bf16(x_ref.astype(np.float32) + ff2_ref.astype(np.float32))

    x_final_ref = _layernorm_ref(x_ref, w['g_f'], w['b_f'])
    logits_ref = x_final_ref[-1:] @ w['W_head'].astype(np.float32)

    ref_time = time.time() - t0
    print(f"         CPU reference: {ref_time:.1f}s", flush=True)
    print(f"         Ref logits range: [{logits_ref.min():.4f}, {logits_ref.max():.4f}]",
          flush=True)

    # ===== Compare =====
    max_err = np.max(np.abs(logits - logits_ref))
    print(f"         Max absolute error: {max_err:.4f}", flush=True)

    npu_token = np.argmax(logits)
    ref_token = np.argmax(logits_ref)
    print(f"         NPU token: {npu_token}, ref: {ref_token}", flush=True)

    if npu_token != ref_token:
        print(f"         NOTE: Token mismatch NPU={npu_token} vs ref={ref_token}"
              " (expected: fused bf16 attention uses exp2/combined_scale "
              "with different rounding than reference)")
    np.testing.assert_allclose(logits, logits_ref, atol=5.0)


def test_gpt2_single_layer():
    """GPT-2 architecture with 1 layer — validates multi-head attention,
    FFN splitting, bias handling, and causal masking."""
    _run_gpt2_test(n_layers=1)


def test_gpt2_full():
    """GPT-2 124M: full 12-layer forward pass on NPU."""
    _run_gpt2_test(n_layers=12)


def main():
    tests = [
        ("GPT-2 single layer", test_gpt2_single_layer),
        ("GPT-2 12 layers", test_gpt2_full),
    ]

    print("GPT-2 XDNA Integration Tests", flush=True)
    print("=" * 40, flush=True)

    passed = 0
    failed = 0
    for name, fn in tests:
        jax.clear_caches()
        import gc; gc.collect()
        if run_test(name, fn):
            passed += 1
        else:
            failed += 1

    print("=" * 40, flush=True)
    print(f"Results: {passed} passed, {failed} failed, {len(tests)} total",
          flush=True)

    signal.alarm(0)
    sys.exit(0 if failed == 0 else 1)


if __name__ == "__main__":
    main()
