#!/bin/bash
# Run the JAX XDNA integration test with full debugging output.
# Usage: bash xla/pjrt/plugin/xdna/run_jax_test.sh
set -e

cd /home/zach/code/openxla

echo "=== Step 1: Verify device ==="
if [ ! -e /dev/accel/accel0 ]; then
    echo "ERROR: /dev/accel/accel0 not found"
    exit 1
fi
echo "Device exists: $(ls -la /dev/accel/accel0)"

# Check for D-state processes
dstate=$(ps aux 2>/dev/null | awk '$8 ~ /^D/ && /xdna|accel|amdxdna/' | head -5)
if [ -n "$dstate" ]; then
    echo "WARNING: D-state processes detected — device may be stuck:"
    echo "$dstate"
    echo "Reboot may be required."
    exit 1
fi
echo "No D-state processes."

echo ""
echo "=== Step 2: Build plugin ==="
bazel build //xla/pjrt/plugin/xdna:pjrt_c_api_xdna_plugin.so 2>&1 | tail -5
echo "Plugin built."

echo ""
echo "=== Step 3: Run C++ test (verify device health) ==="
bazel test //xla/pjrt/plugin/xdna:xdna_pjrt_client_test \
    --test_filter=CompileAndExecuteAdd \
    --test_output=streamed 2>&1 | tail -20
echo "C++ test done."

echo ""
echo "=== Step 4: Run JAX test ==="
LOGFILE="/tmp/xdna_jax_test_$(date +%Y%m%d_%H%M%S).log"
echo "Stderr log: $LOGFILE"
echo "PID will be shown by test script."
echo ""

source .venv/bin/activate

# Enable core dumps
ulimit -c unlimited 2>/dev/null || true

export PJRT_PLUGIN_PATH=/home/zach/code/openxla/bazel-bin/xla/pjrt/plugin/xdna/pjrt_c_api_xdna_plugin.so

# Run with timeout. Don't use SIGKILL — let the test's own SIGALRM handler
# dump thread stacks first. Use SIGTERM as fallback.
timeout --signal=TERM 60 python xla/pjrt/plugin/xdna/test_jax_xdna.py 2>"$LOGFILE"
rc=$?

echo ""
echo "=== Exit code: $rc ==="
echo ""
echo "=== Stderr output (XDNA traces): ==="
cat "$LOGFILE"
