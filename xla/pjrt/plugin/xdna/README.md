# XDNA NPU PJRT Plugin

A PJRT plugin that exposes AMD XDNA NPUs (Ryzen AI, Strix Halo, etc.) to
XLA-based frameworks (JAX, TensorFlow, PyTorch/XLA) via the
[Xilinx Runtime (XRT)](https://github.com/amd/xdna-driver).

**Status:** Phase 1 — operator dispatch (pre-compiled kernels only, no compiler
integration).

## Prerequisites

### Hardware

- AMD processor with XDNA NPU (e.g., Ryzen AI MAX+ 395 / Strix Halo)
- NPU device visible at `/dev/accel/accel0`

### Kernel

The `amdxdna` kernel module must be loaded. IOMMU must be enabled:

```bash
# Check IOMMU is enabled
dmesg | grep -i iommu

# If not, add to your bootloader kernel cmdline:
#   amd_iommu=on iommu=pt
# For systemd-boot: edit /boot/loader/entries/*.conf
# For GRUB: edit /etc/default/grub GRUB_CMDLINE_LINUX_DEFAULT
```

Also verify in BIOS: enable "IOMMU" and/or "SVM" (Secure Virtual Machine).

### Resource Limits

XRT requires mapping large device memory regions. The default locked memory
limit (typically 8 MB) is too small. Set it to unlimited:

```bash
# Add to /etc/security/limits.conf:
echo "* - memlock unlimited" | sudo tee -a /etc/security/limits.conf
```

Reboot (or log out and back in) for the change to take effect. Verify with
`ulimit -l` — it should show `unlimited`.

### XRT Installation (Arch Linux)

XRT must be installed from the
[amd/xdna-driver](https://github.com/amd/xdna-driver) repository. Three
packages are required, built and installed in order:

```bash
# Clone and initialize
git clone https://github.com/amd/xdna-driver.git
cd xdna-driver
git submodule update --init --recursive

# 1. Build xrt-base and xrt-npu
cd xrt/build/arch
makepkg -p PKGBUILD-xrt-base -s
makepkg -p PKGBUILD-xrt-npu -s
sudo pacman -U xrt-base-*.pkg.tar.zst xrt-npu-*.pkg.tar.zst

# 2. Build the XDNA driver plugin (from top-level build/)
cd ../../../build
./build.sh -release

# 3. Package and install the plugin
cd arch
makepkg -p PKGBUILD-xrt-plugin -s
sudo pacman -U xrt-plugin-amdxdna-*.pkg.tar.zst
```

**Reboot** after installing `xrt-plugin-amdxdna` (it includes a DKMS kernel
module).

### Verify Installation

```bash
# XRT libraries
ls /opt/xilinx/xrt/lib/libxrt_coreutil.so

# XRT headers
ls /opt/xilinx/xrt/include/xrt/xrt_device.h

# libuuid headers (required by XRT)
ls /usr/include/uuid/uuid.h
# On Arch: pacman -S util-linux-libs

# Device access
/opt/xilinx/xrt/bin/xrt-smi examine
# Should show "1 devices found" with device details
```

### Build Dependencies

- **Bazel 7.7.0** (see `.bazelversion`; install via `asdf install bazel 7.7.0`)
- **libuuid headers** (`uuid/uuid.h`) — on Arch Linux: `pacman -S util-linux-libs`

## Building

```bash
# Build the plugin shared library
bazel build //xla/pjrt/plugin/xdna:pjrt_c_api_xdna_plugin.so

# Run tests
bazel test //xla/pjrt/plugin/xdna:xdna_pjrt_client_test
```

The plugin builds and tests pass regardless of whether NPU hardware is present.
When no device is available, the client runs in host-only mode.

## WORKSPACE Configuration

The build requires two external repository entries in the top-level `WORKSPACE`
file (already configured):

```python
# libuuid headers (required by XRT)
new_local_repository(
    name = "libuuid",
    build_file_content = """...""",
    path = "/usr/include/uuid",
)

# XRT (Xilinx Runtime)
new_local_repository(
    name = "xrt",
    build_file_content = """...""",
    path = "/opt/xilinx/xrt",
)
```

## Architecture

```
JAX / TensorFlow / PyTorch
        |
    PJRT C API  (pjrt_c_api_xdna_plugin.so)
        |
    XdnaPjrtClient  (C++ PjRtClient subclass)
        |
    XRT (libxrt_coreutil.so)
        |
    amdxdna kernel module
        |
    /dev/accel/accel0  (NPU hardware)
```

### Source Files

| File | Description |
|------|-------------|
| `xdna_pjrt_client.h/cc` | `XdnaPjrtClient` — PjRtClient subclass, XRT device management |
| `xdna_pjrt_device.h/cc` | `XdnaDevice`, `XdnaMemorySpace`, `XdnaDeviceDescription` |
| `xdna_pjrt_buffer.h/cc` | `XdnaBuffer` — PjRtBuffer with host-side storage (Phase 1) |
| `xdna_pjrt_executable.h/cc` | `XdnaExecutable` — PjRtLoadedExecutable (stub, Phase 1) |
| `xdna_c_pjrt.h/cc` | C API entry point (`GetPjrtApi()`) |
| `xdna_c_pjrt_internal.h/cc` | C API wrapper using `pjrt::CreatePjrtApi()` |

## Troubleshooting

### "0 devices found" in xrt-smi

1. Ensure all three packages are installed: `pacman -Q | grep xrt`
   - `xrt-base`, `xrt-npu`, `xrt-plugin-amdxdna`
2. Reboot after installing `xrt-plugin-amdxdna`
3. Check IOMMU: `dmesg | grep -i iommu` — must be enabled
4. Check kernel log: `dmesg | grep -i xdna`
   - "SVA bind device failed" → IOMMU not enabled
   - "IOMMU is off, require carveout memory" → enable IOMMU in BIOS + kernel cmdline
   - "Hardware init failed, ret -19" → reboot needed after driver install

### mmap failed "Resource temporarily unavailable"

The locked memory limit is too low. XRT needs to map ~64 MB of device memory:
```bash
# Check current limit
ulimit -l
# If not "unlimited", add to /etc/security/limits.conf:
echo "* - memlock unlimited" | sudo tee -a /etc/security/limits.conf
# Then reboot or re-login
```

### Build fails with "uuid/uuid.h not found"

Install libuuid development headers:
```bash
# Arch Linux
sudo pacman -S util-linux-libs

# Ubuntu/Debian
sudo apt install uuid-dev
```

### Runtime error "libxrt_coreutil.so.2: cannot open shared object"

The XRT library path may not be in the linker search path:
```bash
export LD_LIBRARY_PATH=/opt/xilinx/xrt/lib:$LD_LIBRARY_PATH
```
