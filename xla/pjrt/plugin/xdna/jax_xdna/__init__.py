"""JAX plugin registration for the XDNA NPU PJRT plugin.

Registers the XDNA PJRT plugin .so with JAX. The plugin is discovered
automatically by JAX via the jax_plugins namespace package mechanism,
or can be loaded manually by importing this module.

The .so is located via (in order):
  1. XDNA_PLUGIN_PATH environment variable
  2. bazel-bin output directory (for development)
"""

import os
import pathlib

import jax._src.xla_bridge as xb

_PLUGIN_NAME = "xdna"
_SO_NAME = "pjrt_c_api_xdna_plugin.so"


def _find_plugin_so() -> str:
    # 1. Explicit env var
    env_path = os.environ.get("XDNA_PLUGIN_PATH")
    if env_path:
        if os.path.isfile(env_path):
            return env_path
        raise FileNotFoundError(
            f"XDNA_PLUGIN_PATH={env_path} does not exist"
        )

    # 2. Adjacent to this file (for installed packages)
    here = pathlib.Path(__file__).resolve().parent
    adjacent = here / _SO_NAME
    if adjacent.is_file():
        return str(adjacent)

    # 3. bazel-bin output (for development)
    # Walk up from xla/pjrt/plugin/xdna/jax_xdna/ to the workspace root
    workspace = here.parent.parent.parent.parent.parent
    bazel_bin = workspace / "bazel-bin" / "xla" / "pjrt" / "plugin" / "xdna" / _SO_NAME
    if bazel_bin.is_file():
        return str(bazel_bin)

    raise FileNotFoundError(
        f"Cannot find {_SO_NAME}. Set XDNA_PLUGIN_PATH or run:\n"
        f"  bazel build //xla/pjrt/plugin/xdna:{_SO_NAME}"
    )


def initialize():
    """Called by JAX's plugin discovery to register the XDNA backend."""
    path = _find_plugin_so()
    xb.register_plugin(_PLUGIN_NAME, priority=500, library_path=path, options=None)
