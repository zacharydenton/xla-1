/* Copyright 2025 The OpenXLA Authors.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

#ifndef XLA_PJRT_PLUGIN_XDNA_XDNA_CODEGEN_H_
#define XLA_PJRT_PLUGIN_XDNA_XDNA_CODEGEN_H_

#include <cstdint>
#include <string>
#include <vector>

#include "absl/status/statusor.h"

namespace xla {

// Generates an ELF binary from AIE dialect MLIR text.
//
// Uses external tools via subprocess invocation:
// 1. aie-opt: Run AIE optimization/lowering passes
// 2. aie-translate: Generate CDO and NPU instruction binaries
// 3. Peano (clang): Compile per-core code to AIE object files
// 4. aiebu-asm: Package everything into a self-contained ELF
//
// Tool paths (searched in order):
// - /opt/mlir-aie/bin/aie-opt, /opt/mlir-aie/bin/aie-translate
// - /opt/peano/bin/clang
// - /opt/xilinx/xrt/bin/aiebu-asm
//
// Returns raw ELF bytes suitable for LoadSerializedExecutable().
absl::StatusOr<std::vector<uint8_t>> GenerateElfFromAie(
    const std::string& aie_mlir);

}  // namespace xla

#endif  // XLA_PJRT_PLUGIN_XDNA_XDNA_CODEGEN_H_
