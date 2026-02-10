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

// Generates an ELF binary from AIE dialect MLIR.
//
// The full pipeline:
// 1. Per-core code generation: extract each core's computation, lower to
//    LLVM IR, compile with Peano (llc --march=aie2p) to per-core object files.
// 2. CDO generation: AIETranslateToCDODirect() generates configuration data
//    for DMA, locks, and routing.
// 3. PDI packaging: bootgen or direct CDO packing into a PDI.
// 4. ELF assembly: aiebu-asm packages PDI + NPU instructions into a
//    self-contained ELF loadable by XRT.
//
// Currently returns Unimplemented — requires Peano and mlir-aie library
// linkage.
//
// `aie_mlir` is serialized MLIR bytecode in the AIE dialect.
// Returns raw ELF bytes suitable for LoadSerializedExecutable().
absl::StatusOr<std::vector<uint8_t>> GenerateElfFromAie(
    const std::string& aie_mlir);

}  // namespace xla

#endif  // XLA_PJRT_PLUGIN_XDNA_XDNA_CODEGEN_H_
