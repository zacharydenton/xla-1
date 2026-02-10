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

#ifndef XLA_PJRT_PLUGIN_XDNA_XDNA_COMPILER_H_
#define XLA_PJRT_PLUGIN_XDNA_XDNA_COMPILER_H_

#include <cstdint>
#include <memory>
#include <vector>

#include "absl/status/statusor.h"
#include "xla/hlo/ir/hlo_module.h"

namespace xla {

// Compiles an HloModule into an ELF binary for the XDNA NPU.
//
// Pipeline:
//   HloModule
//     → MHLO dialect (via ConvertHloToMlirHlo)
//     → linalg-on-tensors (via mhlo-to-linalg pass)
//     → AIE dialect (via LowerLinalgToAie — requires mlir-aie)
//     → ELF binary (via GenerateElfFromAie — requires Peano + mlir-aie)
//
// Currently only the HLO → MHLO → linalg path is implemented. The AIE
// lowering and codegen steps return Unimplemented until the toolchain
// (Peano + mlir-aie) is built and linked.
class XdnaCompiler {
 public:
  // Compiles an optimized HloModule to ELF bytes.
  // The module should have already been through RunHloPasses().
  static absl::StatusOr<std::vector<uint8_t>> Compile(
      std::unique_ptr<HloModule> hlo_module);
};

}  // namespace xla

#endif  // XLA_PJRT_PLUGIN_XDNA_XDNA_COMPILER_H_
