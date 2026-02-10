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

#include <memory>

#include "absl/status/statusor.h"
#include "xla/hlo/ir/hlo_module.h"
#include "xla/pjrt/plugin/xdna/xdna_codegen.h"

namespace xla {

// Compiles an HloModule into an xclbin for the XDNA NPU.
//
// Pipeline:
//   HloModule
//     → StableHLO dialect (via ConvertHloToMlirHlo)
//     → linalg-on-tensors (via stablehlo-legalize-to-linalg)
//     → AIE dialect MLIR text (via LowerLinalgToAie)
//     → xclbin (via GenerateXclbinFromAie: aie-opt + Peano + bootgen + xclbinutil)
//
// External tools are invoked via subprocess to avoid LLVM ABI conflicts
// between OpenXLA's LLVM and Peano/mlir-aie's LLVM.
class XdnaCompiler {
 public:
  // Compiles an optimized HloModule to an xclbin.
  // The module should have already been through RunXdnaHloPasses().
  static absl::StatusOr<XdnaCodegenResult> Compile(
      std::unique_ptr<HloModule> hlo_module);
};

}  // namespace xla

#endif  // XLA_PJRT_PLUGIN_XDNA_XDNA_COMPILER_H_
