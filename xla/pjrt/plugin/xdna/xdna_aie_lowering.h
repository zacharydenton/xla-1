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

#ifndef XLA_PJRT_PLUGIN_XDNA_XDNA_AIE_LOWERING_H_
#define XLA_PJRT_PLUGIN_XDNA_XDNA_AIE_LOWERING_H_

#include <string>

#include "absl/status/statusor.h"
#include "mlir/IR/BuiltinOps.h"
#include "xla/pjrt/plugin/xdna/xdna_target_caps.h"

namespace xla {

// Configuration for AIE tile placement and data movement.
struct AieLoweringConfig {
  // Number of compute columns to use (1-4 for XDNA 2).
  int num_columns = 1;
  // Number of compute tiles per column (1-8 for XDNA 2).
  int num_cores_per_column = 1;
  // L1 tile size in bytes (per compute tile).
  int l1_tile_bytes = 65536;
  // L2 tile size in bytes (per memory tile).
  int l2_tile_bytes = 524288;
};

// Result of AIE lowering: MLIR text and actual core count used.
struct AieLoweringResult {
  std::string aie_mlir;
  int num_cores;  // Actual columns used (may be < requested).
  bool use_aievec = false;  // Enable --convert-aievec-to-llvm pass.
  bool convert_vector_to_aievec = false;  // Enable --convert-vector-to-aievec pass.
  bool needs_softfloat_stubs = false;  // Kernel uses soft-float comparisons.
};

// Lowers a linalg-on-tensors MLIR module to AIE dialect MLIR text.
//
// Analyzes the linalg operations in the module and generates AIE dialect
// MLIR text for the XDNA NPU, including:
// - Physical tile placement (aie.tile)
// - ObjectFIFO data movement (aie.objectfifo)
// - Per-core computation (aie.core)
// - NPU DMA instruction sequence
//
// The output text is suitable for processing by aie-opt and aie-translate.
//
// Supports element-wise ops (add, subtract, multiply, negate, etc.)
// across multiple compute columns for data-parallel execution.
//
// Returns AieLoweringResult with AIE dialect MLIR text and actual core count.
absl::StatusOr<AieLoweringResult> LowerLinalgToAie(
    mlir::ModuleOp linalg_module, const AieLoweringConfig& config,
    const TargetCaps& caps);

}  // namespace xla

#endif  // XLA_PJRT_PLUGIN_XDNA_XDNA_AIE_LOWERING_H_
