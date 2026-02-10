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

// Lowers linalg-on-tensors MLIR to AIE dialect IR.
//
// Takes MLIR in the linalg dialect (output of mhlo-to-linalg conversion) and
// produces MLIR in the AIE dialect with:
// - Physical tile placement (aie.tile)
// - ObjectFIFO data movement (aie.objectfifo)
// - Per-core computation (aie.core)
// - DMA and lock configuration
//
// Currently returns Unimplemented — requires mlir-aie library linkage.
//
// `linalg_mlir` is serialized MLIR bytecode in the linalg dialect.
// Returns serialized MLIR bytecode in the AIE dialect.
absl::StatusOr<std::string> LowerLinalgToAie(
    const std::string& linalg_mlir, const AieLoweringConfig& config);

}  // namespace xla

#endif  // XLA_PJRT_PLUGIN_XDNA_XDNA_AIE_LOWERING_H_
