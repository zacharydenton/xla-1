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

#ifndef XLA_PJRT_PLUGIN_XDNA_XDNA_HLO_PASSES_H_
#define XLA_PJRT_PLUGIN_XDNA_XDNA_HLO_PASSES_H_

#include <memory>

#include "absl/status/statusor.h"
#include "xla/hlo/ir/hlo_module.h"

namespace xla {

// Runs XDNA-specific HLO optimization passes on the module.
//
// Includes op expanders (TopK, Cholesky, QR, Eigh, TriangularSolve,
// BatchNorm), layout assignment, and type canonicalization. These simplify
// the HLO graph before lowering to MLIR.
absl::StatusOr<std::unique_ptr<HloModule>> RunXdnaHloPasses(
    std::unique_ptr<HloModule> hlo_module);

}  // namespace xla

#endif  // XLA_PJRT_PLUGIN_XDNA_XDNA_HLO_PASSES_H_
