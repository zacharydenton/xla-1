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

#include "xla/pjrt/plugin/xdna/xdna_aie_lowering.h"

#include <string>

#include "absl/status/status.h"
#include "absl/status/statusor.h"

namespace xla {

absl::StatusOr<std::string> LowerLinalgToAie(
    const std::string& linalg_mlir, const AieLoweringConfig& config) {
  return absl::UnimplementedError(
      "AIE lowering requires mlir-aie library linkage. "
      "Build mlir-aie from source and link via @mlir_aie.");
}

}  // namespace xla
