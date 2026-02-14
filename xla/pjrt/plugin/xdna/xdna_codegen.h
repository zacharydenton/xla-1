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
#include "xla/pjrt/plugin/xdna/xdna_target_caps.h"

namespace xla {

// Result of AIE codegen: xclbin + NPU instructions + kernel metadata.
struct XdnaCodegenResult {
  std::vector<uint8_t> xclbin_bytes;
  std::string kernel_name;
  int num_kernel_args;  // Total args including opcode/instr/ninstr prefix.
  // NPU instruction stream (uint32 words). Loaded as a cacheable BO and
  // passed as kernel arg 1 at execution time.
  std::vector<uint32_t> instr_words;
};

// Generates an xclbin from AIE dialect MLIR text.
//
// Full pipeline via external tools:
// 1. aie-opt: Lower ObjectFIFOs, route flows, assign resources
// 2. Peano (opt/llc/clang): Compile per-core code to AIE ELFs
// 3. aie-translate --aie-generate-cdo: Generate CDO binaries
// 4. bootgen: Package CDOs into a PDI (Programmable Device Image)
// 5. xclbinutil: Assemble xclbin with PDI + kernel metadata
//
// Tool paths default to /opt/{mlir-aie,peano,xilinx/xrt}/bin/;
// override via XDNA_AIE_OPT, XDNA_PEANO_CLANG, etc.
//
// `num_data_args` is the number of data buffer arguments (inputs + outputs)
// in the kernel. Used to generate kernel metadata in the xclbin.
// `num_cores` is the number of compute columns to compile ELFs for.
absl::StatusOr<XdnaCodegenResult> GenerateXclbinFromAie(
    const std::string& aie_mlir, int num_data_args, const TargetCaps& caps,
    int num_cores, bool use_aievec = false,
    bool convert_vector_to_aievec = false,
    bool needs_matmul_workarounds = false,
    bool needs_softfloat_stubs = false,
    bool needs_softmax_kernel = false);

}  // namespace xla

#endif  // XLA_PJRT_PLUGIN_XDNA_XDNA_CODEGEN_H_
