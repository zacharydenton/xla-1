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

#ifndef XLA_PJRT_PLUGIN_XDNA_XDNA_TARGET_CAPS_H_
#define XLA_PJRT_PLUGIN_XDNA_XDNA_TARGET_CAPS_H_

#include <string>

namespace xla {

// Hardware capabilities for an XDNA NPU target. Captures per-SKU facts
// (device name, ISA, memory sizes, tile grid) separate from compilation
// choices (AieLoweringConfig).
struct TargetCaps {
  // Device identifier for aie.device() directive.
  std::string device_name = "npu2";
  // ISA target for Peano backend (--march, --target).
  std::string isa_target = "aie2p";
  // Tile grid dimensions.
  int num_columns = 4;               // usable compute columns (Strix Point)
  int num_rows = 4;                  // compute rows per column
  // Memory sizes (bytes).
  int l1_data_memory_bytes = 65536;  // 64KB data memory per compute tile
  int l1_program_memory_bytes = 16384;  // 16KB program memory per compute tile
  int l2_memory_bytes = 524288;      // 512KB per memory tile
  // L1 usable budget = l1_data - overhead for stack. Default 48KB.
  int l1_usable_bytes = 49152;
  // Tile row assignments.
  int shim_row = 0;
  int mem_tile_row = 1;
  int first_compute_row = 2;
  // Memory tile DMA channels per direction (MM2S and S2MM).
  // NPU2 mem tile has 6 channels per direction. Used to decide whether
  // distribute/join can route all FIFOs through a single mem tile.
  int mem_tile_dma_channels = 6;
  // Shim DMA BD d3 dimension max (outermost wrap count). Hardware limit
  // on the number of outermost iterations in a single BD descriptor.
  int shim_dma_max_d3 = 64;
  // xclbin partition metadata.
  int partition_column_width = 8;
  int partition_start_column = 0;
};

// Returns the partition-aware device name for a given number of columns.
// mlir-aie provides npu2_1col through npu2_7col which restrict tile
// allocation to exact column counts, preventing resource conflicts when
// other applications share the NPU.
inline std::string DeviceNameForColumns(int num_columns) {
  if (num_columns <= 0 || num_columns > 7) return "npu2";
  return "npu2_" + std::to_string(num_columns) + "col";
}

// Factory for Strix Point (XDNA 2 / Ryzen AI 300 / AIE2PS).
inline TargetCaps StrixPointTargetCaps() { return TargetCaps{}; }

}  // namespace xla

#endif  // XLA_PJRT_PLUGIN_XDNA_XDNA_TARGET_CAPS_H_
