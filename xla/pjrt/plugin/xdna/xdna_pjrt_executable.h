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

#ifndef XLA_PJRT_PLUGIN_XDNA_XDNA_PJRT_EXECUTABLE_H_
#define XLA_PJRT_PLUGIN_XDNA_XDNA_PJRT_EXECUTABLE_H_

#include <cstdint>
#include <memory>
#include <optional>
#include <string>
#include <vector>

#include "absl/status/statusor.h"
#include "absl/strings/string_view.h"
#include "absl/types/span.h"
#include "xla/hlo/ir/hlo_module.h"
#include "xla/pjrt/pjrt_client.h"
#include "xla/pjrt/pjrt_executable.h"
#include "xla/pjrt/pjrt_future.h"
#include "xla/service/computation_placer.h"
#include "xla/shape.h"
#include "xrt/experimental/xrt_elf.h"
#include "xrt/experimental/xrt_hw_context.h"
#include "xrt/experimental/xrt_xclbin.h"
#include "xrt/xrt_kernel.h"

namespace xla {

// Calling convention for kernel argument binding.
enum class XdnaKernelConvention {
  // Phase 1 / legacy: data BOs start at arg 0.
  kDirect,
  // Standard DPU convention used by mlir-aie compiled kernels:
  //   arg 0: opcode (uint64, value 3 for DPU dispatch)
  //   arg 1: instruction BO (or 0 in ELF flow)
  //   arg 2: instruction count (or 0 in ELF flow)
  //   args 3+: data buffer objects
  kDpu,
};

// PjRtLoadedExecutable for XDNA NPU that dispatches pre-compiled ELF kernels.
//
// Two modes:
// - kDirect: Phase 1 legacy. All argument_handles are bound as kernel args
//   starting at arg 0. Buffers are modified in-place; returns empty output.
// - kDpu: Compiled kernels. Args 0-2 are opcode/instr/count. Input buffers
//   bound at args 3+, output buffers allocated and bound after inputs.
//   Returns newly allocated output buffers.
class XdnaExecutable : public PjRtLoadedExecutable {
 public:
  // Construct from ELF (Phase 1 / LoadSerializedExecutable).
  // Uses kDirect convention by default.
  XdnaExecutable(PjRtClient* client,
                 absl::Span<PjRtDevice* const> addressable_devices,
                 xrt::elf elf, xrt::hw_context hw_context,
                 xrt::kernel kernel, absl::string_view name,
                 XdnaKernelConvention convention = XdnaKernelConvention::kDirect,
                 int num_inputs = -1,
                 std::vector<Shape> output_shapes = {});

  // Construct from xclbin (compiled kernels).
  XdnaExecutable(PjRtClient* client,
                 absl::Span<PjRtDevice* const> addressable_devices,
                 xrt::xclbin xclbin, xrt::hw_context hw_context,
                 xrt::kernel kernel, absl::string_view name,
                 XdnaKernelConvention convention,
                 int num_inputs,
                 std::vector<Shape> output_shapes,
                 std::vector<uint32_t> instr_words,
                 std::shared_ptr<HloModule> hlo_module = nullptr);

  ~XdnaExecutable() override = default;

  PjRtClient* client() const override;
  int num_replicas() const override;
  int num_partitions() const override;
  const DeviceAssignment& device_assignment() const override;

  absl::Span<const LogicalDeviceIds> addressable_device_logical_ids()
      const override;
  absl::Span<PjRtDevice* const> addressable_devices() const override;

  absl::StatusOr<std::vector<std::vector<std::unique_ptr<PjRtBuffer>>>>
  Execute(absl::Span<const std::vector<PjRtBuffer*>> argument_handles,
          const ExecuteOptions& options,
          std::optional<std::vector<Future<>>>& returned_futures)
      const override;

  absl::StatusOr<std::vector<std::unique_ptr<PjRtBuffer>>> ExecuteSharded(
      absl::Span<PjRtBuffer* const> argument_handles, PjRtDevice* device,
      const ExecuteOptions& options,
      std::optional<Future<>>& returned_future,
      bool fill_future) const override;

  absl::StatusOr<std::vector<std::unique_ptr<PjRtBuffer>>> ExecutePortable(
      absl::Span<PjRtBuffer* const> argument_handles, PjRtDevice* device,
      const ExecuteOptions& options,
      std::optional<Future<>>& returned_future,
      bool fill_future) const override;

  void Delete() override;
  bool IsDeleted() const override;

  absl::string_view name() const override;

  absl::StatusOr<std::string> FingerprintExecutable() const override;
  absl::StatusOr<std::string> SerializeExecutable() const override;
  absl::StatusOr<struct CompileOptions> GetCompileOptions() const override;
  absl::StatusOr<std::vector<std::shared_ptr<HloModule>>>
  GetHloModules() const override;
  absl::StatusOr<std::vector<Shape>> GetOutputShapes() const override;
  absl::StatusOr<std::vector<std::vector<PrimitiveType>>>
  GetOutputElementTypes() const override;
  absl::StatusOr<std::vector<std::vector<DimensionVector>>>
  GetOutputDimensions() const override;
  absl::StatusOr<std::vector<std::vector<absl::string_view>>>
  GetOutputMemoryKinds() const override;
  absl::StatusOr<std::vector<std::shared_ptr<const PjRtLayout>>>
  GetParameterLayouts() const override;
  absl::StatusOr<std::vector<std::shared_ptr<const PjRtLayout>>>
  GetOutputLayouts() const override;
  absl::StatusOr<CompiledMemoryStats> GetCompiledMemoryStats() const override;

  // Override to avoid default impl calling GetHloModules().
  std::optional<std::vector<OpSharding>> GetOutputShardings() const override;
  std::optional<std::vector<OpSharding>> GetParameterShardings() const override;

  // Override to avoid forwarder infinite recursion.
  int64_t SizeOfGeneratedCodeInBytes() const override;

 private:
  PjRtClient* client_;
  DeviceAssignment device_assignment_;
  std::vector<LogicalDeviceIds> addressable_device_logical_ids_;
  std::vector<PjRtDevice*> addressable_devices_;
  std::string name_;
  bool is_deleted_ = false;

  // Backing object: ELF for Phase 1, xclbin for compiled kernels.
  std::optional<xrt::elf> elf_;
  std::optional<xrt::xclbin> xclbin_;
  xrt::hw_context hw_context_;
  xrt::kernel kernel_;
  XdnaKernelConvention convention_;

  // For kDpu convention: number of input args and output shapes.
  // When num_inputs_ >= 0, ExecuteSharded will allocate output buffers.
  int num_inputs_ = -1;
  std::vector<Shape> output_shapes_;

  // NPU instruction stream for xclbin-based executables.
  // Loaded as a cacheable BO and passed as kernel arg 1.
  std::vector<uint32_t> instr_words_;

  // The HLO module from compilation, used to satisfy GetHloModules() queries.
  std::shared_ptr<HloModule> hlo_module_;
};

}  // namespace xla

#endif  // XLA_PJRT_PLUGIN_XDNA_XDNA_PJRT_EXECUTABLE_H_
