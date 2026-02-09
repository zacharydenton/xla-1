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

#include <memory>
#include <optional>
#include <string>
#include <vector>

#include "absl/status/statusor.h"
#include "absl/strings/string_view.h"
#include "absl/types/span.h"
#include "xla/pjrt/pjrt_client.h"
#include "xla/pjrt/pjrt_executable.h"
#include "xla/pjrt/pjrt_future.h"
#include "xla/service/computation_placer.h"
#include "xrt/experimental/xrt_elf.h"
#include "xrt/experimental/xrt_hw_context.h"
#include "xrt/xrt_kernel.h"

namespace xla {

// PjRtLoadedExecutable for XDNA NPU that dispatches pre-compiled ELF kernels.
//
// Phase 1: No compiler integration. Users provide pre-compiled ELF binaries
// via LoadSerializedExecutable(). Execute() binds user-provided buffers to
// kernel arguments and runs the kernel on the NPU.
//
// All argument_handles passed to Execute are bound as kernel global arguments
// in order. The kernel modifies buffers in-place; Execute returns empty output.
class XdnaExecutable : public PjRtLoadedExecutable {
 public:
  // Construct an executable from XRT objects loaded from an ELF.
  XdnaExecutable(PjRtClient* client,
                 absl::Span<PjRtDevice* const> addressable_devices,
                 xrt::elf elf, xrt::hw_context hw_context,
                 xrt::kernel kernel, absl::string_view name);

  ~XdnaExecutable() override = default;

  PjRtClient* client() const override;
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

 private:
  PjRtClient* client_;
  DeviceAssignment device_assignment_;
  std::vector<LogicalDeviceIds> addressable_device_logical_ids_;
  std::vector<PjRtDevice*> addressable_devices_;
  std::string name_;
  bool is_deleted_ = false;

  xrt::elf elf_;
  xrt::hw_context hw_context_;
  xrt::kernel kernel_;
};

}  // namespace xla

#endif  // XLA_PJRT_PLUGIN_XDNA_XDNA_PJRT_EXECUTABLE_H_
