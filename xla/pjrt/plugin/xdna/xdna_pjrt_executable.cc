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

#include "xla/pjrt/plugin/xdna/xdna_pjrt_executable.h"

#include <memory>
#include <optional>
#include <string>
#include <utility>
#include <vector>

#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/string_view.h"
#include "absl/types/span.h"
#include "xla/pjrt/pjrt_client.h"
#include "xla/pjrt/pjrt_future.h"
#include "xla/service/computation_placer.h"

namespace xla {

XdnaExecutable::XdnaExecutable(
    PjRtClient* client, absl::Span<PjRtDevice* const> addressable_devices,
    absl::string_view name)
    : client_(client),
      device_assignment_(1, 1),
      name_(name) {
  addressable_devices_.assign(addressable_devices.begin(),
                              addressable_devices.end());
  addressable_device_logical_ids_.push_back(
      LogicalDeviceIds{/*replica=*/0, /*partition=*/0});
  // Set device assignment to the first addressable device.
  if (!addressable_devices_.empty()) {
    device_assignment_(0, 0) =
        addressable_devices_.front()->global_device_id().value();
  }
}

PjRtClient* XdnaExecutable::client() const { return client_; }

const DeviceAssignment& XdnaExecutable::device_assignment() const {
  return device_assignment_;
}

absl::Span<const PjRtLoadedExecutable::LogicalDeviceIds>
XdnaExecutable::addressable_device_logical_ids() const {
  return addressable_device_logical_ids_;
}

absl::Span<PjRtDevice* const> XdnaExecutable::addressable_devices() const {
  return addressable_devices_;
}

absl::StatusOr<std::vector<std::vector<std::unique_ptr<PjRtBuffer>>>>
XdnaExecutable::Execute(
    absl::Span<const std::vector<PjRtBuffer*>> argument_handles,
    const ExecuteOptions& options,
    std::optional<std::vector<Future<>>>& returned_futures) const {
  return absl::UnimplementedError(
      "Execute not yet implemented for XDNA. Phase 1 requires pre-compiled "
      "XCLBINs loaded via XRT directly.");
}

absl::StatusOr<std::vector<std::unique_ptr<PjRtBuffer>>>
XdnaExecutable::ExecuteSharded(
    absl::Span<PjRtBuffer* const> argument_handles, PjRtDevice* device,
    const ExecuteOptions& options, std::optional<Future<>>& returned_future,
    bool fill_future) const {
  return absl::UnimplementedError(
      "ExecuteSharded not yet implemented for XDNA.");
}

absl::StatusOr<std::vector<std::unique_ptr<PjRtBuffer>>>
XdnaExecutable::ExecutePortable(
    absl::Span<PjRtBuffer* const> argument_handles, PjRtDevice* device,
    const ExecuteOptions& options, std::optional<Future<>>& returned_future,
    bool fill_future) const {
  return absl::UnimplementedError(
      "ExecutePortable not yet implemented for XDNA.");
}

void XdnaExecutable::Delete() { is_deleted_ = true; }

bool XdnaExecutable::IsDeleted() const { return is_deleted_; }

absl::string_view XdnaExecutable::name() const { return name_; }

}  // namespace xla
