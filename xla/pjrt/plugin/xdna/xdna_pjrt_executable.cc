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

#include <cstring>
#include <exception>
#include <memory>
#include <optional>
#include <string>
#include <utility>
#include <vector>

#include "absl/log/log.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/string_view.h"
#include "absl/types/span.h"
#include "xla/pjrt/pjrt_client.h"
#include "xla/pjrt/pjrt_future.h"
#include "xla/pjrt/plugin/xdna/xdna_pjrt_buffer.h"
#include "xla/service/computation_placer.h"

namespace xla {

XdnaExecutable::XdnaExecutable(
    PjRtClient* client, absl::Span<PjRtDevice* const> addressable_devices,
    xrt::elf elf, xrt::hw_context hw_context, xrt::kernel kernel,
    absl::string_view name)
    : client_(client),
      device_assignment_(1, 1),
      name_(name),
      elf_(std::move(elf)),
      hw_context_(std::move(hw_context)),
      kernel_(std::move(kernel)) {
  addressable_devices_.assign(addressable_devices.begin(),
                              addressable_devices.end());
  addressable_device_logical_ids_.push_back(
      LogicalDeviceIds{/*replica=*/0, /*partition=*/0});
  if (!addressable_devices_.empty()) {
    device_assignment_(0, 0) =
        addressable_devices_.front()->global_device_id().value();
  }
  LOG(INFO) << "XDNA: Loaded executable '" << name_ << "'";
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
  if (is_deleted_) {
    return absl::InternalError("Executable has been deleted.");
  }
  if (argument_handles.size() != 1) {
    return absl::InvalidArgumentError(absl::StrCat(
        "XDNA supports single-device execution only, got ",
        argument_handles.size(), " device argument sets."));
  }

  std::optional<Future<>> returned_future;
  auto result = ExecuteSharded(argument_handles[0],
                               addressable_devices_.front(), options,
                               returned_future, /*fill_future=*/false);
  if (!result.ok()) {
    return result.status();
  }

  std::vector<std::vector<std::unique_ptr<PjRtBuffer>>> outputs;
  outputs.push_back(std::move(*result));

  if (returned_futures.has_value()) {
    returned_futures->push_back(
        returned_future.value_or(Future<>(absl::OkStatus())));
  }

  return outputs;
}

absl::StatusOr<std::vector<std::unique_ptr<PjRtBuffer>>>
XdnaExecutable::ExecuteSharded(
    absl::Span<PjRtBuffer* const> argument_handles, PjRtDevice* device,
    const ExecuteOptions& options, std::optional<Future<>>& returned_future,
    bool fill_future) const {
  if (is_deleted_) {
    return absl::InternalError("Executable has been deleted.");
  }

  try {
    xrt::run run(kernel_);

    // Allocate device-mapped BOs through the hw_context and copy data in.
    std::vector<xrt::bo> device_bos;
    device_bos.reserve(argument_handles.size());
    for (int i = 0; i < static_cast<int>(argument_handles.size()); ++i) {
      auto* xdna_buf = dynamic_cast<XdnaBuffer*>(argument_handles[i]);
      if (xdna_buf == nullptr) {
        return absl::InvalidArgumentError(
            absl::StrCat("Argument ", i, " is not an XdnaBuffer."));
      }
      if (xdna_buf->IsDeleted()) {
        return absl::InvalidArgumentError(
            absl::StrCat("Argument ", i, " has been deleted."));
      }

      // Allocate a device-mapped BO with proper IOMMU mapping.
      auto raw = xdna_buf->raw_data();
      int grp = kernel_.group_id(i);
      xrt::bo dev_bo(hw_context_, raw.size(), xrt::bo::flags::normal, grp);

      // Copy host data into the mapped region (one memcpy, same physical
      // memory — sync is just a cache flush).
      std::memcpy(dev_bo.map<void*>(), raw.data(), raw.size());
      dev_bo.sync(XCL_BO_SYNC_BO_TO_DEVICE);

      run.set_arg(i, dev_bo);
      device_bos.push_back(std::move(dev_bo));
    }

    run.start();
    run.wait2();

    // Copy results back: cache invalidate, then read from mapped pointer.
    for (int i = 0; i < static_cast<int>(argument_handles.size()); ++i) {
      auto* xdna_buf = dynamic_cast<XdnaBuffer*>(argument_handles[i]);
      device_bos[i].sync(XCL_BO_SYNC_BO_FROM_DEVICE);
      auto dst = xdna_buf->mutable_raw_data();
      std::memcpy(dst.data(), device_bos[i].map<const void*>(), dst.size());
    }

    LOG(INFO) << "XDNA: Kernel '" << name_ << "' execution completed.";
  } catch (const xrt::run::command_error& e) {
    return absl::InternalError(
        absl::StrCat("XDNA kernel execution failed: ", e.what()));
  } catch (const std::exception& e) {
    return absl::InternalError(
        absl::StrCat("XDNA kernel execution failed: ", e.what()));
  }

  if (fill_future) {
    returned_future = Future<>(absl::OkStatus());
  }

  // Buffers updated in-place via copy-back above.
  return std::vector<std::unique_ptr<PjRtBuffer>>{};
}

absl::StatusOr<std::vector<std::unique_ptr<PjRtBuffer>>>
XdnaExecutable::ExecutePortable(
    absl::Span<PjRtBuffer* const> argument_handles, PjRtDevice* device,
    const ExecuteOptions& options, std::optional<Future<>>& returned_future,
    bool fill_future) const {
  return ExecuteSharded(argument_handles, device, options, returned_future,
                        fill_future);
}

void XdnaExecutable::Delete() { is_deleted_ = true; }

bool XdnaExecutable::IsDeleted() const { return is_deleted_; }

absl::string_view XdnaExecutable::name() const { return name_; }

}  // namespace xla
