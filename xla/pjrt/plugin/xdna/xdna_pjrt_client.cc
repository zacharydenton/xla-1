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

#include "xla/pjrt/plugin/xdna/xdna_pjrt_client.h"

#include <cstdint>
#include <cstring>
#include <memory>
#include <optional>
#include <utility>
#include <vector>

#include "absl/functional/any_invocable.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/string_view.h"
#include "absl/types/span.h"
#include "xla/pjrt/pjrt_client.h"
#include "xla/pjrt/plugin/xdna/xdna_pjrt_buffer.h"
#include "xla/pjrt/plugin/xdna/xdna_pjrt_device.h"
#include "xla/shape_util.h"
#include "tsl/platform/fingerprint.h"

namespace xla {

namespace {
constexpr char kXdnaPlatformName[] = "xdna";
constexpr char kXdnaBackendName[] = "xdna";
}  // namespace

XdnaPjrtClient::XdnaPjrtClient() {
  // Create one device and one memory space.
  device_ = std::make_unique<XdnaDevice>(this, /*id=*/0, /*process_index=*/0);
  memory_space_ = std::make_unique<XdnaMemorySpace>(this, /*id=*/0);

  // Wire device and memory space together.
  device_->SetMemorySpace(memory_space_.get());
  memory_space_->AddDevice(device_.get());

  devices_.push_back(device_.get());
  addressable_devices_.push_back(device_.get());
  memory_spaces_.push_back(memory_space_.get());

  platform_version_ = "XDNA NPU Phase 1 (operator dispatch)";
}

absl::string_view XdnaPjrtClient::platform_name() const {
  return kXdnaPlatformName;
}

int XdnaPjrtClient::process_index() const { return 0; }

PjRtPlatformId XdnaPjrtClient::platform_id() const {
  static const uint64_t kXdnaPlatformId =
      tsl::Fingerprint64(kXdnaBackendName);
  return kXdnaPlatformId;
}

int XdnaPjrtClient::device_count() const {
  return static_cast<int>(devices_.size());
}

int XdnaPjrtClient::addressable_device_count() const {
  return static_cast<int>(addressable_devices_.size());
}

absl::Span<PjRtDevice* const> XdnaPjrtClient::devices() const {
  return devices_;
}

absl::Span<PjRtDevice* const> XdnaPjrtClient::addressable_devices() const {
  return addressable_devices_;
}

absl::Span<PjRtMemorySpace* const> XdnaPjrtClient::memory_spaces() const {
  return memory_spaces_;
}

absl::string_view XdnaPjrtClient::platform_version() const {
  return platform_version_;
}

absl::StatusOr<PjRtDevice*> XdnaPjrtClient::LookupDevice(
    PjRtGlobalDeviceId global_device_id) const {
  for (PjRtDevice* device : devices_) {
    if (device->global_device_id() == global_device_id) {
      return device;
    }
  }
  return absl::NotFoundError(
      absl::StrCat("No XDNA device with global_device_id ",
                    global_device_id.value()));
}

absl::StatusOr<PjRtDevice*> XdnaPjrtClient::LookupAddressableDevice(
    PjRtLocalDeviceId local_device_id) const {
  for (PjRtDevice* device : addressable_devices_) {
    if (device->local_device_id() == local_device_id) {
      return device;
    }
  }
  return absl::NotFoundError(
      absl::StrCat("No XDNA addressable device with local_device_id ",
                    local_device_id.value()));
}

absl::StatusOr<std::unique_ptr<PjRtBuffer>>
XdnaPjrtClient::BufferFromHostBuffer(
    const void* data, PrimitiveType type, absl::Span<int64_t const> dims,
    std::optional<absl::Span<int64_t const>> byte_strides,
    HostBufferSemantics host_buffer_semantics,
    absl::AnyInvocable<void() &&> on_done_with_host_buffer,
    PjRtMemorySpace* memory_space, const Layout* device_layout) {
  Shape shape = ShapeUtil::MakeShape(type, dims);
  int64_t byte_size = ShapeUtil::ByteSizeOf(shape);

  // Copy data into a host-side buffer.
  std::vector<uint8_t> buffer_data(byte_size);
  if (data != nullptr && byte_size > 0) {
    std::memcpy(buffer_data.data(), data, byte_size);
  }

  // Notify caller we're done with their host buffer.
  if (on_done_with_host_buffer) {
    std::move(on_done_with_host_buffer)();
  }

  // Determine which device and memory space to use.
  PjRtDevice* device = device_.get();
  if (memory_space == nullptr) {
    memory_space = memory_space_.get();
  }

  return std::make_unique<XdnaBuffer>(this, device, memory_space,
                                      std::move(shape),
                                      std::move(buffer_data));
}

std::unique_ptr<PjRtClient> CreateXdnaPjrtClient() {
  return std::make_unique<XdnaPjrtClient>();
}

}  // namespace xla
