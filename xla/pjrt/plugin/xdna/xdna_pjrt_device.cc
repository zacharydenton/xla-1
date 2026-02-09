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

#include "xla/pjrt/plugin/xdna/xdna_pjrt_device.h"

#include <memory>
#include <string>

#include "absl/container/flat_hash_map.h"
#include "absl/status/status.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/string_view.h"
#include "absl/types/span.h"
#include "xla/pjrt/pjrt_client.h"
#include "xla/pjrt/pjrt_common.h"

namespace xla {

// --- XdnaDeviceDescription ---

XdnaDeviceDescription::XdnaDeviceDescription(int id, int process_index)
    : id_(id), process_index_(process_index) {
  debug_string_ = absl::StrCat("XdnaDevice(id=", id_, ")");
  to_string_ = debug_string_;
  attributes_["device_type"] = PjRtDeviceAttribute(std::string("XDNA NPU"));
}

int XdnaDeviceDescription::id() const { return id_; }

int XdnaDeviceDescription::process_index() const { return process_index_; }

absl::string_view XdnaDeviceDescription::device_kind() const {
  static constexpr char kDeviceKind[] = "XDNA";
  return kDeviceKind;
}

absl::string_view XdnaDeviceDescription::DebugString() const {
  return debug_string_;
}

absl::string_view XdnaDeviceDescription::ToString() const {
  return to_string_;
}

const absl::flat_hash_map<std::string, PjRtDeviceAttribute>&
XdnaDeviceDescription::Attributes() const {
  return attributes_;
}

// --- XdnaMemorySpace ---

XdnaMemorySpace::XdnaMemorySpace(PjRtClient* client, int id)
    : client_(client), id_(id) {
  debug_string_ = absl::StrCat("XdnaMemorySpace(id=", id_, ")");
  to_string_ = debug_string_;
}

PjRtClient* XdnaMemorySpace::client() const { return client_; }

absl::Span<PjRtDevice* const> XdnaMemorySpace::devices() const {
  return devices_;
}

int XdnaMemorySpace::id() const { return id_; }

absl::string_view XdnaMemorySpace::kind() const {
  static constexpr char kKind[] = "device";
  return kKind;
}

int XdnaMemorySpace::kind_id() const { return 0; }

absl::string_view XdnaMemorySpace::DebugString() const {
  return debug_string_;
}

absl::string_view XdnaMemorySpace::ToString() const { return to_string_; }

void XdnaMemorySpace::AddDevice(PjRtDevice* device) {
  devices_.push_back(device);
}

// --- XdnaDevice ---

XdnaDevice::XdnaDevice(PjRtClient* client, int id, int process_index)
    : client_(client), description_(id, process_index) {}

PjRtClient* XdnaDevice::client() const { return client_; }

bool XdnaDevice::IsAddressable() const { return true; }

const PjRtDeviceDescription& XdnaDevice::description() const {
  return description_;
}

PjRtLocalHardwareId XdnaDevice::local_hardware_id() const {
  return PjRtLocalHardwareId(description_.id());
}

absl::Span<PjRtMemorySpace* const> XdnaDevice::memory_spaces() const {
  return memory_spaces_;
}

absl::StatusOr<PjRtMemorySpace*> XdnaDevice::default_memory_space() const {
  if (memory_spaces_.empty()) {
    return absl::InternalError("No memory spaces available for XdnaDevice.");
  }
  return memory_spaces_.front();
}

std::unique_ptr<ScopedAsyncTrackingEvent>
XdnaDevice::CreateAsyncTrackingEvent(absl::string_view description) const {
  return nullptr;
}

absl::Status XdnaDevice::TransferToInfeed(const LiteralSlice& literal) {
  return absl::UnimplementedError("XDNA does not support infeed.");
}

absl::Status XdnaDevice::TransferFromOutfeed(
    MutableBorrowingLiteral literal) {
  return absl::UnimplementedError("XDNA does not support outfeed.");
}

void XdnaDevice::SetMemorySpace(PjRtMemorySpace* memory_space) {
  memory_spaces_ = {memory_space};
}

}  // namespace xla
