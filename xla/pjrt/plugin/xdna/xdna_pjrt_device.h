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

#ifndef XLA_PJRT_PLUGIN_XDNA_XDNA_PJRT_DEVICE_H_
#define XLA_PJRT_PLUGIN_XDNA_XDNA_PJRT_DEVICE_H_

#include <memory>
#include <string>

#include "absl/container/flat_hash_map.h"
#include "absl/status/status.h"
#include "absl/strings/string_view.h"
#include "absl/types/span.h"
#include "xla/pjrt/pjrt_client.h"
#include "xla/pjrt/pjrt_common.h"
#include "xla/pjrt/pjrt_device_description.h"

namespace xla {

class XdnaDevice;

class XdnaDeviceDescription : public PjRtDeviceDescription {
 public:
  XdnaDeviceDescription(int id, int process_index);

  int id() const override;
  int process_index() const override;
  absl::string_view device_kind() const override;
  absl::string_view DebugString() const override;
  absl::string_view ToString() const override;
  const absl::flat_hash_map<std::string, PjRtDeviceAttribute>& Attributes()
      const override;

 private:
  int id_;
  int process_index_;
  std::string debug_string_;
  std::string to_string_;
  absl::flat_hash_map<std::string, PjRtDeviceAttribute> attributes_;
};

class XdnaMemorySpace : public PjRtMemorySpace {
 public:
  XdnaMemorySpace(PjRtClient* client, int id);

  PjRtClient* client() const override;
  absl::Span<PjRtDevice* const> devices() const override;
  int id() const override;
  absl::string_view kind() const override;
  int kind_id() const override;
  absl::string_view DebugString() const override;
  absl::string_view ToString() const override;

  void AddDevice(PjRtDevice* device);

 private:
  PjRtClient* client_;
  int id_;
  std::vector<PjRtDevice*> devices_;
  std::string debug_string_;
  std::string to_string_;
};

class XdnaDevice : public PjRtDevice {
 public:
  XdnaDevice(PjRtClient* client, int id, int process_index);

  PjRtClient* client() const override;
  bool IsAddressable() const override;
  const PjRtDeviceDescription& description() const override;
  PjRtLocalHardwareId local_hardware_id() const override;

  absl::Span<PjRtMemorySpace* const> memory_spaces() const override;
  absl::StatusOr<PjRtMemorySpace*> default_memory_space() const override;

  std::unique_ptr<ScopedAsyncTrackingEvent> CreateAsyncTrackingEvent(
      absl::string_view description) const override;
  absl::Status TransferToInfeed(const LiteralSlice& literal) override;
  absl::Status TransferFromOutfeed(MutableBorrowingLiteral literal) override;

  void SetMemorySpace(PjRtMemorySpace* memory_space);

 private:
  PjRtClient* client_;
  XdnaDeviceDescription description_;
  std::vector<PjRtMemorySpace*> memory_spaces_;
};

}  // namespace xla

#endif  // XLA_PJRT_PLUGIN_XDNA_XDNA_PJRT_DEVICE_H_
