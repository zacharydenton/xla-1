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

#ifndef XLA_PJRT_PLUGIN_XDNA_XDNA_PJRT_BUFFER_H_
#define XLA_PJRT_PLUGIN_XDNA_XDNA_PJRT_BUFFER_H_

#include <cstdint>
#include <functional>
#include <memory>
#include <string>
#include <vector>

#include "absl/functional/any_invocable.h"
#include "absl/status/statusor.h"
#include "absl/types/span.h"
#include "xla/literal.h"
#include "xla/pjrt/pjrt_client.h"
#include "xla/pjrt/pjrt_future.h"
#include "xla/shape.h"

namespace xla {

class XdnaBuffer : public PjRtBuffer {
 public:
  XdnaBuffer(PjRtClient* client, PjRtDevice* device,
             PjRtMemorySpace* memory_space, Shape on_device_shape,
             std::vector<uint8_t> data);

  ~XdnaBuffer() override = default;

  const Shape& on_device_shape() const override;

  PjRtMemorySpace* memory_space() const override;
  PjRtDevice* device() const override;
  PjRtClient* client() const override;

  absl::StatusOr<std::unique_ptr<ExternalReference>>
  AcquireExternalReference() override;

  Future<> ToLiteral(MutableLiteralBase* literal) override;
  Future<> LazyToLiteral(
      absl::AnyInvocable<Future<MutableLiteralBase*>() &&> generator) override;

  absl::StatusOr<size_t> GetOnDeviceSizeInBytes() const override;
  Future<> CopyRawToHost(void* dst, int64_t offset,
                         int64_t transfer_size) override;

  void Delete() override;
  absl::StatusOr<std::unique_ptr<ExternalReference>>
  ReleaseDeviceMemoryOwnership(bool wait_for_operations_to_complete) override;
  bool IsDeleted() const override;

  absl::StatusOr<std::unique_ptr<PjRtBuffer>> CopyToMemorySpace(
      PjRtMemorySpace* dst_memory_space) override;

  void CopyToRemoteDevice(Future<std::string> serialized_descriptor,
                           RemoteSendCallback on_done) override;

  Future<> GetReadyFuture() override;
  bool IsOnCpu() const override;

  const std::vector<uint8_t>& raw_data() const { return data_; }

 private:
  PjRtClient* client_;
  PjRtDevice* device_;
  PjRtMemorySpace* memory_space_;
  Shape on_device_shape_;
  std::vector<uint8_t> data_;
  bool is_deleted_ = false;
};

}  // namespace xla

#endif  // XLA_PJRT_PLUGIN_XDNA_XDNA_PJRT_BUFFER_H_
