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

#include "xla/pjrt/plugin/xdna/xdna_pjrt_buffer.h"

#include <cstdint>
#include <cstring>
#include <memory>
#include <utility>
#include <vector>

#include "absl/functional/any_invocable.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "xla/literal.h"
#include "xla/pjrt/pjrt_client.h"
#include "xla/pjrt/pjrt_future.h"
#include "xla/shape.h"
#include "xla/shape_util.h"

namespace xla {

XdnaBuffer::XdnaBuffer(PjRtClient* client, PjRtDevice* device,
                       PjRtMemorySpace* memory_space, Shape on_device_shape,
                       std::vector<uint8_t> data)
    : client_(client),
      device_(device),
      memory_space_(memory_space),
      on_device_shape_(std::move(on_device_shape)),
      data_(std::move(data)) {}

const Shape& XdnaBuffer::on_device_shape() const { return on_device_shape_; }

PjRtMemorySpace* XdnaBuffer::memory_space() const { return memory_space_; }

PjRtDevice* XdnaBuffer::device() const { return device_; }

PjRtClient* XdnaBuffer::client() const { return client_; }

absl::StatusOr<std::unique_ptr<PjRtBuffer::ExternalReference>>
XdnaBuffer::AcquireExternalReference() {
  return absl::UnimplementedError(
      "AcquireExternalReference not supported for XDNA buffers.");
}

Future<> XdnaBuffer::ToLiteral(MutableLiteralBase* literal) {
  if (is_deleted_) {
    return Future<>(
        absl::InternalError("Buffer has been deleted."));
  }
  std::memcpy(literal->untyped_data(), data_.data(),
              std::min(data_.size(),
                       static_cast<size_t>(ShapeUtil::ByteSizeOf(
                           on_device_shape_))));
  return Future<>(absl::OkStatus());
}

Future<> XdnaBuffer::LazyToLiteral(
    absl::AnyInvocable<Future<MutableLiteralBase*>() &&> generator) {
  Future<MutableLiteralBase*> literal_future = std::move(generator)();
  absl::StatusOr<MutableLiteralBase*> literal_or = literal_future.Await();
  if (!literal_or.ok()) {
    return Future<>(literal_or.status());
  }
  MutableLiteralBase* literal = *literal_or;
  if (is_deleted_) {
    return Future<>(absl::InternalError("Buffer has been deleted."));
  }
  std::memcpy(literal->untyped_data(), data_.data(),
              std::min(data_.size(),
                       static_cast<size_t>(
                           ShapeUtil::ByteSizeOf(on_device_shape_))));
  return Future<>(absl::OkStatus());
}

absl::StatusOr<size_t> XdnaBuffer::GetOnDeviceSizeInBytes() const {
  return data_.size();
}

Future<> XdnaBuffer::CopyRawToHost(void* dst, int64_t offset,
                                   int64_t transfer_size) {
  if (is_deleted_) {
    return Future<>(
        absl::InternalError("Buffer has been deleted."));
  }
  if (offset + transfer_size > static_cast<int64_t>(data_.size())) {
    return Future<>(absl::InvalidArgumentError(
        "CopyRawToHost: offset + transfer_size exceeds buffer size."));
  }
  std::memcpy(dst, data_.data() + offset, transfer_size);
  return Future<>(absl::OkStatus());
}

void XdnaBuffer::Delete() { is_deleted_ = true; }

absl::StatusOr<std::unique_ptr<PjRtBuffer::ExternalReference>>
XdnaBuffer::ReleaseDeviceMemoryOwnership(
    bool wait_for_operations_to_complete) {
  return absl::UnimplementedError(
      "ReleaseDeviceMemoryOwnership not supported for XDNA buffers.");
}

bool XdnaBuffer::IsDeleted() const { return is_deleted_; }

absl::StatusOr<std::unique_ptr<PjRtBuffer>> XdnaBuffer::CopyToMemorySpace(
    PjRtMemorySpace* dst_memory_space) {
  if (is_deleted_) {
    return absl::InternalError("Buffer has been deleted.");
  }
  return std::make_unique<XdnaBuffer>(client_, device_, dst_memory_space,
                                      on_device_shape_, data_);
}

void XdnaBuffer::CopyToRemoteDevice(Future<std::string> serialized_descriptor,
                                     RemoteSendCallback on_done) {
  on_done(absl::UnimplementedError(
              "CopyToRemoteDevice not supported for XDNA buffers."),
          false);
}

Future<> XdnaBuffer::GetReadyFuture() {
  return Future<>(absl::OkStatus());
}

bool XdnaBuffer::IsOnCpu() const { return false; }

}  // namespace xla
