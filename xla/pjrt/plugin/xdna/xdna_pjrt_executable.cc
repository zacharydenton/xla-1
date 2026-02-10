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
#include "xla/shape_util.h"

namespace xla {

XdnaExecutable::XdnaExecutable(
    PjRtClient* client, absl::Span<PjRtDevice* const> addressable_devices,
    xrt::elf elf, xrt::hw_context hw_context, xrt::kernel kernel,
    absl::string_view name, XdnaKernelConvention convention,
    int num_inputs, std::vector<Shape> output_shapes)
    : client_(client),
      device_assignment_(1, 1),
      name_(name),
      elf_(std::move(elf)),
      hw_context_(std::move(hw_context)),
      kernel_(std::move(kernel)),
      convention_(convention),
      num_inputs_(num_inputs),
      output_shapes_(std::move(output_shapes)) {
  addressable_devices_.assign(addressable_devices.begin(),
                              addressable_devices.end());
  addressable_device_logical_ids_.push_back(
      LogicalDeviceIds{/*replica=*/0, /*partition=*/0});
  if (!addressable_devices_.empty()) {
    device_assignment_(0, 0) =
        addressable_devices_.front()->global_device_id().value();
  }
  LOG(INFO) << "XDNA: Loaded executable '" << name_ << "' (ELF)";
}

XdnaExecutable::XdnaExecutable(
    PjRtClient* client, absl::Span<PjRtDevice* const> addressable_devices,
    xrt::xclbin xclbin, xrt::hw_context hw_context, xrt::kernel kernel,
    absl::string_view name, XdnaKernelConvention convention,
    int num_inputs, std::vector<Shape> output_shapes,
    std::vector<uint32_t> instr_words)
    : client_(client),
      device_assignment_(1, 1),
      name_(name),
      xclbin_(std::move(xclbin)),
      hw_context_(std::move(hw_context)),
      kernel_(std::move(kernel)),
      convention_(convention),
      num_inputs_(num_inputs),
      output_shapes_(std::move(output_shapes)),
      instr_words_(std::move(instr_words)) {
  addressable_devices_.assign(addressable_devices.begin(),
                              addressable_devices.end());
  addressable_device_logical_ids_.push_back(
      LogicalDeviceIds{/*replica=*/0, /*partition=*/0});
  if (!addressable_devices_.empty()) {
    device_assignment_(0, 0) =
        addressable_devices_.front()->global_device_id().value();
  }
  LOG(INFO) << "XDNA: Loaded executable '" << name_ << "' (xclbin)";
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

  // For kDpu convention with output shapes: inputs are argument_handles,
  // outputs are allocated fresh. For kDirect: all args are in-place.
  bool has_outputs = (convention_ == XdnaKernelConvention::kDpu &&
                      !output_shapes_.empty());

  try {
    xrt::run run(kernel_);
    std::vector<xrt::bo> input_bos;
    std::vector<xrt::bo> output_bos;
    xrt::bo instr_bo;

    // For kDpu: set up instruction BO first, then data BOs.
    if (convention_ == XdnaKernelConvention::kDpu && !instr_words_.empty()) {
      size_t instr_size = instr_words_.size() * sizeof(uint32_t);
      int instr_grp = kernel_.group_id(1);
      instr_bo = xrt::bo(hw_context_, instr_size,
                         xrt::bo::flags::cacheable, instr_grp);
      std::memcpy(instr_bo.map<void*>(), instr_words_.data(), instr_size);
      instr_bo.sync(XCL_BO_SYNC_BO_TO_DEVICE);

      run.set_arg(0, static_cast<uint64_t>(3));  // opcode: DPU dispatch
      run.set_arg(1, instr_bo);
      run.set_arg(2, static_cast<uint32_t>(instr_words_.size()));
    }

    // Bind input buffers.
    input_bos.reserve(argument_handles.size());
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

      auto raw = xdna_buf->raw_data();
      int kernel_arg_idx =
          (convention_ == XdnaKernelConvention::kDpu) ? i + 3 : i;
      int grp = kernel_.group_id(kernel_arg_idx);
      // xclbin-based kernels (kDpu) need host_only BOs; NPU accesses host
      // DDR via DMA. ELF-based kernels (kDirect) use normal BOs.
      auto bo_flags = (convention_ == XdnaKernelConvention::kDpu)
                          ? xrt::bo::flags::host_only
                          : xrt::bo::flags::normal;
      xrt::bo dev_bo(hw_context_, raw.size(), bo_flags, grp);

      std::memcpy(dev_bo.map<void*>(), raw.data(), raw.size());
      dev_bo.sync(XCL_BO_SYNC_BO_TO_DEVICE);

      run.set_arg(kernel_arg_idx, dev_bo);
      input_bos.push_back(std::move(dev_bo));
    }

    // Allocate and bind output buffers.
    if (has_outputs) {
      int num_inputs = static_cast<int>(argument_handles.size());
      output_bos.reserve(output_shapes_.size());
      for (int i = 0; i < static_cast<int>(output_shapes_.size()); ++i) {
        int64_t byte_size = ShapeUtil::ByteSizeOf(output_shapes_[i]);
        int kernel_arg_idx = num_inputs + i + 3;  // after DPU prefix + inputs
        int grp = kernel_.group_id(kernel_arg_idx);
        xrt::bo dev_bo(hw_context_, byte_size, xrt::bo::flags::host_only, grp);

        // Zero-initialize output buffer.
        std::memset(dev_bo.map<void*>(), 0, byte_size);
        dev_bo.sync(XCL_BO_SYNC_BO_TO_DEVICE);

        run.set_arg(kernel_arg_idx, dev_bo);
        output_bos.push_back(std::move(dev_bo));
      }
    }

    run.start();
    run.wait2();

    // Build output buffers from device results.
    std::vector<std::unique_ptr<PjRtBuffer>> results;
    if (has_outputs) {
      for (int i = 0; i < static_cast<int>(output_shapes_.size()); ++i) {
        output_bos[i].sync(XCL_BO_SYNC_BO_FROM_DEVICE);
        int64_t byte_size = ShapeUtil::ByteSizeOf(output_shapes_[i]);
        std::vector<uint8_t> result_data(byte_size);
        std::memcpy(result_data.data(), output_bos[i].map<const void*>(),
                    byte_size);

        PjRtMemorySpace* mem_space = device->default_memory_space().value();
        results.push_back(std::make_unique<XdnaBuffer>(
            client_, device, mem_space, output_shapes_[i],
            std::move(result_data)));
      }
    } else {
      // kDirect: copy results back to input buffers in-place.
      for (int i = 0; i < static_cast<int>(argument_handles.size()); ++i) {
        auto* xdna_buf = dynamic_cast<XdnaBuffer*>(argument_handles[i]);
        input_bos[i].sync(XCL_BO_SYNC_BO_FROM_DEVICE);
        auto dst = xdna_buf->mutable_raw_data();
        std::memcpy(dst.data(), input_bos[i].map<const void*>(), dst.size());
      }
    }

    LOG(INFO) << "XDNA: Kernel '" << name_ << "' execution completed.";

    if (fill_future) {
      returned_future = Future<>(absl::OkStatus());
    }

    return results;
  } catch (const xrt::run::command_error& e) {
    return absl::InternalError(
        absl::StrCat("XDNA kernel execution failed: ", e.what()));
  } catch (const std::exception& e) {
    return absl::InternalError(
        absl::StrCat("XDNA kernel execution failed: ", e.what()));
  }
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
