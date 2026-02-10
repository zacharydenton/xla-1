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

#include <cstdio>
#include <cstring>
#include <exception>
#include <memory>
#include <optional>
#include <string>
#include <unistd.h>
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

// Debug trace that bypasses libc buffering (hermetic toolchain may have
// different stderr buffering behavior in dlopen'd libraries).
#define XDNA_TRACE(msg) \
  do { \
    char buf[256]; \
    int n = snprintf(buf, sizeof(buf), "%s\n", msg); \
    (void)write(STDERR_FILENO, buf, n); \
  } while (0)

#define XDNA_TRACEF(fmt, ...) \
  do { \
    char buf[512]; \
    int n = snprintf(buf, sizeof(buf), fmt "\n", ##__VA_ARGS__); \
    (void)write(STDERR_FILENO, buf, n); \
  } while (0)

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
    std::vector<uint32_t> instr_words,
    std::shared_ptr<HloModule> hlo_module)
    : client_(client),
      device_assignment_(1, 1),
      name_(name),
      xclbin_(std::move(xclbin)),
      hw_context_(std::move(hw_context)),
      kernel_(std::move(kernel)),
      convention_(convention),
      num_inputs_(num_inputs),
      output_shapes_(std::move(output_shapes)),
      instr_words_(std::move(instr_words)),
      hlo_module_(std::move(hlo_module)) {
  addressable_devices_.assign(addressable_devices.begin(),
                              addressable_devices.end());
  addressable_device_logical_ids_.push_back(
      LogicalDeviceIds{/*replica=*/0, /*partition=*/0});
  if (!addressable_devices_.empty()) {
    device_assignment_(0, 0) =
        addressable_devices_.front()->global_device_id().value();
  }
  LOG(INFO) << "XDNA: Loaded executable '" << name_ << "' (xclbin)";
  XDNA_TRACEF("XDNA: XdnaExecutable xclbin ctor done ('%s')", name_.c_str());
}

PjRtClient* XdnaExecutable::client() const { return client_; }

int XdnaExecutable::num_replicas() const { return 1; }

int XdnaExecutable::num_partitions() const { return 1; }

const DeviceAssignment& XdnaExecutable::device_assignment() const {
  XDNA_TRACE("XDNA: device_assignment() called");
  return device_assignment_;
}

absl::Span<const PjRtLoadedExecutable::LogicalDeviceIds>
XdnaExecutable::addressable_device_logical_ids() const {
  XDNA_TRACE("XDNA: addressable_device_logical_ids() called");
  return addressable_device_logical_ids_;
}

absl::Span<PjRtDevice* const> XdnaExecutable::addressable_devices() const {
  XDNA_TRACE("XDNA: addressable_devices() called");
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

  XDNA_TRACEF("XDNA: Execute called with %zu args, returned_futures=%d",
              argument_handles[0].size(),
              static_cast<int>(returned_futures.has_value()));

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

  XDNA_TRACEF("XDNA: ExecuteSharded '%s' with %zu args, convention=%s, "
              "has_outputs=%d, num_output_shapes=%zu, instr_words=%zu",
              name_.c_str(), argument_handles.size(),
              (convention_ == XdnaKernelConvention::kDpu ? "kDpu" : "kDirect"),
              static_cast<int>(has_outputs), output_shapes_.size(),
              instr_words_.size());

  try {
    xrt::run run(kernel_);
    std::vector<xrt::bo> input_bos;
    std::vector<xrt::bo> output_bos;
    xrt::bo instr_bo;

    // For kDpu: set up instruction BO first, then data BOs.
    if (convention_ == XdnaKernelConvention::kDpu && !instr_words_.empty()) {
      size_t instr_size = instr_words_.size() * sizeof(uint32_t);
      int instr_grp = kernel_.group_id(1);
      XDNA_TRACEF("XDNA: Setting up instruction BO, size=%zu words=%zu grp=%d",
                  instr_size, instr_words_.size(), instr_grp);
      instr_bo = xrt::bo(hw_context_, instr_size,
                         xrt::bo::flags::cacheable, instr_grp);
      std::memcpy(instr_bo.map<void*>(), instr_words_.data(), instr_size);
      instr_bo.sync(XCL_BO_SYNC_BO_TO_DEVICE);

      run.set_arg(0, static_cast<uint64_t>(3));  // opcode: DPU dispatch
      run.set_arg(1, instr_bo);
      run.set_arg(2, static_cast<uint32_t>(instr_words_.size()));
      XDNA_TRACE("XDNA: Instruction BO set up.");
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
      XDNA_TRACEF("XDNA: Input BO %d: size=%zu kernel_arg_idx=%d grp=%d",
                  i, raw.size(), kernel_arg_idx, grp);
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
        XDNA_TRACEF("XDNA: Output BO %d: size=%ld kernel_arg_idx=%d grp=%d",
                    i, byte_size, kernel_arg_idx, grp);
        xrt::bo dev_bo(hw_context_, byte_size, xrt::bo::flags::host_only, grp);

        // Zero-initialize output buffer.
        std::memset(dev_bo.map<void*>(), 0, byte_size);
        dev_bo.sync(XCL_BO_SYNC_BO_TO_DEVICE);

        run.set_arg(kernel_arg_idx, dev_bo);
        output_bos.push_back(std::move(dev_bo));
      }
    }

    XDNA_TRACE("XDNA: Starting kernel run...");
    run.start();
    XDNA_TRACE("XDNA: Kernel started, waiting for completion...");
    run.wait2();
    XDNA_TRACE("XDNA: Kernel completed.");

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
      // kDirect: sync results back from device and return as output buffers.
      PjRtMemorySpace* mem_space = device->default_memory_space().value();
      for (int i = 0; i < static_cast<int>(argument_handles.size()); ++i) {
        auto* xdna_buf = dynamic_cast<XdnaBuffer*>(argument_handles[i]);
        input_bos[i].sync(XCL_BO_SYNC_BO_FROM_DEVICE);
        int64_t byte_size =
            ShapeUtil::ByteSizeOf(xdna_buf->on_device_shape());
        std::vector<uint8_t> result_data(byte_size);
        std::memcpy(result_data.data(), input_bos[i].map<const void*>(),
                    byte_size);
        results.push_back(std::make_unique<XdnaBuffer>(
            client_, device, mem_space, xdna_buf->on_device_shape(),
            std::move(result_data)));
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

absl::string_view XdnaExecutable::name() const {
  XDNA_TRACEF("XDNA: name() called -> '%s'", name_.c_str());
  return name_;
}

absl::StatusOr<std::string> XdnaExecutable::FingerprintExecutable() const {
  XDNA_TRACE("XDNA: FingerprintExecutable() called");
  return name_;
}

absl::StatusOr<std::string> XdnaExecutable::SerializeExecutable() const {
  XDNA_TRACE("XDNA: SerializeExecutable() called");
  return absl::UnimplementedError(
      "XdnaExecutable::SerializeExecutable is not implemented.");
}

absl::StatusOr<struct CompileOptions>
XdnaExecutable::GetCompileOptions() const {
  XDNA_TRACE("XDNA: GetCompileOptions() called");
  CompileOptions options;
  options.executable_build_options.set_num_replicas(1);
  options.executable_build_options.set_num_partitions(1);
  return options;
}

absl::StatusOr<std::vector<std::shared_ptr<HloModule>>>
XdnaExecutable::GetHloModules() const {
  XDNA_TRACE("XDNA: GetHloModules() called");
  if (hlo_module_ == nullptr) {
    return absl::UnimplementedError(
        "XdnaExecutable::GetHloModules: no HLO module available "
        "(ELF/legacy path).");
  }
  return std::vector<std::shared_ptr<HloModule>>{hlo_module_};
}

absl::StatusOr<std::vector<Shape>> XdnaExecutable::GetOutputShapes() const {
  XDNA_TRACE("XDNA: GetOutputShapes() called");
  return absl::UnimplementedError(
      "XdnaExecutable::GetOutputShapes is not implemented.");
}

absl::StatusOr<std::vector<std::vector<PrimitiveType>>>
XdnaExecutable::GetOutputElementTypes() const {
  XDNA_TRACE("XDNA: GetOutputElementTypes() called");
  return absl::UnimplementedError(
      "XdnaExecutable::GetOutputElementTypes is not implemented.");
}

absl::StatusOr<std::vector<std::vector<DimensionVector>>>
XdnaExecutable::GetOutputDimensions() const {
  XDNA_TRACE("XDNA: GetOutputDimensions() called");
  return absl::UnimplementedError(
      "XdnaExecutable::GetOutputDimensions is not implemented.");
}

absl::StatusOr<std::vector<std::vector<absl::string_view>>>
XdnaExecutable::GetOutputMemoryKinds() const {
  XDNA_TRACE("XDNA: GetOutputMemoryKinds() called");
  return absl::UnimplementedError(
      "XdnaExecutable::GetOutputMemoryKinds is not implemented.");
}

absl::StatusOr<std::vector<std::shared_ptr<const PjRtLayout>>>
XdnaExecutable::GetParameterLayouts() const {
  XDNA_TRACE("XDNA: GetParameterLayouts() called");
  return absl::UnimplementedError(
      "XdnaExecutable::GetParameterLayouts is not implemented.");
}

absl::StatusOr<std::vector<std::shared_ptr<const PjRtLayout>>>
XdnaExecutable::GetOutputLayouts() const {
  XDNA_TRACE("XDNA: GetOutputLayouts() called");
  return absl::UnimplementedError(
      "XdnaExecutable::GetOutputLayouts is not implemented.");
}

absl::StatusOr<CompiledMemoryStats>
XdnaExecutable::GetCompiledMemoryStats() const {
  XDNA_TRACE("XDNA: GetCompiledMemoryStats() called");
  return absl::UnimplementedError(
      "XdnaExecutable::GetCompiledMemoryStats is not implemented.");
}

std::optional<std::vector<OpSharding>>
XdnaExecutable::GetOutputShardings() const {
  XDNA_TRACE("XDNA: GetOutputShardings() called");
  return std::nullopt;
}

std::optional<std::vector<OpSharding>>
XdnaExecutable::GetParameterShardings() const {
  XDNA_TRACE("XDNA: GetParameterShardings() called");
  return std::nullopt;
}

int64_t XdnaExecutable::SizeOfGeneratedCodeInBytes() const {
  XDNA_TRACE("XDNA: SizeOfGeneratedCodeInBytes() called");
  return 0;
}

}  // namespace xla
