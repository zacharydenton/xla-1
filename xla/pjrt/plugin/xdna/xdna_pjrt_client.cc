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
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <unistd.h>
#include <exception>
#include <memory>
#include <optional>
#include <string>
#include <utility>
#include <vector>

#include "absl/functional/any_invocable.h"
#include "absl/log/log.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/string_view.h"
#include "absl/types/span.h"
#include "mlir/IR/BuiltinOps.h"
#include "xla/client/executable_build_options.h"
#include "xla/hlo/builder/xla_computation.h"
#include "xla/hlo/ir/hlo_module.h"
#include "xla/layout_util.h"
#include "xla/pjrt/mlir_to_hlo.h"
#include "xla/pjrt/pjrt_client.h"
#include "xla/pjrt/plugin/xdna/xdna_compiler.h"
#include "xla/pjrt/plugin/xdna/xdna_hlo_passes.h"
#include "xla/pjrt/plugin/xdna/xdna_pjrt_buffer.h"
#include "xla/pjrt/plugin/xdna/xdna_pjrt_device.h"
#include "xla/pjrt/plugin/xdna/xdna_codegen.h"
#include "xla/pjrt/plugin/xdna/xdna_pjrt_executable.h"
#include "xla/pjrt/utils.h"
#include "xla/service/hlo_module_config.h"
#include "xla/service/hlo_module_util.h"
#include "xla/shape.h"
#include "xla/shape_util.h"
#include "xla/tsl/platform/errors.h"
#include "xla/tsl/platform/statusor.h"
#include "xla/xla.pb.h"
#include "tsl/platform/fingerprint.h"

#define XDNA_TRACE(msg) \
  do { \
    char buf_[256]; \
    int n_ = snprintf(buf_, sizeof(buf_), "%s\n", msg); \
    (void)write(STDERR_FILENO, buf_, n_); \
  } while (0)

#define XDNA_TRACEF(fmt, ...) \
  do { \
    char buf_[512]; \
    int n_ = snprintf(buf_, sizeof(buf_), fmt "\n", ##__VA_ARGS__); \
    (void)write(STDERR_FILENO, buf_, n_); \
  } while (0)

namespace xla {

namespace {
constexpr char kXdnaPlatformName[] = "xdna";
constexpr char kXdnaBackendName[] = "xdna";
}  // namespace

XdnaPjrtClient::XdnaPjrtClient(xrt::device device)
    : xrt_device_(std::move(device)) {
  std::string device_name =
      xrt_device_.get_info<xrt::info::device::name>();
  LOG(INFO) << "XDNA: Opened XRT device: " << device_name;
  platform_version_ = absl::StrCat("XDNA NPU (", device_name, ")");

  // Create one device and one memory space.
  device_ = std::make_unique<XdnaDevice>(this, /*id=*/0, /*process_index=*/0);
  memory_space_ = std::make_unique<XdnaMemorySpace>(this, /*id=*/0);

  // Wire device and memory space together.
  device_->SetMemorySpace(memory_space_.get());
  memory_space_->AddDevice(device_.get());

  devices_.push_back(device_.get());
  addressable_devices_.push_back(device_.get());
  memory_spaces_.push_back(memory_space_.get());
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
  XDNA_TRACEF("XDNA: BufferFromHostBuffer type=%d byte_size=%ld",
              static_cast<int>(type), byte_size);

  // Copy host data into a raw byte buffer. Device-accessible BOs are allocated
  // at execution time through the hw_context.
  std::vector<uint8_t> buffer_data(byte_size, 0);
  if (data != nullptr && byte_size > 0) {
    std::memcpy(buffer_data.data(), data, byte_size);
  }

  // Notify caller we're done with their host buffer.
  if (on_done_with_host_buffer) {
    std::move(on_done_with_host_buffer)();
  }

  PjRtDevice* device = device_.get();
  if (memory_space == nullptr) {
    memory_space = memory_space_.get();
  }

  XDNA_TRACE("XDNA: BufferFromHostBuffer done.");
  return std::make_unique<XdnaBuffer>(this, device, memory_space,
                                      std::move(shape),
                                      std::move(buffer_data));
}

namespace {
absl::StatusOr<Shape> ChooseCompactLayoutForShape(Shape shape) {
  LayoutUtil::SetToDefaultLayout(&shape);
  return shape;
}
}  // namespace

absl::StatusOr<std::unique_ptr<PjRtLoadedExecutable>>
XdnaPjrtClient::CompileAndLoad(const XlaComputation& computation,
                                CompileOptions options) {
  std::vector<const Shape*> argument_layout_pointers;
  const ExecutableBuildOptions& build_options =
      options.executable_build_options;
  const bool allow_auto_layout =
      build_options.has_debug_options() &&
      build_options.debug_options().xla_pjrt_allow_auto_layout_in_hlo();
  TF_RETURN_IF_ERROR(DetermineArgumentLayoutsFromCompileOptions(
      computation,
      [allow_auto_layout](Shape shape) -> absl::StatusOr<Shape> {
        if (allow_auto_layout && !shape.has_layout()) {
          return shape;
        }
        return ChooseCompactLayoutForShape(shape);
      },
      options.argument_layouts, &options.executable_build_options,
      &argument_layout_pointers));
  return CompileInternal(computation, argument_layout_pointers, options);
}

absl::StatusOr<std::unique_ptr<PjRtLoadedExecutable>>
XdnaPjrtClient::CompileAndLoad(mlir::ModuleOp module,
                                CompileOptions options) {
  XlaComputation xla_computation;
  ExecutableBuildOptions& exec_build_options = options.executable_build_options;
  TF_RETURN_IF_ERROR(MlirToXlaComputation(
      module, xla_computation,
      /*use_tuple_args=*/options.parameter_is_tupled_arguments,
      /*return_tuple=*/false, &exec_build_options));
  return CompileAndLoad(xla_computation, options);
}

absl::StatusOr<std::unique_ptr<PjRtLoadedExecutable>>
XdnaPjrtClient::CompileInternal(
    const XlaComputation& computation,
    const std::vector<const Shape*>& argument_shapes,
    CompileOptions options) {
  TF_RETURN_IF_ERROR(options.ApplyAllOptionOverrides());

  TF_ASSIGN_OR_RETURN(ProgramShape program_shape,
                      computation.GetProgramShape());

  const ExecutableBuildOptions& build_options =
      options.executable_build_options;
  ExecutionOptions execution_options =
      CreateExecutionOptions(build_options, &program_shape);

  TF_ASSIGN_OR_RETURN(
      std::unique_ptr<HloModuleConfig> hlo_module_config,
      CreateModuleConfig(program_shape, argument_shapes, &execution_options,
                         execution_options.num_replicas(),
                         /*num_threads=*/std::nullopt,
                         /*aot_options=*/nullptr));

  TF_ASSIGN_OR_RETURN(
      std::unique_ptr<HloModule> hlo_module,
      HloModule::CreateFromProto(computation.proto(), *hlo_module_config));

  if (build_options.num_partitions() != 1) {
    return absl::UnimplementedError(
        "XDNA only supports num_partitions=1.");
  }

  // Run HLO optimization passes.
  if (!build_options.run_backend_only()) {
    TF_ASSIGN_OR_RETURN(hlo_module, RunXdnaHloPasses(std::move(hlo_module)));
  }

  // Check compilation cache.
  bool use_cache = !std::getenv("XDNA_NO_CACHE");
  uint64_t cache_key = 0;
  if (use_cache) {
    // Use the entry computation's text representation as cache key.
    // This is stable across compilations of the same HLO (unlike ToProto()
    // which includes varying module IDs).
    // Include XDNA_NUM_CORES override in key when set, so debug overrides
    // don't get stale cache hits.
    std::string cache_input = hlo_module->entry_computation()->ToString();
    const char* nc = std::getenv("XDNA_NUM_CORES");
    if (nc) absl::StrAppend(&cache_input, ":cores=", nc);
    cache_key = tsl::Fingerprint64(cache_input);

    auto it = compilation_cache_.find(cache_key);
    if (it != compilation_cache_.end()) {
      XDNA_TRACE("XDNA: Compilation cache hit — skipping codegen.");
      const auto& cached = it->second;
      try {
        std::vector<char> xclbin_data(cached.xclbin_bytes.begin(),
                                       cached.xclbin_bytes.end());
        xrt::xclbin xclbin(xclbin_data);
        auto uuid = xrt_device_.register_xclbin(xclbin);
        xrt::hw_context hw_ctx(xrt_device_, uuid);
        xrt::kernel kernel(hw_ctx, cached.kernel_name);

        return std::make_unique<XdnaExecutable>(
            this, addressable_devices_, std::move(xclbin), std::move(hw_ctx),
            std::move(kernel), cached.kernel_name,
            XdnaKernelConvention::kDpu, cached.num_inputs,
            std::vector<Shape>(cached.output_shapes),
            std::vector<uint32_t>(cached.instr_words),
            cached.hlo_module->Clone());
      } catch (const std::exception& e) {
        LOG(WARNING) << "XDNA: Cache hit but xclbin load failed: " << e.what()
                     << ". Recompiling.";
        // Fall through to full compilation.
      }
    }
  }

  // Extract output shapes before compilation consumes the module.
  const Shape& result_shape = hlo_module->result_shape();
  std::vector<Shape> output_shapes;
  if (result_shape.IsTuple()) {
    for (const Shape& elem : result_shape.tuple_shapes()) {
      output_shapes.push_back(elem);
    }
  } else {
    output_shapes.push_back(result_shape);
  }

  int num_inputs = hlo_module->entry_computation()
                       ->num_parameters();

  // Keep a copy of the HLO module for GetHloModules() queries.
  auto hlo_module_copy = hlo_module->Clone();

  // Compile HLO to xclbin.
  TF_ASSIGN_OR_RETURN(XdnaCodegenResult codegen_result,
                      XdnaCompiler::Compile(std::move(hlo_module)));

  XDNA_TRACEF("XDNA: Loading xclbin (%zu bytes), kernel='%s'",
              codegen_result.xclbin_bytes.size(),
              codegen_result.kernel_name.c_str());

  // Store in cache before moving data into the executable.
  if (use_cache) {
    compilation_cache_[cache_key] = CachedCompilation{
        .xclbin_bytes = codegen_result.xclbin_bytes,
        .kernel_name = codegen_result.kernel_name,
        .instr_words = codegen_result.instr_words,
        .output_shapes = output_shapes,
        .num_inputs = num_inputs,
        .hlo_module = std::shared_ptr<const HloModule>(
            hlo_module_copy->Clone()),
    };
    XDNA_TRACE("XDNA: Compilation result cached.");
  }

  // Load the xclbin and create an executable with kDpu convention.
  try {
    // Construct xrt::xclbin from raw bytes.
    std::vector<char> xclbin_data(codegen_result.xclbin_bytes.begin(),
                                  codegen_result.xclbin_bytes.end());
    XDNA_TRACE("XDNA: Constructing xrt::xclbin...");
    xrt::xclbin xclbin(xclbin_data);

    // Register xclbin with device and create hw_context from UUID.
    XDNA_TRACE("XDNA: Registering xclbin with device...");
    auto uuid = xrt_device_.register_xclbin(xclbin);
    XDNA_TRACE("XDNA: Creating hw_context...");
    xrt::hw_context hw_ctx(xrt_device_, uuid);
    XDNA_TRACE("XDNA: Creating kernel...");
    xrt::kernel kernel(hw_ctx, codegen_result.kernel_name);
    XDNA_TRACE("XDNA: Compile complete, creating executable.");

    return std::make_unique<XdnaExecutable>(
        this, addressable_devices_, std::move(xclbin), std::move(hw_ctx),
        std::move(kernel), codegen_result.kernel_name,
        XdnaKernelConvention::kDpu, num_inputs, std::move(output_shapes),
        std::move(codegen_result.instr_words),
        std::move(hlo_module_copy));
  } catch (const std::exception& e) {
    return absl::InternalError(
        absl::StrCat("Failed to load compiled xclbin: ", e.what()));
  }
}

absl::StatusOr<std::unique_ptr<PjRtLoadedExecutable>>
XdnaPjrtClient::LoadSerializedExecutable(
    absl::string_view serialized, std::optional<CompileOptions> options,
    const LoadOptions& load_options) {
  if (serialized.empty()) {
    return absl::InvalidArgumentError("Serialized executable is empty.");
  }

  try {
    // Parse the raw ELF bytes.
    xrt::elf elf(serialized.data(), serialized.size());

    // Create a hardware context from the ELF.
    xrt::hw_context hw_ctx(xrt_device_, elf);

    // Extract kernel name from the ELF.
    auto kernels = elf.get_kernels();
    if (kernels.empty()) {
      return absl::InvalidArgumentError("ELF contains no kernels.");
    }
    std::string kernel_name = kernels[0].get_name();

    // Create a kernel handle.
    xrt::kernel kernel(hw_ctx, kernel_name);

    return std::make_unique<XdnaExecutable>(
        this, addressable_devices_, std::move(elf), std::move(hw_ctx),
        std::move(kernel), kernel_name);
  } catch (const std::exception& e) {
    return absl::InternalError(
        absl::StrCat("Failed to load ELF executable: ", e.what()));
  }
}

absl::StatusOr<std::unique_ptr<PjRtClient>> CreateXdnaPjrtClient() {
  try {
    xrt::device device(0);
    return std::make_unique<XdnaPjrtClient>(std::move(device));
  } catch (const std::exception& e) {
    return absl::UnavailableError(
        absl::StrCat("Failed to open XDNA device: ", e.what()));
  }
}

}  // namespace xla
