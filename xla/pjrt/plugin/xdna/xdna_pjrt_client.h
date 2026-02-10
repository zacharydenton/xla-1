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

#ifndef XLA_PJRT_PLUGIN_XDNA_XDNA_PJRT_CLIENT_H_
#define XLA_PJRT_PLUGIN_XDNA_XDNA_PJRT_CLIENT_H_

#include <cstdint>
#include <memory>
#include <optional>
#include <string>
#include <vector>

#include "absl/container/flat_hash_map.h"
#include "absl/status/statusor.h"
#include "absl/strings/string_view.h"
#include "absl/types/span.h"
#include "mlir/IR/BuiltinOps.h"
#include "xla/hlo/builder/xla_computation.h"
#include "xla/pjrt/pjrt_client.h"
#include "xla/pjrt/pjrt_compiler.h"
#include "xla/pjrt/plugin/xdna/xdna_pjrt_buffer.h"
#include "xla/pjrt/plugin/xdna/xdna_pjrt_device.h"
#include "xla/pjrt/plugin/xdna/xdna_pjrt_executable.h"
#include "xla/shape.h"
#include "xrt/xrt_device.h"

namespace xla {

// Creates the XDNA PJRT client. Returns an error if no NPU device is available.
absl::StatusOr<std::unique_ptr<PjRtClient>> CreateXdnaPjrtClient();

class XdnaPjrtClient : public PjRtClient {
 public:
  explicit XdnaPjrtClient(xrt::device device);
  ~XdnaPjrtClient() override = default;

  absl::string_view platform_name() const override;
  int process_index() const override;
  int device_count() const override;
  int addressable_device_count() const override;
  absl::Span<PjRtDevice* const> devices() const override;
  absl::Span<PjRtDevice* const> addressable_devices() const override;
  absl::Span<PjRtMemorySpace* const> memory_spaces() const override;
  PjRtPlatformId platform_id() const override;
  absl::string_view platform_version() const override;

  absl::StatusOr<PjRtDevice*> LookupDevice(
      PjRtGlobalDeviceId global_device_id) const override;
  absl::StatusOr<PjRtDevice*> LookupAddressableDevice(
      PjRtLocalDeviceId local_device_id) const override;

  absl::StatusOr<std::unique_ptr<PjRtBuffer>> BufferFromHostBuffer(
      const void* data, PrimitiveType type, absl::Span<int64_t const> dims,
      std::optional<absl::Span<int64_t const>> byte_strides,
      HostBufferSemantics host_buffer_semantics,
      absl::AnyInvocable<void() &&> on_done_with_host_buffer,
      PjRtMemorySpace* memory_space, const Layout* device_layout) override;

  absl::StatusOr<std::unique_ptr<PjRtLoadedExecutable>> CompileAndLoad(
      const XlaComputation& computation, CompileOptions options) override;

  absl::StatusOr<std::unique_ptr<PjRtLoadedExecutable>> CompileAndLoad(
      mlir::ModuleOp module, CompileOptions options) override;

  absl::StatusOr<std::unique_ptr<PjRtLoadedExecutable>>
  LoadSerializedExecutable(absl::string_view serialized,
                           std::optional<CompileOptions> options,
                           const LoadOptions& load_options) override;

 private:
  absl::StatusOr<std::unique_ptr<PjRtLoadedExecutable>> CompileInternal(
      const XlaComputation& computation,
      const std::vector<const Shape*>& argument_shapes,
      CompileOptions options);

  // Cached compilation result for xclbin reuse.
  struct CachedCompilation {
    std::vector<uint8_t> xclbin_bytes;
    std::string kernel_name;
    std::vector<uint32_t> instr_words;
    std::vector<Shape> output_shapes;
    int num_inputs;
    std::shared_ptr<const HloModule> hlo_module;
  };

  std::unique_ptr<XdnaDevice> device_;
  std::unique_ptr<XdnaMemorySpace> memory_space_;
  std::vector<PjRtDevice*> devices_;
  std::vector<PjRtDevice*> addressable_devices_;
  std::vector<PjRtMemorySpace*> memory_spaces_;
  std::string platform_version_;
  xrt::device xrt_device_;

  // In-memory compilation cache: maps HLO fingerprint to compiled xclbin.
  // Disable with XDNA_NO_CACHE=1 environment variable.
  absl::flat_hash_map<uint64_t, CachedCompilation> compilation_cache_;
};

}  // namespace xla

#endif  // XLA_PJRT_PLUGIN_XDNA_XDNA_PJRT_CLIENT_H_
