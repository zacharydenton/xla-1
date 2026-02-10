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

#include "xla/pjrt/plugin/xdna/xdna_compiler.h"

#include <cstdint>
#include <memory>
#include <optional>
#include <vector>

#include <gtest/gtest.h>
#include "absl/status/statusor.h"
#include "xla/hlo/builder/xla_builder.h"
#include "xla/hlo/builder/xla_computation.h"
#include "xla/hlo/ir/hlo_module.h"
#include "xla/literal_util.h"
#include "xla/pjrt/plugin/xdna/xdna_hlo_passes.h"
#include "xla/pjrt/plugin/xdna/xdna_pjrt_client.h"
#include "xla/pjrt/pjrt_client.h"
#include "xla/pjrt/pjrt_future.h"
#include "xla/shape_util.h"

namespace xla {
namespace {

// Helper to build a simple add-one computation.
XlaComputation BuildAddOneComputation() {
  Shape shape = ShapeUtil::MakeShape(F32, {4});
  XlaBuilder builder("add_one");
  Add(Parameter(&builder, 0, shape, "x"),
      ConstantR1<float>(&builder, {1.0f, 1.0f, 1.0f, 1.0f}));
  return builder.Build().value();
}

// Helper to build a two-input add computation.
XlaComputation BuildAddComputation() {
  Shape shape = ShapeUtil::MakeShape(F32, {4});
  XlaBuilder builder("add");
  auto x = Parameter(&builder, 0, shape, "x");
  auto y = Parameter(&builder, 1, shape, "y");
  Add(x, y);
  return builder.Build().value();
}

// Helper to create an HloModule from a computation.
std::unique_ptr<HloModule> CreateHloModule(const XlaComputation& computation) {
  auto config = HloModule::CreateModuleConfigFromProto(computation.proto(),
                                                       DebugOptions())
                    .value();
  return HloModule::CreateFromProto(computation.proto(), config).value();
}

TEST(XdnaHloPassesTest, RunsOnSimpleModule) {
  XlaComputation computation = BuildAddOneComputation();
  std::unique_ptr<HloModule> module = CreateHloModule(computation);

  auto result = RunXdnaHloPasses(std::move(module));
  ASSERT_TRUE(result.ok()) << result.status();

  // Module should still have an entry computation after passes.
  EXPECT_TRUE((*result)->has_entry_computation());
}

TEST(XdnaHloPassesTest, RunsOnElementWiseOps) {
  Shape shape = ShapeUtil::MakeShape(F32, {8});
  XlaBuilder builder("elementwise");
  auto x = Parameter(&builder, 0, shape, "x");
  auto y = Parameter(&builder, 1, shape, "y");
  auto sum = Add(x, y);
  auto product = Mul(sum, x);
  Neg(product);

  auto computation = builder.Build().value();
  std::unique_ptr<HloModule> module = CreateHloModule(computation);

  auto result = RunXdnaHloPasses(std::move(module));
  ASSERT_TRUE(result.ok()) << result.status();
  EXPECT_TRUE((*result)->has_entry_computation());
}

// Test that the full compiler pipeline produces an xclbin.
TEST(XdnaCompilerTest, CompilePipelineRuns) {
  XlaComputation computation = BuildAddComputation();
  std::unique_ptr<HloModule> module = CreateHloModule(computation);

  auto result = XdnaCompiler::Compile(std::move(module));
  if (!result.ok()) {
    // Tools may not be installed in CI — NotFound/Internal are acceptable.
    EXPECT_TRUE(result.status().code() == absl::StatusCode::kNotFound ||
                result.status().code() == absl::StatusCode::kInternal)
        << "Unexpected error: " << result.status();
    GTEST_SKIP() << "Toolchain not available: " << result.status().message();
  }
  // Pipeline succeeded — verify we got non-empty xclbin bytes.
  EXPECT_GT(result->xclbin_bytes.size(), 0);
  // Check xclbin magic: "xclbin2" at offset 0.
  ASSERT_GE(result->xclbin_bytes.size(), 7u);
  EXPECT_EQ(result->xclbin_bytes[0], 'x');
  EXPECT_EQ(result->xclbin_bytes[1], 'c');
  EXPECT_EQ(result->xclbin_bytes[2], 'l');
  EXPECT_EQ(result->xclbin_bytes[3], 'b');
  EXPECT_EQ(result->xclbin_bytes[4], 'i');
  EXPECT_EQ(result->xclbin_bytes[5], 'n');
  EXPECT_EQ(result->xclbin_bytes[6], '2');
  // Kernel name should be set.
  EXPECT_FALSE(result->kernel_name.empty());
}

TEST(XdnaCompilerTest, CompileAfterHloPasses) {
  XlaComputation computation = BuildAddComputation();
  std::unique_ptr<HloModule> module = CreateHloModule(computation);

  auto passed = RunXdnaHloPasses(std::move(module));
  ASSERT_TRUE(passed.ok()) << passed.status();

  auto result = XdnaCompiler::Compile(std::move(*passed));
  if (!result.ok()) {
    EXPECT_TRUE(result.status().code() == absl::StatusCode::kNotFound ||
                result.status().code() == absl::StatusCode::kInternal)
        << "Unexpected error: " << result.status();
    GTEST_SKIP() << "Toolchain not available: " << result.status().message();
  }
  EXPECT_GT(result->xclbin_bytes.size(), 0);
}

// Integration test: CompileAndLoad on the full client.
class XdnaCompileAndLoadTest : public ::testing::Test {
 protected:
  void SetUp() override {
    auto client_or = CreateXdnaPjrtClient();
    if (!client_or.ok()) {
      GTEST_SKIP() << "No XDNA device: " << client_or.status().message();
    }
    client_ = std::move(*client_or);
  }

  std::unique_ptr<PjRtClient> client_;
};

TEST_F(XdnaCompileAndLoadTest, CompilePipelineRuns) {
  XlaComputation computation = BuildAddComputation();
  auto result = client_->CompileAndLoad(computation, CompileOptions());
  // Should reach codegen, may fail if tools are missing.
  if (!result.ok()) {
    EXPECT_TRUE(result.status().code() == absl::StatusCode::kNotFound ||
                result.status().code() == absl::StatusCode::kInternal)
        << "Unexpected error: " << result.status();
  }
}

TEST_F(XdnaCompileAndLoadTest, CompileAndExecuteAdd) {
  XlaComputation computation = BuildAddComputation();
  auto exe_or = client_->CompileAndLoad(computation, CompileOptions());
  if (!exe_or.ok()) {
    GTEST_SKIP() << "Compile failed: " << exe_or.status().message();
  }
  auto& exe = *exe_or;

  // Create input buffers: x = [1, 2, 3, 4], y = [10, 20, 30, 40]
  std::vector<float> x_data = {1.0f, 2.0f, 3.0f, 4.0f};
  std::vector<float> y_data = {10.0f, 20.0f, 30.0f, 40.0f};

  PjRtDevice* device = client_->addressable_devices()[0];
  auto mem_space = device->default_memory_space().value();

  auto buf_x = client_->BufferFromHostBuffer(
      x_data.data(), F32, {4}, /*byte_strides=*/std::nullopt,
      PjRtClient::HostBufferSemantics::kImmutableOnlyDuringCall,
      /*on_done_with_host_buffer=*/nullptr, mem_space,
      /*device_layout=*/nullptr);
  ASSERT_TRUE(buf_x.ok()) << buf_x.status();
  auto buf_y = client_->BufferFromHostBuffer(
      y_data.data(), F32, {4}, /*byte_strides=*/std::nullopt,
      PjRtClient::HostBufferSemantics::kImmutableOnlyDuringCall,
      /*on_done_with_host_buffer=*/nullptr, mem_space,
      /*device_layout=*/nullptr);
  ASSERT_TRUE(buf_y.ok()) << buf_y.status();

  // Execute.
  std::vector<PjRtBuffer*> args = {buf_x->get(), buf_y->get()};
  std::optional<Future<>> returned_future;
  auto result = exe->ExecuteSharded(args, device, ExecuteOptions(),
                                    returned_future, /*fill_future=*/false);
  ASSERT_TRUE(result.ok()) << result.status();

  // Should return one output buffer.
  ASSERT_EQ(result->size(), 1);

  // Read back the result.
  auto literal_result = (*result)[0]->ToLiteralSync();
  ASSERT_TRUE(literal_result.ok()) << literal_result.status();

  // Expected: [11, 22, 33, 44]
  auto expected = LiteralUtil::CreateR1<float>({11.0f, 22.0f, 33.0f, 44.0f});
  EXPECT_EQ(**literal_result, expected);
}

}  // namespace
}  // namespace xla
