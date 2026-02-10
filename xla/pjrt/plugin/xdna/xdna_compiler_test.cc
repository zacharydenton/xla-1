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

#include <gtest/gtest.h>
#include "absl/status/statusor.h"
#include "xla/hlo/builder/xla_builder.h"
#include "xla/hlo/builder/xla_computation.h"
#include "xla/hlo/ir/hlo_module.h"
#include "xla/literal_util.h"
#include "xla/pjrt/plugin/xdna/xdna_hlo_passes.h"
#include "xla/pjrt/plugin/xdna/xdna_pjrt_client.h"
#include "xla/pjrt/pjrt_client.h"
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

TEST(XdnaCompilerTest, CompileReturnsUnimplemented) {
  XlaComputation computation = BuildAddOneComputation();
  std::unique_ptr<HloModule> module = CreateHloModule(computation);

  auto result = XdnaCompiler::Compile(std::move(module));
  // Currently returns Unimplemented because AIE lowering + codegen
  // require the Peano/mlir-aie toolchain.
  EXPECT_FALSE(result.ok());
  EXPECT_EQ(result.status().code(), absl::StatusCode::kUnimplemented);
}

TEST(XdnaCompilerTest, CompileAfterHloPasses) {
  XlaComputation computation = BuildAddOneComputation();
  std::unique_ptr<HloModule> module = CreateHloModule(computation);

  // Passes should succeed.
  auto passed = RunXdnaHloPasses(std::move(module));
  ASSERT_TRUE(passed.ok()) << passed.status();

  // Compile should fail with Unimplemented.
  auto result = XdnaCompiler::Compile(std::move(*passed));
  EXPECT_FALSE(result.ok());
  EXPECT_EQ(result.status().code(), absl::StatusCode::kUnimplemented);
}

// Integration test: CompileAndLoad on the full client should propagate the
// Unimplemented error from the compiler.
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

TEST_F(XdnaCompileAndLoadTest, ReturnsUnimplemented) {
  XlaComputation computation = BuildAddOneComputation();
  auto result = client_->CompileAndLoad(computation, CompileOptions());
  EXPECT_FALSE(result.ok());
  // The error should indicate that compilation is not yet fully available.
  EXPECT_EQ(result.status().code(), absl::StatusCode::kUnimplemented);
}

}  // namespace
}  // namespace xla
