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
#include <memory>
#include <vector>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "xla/literal_util.h"
#include "xla/pjrt/pjrt_client.h"
#include "xla/pjrt/plugin/xdna/xdna_c_pjrt.h"
#include "xla/pjrt/c/pjrt_c_api.h"

namespace xla {
namespace {

TEST(XdnaPjrtClientTest, PlatformName) {
  auto client = CreateXdnaPjrtClient();
  EXPECT_EQ(client->platform_name(), "xdna");
}

TEST(XdnaPjrtClientTest, PlatformVersion) {
  auto client = CreateXdnaPjrtClient();
  EXPECT_FALSE(client->platform_version().empty());
}

TEST(XdnaPjrtClientTest, DeviceCount) {
  auto client = CreateXdnaPjrtClient();
  EXPECT_EQ(client->device_count(), 1);
}

TEST(XdnaPjrtClientTest, AddressableDeviceCount) {
  auto client = CreateXdnaPjrtClient();
  EXPECT_EQ(client->addressable_device_count(), 1);
}

TEST(XdnaPjrtClientTest, DevicesNotEmpty) {
  auto client = CreateXdnaPjrtClient();
  EXPECT_FALSE(client->devices().empty());
  EXPECT_EQ(client->devices().size(), 1);
}

TEST(XdnaPjrtClientTest, DeviceIsAddressable) {
  auto client = CreateXdnaPjrtClient();
  EXPECT_TRUE(client->devices()[0]->IsAddressable());
}

TEST(XdnaPjrtClientTest, DeviceKind) {
  auto client = CreateXdnaPjrtClient();
  EXPECT_EQ(client->devices()[0]->device_kind(), "XDNA");
}

TEST(XdnaPjrtClientTest, MemorySpacesNotEmpty) {
  auto client = CreateXdnaPjrtClient();
  EXPECT_FALSE(client->memory_spaces().empty());
  EXPECT_EQ(client->memory_spaces().size(), 1);
}

TEST(XdnaPjrtClientTest, LookupDevice) {
  auto client = CreateXdnaPjrtClient();
  auto device = client->LookupDevice(PjRtGlobalDeviceId(0));
  ASSERT_TRUE(device.ok());
  EXPECT_EQ((*device)->global_device_id(), PjRtGlobalDeviceId(0));
}

TEST(XdnaPjrtClientTest, LookupDeviceNotFound) {
  auto client = CreateXdnaPjrtClient();
  auto device = client->LookupDevice(PjRtGlobalDeviceId(999));
  EXPECT_FALSE(device.ok());
}

TEST(XdnaPjrtClientTest, BufferRoundTrip) {
  auto client = CreateXdnaPjrtClient();
  // Create a buffer with known data.
  std::vector<float> host_data = {1.0f, 2.0f, 3.0f, 4.0f};
  auto buffer_or = client->BufferFromHostBuffer(
      host_data.data(), PrimitiveType::F32, {4},
      /*byte_strides=*/std::nullopt,
      PjRtClient::HostBufferSemantics::kImmutableOnlyDuringCall,
      /*on_done_with_host_buffer=*/nullptr,
      client->memory_spaces()[0],
      /*device_layout=*/nullptr);
  ASSERT_TRUE(buffer_or.ok());
  auto buffer = std::move(*buffer_or);

  // Read the data back.
  auto literal = LiteralUtil::CreateR1<float>({0.0f, 0.0f, 0.0f, 0.0f});
  auto status = buffer->ToLiteral(&literal);
  ASSERT_TRUE(status.Await().ok());

  EXPECT_EQ(literal.Get<float>({0}), 1.0f);
  EXPECT_EQ(literal.Get<float>({1}), 2.0f);
  EXPECT_EQ(literal.Get<float>({2}), 3.0f);
  EXPECT_EQ(literal.Get<float>({3}), 4.0f);
}

TEST(XdnaPjrtClientTest, BufferDelete) {
  auto client = CreateXdnaPjrtClient();
  std::vector<float> host_data = {1.0f};
  auto buffer_or = client->BufferFromHostBuffer(
      host_data.data(), PrimitiveType::F32, {1},
      /*byte_strides=*/std::nullopt,
      PjRtClient::HostBufferSemantics::kImmutableOnlyDuringCall,
      /*on_done_with_host_buffer=*/nullptr,
      client->memory_spaces()[0],
      /*device_layout=*/nullptr);
  ASSERT_TRUE(buffer_or.ok());
  auto buffer = std::move(*buffer_or);
  EXPECT_FALSE(buffer->IsDeleted());
  buffer->Delete();
  EXPECT_TRUE(buffer->IsDeleted());
}

// C API tests.
TEST(XdnaCApiTest, CreatesPjRtApi) {
  const PJRT_Api* api = GetPjrtApi();
  EXPECT_THAT(api, ::testing::NotNull());
}

TEST(XdnaCApiTest, CanCreateClient) {
  const PJRT_Api* api = GetPjrtApi();
  ASSERT_THAT(api, ::testing::NotNull());
  PJRT_Client_Create_Args args;
  args.struct_size = PJRT_Client_Create_Args_STRUCT_SIZE;
  args.extension_start = nullptr;
  args.create_options = nullptr;
  args.num_options = 0;
  args.kv_get_callback = nullptr;
  args.kv_put_callback = nullptr;
  args.kv_get_user_arg = nullptr;
  args.kv_put_user_arg = nullptr;
  PJRT_Error* error = api->PJRT_Client_Create(&args);
  EXPECT_THAT(error, ::testing::IsNull());
  if (args.client != nullptr) {
    PJRT_Client_Destroy_Args destroy_args;
    destroy_args.struct_size = PJRT_Client_Destroy_Args_STRUCT_SIZE;
    destroy_args.extension_start = nullptr;
    destroy_args.client = args.client;
    api->PJRT_Client_Destroy(&destroy_args);
  }
}

}  // namespace
}  // namespace xla
