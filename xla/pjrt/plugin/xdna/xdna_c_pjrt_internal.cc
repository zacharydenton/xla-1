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

#include "xla/pjrt/plugin/xdna/xdna_c_pjrt_internal.h"

#include <memory>
#include <utility>

#include "absl/status/status.h"
#include "xla/pjrt/c/pjrt_c_api.h"
#include "xla/pjrt/c/pjrt_c_api_layouts_extension.h"
#include "xla/pjrt/c/pjrt_c_api_wrapper_impl.h"
#include "xla/pjrt/pjrt_client.h"
#include "xla/pjrt/plugin/xdna/xdna_pjrt_client.h"

namespace xdna_pjrt {

PJRT_Error* PJRT_XdnaClient_Create(PJRT_Client_Create_Args* args) {
  absl::StatusOr<std::unique_ptr<xla::PjRtClient>> client_or =
      xla::CreateXdnaPjrtClient();
  if (!client_or.ok()) {
    return new PJRT_Error{client_or.status()};
  }
  args->client = pjrt::CreateWrapperClient(std::move(*client_or));
  return nullptr;
}

PJRT_Error* PJRT_XdnaExecuteContext_Create(
    PJRT_ExecuteContext_Create_Args* args) {
  return new PJRT_Error{absl::UnimplementedError(
      "ExecuteContext not supported for XDNA.")};
}

PJRT_Error* PJRT_XdnaDeviceTopology_Create(
    PJRT_TopologyDescription_Create_Args* args) {
  return new PJRT_Error{absl::UnimplementedError(
      "Topology not supported for XDNA.")};
}

const PJRT_Api* GetXdnaPjrtApi() {
  static PJRT_Layouts_Extension layouts_extension =
      pjrt::CreateLayoutsExtension(nullptr);

  static const PJRT_Api pjrt_api = pjrt::CreatePjrtApi(
      xdna_pjrt::PJRT_XdnaClient_Create,
      xdna_pjrt::PJRT_XdnaExecuteContext_Create,
      xdna_pjrt::PJRT_XdnaDeviceTopology_Create,
      pjrt::PJRT_Plugin_Initialize_NoOp,
      &layouts_extension.base,
      pjrt::PJRT_Plugin_Attributes_Xla);

  return &pjrt_api;
}

}  // namespace xdna_pjrt
