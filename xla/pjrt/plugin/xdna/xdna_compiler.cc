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
#include <string>
#include <vector>

#include "absl/log/log.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_cat.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/OwningOpRef.h"
#include "xla/hlo/ir/hlo_module.h"
#include "xla/hlo/translate/hlo_to_mhlo/hlo_to_mlir_hlo.h"
#include "xla/pjrt/mlir_to_hlo.h"
#include "xla/pjrt/plugin/xdna/xdna_aie_lowering.h"
#include "xla/pjrt/plugin/xdna/xdna_codegen.h"
#include "xla/tsl/platform/errors.h"

namespace xla {

absl::StatusOr<std::vector<uint8_t>> XdnaCompiler::Compile(
    std::unique_ptr<HloModule> hlo_module) {
  LOG(INFO) << "XDNA compiler: compiling HLO module '"
            << hlo_module->name() << "'";

  // Step 1: Convert HLO to MHLO dialect.
  mlir::DialectRegistry registry;
  RegisterAllHloDialects(registry);
  mlir::MLIRContext context(registry);
  context.loadAllAvailableDialects();

  TF_ASSIGN_OR_RETURN(
      mlir::OwningOpRef<mlir::ModuleOp> mlir_module,
      ConvertHloToMlirHlo(context, hlo_module.get()));

  LOG(INFO) << "XDNA compiler: HLO → MHLO conversion succeeded.";

  // Step 2: MHLO → linalg lowering.
  // TODO: Apply mhlo-to-linalg pass pipeline here.
  // This requires running:
  //   mlir::PassManager pm(&context);
  //   pm.addPass(mlir::mhlo::createHloLegalizeToLinalgPass());
  //   pm.run(*mlir_module);

  // Step 3: linalg → AIE dialect lowering (requires mlir-aie).
  // Step 4: AIE → ELF codegen (requires Peano + mlir-aie).
  //
  // These steps are not yet implemented. The toolchain (Peano + mlir-aie)
  // must be built and linked before these can work.

  return absl::UnimplementedError(absl::StrCat(
      "XDNA compiler: full compilation pipeline not yet available for "
      "module '",
      hlo_module->name(),
      "'. The HLO → MHLO conversion succeeded, but linalg lowering and "
      "AIE codegen require Peano and mlir-aie toolchain linkage."));
}

}  // namespace xla
