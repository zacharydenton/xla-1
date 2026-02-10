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
#include "llvm/Support/raw_ostream.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/OwningOpRef.h"
#include "mlir/Pass/PassManager.h"
#include "stablehlo/conversions/linalg/transforms/Passes.h"
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

  // Step 1: Convert HLO to StableHLO dialect.
  mlir::DialectRegistry registry;
  RegisterAllHloDialects(registry);
  mlir::MLIRContext context(registry);
  context.loadAllAvailableDialects();

  TF_ASSIGN_OR_RETURN(
      mlir::OwningOpRef<mlir::ModuleOp> mlir_module,
      ConvertHloToMlirHlo(context, hlo_module.get(),
                          /*import_all_computations=*/false,
                          /*flatten_computation_args_result=*/false,
                          /*emit_stablehlo=*/true));

  LOG(INFO) << "XDNA compiler: HLO → StableHLO conversion succeeded.";

  // Step 2: StableHLO → linalg-on-tensors.
  mlir::PassManager pm(&context);
  pm.addPass(mlir::stablehlo::createStablehloLegalizeToLinalgPass());
  if (mlir::failed(pm.run(*mlir_module))) {
    return absl::InternalError(
        "XDNA compiler: StableHLO → linalg conversion failed.");
  }

  LOG(INFO) << "XDNA compiler: StableHLO → linalg conversion succeeded.";

  // Debug: print the linalg module.
  std::string linalg_ir;
  llvm::raw_string_ostream os(linalg_ir);
  mlir_module->print(os);
  LOG(INFO) << "XDNA compiler: linalg IR:\n" << linalg_ir;

  // Step 3: linalg → AIE dialect MLIR (template-based generation).
  AieLoweringConfig config;
  TF_ASSIGN_OR_RETURN(std::string aie_mlir,
                      LowerLinalgToAie(*mlir_module, config));

  LOG(INFO) << "XDNA compiler: linalg → AIE lowering succeeded.";

  // Step 4: AIE → ELF codegen (subprocess: aie-opt + Peano + aiebu).
  TF_ASSIGN_OR_RETURN(std::vector<uint8_t> elf_bytes,
                      GenerateElfFromAie(aie_mlir));

  LOG(INFO) << "XDNA compiler: AIE → ELF codegen succeeded. ELF size: "
            << elf_bytes.size() << " bytes.";

  return elf_bytes;
}

}  // namespace xla
