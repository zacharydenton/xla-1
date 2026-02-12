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

#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <memory>
#include <string>
#include <unistd.h>
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
#include "mlir/Transforms/Passes.h"
#include "stablehlo/conversions/linalg/transforms/Passes.h"
#include "xla/hlo/ir/hlo_module.h"
#include "xla/hlo/translate/hlo_to_mhlo/hlo_to_mlir_hlo.h"
#include "xla/shape.h"
#include "xla/pjrt/mlir_to_hlo.h"
#include "xla/pjrt/plugin/xdna/xdna_aie_lowering.h"
#include "xla/pjrt/plugin/xdna/xdna_codegen.h"
#include "xla/pjrt/plugin/xdna/xdna_target_caps.h"
#include "xla/tsl/platform/errors.h"

namespace xla {

absl::StatusOr<XdnaCodegenResult> XdnaCompiler::Compile(
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
  // Inline private helper functions so that constants (e.g., relu zero)
  // are directly visible to AnalyzeLinalgGeneric.
  pm.addPass(mlir::createInlinerPass());
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
  TargetCaps caps = StrixPointTargetCaps();
  AieLoweringConfig config;
  config.num_columns = caps.num_columns;  // Auto-scale to hardware (4 on Strix Point).
  const char* num_cores_env = std::getenv("XDNA_NUM_CORES");
  if (num_cores_env)
    config.num_columns = std::max(1, std::min(std::atoi(num_cores_env),
                                               caps.num_columns));
  LOG(INFO) << "XDNA compiler: target=" << caps.device_name
            << " isa=" << caps.isa_target
            << " l1_usable=" << caps.l1_usable_bytes << "B"
            << " columns=" << config.num_columns;
  TF_ASSIGN_OR_RETURN(AieLoweringResult lowering,
                      LowerLinalgToAie(*mlir_module, config, caps));

  LOG(INFO) << "XDNA compiler: linalg → AIE lowering succeeded ("
            << lowering.num_cores << " core(s)).";
  // Write to stderr directly so subprocess tests can observe core count.
  {
    char buf[80];
    int n = snprintf(buf, sizeof(buf), "XDNA: using %d core(s)\n",
                     lowering.num_cores);
    (void)write(STDERR_FILENO, buf, n);
  }

  // Compute number of data buffer args: inputs + outputs.
  int num_inputs = hlo_module->entry_computation()->num_parameters();
  const Shape& result_shape = hlo_module->result_shape();
  int num_outputs = result_shape.IsTuple()
                        ? result_shape.tuple_shapes_size()
                        : 1;
  int num_data_args = num_inputs + num_outputs;

  // Step 4: AIE → xclbin codegen (aie-opt + Peano + bootgen + xclbinutil).
  TF_ASSIGN_OR_RETURN(XdnaCodegenResult result,
                      GenerateXclbinFromAie(lowering.aie_mlir, num_data_args,
                                            caps, lowering.num_cores,
                                            lowering.use_aievec,
                                            lowering.needs_softfloat_stubs));

  LOG(INFO) << "XDNA compiler: xclbin generated, "
            << result.xclbin_bytes.size() << " bytes.";

  return result;
}

}  // namespace xla
