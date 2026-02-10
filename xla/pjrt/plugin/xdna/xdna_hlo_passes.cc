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

#include "xla/pjrt/plugin/xdna/xdna_hlo_passes.h"

#include <memory>

#include "absl/status/statusor.h"
#include "xla/hlo/ir/hlo_module.h"
#include "xla/hlo/pass/hlo_pass_pipeline.h"
#include "xla/hlo/transforms/expanders/cholesky_expander.h"
#include "xla/hlo/transforms/expanders/convolution_type_canonicalizer.h"
#include "xla/hlo/transforms/expanders/dynamic_index_splitter.h"
#include "xla/hlo/transforms/expanders/eigh_expander.h"
#include "xla/hlo/transforms/expanders/qr_expander.h"
#include "xla/service/batchnorm_expander.h"
#include "xla/service/layout_assignment.h"
#include "xla/service/topk_rewriter.h"
#include "xla/service/triangular_solve_expander.h"
#include "xla/tsl/platform/errors.h"

namespace xla {

absl::StatusOr<std::unique_ptr<HloModule>> RunXdnaHloPasses(
    std::unique_ptr<HloModule> hlo_module) {
  HloPassPipeline pipeline("XDNA");

  // TopkDecomposer generates compare ops with type=TOTALORDER and must run
  // before any pass that rewrites such comparisons.
  pipeline.AddPass<TopkDecomposer>();
  pipeline.AddPass<DynamicIndexSplitter>();

  // Expand complex ops into simpler primitives.
  pipeline.AddPass<CholeskyExpander>();
  pipeline.AddPass<QrExpander>();
  pipeline.AddPass<EighExpander>();
  pipeline.AddPass<TriangularSolveExpander>();
  pipeline.AddPass<BatchNormExpander>(
      /*rewrite_training_op=*/true,
      /*rewrite_inference_op=*/true,
      /*rewrite_grad_op=*/true);

  // Assign default layouts.
  pipeline.AddPass<LayoutAssignment>(
      hlo_module->mutable_entry_computation_layout());
  pipeline.AddPass<ConvolutionTypeCanonicalizer>();

  TF_RETURN_IF_ERROR(pipeline.Run(hlo_module.get()).status());
  return hlo_module;
}

}  // namespace xla
