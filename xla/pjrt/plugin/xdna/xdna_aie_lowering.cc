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

#include "xla/pjrt/plugin/xdna/xdna_aie_lowering.h"

#include <cstdint>
#include <cstdlib>
#include <string>
#include <vector>

#include "absl/log/log.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/str_format.h"
#include "absl/strings/str_join.h"
#include "llvm/Support/raw_ostream.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Operation.h"
#include "mlir/IR/Types.h"

namespace xla {
namespace {

// Returns the MLIR type string for a given element type.
std::string GetElementTypeStr(mlir::Type type) {
  if (type.isF32()) return "f32";
  if (type.isF16()) return "f16";
  if (type.isBF16()) return "bf16";
  if (type.isInteger(32)) return "i32";
  if (type.isInteger(16)) return "i16";
  if (type.isInteger(8)) return "i8";
  return "f32";  // default
}

// Returns the linalg kernel operation for a given linalg named op.
// Maps the linalg op to the computation that should run on the AIE core.
std::string GetCoreKernelOp(llvm::StringRef op_name) {
  if (op_name == "linalg.add") return "arith.addf";
  if (op_name == "linalg.sub") return "arith.subf";
  if (op_name == "linalg.mul") return "arith.mulf";
  if (op_name == "linalg.negf") return "arith.negf";
  if (op_name == "linalg.exp") return "math.exp";
  if (op_name == "linalg.log") return "math.log";
  return "";
}

// Returns the number of input operands for a given linalg op.
int GetNumInputs(llvm::StringRef op_name) {
  if (op_name == "linalg.add") return 2;
  if (op_name == "linalg.sub") return 2;
  if (op_name == "linalg.mul") return 2;
  if (op_name == "linalg.negf") return 1;
  if (op_name == "linalg.exp") return 1;
  if (op_name == "linalg.log") return 1;
  return -1;  // unsupported
}

// Describes a linalg operation extracted from the module.
struct LinalgOpInfo {
  std::string op_name;        // e.g., "linalg.add"
  int num_inputs;             // 1 or 2
  int64_t num_elements;       // total element count
  std::string element_type;   // e.g., "f32"
  std::string kernel_op;      // e.g., "arith.addf"
};

// Analyzes the linalg module to extract operation info.
// Returns an error if no supported op is found, or if multiple ops are found
// (multi-op programs are not yet supported).
absl::StatusOr<LinalgOpInfo> AnalyzeLinalgModule(mlir::ModuleOp module) {
  LinalgOpInfo info;

  // Find the entry function.
  mlir::func::FuncOp entry_func;
  module.walk([&](mlir::func::FuncOp func) {
    if (func.getName() == "main" || !entry_func) {
      entry_func = func;
    }
  });

  if (!entry_func) {
    return absl::InvalidArgumentError(
        "No function found in linalg module.");
  }

  // Count all supported linalg ops to detect multi-op programs.
  int op_count = 0;

  // Walk the function body to find linalg named ops.
  entry_func.walk([&](mlir::Operation* op) {
    llvm::StringRef name = op->getName().getStringRef();
    if (!name.starts_with("linalg.")) return;

    // Skip linalg.generic — handled separately below.
    if (name == "linalg.generic") return;

    int num_inputs = GetNumInputs(name);
    if (num_inputs < 0) return;

    std::string kernel_op = GetCoreKernelOp(name);
    if (kernel_op.empty()) return;

    // Get shape info from the result type.
    if (op->getNumResults() < 1) return;
    auto result_type = mlir::dyn_cast<mlir::RankedTensorType>(
        op->getResult(0).getType());
    if (!result_type) return;

    op_count++;
    info.op_name = name.str();
    info.num_inputs = num_inputs;
    info.element_type = GetElementTypeStr(result_type.getElementType());
    info.kernel_op = kernel_op;

    // Compute total elements.
    info.num_elements = 1;
    for (int64_t dim : result_type.getShape()) {
      info.num_elements *= dim;
    }
  });

  // Also walk linalg.generic ops.
  entry_func.walk([&](mlir::linalg::GenericOp generic) {
    // Check that all iterator types are parallel (element-wise).
    auto iter_types = generic.getIteratorTypesArray();
    for (auto it : iter_types) {
      if (it != mlir::utils::IteratorType::parallel) return;
    }

    // Look at the body to determine the operation.
    mlir::Block& body = generic.getRegion().front();
    LinalgOpInfo generic_info;
    bool found_in_body = false;
    for (mlir::Operation& body_op : body.without_terminator()) {
      llvm::StringRef body_op_name = body_op.getName().getStringRef();
      if (body_op_name == "arith.addf" || body_op_name == "arith.addi") {
        generic_info.kernel_op = body_op_name.str();
        generic_info.op_name = "linalg.generic(add)";
        generic_info.num_inputs = 2;
      } else if (body_op_name == "arith.subf" ||
                 body_op_name == "arith.subi") {
        generic_info.kernel_op = body_op_name.str();
        generic_info.op_name = "linalg.generic(sub)";
        generic_info.num_inputs = 2;
      } else if (body_op_name == "arith.mulf" ||
                 body_op_name == "arith.muli") {
        generic_info.kernel_op = body_op_name.str();
        generic_info.op_name = "linalg.generic(mul)";
        generic_info.num_inputs = 2;
      } else if (body_op_name == "arith.negf") {
        generic_info.kernel_op = body_op_name.str();
        generic_info.op_name = "linalg.generic(neg)";
        generic_info.num_inputs = 1;
      } else if (body_op_name == "math.exp") {
        generic_info.kernel_op = body_op_name.str();
        generic_info.op_name = "linalg.generic(exp)";
        generic_info.num_inputs = 1;
      } else {
        continue;
      }
      found_in_body = true;
      break;
    }

    if (!found_in_body) return;

    // Get shape from first input.
    if (generic.getInputs().empty()) return;
    auto input_type = mlir::dyn_cast<mlir::RankedTensorType>(
        generic.getInputs().front().getType());
    if (!input_type) return;

    generic_info.element_type = GetElementTypeStr(input_type.getElementType());
    generic_info.num_elements = 1;
    for (int64_t dim : input_type.getShape()) {
      generic_info.num_elements *= dim;
    }

    op_count++;
    info = generic_info;
  });

  if (op_count > 1) {
    return absl::UnimplementedError(absl::StrFormat(
        "XDNA plugin currently supports single-op programs only. Found %d "
        "linalg operations. Consider decomposing your computation.",
        op_count));
  }

  if (op_count == 0) {
    // Print the module for debugging.
    std::string module_str;
    llvm::raw_string_ostream os(module_str);
    module.print(os);
    return absl::UnimplementedError(absl::StrCat(
        "No supported linalg operation found in module. "
        "Currently supported: add, sub, mul, neg, exp, log. "
        "Module IR:\n", module_str));
  }

  LOG(INFO) << "XDNA AIE lowering: found " << info.op_name
            << " on " << info.num_elements << "x" << info.element_type;

  return info;
}

// Generates a single-core element-wise loop as AIE core body text.
std::string GenerateCoreBody(const LinalgOpInfo& info, int64_t chunk_size) {
  std::string body;
  const std::string& ty = info.element_type;
  std::string memref_ty = absl::StrFormat("memref<%dx%s>", chunk_size, ty);

  // Acquire input/output ObjectFIFO elements.
  absl::StrAppend(&body,
      "      %subview_in0 = aie.objectfifo.acquire @in0(Consume, 1) "
      ": !aie.objectfifosubview<", memref_ty, ">\n");
  absl::StrAppend(&body,
      "      %elem_in0 = aie.objectfifo.subview.access %subview_in0[0] "
      ": !aie.objectfifosubview<", memref_ty, "> -> ", memref_ty, "\n");

  if (info.num_inputs == 2) {
    absl::StrAppend(&body,
        "      %subview_in1 = aie.objectfifo.acquire @in1(Consume, 1) "
        ": !aie.objectfifosubview<", memref_ty, ">\n");
    absl::StrAppend(&body,
        "      %elem_in1 = aie.objectfifo.subview.access %subview_in1[0] "
        ": !aie.objectfifosubview<", memref_ty, "> -> ", memref_ty, "\n");
  }

  absl::StrAppend(&body,
      "      %subview_out = aie.objectfifo.acquire @out0(Produce, 1) "
      ": !aie.objectfifosubview<", memref_ty, ">\n");
  absl::StrAppend(&body,
      "      %elem_out = aie.objectfifo.subview.access %subview_out[0] "
      ": !aie.objectfifosubview<", memref_ty, "> -> ", memref_ty, "\n");

  // Generate the element-wise loop.
  absl::StrAppend(&body,
      "      %c0 = arith.constant 0 : index\n"
      "      %c1 = arith.constant 1 : index\n");
  absl::StrAppend(&body, absl::StrFormat(
      "      %%c%d = arith.constant %d : index\n", chunk_size, chunk_size));
  absl::StrAppend(&body, absl::StrFormat(
      "      scf.for %%i = %%c0 to %%c%d step %%c1 {\n", chunk_size));

  // Load, compute, store.
  //
  // For f32 multiply: AIE2p has native bf16 multiply (vmul.f) but no f32
  // multiply instruction. Peano's legalizer has custom scalar bf16 fmul
  // (insert→vmul→extract) but falls back to __mulsf3 libcall for f32.
  // We truncate f32→bf16 before multiply and extend bf16→f32 after.
  bool needs_bf16_cast =
      (info.kernel_op == "arith.mulf" && info.element_type == "f32");

  absl::StrAppend(&body,
      "        %val_in0 = memref.load %elem_in0[%i] : ", memref_ty, "\n");
  if (info.num_inputs == 2) {
    absl::StrAppend(&body,
        "        %val_in1 = memref.load %elem_in1[%i] : ", memref_ty, "\n");
    if (needs_bf16_cast) {
      absl::StrAppend(&body,
          "        %bf16_in0 = arith.truncf %val_in0 : f32 to bf16\n"
          "        %bf16_in1 = arith.truncf %val_in1 : f32 to bf16\n"
          "        %bf16_out = arith.mulf %bf16_in0, %bf16_in1 : bf16\n"
          "        %val_out = arith.extf %bf16_out : bf16 to f32\n");
    } else {
      absl::StrAppend(&body,
          "        %val_out = ", info.kernel_op,
          " %val_in0, %val_in1 : ", ty, "\n");
    }
  } else {
    absl::StrAppend(&body,
        "        %val_out = ", info.kernel_op,
        " %val_in0 : ", ty, "\n");
  }
  absl::StrAppend(&body,
      "        memref.store %val_out, %elem_out[%i] : ", memref_ty, "\n");
  absl::StrAppend(&body,
      "      }\n");

  // Release ObjectFIFO elements.
  absl::StrAppend(&body,
      "      aie.objectfifo.release @in0(Consume, 1)\n");
  if (info.num_inputs == 2) {
    absl::StrAppend(&body,
        "      aie.objectfifo.release @in1(Consume, 1)\n");
  }
  absl::StrAppend(&body,
      "      aie.objectfifo.release @out0(Produce, 1)\n");

  return body;
}

// Generates the NPU instruction sequence for DMA setup.
// Follows the pattern from mlir-aie examples (e.g., add_314_using_dma_op):
//   1. Set up input DMA transfers
//   2. Set up output DMA with issue_token=true (enables completion tracking)
//   3. dma_wait on the output ObjectFIFO
std::string GenerateNpuSequence(const LinalgOpInfo& info,
                                int64_t chunk_size) {
  const std::string& ty = info.element_type;
  std::string memref_ty = absl::StrFormat("memref<%dx%s>", chunk_size, ty);

  std::string seq;

  // Runtime sequence: inputs + output.
  // Uses aiex.runtime_sequence (not func.func @sequence) so that
  // aie-dma-to-npu can lower the DMA ops to NPU instructions.
  if (info.num_inputs == 2) {
    absl::StrAppend(&seq, absl::StrFormat(
        "    aiex.runtime_sequence(%%arg0: %s, %%arg1: %s, %%arg2: %s) {\n",
        memref_ty, memref_ty, memref_ty));
  } else {
    absl::StrAppend(&seq, absl::StrFormat(
        "    aiex.runtime_sequence(%%arg0: %s, %%arg1: %s) {\n",
        memref_ty, memref_ty));
  }

  // DMA transfers: inputs first.
  absl::StrAppend(&seq, absl::StrFormat(
      "      aiex.npu.dma_memcpy_nd(0, 0, %%arg0[0, 0, 0, 0]"
      "[1, 1, 1, %d][0, 0, 0, 1]) "
      "{id = 0 : i64, metadata = @in0} : %s\n",
      chunk_size, memref_ty));

  if (info.num_inputs == 2) {
    absl::StrAppend(&seq, absl::StrFormat(
        "      aiex.npu.dma_memcpy_nd(0, 0, %%arg1[0, 0, 0, 0]"
        "[1, 1, 1, %d][0, 0, 0, 1]) "
        "{id = 1 : i64, metadata = @in1} : %s\n",
        chunk_size, memref_ty));
  }

  // DMA transfer: output with issue_token for completion tracking.
  int out_arg = info.num_inputs;
  int out_id = info.num_inputs;
  absl::StrAppend(&seq, absl::StrFormat(
      "      aiex.npu.dma_memcpy_nd(0, 0, %%arg%d[0, 0, 0, 0]"
      "[1, 1, 1, %d][0, 0, 0, 1]) "
      "{id = %d : i64, metadata = @out0, issue_token = true} : %s\n",
      out_arg, chunk_size, out_id, memref_ty));

  // Wait for output DMA completion (lowered to npu.sync by aie-dma-to-npu).
  absl::StrAppend(&seq,
      "      aiex.npu.dma_wait { symbol = @out0 }\n");

  absl::StrAppend(&seq, "    }\n");

  return seq;
}

}  // namespace

// Returns the byte size of a single element for the given type string.
int GetElementBytes(const std::string& element_type) {
  if (element_type == "f32" || element_type == "i32") return 4;
  if (element_type == "f16" || element_type == "bf16" || element_type == "i16")
    return 2;
  if (element_type == "i8") return 1;
  return 4;  // default
}

absl::StatusOr<std::string> LowerLinalgToAie(
    mlir::ModuleOp linalg_module, const AieLoweringConfig& config) {
  // Step 1: Analyze the linalg module.
  auto info_result = AnalyzeLinalgModule(linalg_module);
  if (!info_result.ok()) return info_result.status();
  LinalgOpInfo info = std::move(*info_result);

  // Step 1b: Validate L1 memory usage.
  // Each ObjectFIFO buffer uses ping-pong depth of 2, so total L1 is:
  //   num_buffers * num_elements * element_bytes * 2 (ping-pong)
  int element_bytes = GetElementBytes(info.element_type);
  int num_buffers = info.num_inputs + 1;  // inputs + output
  int64_t total_l1_bytes = num_buffers * info.num_elements * element_bytes * 2;

  int64_t l1_limit = 48 * 1024;  // 48KB default (leaves headroom for stack/code)
  const char* limit_env = std::getenv("XDNA_L1_LIMIT_BYTES");
  if (limit_env) l1_limit = std::atol(limit_env);

  if (total_l1_bytes > l1_limit) {
    return absl::InvalidArgumentError(absl::StrFormat(
        "Tensor buffers (%d bytes) exceed AIE L1 memory limit (%d bytes). "
        "The operation requires %d buffers of %d elements x %d bytes x 2 "
        "(ping-pong). Consider using smaller tensors (max ~%d elements for "
        "%s). Override limit with XDNA_L1_LIMIT_BYTES env var.",
        total_l1_bytes, l1_limit, num_buffers, info.num_elements,
        element_bytes, l1_limit / (num_buffers * element_bytes * 2),
        info.element_type));
  }

  // For single-core execution, the chunk size is the full tensor.
  int64_t chunk_size = info.num_elements;

  // Step 2: Generate AIE dialect MLIR text.
  //
  // Architecture: npu2 (XDNA 2 / Strix Halo / AIE2PS)
  // Layout: single column (col 0)
  //   - Shim tile (0, 0): host DMA interface
  //   - Mem tile (0, 1): L2 buffer (unused for single-core)
  //   - Compute tile (0, 2): runs the kernel

  std::string aie_mlir;
  const std::string& ty = info.element_type;
  std::string memref_ty = absl::StrFormat("memref<%dx%s>", chunk_size, ty);

  absl::StrAppend(&aie_mlir, "module {\n");
  absl::StrAppend(&aie_mlir, "  aie.device(npu2) {\n");

  // Tiles.
  absl::StrAppend(&aie_mlir,
      "    %tile_0_0 = aie.tile(0, 0)\n"
      "    %tile_0_2 = aie.tile(0, 2)\n");

  // ObjectFIFOs for data movement: shim → compute.
  absl::StrAppend(&aie_mlir, absl::StrFormat(
      "    aie.objectfifo @in0(%%tile_0_0, {%%tile_0_2}, 2 : i32) "
      ": !aie.objectfifo<%s>\n", memref_ty));
  if (info.num_inputs == 2) {
    absl::StrAppend(&aie_mlir, absl::StrFormat(
        "    aie.objectfifo @in1(%%tile_0_0, {%%tile_0_2}, 2 : i32) "
        ": !aie.objectfifo<%s>\n", memref_ty));
  }
  absl::StrAppend(&aie_mlir, absl::StrFormat(
      "    aie.objectfifo @out0(%%tile_0_2, {%%tile_0_0}, 2 : i32) "
      ": !aie.objectfifo<%s>\n", memref_ty));

  // Core body.
  absl::StrAppend(&aie_mlir,
      "    %core_0_2 = aie.core(%tile_0_2) {\n");
  absl::StrAppend(&aie_mlir, GenerateCoreBody(info, chunk_size));
  absl::StrAppend(&aie_mlir,
      "      aie.end\n"
      "    }\n");

  // NPU instruction sequence.
  absl::StrAppend(&aie_mlir, GenerateNpuSequence(info, chunk_size));

  absl::StrAppend(&aie_mlir, "  }\n");
  absl::StrAppend(&aie_mlir, "}\n");

  LOG(INFO) << "XDNA AIE lowering: generated AIE MLIR:\n" << aie_mlir;

  return aie_mlir;
}

}  // namespace xla
