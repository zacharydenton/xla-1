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
#include <optional>
#include <string>
#include <vector>

#include "absl/log/log.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/str_format.h"
#include "absl/strings/str_join.h"
#include "llvm/ADT/DenseMap.h"
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

// Returns the vector width for an (element_type, kernel_op) pair, or 0 if
// vectorization is not possible. Peano (Jan 2025) GlobalISel legalizes:
//   - fmul on <32 x bfloat> and <32 x half>  (native 16-bit float multiply)
//   - integer add/sub on <64 x i8>, <32 x i16>, <16 x i32>
//   - integer xor on <32 x i16>, <16 x i32> (used for float negate)
// It CANNOT legalize: fadd/fsub/fneg on any float vector, any integer mul
// on vectors, fptrunc/fpext on vectors, or any f32 vector float op.
// We work around fneg by XOR with the sign bit (bitcast float→int, xor).
int GetVectorWidth(const std::string& element_type,
                   const std::string& kernel_op) {
  // Float multiply: bf16 and f16 have native 32-wide fmul.
  if ((element_type == "bf16" || element_type == "f16") &&
      kernel_op == "arith.mulf")
    return 32;
  // Float negate: XOR sign bit works for all float types.
  if (element_type == "bf16" && kernel_op == "arith.negf") return 32;
  if (element_type == "f16" && kernel_op == "arith.negf") return 32;
  if (element_type == "f32" && kernel_op == "arith.negf") return 16;
  // Integer add/sub: native vector support (512-bit register).
  if (element_type == "i8" &&
      (kernel_op == "arith.addi" || kernel_op == "arith.subi"))
    return 64;
  if (element_type == "i16" &&
      (kernel_op == "arith.addi" || kernel_op == "arith.subi"))
    return 32;
  if (element_type == "i32" &&
      (kernel_op == "arith.addi" || kernel_op == "arith.subi"))
    return 16;
  return 0;
}

// Describes a linalg operation extracted from the module.
struct LinalgOpInfo {
  std::string op_name;        // e.g., "linalg.add"
  int num_inputs;             // 1 or 2
  int64_t num_elements;       // total element count
  std::string element_type;   // e.g., "f32"
  std::string kernel_op;      // e.g., "arith.addf"
};

// Full program description for AIE lowering.
struct LinalgProgramInfo {
  // For single-op programs, populated → delegates to existing scalar/vector paths.
  std::optional<LinalgOpInfo> single_op;

  // For multi-op generics, the GenericOp handle — body is walked directly
  // during codegen instead of building an intermediate DAG representation.
  mlir::linalg::GenericOp generic_op;

  int num_inputs = 0;
  int64_t num_elements = 0;
  std::string storage_type;  // memref element type (e.g., "f16")

  bool is_multi_op() const {
    return !single_op.has_value() && generic_op;
  }
};

// Returns true if the given op name is a supported body op for DAG analysis.
bool IsSupportedBodyOp(llvm::StringRef name) {
  return name == "arith.addf" || name == "arith.subf" || name == "arith.mulf" ||
         name == "arith.negf" || name == "arith.addi" || name == "arith.subi" ||
         name == "arith.muli" || name == "arith.extf" || name == "arith.truncf" ||
         name == "math.exp" || name == "math.log";
}

// Maps a single-body-op kernel_op name to a LinalgOpInfo op_name.
std::string GetSingleOpName(llvm::StringRef kernel_op) {
  if (kernel_op == "arith.addf" || kernel_op == "arith.addi") return "linalg.generic(add)";
  if (kernel_op == "arith.subf" || kernel_op == "arith.subi") return "linalg.generic(sub)";
  if (kernel_op == "arith.mulf" || kernel_op == "arith.muli") return "linalg.generic(mul)";
  if (kernel_op == "arith.negf") return "linalg.generic(neg)";
  if (kernel_op == "math.exp") return "linalg.generic(exp)";
  if (kernel_op == "math.log") return "linalg.generic(log)";
  return "linalg.generic(unknown)";
}

// Returns the number of inputs for a single-body-op kernel_op.
int GetSingleOpNumInputs(llvm::StringRef kernel_op) {
  if (kernel_op == "arith.negf" || kernel_op == "math.exp" || kernel_op == "math.log")
    return 1;
  return 2;
}

// Analyzes a linalg.generic to build a LinalgProgramInfo.
// Returns nullopt if the generic is not a supported element-wise computation.
std::optional<LinalgProgramInfo> AnalyzeLinalgGeneric(
    mlir::linalg::GenericOp generic) {
  // Check that all iterator types are parallel (element-wise).
  auto iter_types = generic.getIteratorTypesArray();
  for (auto it : iter_types) {
    if (it != mlir::utils::IteratorType::parallel) return std::nullopt;
  }

  if (generic.getInputs().empty()) return std::nullopt;

  // Reject multi-output generics (we only support single output).
  if (generic.getNumDpsInits() != 1) return std::nullopt;

  // Validate all indexing maps are identity (flat element-wise).
  for (auto map : generic.getIndexingMapsArray()) {
    if (!map.isIdentity()) return std::nullopt;
  }

  // Get shape and element type from first input.
  auto input_type = mlir::dyn_cast<mlir::RankedTensorType>(
      generic.getInputs().front().getType());
  if (!input_type) return std::nullopt;

  // Verify all inputs have the same shape.
  for (auto input : generic.getInputs()) {
    auto ty = mlir::dyn_cast<mlir::RankedTensorType>(input.getType());
    if (!ty || ty.getShape() != input_type.getShape()) return std::nullopt;
  }

  std::string storage_type = GetElementTypeStr(input_type.getElementType());

  // Reject f16: Peano's AIE2p backend treats f16 (half) as bf16 in its
  // fpext/fptrunc lowering (left-shift-by-16 and vconv.bf16.fp32), producing
  // incorrect results. All f16 ops are broken until Peano adds proper support.
  if (storage_type == "f16") return std::nullopt;

  // Verify output element type matches input element type. Our lowering uses
  // a single storage_type for all ObjectFIFOs (inputs + output). If the output
  // type differs, the generated memref types would be wrong.
  auto output_type = mlir::dyn_cast<mlir::RankedTensorType>(
      generic.getDpsInits()[0].getType());
  if (!output_type) return std::nullopt;
  if (GetElementTypeStr(output_type.getElementType()) != storage_type)
    return std::nullopt;

  int64_t num_elements = 1;
  for (int64_t dim : input_type.getShape()) {
    num_elements *= dim;
  }

  mlir::Block& body = generic.getRegion().front();

  // Validate all body ops are supported. Count total ops and compute ops
  // (non-cast) to decide single-op vs multi-op path.
  int total_op_count = 0;
  int compute_op_count = 0;
  std::string sole_kernel_op;
  for (mlir::Operation& body_op : body.without_terminator()) {
    llvm::StringRef name = body_op.getName().getStringRef();
    if (!IsSupportedBodyOp(name)) return std::nullopt;
    total_op_count++;
    if (name != "arith.extf" && name != "arith.truncf") {
      compute_op_count++;
      sole_kernel_op = name.str();
    }
  }

  if (compute_op_count == 0) return std::nullopt;

  LinalgProgramInfo program;
  program.num_inputs = static_cast<int>(generic.getInputs().size());
  program.num_elements = num_elements;
  program.storage_type = storage_type;

  // Single op with NO casts → use single_op path for vectorization and
  // bf16 multiply workaround. If casts are present (e.g., f16 extf→addf→truncf),
  // use multi-op path so the casts are correctly emitted.
  if (total_op_count == 1 && compute_op_count == 1) {
    LinalgOpInfo single;
    single.kernel_op = sole_kernel_op;
    single.op_name = GetSingleOpName(sole_kernel_op);
    single.num_inputs = GetSingleOpNumInputs(sole_kernel_op);
    single.element_type = storage_type;
    single.num_elements = num_elements;
    program.single_op = single;
  } else {
    // Multi-op: store the GenericOp handle for direct body walking in codegen.
    program.generic_op = generic;
  }

  return program;
}

// Analyzes the linalg module to extract program info.
absl::StatusOr<LinalgProgramInfo> AnalyzeLinalgModule(mlir::ModuleOp module) {
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
  LinalgProgramInfo program;

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

    LinalgOpInfo info;
    info.op_name = name.str();
    info.num_inputs = num_inputs;
    info.element_type = GetElementTypeStr(result_type.getElementType());
    if (info.element_type == "f16") return;  // f16 broken in Peano
    info.kernel_op = kernel_op;
    info.num_elements = 1;
    for (int64_t dim : result_type.getShape()) {
      info.num_elements *= dim;
    }

    program.single_op = info;
    program.num_inputs = num_inputs;
    program.num_elements = info.num_elements;
    program.storage_type = info.element_type;
  });

  // Also walk linalg.generic ops.
  entry_func.walk([&](mlir::linalg::GenericOp generic) {
    auto result = AnalyzeLinalgGeneric(generic);
    if (!result.has_value()) return;
    op_count++;
    program = std::move(*result);
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

  if (program.single_op.has_value()) {
    LOG(INFO) << "XDNA AIE lowering: found " << program.single_op->op_name
              << " on " << program.single_op->num_elements << "x"
              << program.single_op->element_type;
  } else {
    LOG(INFO) << "XDNA AIE lowering: found multi-op generic with "
              << program.num_inputs << " inputs, "
              << program.num_elements << "x" << program.storage_type;
  }

  return program;
}

// Generates a scalar element-wise loop body (memref.load/store, step 1).
std::string GenerateScalarLoop(const LinalgOpInfo& info,
                               const std::string& memref_ty,
                               int64_t chunk_size) {
  std::string body;
  const std::string& ty = info.element_type;

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

  return body;
}

// Generates a vectorized element-wise loop body (vector.load/store).
// Peano (Jan 2025) supports a limited set of vector operations:
//   - fmul <32 x bfloat> (native bf16 multiply)
//   - xor <32 x i16> / xor <16 x i32> (used for negate via sign-bit flip)
// We use bitcast + XOR for negate since fneg on vectors is not legalizable.
std::string GenerateVectorLoop(const LinalgOpInfo& info,
                               const std::string& memref_ty,
                               int64_t chunk_size, int vector_width) {
  std::string body;
  const std::string& ty = info.element_type;
  std::string vec_ty = absl::StrFormat("vector<%dx%s>", vector_width, ty);

  absl::StrAppend(&body,
      "      %c0 = arith.constant 0 : index\n");
  absl::StrAppend(&body, absl::StrFormat(
      "      %%c%d = arith.constant %d : index\n", chunk_size, chunk_size));
  absl::StrAppend(&body, absl::StrFormat(
      "      %%c%d = arith.constant %d : index\n", vector_width, vector_width));

  // For negate, pre-compute the sign-bit mask outside the loop.
  bool is_negate = (info.kernel_op == "arith.negf");
  if (is_negate) {
    if (ty == "bf16" || ty == "f16") {
      // bf16/f16 sign bit is 0x8000 (bit 15), integer type is i16
      absl::StrAppend(&body, absl::StrFormat(
          "      %%sign_mask = arith.constant dense<-32768>"
          " : vector<%dxi16>\n", vector_width));
    } else {
      // f32 sign bit is 0x80000000 (bit 31), integer type is i32
      absl::StrAppend(&body, absl::StrFormat(
          "      %%sign_mask = arith.constant dense<-2147483648>"
          " : vector<%dxi32>\n", vector_width));
    }
  }

  absl::StrAppend(&body, absl::StrFormat(
      "      scf.for %%i = %%c0 to %%c%d step %%c%d {\n",
      chunk_size, vector_width));

  absl::StrAppend(&body, absl::StrFormat(
      "        %%v0 = vector.load %%elem_in0[%%i] : %s, %s\n",
      memref_ty, vec_ty));

  if (is_negate) {
    // Negate via bitcast to integer, XOR with sign bit, bitcast back.
    // This avoids fneg which Peano's GlobalISel cannot legalize on vectors.
    std::string int_ty = (ty == "bf16" || ty == "f16") ? "i16" : "i32";
    std::string int_vec_ty = absl::StrFormat("vector<%dx%s>",
                                              vector_width, int_ty);
    absl::StrAppend(&body, absl::StrFormat(
        "        %%v0_int = arith.bitcast %%v0 : %s to %s\n",
        vec_ty, int_vec_ty));
    absl::StrAppend(&body, absl::StrFormat(
        "        %%vr_int = arith.xori %%v0_int, %%sign_mask : %s\n",
        int_vec_ty));
    absl::StrAppend(&body, absl::StrFormat(
        "        %%vr = arith.bitcast %%vr_int : %s to %s\n",
        int_vec_ty, vec_ty));
  } else if (info.num_inputs == 2) {
    absl::StrAppend(&body, absl::StrFormat(
        "        %%v1 = vector.load %%elem_in1[%%i] : %s, %s\n",
        memref_ty, vec_ty));
    absl::StrAppend(&body, absl::StrFormat(
        "        %%vr = %s %%v0, %%v1 : %s\n", info.kernel_op, vec_ty));
  }

  absl::StrAppend(&body, absl::StrFormat(
      "        vector.store %%vr, %%elem_out[%%i] : %s, %s\n",
      memref_ty, vec_ty));
  absl::StrAppend(&body,
      "      }\n");

  return body;
}

// Resolves an MLIR Value to its generated SSA name, or returns "???" if
// unmapped (indicates a bug in analysis — should never happen for validated ops).
std::string ResolveName(const llvm::DenseMap<mlir::Value, std::string>& names,
                        mlir::Value value) {
  auto it = names.find(value);
  if (it != names.end()) return it->second;
  LOG(ERROR) << "XDNA: unmapped value in multi-op body";
  return "???";
}

// Generates a scalar loop for multi-op linalg.generic bodies (e.g., bodies
// with extf/truncf casts around compute ops). Walks the GenericOp's body IR
// directly, using a value→name map to emit each op as text.
std::string GenerateMultiOpScalarLoop(mlir::linalg::GenericOp generic,
                                      int num_inputs,
                                      const std::string& memref_ty,
                                      int64_t chunk_size) {
  std::string body;

  absl::StrAppend(&body,
      "      %c0 = arith.constant 0 : index\n"
      "      %c1 = arith.constant 1 : index\n");
  absl::StrAppend(&body, absl::StrFormat(
      "      %%c%d = arith.constant %d : index\n", chunk_size, chunk_size));
  absl::StrAppend(&body, absl::StrFormat(
      "      scf.for %%i = %%c0 to %%c%d step %%c1 {\n", chunk_size));

  // Load all inputs.
  for (int i = 0; i < num_inputs; i++) {
    absl::StrAppend(&body, absl::StrFormat(
        "        %%val_in%d = memref.load %%elem_in%d[%%i] : %s\n",
        i, i, memref_ty));
  }

  // Map MLIR Values to generated SSA names.
  // A linalg.generic block has (num_inputs + num_outputs) arguments. The output
  // init arg (block.getArgument(num_inputs)) can be referenced by body ops
  // (e.g., accumulation patterns). We load it from the output buffer.
  mlir::Block& block = generic.getRegion().front();
  llvm::DenseMap<mlir::Value, std::string> names;
  for (int i = 0; i < num_inputs; i++) {
    names[block.getArgument(i)] = absl::StrFormat("%%val_in%d", i);
  }
  // Map the output init arg — load the current output element for read-modify.
  if (block.getNumArguments() > static_cast<unsigned>(num_inputs)) {
    absl::StrAppend(&body, absl::StrFormat(
        "        %%val_out_init = memref.load %%elem_out[%%i] : %s\n",
        memref_ty));
    names[block.getArgument(num_inputs)] = "%val_out_init";
  }

  // Emit each body op by walking the IR directly.
  int node_idx = 0;
  for (mlir::Operation& op : block.without_terminator()) {
    std::string result_name = absl::StrFormat("%%t%d", node_idx);
    llvm::StringRef op_name = op.getName().getStringRef();
    std::string result_type = GetElementTypeStr(op.getResult(0).getType());

    if (op_name == "arith.extf" || op_name == "arith.truncf") {
      // Cast ops: %tN = arith.extf %src : src_type to dst_type
      std::string src_type =
          GetElementTypeStr(op.getOperand(0).getType());
      absl::StrAppend(&body, absl::StrFormat(
          "        %s = %s %s : %s to %s\n",
          result_name, op_name, ResolveName(names, op.getOperand(0)),
          src_type, result_type));
    } else if (op_name == "arith.mulf" && result_type == "f32") {
      // f32 multiply workaround: AIE2p has no f32 mul instruction.
      // Truncate to bf16, multiply natively, extend back to f32.
      std::string lhs = ResolveName(names, op.getOperand(0));
      std::string rhs = ResolveName(names, op.getOperand(1));
      std::string t_lhs = absl::StrFormat("%%t%d_bf16_l", node_idx);
      std::string t_rhs = absl::StrFormat("%%t%d_bf16_r", node_idx);
      std::string t_mul = absl::StrFormat("%%t%d_bf16", node_idx);
      absl::StrAppend(&body, absl::StrFormat(
          "        %s = arith.truncf %s : f32 to bf16\n", t_lhs, lhs));
      absl::StrAppend(&body, absl::StrFormat(
          "        %s = arith.truncf %s : f32 to bf16\n", t_rhs, rhs));
      absl::StrAppend(&body, absl::StrFormat(
          "        %s = arith.mulf %s, %s : bf16\n", t_mul, t_lhs, t_rhs));
      absl::StrAppend(&body, absl::StrFormat(
          "        %s = arith.extf %s : bf16 to f32\n",
          result_name, t_mul));
    } else if (op.getNumOperands() == 1) {
      // Unary ops: %tN = arith.negf %src : type
      absl::StrAppend(&body, absl::StrFormat(
          "        %s = %s %s : %s\n",
          result_name, op_name,
          ResolveName(names, op.getOperand(0)), result_type));
    } else {
      // Binary ops: %tN = arith.addf %lhs, %rhs : type
      absl::StrAppend(&body, absl::StrFormat(
          "        %s = %s %s, %s : %s\n",
          result_name, op_name,
          ResolveName(names, op.getOperand(0)),
          ResolveName(names, op.getOperand(1)), result_type));
    }

    names[op.getResult(0)] = result_name;
    node_idx++;
  }

  // Store yield result.
  auto* yield = block.getTerminator();
  absl::StrAppend(&body, absl::StrFormat(
      "        memref.store %s, %%elem_out[%%i] : %s\n",
      ResolveName(names, yield->getOperand(0)), memref_ty));
  absl::StrAppend(&body, "      }\n");

  return body;
}

// Computes the largest chunk size that evenly divides num_elements, fits
// within max_chunk elements, and is aligned to vector_width (if nonzero).
// Returns 0 if no valid chunk size exists.
int64_t ComputeChunkSize(int64_t num_elements, int64_t max_chunk,
                          int vector_width) {
  int alignment = std::max(1, vector_width);
  // Round max_chunk down to a multiple of alignment.
  int64_t chunk = (max_chunk / alignment) * alignment;
  while (chunk > 0) {
    if (num_elements % chunk == 0) return chunk;
    chunk -= alignment;
  }
  return 0;
}

// Generates a single-core element-wise loop as AIE core body text.
// When num_chunks > 1, wraps the acquire/compute/release in an outer
// scf.for loop for streaming tiled execution. The DMA fills ObjectFIFO
// ping-pong buffers while the core processes the previous chunk.
std::string GenerateCoreBody(const LinalgProgramInfo& program,
                             int64_t chunk_size, int64_t num_chunks) {
  std::string body;
  const std::string& ty = program.storage_type;
  std::string memref_ty = absl::StrFormat("memref<%dx%s>", chunk_size, ty);

  // Outer loop over chunks (streaming DMA fills ObjectFIFO buffers).
  if (num_chunks > 1) {
    absl::StrAppend(&body, absl::StrFormat(
        "      %%n_chunks = arith.constant %d : index\n"
        "      %%c0_outer = arith.constant 0 : index\n"
        "      %%c1_outer = arith.constant 1 : index\n"
        "      scf.for %%_ = %%c0_outer to %%n_chunks step %%c1_outer {\n",
        num_chunks));
  }

  // Acquire input ObjectFIFO elements.
  for (int i = 0; i < program.num_inputs; i++) {
    absl::StrAppend(&body, absl::StrFormat(
        "      %%subview_in%d = aie.objectfifo.acquire @in%d(Consume, 1) "
        ": !aie.objectfifosubview<%s>\n", i, i, memref_ty));
    absl::StrAppend(&body, absl::StrFormat(
        "      %%elem_in%d = aie.objectfifo.subview.access %%subview_in%d[0] "
        ": !aie.objectfifosubview<%s> -> %s\n", i, i, memref_ty, memref_ty));
  }

  // Acquire output ObjectFIFO element.
  absl::StrAppend(&body,
      "      %subview_out = aie.objectfifo.acquire @out0(Produce, 1) "
      ": !aie.objectfifosubview<", memref_ty, ">\n");
  absl::StrAppend(&body,
      "      %elem_out = aie.objectfifo.subview.access %subview_out[0] "
      ": !aie.objectfifosubview<", memref_ty, "> -> ", memref_ty, "\n");

  // Choose loop generation path.
  if (program.is_multi_op()) {
    absl::StrAppend(&body,
                    GenerateMultiOpScalarLoop(program.generic_op,
                                             program.num_inputs,
                                             memref_ty, chunk_size));
  } else if (program.single_op.has_value()) {
    const LinalgOpInfo& info = *program.single_op;
    int vector_width = GetVectorWidth(info.element_type, info.kernel_op);
    if (vector_width > 0 && chunk_size >= vector_width &&
        chunk_size % vector_width == 0) {
      absl::StrAppend(&body,
                      GenerateVectorLoop(info, memref_ty, chunk_size,
                                         vector_width));
    } else {
      absl::StrAppend(&body,
                      GenerateScalarLoop(info, memref_ty, chunk_size));
    }
  }

  // Release ObjectFIFO elements.
  for (int i = 0; i < program.num_inputs; i++) {
    absl::StrAppend(&body, absl::StrFormat(
        "      aie.objectfifo.release @in%d(Consume, 1)\n", i));
  }
  absl::StrAppend(&body,
      "      aie.objectfifo.release @out0(Produce, 1)\n");

  // Close outer chunk loop.
  if (num_chunks > 1) {
    absl::StrAppend(&body, "      }\n");
  }

  return body;
}

// Generates the NPU instruction sequence for DMA setup.
// Follows the pattern from mlir-aie examples (e.g., add_314_using_dma_op):
//   1. Set up input DMA transfers
//   2. Set up output DMA with issue_token=true (enables completion tracking)
//   3. dma_wait on the output ObjectFIFO
//
// The shim DMA d0 (innermost) dimension is limited to 1023 32-bit words.
// When chunk_size exceeds this, we split each chunk into sub-transfers
// using the d1 dimension. Combined with tiling (num_chunks), the 4D
// addressing is:
//   sizes:   [1, num_chunks, d0_reps, d0_size]
//   strides: [0, chunk_size, d0_size, 1]
// Unused strides (for dims with size=1) are set to 0 to satisfy the
// aie-opt verifier on rank-1 host buffers.
std::string GenerateNpuSequence(const LinalgProgramInfo& program,
                                int64_t chunk_size, int64_t num_chunks) {
  const std::string& ty = program.storage_type;
  int64_t total_elements = chunk_size * num_chunks;
  // Runtime sequence args use full tensor memref; ObjectFIFOs use chunk memref.
  std::string host_memref_ty =
      absl::StrFormat("memref<%dx%s>", total_elements, ty);

  // Compute element size for DMA d0 limit.
  int element_bytes = 4;
  if (ty == "bf16" || ty == "f16" || ty == "i16") element_bytes = 2;
  else if (ty == "i8") element_bytes = 1;

  // Shim DMA d0 dimension is limited to 1023 32-bit words.
  // Split chunk into d0_reps sub-transfers of d0_size elements if needed.
  int64_t max_d0_elements = static_cast<int64_t>(1023) * 4 / element_bytes;
  int64_t d0_size = chunk_size;
  int64_t d0_reps = 1;
  if (chunk_size > max_d0_elements) {
    d0_size = max_d0_elements;
    while (d0_size > 0 && chunk_size % d0_size != 0) d0_size--;
    if (d0_size <= 0) d0_size = 1;
    d0_reps = chunk_size / d0_size;
  }

  // Strides: set to 0 when the corresponding dim has size 1 (unused),
  // to satisfy the aie-opt verifier on rank-1 memrefs.
  int64_t s1 = (d0_reps > 1) ? d0_size : 0;
  int64_t s2 = (num_chunks > 1) ? chunk_size : 0;

  std::string seq;

  // Runtime sequence: N inputs + 1 output.
  // Uses aiex.runtime_sequence (not func.func @sequence) so that
  // aie-dma-to-npu can lower the DMA ops to NPU instructions.
  std::vector<std::string> arg_types;
  for (int i = 0; i < program.num_inputs + 1; i++) {
    arg_types.push_back(absl::StrFormat("%%arg%d: %s", i, host_memref_ty));
  }
  absl::StrAppend(&seq, absl::StrFormat(
      "    aiex.runtime_sequence(%s) {\n",
      absl::StrJoin(arg_types, ", ")));

  // DMA transfers: inputs first.
  // 4D addressing: [1, num_chunks, d0_reps, d0_size]
  //   d0: innermost contiguous transfer (≤ 1023 32-bit words)
  //   d1: sub-chunks within a tile chunk (stride = d0_size)
  //   d2: tile chunks (stride = chunk_size)
  for (int i = 0; i < program.num_inputs; i++) {
    absl::StrAppend(&seq, absl::StrFormat(
        "      aiex.npu.dma_memcpy_nd(0, 0, %%arg%d[0, 0, 0, 0]"
        "[1, %d, %d, %d][0, %d, %d, 1]) "
        "{id = %d : i64, metadata = @in%d} : %s\n",
        i, num_chunks, d0_reps, d0_size,
        s2, s1,
        i, i, host_memref_ty));
  }

  // DMA transfer: output with issue_token for completion tracking.
  int out_arg = program.num_inputs;
  int out_id = program.num_inputs;
  absl::StrAppend(&seq, absl::StrFormat(
      "      aiex.npu.dma_memcpy_nd(0, 0, %%arg%d[0, 0, 0, 0]"
      "[1, %d, %d, %d][0, %d, %d, 1]) "
      "{id = %d : i64, metadata = @out0, issue_token = true} : %s\n",
      out_arg, num_chunks, d0_reps, d0_size,
      s2, s1,
      out_id, host_memref_ty));

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
    mlir::ModuleOp linalg_module, const AieLoweringConfig& config,
    const TargetCaps& caps) {
  // Step 1: Analyze the linalg module.
  auto program_result = AnalyzeLinalgModule(linalg_module);
  if (!program_result.ok()) return program_result.status();
  LinalgProgramInfo program = std::move(*program_result);

  // Step 1b: Compute tiling parameters for L1 memory budget.
  // Each ObjectFIFO buffer uses ping-pong depth of 2, so L1 per chunk is:
  //   num_buffers * chunk_size * element_bytes * 2 (ping-pong)
  int element_bytes = GetElementBytes(program.storage_type);
  int num_buffers = program.num_inputs + 1;  // inputs + output

  int64_t l1_limit = caps.l1_usable_bytes;
  const char* limit_env = std::getenv("XDNA_L1_LIMIT_BYTES");
  if (limit_env) l1_limit = std::atol(limit_env);

  // Max elements per chunk that fit in L1 with ping-pong ObjectFIFO buffers.
  int64_t max_chunk = l1_limit / (num_buffers * element_bytes * 2);

  // Determine vector width for chunk alignment (0 if not vectorized).
  int vector_width = 0;
  if (program.single_op.has_value()) {
    vector_width = GetVectorWidth(program.single_op->element_type,
                                   program.single_op->kernel_op);
  }

  int64_t chunk_size;
  int64_t num_chunks;
  if (program.num_elements <= max_chunk) {
    chunk_size = program.num_elements;
    num_chunks = 1;
  } else {
    chunk_size = ComputeChunkSize(program.num_elements, max_chunk,
                                   vector_width);
    if (chunk_size <= 0) {
      return absl::InvalidArgumentError(absl::StrFormat(
          "Cannot tile %d elements into L1 (%d bytes). "
          "Max %d elements per chunk (%d buffers x %d bytes x 2 ping-pong). "
          "No valid chunk size found (vector_width=%d). "
          "Consider using a tensor size with more factors. "
          "Override L1 limit with XDNA_L1_LIMIT_BYTES env var.",
          program.num_elements, l1_limit, max_chunk, num_buffers,
          element_bytes, vector_width));
    }
    num_chunks = program.num_elements / chunk_size;
  }

  LOG(INFO) << "XDNA AIE lowering: tiling " << program.num_elements
            << " elements into " << num_chunks << " chunk(s) of "
            << chunk_size << " (L1 budget: " << l1_limit << "B)";

  // Step 2: Generate AIE dialect MLIR text.
  //
  // Architecture: npu2 (XDNA 2 / Strix Halo / AIE2PS)
  // Layout: single column (col 0)
  //   - Shim tile (0, 0): host DMA interface
  //   - Mem tile (0, 1): L2 buffer (unused for single-core)
  //   - Compute tile (0, 2): runs the kernel

  std::string aie_mlir;
  const std::string& ty = program.storage_type;
  std::string memref_ty = absl::StrFormat("memref<%dx%s>", chunk_size, ty);

  int col = caps.partition_start_column;
  int shim_row = caps.shim_row;
  int compute_row = caps.first_compute_row;
  std::string shim_tile = absl::StrFormat("tile_%d_%d", col, shim_row);
  std::string compute_tile = absl::StrFormat("tile_%d_%d", col, compute_row);

  absl::StrAppend(&aie_mlir, "module {\n");
  absl::StrAppend(&aie_mlir,
      absl::StrFormat("  aie.device(%s) {\n", caps.device_name));

  // Tiles.
  absl::StrAppend(&aie_mlir, absl::StrFormat(
      "    %%%s = aie.tile(%d, %d)\n", shim_tile, col, shim_row));
  absl::StrAppend(&aie_mlir, absl::StrFormat(
      "    %%%s = aie.tile(%d, %d)\n", compute_tile, col, compute_row));

  // ObjectFIFOs for data movement: shim → compute.
  for (int i = 0; i < program.num_inputs; i++) {
    absl::StrAppend(&aie_mlir, absl::StrFormat(
        "    aie.objectfifo @in%d(%%%s, {%%%s}, 2 : i32) "
        ": !aie.objectfifo<%s>\n", i, shim_tile, compute_tile, memref_ty));
  }
  absl::StrAppend(&aie_mlir, absl::StrFormat(
      "    aie.objectfifo @out0(%%%s, {%%%s}, 2 : i32) "
      ": !aie.objectfifo<%s>\n", compute_tile, shim_tile, memref_ty));

  // Core body.
  absl::StrAppend(&aie_mlir, absl::StrFormat(
      "    %%core_%d_%d = aie.core(%%%s) {\n", col, compute_row,
      compute_tile));
  absl::StrAppend(&aie_mlir, GenerateCoreBody(program, chunk_size, num_chunks));
  absl::StrAppend(&aie_mlir,
      "      aie.end\n"
      "    }\n");

  // NPU instruction sequence.
  absl::StrAppend(&aie_mlir, GenerateNpuSequence(program, chunk_size,
                                                   num_chunks));

  absl::StrAppend(&aie_mlir, "  }\n");
  absl::StrAppend(&aie_mlir, "}\n");

  LOG(INFO) << "XDNA AIE lowering: generated AIE MLIR:\n" << aie_mlir;

  return aie_mlir;
}

}  // namespace xla
