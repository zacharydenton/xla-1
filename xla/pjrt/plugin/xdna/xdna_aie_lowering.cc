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

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <optional>
#include <string>
#include <vector>

#include "absl/strings/numbers.h"

#include "absl/log/check.h"
#include "absl/log/log.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/str_format.h"
#include "absl/strings/str_join.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/SmallPtrSet.h"
#include "llvm/ADT/SmallVector.h"
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
//
// For maximumf/minimumf: mlir-aie's --convert-vector-to-aievec lowers
// arith.maximumf/minimumf on 512-bit vectors to aievec.max/min → LLVM
// intrinsics (VectorMaxLtBf16, VectorMaxLt32, etc.). This bypasses Peano's
// GlobalISel entirely. AIE2p supports bf16 (32-wide) but NOT f32 max/min.
//
// For addf/subf: mlir-aie's --convert-vector-to-aievec lowers these to
// aievec.add_elem/sub_elem → ACC2048 accumulator intrinsics. Both f32 and
// bf16 use 16-wide vectors (bf16 goes through UPS→f32 add_elem→SRS path).
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
  // Float add/sub: 16-wide via aievec ACC2048 accumulator instructions.
  // bf16 uses UPS(shift=0) → f32 add_elem → SRS(shift=0) path.
  if ((element_type == "f32" || element_type == "bf16") &&
      (kernel_op == "arith.addf" || kernel_op == "arith.subf"))
    return 16;
  // Float max/min: bf16 has native 32-wide via aievec (no f32 intrinsic).
  if (element_type == "bf16" &&
      (kernel_op == "arith.maximumf" || kernel_op == "arith.minimumf"))
    return 32;
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

// Returns true if the given kernel_op at this vector width requires the
// aievec pipeline (--convert-vector-to-aievec + --convert-aievec-to-llvm).
bool NeedsAievecPipeline(const std::string& kernel_op) {
  return kernel_op == "arith.maximumf" || kernel_op == "arith.minimumf" ||
         kernel_op == "arith.addf" || kernel_op == "arith.subf";
}

// Describes a linalg operation extracted from the module.
struct LinalgOpInfo {
  std::string op_name;        // e.g., "linalg.add"
  int num_inputs;             // 1 or 2 (DMA inputs, excluding embedded constants)
  int64_t num_elements;       // total element count
  std::string element_type;   // e.g., "f32"
  std::string kernel_op;      // e.g., "arith.addf"
  // For ops with a broadcast constant (e.g., relu = max(x, 0)):
  // embedded_constant holds the MLIR literal (e.g., "0.000000e+00 : f32").
  // The op is a 2-operand op at the MLIR level but only 1 DMA input.
  std::string embedded_constant;
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

  // Per DMA-input: true if the input is a broadcast (suffix projection).
  // Size = num_inputs (DMA inputs only, not embedded constants).
  std::vector<bool> broadcast_inputs;
  int64_t broadcast_inner_dim = 0;   // N (row length, broadcast dim)
  int64_t broadcast_outer_dim = 0;   // M (number of rows)

  bool is_multi_op() const {
    return !single_op.has_value() && generic_op;
  }
};

// Matmul program description for AIE lowering.
struct MatmulProgramInfo {
  int64_t M, K, N;
  std::string element_type;  // "f32" or "bf16"
  bool b_transposed = false;  // B input has transposed host layout (N×K not K×N)
};

// Matmul tile sizes for L1 tiling.
struct MatmulTileConfig {
  int64_t m, k, n;  // tile dimensions
};

// Softmax program description for AIE lowering.
struct SoftmaxProgramInfo {
  int64_t num_rows;        // product of all dims except last (batch dims)
  int64_t row_length;      // last dimension (softmax axis)
  std::string element_type;  // "bf16"
};

// Fused attention program description for AIE lowering.
// Represents softmax(Q @ K^T / sqrt(dk)) @ V fused into a single kernel.
struct AttentionProgramInfo {
  int64_t num_rows;      // M dimension (Q rows, = seq_len for self-attention)
  int64_t seq_len;       // S dimension (K/V rows, scores width)
  int64_t dk;            // shared dimension (Q/K/V columns)
  std::string element_type;  // "bf16" only
  bool is_causal = false;  // Upper-triangular causal mask detected.
};

// GELU program description for AIE lowering.
// GELU(x) = 0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))
struct GeluProgramInfo {
  int64_t num_elements;
  std::string element_type;  // "bf16"
};

// LayerNorm program description for AIE lowering.
// Y = gamma * (X - mean) / sqrt(var + eps) + beta
struct LayerNormProgramInfo {
  int64_t num_rows;        // product of all dims except last
  int64_t row_length;      // last dimension (normalization axis)
  std::string element_type;  // "bf16"
};

// Returns true if the given op name is a supported body op for DAG analysis.
bool IsSupportedBodyOp(llvm::StringRef name) {
  return name == "arith.addf" || name == "arith.subf" || name == "arith.mulf" ||
         name == "arith.negf" || name == "arith.addi" || name == "arith.subi" ||
         name == "arith.muli" || name == "arith.extf" || name == "arith.truncf" ||
         name == "arith.maximumf" || name == "arith.maxsi" ||
         name == "arith.minimumf" || name == "arith.minsi" ||
         name == "math.exp" || name == "math.log";
}

// Maps a single-body-op kernel_op name to a LinalgOpInfo op_name.
std::string GetSingleOpName(llvm::StringRef kernel_op) {
  if (kernel_op == "arith.addf" || kernel_op == "arith.addi") return "linalg.generic(add)";
  if (kernel_op == "arith.subf" || kernel_op == "arith.subi") return "linalg.generic(sub)";
  if (kernel_op == "arith.mulf" || kernel_op == "arith.muli") return "linalg.generic(mul)";
  if (kernel_op == "arith.negf") return "linalg.generic(neg)";
  if (kernel_op == "arith.maximumf" || kernel_op == "arith.maxsi") return "linalg.generic(max)";
  if (kernel_op == "arith.minimumf" || kernel_op == "arith.minsi") return "linalg.generic(min)";
  if (kernel_op == "math.exp") return "linalg.generic(exp)";
  if (kernel_op == "math.log") return "linalg.generic(log)";
  return "linalg.generic(unknown)";
}

// Returns the number of inputs for a single-body-op kernel_op.
int GetSingleOpNumInputs(llvm::StringRef kernel_op) {
  if (kernel_op == "arith.negf" || kernel_op == "math.exp" || kernel_op == "math.log")
    return 1;
  // arith.maximumf/maxsi/minimumf/minsi are 2-input ops (e.g., relu = max(x, 0))
  return 2;
}

// Traces a Value through function call boundaries to find its origin.
// If `value` is a block argument of a private function, finds the corresponding
// actual argument at the call site. Returns the original value otherwise.
mlir::Value TraceBlockArgToCallSite(mlir::Value value,
                                    mlir::ModuleOp module) {
  auto block_arg = mlir::dyn_cast<mlir::BlockArgument>(value);
  if (!block_arg) return value;  // Already an SSA value with defining op.

  // Get the parent function.
  auto parent_func = mlir::dyn_cast<mlir::func::FuncOp>(
      block_arg.getOwner()->getParentOp());
  if (!parent_func) return value;

  // Find a call site of this function in the module.
  llvm::StringRef func_name = parent_func.getSymName();
  mlir::Value result = value;
  module.walk([&](mlir::func::CallOp call) {
    if (call.getCallee() == func_name) {
      unsigned arg_idx = block_arg.getArgNumber();
      if (arg_idx < call.getNumOperands()) {
        result = call.getOperand(arg_idx);
      }
    }
  });
  return result;
}

// Formats a scalar constant as an MLIR literal string (e.g. "0.000000e+00 : f32").
std::string FormatScalarConstant(mlir::Type elem_type, mlir::Attribute splat_val) {
  std::string type_str = GetElementTypeStr(elem_type);
  if (elem_type.isIntOrIndex()) {
    auto int_attr = mlir::dyn_cast<mlir::IntegerAttr>(splat_val);
    if (!int_attr) return "";
    return absl::StrFormat("%d : %s", int_attr.getInt(), type_str);
  }
  auto float_attr = mlir::dyn_cast<mlir::FloatAttr>(splat_val);
  if (!float_attr) return "";
  auto val = float_attr.getValue();
  llvm::SmallString<32> float_str;
  val.toString(float_str, /*FormatPrecision=*/6, /*FormatMaxPadding=*/0);
  std::string s = float_str.str().str();
  if (s.find('.') == std::string::npos && s.find('E') == std::string::npos &&
      s.find('e') == std::string::npos) {
    s += ".0";
  }
  return absl::StrCat(s, " : ", type_str);
}

// Tries to extract a scalar constant value from an MLIR Value.
// Returns the MLIR literal string (e.g., "0.000000e+00 : f32") or empty.
// Handles:
//   1. arith.constant dense<splat> : tensor<...>  (broadcast map)
//   2. linalg.fill ins(%cst) outs(%empty)         (identity-mapped splat tensor)
std::string TryExtractScalarConstant(mlir::Value value) {
  auto* def_op = value.getDefiningOp();
  if (!def_op) return "";

  llvm::StringRef op_name = def_op->getName().getStringRef();

  // Case 1: arith.constant dense<splat>
  if (op_name == "arith.constant") {
    auto attr = def_op->getAttr("value");
    if (!attr) return "";
    auto dense = mlir::dyn_cast<mlir::DenseElementsAttr>(attr);
    if (!dense || !dense.isSplat()) return "";
    auto elem_type = dense.getElementType();
    std::string type_str = GetElementTypeStr(elem_type);
    if (elem_type.isIntOrIndex()) {
      auto val = dense.getSplatValue<mlir::APInt>();
      return absl::StrFormat("%d : %s", val.getSExtValue(), type_str);
    }
    auto val = dense.getSplatValue<mlir::APFloat>();
    llvm::SmallString<32> float_str;
    val.toString(float_str, /*FormatPrecision=*/6, /*FormatMaxPadding=*/0);
    std::string s = float_str.str().str();
    if (s.find('.') == std::string::npos && s.find('E') == std::string::npos &&
        s.find('e') == std::string::npos) {
      s += ".0";
    }
    return absl::StrCat(s, " : ", type_str);
  }

  // Case 2: linalg.fill ins(%cst : scalar) outs(%empty)
  // stablehlo→linalg produces this for broadcast constants (e.g., relu zero).
  if (op_name == "linalg.fill") {
    if (def_op->getNumOperands() < 1) return "";
    // linalg.fill DPS: first operand is the scalar fill value.
    mlir::Value fill_val = def_op->getOperand(0);
    auto* fill_def = fill_val.getDefiningOp();
    if (!fill_def || fill_def->getName().getStringRef() != "arith.constant")
      return "";
    auto attr = fill_def->getAttr("value");
    if (!attr) return "";
    auto elem_type = fill_val.getType();
    return FormatScalarConstant(elem_type, attr);
  }

  // Case 3: linalg.generic that broadcasts a scalar constant to a tensor.
  // stablehlo.broadcast_in_dim(%cst) lowers to a linalg.generic with a
  // scalar input (map: (d0)->()) and body that just yields the input.
  if (op_name == "linalg.generic") {
    auto generic = mlir::dyn_cast<mlir::linalg::GenericOp>(def_op);
    if (!generic) return "";
    // Must have exactly 1 input (the scalar to broadcast).
    if (generic.getInputs().size() != 1) return "";
    // Body must be trivial: just yield the input block arg.
    mlir::Block& body = generic.getRegion().front();
    int non_yield = 0;
    for (mlir::Operation& op : body.without_terminator()) {
      (void)op;
      non_yield++;
    }
    if (non_yield != 0) return "";
    // The single input should be a constant.
    return TryExtractScalarConstant(generic.getInputs()[0]);
  }

  return "";
}

// Returns the float value of a scalar constant, or nullopt.
std::optional<float> TryExtractScalarFloat(mlir::Value value) {
  std::string s = TryExtractScalarConstant(value);
  if (s.empty()) return std::nullopt;
  // Parse the float from "0.176777 : bf16" format.
  size_t colon = s.find(" : ");
  if (colon == std::string::npos) return std::nullopt;
  float val;
  if (!absl::SimpleAtof(s.substr(0, colon), &val)) return std::nullopt;
  return val;
}

// Returns true if `map` is a suffix projection: the results are the trailing
// dimensions of the input, dropping at least one leading dimension.
// E.g., (d0, d1) -> (d1) maps a 2D iteration space to the last dim.
bool IsSuffixProjection(mlir::AffineMap map) {
  unsigned num_dims = map.getNumDims();
  unsigned num_results = map.getNumResults();
  if (num_results == 0 || num_results >= num_dims) return false;
  for (unsigned r = 0; r < num_results; r++) {
    auto expr = map.getResult(r);
    auto dim = mlir::dyn_cast<mlir::AffineDimExpr>(expr);
    if (!dim) return false;
    if (dim.getPosition() != num_dims - num_results + r) return false;
  }
  return true;
}

// Analyzes a linalg.generic to build a LinalgProgramInfo.
// Returns nullopt if the generic is not a supported element-wise computation.
//
// Supports three patterns:
//   1. All-identity indexing maps (standard elementwise: add, sub, mul, neg)
//   2. Mixed identity + scalar-broadcast maps where broadcast inputs are
//      compile-time constants (e.g., relu = max(x, 0) where 0 is broadcast)
//   3. Mixed identity + suffix-projection maps (DMA broadcast: [M,N] + [N])
std::optional<LinalgProgramInfo> AnalyzeLinalgGeneric(
    mlir::linalg::GenericOp generic, mlir::ModuleOp module) {
  // Check that all iterator types are parallel (element-wise).
  auto iter_types = generic.getIteratorTypesArray();
  for (auto it : iter_types) {
    if (it != mlir::utils::IteratorType::parallel) return std::nullopt;
  }

  if (generic.getInputs().empty()) return std::nullopt;

  // Reject multi-output generics (we only support single output).
  if (generic.getNumDpsInits() != 1) return std::nullopt;

  // Classify indexing maps: identity (DMA input), scalar broadcast (constant),
  // or suffix projection (DMA broadcast like [M,N] + [N]).
  auto maps = generic.getIndexingMapsArray();
  int num_total_inputs = static_cast<int>(generic.getInputs().size());
  std::vector<bool> is_broadcast(num_total_inputs, false);
  std::vector<std::string> broadcast_values(num_total_inputs);
  std::vector<bool> is_dma_broadcast(num_total_inputs, false);
  // For DMA broadcast inputs detected via broadcast generic: the original
  // (pre-broadcast) input type. Used to compute broadcast dimensions.
  std::vector<mlir::RankedTensorType> dma_broadcast_original_types(
      num_total_inputs);

  for (int i = 0; i < num_total_inputs; i++) {
    auto map = maps[i];
    if (map.isIdentity()) {
      // Identity map: could be a DMA input, a constant splat tensor, or
      // the result of a broadcast generic (stablehlo broadcast_in_dim).
      mlir::Value inp = generic.getInputs()[i];
      mlir::Value traced = TraceBlockArgToCallSite(inp, module);

      // Check for scalar constant broadcast.
      std::string constant = TryExtractScalarConstant(traced);
      if (!constant.empty()) {
        is_broadcast[i] = true;
        broadcast_values[i] = constant;
        continue;
      }

      // Check if the input is produced by a broadcast generic:
      // linalg.generic with a suffix-projection input map and trivial body
      // (just yields the input). XLA decomposes [M,N]+[N] into:
      //   %bcast = linalg.generic {maps=[(d0,d1)->(d1), (d0,d1)->(d0,d1)]}
      //            ins(%bias) outs(%empty) { yield %in }
      //   %result = linalg.generic {maps=[identity, identity, identity]}
      //             ins(%x, %bcast) outs(%out) { addf }
      auto* def_op = traced.getDefiningOp();
      if (def_op) {
        auto bcast_generic = mlir::dyn_cast<mlir::linalg::GenericOp>(def_op);
        if (bcast_generic && bcast_generic.getInputs().size() == 1) {
          // Check: trivial body (just yields input block arg).
          mlir::Block& bcast_body = bcast_generic.getRegion().front();
          bool trivial_body = true;
          for (mlir::Operation& op : bcast_body.without_terminator()) {
            (void)op;
            trivial_body = false;
            break;
          }
          if (trivial_body) {
            // Check: input has a suffix projection map.
            auto bcast_maps = bcast_generic.getIndexingMapsArray();
            if (bcast_maps.size() >= 1 && IsSuffixProjection(bcast_maps[0])) {
              is_dma_broadcast[i] = true;
              dma_broadcast_original_types[i] =
                  mlir::dyn_cast<mlir::RankedTensorType>(
                      bcast_generic.getInputs()[0].getType());
              continue;
            }
          }
        }
      }
      continue;
    }
    // Allow scalar broadcast: (d0, ...) -> () — all dims dropped.
    if (map.getNumResults() == 0) {
      // Must be a compile-time constant for us to embed it.
      std::string constant = TryExtractScalarConstant(generic.getInputs()[i]);
      if (constant.empty()) return std::nullopt;
      is_broadcast[i] = true;
      broadcast_values[i] = constant;
    } else if (IsSuffixProjection(map)) {
      // Suffix projection: e.g., (d0,d1) -> (d1). This input has fewer
      // dimensions than the output and needs DMA broadcast to all cores.
      is_dma_broadcast[i] = true;
    } else {
      return std::nullopt;  // Unsupported indexing pattern.
    }
  }

  // Output map must be identity.
  auto output_map = maps[num_total_inputs];  // output is after all inputs
  if (!output_map.isIdentity()) return std::nullopt;

  // Get shape and element type from the output (identity-mapped, full shape).
  auto output_type = mlir::dyn_cast<mlir::RankedTensorType>(
      generic.getDpsInits()[0].getType());
  if (!output_type) return std::nullopt;

  std::string storage_type = GetElementTypeStr(output_type.getElementType());

  // Reject f16: Peano's AIE2p backend treats f16 (half) as bf16 in its
  // fpext/fptrunc lowering (left-shift-by-16 and vconv.bf16.fp32), producing
  // incorrect results. All f16 ops are broken until Peano adds proper support.
  if (storage_type == "f16") return std::nullopt;

  // Get the first identity-mapped (non-broadcast) input for shape reference.
  mlir::RankedTensorType identity_input_type;
  for (int i = 0; i < num_total_inputs; i++) {
    if (is_broadcast[i] || is_dma_broadcast[i]) continue;
    identity_input_type = mlir::dyn_cast<mlir::RankedTensorType>(
        generic.getInputs()[i].getType());
    break;
  }
  // At least one identity input is required (the activation).
  if (!identity_input_type) return std::nullopt;

  // Verify identity input element type matches output.
  if (GetElementTypeStr(identity_input_type.getElementType()) != storage_type)
    return std::nullopt;

  // Verify all identity-mapped inputs have the same shape as the output.
  for (int i = 0; i < num_total_inputs; i++) {
    if (is_broadcast[i] || is_dma_broadcast[i]) continue;
    auto ty = mlir::dyn_cast<mlir::RankedTensorType>(
        generic.getInputs()[i].getType());
    if (!ty || ty.getShape() != output_type.getShape()) return std::nullopt;
  }

  // Verify DMA broadcast inputs: trailing dims must match output's trailing dims,
  // and element type must match.
  for (int i = 0; i < num_total_inputs; i++) {
    if (!is_dma_broadcast[i]) continue;
    auto ty = mlir::dyn_cast<mlir::RankedTensorType>(
        generic.getInputs()[i].getType());
    if (!ty) return std::nullopt;
    if (GetElementTypeStr(ty.getElementType()) != storage_type)
      return std::nullopt;
    auto out_shape = output_type.getShape();
    auto in_shape = ty.getShape();
    // Trailing dims of broadcast input must match trailing dims of output.
    int offset = out_shape.size() - in_shape.size();
    for (unsigned r = 0; r < in_shape.size(); r++) {
      if (in_shape[r] != out_shape[offset + r]) return std::nullopt;
    }
  }

  int64_t num_elements = 1;
  for (int64_t dim : output_type.getShape()) {
    num_elements *= dim;
  }

  mlir::Block& body = generic.getRegion().front();

  // Validate all body ops are supported.
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

  // Count DMA inputs (non-scalar-broadcast; includes DMA broadcast inputs).
  int dma_inputs = 0;
  bool has_dma_broadcast = false;
  for (int i = 0; i < num_total_inputs; i++) {
    if (!is_broadcast[i]) {
      dma_inputs++;
      if (is_dma_broadcast[i]) has_dma_broadcast = true;
    }
  }

  LinalgProgramInfo program;
  program.num_inputs = dma_inputs;
  program.num_elements = num_elements;
  program.storage_type = storage_type;

  // Build broadcast_inputs vector (indexed by DMA input order).
  if (has_dma_broadcast) {
    int dma_idx = 0;
    program.broadcast_inputs.resize(dma_inputs, false);
    for (int i = 0; i < num_total_inputs; i++) {
      if (is_broadcast[i]) continue;  // skip scalar constants
      if (is_dma_broadcast[i]) {
        program.broadcast_inputs[dma_idx] = true;
      }
      dma_idx++;
    }
    // Compute broadcast dimensions from the original (pre-broadcast) shape.
    // For broadcast-generic pattern: original type is stored.
    // For suffix projection pattern: use the input type directly.
    for (int i = 0; i < num_total_inputs; i++) {
      if (!is_dma_broadcast[i]) continue;
      mlir::RankedTensorType orig_ty = dma_broadcast_original_types[i];
      if (!orig_ty) {
        // Suffix projection path: input type IS the original shape.
        orig_ty = mlir::dyn_cast<mlir::RankedTensorType>(
            generic.getInputs()[i].getType());
      }
      if (!orig_ty) continue;
      int64_t inner = 1;
      for (int64_t d : orig_ty.getShape()) inner *= d;
      program.broadcast_inner_dim = inner;
      program.broadcast_outer_dim = num_elements / inner;
      break;
    }
  }

  // Single op with NO casts → use single_op path.
  if (total_op_count == 1 && compute_op_count == 1) {
    LinalgOpInfo single;
    single.kernel_op = sole_kernel_op;
    single.op_name = GetSingleOpName(sole_kernel_op);
    single.num_inputs = dma_inputs;
    single.element_type = storage_type;
    single.num_elements = num_elements;
    // If there's a broadcast constant, embed it.
    for (int i = 0; i < num_total_inputs; i++) {
      if (is_broadcast[i]) {
        single.embedded_constant = broadcast_values[i];
        break;
      }
    }
    program.single_op = single;
  } else {
    // Multi-op: store the GenericOp handle for direct body walking in codegen.
    program.generic_op = generic;
  }

  return program;
}

// Selects matmul tile sizes (m, k, n) that fit in L1 memory.
//
// L1 budget with depth-2 ObjectFIFOs (ping-pong):
//   bf16: 4*(m*k + k*n + 2*m*n) bytes  (A,B bf16 FIFOs + C bf16 FIFO + f32 acc)
//   f32:  8*(m*k + k*n + m*n)   bytes  (A,B,C f32 FIFOs, accumulate in C)
absl::StatusOr<MatmulTileConfig> SelectMatmulTiles(
    const MatmulProgramInfo& info, const TargetCaps& caps,
    int num_cores) {
  int64_t l1_limit = caps.l1_usable_bytes;
  const char* limit_env = std::getenv("XDNA_L1_LIMIT_BYTES");
  if (limit_env) l1_limit = std::atol(limit_env);

  bool is_bf16 = (info.element_type == "bf16");

  // Tile candidates: try largest first for m; scalar uses all for n.
  // Peano's VLIW scheduler (Jan 2025) has a bug at opt O2, but we use
  // opt O0 + llc O2 to avoid it. Larger tiles reduce DMA command count
  // (limited by BD slots, ~16 per channel).
  static constexpr int64_t kMNCandidates[] = {64, 48, 32, 16, 8, 4};
  static constexpr int64_t kKCandidatesScalar[] = {4};

  // For bf16 vectorized matmul (hardware MAC 4x8x8):
  //   k=8 fixed (MAC reduction width),
  //   m must be multiple of 4 (MAC output rows), prefer large m to
  //   reduce Mt (DMA blocks), since shim DMA has ~16 BD slots.
  //   n=8 (MAC output width, one MAC column per tile).
  //   The inner loop iterates over m in steps of 4, so any m works.
  static constexpr int64_t kMCandidatesVec[] = {32, 16, 8, 4};
  static constexpr int64_t kNCandidatesVec[] = {8};

  // Use vectorized path for bf16 when K is divisible by 8.
  // Transposed B disables vectorization: the MAC expects B in K×N row-major
  // layout, but transposed B arrives as N×K tiles in L1.
  bool use_vectorized = is_bf16 && (info.K % 8 == 0) && !info.b_transposed;

  auto fits_l1 = [&](int64_t m, int64_t k, int64_t n) -> bool {
    int64_t bytes;
    if (is_bf16) {
      // A(bf16) + B(bf16) + C(bf16) FIFOs each depth 2 = 4*elem_count*2
      // Plus f32 accumulator buffer (not a FIFO, single copy) = 4*m*n
      bytes = 4 * (m * k + k * n + m * n) + 4 * m * n;
    } else {
      // A,B,C f32 FIFOs each depth 2 = 4*elem_count*2 = 8*elem_count
      bytes = 8 * (m * k + k * n + m * n);
    }
    return bytes <= l1_limit;
  };

  if (use_vectorized) {
    // Vectorized: m from kMCandidatesVec, n=8, k=8 (fixed MAC width).
    // The K-tile loop is fully unrolled at the MLIR level to avoid Peano
    // miscompilation of loops containing BFP MAC intrinsics.
    static constexpr int64_t kKCandidatesVec[] = {8};
    for (int64_t m : kMCandidatesVec) {
      if (info.M % m != 0) continue;
      for (int64_t n : kNCandidatesVec) {
        if (info.N % n != 0) continue;
        int64_t Nt = info.N / n;
        if (Nt % num_cores != 0) continue;
        if (Nt / num_cores > 64) continue;  // DMA d3 limit per core
        for (int64_t k : kKCandidatesVec) {
          if (info.K % k != 0) continue;
          if (fits_l1(m, k, n)) {
            return MatmulTileConfig{m, k, n};
          }
        }
      }
    }
    // Fall back to scalar if vectorized tiles don't fit.
  }

  // Scalar: try all m,n candidates with k=4.
  auto try_scalar = [&]() -> std::optional<MatmulTileConfig> {
    for (int64_t m : kMNCandidates) {
      if (info.M % m != 0) continue;
      for (int64_t k : kKCandidatesScalar) {
        if (info.K % k != 0) continue;
        for (int64_t n : kMNCandidates) {
          if (info.N % n != 0) continue;
          int64_t Nt = info.N / n;
          if (Nt % num_cores != 0) continue;
          if (Nt / num_cores > 64) continue;  // DMA d3 limit per core
          if (fits_l1(m, k, n)) {
            return MatmulTileConfig{m, k, n};
          }
        }
      }
    }
    return std::nullopt;
  };

  auto result = try_scalar();
  if (result.has_value()) return *result;

  return absl::InvalidArgumentError(absl::StrFormat(
      "Cannot tile matmul [%d,%d]@[%d,%d] %s into L1 (%d bytes). "
      "No valid (m,k,n) tile found from candidates {64,48,32,16,8,4}.",
      info.M, info.K, info.K, info.N, info.element_type, l1_limit));
}

// Forward declaration.
absl::StatusOr<AieLoweringResult> LowerMatmulToAieInternal(
    const MatmulProgramInfo& info, const TargetCaps& caps,
    int max_columns);

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
  // Also collect unsupported op names for better error messages.
  // Walk the module (not just entry_func) because XLA may generate private
  // helper functions (e.g., @relu.1) containing linalg ops. The MLIR inliner
  // may not inline all functions (depends on call graph structure).
  // Note: if the inliner creates duplicates (inlined copy + dead original),
  // the original should be deleted by the inliner. If not, we may overcount
  // but this is handled by the op_count > 1 check producing a clear error.
  int op_count = 0;
  std::vector<std::string> unsupported_ops;
  LinalgProgramInfo program;

  // Walk the module to find linalg named ops.
  module.walk([&](mlir::Operation* op) {
    llvm::StringRef name = op->getName().getStringRef();
    if (!name.starts_with("linalg.")) return;

    // Skip linalg.generic — handled separately below.
    if (name == "linalg.generic") return;
    // Skip linalg.fill — used for zero-init before matmul.
    if (name == "linalg.fill") return;
    // Skip linalg.yield — it's a block terminator, not an op.
    if (name == "linalg.yield") return;

    int num_inputs = GetNumInputs(name);
    if (num_inputs < 0) {
      unsupported_ops.push_back(name.str());
      return;
    }

    std::string kernel_op = GetCoreKernelOp(name);
    if (kernel_op.empty()) {
      unsupported_ops.push_back(name.str());
      return;
    }

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

  // Also walk linalg.generic ops. Collect unsupported body ops for diagnostics.
  module.walk([&](mlir::linalg::GenericOp generic) {
    auto result = AnalyzeLinalgGeneric(generic, module);
    if (!result.has_value()) {
      // Collect unsupported body ops for error messages.
      mlir::Block& body = generic.getRegion().front();
      for (mlir::Operation& body_op : body.without_terminator()) {
        llvm::StringRef bname = body_op.getName().getStringRef();
        if (!IsSupportedBodyOp(bname) && bname != "linalg.yield") {
          unsupported_ops.push_back(bname.str());
        }
      }
      return;
    }
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
    // Build descriptive error message with unsupported ops.
    std::string unsupported_str;
    if (!unsupported_ops.empty()) {
      // Deduplicate unsupported ops.
      std::sort(unsupported_ops.begin(), unsupported_ops.end());
      unsupported_ops.erase(
          std::unique(unsupported_ops.begin(), unsupported_ops.end()),
          unsupported_ops.end());
      unsupported_str = absl::StrCat(
          " Unsupported ops: ", absl::StrJoin(unsupported_ops, ", "),
          ".");
    }
    // Log the full module IR for debugging (visible via LOG(INFO)).
    std::string module_str;
    llvm::raw_string_ostream os(module_str);
    module.print(os);
    LOG(INFO) << "XDNA: unsupported module IR:\n" << module_str;
    return absl::UnimplementedError(absl::StrCat(
        "XDNA cannot compile this operation.",
        unsupported_str,
        " Supported: elementwise (add, sub, mul, neg, max, min),"
        " matmul. Use jax.jit(fn, backend='cpu') for unsupported ops."));
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
  if (chunk_size > 1) {
    absl::StrAppend(&body, absl::StrFormat(
        "      %%c%d = arith.constant %d : index\n", chunk_size, chunk_size));
  }

  // Emit embedded constant before the loop (e.g., relu zero constant).
  bool has_constant = !info.embedded_constant.empty();
  if (has_constant) {
    absl::StrAppend(&body, absl::StrFormat(
        "      %%const_val = arith.constant %s\n", info.embedded_constant));
  }

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

  // Determine if this is a max/min op that needs cmpf+select lowering.
  // arith.maximumf/minimumf lower to llvm.intr.maximum/minimum which call
  // __unordsf2 (soft-float NaN check) — unavailable in Peano's baremetal env.
  // arith.maxsi/minsi lower to llvm.smax/smin which may also need intrinsics.
  // We use compare+select instead, which Peano handles natively.
  bool is_maxf = (info.kernel_op == "arith.maximumf");
  bool is_minf = (info.kernel_op == "arith.minimumf");
  bool is_maxi = (info.kernel_op == "arith.maxsi");
  bool is_mini = (info.kernel_op == "arith.minsi");
  bool is_cmp_select_op = is_maxf || is_minf || is_maxi || is_mini;

  absl::StrAppend(&body,
      "        %val_in0 = memref.load %elem_in0[%i] : ", memref_ty, "\n");

  // Determine the second operand name.
  std::string rhs_name;
  if (has_constant) {
    rhs_name = "%const_val";
  } else if (info.num_inputs == 2) {
    absl::StrAppend(&body,
        "        %val_in1 = memref.load %elem_in1[%i] : ", memref_ty, "\n");
    rhs_name = "%val_in1";
  }

  if (is_cmp_select_op && !rhs_name.empty()) {
    if (is_maxi || is_mini) {
      // Integer max/min: use arith.cmpi directly (native on AIE).
      std::string cmp_pred = is_maxi ? "sgt" : "slt";
      absl::StrAppend(&body, absl::StrFormat(
          "        %%cmp = arith.cmpi %s, %%val_in0, %s : %s\n",
          cmp_pred, rhs_name, ty));
      absl::StrAppend(&body, absl::StrFormat(
          "        %%val_out = arith.select %%cmp, %%val_in0, %s : %s\n",
          rhs_name, ty));
    } else if (ty == "f32") {
      // Float max/min for f32: use integer bitcast comparison.
      // AIE2p has no native f32 comparison instruction; arith.cmpf would
      // lower to __gtsf2/__ltsf2 soft-float calls which crash/hang at -O0.
      // IEEE 754 f32 sign bit: positive floats have bit 31 = 0.
      // For max(x, y): bitcast to i32, check signs, compare magnitudes.
      // Simplified for common case (relu: max(x, 0.0)):
      //   - positive x: sign bit 0 → i32 >= 0 → x > 0.0 → output x
      //   - negative x: sign bit 1 → i32 < 0  → x < 0.0 → output 0.0
      std::string cmp_pred = is_maxf ? "sgt" : "slt";
      absl::StrAppend(&body, absl::StrFormat(
          "        %%bits_a = arith.bitcast %%val_in0 : f32 to i32\n"
          "        %%bits_b = arith.bitcast %s : f32 to i32\n"
          "        %%cmp = arith.cmpi %s, %%bits_a, %%bits_b : i32\n"
          "        %%val_out = arith.select %%cmp, %%val_in0, %s : f32\n",
          rhs_name, cmp_pred, rhs_name));
    } else {
      // bf16/f16: use arith.cmpf (native bf16 comparison on AIE2p).
      // Use ordered predicates (ogt/olt). NaN propagation is not guaranteed:
      // if either operand is NaN, ogt/olt returns false, selecting RHS.
      // This matches JAX's relu behavior (no NaN inputs expected).
      // TODO: use ugt/ult for proper IEEE 754-2019 maximumf NaN semantics
      // once we verify Peano supports unordered float compares on AIE2p.
      std::string cmp_pred = is_maxf ? "ogt" : "olt";
      absl::StrAppend(&body, absl::StrFormat(
          "        %%cmp = arith.cmpf %s, %%val_in0, %s : %s\n",
          cmp_pred, rhs_name, ty));
      absl::StrAppend(&body, absl::StrFormat(
          "        %%val_out = arith.select %%cmp, %%val_in0, %s : %s\n",
          rhs_name, ty));
    }
  } else if (has_constant) {
    // Binary op with one DMA input and one embedded constant.
    absl::StrAppend(&body,
        "        %val_out = ", info.kernel_op,
        " %val_in0, %const_val : ", ty, "\n");
  } else if (info.num_inputs == 2) {
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
  if (vector_width != chunk_size) {
    absl::StrAppend(&body, absl::StrFormat(
        "      %%c%d = arith.constant %d : index\n", vector_width,
        vector_width));
  }

  // Pre-compute constants outside the loop.
  bool is_negate = (info.kernel_op == "arith.negf");
  bool has_constant = !info.embedded_constant.empty();
  if (is_negate) {
    if (ty == "bf16" || ty == "f16") {
      absl::StrAppend(&body, absl::StrFormat(
          "      %%sign_mask = arith.constant dense<-32768>"
          " : vector<%dxi16>\n", vector_width));
    } else {
      absl::StrAppend(&body, absl::StrFormat(
          "      %%sign_mask = arith.constant dense<-2147483648>"
          " : vector<%dxi32>\n", vector_width));
    }
  }
  if (has_constant) {
    // Splat the embedded constant to a vector for the binary op.
    absl::StrAppend(&body, absl::StrFormat(
        "      %%scalar_const = arith.constant %s\n", info.embedded_constant));
    absl::StrAppend(&body, absl::StrFormat(
        "      %%v_const = vector.broadcast %%scalar_const : %s to %s\n",
        ty, vec_ty));
  }

  absl::StrAppend(&body, absl::StrFormat(
      "      scf.for %%i = %%c0 to %%c%d step %%c%d {\n",
      chunk_size, vector_width));

  absl::StrAppend(&body, absl::StrFormat(
      "        %%v0 = vector.load %%elem_in0[%%i] : %s, %s\n",
      memref_ty, vec_ty));

  if (is_negate) {
    // Negate via bitcast to integer, XOR with sign bit, bitcast back.
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
  } else if (has_constant) {
    // Binary op with one DMA input vector and one splatted constant vector.
    absl::StrAppend(&body, absl::StrFormat(
        "        %%vr = %s %%v0, %%v_const : %s\n", info.kernel_op, vec_ty));
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
  if (chunk_size > 1) {
    absl::StrAppend(&body, absl::StrFormat(
        "      %%c%d = arith.constant %d : index\n", chunk_size, chunk_size));
  }
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
    } else if (op_name == "arith.maximumf" || op_name == "arith.minimumf" ||
               op_name == "arith.maxsi" || op_name == "arith.minsi") {
      // Max/min ops: use cmpf/cmpi + select to avoid __unordsf2 libcall.
      // Float: use ordered predicates (ogt/olt). NaN not propagated.
      bool is_float = (op_name == "arith.maximumf" || op_name == "arith.minimumf");
      std::string cmp_op = is_float ? "arith.cmpf" : "arith.cmpi";
      std::string pred = (op_name == "arith.maximumf") ? "ogt"
                       : (op_name == "arith.minimumf") ? "olt"
                       : (op_name == "arith.maxsi") ? "sgt" : "slt";
      std::string lhs = ResolveName(names, op.getOperand(0));
      std::string rhs = ResolveName(names, op.getOperand(1));
      std::string cmp_name = absl::StrFormat("%%t%d_cmp", node_idx);
      absl::StrAppend(&body, absl::StrFormat(
          "        %s = %s %s, %s, %s : %s\n",
          cmp_name, cmp_op, pred, lhs, rhs, result_type));
      absl::StrAppend(&body, absl::StrFormat(
          "        %s = arith.select %s, %s, %s : %s\n",
          result_name, cmp_name, lhs, rhs, result_type));
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

// Shim DMA BD size fields are 10 bits for d0 and d1, giving a max of 1023
// 32-bit words per dimension.
static constexpr int64_t kMaxBdWords = 1023;

// Tries to split chunk_size elements into d0_size * d0_reps where both
// fit within the shim DMA BD 10-bit fields (≤ 1023 32-bit words).
// Returns true and sets d0_size/d0_reps on success; false if no valid split
// exists.
bool FindDmaSplit(int64_t chunk_size, int element_bytes,
                  int64_t& d0_size, int64_t& d0_reps) {
  int64_t max_d0_elements = kMaxBdWords * 4 / element_bytes;
  if (chunk_size <= max_d0_elements) {
    d0_size = chunk_size;
    d0_reps = 1;
    return true;
  }
  // Find largest d0_size ≤ max_d0_elements that evenly divides chunk_size
  // and yields d0_reps ≤ kMaxBdWords.
  d0_size = max_d0_elements;
  while (d0_size > 0) {
    if (chunk_size % d0_size == 0) {
      d0_reps = chunk_size / d0_size;
      if (d0_reps <= kMaxBdWords) return true;
    }
    d0_size--;
  }
  return false;
}

// Computes the largest chunk size that evenly divides num_elements, fits
// within max_chunk elements, is aligned to vector_width (if nonzero), and
// has a valid DMA d0/d1 split (both ≤ 1023 32-bit words).
// Returns 0 if no valid chunk size exists.
int64_t ComputeChunkSize(int64_t num_elements, int64_t max_chunk,
                          int vector_width, int element_bytes) {
  int alignment = std::max(1, vector_width);
  // Round max_chunk down to a multiple of alignment.
  int64_t chunk = (max_chunk / alignment) * alignment;
  while (chunk > 0) {
    if (num_elements % chunk == 0) {
      // Verify this chunk has a valid DMA split.
      int64_t d0, d1;
      if (FindDmaSplit(chunk, element_bytes, d0, d1)) return chunk;
    }
    chunk -= alignment;
  }
  return 0;
}

// Generates a single-core element-wise loop as AIE core body text.
// When num_chunks > 1, wraps the acquire/compute/release in an outer
// scf.for loop for streaming tiled execution. The DMA fills ObjectFIFO
// ping-pong buffers while the core processes the previous chunk.
// fifo_prefix: prefix for ObjectFIFO names (e.g., "c0_" for multi-core,
// "" for single-core backward compatibility).
std::string GenerateCoreBody(const LinalgProgramInfo& program,
                             int64_t chunk_size, int64_t num_chunks,
                             const std::string& fifo_prefix) {
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
        "      %%subview_in%d = aie.objectfifo.acquire @%sin%d(Consume, 1) "
        ": !aie.objectfifosubview<%s>\n", i, fifo_prefix, i, memref_ty));
    absl::StrAppend(&body, absl::StrFormat(
        "      %%elem_in%d = aie.objectfifo.subview.access %%subview_in%d[0] "
        ": !aie.objectfifosubview<%s> -> %s\n", i, i, memref_ty, memref_ty));
  }

  // Acquire output ObjectFIFO element.
  absl::StrAppend(&body, absl::StrFormat(
      "      %%subview_out = aie.objectfifo.acquire @%sout0(Produce, 1) "
      ": !aie.objectfifosubview<%s>\n", fifo_prefix, memref_ty));
  absl::StrAppend(&body, absl::StrFormat(
      "      %%elem_out = aie.objectfifo.subview.access %%subview_out[0] "
      ": !aie.objectfifosubview<%s> -> %s\n", memref_ty, memref_ty));

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
        "      aie.objectfifo.release @%sin%d(Consume, 1)\n",
        fifo_prefix, i));
  }
  absl::StrAppend(&body, absl::StrFormat(
      "      aie.objectfifo.release @%sout0(Produce, 1)\n", fifo_prefix));

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
// Shim DMA ND addressing uses 4 dimensions with the following limits
// (empirically verified against aie-opt from mlir-aie commit 0d03400dca76):
//
//   sizes:   [d3,         d2,         d1,        d0        ]
//   limits:  [1:64]       [no limit]  [0:1023]   [0:1023]
//
// d0 and d1 limits are in 32-bit words — verified empirically:
//   f32[1024] (1024 words) fails when d1 stride is non-zero
//   bf16[1024] (512 words) passes in same context
// The verifier only enforces d0/d1 limits when the dimension is active
// (i.e., its stride is non-zero). Single-transfer cases with stride=0
// bypass the check.
//
// Our 4D addressing pattern:
//   sizes:   [1, num_chunks, d0_reps, d0_size]
//   strides: [0, chunk_size, d0_size, 1]
// Unused strides (for dims with size=1) are set to 0 to satisfy the
// aie-opt verifier on rank-1 host buffers.
//
// Dimension mapping:
//   d0 = d0_size: innermost contiguous transfer (≤ 1023 32-bit words)
//   d1 = d0_reps: sub-chunks within a tile chunk (≤ 1023)
//   d2 = num_chunks: tile chunks (no practical limit)
//   d3 = 1: always (limit [1:64])
absl::StatusOr<std::string> GenerateNpuSequence(
    const LinalgProgramInfo& program,
    int64_t per_core_elements, int64_t chunk_size, int64_t num_chunks,
    bool use_mem_tile, int num_cores, int start_col,
    bool use_distribute) {
  const std::string& ty = program.storage_type;
  int64_t total_elements = per_core_elements * num_cores;
  // Runtime sequence args use full tensor memref; ObjectFIFOs use chunk memref.
  std::string host_memref_ty =
      absl::StrFormat("memref<%dx%s>", total_elements, ty);

  // Compute element size for DMA d0/d1 limits.
  int element_bytes = 4;
  if (ty == "bf16" || ty == "f16" || ty == "i16") element_bytes = 2;
  else if (ty == "i8") element_bytes = 1;

  // For single-chunk contiguous transfers (num_chunks==1), all higher strides
  // are 0, so d0 can be arbitrarily large — the verifier only enforces the
  // 1023-word limit when the dimension's stride is non-zero.
  int64_t d0_size = chunk_size;
  int64_t d0_reps = 1;
  if (num_chunks > 1) {
    // Tiled case: chunk planning (ComputeChunkSize) already verified a valid
    // DMA split exists for this chunk_size.
    bool ok = FindDmaSplit(chunk_size, element_bytes, d0_size, d0_reps);
    if (!ok) {
      return absl::InvalidArgumentError(absl::StrFormat(
          "No valid DMA split for chunk_size=%d (element_bytes=%d). "
          "This should have been caught by ComputeChunkSize.",
          chunk_size, element_bytes));
    }
  }

  // Metadata names for shim-side ObjectFIFOs.
  // With mem tile routing, shim connects to @in*_L2 / @out0_L2.
  std::string l2 = use_mem_tile ? "_L2" : "";

  // FIFO name helper: multi-core uses "c{col}_" prefix, single-core uses "".
  auto fifo_name = [&](int c, const std::string& base) -> std::string {
    return num_cores > 1 ? absl::StrFormat("c%d_%s", c, base) : base;
  };

  std::string seq;

  // Runtime sequence: N inputs + 1 output.
  // Uses aie.runtime_sequence (not func.func @sequence) so that
  // aie-dma-to-npu can lower the DMA ops to NPU instructions.
  std::vector<std::string> arg_types;
  for (int i = 0; i < program.num_inputs + 1; i++) {
    arg_types.push_back(absl::StrFormat("%%arg%d: %s", i, host_memref_ty));
  }
  absl::StrAppend(&seq, absl::StrFormat(
      "    aie.runtime_sequence(%s) {\n",
      absl::StrJoin(arg_types, ", ")));

  if (use_distribute) {
    // Distribute/join: single DMA per input/output with ND gather/scatter.
    // Host memory layout: [core0_all | core1_all | ... | coreN_all]
    // Distribute expects interleaved: [core0_chunk0 | core1_chunk0 | ...]
    //
    // When d0_reps == 1 (chunk fits in d0):
    //   [1, num_chunks, num_cores, chunk_size]
    //   strides: [0, chunk_size, per_core_elements, 1]
    //
    // When d0_reps > 1 (chunk needs d0/d1 split):
    //   [num_chunks, num_cores, d0_reps, d0_size]
    //   strides: [chunk_size, per_core_elements, d0_size, 1]

    int64_t ds3, ds2, ds1, ds0;  // sizes
    int64_t st3, st2, st1;        // strides (st0 = 1 always)
    if (d0_reps == 1) {
      ds3 = 1;             ds2 = num_chunks;          ds1 = num_cores;
      ds0 = chunk_size;
      st3 = 0;             st2 = chunk_size;           st1 = per_core_elements;
    } else {
      ds3 = num_chunks;    ds2 = num_cores;            ds1 = d0_reps;
      ds0 = d0_size;
      st3 = chunk_size;    st2 = per_core_elements;    st1 = d0_size;
    }

    // Input DMAs: one per input, all through single shim's BD space.
    for (int i = 0; i < program.num_inputs; i++) {
      absl::StrAppend(&seq, absl::StrFormat(
          "      aiex.npu.dma_memcpy_nd(%%arg%d[0, 0, 0, 0]"
          "[%d, %d, %d, %d][%d, %d, %d, 1]) "
          "{id = %d : i64, metadata = @in%d_L2} : %s\n",
          i,
          ds3, ds2, ds1, ds0,
          st3, st2, st1,
          i, i, host_memref_ty));
    }

    // Output DMA with issue_token for completion tracking.
    int out_arg = program.num_inputs;
    absl::StrAppend(&seq, absl::StrFormat(
        "      aiex.npu.dma_memcpy_nd(%%arg%d[0, 0, 0, 0]"
        "[%d, %d, %d, %d][%d, %d, %d, 1]) "
        "{id = %d : i64, metadata = @out0_L2, issue_token = true} : %s\n",
        out_arg,
        ds3, ds2, ds1, ds0,
        st3, st2, st1,
        program.num_inputs, host_memref_ty));

    // Single dma_wait for the one output FIFO.
    absl::StrAppend(&seq, "      aiex.npu.dma_wait { symbol = @out0_L2 }\n");
  } else {
    // Per-column independent: each column gets its own DMA commands.
    // Strides: set to 0 when the corresponding dim has size 1 (unused),
    // to satisfy the aie-opt verifier on rank-1 memrefs.
    int64_t s1 = (d0_reps > 1) ? d0_size : 0;
    int64_t s2 = (num_chunks > 1) ? chunk_size : 0;

    // 4D addressing: [1, num_chunks, d0_reps, d0_size]
    //   d0: innermost contiguous transfer (≤ 1023 32-bit words)
    //   d1: sub-chunks within a tile chunk (stride = d0_size)
    //   d2: tile chunks (stride = chunk_size)
    // Each column's DMA starts at offset c * per_core_elements into the
    // host buffer, selecting that core's slice of the tensor.
    for (int c = 0; c < num_cores; c++) {
      int64_t offset = c * per_core_elements;

      // Input DMAs for this column.
      // BD IDs are per-core (0..num_inputs), not global. Each column's shim
      // has its own BD space, so different columns can reuse the same IDs.
      for (int i = 0; i < program.num_inputs; i++) {
        int dma_id = i;
        std::string meta = fifo_name(c, absl::StrFormat("in%d%s", i, l2));
        absl::StrAppend(&seq, absl::StrFormat(
            "      aiex.npu.dma_memcpy_nd(%%arg%d[0, 0, 0, %d]"
            "[1, %d, %d, %d][0, %d, %d, 1]) "
            "{id = %d : i64, metadata = @%s} : %s\n",
            i, offset,
            num_chunks, d0_reps, d0_size,
            s2, s1,
            dma_id, meta, host_memref_ty));
      }

      // Output DMA for this column with issue_token for completion tracking.
      int out_arg = program.num_inputs;
      int out_dma_id = program.num_inputs;
      std::string out_meta = fifo_name(c, absl::StrFormat("out0%s", l2));
      absl::StrAppend(&seq, absl::StrFormat(
          "      aiex.npu.dma_memcpy_nd(%%arg%d[0, 0, 0, %d]"
          "[1, %d, %d, %d][0, %d, %d, 1]) "
          "{id = %d : i64, metadata = @%s, issue_token = true} : %s\n",
          out_arg, offset,
          num_chunks, d0_reps, d0_size,
          s2, s1,
          out_dma_id, out_meta, host_memref_ty));
    }

    // Wait for ALL columns' output DMA completion.
    for (int c = 0; c < num_cores; c++) {
      std::string out_meta = fifo_name(c, absl::StrFormat("out0%s", l2));
      absl::StrAppend(&seq, absl::StrFormat(
          "      aiex.npu.dma_wait { symbol = @%s }\n", out_meta));
    }
  }

  absl::StrAppend(&seq, "    }\n");

  return seq;
}

// Generates a vectorized matmul core body using hardware MAC intrinsics.
// Emits aievec.matmul_aie2p directly (4x8x8 shape: A: 4x8 bf16, B: 8x8 bf16,
// C: 4x8 f32). The 4x8x8 shape naturally fills ACC1024 (32 f32) without
// padding, matching mlir-aie reference kernels. We emit the aievec op directly
// rather than vector.contract because --convert-vector-to-aievec also converts
// vector.transfer_read into broken half-register aievec.upd loads. By emitting
// aievec.matmul_aie2p ourselves, we only need --convert-aievec-to-llvm (which
// correctly lowers matmul_aie2p → xllvm intrinsics) while vector.transfer_read
// /write are handled by --convert-vector-to-llvm.
// bf16 only — k=8 fixed (MAC width), n=8 (MAC output width).
// The K-tile loop is kept as scf.for to minimize code size. Each iteration
// acquires A/B tiles, performs MAC over 4-row blocks, then releases.
// The objectFifo-stateful-transform pass handles ping-pong buffering.
std::string GenerateVectorizedMatmulCoreBody(
    const MatmulProgramInfo& info, const MatmulTileConfig& tile,
    int64_t Nt_per_core, const std::string& fifo_prefix,
    const std::string& acc_name) {
  std::string body;

  int64_t Mt = info.M / tile.m;
  int64_t Kt = info.K / tile.k;

  // 2D memref types (for ObjectFIFO acquire/release).
  std::string a_memref = absl::StrFormat("memref<%dx%dxbf16>", tile.m, tile.k);
  std::string b_memref = absl::StrFormat("memref<%dx%dxbf16>", tile.k, tile.n);
  std::string c_memref = absl::StrFormat("memref<%dx%dxbf16>", tile.m, tile.n);
  std::string acc_memref = absl::StrFormat("memref<%dx%dxf32>", tile.m, tile.n);

  // 1D flattened memref types (for vector.transfer_read/write).
  // convert-vector-to-llvm can't lower 2D transfers; we flatten to 1D.
  int64_t a_flat_size = tile.m * tile.k;
  int64_t b_flat_size = tile.k * tile.n;
  int64_t acc_flat_size = tile.m * tile.n;
  std::string a_flat = absl::StrFormat("memref<%dxbf16>", a_flat_size);
  std::string b_flat = absl::StrFormat("memref<%dxbf16>", b_flat_size);
  std::string acc_flat = absl::StrFormat("memref<%dxf32>", acc_flat_size);

  // Vector types (1D for transfers, 2D for MAC).
  int64_t mac_m = 4, mac_n = 8, mac_k = 8;
  int64_t acc_vec_1d = mac_m * mac_n;   // 32
  int64_t a_vec_1d = mac_m * mac_k;     // 32
  int64_t b_vec_1d = mac_k * mac_n;     // 64

  // Constants.
  absl::StrAppend(&body,
      "      %c0 = arith.constant 0 : index\n"
      "      %c1 = arith.constant 1 : index\n"
      "      %c4 = arith.constant 4 : index\n");
  absl::StrAppend(&body, absl::StrFormat(
      "      %%cm = arith.constant %d : index\n", tile.m));
  absl::StrAppend(&body, absl::StrFormat(
      "      %%cn = arith.constant %d : index\n", tile.n));
  absl::StrAppend(&body, absl::StrFormat(
      "      %%ck = arith.constant %d : index\n", tile.k));
  absl::StrAppend(&body,
      "      %pad_f32 = arith.constant 0.0 : f32\n"
      "      %pad_bf16 = arith.constant 0.0 : bf16\n");
  int64_t total_out_tiles = Mt * Nt_per_core;
  absl::StrAppend(&body, absl::StrFormat(
      "      %%n_out_tiles = arith.constant %d : index\n", total_out_tiles));

  // Flatten accumulator memref to 1D for vector transfers.
  absl::StrAppend(&body, absl::StrFormat(
      "      %%flat_acc = memref.collapse_shape %%%s [[0, 1]]"
      " : %s into %s\n", acc_name, acc_memref, acc_flat));

  // Outer loop over output tiles.
  absl::StrAppend(&body,
      "      scf.for %ot = %c0 to %n_out_tiles step %c1 {\n");

  // Acquire C (Produce).
  absl::StrAppend(&body, absl::StrFormat(
      "        %%subview_c = aie.objectfifo.acquire @%soutC(Produce, 1) "
      ": !aie.objectfifosubview<%s>\n", fifo_prefix, c_memref));
  absl::StrAppend(&body, absl::StrFormat(
      "        %%elem_c = aie.objectfifo.subview.access %%subview_c[0] "
      ": !aie.objectfifosubview<%s> -> %s\n", c_memref, c_memref));

  // Zero the f32 accumulator buffer with scalar stores.
  // Using scalar stores instead of vector.transfer_write to avoid memset
  // at llc -O0 (required to work around Peano VLIW scheduler bugs).
  absl::StrAppend(&body,
      "        scf.for %zi = %c0 to %cm step %c1 {\n"
      "          scf.for %zj = %c0 to %cn step %c1 {\n");
  absl::StrAppend(&body, absl::StrFormat(
      "            memref.store %%pad_f32, %%%s[%%zi, %%zj] : %s\n",
      acc_name, acc_memref));
  absl::StrAppend(&body,
      "          }\n"
      "        }\n");

  // K-tile loop: kept as scf.for to minimize code size.
  // Each iteration acquires one A/B tile, does MAC, releases.
  // The objectFifo-stateful-transform pass handles ping-pong buffering.
  absl::StrAppend(&body, absl::StrFormat(
      "        %%n_k_tiles = arith.constant %d : index\n", Kt));
  absl::StrAppend(&body,
      "        scf.for %kt = %c0 to %n_k_tiles step %c1 {\n");

  {
    // Acquire A and B (Consume), flatten to 1D.
    absl::StrAppend(&body, absl::StrFormat(
        "          %%subview_a = aie.objectfifo.acquire @%sinA(Consume, 1) "
        ": !aie.objectfifosubview<%s>\n", fifo_prefix, a_memref));
    absl::StrAppend(&body, absl::StrFormat(
        "          %%elem_a = aie.objectfifo.subview.access %%subview_a[0] "
        ": !aie.objectfifosubview<%s> -> %s\n", a_memref, a_memref));
    absl::StrAppend(&body, absl::StrFormat(
        "          %%flat_a = memref.collapse_shape %%elem_a [[0, 1]]"
        " : %s into %s\n", a_memref, a_flat));
    absl::StrAppend(&body, absl::StrFormat(
        "          %%subview_b = aie.objectfifo.acquire @%sinB(Consume, 1) "
        ": !aie.objectfifosubview<%s>\n", fifo_prefix, b_memref));
    absl::StrAppend(&body, absl::StrFormat(
        "          %%elem_b = aie.objectfifo.subview.access %%subview_b[0] "
        ": !aie.objectfifosubview<%s> -> %s\n", b_memref, b_memref));
    absl::StrAppend(&body, absl::StrFormat(
        "          %%flat_b = memref.collapse_shape %%elem_b [[0, 1]]"
        " : %s into %s\n", b_memref, b_flat));

    // Inner MAC loop over 4-row blocks.
    absl::StrAppend(&body,
        "          scf.for %mi = %c0 to %cm step %c4 {\n");

    // Load accumulator from memory.
    absl::StrAppend(&body,
        "            %acc_off = arith.muli %mi, %cn : index\n");
    absl::StrAppend(&body, absl::StrFormat(
        "            %%acc_1d = vector.transfer_read %%flat_acc[%%acc_off], %%pad_f32"
        " {in_bounds = [true]} : %s, vector<%dxf32>\n",
        acc_flat.c_str(), acc_vec_1d));
    absl::StrAppend(&body, absl::StrFormat(
        "            %%acc_init = vector.shape_cast %%acc_1d"
        " : vector<%dxf32> to vector<%dx%dxf32>\n",
        acc_vec_1d, mac_m, mac_n));

    // A offset: mi * k (flat index into m×k tile). k=8 so contiguous read works.
    absl::StrAppend(&body,
        "            %a_off = arith.muli %mi, %ck : index\n");
    absl::StrAppend(&body, absl::StrFormat(
        "            %%a_1d = vector.transfer_read %%flat_a[%%a_off], %%pad_bf16"
        " {in_bounds = [true]} : %s, vector<%dxbf16>\n",
        a_flat.c_str(), a_vec_1d));
    absl::StrAppend(&body, absl::StrFormat(
        "            %%a_tile = vector.shape_cast %%a_1d"
        " : vector<%dxbf16> to vector<%dx%dxbf16>\n",
        a_vec_1d, mac_m, mac_k));

    // B offset: 0 (entire 8×8 tile).
    absl::StrAppend(&body, absl::StrFormat(
        "            %%b_1d = vector.transfer_read %%flat_b[%%c0], %%pad_bf16"
        " {in_bounds = [true]} : %s, vector<%dxbf16>\n",
        b_flat.c_str(), b_vec_1d));
    absl::StrAppend(&body, absl::StrFormat(
        "            %%b_tile = vector.shape_cast %%b_1d"
        " : vector<%dxbf16> to vector<%dx%dxbf16>\n",
        b_vec_1d, mac_k, mac_n));

    // Hardware MAC.
    absl::StrAppend(&body, absl::StrFormat(
        "            %%mac_result = aievec.matmul_aie2p"
        " %%a_tile, %%b_tile, %%acc_init"
        " : vector<%dx%dxbf16>, vector<%dx%dxbf16> into vector<%dx%dxf32>\n",
        mac_m, mac_k, mac_k, mac_n, mac_m, mac_n));

    // Store accumulator back to memory.
    absl::StrAppend(&body, absl::StrFormat(
        "            %%result_1d = vector.shape_cast %%mac_result"
        " : vector<%dx%dxf32> to vector<%dxf32>\n",
        mac_m, mac_n, acc_vec_1d));
    absl::StrAppend(&body, absl::StrFormat(
        "            vector.transfer_write %%result_1d, %%flat_acc[%%acc_off]"
        " {in_bounds = [true]}"
        " : vector<%dxf32>, %s\n",
        acc_vec_1d, acc_flat.c_str()));

    // Close mi loop.
    absl::StrAppend(&body,
        "          }\n");

    // Release A and B.
    absl::StrAppend(&body, absl::StrFormat(
        "          aie.objectfifo.release @%sinA(Consume, 1)\n"
        "          aie.objectfifo.release @%sinB(Consume, 1)\n",
        fifo_prefix, fifo_prefix));
  }

  // Close K-tile loop.
  absl::StrAppend(&body,
      "        }\n");

  // Truncate f32 accumulator → bf16 into C buffer (scalar).
  // Peano (Jan 2025) cannot legalize fptrunc on any vector type, so we
  // must truncate element-by-element.
  absl::StrAppend(&body,
      "        scf.for %ci = %c0 to %cm step %c1 {\n"
      "          scf.for %cj = %c0 to %cn step %c1 {\n");
  absl::StrAppend(&body, absl::StrFormat(
      "            %%acc_val = memref.load %%%s[%%ci, %%cj] : %s\n",
      acc_name, acc_memref));
  absl::StrAppend(&body,
      "            %bf16_val = arith.truncf %acc_val : f32 to bf16\n");
  absl::StrAppend(&body, absl::StrFormat(
      "            memref.store %%bf16_val, %%elem_c[%%ci, %%cj] : %s\n",
      c_memref));
  absl::StrAppend(&body,
      "          }\n"
      "        }\n");

  // Release C.
  absl::StrAppend(&body, absl::StrFormat(
      "        aie.objectfifo.release @%soutC(Produce, 1)\n", fifo_prefix));

  // Close output tile loop.
  absl::StrAppend(&body,
      "      }\n");

  return body;
}

// Generates the matmul core body for AIE dialect.
// Pattern: for each output tile (M/m * N/n iterations):
//   acquire C → zero accumulator → for kt in K/k: acquire A,B → MAC → release
//   → [bf16: truncate acc→C] → release C
std::string GenerateMatmulCoreBody(
    const MatmulProgramInfo& info, const MatmulTileConfig& tile,
    int64_t Nt_per_core, const std::string& fifo_prefix,
    const std::string& acc_name) {
  std::string body;
  const std::string& ty = info.element_type;
  bool is_bf16 = (ty == "bf16");

  int64_t Mt = info.M / tile.m;
  int64_t Kt = info.K / tile.k;

  std::string a_memref = absl::StrFormat("memref<%dx%dx%s>", tile.m, tile.k, ty);
  // When b_transposed, DMA transfers B_orig tiles as n×k (not k×n).
  std::string b_memref = info.b_transposed
      ? absl::StrFormat("memref<%dx%dx%s>", tile.n, tile.k, ty)
      : absl::StrFormat("memref<%dx%dx%s>", tile.k, tile.n, ty);
  std::string c_memref = absl::StrFormat("memref<%dx%dx%s>", tile.m, tile.n, ty);
  std::string acc_memref = absl::StrFormat("memref<%dx%dxf32>", tile.m, tile.n);

  // Constants.
  absl::StrAppend(&body,
      "      %c0 = arith.constant 0 : index\n"
      "      %c1 = arith.constant 1 : index\n");
  absl::StrAppend(&body, absl::StrFormat(
      "      %%cm = arith.constant %d : index\n", tile.m));
  absl::StrAppend(&body, absl::StrFormat(
      "      %%cn = arith.constant %d : index\n", tile.n));
  absl::StrAppend(&body, absl::StrFormat(
      "      %%ck = arith.constant %d : index\n", tile.k));
  if (is_bf16) {
    absl::StrAppend(&body,
        "      %zero_f32 = arith.constant 0.0 : f32\n");
  } else {
    absl::StrAppend(&body, absl::StrFormat(
        "      %%zero_%s = arith.constant 0.0 : %s\n", ty, ty));
  }

  int64_t total_out_tiles = Mt * Nt_per_core;
  absl::StrAppend(&body, absl::StrFormat(
      "      %%n_out_tiles = arith.constant %d : index\n", total_out_tiles));
  absl::StrAppend(&body, absl::StrFormat(
      "      %%n_k_tiles = arith.constant %d : index\n", Kt));

  // Outer loop over output tiles.
  absl::StrAppend(&body,
      "      scf.for %ot = %c0 to %n_out_tiles step %c1 {\n");

  // Acquire C (Produce).
  absl::StrAppend(&body, absl::StrFormat(
      "        %%subview_c = aie.objectfifo.acquire @%soutC(Produce, 1) "
      ": !aie.objectfifosubview<%s>\n", fifo_prefix, c_memref));
  absl::StrAppend(&body, absl::StrFormat(
      "        %%elem_c = aie.objectfifo.subview.access %%subview_c[0] "
      ": !aie.objectfifosubview<%s> -> %s\n", c_memref, c_memref));

  // Zero the accumulator.
  if (is_bf16) {
    // Zero the f32 accumulator buffer.
    absl::StrAppend(&body,
        "        scf.for %zi = %c0 to %cm step %c1 {\n"
        "          scf.for %zj = %c0 to %cn step %c1 {\n");
    absl::StrAppend(&body, absl::StrFormat(
        "            memref.store %%zero_f32, %%%s[%%zi, %%zj] : %s\n",
        acc_name, acc_memref));
    absl::StrAppend(&body,
        "          }\n"
        "        }\n");
  } else {
    // Zero the C buffer directly.
    absl::StrAppend(&body,
        "        scf.for %zi = %c0 to %cm step %c1 {\n"
        "          scf.for %zj = %c0 to %cn step %c1 {\n");
    absl::StrAppend(&body, absl::StrFormat(
        "            memref.store %%zero_%s, %%elem_c[%%zi, %%zj] : %s\n",
        ty, c_memref));
    absl::StrAppend(&body,
        "          }\n"
        "        }\n");
  }

  // K-tile loop.
  absl::StrAppend(&body,
      "        scf.for %kt = %c0 to %n_k_tiles step %c1 {\n");

  // Acquire A and B (Consume).
  absl::StrAppend(&body, absl::StrFormat(
      "          %%subview_a = aie.objectfifo.acquire @%sinA(Consume, 1) "
      ": !aie.objectfifosubview<%s>\n", fifo_prefix, a_memref));
  absl::StrAppend(&body, absl::StrFormat(
      "          %%elem_a = aie.objectfifo.subview.access %%subview_a[0] "
      ": !aie.objectfifosubview<%s> -> %s\n", a_memref, a_memref));
  absl::StrAppend(&body, absl::StrFormat(
      "          %%subview_b = aie.objectfifo.acquire @%sinB(Consume, 1) "
      ": !aie.objectfifosubview<%s>\n", fifo_prefix, b_memref));
  absl::StrAppend(&body, absl::StrFormat(
      "          %%elem_b = aie.objectfifo.subview.access %%subview_b[0] "
      ": !aie.objectfifosubview<%s> -> %s\n", b_memref, b_memref));

  // Triple nested MAC loop.
  absl::StrAppend(&body,
      "          scf.for %i = %c0 to %cm step %c1 {\n"
      "            scf.for %j = %c0 to %cn step %c1 {\n"
      "              scf.for %p = %c0 to %ck step %c1 {\n");

  // B access: [%p, %j] for normal, [%j, %p] for transposed (B_orig[N][K] in L1).
  std::string b_idx0 = info.b_transposed ? "%j" : "%p";
  std::string b_idx1 = info.b_transposed ? "%p" : "%j";

  if (is_bf16) {
    // bf16: native multiply, f32 accumulator.
    absl::StrAppend(&body, absl::StrFormat(
        "                %%a_val = memref.load %%elem_a[%%i, %%p] : %s\n",
        a_memref));
    absl::StrAppend(&body, absl::StrFormat(
        "                %%b_val = memref.load %%elem_b[%s, %s] : %s\n",
        b_idx0, b_idx1, b_memref));
    absl::StrAppend(&body,
        "                %prod_bf = arith.mulf %a_val, %b_val : bf16\n"
        "                %prod_f32 = arith.extf %prod_bf : bf16 to f32\n");
    absl::StrAppend(&body, absl::StrFormat(
        "                %%c_old = memref.load %%%s[%%i, %%j] : %s\n",
        acc_name, acc_memref));
    absl::StrAppend(&body,
        "                %c_new = arith.addf %c_old, %prod_f32 : f32\n");
    absl::StrAppend(&body, absl::StrFormat(
        "                memref.store %%c_new, %%%s[%%i, %%j] : %s\n",
        acc_name, acc_memref));
  } else {
    // f32: truncate→bf16 multiply→extend workaround, accumulate in C.
    absl::StrAppend(&body, absl::StrFormat(
        "                %%a_val = memref.load %%elem_a[%%i, %%p] : %s\n",
        a_memref));
    absl::StrAppend(&body, absl::StrFormat(
        "                %%b_val = memref.load %%elem_b[%s, %s] : %s\n",
        b_idx0, b_idx1, b_memref));
    absl::StrAppend(&body,
        "                %a_bf = arith.truncf %a_val : f32 to bf16\n"
        "                %b_bf = arith.truncf %b_val : f32 to bf16\n"
        "                %prod_bf = arith.mulf %a_bf, %b_bf : bf16\n"
        "                %prod_f32 = arith.extf %prod_bf : bf16 to f32\n");
    absl::StrAppend(&body, absl::StrFormat(
        "                %%c_old = memref.load %%elem_c[%%i, %%j] : %s\n",
        c_memref));
    absl::StrAppend(&body,
        "                %c_new = arith.addf %c_old, %prod_f32 : f32\n");
    absl::StrAppend(&body, absl::StrFormat(
        "                memref.store %%c_new, %%elem_c[%%i, %%j] : %s\n",
        c_memref));
  }

  // Close triple loop.
  absl::StrAppend(&body,
      "              }\n"
      "            }\n"
      "          }\n");

  // Release A and B.
  absl::StrAppend(&body, absl::StrFormat(
      "          aie.objectfifo.release @%sinA(Consume, 1)\n"
      "          aie.objectfifo.release @%sinB(Consume, 1)\n",
      fifo_prefix, fifo_prefix));

  // Close K-tile loop.
  absl::StrAppend(&body,
      "        }\n");

  // bf16: truncate f32 accumulator → bf16 into C buffer.
  if (is_bf16) {
    absl::StrAppend(&body,
        "        scf.for %ci = %c0 to %cm step %c1 {\n"
        "          scf.for %cj = %c0 to %cn step %c1 {\n");
    absl::StrAppend(&body, absl::StrFormat(
        "            %%acc_val = memref.load %%%s[%%ci, %%cj] : %s\n",
        acc_name, acc_memref));
    absl::StrAppend(&body,
        "            %acc_bf = arith.truncf %acc_val : f32 to bf16\n");
    absl::StrAppend(&body, absl::StrFormat(
        "            memref.store %%acc_bf, %%elem_c[%%ci, %%cj] : %s\n",
        c_memref));
    absl::StrAppend(&body,
        "          }\n"
        "        }\n");
  }

  // Release C.
  absl::StrAppend(&body, absl::StrFormat(
      "        aie.objectfifo.release @%soutC(Produce, 1)\n", fifo_prefix));

  // Close output tile loop.
  absl::StrAppend(&body,
      "      }\n");

  return body;
}

// Generates the NPU DMA sequence for matmul.
//
// Data layout: A[M,K] row-major, B[K,N] row-major, C[M,N] row-major.
// Host memrefs are 1D (flat). We unroll the M-tile loop, emitting one DMA
// command block per mt value. Within each block, 4D addressing covers
// N-tiles and K-chunks.
//
// Core consumes tiles in order: for each output tile (mt, nt):
//   for kt in K/k: A[mt*m..mt*m+m, kt*k..kt*k+k], B[kt*k..kt*k+k, nt*n..nt*n+n]
//   then produces C[mt*m..mt*m+m, nt*n..nt*n+n]
//
// So the total order of tile consumption is:
//   for mt in M/m:
//     for nt in N/n:
//       for kt in K/k: one A tile, one B tile
//     one C tile
//
// DMA must deliver tiles in exactly this order.
absl::StatusOr<std::string> GenerateMatmulNpuSequence(
    const MatmulProgramInfo& info, const MatmulTileConfig& tile,
    const TargetCaps& caps, int num_cores) {
  const std::string& ty = info.element_type;
  int64_t Mt = info.M / tile.m;
  int64_t Nt = info.N / tile.n;
  int64_t Kt = info.K / tile.k;
  int64_t Nt_per_core = Nt / num_cores;

  int element_bytes = (ty == "bf16") ? 2 : 4;

  std::string a_host_memref = absl::StrFormat("memref<%dx%s>",
      info.M * info.K, ty);
  std::string b_host_memref = absl::StrFormat("memref<%dx%s>",
      info.K * info.N, ty);
  std::string c_host_memref = absl::StrFormat("memref<%dx%s>",
      info.M * info.N, ty);

  // FIFO name helper: multi-core uses "c{col}_" prefix, single-core uses "".
  auto fifo_name = [&](int c, const std::string& base) -> std::string {
    return num_cores > 1 ? absl::StrFormat("c%d_%s", c, base) : base;
  };

  std::string seq;
  absl::StrAppend(&seq, absl::StrFormat(
      "    aie.runtime_sequence(%%arg0: %s, %%arg1: %s, %%arg2: %s) {\n",
      a_host_memref, b_host_memref, c_host_memref));

  // DMA limit checks (same for all cores — tile sizes are identical).
  int64_t a_d0_words = tile.k * element_bytes / 4;
  int64_t a_d1 = tile.m;
  if (a_d0_words > kMaxBdWords || a_d1 > kMaxBdWords) {
    return absl::InvalidArgumentError(absl::StrFormat(
        "Matmul A DMA exceeds BD limits: d0=%d words, d1=%d (max 1023)",
        a_d0_words, a_d1));
  }
  // For transposed B, DMA reads tiles as n×k (not k×n).
  int64_t b_d0_words = (info.b_transposed ? tile.k : tile.n) * element_bytes / 4;
  int64_t b_d1 = info.b_transposed ? tile.n : tile.k;
  if (b_d0_words > kMaxBdWords || b_d1 > kMaxBdWords) {
    return absl::InvalidArgumentError(absl::StrFormat(
        "Matmul B DMA exceeds BD limits: d0=%d words, d1=%d (max 1023)",
        b_d0_words, b_d1));
  }
  int64_t c_d0_words = tile.n * element_bytes / 4;
  int64_t c_d1 = tile.m;
  if (c_d0_words > kMaxBdWords || c_d1 > kMaxBdWords) {
    return absl::InvalidArgumentError(absl::StrFormat(
        "Matmul C DMA exceeds BD limits: d0=%d words, d1=%d (max 1023)",
        c_d0_words, c_d1));
  }
  if (Nt_per_core > 64) {
    return absl::InvalidArgumentError(absl::StrFormat(
        "Matmul DMA d3=%d exceeds limit 64", Nt_per_core));
  }

  for (int64_t mt = 0; mt < Mt; mt++) {
    // Reset BD IDs each mt block. After the dma_wait barrier at the end of
    // each block, all BDs are complete and their slots can be reused.
    // The hardware has only 16 BD slots (4-bit ID field: 0-15).
    int dma_id = 0;

    // Per-core DMAs for this M-tile row.
    // A is broadcast (same rows for all cores), B and C are column-partitioned.
    for (int c = 0; c < num_cores; c++) {
      // A DMA: [Nt_per_core, Kt, m, k] strides [0, k, K, 1]
      //   d3=Nt_per_core: repeat for each N-tile (stride 0 = broadcast)
      //   d2=Kt: iterate over K-tiles (stride k elements)
      //   d1=m: rows of tile (stride K = row stride in A)
      //   d0=k: elements per row (stride 1)
      int64_t a_offset = mt * tile.m * info.K;
      absl::StrAppend(&seq, absl::StrFormat(
          "      aiex.npu.dma_memcpy_nd(%%arg0"
          "[0, 0, 0, %d][%d, %d, %d, %d][0, %d, %d, 1]) "
          "{id = %d : i64, metadata = @%s, issue_token = true} : %s\n",
          a_offset,
          Nt_per_core, Kt, tile.m, tile.k,
          tile.k, info.K,
          dma_id++, fifo_name(c, "inA_L2"), a_host_memref));

      // B DMA: depends on whether B is transposed.
      // Normal:     B[K,N] row-major → tiles of k×n
      //   [Nt_per_core, Kt, k, n] strides [n, k*N, N, 1]
      // Transposed: B_orig[N,K] row-major → tiles of n×k
      //   [Nt_per_core, Kt, n, k] strides [n*K, k, K, 1]
      // Each core reads a different column slice of B.
      if (info.b_transposed) {
        // B_orig[N,K]: each row is K elements wide.
        // Tile (nt, kt): n rows starting at row nt*n, k cols starting at col kt*k.
        int64_t b_offset = c * Nt_per_core * tile.n * info.K;
        absl::StrAppend(&seq, absl::StrFormat(
            "      aiex.npu.dma_memcpy_nd(%%arg1"
            "[0, 0, 0, %d][%d, %d, %d, %d][%d, %d, %d, 1]) "
            "{id = %d : i64, metadata = @%s, issue_token = true} : %s\n",
            b_offset,
            Nt_per_core, Kt, tile.n, tile.k,
            tile.n * info.K, tile.k, info.K,
            dma_id++, fifo_name(c, "inB_L2"), b_host_memref));
      } else {
        int64_t b_offset = c * Nt_per_core * tile.n;
        absl::StrAppend(&seq, absl::StrFormat(
            "      aiex.npu.dma_memcpy_nd(%%arg1"
            "[0, 0, 0, %d][%d, %d, %d, %d][%d, %d, %d, 1]) "
            "{id = %d : i64, metadata = @%s, issue_token = true} : %s\n",
            b_offset,
            Nt_per_core, Kt, tile.k, tile.n,
            tile.n, tile.k * info.N, info.N,
            dma_id++, fifo_name(c, "inB_L2"), b_host_memref));
      }

      // C DMA: [1, Nt_per_core, m, n] strides [0, n, N, 1]
      // Each core writes a different column slice of C.
      int64_t c_offset = mt * tile.m * info.N + c * Nt_per_core * tile.n;
      absl::StrAppend(&seq, absl::StrFormat(
          "      aiex.npu.dma_memcpy_nd(%%arg2"
          "[0, 0, 0, %d][1, %d, %d, %d][0, %d, %d, 1]) "
          "{id = %d : i64, metadata = @%s, issue_token = true} : %s\n",
          c_offset,
          Nt_per_core, tile.m, tile.n,
          tile.n, info.N,
          dma_id++, fifo_name(c, "outC_L2"), c_host_memref));
    }

    // Wait for ALL cores' DMAs before next mt block.
    // This ensures all DMA channel states are fully reset before the
    // next block's BDs are submitted.
    for (int c = 0; c < num_cores; c++) {
      absl::StrAppend(&seq, absl::StrFormat(
          "      aiex.npu.dma_wait { symbol = @%s }\n",
          fifo_name(c, "outC_L2")));
      absl::StrAppend(&seq, absl::StrFormat(
          "      aiex.npu.dma_wait { symbol = @%s }\n",
          fifo_name(c, "inA_L2")));
      absl::StrAppend(&seq, absl::StrFormat(
          "      aiex.npu.dma_wait { symbol = @%s }\n",
          fifo_name(c, "inB_L2")));
    }
  }
  absl::StrAppend(&seq, "    }\n");

  return seq;
}

// Lowers a matmul to AIE dialect MLIR (multi-core parallel on N dimension).
absl::StatusOr<AieLoweringResult> LowerMatmulToAieInternal(
    const MatmulProgramInfo& info, const TargetCaps& caps,
    int max_columns) {
  // Auto-scale number of cores. max_columns is already clamped by the
  // compiler (respects XDNA_NUM_CORES env var and caps.num_columns).
  int max_cores = max_columns;

  int num_cores = max_cores;
  MatmulTileConfig tile;
  while (num_cores >= 1) {
    auto tile_or = SelectMatmulTiles(info, caps, num_cores);
    if (tile_or.ok()) { tile = *tile_or; break; }
    num_cores--;
  }
  if (num_cores < 1) {
    return absl::InvalidArgumentError(absl::StrFormat(
        "Cannot tile matmul [%d,%d]@[%d,%d] %s for any core count 1-%d",
        info.M, info.K, info.K, info.N, info.element_type, max_cores));
  }

  bool is_bf16 = (info.element_type == "bf16");
  bool use_vectorized = is_bf16 && tile.k == 8;
  int64_t Nt = info.N / tile.n;
  int64_t Nt_per_core = Nt / num_cores;
  LOG(INFO) << "XDNA matmul lowering: tiles m=" << tile.m << " k=" << tile.k
            << " n=" << tile.n << " (M=" << info.M << " K=" << info.K
            << " N=" << info.N << " " << info.element_type
            << (use_vectorized ? " VECTORIZED" : " scalar")
            << (info.b_transposed ? " B_TRANSPOSED" : "")
            << ") using " << num_cores << " core(s)";
  const std::string& ty = info.element_type;
  int start_col = caps.partition_start_column;
  int shim_row = caps.shim_row;
  int compute_row = caps.first_compute_row;

  std::string a_memref = absl::StrFormat("memref<%dx%dx%s>", tile.m, tile.k, ty);
  // Transposed B: DMA transfers tiles as n×k (original B_orig layout).
  std::string b_memref = info.b_transposed
      ? absl::StrFormat("memref<%dx%dx%s>", tile.n, tile.k, ty)
      : absl::StrFormat("memref<%dx%dx%s>", tile.k, tile.n, ty);
  std::string c_memref = absl::StrFormat("memref<%dx%dx%s>", tile.m, tile.n, ty);

  // FIFO name helper: multi-core uses "c{col}_" prefix, single-core uses "".
  auto fifo_name = [&](int c, const std::string& base) -> std::string {
    return num_cores > 1 ? absl::StrFormat("c%d_%s", c, base) : base;
  };
  auto fifo_prefix = [&](int c) -> std::string {
    return num_cores > 1 ? absl::StrFormat("c%d_", c) : "";
  };
  auto acc_name = [&](int c) -> std::string {
    return num_cores > 1 ? absl::StrFormat("acc_%d", c) : "acc";
  };

  int fifo_depth = 2;

  std::string aie_mlir;
  absl::StrAppend(&aie_mlir, "module {\n");
  absl::StrAppend(&aie_mlir,
      absl::StrFormat("  aie.device(%s) {\n",
                      DeviceNameForColumns(num_cores)));

  // Pass 1: All tile declarations.
  for (int c = 0; c < num_cores; c++) {
    int col = start_col + c;
    absl::StrAppend(&aie_mlir, absl::StrFormat(
        "    %%tile_%d_%d = aie.tile(%d, %d)\n",
        col, shim_row, col, shim_row));
    absl::StrAppend(&aie_mlir, absl::StrFormat(
        "    %%tile_%d_%d = aie.tile(%d, %d)\n",
        col, caps.mem_tile_row, col, caps.mem_tile_row));
    absl::StrAppend(&aie_mlir, absl::StrFormat(
        "    %%tile_%d_%d = aie.tile(%d, %d)\n",
        col, compute_row, col, compute_row));
  }

  // Pass 2: All ObjectFIFOs, links, and accumulator buffers.
  for (int c = 0; c < num_cores; c++) {
    int col = start_col + c;
    std::string shim_tile = absl::StrFormat("tile_%d_%d", col, shim_row);
    std::string mem_tile = absl::StrFormat("tile_%d_%d", col, caps.mem_tile_row);
    std::string compute_tile = absl::StrFormat("tile_%d_%d", col, compute_row);

    // ObjectFIFOs: shim ↔ mem tile (L2).
    absl::StrAppend(&aie_mlir, absl::StrFormat(
        "    aie.objectfifo @%s(%%%s, {%%%s}, %d : i32) "
        ": !aie.objectfifo<%s>\n",
        fifo_name(c, "inA_L2"), shim_tile, mem_tile, fifo_depth, a_memref));
    absl::StrAppend(&aie_mlir, absl::StrFormat(
        "    aie.objectfifo @%s(%%%s, {%%%s}, %d : i32) "
        ": !aie.objectfifo<%s>\n",
        fifo_name(c, "inB_L2"), shim_tile, mem_tile, fifo_depth, b_memref));
    absl::StrAppend(&aie_mlir, absl::StrFormat(
        "    aie.objectfifo @%s(%%%s, {%%%s}, %d : i32) "
        ": !aie.objectfifo<%s>\n",
        fifo_name(c, "outC_L2"), mem_tile, shim_tile, fifo_depth, c_memref));

    // ObjectFIFOs: mem tile ↔ compute (L1).
    absl::StrAppend(&aie_mlir, absl::StrFormat(
        "    aie.objectfifo @%s(%%%s, {%%%s}, %d : i32) "
        ": !aie.objectfifo<%s>\n",
        fifo_name(c, "inA"), mem_tile, compute_tile, fifo_depth, a_memref));
    absl::StrAppend(&aie_mlir, absl::StrFormat(
        "    aie.objectfifo @%s(%%%s, {%%%s}, %d : i32) "
        ": !aie.objectfifo<%s>\n",
        fifo_name(c, "inB"), mem_tile, compute_tile, fifo_depth, b_memref));
    absl::StrAppend(&aie_mlir, absl::StrFormat(
        "    aie.objectfifo @%s(%%%s, {%%%s}, %d : i32) "
        ": !aie.objectfifo<%s>\n",
        fifo_name(c, "outC"), compute_tile, mem_tile, fifo_depth, c_memref));

    // Links through memory tile.
    absl::StrAppend(&aie_mlir, absl::StrFormat(
        "    aie.objectfifo.link [@%s] -> [@%s] ([] [])\n",
        fifo_name(c, "inA_L2"), fifo_name(c, "inA")));
    absl::StrAppend(&aie_mlir, absl::StrFormat(
        "    aie.objectfifo.link [@%s] -> [@%s] ([] [])\n",
        fifo_name(c, "inB_L2"), fifo_name(c, "inB")));
    absl::StrAppend(&aie_mlir, absl::StrFormat(
        "    aie.objectfifo.link [@%s] -> [@%s] ([] [])\n",
        fifo_name(c, "outC"), fifo_name(c, "outC_L2")));

    // Local accumulator buffer (bf16 only).
    if (is_bf16) {
      absl::StrAppend(&aie_mlir, absl::StrFormat(
          "    %%%s = aie.buffer(%%%s) {sym_name = \"%s\"} "
          ": memref<%dx%dxf32>\n",
          acc_name(c), compute_tile, acc_name(c), tile.m, tile.n));
    }
  }

  // Pass 3: All core bodies.
  for (int c = 0; c < num_cores; c++) {
    int col = start_col + c;
    std::string compute_tile = absl::StrFormat("tile_%d_%d", col, compute_row);

    absl::StrAppend(&aie_mlir, absl::StrFormat(
        "    %%core_%d_%d = aie.core(%%%s) {\n", col, compute_row,
        compute_tile));
    if (use_vectorized) {
      absl::StrAppend(&aie_mlir,
          GenerateVectorizedMatmulCoreBody(info, tile, Nt_per_core,
                                           fifo_prefix(c), acc_name(c)));
    } else {
      absl::StrAppend(&aie_mlir,
          GenerateMatmulCoreBody(info, tile, Nt_per_core,
                                 fifo_prefix(c), acc_name(c)));
    }
    absl::StrAppend(&aie_mlir,
        "      aie.end\n"
        "    }\n");
  }

  // NPU DMA sequence.
  auto npu_seq_or = GenerateMatmulNpuSequence(info, tile, caps, num_cores);
  if (!npu_seq_or.ok()) return npu_seq_or.status();
  absl::StrAppend(&aie_mlir, *npu_seq_or);

  absl::StrAppend(&aie_mlir, "  }\n");
  absl::StrAppend(&aie_mlir, "}\n");

  LOG(INFO) << "XDNA matmul lowering: generated AIE MLIR:\n" << aie_mlir;

  return AieLoweringResult{aie_mlir, num_cores,
                           /*use_aievec=*/use_vectorized,
                           /*convert_vector_to_aievec=*/false,
                           /*needs_matmul_workarounds=*/use_vectorized};
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

// Detects a matmul program in the module: linalg.fill + linalg.matmul.
// Returns MatmulProgramInfo if found, nullopt otherwise.
std::optional<MatmulProgramInfo> DetectMatmul(mlir::ModuleOp module) {
  mlir::func::FuncOp entry_func;
  module.walk([&](mlir::func::FuncOp func) {
    if (func.getName() == "main" || !entry_func) entry_func = func;
  });
  if (!entry_func) return std::nullopt;

  // Count linalg ops: we accept {fill, matmul} or {transpose_generic, fill, matmul}.
  int fill_count = 0;
  int matmul_count = 0;
  int generic_count = 0;
  int other_linalg_count = 0;
  mlir::Operation* matmul_op = nullptr;
  mlir::Operation* fill_op = nullptr;
  mlir::Operation* generic_op = nullptr;

  entry_func.walk([&](mlir::Operation* op) {
    llvm::StringRef name = op->getName().getStringRef();
    if (!name.starts_with("linalg.")) return;
    if (name == "linalg.yield") return;  // Skip terminators inside linalg regions.
    if (name == "linalg.fill") {
      fill_count++;
      fill_op = op;
    } else if (name == "linalg.matmul") {
      matmul_count++;
      matmul_op = op;
    } else if (name == "linalg.generic") {
      generic_count++;
      generic_op = op;
    } else {
      other_linalg_count++;
    }
  });

  if (matmul_count != 1 || other_linalg_count != 0) return std::nullopt;
  if (fill_count > 1) return std::nullopt;  // At most one fill
  if (generic_count > 1) return std::nullopt;  // At most one generic (transpose)

  // Validate fill value is zero — our lowering unconditionally zeroes the
  // accumulator, so a non-zero fill would be silently dropped.
  if (fill_count == 1 && fill_op) {
    // linalg.fill: operand 0 is the scalar fill value (DPS input).
    mlir::Value fill_val = fill_op->getOperand(0);
    auto* def_op = fill_val.getDefiningOp();
    bool is_zero = false;
    if (def_op && def_op->getName().getStringRef() == "arith.constant") {
      if (auto fa = mlir::dyn_cast_or_null<mlir::FloatAttr>(
              def_op->getAttr("value"))) {
        is_zero = fa.getValue().isZero();
      } else if (auto ia = mlir::dyn_cast_or_null<mlir::IntegerAttr>(
                     def_op->getAttr("value"))) {
        is_zero = ia.getValue().isZero();
      }
    }
    if (!is_zero) {
      LOG(INFO) << "XDNA: linalg.fill value is not zero, cannot lower matmul "
                << "(non-zero init not supported)";
      return std::nullopt;
    }
  }

  // Check if the generic op is a pure transpose that feeds into the matmul's
  // B input. Validates shapes, indexing maps, AND body semantics.
  bool b_transposed = false;
  if (generic_count == 1 && generic_op) {
    // Transpose generic: 1 input (DPS), 1 output (DPS) = 2 operands.
    if (generic_op->getNumOperands() != 2) return std::nullopt;
    auto gin_type = mlir::dyn_cast<mlir::RankedTensorType>(
        generic_op->getOperand(0).getType());
    auto gout_type = mlir::dyn_cast<mlir::RankedTensorType>(
        generic_op->getResult(0).getType());
    if (!gin_type || !gout_type) return std::nullopt;
    if (gin_type.getRank() != 2 || gout_type.getRank() != 2) return std::nullopt;
    // Shape check: input (a, b), output (b, a).
    if (gin_type.getShape()[0] != gout_type.getShape()[1] ||
        gin_type.getShape()[1] != gout_type.getShape()[0]) {
      LOG(INFO) << "XDNA: linalg.generic in matmul module is not a transpose";
      return std::nullopt;
    }
    // Verify indexing maps are transpose: input map swaps dims, output is identity.
    // Expected: input = (d0, d1) -> (d1, d0), output = (d0, d1) -> (d0, d1).
    auto generic_typed = mlir::dyn_cast<mlir::linalg::GenericOp>(generic_op);
    if (!generic_typed) return std::nullopt;
    auto maps = generic_typed.getIndexingMapsArray();
    if (maps.size() != 2) return std::nullopt;
    // Input map must be a permutation that swaps the two dims.
    auto in_map = maps[0];
    if (in_map.getNumDims() != 2 || in_map.getNumResults() != 2)
      return std::nullopt;
    auto r0 = mlir::dyn_cast<mlir::AffineDimExpr>(in_map.getResult(0));
    auto r1 = mlir::dyn_cast<mlir::AffineDimExpr>(in_map.getResult(1));
    if (!r0 || !r1 || r0.getPosition() != 1 || r1.getPosition() != 0) {
      LOG(INFO) << "XDNA: linalg.generic input map is not a transpose";
      return std::nullopt;
    }
    // Output map must be identity.
    if (!maps[1].isIdentity()) {
      LOG(INFO) << "XDNA: linalg.generic output map is not identity";
      return std::nullopt;
    }
    // Body must be a pure copy: just yields the input block argument.
    mlir::Block& body = generic_typed.getRegion().front();
    auto body_ops = body.without_terminator();
    if (body_ops.begin() != body_ops.end()) {
      LOG(INFO) << "XDNA: linalg.generic body is not a pure copy";
      return std::nullopt;
    }
    auto yield = mlir::dyn_cast<mlir::linalg::YieldOp>(body.getTerminator());
    if (!yield || yield.getNumOperands() != 1 ||
        yield.getOperand(0) != body.getArgument(0)) {
      LOG(INFO) << "XDNA: linalg.generic yield does not pass through input";
      return std::nullopt;
    }
    // Verify the transpose output feeds into the matmul's B input (operand 1).
    if (generic_op->getResult(0) != matmul_op->getOperand(1)) {
      LOG(INFO) << "XDNA: transpose generic does not feed matmul B input";
      return std::nullopt;
    }
    b_transposed = true;
  }

  // Extract M, K, N from matmul operand shapes.
  // linalg.matmul ins(%A : tensor<MxKxTy>, %B : tensor<KxNxTy>)
  //               outs(%C : tensor<MxNxTy>)
  // When b_transposed, B is the transposed result (K×N) but the host buffer
  // is the original layout (N×K).
  if (matmul_op->getNumOperands() < 3) return std::nullopt;

  auto a_type = mlir::dyn_cast<mlir::RankedTensorType>(
      matmul_op->getOperand(0).getType());
  auto b_type = mlir::dyn_cast<mlir::RankedTensorType>(
      matmul_op->getOperand(1).getType());
  if (!a_type || !b_type) return std::nullopt;
  if (a_type.getRank() != 2 || b_type.getRank() != 2) return std::nullopt;

  std::string elem_type = GetElementTypeStr(a_type.getElementType());
  // V1: only f32 and bf16.
  if (elem_type != "f32" && elem_type != "bf16") return std::nullopt;

  int64_t M = a_type.getShape()[0];
  int64_t K = a_type.getShape()[1];
  int64_t N = b_type.getShape()[1];

  // Sanity: B's first dim should be K.
  if (b_type.getShape()[0] != K) return std::nullopt;

  MatmulProgramInfo info;
  info.M = M;
  info.K = K;
  info.N = N;
  info.element_type = elem_type;
  info.b_transposed = b_transposed;
  if (b_transposed) {
    LOG(INFO) << "XDNA: detected transposed matmul (Q @ K^T pattern)";
  }
  return info;
}

// Detects a softmax pattern in the module.
// XLA decomposes jax.nn.softmax into multiple linalg ops:
//   reduce(maximumf) + exp + reduce(addf) + divf
// We detect this signature and fuse into a single AIE API C++ kernel call.
std::optional<SoftmaxProgramInfo> DetectSoftmax(mlir::ModuleOp module) {
  mlir::func::FuncOp entry_func;
  module.walk([&](mlir::func::FuncOp func) {
    if (func.getName() == "main" || !entry_func) entry_func = func;
  });
  if (!entry_func) return std::nullopt;

  // Verify function signature: single input tensor, single result tensor.
  auto func_type = entry_func.getFunctionType();
  if (func_type.getNumInputs() != 1 || func_type.getNumResults() != 1)
    return std::nullopt;

  // Validate input/output types: both must be the same ranked tensor type.
  auto input = mlir::dyn_cast<mlir::RankedTensorType>(func_type.getInput(0));
  auto output = mlir::dyn_cast<mlir::RankedTensorType>(func_type.getResult(0));
  if (!input || !output) return std::nullopt;
  if (input.getRank() < 1) return std::nullopt;
  if (input != output) {
    LOG(INFO) << "XDNA: softmax input/output type mismatch";
    return std::nullopt;
  }

  std::string elem_type = GetElementTypeStr(input.getElementType());
  if (elem_type != "bf16") return std::nullopt;  // bf16 only for now

  // Count all linalg ops and look for the softmax decomposition pattern:
  //   fill(-inf) + reduce(maximumf) + clamp + broadcast_max +
  //   subf + exp + extf + fill(0) + reduce(addf) + truncf +
  //   broadcast_sum + divf = 2 fills + 10 generics = 12 ops total.
  // Reject if unexpected linalg ops are present (e.g. matmul).
  bool has_max_reduce = false, has_exp = false;
  bool has_sum_reduce = false, has_div = false;
  mlir::Operation* div_op = nullptr;
  int fill_count = 0, generic_count = 0, other_linalg_count = 0;

  entry_func.walk([&](mlir::Operation* op) {
    llvm::StringRef op_name = op->getName().getStringRef();
    if (!op_name.starts_with("linalg.")) return;
    if (op_name == "linalg.yield") return;
    if (op_name == "linalg.fill") { fill_count++; return; }
    if (op_name == "linalg.generic") { generic_count++; } else {
      other_linalg_count++;
      return;
    }
  });

  // Reject if non-fill/generic linalg ops exist or total count is unexpected.
  // XLA's softmax decomposition produces exactly 2 fills + 10 generics = 12.
  // Allow some tolerance for XLA version variation, but reject clearly
  // non-softmax modules.
  if (other_linalg_count > 0) return std::nullopt;
  if (fill_count + generic_count > 15) return std::nullopt;

  // The last dimension index — softmax must reduce over this axis.
  int64_t last_dim_idx = input.getRank() - 1;

  entry_func.walk([&](mlir::linalg::GenericOp generic) {
    auto iterators = generic.getIteratorTypesArray();

    // Find which positions are reductions.
    bool is_reduction = false;
    for (auto it : iterators) {
      if (it == mlir::utils::IteratorType::reduction) {
        is_reduction = true;
        break;
      }
    }

    // For reductions, verify the reduction dimension maps to the input's
    // last axis. XLA's softmax reduces along the last dimension; if this
    // reduces a different axis, it's not last-axis softmax.
    if (is_reduction) {
      auto maps = generic.getIndexingMapsArray();
      // The first map is the input map. Find which loop dim(s) are reductions
      // and check they correspond to the last input dimension.
      bool reduces_last_dim = false;
      if (!maps.empty()) {
        auto input_map = maps[0];
        for (unsigned d = 0; d < iterators.size(); d++) {
          if (iterators[d] != mlir::utils::IteratorType::reduction) continue;
          // Check if this reduction loop dim maps to the last input dim.
          // For softmax, the input map is (d0, d1) -> (d0, d1) and d1 is
          // the reduction dim, which maps to position 1 = last_dim_idx.
          for (unsigned r = 0; r < input_map.getNumResults(); r++) {
            auto expr = input_map.getResult(r);
            if (auto dim = mlir::dyn_cast<mlir::AffineDimExpr>(expr)) {
              if (dim.getPosition() == d &&
                  static_cast<int64_t>(r) == last_dim_idx) {
                reduces_last_dim = true;
              }
            }
          }
        }
      }
      if (!reduces_last_dim) {
        // This reduction doesn't reduce the last axis — not last-axis softmax.
        // Clear the flags so detection fails.
        has_max_reduce = false;
        has_sum_reduce = false;
        return;
      }
    }

    mlir::Block& body = generic.getRegion().front();
    for (mlir::Operation& body_op : body.without_terminator()) {
      llvm::StringRef name = body_op.getName().getStringRef();
      if (name == "arith.maximumf" && is_reduction) has_max_reduce = true;
      if (name == "math.exp") has_exp = true;
      if (name == "arith.addf" && is_reduction) has_sum_reduce = true;
      if (name == "arith.divf") { has_div = true; div_op = generic; }
    }
  });

  if (!has_max_reduce || !has_exp || !has_sum_reduce || !has_div)
    return std::nullopt;

  // Verify the divf generic's output feeds into the function return value.
  // This ensures the module's output IS the softmax result, not some other
  // computation that happens to contain softmax-like ops.
  if (div_op) {
    mlir::Value div_result = div_op->getResult(0);
    bool feeds_return = false;
    for (mlir::Operation* user : div_result.getUsers()) {
      // The result may pass through stablehlo.custom_call @xla.sdy.FuncResultSharding
      // before reaching the return op.
      if (mlir::isa<mlir::func::ReturnOp>(user)) {
        feeds_return = true;
      } else if (user->getName().getStringRef() == "stablehlo.custom_call") {
        for (mlir::Operation* inner_user : user->getResult(0).getUsers()) {
          if (mlir::isa<mlir::func::ReturnOp>(inner_user))
            feeds_return = true;
        }
      }
    }
    if (!feeds_return) {
      LOG(INFO) << "XDNA: divf output does not feed function return, "
                << "not a pure softmax module";
      return std::nullopt;
    }
  }

  int64_t row_length = input.getShape().back();
  int64_t num_rows = 1;
  for (int i = 0; i < input.getRank() - 1; i++)
    num_rows *= input.getShape()[i];

  // AIE API softmax kernel uses 32-wide bf16 vectors.
  if (row_length % 32 != 0) return std::nullopt;
  if (num_rows < 1) return std::nullopt;

  SoftmaxProgramInfo sinfo;
  sinfo.num_rows = num_rows;
  sinfo.row_length = row_length;
  sinfo.element_type = elem_type;
  return sinfo;
}

// Helper: classify a Value as a 2D broadcast of a 1D iota (index vector).
// Returns 0 if not an index tensor, 1 if row indices, 2 if column indices.
// XLA lowers jnp.triu into: iota (linalg.index 0 → i32) → broadcast to 2D.
// The broadcast indexing map determines row vs column:
//   (d0, d1) -> (d0) = row indices,  (d0, d1) -> (d1) = column indices.
static int ClassifyIndexTensor(mlir::Value val) {
  auto* def = val.getDefiningOp();
  if (!def) return 0;
  auto bcast = mlir::dyn_cast<mlir::linalg::GenericOp>(def);
  if (!bcast) return 0;

  // Must be a broadcast: 1 input (1D i32), 1 output (2D i32), body = yield.
  if (bcast.getInputs().size() != 1) return 0;
  auto in_type = mlir::dyn_cast<mlir::RankedTensorType>(
      bcast.getInputs()[0].getType());
  auto out_type = mlir::dyn_cast<mlir::RankedTensorType>(
      bcast.getResult(0).getType());
  if (!in_type || !out_type) return 0;
  if (in_type.getRank() != 1 || out_type.getRank() != 2) return 0;

  // Body should be a pure yield (broadcast — no computation).
  mlir::Block& body = bcast.getRegion().front();
  auto body_ops = body.without_terminator();
  if (body_ops.begin() != body_ops.end()) return 0;

  // Input indexing map determines row vs column broadcast.
  auto maps = bcast.getIndexingMapsArray();
  if (maps.empty()) return 0;
  auto in_map = maps[0];
  if (in_map.getNumResults() != 1) return 0;
  auto dim = mlir::dyn_cast<mlir::AffineDimExpr>(in_map.getResult(0));
  if (!dim) return 0;

  // Verify the 1D source is an iota pattern (contains linalg.index).
  auto* iota_def = bcast.getInputs()[0].getDefiningOp();
  if (!iota_def) return 0;
  auto iota = mlir::dyn_cast<mlir::linalg::GenericOp>(iota_def);
  if (!iota) return 0;
  bool has_index = false;
  for (auto& op : iota.getRegion().front()) {
    if (mlir::isa<mlir::linalg::IndexOp>(&op)) has_index = true;
  }
  if (!has_index) return 0;

  if (dim.getPosition() == 0) return 1;  // row broadcast
  if (dim.getPosition() == 1) return 2;  // column broadcast
  return 0;
}

// Detects whether the attention module has a causal (upper-triangular) mask.
// XLA decomposes jnp.triu(jnp.ones((S,S), bool), k=1) into multiple separate
// linalg.generic ops:
//   1. iota (linalg.index 0) → 1D row/col index vectors
//   2. broadcast to 2D via (d0,d1)->(d0) or (d0,d1)->(d1)
//   3. arith.cmpi sge (row_indices, col_indices) → lower triangle
//   4. arith.select(lower_tri, false, true) → upper triangle
//   5. arith.select(upper_tri, neg_value, scores) → masked scores
//
// Strategy 1: Find a generic whose body contains arith.cmpi with row/col
//   index tensor inputs (traced through broadcast→iota via ClassifyIndexTensor).
//   The cmpi and select are in SEPARATE generics (XLA does not fuse them).
// Strategy 2: Constant boolean tensor input that is upper-triangular.
bool DetectCausalMask(mlir::func::FuncOp entry_func, int64_t M, int64_t S) {
  // Causal masking only makes sense for self-attention (M == S).
  if (M != S) {
    LOG(INFO) << "XDNA: M=" << M << " != S=" << S
              << ", not self-attention — cannot be causal";
    return false;
  }

  bool found_causal = false;
  entry_func.walk([&](mlir::linalg::GenericOp generic) {
    if (found_causal) return;
    mlir::Block& body = generic.getRegion().front();

    // Strategy 1: Find arith.cmpi in body that compares row vs column indices.
    // XLA decomposes the mask into separate generics, so the cmpi is in its
    // own generic that takes 2D broadcast index tensors as inputs.
    // We validate:
    //   (a) Both operands trace through broadcast→iota to row/col indices
    //   (b) Predicate is consistent with causal masking (k=1):
    //       slt(row,col) or sgt(col,row) → upper triangle (strict)
    //       sge(row,col) or sle(col,row) → lower triangle (complement)
    //   (c) The cmpi result reaches a bf16 arith.select (score masking)
    // CmpIPredicate enum: eq=0, ne=1, slt=2, sle=3, sgt=4, sge=5.
    for (mlir::Operation& op : body.without_terminator()) {
      if (op.getName().getStringRef() != "arith.cmpi") continue;

      auto lhs_arg = mlir::dyn_cast<mlir::BlockArgument>(op.getOperand(0));
      auto rhs_arg = mlir::dyn_cast<mlir::BlockArgument>(op.getOperand(1));
      if (!lhs_arg || !rhs_arg) continue;
      unsigned lhs_idx = lhs_arg.getArgNumber();
      unsigned rhs_idx = rhs_arg.getArgNumber();
      if (lhs_idx >= generic.getInputs().size()) continue;
      if (rhs_idx >= generic.getInputs().size()) continue;

      int lhs_class = ClassifyIndexTensor(generic.getInputs()[lhs_idx]);
      int rhs_class = ClassifyIndexTensor(generic.getInputs()[rhs_idx]);

      // One must be row indices (1), other must be column indices (2).
      if (!((lhs_class == 1 && rhs_class == 2) ||
            (lhs_class == 2 && rhs_class == 1))) {
        continue;
      }

      // Validate predicate direction for causal masking.
      auto pred_attr = op.getAttrOfType<mlir::IntegerAttr>("predicate");
      if (!pred_attr) continue;
      int64_t pred = pred_attr.getInt();

      bool valid_causal = false;
      if (lhs_class == 1 && rhs_class == 2) {
        // lhs=row, rhs=col: slt (row < col) or sge (row >= col)
        valid_causal = (pred == 2 || pred == 5);
      } else {
        // lhs=col, rhs=row: sgt (col > row) or sle (col <= row)
        valid_causal = (pred == 4 || pred == 3);
      }
      if (!valid_causal) continue;

      // Verify the cmpi result reaches a bf16 arith.select (the score
      // masking op). BFS through tensor-level SSA users of this generic's
      // result, following through intermediate generics (e.g., the i1
      // mask-inversion select), until we find a generic whose body has
      // arith.select on bf16.
      bool reaches_bf16_select = false;
      llvm::SmallVector<mlir::Value, 8> worklist;
      llvm::SmallPtrSet<mlir::Operation*, 8> visited;
      worklist.push_back(generic.getResult(0));
      while (!worklist.empty() && !reaches_bf16_select) {
        mlir::Value v = worklist.pop_back_val();
        for (mlir::Operation* user : v.getUsers()) {
          if (!visited.insert(user).second) continue;
          auto user_generic =
              mlir::dyn_cast<mlir::linalg::GenericOp>(user);
          if (!user_generic) continue;
          mlir::Block& ubody = user_generic.getRegion().front();
          for (mlir::Operation& uop : ubody.without_terminator()) {
            if (uop.getName().getStringRef() == "arith.select" &&
                uop.getOperand(1).getType().isF16() == false &&
                uop.getOperand(1).getType().isBF16()) {
              reaches_bf16_select = true;
              break;
            }
          }
          // Follow through this generic's results for transitive users.
          for (mlir::OpResult res : user_generic->getResults()) {
            worklist.push_back(res);
          }
        }
      }
      if (!reaches_bf16_select) continue;

      LOG(INFO) << "XDNA: detected causal mask via row/col index comparison"
                << " (pred=" << pred << ", lhs=" << lhs_class
                << ", rhs=" << rhs_class << ")";
      found_causal = true;
      return;
    }

    // Strategy 2: Constant boolean tensor input that is upper-triangular.
    // (Handles potential future XLA constant-folding of the mask.)
    mlir::Value select_cond;
    for (mlir::Operation& op : body.without_terminator()) {
      if (op.getName().getStringRef() == "arith.select") {
        select_cond = op.getOperand(0);
        break;
      }
    }
    if (!select_cond) return;
    if (auto block_arg = mlir::dyn_cast<mlir::BlockArgument>(select_cond)) {
      unsigned arg_idx = block_arg.getArgNumber();
      if (arg_idx < generic.getInputs().size()) {
        mlir::Value input = generic.getInputs()[arg_idx];
        if (auto* def_op = input.getDefiningOp()) {
          if (def_op->getName().getStringRef() == "arith.constant") {
            auto attr = def_op->getAttr("value");
            auto dense = mlir::dyn_cast_or_null<mlir::DenseElementsAttr>(attr);
            if (dense) {
              auto shaped = dense.getType();
              if (shaped.getRank() == 2 &&
                  shaped.getDimSize(0) == M &&
                  shaped.getDimSize(1) == S) {
                // Verify upper-triangular: true above diagonal, false on/below.
                bool is_upper_tri = true;
                int64_t idx = 0;
                for (int64_t r = 0; r < M && is_upper_tri; r++) {
                  for (int64_t c = 0; c < S && is_upper_tri; c++, idx++) {
                    bool expected = (c > r);
                    bool actual;
                    if (dense.getElementType().isInteger(1)) {
                      actual = dense.getValues<bool>()[idx];
                    } else if (dense.getElementType().isIntOrIndex()) {
                      actual = dense.getValues<mlir::APInt>()[idx].getBoolValue();
                    } else {
                      is_upper_tri = false;
                      break;
                    }
                    if (actual != expected) is_upper_tri = false;
                  }
                }
                if (is_upper_tri) {
                  LOG(INFO) << "XDNA: detected causal mask via constant tensor";
                  found_causal = true;
                  return;
                }
              }
            }
          }
        }
      }
    }
  });

  return found_causal;
}

// Detects a fused attention pattern: softmax(Q @ K^T * scale) @ V.
// The module must contain exactly 2 linalg.matmul ops, a transpose generic
// feeding the first matmul's B input, and softmax-like ops (reduce(maximumf),
// exp, reduce(addf), divf) between the matmuls. 3 inputs, 1 output.
std::optional<AttentionProgramInfo> DetectAttention(mlir::ModuleOp module) {
  mlir::func::FuncOp entry_func;
  module.walk([&](mlir::func::FuncOp func) {
    if (func.getName() == "main" || !entry_func) entry_func = func;
  });
  if (!entry_func) {
    LOG(INFO) << "XDNA DetectAttention: no entry func";
    return std::nullopt;
  }

  // Verify function signature: 3 input tensors, 1 result tensor.
  auto func_type = entry_func.getFunctionType();
  if (func_type.getNumInputs() != 3 || func_type.getNumResults() != 1) {
    LOG(INFO) << "XDNA DetectAttention: sig mismatch inputs="
               << func_type.getNumInputs() << " results="
               << func_type.getNumResults();
    return std::nullopt;
  }

  // All inputs must be 2D bf16 ranked tensors.
  for (unsigned i = 0; i < 3; i++) {
    auto t = mlir::dyn_cast<mlir::RankedTensorType>(func_type.getInput(i));
    if (!t || t.getRank() != 2) return std::nullopt;
    if (GetElementTypeStr(t.getElementType()) != "bf16") return std::nullopt;
  }
  auto result_type = mlir::dyn_cast<mlir::RankedTensorType>(
      func_type.getResult(0));
  if (!result_type || result_type.getRank() != 2) return std::nullopt;
  if (GetElementTypeStr(result_type.getElementType()) != "bf16")
    return std::nullopt;

  // Count linalg ops.
  int matmul_count = 0;
  int fill_count = 0;
  int generic_count = 0;
  int other_linalg_count = 0;
  std::vector<mlir::Operation*> matmul_ops;

  entry_func.walk([&](mlir::Operation* op) {
    llvm::StringRef name = op->getName().getStringRef();
    if (!name.starts_with("linalg.")) return;
    if (name == "linalg.yield" || name == "linalg.index") return;
    if (name == "linalg.fill") {
      fill_count++;
    } else if (name == "linalg.matmul") {
      matmul_count++;
      matmul_ops.push_back(op);
    } else if (name == "linalg.generic") {
      generic_count++;
    } else {
      other_linalg_count++;
    }
  });

  // Must have exactly 2 matmuls, no other linalg ops (e.g., conv).
  if (matmul_count != 2 || other_linalg_count != 0) {
    LOG(INFO) << "XDNA DetectAttention: op counts matmul=" << matmul_count
               << " other=" << other_linalg_count
               << " fill=" << fill_count << " generic=" << generic_count;
    return std::nullopt;
  }
  // At least 2 fills (one per matmul), plus generics for transpose/scale/softmax.
  if (fill_count < 2) {
    LOG(INFO) << "XDNA DetectAttention: fill_count=" << fill_count;
    return std::nullopt;
  }
  // Softmax decomposes into ~10 generics + transpose + scale = ~12+ generics.
  if (generic_count < 5) {
    LOG(INFO) << "XDNA DetectAttention: generic_count=" << generic_count;
    return std::nullopt;
  }

  // Check for softmax markers in the generics.
  bool has_max_reduce = false, has_exp = false;
  bool has_sum_reduce = false, has_div = false;

  entry_func.walk([&](mlir::linalg::GenericOp generic) {
    auto iterators = generic.getIteratorTypesArray();
    bool is_reduction = false;
    for (auto it : iterators) {
      if (it == mlir::utils::IteratorType::reduction) {
        is_reduction = true;
        break;
      }
    }
    mlir::Block& body = generic.getRegion().front();
    for (mlir::Operation& body_op : body.without_terminator()) {
      llvm::StringRef name = body_op.getName().getStringRef();
      if (name == "arith.maximumf" && is_reduction) has_max_reduce = true;
      if (name == "math.exp") has_exp = true;
      if (name == "arith.addf" && is_reduction) has_sum_reduce = true;
      if (name == "arith.divf") has_div = true;
    }
  });

  if (!has_max_reduce || !has_exp || !has_sum_reduce || !has_div) {
    LOG(INFO) << "XDNA: 2 matmuls found but no softmax pattern, "
              << "not fused attention";
    return std::nullopt;
  }

  // Check for masking (arith.select). If present, try to detect a causal
  // (upper-triangular) mask pattern. Non-causal masks are rejected.
  bool has_select = false;
  entry_func.walk([&](mlir::linalg::GenericOp generic) {
    mlir::Block& body = generic.getRegion().front();
    for (mlir::Operation& op : body.without_terminator()) {
      if (op.getName().getStringRef() == "arith.select")
        has_select = true;
    }
  });
  bool is_causal = false;
  if (has_select) {
    // Dimensions not yet extracted — we need them for DetectCausalMask.
    // Peek at matmul shapes to get M and S for the causal check.
    auto peek_q = mlir::dyn_cast<mlir::RankedTensorType>(
        matmul_ops[0]->getOperand(0).getType());
    auto peek_kt = mlir::dyn_cast<mlir::RankedTensorType>(
        matmul_ops[0]->getOperand(1).getType());
    if (!peek_q || !peek_kt || peek_q.getRank() != 2 || peek_kt.getRank() != 2) {
      LOG(INFO) << "XDNA: attention has masking but can't peek dimensions";
      return std::nullopt;
    }
    // Determine matmul order: the matmul whose result feeds the return is #2.
    int64_t peek_M, peek_S;
    bool m1_feeds_return = false;
    for (mlir::Operation* user : matmul_ops[0]->getResult(0).getUsers()) {
      if (mlir::isa<mlir::func::ReturnOp>(user)) m1_feeds_return = true;
      if (user->getName().getStringRef() == "stablehlo.custom_call") {
        for (mlir::Operation* inner : user->getResult(0).getUsers()) {
          if (mlir::isa<mlir::func::ReturnOp>(inner)) m1_feeds_return = true;
        }
      }
    }
    if (m1_feeds_return) {
      // matmul_ops[1] is the Q@K^T matmul
      auto q2 = mlir::dyn_cast<mlir::RankedTensorType>(
          matmul_ops[1]->getOperand(0).getType());
      auto kt2 = mlir::dyn_cast<mlir::RankedTensorType>(
          matmul_ops[1]->getOperand(1).getType());
      if (!q2 || !kt2) { return std::nullopt; }
      peek_M = q2.getShape()[0];
      peek_S = kt2.getShape()[1];
    } else {
      peek_M = peek_q.getShape()[0];
      peek_S = peek_kt.getShape()[1];
    }
    is_causal = DetectCausalMask(entry_func, peek_M, peek_S);
    if (!is_causal) {
      LOG(INFO) << "XDNA: attention has non-causal masking (arith.select) — "
                << "not fusing";
      return std::nullopt;
    }
    LOG(INFO) << "XDNA: detected causal mask in attention pattern";
  }

  // Check for transpose generic that feeds the first matmul's B input.
  // The first matmul (in program order) computes Q @ K^T.
  mlir::Operation* matmul1 = matmul_ops[0];
  mlir::Operation* matmul2 = matmul_ops[1];

  // Determine matmul order: the first matmul's result should NOT be the
  // function return. The second matmul's result should feed the return.
  // If matmul_ops[1]'s result feeds into matmul_ops[0] (reverse order),
  // swap them.
  bool mat1_feeds_return = false;
  for (mlir::Operation* user : matmul1->getResult(0).getUsers()) {
    if (mlir::isa<mlir::func::ReturnOp>(user)) mat1_feeds_return = true;
    if (user->getName().getStringRef() == "stablehlo.custom_call") {
      for (mlir::Operation* inner : user->getResult(0).getUsers()) {
        if (mlir::isa<mlir::func::ReturnOp>(inner)) mat1_feeds_return = true;
      }
    }
  }
  if (mat1_feeds_return) {
    std::swap(matmul1, matmul2);
  }

  // Verify matmul1's B input comes from a transpose generic.
  bool has_transpose = false;
  mlir::Value matmul1_b = matmul1->getOperand(1);
  if (auto* def_op = matmul1_b.getDefiningOp()) {
    if (auto generic = mlir::dyn_cast<mlir::linalg::GenericOp>(def_op)) {
      // Check if it's a pure transpose: 2 operands, swapped indexing maps,
      // body is a pure yield of input.
      if (generic->getNumOperands() == 2) {
        auto gin = mlir::dyn_cast<mlir::RankedTensorType>(
            generic->getOperand(0).getType());
        auto gout = mlir::dyn_cast<mlir::RankedTensorType>(
            generic->getResult(0).getType());
        if (gin && gout && gin.getRank() == 2 && gout.getRank() == 2 &&
            gin.getShape()[0] == gout.getShape()[1] &&
            gin.getShape()[1] == gout.getShape()[0]) {
          auto maps = generic.getIndexingMapsArray();
          if (maps.size() == 2) {
            auto r0 = mlir::dyn_cast<mlir::AffineDimExpr>(
                maps[0].getResult(0));
            auto r1 = mlir::dyn_cast<mlir::AffineDimExpr>(
                maps[0].getResult(1));
            if (r0 && r1 && r0.getPosition() == 1 && r1.getPosition() == 0 &&
                maps[1].isIdentity()) {
              mlir::Block& body = generic.getRegion().front();
              auto body_ops = body.without_terminator();
              if (body_ops.begin() == body_ops.end()) {
                auto yield = mlir::dyn_cast<mlir::linalg::YieldOp>(
                    body.getTerminator());
                if (yield && yield.getNumOperands() == 1 &&
                    yield.getOperand(0) == body.getArgument(0)) {
                  has_transpose = true;
                }
              }
            }
          }
        }
      }
    }
  }

  if (!has_transpose) {
    LOG(INFO) << "XDNA: 2 matmuls + softmax found but no transpose on "
              << "first matmul B input, not Q@K^T attention pattern";
    return std::nullopt;
  }

  // Extract dimensions.
  // matmul1: Q[M, dk] @ K_T[dk, S] → scores[M, S]
  auto q_type = mlir::dyn_cast<mlir::RankedTensorType>(
      matmul1->getOperand(0).getType());
  auto kt_type = mlir::dyn_cast<mlir::RankedTensorType>(
      matmul1->getOperand(1).getType());
  if (!q_type || !kt_type) return std::nullopt;
  if (q_type.getRank() != 2 || kt_type.getRank() != 2) return std::nullopt;

  int64_t M = q_type.getShape()[0];
  int64_t dk = q_type.getShape()[1];
  int64_t S = kt_type.getShape()[1];

  // Verify K^T's first dim matches dk.
  if (kt_type.getShape()[0] != dk) return std::nullopt;

  // matmul2: weights[M, S] @ V[S, dk2] → O[M, dk2]
  auto v_type = mlir::dyn_cast<mlir::RankedTensorType>(
      matmul2->getOperand(1).getType());
  if (!v_type || v_type.getRank() != 2) return std::nullopt;
  int64_t dk2 = v_type.getShape()[1];
  if (v_type.getShape()[0] != S) return std::nullopt;

  // For standard attention, dk == dk2 (Q and V have same model dimension).
  if (dk != dk2) {
    LOG(INFO) << "XDNA: attention dk mismatch: Q dk=" << dk
              << " V dk=" << dk2;
    return std::nullopt;
  }

  // Verify output shape matches [M, dk].
  if (result_type.getShape()[0] != M || result_type.getShape()[1] != dk)
    return std::nullopt;

  // Constraints for the fused kernel.
  if (dk % 32 != 0) {
    LOG(INFO) << "XDNA: attention dk=" << dk << " not divisible by 32";
    return std::nullopt;
  }
  if (S % 32 != 0) {
    LOG(INFO) << "XDNA: attention seq_len=" << S << " not divisible by 32";
    return std::nullopt;
  }

  // Validate the scale constant matches 1/sqrt(dk). The fused kernel
  // hardcodes log2e/sqrt(dk), so a non-standard temperature would produce
  // wrong results.
  bool found_scale = false;
  entry_func.walk([&](mlir::linalg::GenericOp generic) {
    if (found_scale) return;
    auto iters = generic.getIteratorTypesArray();
    bool all_parallel = std::all_of(iters.begin(), iters.end(),
        [](auto t) { return t == mlir::utils::IteratorType::parallel; });
    if (!all_parallel) return;
    // Check body: single arith.mulf.
    mlir::Block& body = generic.getRegion().front();
    int mulf_count = 0;
    for (mlir::Operation& op : body.without_terminator())
      if (op.getName().getStringRef() == "arith.mulf") mulf_count++;
    if (mulf_count != 1) return;
    // Check inputs for a broadcast constant matching 1/sqrt(dk).
    // The constant may have a scalar map (numResults==0) or an identity map
    // if StableHLO→linalg already broadcast it to a full tensor.
    int n_inputs = generic.getInputs().size();
    for (int i = 0; i < n_inputs; i++) {
      auto val = TryExtractScalarFloat(generic.getInputs()[i]);
      if (val.has_value()) {
        float expected = 1.0f / sqrtf(static_cast<float>(dk));
        // Relative tolerance: bf16 has ~0.8% relative error, use 2% to
        // allow rounding but reject materially wrong temperatures.
        // Floor of 1e-4 prevents division-by-zero-like issues.
        float tol = std::max(1e-4f, 0.02f * expected);
        if (std::abs(*val - expected) <= tol) found_scale = true;
      }
    }
  });
  if (!found_scale) {
    LOG(INFO) << "XDNA: attention scale not found or doesn't match "
              << "1/sqrt(dk=" << dk << ") — not fusing";
    return std::nullopt;
  }

  LOG(INFO) << "XDNA: detected fused attention pattern: M=" << M
            << " S=" << S << " dk=" << dk
            << (is_causal ? " (causal)" : "");

  AttentionProgramInfo info;
  info.num_rows = M;
  info.seq_len = S;
  info.dk = dk;
  info.element_type = "bf16";
  info.is_causal = is_causal;
  return info;
}

// Selects the KV block size (Sb) for tiled attention.
// Returns the largest Sb ≤ max_sb that evenly divides seq_len.
static int64_t SelectKvBlockSize(int64_t seq_len, int64_t dk,
                                  int64_t m_per_core, int64_t l1_budget) {
  // L1 budget for tiled path with combined KV FIFO (depth=2, acquire(2) → 3 buffers):
  //   Q: 1 * m_pc * dk * 2    (depth=1, single transfer)
  //   KV: 3 * Sb * dk * 2     (depth=2, acquire(2))
  //   O: 1 * m_pc * dk * 2    (depth=1, single transfer)
  //   State: m_pc * 2 * 2     (max_val + total_sum per row)
  // Total = 4*m_pc*dk + 6*Sb*dk + 4*m_pc  ≤ l1_budget
  // => Sb ≤ (l1_budget - 4*m_pc*dk - 4*m_pc) / (6*dk)
  int64_t max_sb = (l1_budget - 4 * dk * m_per_core - 4 * m_per_core) /
                   (6 * dk);
  // Find largest divisor of seq_len that fits.
  for (int64_t sb = max_sb; sb >= 1; sb--) {
    if (seq_len % sb == 0) return sb;
  }
  return 1;  // Fallback: process one row at a time.
}

// Generates AIE MLIR for TILED fused attention (large seq_len).
// Streams K and V through L1 in blocks of Sb rows using a combined KV
// ObjectFIFO (depth=2, acquire(2) per block). K in slot[0], V in slot[1].
// An aie.buffer holds online softmax state (max_val, total_sum) across blocks.
absl::StatusOr<AieLoweringResult> LowerTiledAttentionToAie(
    const AttentionProgramInfo& info, const TargetCaps& caps,
    int num_cores, int64_t m_per_core, int64_t sb) {
  int64_t num_blocks = info.seq_len / sb;

  // Verify L1 budget:
  // Q(depth=1) + KV(3 buffers: depth=2+acquire(2)) + O(depth=1) + state.
  int64_t tiled_l1 = 4 * m_per_core * info.dk +
                     6 * sb * info.dk +
                     4 * m_per_core;
  if (tiled_l1 > caps.l1_usable_bytes) {
    return absl::InvalidArgumentError(absl::StrFormat(
        "Tiled attention L1 exceeded: M_pc=%d Sb=%d dk=%d needs %dB, "
        "have %dB", m_per_core, sb, info.dk, tiled_l1, caps.l1_usable_bytes));
  }

  // DMA dimension limits.
  int64_t d0_words_dk = info.dk * 2 / 4;
  if (d0_words_dk > kMaxBdWords) {
    return absl::InvalidArgumentError(absl::StrFormat(
        "Attention dk=%d exceeds DMA d0 limit (%d words, max %d)",
        info.dk, d0_words_dk, kMaxBdWords));
  }
  if (sb > kMaxBdWords) {
    return absl::InvalidArgumentError(absl::StrFormat(
        "Attention Sb=%d exceeds DMA d1 limit (max %d)", sb, kMaxBdWords));
  }
  if (m_per_core > kMaxBdWords) {
    return absl::InvalidArgumentError(absl::StrFormat(
        "Attention m_per_core=%d exceeds DMA d1 limit (max %d)",
        m_per_core, kMaxBdWords));
  }

  LOG(INFO) << "XDNA tiled attention lowering: M=" << info.num_rows
            << " S=" << info.seq_len << " dk=" << info.dk
            << " Sb=" << sb << " blocks=" << num_blocks
            << " using " << num_cores << " core(s) ("
            << m_per_core << " rows/core)"
            << " L1=" << tiled_l1 << "B";

  int start_col = caps.partition_start_column;
  int shim_row = caps.shim_row;
  int compute_row = caps.first_compute_row;
  int kv_fifo_depth = 2;   // KV: double-buffered for pipelining.
  int qo_fifo_depth = 1;   // Q/O: single transfer, depth=1 saves L1.

  // ObjectFIFO buffer types.
  std::string q_memref = absl::StrFormat("memref<%dxbf16>",
                                          m_per_core * info.dk);
  // Combined KV FIFO: each element = Sb*dk bf16. Depth=2, acquire(2).
  // Slot[0]=K_block, slot[1]=V_block.
  std::string kv_memref = absl::StrFormat("memref<%dxbf16>", sb * info.dk);
  std::string o_memref = absl::StrFormat("memref<%dxbf16>",
                                          m_per_core * info.dk);
  // State buffer: 2 bf16 per query row (max_val + total_sum).
  // Causal: +4 bf16 slots for m_offset (2 slots = i32) + block_counter (2 slots = i32).
  // Causal: +2 bf16 slots for m_offset (1 i32 = 2 bf16).
  int64_t state_elems = m_per_core * 2 + (info.is_causal ? 2 : 0);
  std::string state_memref = absl::StrFormat("memref<%dxbf16>", state_elems);

  // Host buffer types.
  int64_t q_total = info.num_rows * info.dk;
  int64_t kv_total = info.seq_len * info.dk;
  int64_t o_total = info.num_rows * info.dk;
  std::string q_host_memref = absl::StrFormat("memref<%dxbf16>", q_total);
  std::string k_host_memref = absl::StrFormat("memref<%dxbf16>", kv_total);
  std::string v_host_memref = absl::StrFormat("memref<%dxbf16>", kv_total);
  std::string o_host_memref = absl::StrFormat("memref<%dxbf16>", o_total);

  auto fifo_name = [&](int c, const std::string& base) -> std::string {
    return num_cores > 1 ? absl::StrFormat("c%d_%s", c, base) : base;
  };

  std::string aie_mlir;
  absl::StrAppend(&aie_mlir, "module {\n");
  absl::StrAppend(&aie_mlir, absl::StrFormat(
      "  aie.device(%s) {\n", DeviceNameForColumns(num_cores)));

  // Pass 1: Tile declarations.
  for (int c = 0; c < num_cores; c++) {
    int col = start_col + c;
    absl::StrAppend(&aie_mlir, absl::StrFormat(
        "    %%tile_%d_%d = aie.tile(%d, %d)\n", col, shim_row, col, shim_row));
    absl::StrAppend(&aie_mlir, absl::StrFormat(
        "    %%tile_%d_%d = aie.tile(%d, %d)\n",
        col, caps.mem_tile_row, col, caps.mem_tile_row));
    absl::StrAppend(&aie_mlir, absl::StrFormat(
        "    %%tile_%d_%d = aie.tile(%d, %d)\n",
        col, compute_row, col, compute_row));
  }

  // Pass 2: ObjectFIFOs and links.
  // KV broadcast: combined K/V FIFO from shim(start_col) → mem → all compute.
  // Single shim output channel (avoids DMA channel exhaustion).
  {
    std::string kv_shim = absl::StrFormat("tile_%d_%d", start_col, shim_row);
    std::string kv_mem = absl::StrFormat("tile_%d_%d", start_col,
                                          caps.mem_tile_row);
    absl::StrAppend(&aie_mlir, absl::StrFormat(
        "    aie.objectfifo @in_kv_L2(%%%s, {%%%s}, %d : i32) "
        ": !aie.objectfifo<%s>\n",
        kv_shim, kv_mem, kv_fifo_depth, kv_memref));
    std::string kv_consumers;
    for (int c = 0; c < num_cores; c++) {
      if (c > 0) kv_consumers += ", ";
      absl::StrAppendFormat(&kv_consumers, "%%tile_%d_%d",
                            start_col + c, compute_row);
    }
    absl::StrAppend(&aie_mlir, absl::StrFormat(
        "    aie.objectfifo @in_kv(%%%s, {%s}, %d : i32) "
        ": !aie.objectfifo<%s>\n",
        kv_mem, kv_consumers, kv_fifo_depth, kv_memref));
    absl::StrAppend(&aie_mlir,
        "    aie.objectfifo.link [@in_kv_L2] -> [@in_kv] ([] [])\n");
  }

  // Q and O per-column.
  for (int c = 0; c < num_cores; c++) {
    int col = start_col + c;
    std::string shim = absl::StrFormat("tile_%d_%d", col, shim_row);
    std::string mem = absl::StrFormat("tile_%d_%d", col, caps.mem_tile_row);
    std::string compute = absl::StrFormat("tile_%d_%d", col, compute_row);

    // Q input: shim → mem → compute. Depth=1 (single transfer).
    absl::StrAppend(&aie_mlir, absl::StrFormat(
        "    aie.objectfifo @%s(%%%s, {%%%s}, %d : i32) "
        ": !aie.objectfifo<%s>\n",
        fifo_name(c, "in_q_L2"), shim, mem, qo_fifo_depth, q_memref));
    absl::StrAppend(&aie_mlir, absl::StrFormat(
        "    aie.objectfifo @%s(%%%s, {%%%s}, %d : i32) "
        ": !aie.objectfifo<%s>\n",
        fifo_name(c, "in_q"), mem, compute, qo_fifo_depth, q_memref));
    absl::StrAppend(&aie_mlir, absl::StrFormat(
        "    aie.objectfifo.link [@%s] -> [@%s] ([] [])\n",
        fifo_name(c, "in_q_L2"), fifo_name(c, "in_q")));

    // O output: compute → mem → shim. Depth=1 (single transfer).
    absl::StrAppend(&aie_mlir, absl::StrFormat(
        "    aie.objectfifo @%s(%%%s, {%%%s}, %d : i32) "
        ": !aie.objectfifo<%s>\n",
        fifo_name(c, "out"), compute, mem, qo_fifo_depth, o_memref));
    absl::StrAppend(&aie_mlir, absl::StrFormat(
        "    aie.objectfifo @%s(%%%s, {%%%s}, %d : i32) "
        ": !aie.objectfifo<%s>\n",
        fifo_name(c, "out_L2"), mem, shim, qo_fifo_depth, o_memref));
    absl::StrAppend(&aie_mlir, absl::StrFormat(
        "    aie.objectfifo.link [@%s] -> [@%s] ([] [])\n",
        fifo_name(c, "out"), fifo_name(c, "out_L2")));
  }

  // State buffer per compute tile (for online softmax running state).
  for (int c = 0; c < num_cores; c++) {
    int col = start_col + c;
    absl::StrAppend(&aie_mlir, absl::StrFormat(
        "    %%state_%d = aie.buffer(%%tile_%d_%d) : %s\n",
        c, col, compute_row, state_memref));
  }

  // External function declarations for tiled attention kernel.
  // CONTRACT: The kernel loop bounds use compile-time CONST_M_PC/CONST_SB
  // (#define) for Peano VLIW stability. The i32 args have different meanings:
  //   Non-causal: (m_pc, sb, dk) — unused by kernel, must match compile-time.
  //   Causal:     (m_offset, s_offset, dk) — m_pc/sb slots repurposed for
  //               causal offsets since the kernel ignores them anyway.
  // Same function signature for both cases (8 params) to stay within AIE ABI
  // register limits.
  absl::StrAppend(&aie_mlir, absl::StrFormat(
      "    func.func private @attention_init_bf16(%s, %s, i32, i32) -> ()\n",
      o_memref, state_memref));
  absl::StrAppend(&aie_mlir, absl::StrFormat(
      "    func.func private @attention_block_bf16("
      "%s, %s, %s, %s, %s, i32, i32, i32) -> ()\n",
      q_memref, kv_memref, kv_memref, o_memref, state_memref));
  absl::StrAppend(&aie_mlir, absl::StrFormat(
      "    func.func private @attention_normalize_bf16(%s, %s, i32, i32) -> ()\n",
      o_memref, state_memref));

  // Pass 3: Core bodies — scf.for loop over KV blocks.
  for (int c = 0; c < num_cores; c++) {
    int col = start_col + c;
    std::string compute = absl::StrFormat("tile_%d_%d", col, compute_row);
    std::string q_subview_ty = absl::StrFormat(
        "!aie.objectfifosubview<%s>", q_memref);
    std::string kv_subview_ty = absl::StrFormat(
        "!aie.objectfifosubview<%s>", kv_memref);
    std::string o_subview_ty = absl::StrFormat(
        "!aie.objectfifosubview<%s>", o_memref);

    absl::StrAppend(&aie_mlir, absl::StrFormat(
        "    %%core_%d_%d = aie.core(%%%s) {\n", col, compute_row, compute));

    // Acquire Q and O (held for entire computation).
    absl::StrAppend(&aie_mlir, absl::StrFormat(
        "      %%q_sub = aie.objectfifo.acquire @%s(Consume, 1) : %s\n"
        "      %%q_buf = aie.objectfifo.subview.access %%q_sub[0] "
        ": %s -> %s\n",
        fifo_name(c, "in_q"), q_subview_ty, q_subview_ty, q_memref));
    absl::StrAppend(&aie_mlir, absl::StrFormat(
        "      %%o_sub = aie.objectfifo.acquire @%s(Produce, 1) : %s\n"
        "      %%o_buf = aie.objectfifo.subview.access %%o_sub[0] "
        ": %s -> %s\n",
        fifo_name(c, "out"), o_subview_ty, o_subview_ty, o_memref));

    // Initialize O=0, state=(max=-large, sum=0).
    absl::StrAppend(&aie_mlir, absl::StrFormat(
        "      %%m_pc = arith.constant %d : i32\n"
        "      %%sb_val = arith.constant %d : i32\n"
        "      %%dk_val = arith.constant %d : i32\n",
        m_per_core, sb, info.dk));
    if (info.is_causal) {
      // Causal: pass m_offset (= c * m_per_core) as the m_pc arg to init.
      // Init stores it in the state buffer for the block kernel to read.
      absl::StrAppend(&aie_mlir, absl::StrFormat(
          "      %%m_offset = arith.constant %d : i32\n",
          c * m_per_core));
      absl::StrAppend(&aie_mlir, absl::StrFormat(
          "      func.call @attention_init_bf16(%%o_buf, %%state_%d, "
          "%%m_offset, %%dk_val) : (%s, %s, i32, i32) -> ()\n",
          c, o_memref, state_memref));
    } else {
      absl::StrAppend(&aie_mlir, absl::StrFormat(
          "      func.call @attention_init_bf16(%%o_buf, %%state_%d, "
          "%%m_pc, %%dk_val) : (%s, %s, i32, i32) -> ()\n",
          c, o_memref, state_memref));
    }

    // scf.for loop over KV blocks.
    absl::StrAppend(&aie_mlir, absl::StrFormat(
        "      %%c0 = arith.constant 0 : index\n"
        "      %%c1 = arith.constant 1 : index\n"
        "      %%num_blocks = arith.constant %d : index\n",
        num_blocks));
    absl::StrAppend(&aie_mlir,
        "      scf.for %b = %c0 to %num_blocks step %c1 {\n");

    // Acquire KV (2 elements): slot[0]=K_block, slot[1]=V_block.
    absl::StrAppend(&aie_mlir, absl::StrFormat(
        "        %%kv_sub = aie.objectfifo.acquire @in_kv(Consume, 2) : %s\n"
        "        %%k_buf = aie.objectfifo.subview.access %%kv_sub[0] "
        ": %s -> %s\n"
        "        %%v_buf = aie.objectfifo.subview.access %%kv_sub[1] "
        ": %s -> %s\n",
        kv_subview_ty,
        kv_subview_ty, kv_memref, kv_subview_ty, kv_memref));

    // Call block kernel. Same 8-param signature for causal and non-causal.
    // Causal: pass s_offset (= b * sb) as 7th param instead of sb_val.
    if (info.is_causal) {
      absl::StrAppend(&aie_mlir,
          "        %b_idx = arith.index_cast %b : index to i32\n"
          "        %s_offset = arith.muli %b_idx, %sb_val : i32\n");
      absl::StrAppend(&aie_mlir, absl::StrFormat(
          "        func.call @attention_block_bf16(%%q_buf, %%k_buf, %%v_buf, "
          "%%o_buf, %%state_%d, %%m_pc, %%s_offset, %%dk_val) "
          ": (%s, %s, %s, %s, %s, i32, i32, i32) -> ()\n",
          c, q_memref, kv_memref, kv_memref, o_memref, state_memref));
    } else {
      absl::StrAppend(&aie_mlir, absl::StrFormat(
          "        func.call @attention_block_bf16(%%q_buf, %%k_buf, %%v_buf, "
          "%%o_buf, %%state_%d, %%m_pc, %%sb_val, %%dk_val) "
          ": (%s, %s, %s, %s, %s, i32, i32, i32) -> ()\n",
          c, q_memref, kv_memref, kv_memref, o_memref, state_memref));
    }

    // Release KV (2 elements).
    absl::StrAppend(&aie_mlir,
        "        aie.objectfifo.release @in_kv(Consume, 2)\n"
        "      }\n");  // end scf.for

    // Final normalization: O /= total_sum.
    absl::StrAppend(&aie_mlir, absl::StrFormat(
        "      func.call @attention_normalize_bf16(%%o_buf, %%state_%d, "
        "%%m_pc, %%dk_val) : (%s, %s, i32, i32) -> ()\n",
        c, o_memref, state_memref));

    // Release Q and O.
    absl::StrAppend(&aie_mlir, absl::StrFormat(
        "      aie.objectfifo.release @%s(Produce, 1)\n"
        "      aie.objectfifo.release @%s(Consume, 1)\n",
        fifo_name(c, "out"), fifo_name(c, "in_q")));
    absl::StrAppend(&aie_mlir,
        "      aie.end\n    } { stack_size = 0x2000 : i32 }\n");
  }

  // Runtime sequence (DMA): 4 args = Q, K, V, O.
  absl::StrAppend(&aie_mlir, absl::StrFormat(
      "    aie.runtime_sequence(%%arg0: %s, %%arg1: %s, %%arg2: %s, "
      "%%arg3: %s) {\n",
      q_host_memref, k_host_memref, v_host_memref, o_host_memref));

  int dma_id = 0;

  // Q per-column: sent first (needed before KV loop starts).
  for (int c = 0; c < num_cores; c++) {
    int64_t q_offset = c * m_per_core * info.dk;
    absl::StrAppend(&aie_mlir, absl::StrFormat(
        "      aiex.npu.dma_memcpy_nd(%%arg0"
        "[0, 0, 0, %d][1, 1, %d, %d][0, 0, %d, 1]) "
        "{id = %d : i64, metadata = @%s, issue_token = true} : %s\n",
        q_offset, m_per_core, info.dk, info.dk,
        dma_id++, fifo_name(c, "in_q_L2"), q_host_memref));
  }

  // KV blocks interleaved: for each block, send K_block then V_block
  // to the combined KV FIFO. K fills slot[0], V fills slot[1].
  // CRITICAL: dma_wait after each K/V pair to prevent BD interference.
  // Without this, the instruction processor submits later BDs before
  // current ones complete, causing ERT_CMD_STATE_TIMEOUT with >2 blocks.
  for (int64_t b = 0; b < num_blocks; b++) {
    int64_t block_offset = b * sb * info.dk;
    // K block → slot[0].
    absl::StrAppend(&aie_mlir, absl::StrFormat(
        "      aiex.npu.dma_memcpy_nd(%%arg1"
        "[0, 0, 0, %d][1, 1, %d, %d][0, 0, %d, 1]) "
        "{id = %d : i64, metadata = @in_kv_L2, issue_token = true} : %s\n",
        block_offset, sb, info.dk, info.dk, dma_id++, k_host_memref));
    // V block → slot[1].
    absl::StrAppend(&aie_mlir, absl::StrFormat(
        "      aiex.npu.dma_memcpy_nd(%%arg2"
        "[0, 0, 0, %d][1, 1, %d, %d][0, 0, %d, 1]) "
        "{id = %d : i64, metadata = @in_kv_L2, issue_token = true} : %s\n",
        block_offset, sb, info.dk, info.dk, dma_id++, v_host_memref));
    // Wait for this KV pair to complete before submitting the next.
    absl::StrAppend(&aie_mlir,
        "      aiex.npu.dma_wait { symbol = @in_kv_L2 }\n");
  }

  // O per-column.
  for (int c = 0; c < num_cores; c++) {
    int64_t o_offset = c * m_per_core * info.dk;
    absl::StrAppend(&aie_mlir, absl::StrFormat(
        "      aiex.npu.dma_memcpy_nd(%%arg3"
        "[0, 0, 0, %d][1, 1, %d, %d][0, 0, %d, 1]) "
        "{id = %d : i64, metadata = @%s, issue_token = true} : %s\n",
        o_offset, m_per_core, info.dk, info.dk,
        dma_id++, fifo_name(c, "out_L2"), o_host_memref));
  }

  // Wait for all channels.
  absl::StrAppend(&aie_mlir,
      "      aiex.npu.dma_wait { symbol = @in_kv_L2 }\n");
  for (int c = 0; c < num_cores; c++) {
    absl::StrAppend(&aie_mlir, absl::StrFormat(
        "      aiex.npu.dma_wait { symbol = @%s }\n",
        fifo_name(c, "in_q_L2")));
    absl::StrAppend(&aie_mlir, absl::StrFormat(
        "      aiex.npu.dma_wait { symbol = @%s }\n",
        fifo_name(c, "out_L2")));
  }

  absl::StrAppend(&aie_mlir, "    }\n");  // end runtime_sequence
  absl::StrAppend(&aie_mlir, "  }\n");    // end aie.device
  absl::StrAppend(&aie_mlir, "}\n");      // end module

  LOG(INFO) << "XDNA tiled attention lowering: generated AIE MLIR:\n"
            << aie_mlir;

  AieLoweringResult result;
  result.aie_mlir = aie_mlir;
  result.num_cores = num_cores;
  result.needs_softfloat_stubs = true;
  result.needs_attention_kernel = true;
  result.attention_seq_len = info.seq_len;
  result.attention_dk = info.dk;
  result.attention_m_per_core = m_per_core;
  result.attention_kv_block_size = sb;
  result.attention_is_causal = info.is_causal;
  // Sanity: the MLIR passes m_per_core/sb as runtime i32 args, but the kernel
  // uses CONST_M_PC/CONST_SB compile-time defines. These must agree.
  CHECK_GT(result.attention_m_per_core, 0);
  CHECK_GT(result.attention_kv_block_size, 0);
  CHECK_EQ(info.seq_len % sb, 0)
      << "seq_len=" << info.seq_len << " not divisible by Sb=" << sb;
  return result;
}

// Generates AIE MLIR for fused attention using an external C++ kernel.
// Each core processes M_per_core rows of the attention output.
// For small seq_len (fits in L1): non-tiled path — full K/V in L1.
// For large seq_len: tiled path — streams K/V blocks through L1.
absl::StatusOr<AieLoweringResult> LowerAttentionToAie(
    const AttentionProgramInfo& info, const TargetCaps& caps,
    int max_columns) {
  // Auto-scale number of cores.
  int num_cores = std::min(max_columns, caps.num_columns);
  while (num_cores > 1) {
    if (info.num_rows % num_cores != 0) { num_cores--; continue; }
    break;
  }
  int64_t m_per_core = info.num_rows / num_cores;

  // Check if non-tiled path fits in L1.
  // L1 budget: Q (depth 2) + KV (depth 2, holds K and V) + O (depth 2).
  // K and V share a single ObjectFIFO (depth=2): core acquires 2 elements,
  // slot[0]=K and slot[1]=V. Routed through mem tile (shim→mem→compute).
  // Stateful transform creates depth+1=3 buffers for acquire(2).
  int64_t q_l1 = 2 * m_per_core * info.dk * 2;  // depth=2, 2 buffers
  int64_t kv_l1 = 3 * info.seq_len * info.dk * 2;  // depth=2 + acquire(2) → 3 buffers
  int64_t o_l1 = 2 * m_per_core * info.dk * 2;  // depth=2, 2 buffers
  int64_t non_tiled_l1 = q_l1 + kv_l1 + o_l1;

  if (non_tiled_l1 > caps.l1_usable_bytes) {
    // Tiled path: stream K/V in blocks.
    int64_t sb = SelectKvBlockSize(info.seq_len, info.dk, m_per_core,
                                    caps.l1_usable_bytes);
    LOG(INFO) << "XDNA attention: non-tiled L1=" << non_tiled_l1
              << "B exceeds budget, using tiled path with Sb=" << sb;
    return LowerTiledAttentionToAie(info, caps, num_cores, m_per_core, sb);
  }

  // Non-tiled path: full K/V fit in L1.

  // DMA dimension limits.
  int64_t d0_words_dk = info.dk * 2 / 4;  // bf16 elements → 32-bit words
  if (d0_words_dk > kMaxBdWords) {
    return absl::InvalidArgumentError(absl::StrFormat(
        "Attention dk=%d exceeds DMA d0 limit (%d words, max %d)",
        info.dk, d0_words_dk, kMaxBdWords));
  }
  // K/V DMA: d1 = seq_len (rows transferred).
  if (info.seq_len > kMaxBdWords) {
    return absl::InvalidArgumentError(absl::StrFormat(
        "Attention seq_len=%d exceeds DMA d1 limit (max %d)",
        info.seq_len, kMaxBdWords));
  }
  // Q/O DMA: d1 = m_per_core.
  if (m_per_core > kMaxBdWords) {
    return absl::InvalidArgumentError(absl::StrFormat(
        "Attention m_per_core=%d exceeds DMA d1 limit (max %d)",
        m_per_core, kMaxBdWords));
  }

  LOG(INFO) << "XDNA attention lowering: M=" << info.num_rows
            << " S=" << info.seq_len << " dk=" << info.dk
            << " using " << num_cores << " core(s) ("
            << m_per_core << " rows/core)"
            << " L1=" << non_tiled_l1 << "B";

  int start_col = caps.partition_start_column;
  int shim_row = caps.shim_row;
  int compute_row = caps.first_compute_row;
  int fifo_depth = 2;

  // ObjectFIFO buffer types (flattened 1D for simplicity).
  std::string q_memref = absl::StrFormat("memref<%dxbf16>",
                                          m_per_core * info.dk);
  // K and V share one ObjectFIFO — each element is S*dk bf16.
  // The FIFO has depth=2: slot 0 = K, slot 1 = V.
  std::string kv_memref = absl::StrFormat("memref<%dxbf16>",
                                           info.seq_len * info.dk);
  std::string o_memref = absl::StrFormat("memref<%dxbf16>",
                                          m_per_core * info.dk);

  // Host buffer types (flat, full size).
  int64_t q_total = info.num_rows * info.dk;
  int64_t kv_total = info.seq_len * info.dk;
  int64_t o_total = info.num_rows * info.dk;
  std::string q_host_memref = absl::StrFormat("memref<%dxbf16>", q_total);
  std::string k_host_memref = absl::StrFormat("memref<%dxbf16>", kv_total);
  std::string v_host_memref = absl::StrFormat("memref<%dxbf16>", kv_total);
  std::string o_host_memref = absl::StrFormat("memref<%dxbf16>", o_total);

  auto fifo_name = [&](int c, const std::string& base) -> std::string {
    return num_cores > 1 ? absl::StrFormat("c%d_%s", c, base) : base;
  };

  std::string aie_mlir;
  absl::StrAppend(&aie_mlir, "module {\n");
  absl::StrAppend(&aie_mlir, absl::StrFormat(
      "  aie.device(%s) {\n", DeviceNameForColumns(num_cores)));

  // Pass 1: Tile declarations.
  for (int c = 0; c < num_cores; c++) {
    int col = start_col + c;
    absl::StrAppend(&aie_mlir, absl::StrFormat(
        "    %%tile_%d_%d = aie.tile(%d, %d)\n", col, shim_row, col, shim_row));
    absl::StrAppend(&aie_mlir, absl::StrFormat(
        "    %%tile_%d_%d = aie.tile(%d, %d)\n",
        col, caps.mem_tile_row, col, caps.mem_tile_row));
    absl::StrAppend(&aie_mlir, absl::StrFormat(
        "    %%tile_%d_%d = aie.tile(%d, %d)\n",
        col, compute_row, col, compute_row));
  }

  // Pass 2: ObjectFIFOs and links.
  // KV broadcast: single chain from start_col shim → start_col mem → all
  // compute tiles. K and V share one ObjectFIFO (depth=2). Shim sends K
  // (slot 0) then V (slot 1). All cores acquire from the same broadcast FIFO.
  // This sends KV once from host instead of N times (one per core).
  {
    std::string kv_shim = absl::StrFormat("tile_%d_%d", start_col, shim_row);
    std::string kv_mem = absl::StrFormat("tile_%d_%d", start_col,
                                          caps.mem_tile_row);
    // L2: shim(start_col) → mem(start_col).
    absl::StrAppend(&aie_mlir, absl::StrFormat(
        "    aie.objectfifo @in_kv_L2(%%%s, {%%%s}, %d : i32) "
        ": !aie.objectfifo<%s>\n",
        kv_shim, kv_mem, fifo_depth, kv_memref));
    // L1 broadcast: mem(start_col) → {all compute tiles}.
    std::string kv_consumers;
    for (int c = 0; c < num_cores; c++) {
      if (c > 0) kv_consumers += ", ";
      absl::StrAppendFormat(&kv_consumers, "%%tile_%d_%d",
                            start_col + c, compute_row);
    }
    absl::StrAppend(&aie_mlir, absl::StrFormat(
        "    aie.objectfifo @in_kv(%%%s, {%s}, %d : i32) "
        ": !aie.objectfifo<%s>\n",
        kv_mem, kv_consumers, fifo_depth, kv_memref));
    absl::StrAppend(&aie_mlir,
        "    aie.objectfifo.link [@in_kv_L2] -> [@in_kv] ([] [])\n");
  }

  // Q and O per-column: each column has its own shim ↔ mem ↔ compute chain.
  for (int c = 0; c < num_cores; c++) {
    int col = start_col + c;
    std::string shim = absl::StrFormat("tile_%d_%d", col, shim_row);
    std::string mem = absl::StrFormat("tile_%d_%d", col, caps.mem_tile_row);
    std::string compute = absl::StrFormat("tile_%d_%d", col, compute_row);

    // Q input: shim → mem → compute.
    absl::StrAppend(&aie_mlir, absl::StrFormat(
        "    aie.objectfifo @%s(%%%s, {%%%s}, %d : i32) "
        ": !aie.objectfifo<%s>\n",
        fifo_name(c, "in_q_L2"), shim, mem, fifo_depth, q_memref));
    absl::StrAppend(&aie_mlir, absl::StrFormat(
        "    aie.objectfifo @%s(%%%s, {%%%s}, %d : i32) "
        ": !aie.objectfifo<%s>\n",
        fifo_name(c, "in_q"), mem, compute, fifo_depth, q_memref));
    absl::StrAppend(&aie_mlir, absl::StrFormat(
        "    aie.objectfifo.link [@%s] -> [@%s] ([] [])\n",
        fifo_name(c, "in_q_L2"), fifo_name(c, "in_q")));

    // O output: compute → mem → shim.
    absl::StrAppend(&aie_mlir, absl::StrFormat(
        "    aie.objectfifo @%s(%%%s, {%%%s}, %d : i32) "
        ": !aie.objectfifo<%s>\n",
        fifo_name(c, "out"), compute, mem, fifo_depth, o_memref));
    absl::StrAppend(&aie_mlir, absl::StrFormat(
        "    aie.objectfifo @%s(%%%s, {%%%s}, %d : i32) "
        ": !aie.objectfifo<%s>\n",
        fifo_name(c, "out_L2"), mem, shim, fifo_depth, o_memref));
    absl::StrAppend(&aie_mlir, absl::StrFormat(
        "    aie.objectfifo.link [@%s] -> [@%s] ([] [])\n",
        fifo_name(c, "out"), fifo_name(c, "out_L2")));
  }

  // External function declaration (resolved at link time from attention_kernel.o).
  if (info.is_causal) {
    // Causal kernel has extra i32 param: m_offset.
    absl::StrAppend(&aie_mlir, absl::StrFormat(
        "    func.func private @attention_bf16("
        "%s, %s, %s, %s, i32, i32, i32, i32) -> ()\n",
        q_memref, kv_memref, kv_memref, o_memref));
  } else {
    absl::StrAppend(&aie_mlir, absl::StrFormat(
        "    func.func private @attention_bf16("
        "%s, %s, %s, %s, i32, i32, i32) -> ()\n",
        q_memref, kv_memref, kv_memref, o_memref));
  }

  // Pass 3: Core bodies — single acquire/compute/release.
  for (int c = 0; c < num_cores; c++) {
    int col = start_col + c;
    std::string compute = absl::StrFormat("tile_%d_%d", col, compute_row);
    std::string q_subview_ty = absl::StrFormat(
        "!aie.objectfifosubview<%s>", q_memref);
    std::string kv_subview_ty = absl::StrFormat(
        "!aie.objectfifosubview<%s>", kv_memref);
    std::string o_subview_ty = absl::StrFormat(
        "!aie.objectfifosubview<%s>", o_memref);

    absl::StrAppend(&aie_mlir, absl::StrFormat(
        "    %%core_%d_%d = aie.core(%%%s) {\n", col, compute_row, compute));
    // Acquire Q (1 element).
    absl::StrAppend(&aie_mlir, absl::StrFormat(
        "      %%q_sub = aie.objectfifo.acquire @%s(Consume, 1) : %s\n"
        "      %%q_buf = aie.objectfifo.subview.access %%q_sub[0] "
        ": %s -> %s\n",
        fifo_name(c, "in_q"), q_subview_ty, q_subview_ty, q_memref));
    // Acquire KV (2 elements): slot[0]=K, slot[1]=V.
    // All cores acquire from the shared broadcast FIFO @in_kv.
    absl::StrAppend(&aie_mlir, absl::StrFormat(
        "      %%kv_sub = aie.objectfifo.acquire @in_kv(Consume, 2) : %s\n"
        "      %%k_buf = aie.objectfifo.subview.access %%kv_sub[0] "
        ": %s -> %s\n"
        "      %%v_buf = aie.objectfifo.subview.access %%kv_sub[1] "
        ": %s -> %s\n",
        kv_subview_ty,
        kv_subview_ty, kv_memref, kv_subview_ty, kv_memref));
    // Acquire O (1 element).
    absl::StrAppend(&aie_mlir, absl::StrFormat(
        "      %%o_sub = aie.objectfifo.acquire @%s(Produce, 1) : %s\n"
        "      %%o_buf = aie.objectfifo.subview.access %%o_sub[0] "
        ": %s -> %s\n",
        fifo_name(c, "out"), o_subview_ty, o_subview_ty, o_memref));
    absl::StrAppend(&aie_mlir, absl::StrFormat(
        "      %%m_pc = arith.constant %d : i32\n"
        "      %%seq = arith.constant %d : i32\n"
        "      %%dk = arith.constant %d : i32\n",
        m_per_core, info.seq_len, info.dk));
    if (info.is_causal) {
      absl::StrAppend(&aie_mlir, absl::StrFormat(
          "      %%m_offset = arith.constant %d : i32\n",
          c * m_per_core));
      absl::StrAppend(&aie_mlir, absl::StrFormat(
          "      func.call @attention_bf16(%%q_buf, %%k_buf, %%v_buf, "
          "%%o_buf, %%m_pc, %%seq, %%dk, %%m_offset) "
          ": (%s, %s, %s, %s, i32, i32, i32, i32) -> ()\n",
          q_memref, kv_memref, kv_memref, o_memref));
    } else {
      absl::StrAppend(&aie_mlir, absl::StrFormat(
          "      func.call @attention_bf16(%%q_buf, %%k_buf, %%v_buf, "
          "%%o_buf, %%m_pc, %%seq, %%dk) "
          ": (%s, %s, %s, %s, i32, i32, i32) -> ()\n",
          q_memref, kv_memref, kv_memref, o_memref));
    }
    absl::StrAppend(&aie_mlir, absl::StrFormat(
        "      aie.objectfifo.release @%s(Produce, 1)\n"
        "      aie.objectfifo.release @in_kv(Consume, 2)\n"
        "      aie.objectfifo.release @%s(Consume, 1)\n",
        fifo_name(c, "out"), fifo_name(c, "in_q")));
    // stack_size = 0x2000 for attention kernel.
    absl::StrAppend(&aie_mlir,
        "      aie.end\n    } { stack_size = 0x2000 : i32 }\n");
  }

  // Runtime sequence (DMA): 4 args = Q, K, V, O.
  absl::StrAppend(&aie_mlir, absl::StrFormat(
      "    aie.runtime_sequence(%%arg0: %s, %%arg1: %s, %%arg2: %s, "
      "%%arg3: %s) {\n",
      q_host_memref, k_host_memref, v_host_memref, o_host_memref));

  // KV broadcast: send K and V once to @in_kv_L2. The ObjectFIFO broadcast
  // replicates data to all compute tiles via stream routing.
  int dma_id = 0;

  // K DMA: transfer full K to broadcast KV ObjectFIFO (fills slot 0).
  absl::StrAppend(&aie_mlir, absl::StrFormat(
      "      aiex.npu.dma_memcpy_nd(%%arg1"
      "[0, 0, 0, 0][1, 1, %d, %d][0, 0, %d, 1]) "
      "{id = %d : i64, metadata = @in_kv_L2, issue_token = true} : %s\n",
      info.seq_len, info.dk, info.dk, dma_id++, k_host_memref));

  // V DMA: transfer full V to broadcast KV ObjectFIFO (fills slot 1).
  absl::StrAppend(&aie_mlir, absl::StrFormat(
      "      aiex.npu.dma_memcpy_nd(%%arg2"
      "[0, 0, 0, 0][1, 1, %d, %d][0, 0, %d, 1]) "
      "{id = %d : i64, metadata = @in_kv_L2, issue_token = true} : %s\n",
      info.seq_len, info.dk, info.dk, dma_id++, v_host_memref));

  // Q per-column: each core gets its M_per_core rows.
  for (int c = 0; c < num_cores; c++) {
    int64_t q_offset = c * m_per_core * info.dk;
    absl::StrAppend(&aie_mlir, absl::StrFormat(
        "      aiex.npu.dma_memcpy_nd(%%arg0"
        "[0, 0, 0, %d][1, 1, %d, %d][0, 0, %d, 1]) "
        "{id = %d : i64, metadata = @%s, issue_token = true} : %s\n",
        q_offset, m_per_core, info.dk, info.dk,
        dma_id++, fifo_name(c, "in_q_L2"), q_host_memref));
  }

  // O per-column: each core writes its M_per_core rows.
  for (int c = 0; c < num_cores; c++) {
    int64_t o_offset = c * m_per_core * info.dk;
    absl::StrAppend(&aie_mlir, absl::StrFormat(
        "      aiex.npu.dma_memcpy_nd(%%arg3"
        "[0, 0, 0, %d][1, 1, %d, %d][0, 0, %d, 1]) "
        "{id = %d : i64, metadata = @%s, issue_token = true} : %s\n",
        o_offset, m_per_core, info.dk, info.dk,
        dma_id++, fifo_name(c, "out_L2"), o_host_memref));
  }

  // Wait for all channels to complete.
  absl::StrAppend(&aie_mlir,
      "      aiex.npu.dma_wait { symbol = @in_kv_L2 }\n");
  for (int c = 0; c < num_cores; c++) {
    absl::StrAppend(&aie_mlir, absl::StrFormat(
        "      aiex.npu.dma_wait { symbol = @%s }\n",
        fifo_name(c, "in_q_L2")));
    absl::StrAppend(&aie_mlir, absl::StrFormat(
        "      aiex.npu.dma_wait { symbol = @%s }\n",
        fifo_name(c, "out_L2")));
  }

  absl::StrAppend(&aie_mlir, "    }\n");  // end runtime_sequence
  absl::StrAppend(&aie_mlir, "  }\n");    // end aie.device
  absl::StrAppend(&aie_mlir, "}\n");      // end module

  LOG(INFO) << "XDNA attention lowering: generated AIE MLIR:\n" << aie_mlir;

  AieLoweringResult result;
  result.aie_mlir = aie_mlir;
  result.num_cores = num_cores;
  result.needs_softfloat_stubs = true;
  result.needs_attention_kernel = true;
  result.attention_seq_len = info.seq_len;
  result.attention_dk = info.dk;
  result.attention_m_per_core = m_per_core;
  result.attention_is_causal = info.is_causal;
  return result;
}

// Generates AIE MLIR for softmax using an external C++ kernel.
// Each core processes rows_per_core rows, calling softmax_bf16 per row.
absl::StatusOr<AieLoweringResult> LowerSoftmaxToAie(
    const SoftmaxProgramInfo& info, const TargetCaps& caps,
    int max_columns) {
  // Auto-scale number of cores.
  int num_cores = std::min(max_columns, caps.num_columns);
  while (num_cores > 1) {
    if (info.num_rows % num_cores != 0) { num_cores--; continue; }
    break;
  }
  int64_t rows_per_core = info.num_rows / num_cores;
  int64_t total_elements = info.num_rows * info.row_length;

  // L1 budget: 2 FIFOs (in + out) x 2 ping-pong x row_length x 2 bytes.
  int64_t l1_per_row = 2 * 2 * info.row_length * 2;  // 8 * row_length
  if (l1_per_row > caps.l1_usable_bytes) {
    return absl::InvalidArgumentError(absl::StrFormat(
        "Softmax row_length=%d exceeds L1 budget (%d bytes needed, %d available)",
        info.row_length, l1_per_row, caps.l1_usable_bytes));
  }

  // DMA dimension limits (10-bit fields, max 1023 32-bit words).
  int64_t d0_words = info.row_length * 2 / 4;  // bf16 elements → 32-bit words
  if (d0_words > kMaxBdWords) {
    return absl::InvalidArgumentError(absl::StrFormat(
        "Softmax row_length=%d exceeds DMA d0 limit (%d words, max %d)",
        info.row_length, d0_words, kMaxBdWords));
  }
  if (rows_per_core > kMaxBdWords) {
    return absl::InvalidArgumentError(absl::StrFormat(
        "Softmax rows_per_core=%d exceeds DMA d1 limit (max %d)",
        rows_per_core, kMaxBdWords));
  }

  LOG(INFO) << "XDNA softmax lowering: " << info.num_rows << " rows x "
            << info.row_length << " " << info.element_type
            << " using " << num_cores << " core(s) ("
            << rows_per_core << " rows/core)";

  int start_col = caps.partition_start_column;
  int shim_row = caps.shim_row;
  int compute_row = caps.first_compute_row;
  int fifo_depth = 2;

  std::string row_memref = absl::StrFormat("memref<%dxbf16>", info.row_length);
  std::string host_memref = absl::StrFormat("memref<%dxbf16>", total_elements);

  auto fifo_name = [&](int c, const std::string& base) -> std::string {
    return num_cores > 1 ? absl::StrFormat("c%d_%s", c, base) : base;
  };

  std::string aie_mlir;
  absl::StrAppend(&aie_mlir, "module {\n");
  absl::StrAppend(&aie_mlir, absl::StrFormat(
      "  aie.device(%s) {\n", DeviceNameForColumns(num_cores)));

  // Pass 1: Tile declarations.
  for (int c = 0; c < num_cores; c++) {
    int col = start_col + c;
    absl::StrAppend(&aie_mlir, absl::StrFormat(
        "    %%tile_%d_%d = aie.tile(%d, %d)\n", col, shim_row, col, shim_row));
    absl::StrAppend(&aie_mlir, absl::StrFormat(
        "    %%tile_%d_%d = aie.tile(%d, %d)\n",
        col, caps.mem_tile_row, col, caps.mem_tile_row));
    absl::StrAppend(&aie_mlir, absl::StrFormat(
        "    %%tile_%d_%d = aie.tile(%d, %d)\n",
        col, compute_row, col, compute_row));
  }

  // Pass 2: ObjectFIFOs and links (shim ↔ mem ↔ compute).
  for (int c = 0; c < num_cores; c++) {
    int col = start_col + c;
    std::string shim = absl::StrFormat("tile_%d_%d", col, shim_row);
    std::string mem = absl::StrFormat("tile_%d_%d", col, caps.mem_tile_row);
    std::string compute = absl::StrFormat("tile_%d_%d", col, compute_row);

    // Input: shim → mem (L2) → compute (L1).
    absl::StrAppend(&aie_mlir, absl::StrFormat(
        "    aie.objectfifo @%s(%%%s, {%%%s}, %d : i32) "
        ": !aie.objectfifo<%s>\n",
        fifo_name(c, "in_L2"), shim, mem, fifo_depth, row_memref));
    absl::StrAppend(&aie_mlir, absl::StrFormat(
        "    aie.objectfifo @%s(%%%s, {%%%s}, %d : i32) "
        ": !aie.objectfifo<%s>\n",
        fifo_name(c, "in"), mem, compute, fifo_depth, row_memref));
    absl::StrAppend(&aie_mlir, absl::StrFormat(
        "    aie.objectfifo.link [@%s] -> [@%s] ([] [])\n",
        fifo_name(c, "in_L2"), fifo_name(c, "in")));

    // Output: compute → mem (L2) → shim.
    absl::StrAppend(&aie_mlir, absl::StrFormat(
        "    aie.objectfifo @%s(%%%s, {%%%s}, %d : i32) "
        ": !aie.objectfifo<%s>\n",
        fifo_name(c, "out"), compute, mem, fifo_depth, row_memref));
    absl::StrAppend(&aie_mlir, absl::StrFormat(
        "    aie.objectfifo @%s(%%%s, {%%%s}, %d : i32) "
        ": !aie.objectfifo<%s>\n",
        fifo_name(c, "out_L2"), mem, shim, fifo_depth, row_memref));
    absl::StrAppend(&aie_mlir, absl::StrFormat(
        "    aie.objectfifo.link [@%s] -> [@%s] ([] [])\n",
        fifo_name(c, "out"), fifo_name(c, "out_L2")));
  }

  // External function declaration (resolved at link time from softmax_kernel.o).
  absl::StrAppend(&aie_mlir, absl::StrFormat(
      "    func.func private @softmax_bf16(%s, %s, i32) -> ()\n",
      row_memref, row_memref));

  // Pass 3: Core bodies.
  for (int c = 0; c < num_cores; c++) {
    int col = start_col + c;
    std::string compute = absl::StrFormat("tile_%d_%d", col, compute_row);
    std::string subview_ty = absl::StrFormat(
        "!aie.objectfifosubview<%s>", row_memref);

    absl::StrAppend(&aie_mlir, absl::StrFormat(
        "    %%core_%d_%d = aie.core(%%%s) {\n", col, compute_row, compute));
    absl::StrAppend(&aie_mlir, absl::StrFormat(
        "      %%c0 = arith.constant 0 : index\n"
        "      %%c1 = arith.constant 1 : index\n"
        "      %%num_rows = arith.constant %d : index\n"
        "      %%row_len = arith.constant %d : i32\n",
        rows_per_core, info.row_length));
    absl::StrAppend(&aie_mlir, absl::StrFormat(
        "      scf.for %%i = %%c0 to %%num_rows step %%c1 {\n"
        "        %%in_sub = aie.objectfifo.acquire @%s(Consume, 1) : %s\n"
        "        %%in_buf = aie.objectfifo.subview.access %%in_sub[0] "
        ": %s -> %s\n"
        "        %%out_sub = aie.objectfifo.acquire @%s(Produce, 1) : %s\n"
        "        %%out_buf = aie.objectfifo.subview.access %%out_sub[0] "
        ": %s -> %s\n"
        "        func.call @softmax_bf16(%%in_buf, %%out_buf, %%row_len) "
        ": (%s, %s, i32) -> ()\n"
        "        aie.objectfifo.release @%s(Consume, 1)\n"
        "        aie.objectfifo.release @%s(Produce, 1)\n"
        "      }\n",
        fifo_name(c, "in"), subview_ty, subview_ty, row_memref,
        fifo_name(c, "out"), subview_ty, subview_ty, row_memref,
        row_memref, row_memref,
        fifo_name(c, "in"), fifo_name(c, "out")));
    absl::StrAppend(&aie_mlir, "      aie.end\n    }\n");
  }

  // Runtime sequence (DMA).
  absl::StrAppend(&aie_mlir, absl::StrFormat(
      "    aie.runtime_sequence(%%arg0: %s, %%arg1: %s) {\n",
      host_memref, host_memref));

  int dma_id = 0;
  for (int c = 0; c < num_cores; c++) {
    int64_t offset = c * rows_per_core * info.row_length;

    // Input DMA: transfer rows_per_core rows starting at offset.
    absl::StrAppend(&aie_mlir, absl::StrFormat(
        "      aiex.npu.dma_memcpy_nd(%%arg0"
        "[0, 0, 0, %d][1, 1, %d, %d][0, 0, %d, 1]) "
        "{id = %d : i64, metadata = @%s, issue_token = true} : %s\n",
        offset, rows_per_core, info.row_length, info.row_length,
        dma_id++, fifo_name(c, "in_L2"), host_memref));

    // Output DMA: receive rows_per_core rows at same offset.
    absl::StrAppend(&aie_mlir, absl::StrFormat(
        "      aiex.npu.dma_memcpy_nd(%%arg1"
        "[0, 0, 0, %d][1, 1, %d, %d][0, 0, %d, 1]) "
        "{id = %d : i64, metadata = @%s, issue_token = true} : %s\n",
        offset, rows_per_core, info.row_length, info.row_length,
        dma_id++, fifo_name(c, "out_L2"), host_memref));
  }

  // Wait for all DMAs.
  for (int c = 0; c < num_cores; c++) {
    absl::StrAppend(&aie_mlir, absl::StrFormat(
        "      aiex.npu.dma_wait { symbol = @%s }\n",
        fifo_name(c, "out_L2")));
    absl::StrAppend(&aie_mlir, absl::StrFormat(
        "      aiex.npu.dma_wait { symbol = @%s }\n",
        fifo_name(c, "in_L2")));
  }

  absl::StrAppend(&aie_mlir, "    }\n");  // end runtime_sequence
  absl::StrAppend(&aie_mlir, "  }\n");    // end aie.device
  absl::StrAppend(&aie_mlir, "}\n");      // end module

  LOG(INFO) << "XDNA softmax lowering: generated AIE MLIR:\n" << aie_mlir;

  AieLoweringResult result;
  result.aie_mlir = aie_mlir;
  result.num_cores = num_cores;
  result.needs_softfloat_stubs = true;
  result.needs_softmax_kernel = true;
  return result;
}

// Detects a GELU activation pattern in the module.
// XLA decomposes jax.nn.gelu(x, approximate=True) into elementwise ops
// containing math.tanh. Detection checks:
//   1. Function: 1 input, 1 output, same shape/type
//   2. Element type: bf16 only
//   3. At least one linalg.generic body contains math.tanh
//   4. All iterators parallel (no reductions)
//   5. No linalg.matmul ops
std::optional<GeluProgramInfo> DetectGelu(mlir::ModuleOp module) {
  mlir::func::FuncOp entry_func;
  module.walk([&](mlir::func::FuncOp func) {
    if (func.getName() == "main" || !entry_func) entry_func = func;
  });
  if (!entry_func) return std::nullopt;

  // Verify function signature: single input tensor, single result tensor.
  auto func_type = entry_func.getFunctionType();
  if (func_type.getNumInputs() != 1 || func_type.getNumResults() != 1)
    return std::nullopt;

  auto input = mlir::dyn_cast<mlir::RankedTensorType>(func_type.getInput(0));
  auto output = mlir::dyn_cast<mlir::RankedTensorType>(func_type.getResult(0));
  if (!input || !output) return std::nullopt;
  if (input != output) return std::nullopt;

  std::string elem_type = GetElementTypeStr(input.getElementType());
  if (elem_type != "bf16") return std::nullopt;

  // Look for math.tanh as the unique GELU marker. Reject if matmul or
  // reduction iterators are present (those indicate matmul/softmax).
  bool has_tanh = false;
  bool has_matmul = false;
  bool has_reduction = false;

  entry_func.walk([&](mlir::Operation* op) {
    llvm::StringRef name = op->getName().getStringRef();
    if (name == "linalg.matmul") has_matmul = true;
  });

  if (has_matmul) return std::nullopt;

  entry_func.walk([&](mlir::linalg::GenericOp generic) {
    auto iterators = generic.getIteratorTypesArray();
    for (auto it : iterators) {
      if (it == mlir::utils::IteratorType::reduction) {
        has_reduction = true;
        return;
      }
    }
    mlir::Block& body = generic.getRegion().front();
    for (mlir::Operation& body_op : body.without_terminator()) {
      if (body_op.getName().getStringRef() == "math.tanh") has_tanh = true;
    }
  });

  if (!has_tanh || has_reduction) return std::nullopt;

  // Compute total elements.
  int64_t num_elements = 1;
  for (int64_t dim : input.getShape()) num_elements *= dim;

  // GELU kernel uses 32-wide bf16 vectors.
  if (num_elements % 32 != 0) return std::nullopt;
  if (num_elements < 32) return std::nullopt;

  LOG(INFO) << "XDNA: detected GELU pattern: " << num_elements
            << " elements " << elem_type;

  GeluProgramInfo info;
  info.num_elements = num_elements;
  info.element_type = elem_type;
  return info;
}

// Generates AIE MLIR for GELU using an external C++ kernel.
// Same pattern as softmax: 1 input FIFO, 1 output FIFO, chunked processing.
// Multi-core via per-column independent FIFOs (same as softmax).
absl::StatusOr<AieLoweringResult> LowerGeluToAie(
    const GeluProgramInfo& info, const TargetCaps& caps,
    int max_columns) {
  // Auto-scale number of cores.
  int num_cores = std::min(max_columns, caps.num_columns);
  while (num_cores > 1) {
    if (info.num_elements % num_cores != 0) { num_cores--; continue; }
    int64_t per_core = info.num_elements / num_cores;
    if (per_core % 32 != 0) { num_cores--; continue; }
    break;
  }
  int64_t per_core_elements = info.num_elements / num_cores;

  // L1 budget: 2 FIFOs (in + out) x 2 ping-pong x chunk_size x 2 bytes.
  // Choose chunk size that fits L1 and divides per_core_elements.
  int64_t max_chunk = caps.l1_usable_bytes / (2 * 2 * 2);  // 4 * bf16
  // Also cap to shim DMA d0 limit: 1023 32-bit words = 2046 bf16 elements.
  max_chunk = std::min(max_chunk, static_cast<int64_t>(2046));
  // Align chunk to VL=32.
  max_chunk = (max_chunk / 32) * 32;

  int64_t chunk_size;
  int64_t num_chunks;
  if (per_core_elements <= max_chunk) {
    chunk_size = per_core_elements;
    num_chunks = 1;
  } else {
    // Find largest chunk that fits L1 and divides per_core_elements.
    chunk_size = max_chunk;
    while (chunk_size > 32 && per_core_elements % chunk_size != 0)
      chunk_size -= 32;
    if (per_core_elements % chunk_size != 0) {
      return absl::InvalidArgumentError(absl::StrFormat(
          "GELU: cannot tile %d elements into L1 chunks", per_core_elements));
    }
    num_chunks = per_core_elements / chunk_size;
  }

  LOG(INFO) << "XDNA GELU lowering: " << info.num_elements << " elements "
            << info.element_type << " using " << num_cores << " core(s) ("
            << per_core_elements << " per core, " << num_chunks
            << " chunks of " << chunk_size << ")";

  int start_col = caps.partition_start_column;
  int shim_row = caps.shim_row;
  int compute_row = caps.first_compute_row;
  int fifo_depth = 2;

  std::string chunk_memref = absl::StrFormat("memref<%dxbf16>", chunk_size);
  std::string host_memref = absl::StrFormat("memref<%dxbf16>",
                                             info.num_elements);

  auto fifo_name = [&](int c, const std::string& base) -> std::string {
    return num_cores > 1 ? absl::StrFormat("c%d_%s", c, base) : base;
  };

  std::string aie_mlir;
  absl::StrAppend(&aie_mlir, "module {\n");
  absl::StrAppend(&aie_mlir, absl::StrFormat(
      "  aie.device(%s) {\n", DeviceNameForColumns(num_cores)));

  // Tile declarations.
  for (int c = 0; c < num_cores; c++) {
    int col = start_col + c;
    absl::StrAppend(&aie_mlir, absl::StrFormat(
        "    %%tile_%d_%d = aie.tile(%d, %d)\n", col, shim_row, col, shim_row));
    absl::StrAppend(&aie_mlir, absl::StrFormat(
        "    %%tile_%d_%d = aie.tile(%d, %d)\n",
        col, caps.mem_tile_row, col, caps.mem_tile_row));
    absl::StrAppend(&aie_mlir, absl::StrFormat(
        "    %%tile_%d_%d = aie.tile(%d, %d)\n",
        col, compute_row, col, compute_row));
  }

  // ObjectFIFOs and links (shim -> mem -> compute, same as softmax).
  for (int c = 0; c < num_cores; c++) {
    int col = start_col + c;
    std::string shim = absl::StrFormat("tile_%d_%d", col, shim_row);
    std::string mem = absl::StrFormat("tile_%d_%d", col, caps.mem_tile_row);
    std::string compute = absl::StrFormat("tile_%d_%d", col, compute_row);

    // Input: shim -> mem (L2) -> compute (L1).
    absl::StrAppend(&aie_mlir, absl::StrFormat(
        "    aie.objectfifo @%s(%%%s, {%%%s}, %d : i32) "
        ": !aie.objectfifo<%s>\n",
        fifo_name(c, "in_L2"), shim, mem, fifo_depth, chunk_memref));
    absl::StrAppend(&aie_mlir, absl::StrFormat(
        "    aie.objectfifo @%s(%%%s, {%%%s}, %d : i32) "
        ": !aie.objectfifo<%s>\n",
        fifo_name(c, "in"), mem, compute, fifo_depth, chunk_memref));
    absl::StrAppend(&aie_mlir, absl::StrFormat(
        "    aie.objectfifo.link [@%s] -> [@%s] ([] [])\n",
        fifo_name(c, "in_L2"), fifo_name(c, "in")));

    // Output: compute -> mem (L2) -> shim.
    absl::StrAppend(&aie_mlir, absl::StrFormat(
        "    aie.objectfifo @%s(%%%s, {%%%s}, %d : i32) "
        ": !aie.objectfifo<%s>\n",
        fifo_name(c, "out"), compute, mem, fifo_depth, chunk_memref));
    absl::StrAppend(&aie_mlir, absl::StrFormat(
        "    aie.objectfifo @%s(%%%s, {%%%s}, %d : i32) "
        ": !aie.objectfifo<%s>\n",
        fifo_name(c, "out_L2"), mem, shim, fifo_depth, chunk_memref));
    absl::StrAppend(&aie_mlir, absl::StrFormat(
        "    aie.objectfifo.link [@%s] -> [@%s] ([] [])\n",
        fifo_name(c, "out"), fifo_name(c, "out_L2")));
  }

  // External function declaration.
  absl::StrAppend(&aie_mlir, absl::StrFormat(
      "    func.func private @gelu_bf16(%s, %s, i32) -> ()\n",
      chunk_memref, chunk_memref));

  // Core bodies.
  for (int c = 0; c < num_cores; c++) {
    int col = start_col + c;
    std::string compute = absl::StrFormat("tile_%d_%d", col, compute_row);
    std::string subview_ty = absl::StrFormat(
        "!aie.objectfifosubview<%s>", chunk_memref);

    absl::StrAppend(&aie_mlir, absl::StrFormat(
        "    %%core_%d_%d = aie.core(%%%s) {\n", col, compute_row, compute));
    absl::StrAppend(&aie_mlir, absl::StrFormat(
        "      %%c0 = arith.constant 0 : index\n"
        "      %%c1 = arith.constant 1 : index\n"
        "      %%num_chunks = arith.constant %d : index\n"
        "      %%chunk_sz = arith.constant %d : i32\n",
        num_chunks, chunk_size));
    absl::StrAppend(&aie_mlir, absl::StrFormat(
        "      scf.for %%i = %%c0 to %%num_chunks step %%c1 {\n"
        "        %%in_sub = aie.objectfifo.acquire @%s(Consume, 1) : %s\n"
        "        %%in_buf = aie.objectfifo.subview.access %%in_sub[0] "
        ": %s -> %s\n"
        "        %%out_sub = aie.objectfifo.acquire @%s(Produce, 1) : %s\n"
        "        %%out_buf = aie.objectfifo.subview.access %%out_sub[0] "
        ": %s -> %s\n"
        "        func.call @gelu_bf16(%%in_buf, %%out_buf, %%chunk_sz) "
        ": (%s, %s, i32) -> ()\n"
        "        aie.objectfifo.release @%s(Consume, 1)\n"
        "        aie.objectfifo.release @%s(Produce, 1)\n"
        "      }\n",
        fifo_name(c, "in"), subview_ty, subview_ty, chunk_memref,
        fifo_name(c, "out"), subview_ty, subview_ty, chunk_memref,
        chunk_memref, chunk_memref,
        fifo_name(c, "in"), fifo_name(c, "out")));
    absl::StrAppend(&aie_mlir, "      aie.end\n    }\n");
  }

  // Runtime sequence (DMA).
  absl::StrAppend(&aie_mlir, absl::StrFormat(
      "    aie.runtime_sequence(%%arg0: %s, %%arg1: %s) {\n",
      host_memref, host_memref));

  int dma_id = 0;
  for (int c = 0; c < num_cores; c++) {
    int64_t offset = c * per_core_elements;

    // Input DMA.
    absl::StrAppend(&aie_mlir, absl::StrFormat(
        "      aiex.npu.dma_memcpy_nd(%%arg0"
        "[0, 0, 0, %d][1, 1, %d, %d][0, 0, %d, 1]) "
        "{id = %d : i64, metadata = @%s, issue_token = true} : %s\n",
        offset, num_chunks, chunk_size, chunk_size,
        dma_id++, fifo_name(c, "in_L2"), host_memref));

    // Output DMA.
    absl::StrAppend(&aie_mlir, absl::StrFormat(
        "      aiex.npu.dma_memcpy_nd(%%arg1"
        "[0, 0, 0, %d][1, 1, %d, %d][0, 0, %d, 1]) "
        "{id = %d : i64, metadata = @%s, issue_token = true} : %s\n",
        offset, num_chunks, chunk_size, chunk_size,
        dma_id++, fifo_name(c, "out_L2"), host_memref));
  }

  // Wait for all DMAs.
  for (int c = 0; c < num_cores; c++) {
    absl::StrAppend(&aie_mlir, absl::StrFormat(
        "      aiex.npu.dma_wait { symbol = @%s }\n",
        fifo_name(c, "out_L2")));
    absl::StrAppend(&aie_mlir, absl::StrFormat(
        "      aiex.npu.dma_wait { symbol = @%s }\n",
        fifo_name(c, "in_L2")));
  }

  absl::StrAppend(&aie_mlir, "    }\n");  // end runtime_sequence
  absl::StrAppend(&aie_mlir, "  }\n");    // end aie.device
  absl::StrAppend(&aie_mlir, "}\n");      // end module

  LOG(INFO) << "XDNA GELU lowering: generated AIE MLIR:\n" << aie_mlir;

  AieLoweringResult result;
  result.aie_mlir = aie_mlir;
  result.num_cores = num_cores;
  result.needs_gelu_kernel = true;
  return result;
}

// Detects a LayerNorm pattern in the module.
// XLA decomposes layer_norm into: mean reduction + variance reduction +
// rsqrt + normalize + scale/shift. Detection checks:
//   1. Function: 3 inputs, 1 output
//   2. Input shapes: [..., D], [D], [D]; output: [..., D]
//   3. Element type: bf16
//   4. Contains >=2 linalg.generic ops with reduction iterators
//   5. Contains math.rsqrt (or arith.divf with math.sqrt)
//   6. No linalg.matmul ops
std::optional<LayerNormProgramInfo> DetectLayerNorm(mlir::ModuleOp module) {
  mlir::func::FuncOp entry_func;
  module.walk([&](mlir::func::FuncOp func) {
    if (func.getName() == "main" || !entry_func) entry_func = func;
  });
  if (!entry_func) return std::nullopt;

  // Verify function signature: 3 inputs, 1 result.
  auto func_type = entry_func.getFunctionType();
  if (func_type.getNumInputs() != 3 || func_type.getNumResults() != 1)
    return std::nullopt;

  // Input 0 must be a ranked tensor (the data).
  auto x_type = mlir::dyn_cast<mlir::RankedTensorType>(func_type.getInput(0));
  auto gamma_type = mlir::dyn_cast<mlir::RankedTensorType>(
      func_type.getInput(1));
  auto beta_type = mlir::dyn_cast<mlir::RankedTensorType>(
      func_type.getInput(2));
  auto out_type = mlir::dyn_cast<mlir::RankedTensorType>(
      func_type.getResult(0));
  if (!x_type || !gamma_type || !beta_type || !out_type) return std::nullopt;

  // Element type must be bf16.
  std::string elem_type = GetElementTypeStr(x_type.getElementType());
  if (elem_type != "bf16") return std::nullopt;

  // All inputs/outputs must have consistent element type.
  if (GetElementTypeStr(gamma_type.getElementType()) != elem_type)
    return std::nullopt;
  if (GetElementTypeStr(beta_type.getElementType()) != elem_type)
    return std::nullopt;
  if (GetElementTypeStr(out_type.getElementType()) != elem_type)
    return std::nullopt;

  // x must be at least 1D. gamma/beta must be 1D.
  if (x_type.getRank() < 1) return std::nullopt;
  if (gamma_type.getRank() != 1 || beta_type.getRank() != 1)
    return std::nullopt;

  // Last dim of x must match gamma/beta length.
  int64_t row_length = x_type.getShape().back();
  if (gamma_type.getShape()[0] != row_length) return std::nullopt;
  if (beta_type.getShape()[0] != row_length) return std::nullopt;

  // Output must match x shape.
  if (out_type != x_type) return std::nullopt;

  // No matmul ops allowed.
  bool has_matmul = false;
  entry_func.walk([&](mlir::Operation* op) {
    if (op->getName().getStringRef() == "linalg.matmul") has_matmul = true;
  });
  if (has_matmul) return std::nullopt;

  // Count reductions and look for rsqrt/sqrt markers.
  int reduction_count = 0;
  bool has_rsqrt = false;
  bool has_sqrt = false;

  entry_func.walk([&](mlir::linalg::GenericOp generic) {
    auto iterators = generic.getIteratorTypesArray();
    bool is_reduction = false;
    for (auto it : iterators) {
      if (it == mlir::utils::IteratorType::reduction) {
        is_reduction = true;
        break;
      }
    }
    if (is_reduction) reduction_count++;

    mlir::Block& body = generic.getRegion().front();
    for (mlir::Operation& body_op : body.without_terminator()) {
      llvm::StringRef name = body_op.getName().getStringRef();
      if (name == "math.rsqrt") has_rsqrt = true;
      if (name == "math.sqrt") has_sqrt = true;
    }
  });

  // LayerNorm needs at least 2 reductions (mean + variance) and rsqrt/sqrt.
  if (reduction_count < 2) return std::nullopt;
  if (!has_rsqrt && !has_sqrt) return std::nullopt;

  // Kernel uses 32-wide bf16 vectors.
  if (row_length % 32 != 0) return std::nullopt;

  int64_t num_rows = 1;
  for (int i = 0; i < x_type.getRank() - 1; i++)
    num_rows *= x_type.getShape()[i];
  if (num_rows < 1) return std::nullopt;

  LOG(INFO) << "XDNA: detected LayerNorm pattern: " << num_rows << "x"
            << row_length << " " << elem_type;

  LayerNormProgramInfo info;
  info.num_rows = num_rows;
  info.row_length = row_length;
  info.element_type = elem_type;
  return info;
}

// Generates AIE MLIR for LayerNorm using an external C++ kernel.
// Pattern: 3 inputs (X, gamma, beta), 1 output. Gamma/beta broadcast to all
// cores (like K/V in attention). Rows distributed across cores.
absl::StatusOr<AieLoweringResult> LowerLayerNormToAie(
    const LayerNormProgramInfo& info, const TargetCaps& caps,
    int max_columns) {
  // Auto-scale number of cores.
  int num_cores = std::min(max_columns, caps.num_columns);
  while (num_cores > 1) {
    if (info.num_rows % num_cores != 0) { num_cores--; continue; }
    break;
  }
  int64_t rows_per_core = info.num_rows / num_cores;
  int64_t total_elements = info.num_rows * info.row_length;

  // L1 budget: in_x (depth 2) + gb (depth 2, acquire 2 → 3 bufs) + out (depth 2).
  // All buffers are row_length * 2 bytes (bf16).
  int64_t in_l1 = 2 * info.row_length * 2;   // depth 2 ping-pong
  int64_t gb_l1 = 3 * info.row_length * 2;   // depth 2 + acquire(2) → 3 buffers
  int64_t out_l1 = 2 * info.row_length * 2;  // depth 2 ping-pong
  int64_t total_l1 = in_l1 + gb_l1 + out_l1;

  if (total_l1 > caps.l1_usable_bytes) {
    return absl::InvalidArgumentError(absl::StrFormat(
        "LayerNorm row_length=%d exceeds L1 budget (%d bytes needed, "
        "%d available)", info.row_length, total_l1, caps.l1_usable_bytes));
  }

  // DMA dimension limits.
  int64_t d0_words = info.row_length * 2 / 4;  // bf16 elements -> 32-bit words
  if (d0_words > kMaxBdWords) {
    return absl::InvalidArgumentError(absl::StrFormat(
        "LayerNorm row_length=%d exceeds DMA d0 limit (%d words, max %d)",
        info.row_length, d0_words, kMaxBdWords));
  }
  // X/O DMA: d1 = rows_per_core.
  if (rows_per_core > kMaxBdWords) {
    return absl::InvalidArgumentError(absl::StrFormat(
        "LayerNorm rows_per_core=%d exceeds DMA d1 limit (max %d)",
        rows_per_core, kMaxBdWords));
  }

  LOG(INFO) << "XDNA LayerNorm lowering: " << info.num_rows << " rows x "
            << info.row_length << " " << info.element_type
            << " using " << num_cores << " core(s) ("
            << rows_per_core << " rows/core)"
            << " L1=" << total_l1 << "B";

  int start_col = caps.partition_start_column;
  int shim_row = caps.shim_row;
  int compute_row = caps.first_compute_row;
  int fifo_depth = 2;

  std::string row_memref = absl::StrFormat("memref<%dxbf16>", info.row_length);
  std::string x_host_memref = absl::StrFormat("memref<%dxbf16>", total_elements);
  std::string gb_host_memref = absl::StrFormat("memref<%dxbf16>",
                                                info.row_length);
  std::string o_host_memref = absl::StrFormat("memref<%dxbf16>", total_elements);

  auto fifo_name = [&](int c, const std::string& base) -> std::string {
    return num_cores > 1 ? absl::StrFormat("c%d_%s", c, base) : base;
  };

  std::string aie_mlir;
  absl::StrAppend(&aie_mlir, "module {\n");
  absl::StrAppend(&aie_mlir, absl::StrFormat(
      "  aie.device(%s) {\n", DeviceNameForColumns(num_cores)));

  // Tile declarations.
  for (int c = 0; c < num_cores; c++) {
    int col = start_col + c;
    absl::StrAppend(&aie_mlir, absl::StrFormat(
        "    %%tile_%d_%d = aie.tile(%d, %d)\n", col, shim_row, col, shim_row));
    absl::StrAppend(&aie_mlir, absl::StrFormat(
        "    %%tile_%d_%d = aie.tile(%d, %d)\n",
        col, caps.mem_tile_row, col, caps.mem_tile_row));
    absl::StrAppend(&aie_mlir, absl::StrFormat(
        "    %%tile_%d_%d = aie.tile(%d, %d)\n",
        col, compute_row, col, compute_row));
  }

  // Gamma/beta broadcast: single chain from start_col shim -> start_col mem ->
  // all compute tiles. Like K/V in attention: send once, all cores read.
  {
    std::string gb_shim = absl::StrFormat("tile_%d_%d", start_col, shim_row);
    std::string gb_mem = absl::StrFormat("tile_%d_%d", start_col,
                                          caps.mem_tile_row);
    // L2: shim(start_col) -> mem(start_col).
    absl::StrAppend(&aie_mlir, absl::StrFormat(
        "    aie.objectfifo @in_gb_L2(%%%s, {%%%s}, %d : i32) "
        ": !aie.objectfifo<%s>\n",
        gb_shim, gb_mem, fifo_depth, row_memref));
    // L1 broadcast: mem(start_col) -> {all compute tiles}.
    std::string gb_consumers;
    for (int c = 0; c < num_cores; c++) {
      if (c > 0) gb_consumers += ", ";
      absl::StrAppendFormat(&gb_consumers, "%%tile_%d_%d",
                            start_col + c, compute_row);
    }
    absl::StrAppend(&aie_mlir, absl::StrFormat(
        "    aie.objectfifo @in_gb(%%%s, {%s}, %d : i32) "
        ": !aie.objectfifo<%s>\n",
        gb_mem, gb_consumers, fifo_depth, row_memref));
    absl::StrAppend(&aie_mlir,
        "    aie.objectfifo.link [@in_gb_L2] -> [@in_gb] ([] [])\n");
  }

  // X input and output per-column (like softmax).
  for (int c = 0; c < num_cores; c++) {
    int col = start_col + c;
    std::string shim = absl::StrFormat("tile_%d_%d", col, shim_row);
    std::string mem = absl::StrFormat("tile_%d_%d", col, caps.mem_tile_row);
    std::string compute = absl::StrFormat("tile_%d_%d", col, compute_row);

    // X input: shim -> mem -> compute.
    absl::StrAppend(&aie_mlir, absl::StrFormat(
        "    aie.objectfifo @%s(%%%s, {%%%s}, %d : i32) "
        ": !aie.objectfifo<%s>\n",
        fifo_name(c, "in_x_L2"), shim, mem, fifo_depth, row_memref));
    absl::StrAppend(&aie_mlir, absl::StrFormat(
        "    aie.objectfifo @%s(%%%s, {%%%s}, %d : i32) "
        ": !aie.objectfifo<%s>\n",
        fifo_name(c, "in_x"), mem, compute, fifo_depth, row_memref));
    absl::StrAppend(&aie_mlir, absl::StrFormat(
        "    aie.objectfifo.link [@%s] -> [@%s] ([] [])\n",
        fifo_name(c, "in_x_L2"), fifo_name(c, "in_x")));

    // Output: compute -> mem -> shim.
    absl::StrAppend(&aie_mlir, absl::StrFormat(
        "    aie.objectfifo @%s(%%%s, {%%%s}, %d : i32) "
        ": !aie.objectfifo<%s>\n",
        fifo_name(c, "out"), compute, mem, fifo_depth, row_memref));
    absl::StrAppend(&aie_mlir, absl::StrFormat(
        "    aie.objectfifo @%s(%%%s, {%%%s}, %d : i32) "
        ": !aie.objectfifo<%s>\n",
        fifo_name(c, "out_L2"), mem, shim, fifo_depth, row_memref));
    absl::StrAppend(&aie_mlir, absl::StrFormat(
        "    aie.objectfifo.link [@%s] -> [@%s] ([] [])\n",
        fifo_name(c, "out"), fifo_name(c, "out_L2")));
  }

  // External function declaration.
  absl::StrAppend(&aie_mlir, absl::StrFormat(
      "    func.func private @layernorm_bf16(%s, %s, %s, %s, i32) -> ()\n",
      row_memref, row_memref, row_memref, row_memref));

  // Core bodies.
  for (int c = 0; c < num_cores; c++) {
    int col = start_col + c;
    std::string compute = absl::StrFormat("tile_%d_%d", col, compute_row);
    std::string x_subview_ty = absl::StrFormat(
        "!aie.objectfifosubview<%s>", row_memref);
    std::string gb_subview_ty = absl::StrFormat(
        "!aie.objectfifosubview<%s>", row_memref);
    std::string o_subview_ty = absl::StrFormat(
        "!aie.objectfifosubview<%s>", row_memref);

    absl::StrAppend(&aie_mlir, absl::StrFormat(
        "    %%core_%d_%d = aie.core(%%%s) {\n", col, compute_row, compute));
    // Acquire gamma/beta (2 elements from broadcast FIFO).
    absl::StrAppend(&aie_mlir, absl::StrFormat(
        "      %%gb_sub = aie.objectfifo.acquire @in_gb(Consume, 2) : %s\n"
        "      %%gamma_buf = aie.objectfifo.subview.access %%gb_sub[0] "
        ": %s -> %s\n"
        "      %%beta_buf = aie.objectfifo.subview.access %%gb_sub[1] "
        ": %s -> %s\n",
        gb_subview_ty, gb_subview_ty, row_memref,
        gb_subview_ty, row_memref));
    absl::StrAppend(&aie_mlir, absl::StrFormat(
        "      %%c0 = arith.constant 0 : index\n"
        "      %%c1 = arith.constant 1 : index\n"
        "      %%num_rows = arith.constant %d : index\n"
        "      %%row_len = arith.constant %d : i32\n",
        rows_per_core, info.row_length));
    absl::StrAppend(&aie_mlir, absl::StrFormat(
        "      scf.for %%i = %%c0 to %%num_rows step %%c1 {\n"
        "        %%in_sub = aie.objectfifo.acquire @%s(Consume, 1) : %s\n"
        "        %%in_buf = aie.objectfifo.subview.access %%in_sub[0] "
        ": %s -> %s\n"
        "        %%out_sub = aie.objectfifo.acquire @%s(Produce, 1) : %s\n"
        "        %%out_buf = aie.objectfifo.subview.access %%out_sub[0] "
        ": %s -> %s\n"
        "        func.call @layernorm_bf16(%%in_buf, %%gamma_buf, "
        "%%beta_buf, %%out_buf, %%row_len) "
        ": (%s, %s, %s, %s, i32) -> ()\n"
        "        aie.objectfifo.release @%s(Consume, 1)\n"
        "        aie.objectfifo.release @%s(Produce, 1)\n"
        "      }\n",
        fifo_name(c, "in_x"), x_subview_ty, x_subview_ty, row_memref,
        fifo_name(c, "out"), o_subview_ty, o_subview_ty, row_memref,
        row_memref, row_memref, row_memref, row_memref,
        fifo_name(c, "in_x"), fifo_name(c, "out")));
    absl::StrAppend(&aie_mlir,
        "      aie.objectfifo.release @in_gb(Consume, 2)\n");
    absl::StrAppend(&aie_mlir, "      aie.end\n    }\n");
  }

  // Runtime sequence (DMA): 4 args = X, gamma, beta, output.
  absl::StrAppend(&aie_mlir, absl::StrFormat(
      "    aie.runtime_sequence(%%arg0: %s, %%arg1: %s, %%arg2: %s, "
      "%%arg3: %s) {\n",
      x_host_memref, gb_host_memref, gb_host_memref, o_host_memref));

  int dma_id = 0;

  // Gamma broadcast via @in_gb_L2 (fills slot 0).
  absl::StrAppend(&aie_mlir, absl::StrFormat(
      "      aiex.npu.dma_memcpy_nd(%%arg1"
      "[0, 0, 0, 0][1, 1, 1, %d][0, 0, 0, 1]) "
      "{id = %d : i64, metadata = @in_gb_L2, issue_token = true} : %s\n",
      info.row_length, dma_id++, gb_host_memref));

  // Beta broadcast via @in_gb_L2 (fills slot 1).
  absl::StrAppend(&aie_mlir, absl::StrFormat(
      "      aiex.npu.dma_memcpy_nd(%%arg2"
      "[0, 0, 0, 0][1, 1, 1, %d][0, 0, 0, 1]) "
      "{id = %d : i64, metadata = @in_gb_L2, issue_token = true} : %s\n",
      info.row_length, dma_id++, gb_host_memref));

  // X per-column.
  for (int c = 0; c < num_cores; c++) {
    int64_t offset = c * rows_per_core * info.row_length;
    absl::StrAppend(&aie_mlir, absl::StrFormat(
        "      aiex.npu.dma_memcpy_nd(%%arg0"
        "[0, 0, 0, %d][1, 1, %d, %d][0, 0, %d, 1]) "
        "{id = %d : i64, metadata = @%s, issue_token = true} : %s\n",
        offset, rows_per_core, info.row_length, info.row_length,
        dma_id++, fifo_name(c, "in_x_L2"), x_host_memref));
  }

  // O per-column.
  for (int c = 0; c < num_cores; c++) {
    int64_t offset = c * rows_per_core * info.row_length;
    absl::StrAppend(&aie_mlir, absl::StrFormat(
        "      aiex.npu.dma_memcpy_nd(%%arg3"
        "[0, 0, 0, %d][1, 1, %d, %d][0, 0, %d, 1]) "
        "{id = %d : i64, metadata = @%s, issue_token = true} : %s\n",
        offset, rows_per_core, info.row_length, info.row_length,
        dma_id++, fifo_name(c, "out_L2"), o_host_memref));
  }

  // Wait for all channels.
  absl::StrAppend(&aie_mlir,
      "      aiex.npu.dma_wait { symbol = @in_gb_L2 }\n");
  for (int c = 0; c < num_cores; c++) {
    absl::StrAppend(&aie_mlir, absl::StrFormat(
        "      aiex.npu.dma_wait { symbol = @%s }\n",
        fifo_name(c, "out_L2")));
    absl::StrAppend(&aie_mlir, absl::StrFormat(
        "      aiex.npu.dma_wait { symbol = @%s }\n",
        fifo_name(c, "in_x_L2")));
  }

  absl::StrAppend(&aie_mlir, "    }\n");  // end runtime_sequence
  absl::StrAppend(&aie_mlir, "  }\n");    // end aie.device
  absl::StrAppend(&aie_mlir, "}\n");      // end module

  LOG(INFO) << "XDNA LayerNorm lowering: generated AIE MLIR:\n" << aie_mlir;

  AieLoweringResult result;
  result.aie_mlir = aie_mlir;
  result.num_cores = num_cores;
  result.needs_layernorm_kernel = true;
  result.layernorm_row_length = info.row_length;
  return result;
}

// Lowers a broadcast elementwise op to AIE dialect MLIR.
// Pattern: [M, N] op [N] -> [M, N] (e.g., bias add).
// Broadcasts the smaller input (bias) to all compute tiles via mem tile,
// then each core processes M/num_cores rows with vectorized elementwise op.
absl::StatusOr<AieLoweringResult> LowerBroadcastElementwiseToAie(
    const LinalgProgramInfo& program, const TargetCaps& caps,
    int max_columns) {
  // Extract broadcast parameters.
  int64_t N = program.broadcast_inner_dim;  // row length (broadcast dim)
  int64_t M = program.broadcast_outer_dim;  // number of rows
  const std::string& kernel_op = program.single_op->kernel_op;
  const std::string& element_type = program.storage_type;
  int element_bytes = GetElementBytes(element_type);

  // Auto-scale number of cores.
  int num_cores = std::min(max_columns, caps.num_columns);
  while (num_cores > 1) {
    if (M % num_cores != 0) { num_cores--; continue; }
    break;
  }
  int64_t rows_per_core = M / num_cores;

  // L1 budget: bias (depth 2) + act (depth 2) + out (depth 2).
  // All buffers are N * element_bytes.
  int64_t buf_size = N * element_bytes;
  int64_t total_l1 = 6 * buf_size;  // 3 FIFOs x depth 2

  if (total_l1 > caps.l1_usable_bytes) {
    return absl::InvalidArgumentError(absl::StrFormat(
        "Broadcast elementwise N=%d exceeds L1 budget (%d bytes needed, "
        "%d available)", N, total_l1, caps.l1_usable_bytes));
  }

  // DMA dimension limits.
  int64_t d0_words = N * element_bytes / 4;
  if (d0_words > kMaxBdWords) {
    return absl::InvalidArgumentError(absl::StrFormat(
        "Broadcast elementwise N=%d exceeds DMA d0 limit (%d words, max %d)",
        N, d0_words, kMaxBdWords));
  }
  if (rows_per_core > kMaxBdWords) {
    return absl::InvalidArgumentError(absl::StrFormat(
        "Broadcast elementwise rows_per_core=%d exceeds DMA d1 limit (max %d)",
        rows_per_core, kMaxBdWords));
  }

  // Determine which DMA input index is the broadcast (bias) and which is the
  // identity-mapped (activation). For a 2-input op like add(act, bias),
  // broadcast_inputs tells us which.
  int bias_dma_idx = -1;
  int act_dma_idx = -1;
  for (int i = 0; i < program.num_inputs; i++) {
    if (i < static_cast<int>(program.broadcast_inputs.size()) &&
        program.broadcast_inputs[i]) {
      bias_dma_idx = i;
    } else {
      act_dma_idx = i;
    }
  }
  if (bias_dma_idx < 0 || act_dma_idx < 0) {
    return absl::InvalidArgumentError(
        "Broadcast elementwise: could not identify bias and activation inputs");
  }
  // Track operand order: if bias is DMA input 0, it's the LHS in the body.
  // For non-commutative ops (sub), we need to respect this order.
  bool bias_is_lhs = (bias_dma_idx < act_dma_idx);

  // Vectorization.
  int vector_width = GetVectorWidth(element_type, kernel_op);
  bool use_vectorized = (vector_width > 0 && N >= vector_width &&
                         N % vector_width == 0);
  bool use_aievec = use_vectorized && NeedsAievecPipeline(kernel_op);

  int64_t total_elements = M * N;
  LOG(INFO) << "XDNA broadcast elementwise lowering: [" << M << "," << N
            << "] " << kernel_op << " " << element_type
            << " using " << num_cores << " core(s) ("
            << rows_per_core << " rows/core)"
            << " L1=" << total_l1 << "B"
            << (use_vectorized ? " vectorized" : " scalar");

  int start_col = caps.partition_start_column;
  int shim_row = caps.shim_row;
  int compute_row = caps.first_compute_row;
  int fifo_depth = 2;

  std::string row_memref = absl::StrFormat("memref<%dx%s>", N, element_type);
  std::string act_host_memref = absl::StrFormat("memref<%dx%s>",
                                                 total_elements, element_type);
  std::string bias_host_memref = absl::StrFormat("memref<%dx%s>",
                                                  N, element_type);
  std::string out_host_memref = absl::StrFormat("memref<%dx%s>",
                                                 total_elements, element_type);

  auto fifo_name = [&](int c, const std::string& base) -> std::string {
    return num_cores > 1 ? absl::StrFormat("c%d_%s", c, base) : base;
  };

  std::string aie_mlir;
  absl::StrAppend(&aie_mlir, "module {\n");
  absl::StrAppend(&aie_mlir, absl::StrFormat(
      "  aie.device(%s) {\n", DeviceNameForColumns(num_cores)));

  // Tile declarations.
  for (int c = 0; c < num_cores; c++) {
    int col = start_col + c;
    absl::StrAppend(&aie_mlir, absl::StrFormat(
        "    %%tile_%d_%d = aie.tile(%d, %d)\n", col, shim_row, col, shim_row));
    absl::StrAppend(&aie_mlir, absl::StrFormat(
        "    %%tile_%d_%d = aie.tile(%d, %d)\n",
        col, caps.mem_tile_row, col, caps.mem_tile_row));
    absl::StrAppend(&aie_mlir, absl::StrFormat(
        "    %%tile_%d_%d = aie.tile(%d, %d)\n",
        col, compute_row, col, compute_row));
  }

  // Bias broadcast: single chain from start_col shim -> start_col mem ->
  // all compute tiles. Send once, all cores read.
  {
    std::string bias_shim = absl::StrFormat("tile_%d_%d", start_col, shim_row);
    std::string bias_mem = absl::StrFormat("tile_%d_%d", start_col,
                                            caps.mem_tile_row);
    // L2: shim(start_col) -> mem(start_col).
    absl::StrAppend(&aie_mlir, absl::StrFormat(
        "    aie.objectfifo @in_bias_L2(%%%s, {%%%s}, %d : i32) "
        ": !aie.objectfifo<%s>\n",
        bias_shim, bias_mem, fifo_depth, row_memref));
    // L1 broadcast: mem(start_col) -> {all compute tiles}.
    std::string bias_consumers;
    for (int c = 0; c < num_cores; c++) {
      if (c > 0) bias_consumers += ", ";
      absl::StrAppendFormat(&bias_consumers, "%%tile_%d_%d",
                            start_col + c, compute_row);
    }
    absl::StrAppend(&aie_mlir, absl::StrFormat(
        "    aie.objectfifo @in_bias(%%%s, {%s}, %d : i32) "
        ": !aie.objectfifo<%s>\n",
        bias_mem, bias_consumers, fifo_depth, row_memref));
    absl::StrAppend(&aie_mlir,
        "    aie.objectfifo.link [@in_bias_L2] -> [@in_bias] ([] [])\n");
  }

  // Activation input and output per-column.
  for (int c = 0; c < num_cores; c++) {
    int col = start_col + c;
    std::string shim = absl::StrFormat("tile_%d_%d", col, shim_row);
    std::string mem = absl::StrFormat("tile_%d_%d", col, caps.mem_tile_row);
    std::string compute = absl::StrFormat("tile_%d_%d", col, compute_row);

    // Activation input: shim -> mem -> compute.
    absl::StrAppend(&aie_mlir, absl::StrFormat(
        "    aie.objectfifo @%s(%%%s, {%%%s}, %d : i32) "
        ": !aie.objectfifo<%s>\n",
        fifo_name(c, "in_act_L2"), shim, mem, fifo_depth, row_memref));
    absl::StrAppend(&aie_mlir, absl::StrFormat(
        "    aie.objectfifo @%s(%%%s, {%%%s}, %d : i32) "
        ": !aie.objectfifo<%s>\n",
        fifo_name(c, "in_act"), mem, compute, fifo_depth, row_memref));
    absl::StrAppend(&aie_mlir, absl::StrFormat(
        "    aie.objectfifo.link [@%s] -> [@%s] ([] [])\n",
        fifo_name(c, "in_act_L2"), fifo_name(c, "in_act")));

    // Output: compute -> mem -> shim.
    absl::StrAppend(&aie_mlir, absl::StrFormat(
        "    aie.objectfifo @%s(%%%s, {%%%s}, %d : i32) "
        ": !aie.objectfifo<%s>\n",
        fifo_name(c, "out"), compute, mem, fifo_depth, row_memref));
    absl::StrAppend(&aie_mlir, absl::StrFormat(
        "    aie.objectfifo @%s(%%%s, {%%%s}, %d : i32) "
        ": !aie.objectfifo<%s>\n",
        fifo_name(c, "out_L2"), mem, shim, fifo_depth, row_memref));
    absl::StrAppend(&aie_mlir, absl::StrFormat(
        "    aie.objectfifo.link [@%s] -> [@%s] ([] [])\n",
        fifo_name(c, "out"), fifo_name(c, "out_L2")));
  }

  // Core bodies: acquire bias once, loop over rows with vectorized op.
  for (int c = 0; c < num_cores; c++) {
    int col = start_col + c;
    std::string compute = absl::StrFormat("tile_%d_%d", col, compute_row);
    std::string subview_ty = absl::StrFormat(
        "!aie.objectfifosubview<%s>", row_memref);

    absl::StrAppend(&aie_mlir, absl::StrFormat(
        "    %%core_%d_%d = aie.core(%%%s) {\n", col, compute_row, compute));

    // Acquire bias (held for all rows).
    absl::StrAppend(&aie_mlir, absl::StrFormat(
        "      %%bias_sub = aie.objectfifo.acquire @in_bias(Consume, 1) : %s\n"
        "      %%bias_buf = aie.objectfifo.subview.access %%bias_sub[0] "
        ": %s -> %s\n",
        subview_ty, subview_ty, row_memref));

    // Row loop constants.
    absl::StrAppend(&aie_mlir, absl::StrFormat(
        "      %%c0 = arith.constant 0 : index\n"
        "      %%c1 = arith.constant 1 : index\n"
        "      %%num_rows = arith.constant %d : index\n",
        rows_per_core));

    absl::StrAppend(&aie_mlir,
        "      scf.for %i = %c0 to %num_rows step %c1 {\n");

    // Acquire activation and output.
    absl::StrAppend(&aie_mlir, absl::StrFormat(
        "        %%act_sub = aie.objectfifo.acquire @%s(Consume, 1) : %s\n"
        "        %%act_buf = aie.objectfifo.subview.access %%act_sub[0] "
        ": %s -> %s\n"
        "        %%out_sub = aie.objectfifo.acquire @%s(Produce, 1) : %s\n"
        "        %%out_buf = aie.objectfifo.subview.access %%out_sub[0] "
        ": %s -> %s\n",
        fifo_name(c, "in_act"), subview_ty, subview_ty, row_memref,
        fifo_name(c, "out"), subview_ty, subview_ty, row_memref));

    // Generate vectorized or scalar elementwise loop over N elements.
    if (use_vectorized) {
      std::string vec_ty = absl::StrFormat("vector<%dx%s>",
                                            vector_width, element_type);
      absl::StrAppend(&aie_mlir, absl::StrFormat(
          "        %%cv0 = arith.constant 0 : index\n"
          "        %%cv_step = arith.constant %d : index\n"
          "        %%cv_end = arith.constant %d : index\n",
          vector_width, N));
      absl::StrAppend(&aie_mlir,
          "        scf.for %d = %cv0 to %cv_end step %cv_step {\n");
      absl::StrAppend(&aie_mlir, absl::StrFormat(
          "          %%va = vector.load %%act_buf[%%d] : %s, %s\n"
          "          %%vb = vector.load %%bias_buf[%%d] : %s, %s\n",
          row_memref, vec_ty, row_memref, vec_ty));
      // Apply the kernel op (respect operand order for non-commutative ops).
      if (bias_is_lhs) {
        absl::StrAppend(&aie_mlir, absl::StrFormat(
            "          %%vr = %s %%vb, %%va : %s\n", kernel_op, vec_ty));
      } else {
        absl::StrAppend(&aie_mlir, absl::StrFormat(
            "          %%vr = %s %%va, %%vb : %s\n", kernel_op, vec_ty));
      }
      absl::StrAppend(&aie_mlir, absl::StrFormat(
          "          vector.store %%vr, %%out_buf[%%d] : %s, %s\n",
          row_memref, vec_ty));
      absl::StrAppend(&aie_mlir, "        }\n");
    } else {
      // Scalar loop.
      absl::StrAppend(&aie_mlir, absl::StrFormat(
          "        %%cs0 = arith.constant 0 : index\n"
          "        %%cs1 = arith.constant 1 : index\n"
          "        %%cs_end = arith.constant %d : index\n",
          N));
      absl::StrAppend(&aie_mlir,
          "        scf.for %d = %cs0 to %cs_end step %cs1 {\n");
      absl::StrAppend(&aie_mlir, absl::StrFormat(
          "          %%sa = memref.load %%act_buf[%%d] : %s\n"
          "          %%sb = memref.load %%bias_buf[%%d] : %s\n",
          row_memref, row_memref));
      if (bias_is_lhs) {
        absl::StrAppend(&aie_mlir, absl::StrFormat(
            "          %%sr = %s %%sb, %%sa : %s\n", kernel_op, element_type));
      } else {
        absl::StrAppend(&aie_mlir, absl::StrFormat(
            "          %%sr = %s %%sa, %%sb : %s\n", kernel_op, element_type));
      }
      absl::StrAppend(&aie_mlir, absl::StrFormat(
          "          memref.store %%sr, %%out_buf[%%d] : %s\n", row_memref));
      absl::StrAppend(&aie_mlir, "        }\n");
    }

    // Release activation and output.
    absl::StrAppend(&aie_mlir, absl::StrFormat(
        "        aie.objectfifo.release @%s(Consume, 1)\n"
        "        aie.objectfifo.release @%s(Produce, 1)\n",
        fifo_name(c, "in_act"), fifo_name(c, "out")));
    absl::StrAppend(&aie_mlir, "      }\n");  // end scf.for

    // Release bias.
    absl::StrAppend(&aie_mlir,
        "      aie.objectfifo.release @in_bias(Consume, 1)\n");
    absl::StrAppend(&aie_mlir, "      aie.end\n    }\n");
  }

  // Runtime sequence: args follow DMA input order (matching HLO parameter order).
  // arg{act_dma_idx} = activation [M*N], arg{bias_dma_idx} = bias [N],
  // arg{num_inputs} = output [M*N].
  std::string arg_act = absl::StrFormat("%%arg%d", act_dma_idx);
  std::string arg_bias = absl::StrFormat("%%arg%d", bias_dma_idx);
  std::string arg_out = absl::StrFormat("%%arg%d", program.num_inputs);

  // Build arg list in order.
  std::vector<std::string> arg_decls(program.num_inputs + 1);
  arg_decls[act_dma_idx] = absl::StrFormat("%%arg%d: %s",
                                             act_dma_idx, act_host_memref);
  arg_decls[bias_dma_idx] = absl::StrFormat("%%arg%d: %s",
                                              bias_dma_idx, bias_host_memref);
  arg_decls[program.num_inputs] = absl::StrFormat("%%arg%d: %s",
                                                    program.num_inputs,
                                                    out_host_memref);
  absl::StrAppend(&aie_mlir, absl::StrFormat(
      "    aie.runtime_sequence(%s) {\n",
      absl::StrJoin(arg_decls, ", ")));

  int dma_id = 0;

  // Bias broadcast via @in_bias_L2 (single [N] transfer).
  absl::StrAppend(&aie_mlir, absl::StrFormat(
      "      aiex.npu.dma_memcpy_nd(%s"
      "[0, 0, 0, 0][1, 1, 1, %d][0, 0, 0, 1]) "
      "{id = %d : i64, metadata = @in_bias_L2, issue_token = true} : %s\n",
      arg_bias, N, dma_id++, bias_host_memref));

  // Activation per-column.
  for (int c = 0; c < num_cores; c++) {
    int64_t offset = c * rows_per_core * N;
    absl::StrAppend(&aie_mlir, absl::StrFormat(
        "      aiex.npu.dma_memcpy_nd(%s"
        "[0, 0, 0, %d][1, 1, %d, %d][0, 0, %d, 1]) "
        "{id = %d : i64, metadata = @%s, issue_token = true} : %s\n",
        arg_act, offset, rows_per_core, N, N,
        dma_id++, fifo_name(c, "in_act_L2"), act_host_memref));
  }

  // Output per-column.
  for (int c = 0; c < num_cores; c++) {
    int64_t offset = c * rows_per_core * N;
    absl::StrAppend(&aie_mlir, absl::StrFormat(
        "      aiex.npu.dma_memcpy_nd(%s"
        "[0, 0, 0, %d][1, 1, %d, %d][0, 0, %d, 1]) "
        "{id = %d : i64, metadata = @%s, issue_token = true} : %s\n",
        arg_out, offset, rows_per_core, N, N,
        dma_id++, fifo_name(c, "out_L2"), out_host_memref));
  }

  // Wait for all channels.
  absl::StrAppend(&aie_mlir,
      "      aiex.npu.dma_wait { symbol = @in_bias_L2 }\n");
  for (int c = 0; c < num_cores; c++) {
    absl::StrAppend(&aie_mlir, absl::StrFormat(
        "      aiex.npu.dma_wait { symbol = @%s }\n",
        fifo_name(c, "out_L2")));
    absl::StrAppend(&aie_mlir, absl::StrFormat(
        "      aiex.npu.dma_wait { symbol = @%s }\n",
        fifo_name(c, "in_act_L2")));
  }

  absl::StrAppend(&aie_mlir, "    }\n");  // end runtime_sequence
  absl::StrAppend(&aie_mlir, "  }\n");    // end aie.device
  absl::StrAppend(&aie_mlir, "}\n");      // end module

  LOG(INFO) << "XDNA broadcast elementwise lowering: generated AIE MLIR:\n"
            << aie_mlir;

  AieLoweringResult result;
  result.aie_mlir = aie_mlir;
  result.num_cores = num_cores;
  result.use_aievec = use_aievec;
  result.convert_vector_to_aievec = use_aievec;
  return result;
}

absl::StatusOr<AieLoweringResult> LowerLinalgToAie(
    mlir::ModuleOp linalg_module, const AieLoweringConfig& config,
    const TargetCaps& caps) {
  // Check for fused attention before matmul/softmax (attention contains both).
  auto attention_info = DetectAttention(linalg_module);
  if (attention_info.has_value()) {
    LOG(INFO) << "XDNA AIE lowering: detected fused attention M="
              << attention_info->num_rows << " S=" << attention_info->seq_len
              << " dk=" << attention_info->dk << " "
              << attention_info->element_type
              << (attention_info->is_causal ? " (causal)" : "");
    return LowerAttentionToAie(*attention_info, caps, config.num_columns);
  }

  // Check for LayerNorm before matmul (LayerNorm has reductions that could
  // confuse matmul detection if checked first).
  auto layernorm_info = DetectLayerNorm(linalg_module);
  if (layernorm_info.has_value()) {
    LOG(INFO) << "XDNA AIE lowering: detected LayerNorm "
              << layernorm_info->num_rows << "x"
              << layernorm_info->row_length << " "
              << layernorm_info->element_type;
    return LowerLayerNormToAie(*layernorm_info, caps, config.num_columns);
  }

  // Check for matmul before elementwise analysis.
  auto matmul_info = DetectMatmul(linalg_module);
  if (matmul_info.has_value()) {
    LOG(INFO) << "XDNA AIE lowering: detected matmul [" << matmul_info->M
              << "," << matmul_info->K << "]@[" << matmul_info->K << ","
              << matmul_info->N << "] " << matmul_info->element_type;
    return LowerMatmulToAieInternal(*matmul_info, caps, config.num_columns);
  }

  // Check for softmax before elementwise analysis.
  auto softmax_info = DetectSoftmax(linalg_module);
  if (softmax_info.has_value()) {
    LOG(INFO) << "XDNA AIE lowering: detected softmax "
              << softmax_info->num_rows << "x" << softmax_info->row_length
              << " " << softmax_info->element_type;
    return LowerSoftmaxToAie(*softmax_info, caps, config.num_columns);
  }

  // Check for GELU before elementwise analysis.
  auto gelu_info = DetectGelu(linalg_module);
  if (gelu_info.has_value()) {
    LOG(INFO) << "XDNA AIE lowering: detected GELU "
              << gelu_info->num_elements << " " << gelu_info->element_type;
    return LowerGeluToAie(*gelu_info, caps, config.num_columns);
  }

  // Step 1: Analyze the linalg module (elementwise path).
  auto program_result = AnalyzeLinalgModule(linalg_module);
  if (!program_result.ok()) return program_result.status();
  LinalgProgramInfo program = std::move(*program_result);

  // Check for broadcast elementwise (e.g., [M,N] + [N] bias add).
  if (program.broadcast_inner_dim > 0 && program.single_op.has_value()) {
    LOG(INFO) << "XDNA AIE lowering: detected broadcast elementwise ["
              << program.broadcast_outer_dim << ","
              << program.broadcast_inner_dim << "] "
              << program.single_op->kernel_op << " "
              << program.storage_type;
    return LowerBroadcastElementwiseToAie(program, caps, config.num_columns);
  }

  // Step 1b: Determine number of compute columns (cores).
  // All constraints are checked in a single loop to prevent one reduction
  // from breaking an invariant established by a previous check (e.g.,
  // reducing for vector width could break divisibility, silently dropping
  // tail elements via integer truncation).
  int num_cores = std::min(config.num_columns, caps.num_columns);

  // Determine vector width for chunk alignment (0 if not vectorized).
  int vector_width = 0;
  if (program.single_op.has_value()) {
    vector_width = GetVectorWidth(program.single_op->element_type,
                                   program.single_op->kernel_op);
  }
  int min_per_core = std::max(1, vector_width);

  // DMA transfers must be multiples of 4 bytes.
  int element_bytes = GetElementBytes(program.storage_type);
  int64_t min_dma_elements = (element_bytes < 4) ? (4 / element_bytes) : 1;

  while (num_cores > 1) {
    if (program.num_elements % num_cores != 0) { num_cores--; continue; }
    int64_t per_core = program.num_elements / num_cores;
    if (per_core < min_per_core) { num_cores--; continue; }
    if (per_core % min_dma_elements != 0) { num_cores--; continue; }
    break;
  }
  int64_t per_core_elements = program.num_elements / num_cores;

  LOG(INFO) << "XDNA AIE lowering: using " << num_cores << " core(s) for "
            << program.num_elements << " elements ("
            << per_core_elements << " per core)";

  // Step 1c: Compute tiling parameters for L1 memory budget.
  // Each ObjectFIFO buffer uses ping-pong depth of 2, so L1 per chunk is:
  //   num_buffers * chunk_size * element_bytes * 2 (ping-pong)
  int num_buffers = program.num_inputs + 1;  // inputs + output

  int64_t l1_limit = caps.l1_usable_bytes;
  const char* limit_env = std::getenv("XDNA_L1_LIMIT_BYTES");
  if (limit_env) l1_limit = std::atol(limit_env);

  // Max elements per chunk that fit in L1 with ping-pong ObjectFIFO buffers.
  int64_t max_chunk = l1_limit / (num_buffers * element_bytes * 2);

  // Tiling is per-core: each core processes per_core_elements independently.
  int64_t chunk_size;
  int64_t num_chunks;
  if (per_core_elements <= max_chunk) {
    chunk_size = per_core_elements;
    num_chunks = 1;
  } else {
    chunk_size = ComputeChunkSize(per_core_elements, max_chunk,
                                   vector_width, element_bytes);
    if (chunk_size <= 0) {
      return absl::InvalidArgumentError(absl::StrFormat(
          "Cannot tile %d elements into L1 (%d bytes). "
          "Max %d elements per chunk (%d buffers x %d bytes x 2 ping-pong). "
          "No valid chunk size found (vector_width=%d). "
          "Consider using a tensor size with more factors. "
          "Override L1 limit with XDNA_L1_LIMIT_BYTES env var.",
          per_core_elements, l1_limit, max_chunk, num_buffers,
          element_bytes, vector_width));
    }
    num_chunks = per_core_elements / chunk_size;
  }

  bool use_mem_tile = (num_chunks > 1);

  // Early d0/d1 split computation for use_distribute decision.
  int64_t d0_size_early = chunk_size;
  int64_t d0_reps_early = 1;
  if (use_mem_tile) {
    FindDmaSplit(chunk_size, element_bytes, d0_size_early, d0_reps_early);
  }

  // Distribute/join: use a single shim→mem FIFO that fans out to N compute
  // tiles via objectfifo.link, instead of N independent per-column FIFOs.
  // Requires: multi-core, mem tile (tiled), DMA dimensions fit, and
  // mem tile DMA channels sufficient:
  //   MM2S (output from mem): num_cores * num_inputs (L1 FIFOs) + 1 (L2 out)
  //   S2MM (input to mem):    num_inputs (L2 FIFOs) + num_cores (L1 out FIFOs)
  int mem_mm2s = num_cores * program.num_inputs + 1;
  int mem_s2mm = program.num_inputs + num_cores;
  bool channels_fit = (mem_mm2s <= caps.mem_tile_dma_channels &&
                        mem_s2mm <= caps.mem_tile_dma_channels);
  bool use_distribute = (num_cores > 1 && use_mem_tile && channels_fit &&
                          (d0_reps_early == 1 ||
                           num_chunks <= caps.shim_dma_max_d3));

  // If distribute doesn't fit at current num_cores, try fewer cores.
  // Limited to add/sub where compute is a single accumulator op per vector
  // and DMA overhead dominates. Don't reduce below half the original count.
  bool is_addsub = program.single_op.has_value() &&
      (program.single_op->kernel_op == "arith.addf" ||
       program.single_op->kernel_op == "arith.subf" ||
       program.single_op->kernel_op == "arith.addi" ||
       program.single_op->kernel_op == "arith.subi");
  if (!use_distribute && num_cores > 1 && use_mem_tile && is_addsub) {
    int min_try = std::max(2, (num_cores + 1) / 2);
    for (int try_cores = num_cores - 1; try_cores >= min_try; try_cores--) {
      if (program.num_elements % try_cores != 0) continue;
      int64_t try_per_core = program.num_elements / try_cores;
      if (try_per_core < min_per_core) continue;
      if (try_per_core % min_dma_elements != 0) continue;
      int64_t try_chunk, try_num_chunks;
      if (try_per_core <= max_chunk) {
        try_chunk = try_per_core;
        try_num_chunks = 1;
      } else {
        try_chunk = ComputeChunkSize(try_per_core, max_chunk,
                                      vector_width, element_bytes);
        if (try_chunk <= 0) continue;
        try_num_chunks = try_per_core / try_chunk;
      }
      if (try_num_chunks <= 1) continue;  // Need mem tile for distribute.
      int64_t try_d0 = try_chunk, try_d0r = 1;
      FindDmaSplit(try_chunk, element_bytes, try_d0, try_d0r);
      int try_mm2s = try_cores * program.num_inputs + 1;
      int try_s2mm = program.num_inputs + try_cores;
      if (try_mm2s > caps.mem_tile_dma_channels ||
          try_s2mm > caps.mem_tile_dma_channels) continue;
      if (try_d0r > 1 && try_num_chunks > caps.shim_dma_max_d3) continue;
      // Viable: adopt this configuration.
      LOG(INFO) << "XDNA AIE lowering: reduced " << num_cores << " → "
                << try_cores << " core(s) to enable distribute/join";
      num_cores = try_cores;
      per_core_elements = try_per_core;
      chunk_size = try_chunk;
      num_chunks = try_num_chunks;
      d0_size_early = try_d0;
      d0_reps_early = try_d0r;
      channels_fit = true;
      use_distribute = true;
      break;
    }
  }

  LOG(INFO) << "XDNA AIE lowering: tiling " << per_core_elements
            << " elements/core into " << num_chunks << " chunk(s) of "
            << chunk_size << " (L1 budget: " << l1_limit << "B"
            << (use_mem_tile ? ", mem tile staging" : ", direct")
            << (use_distribute ? ", distribute/join" : "") << ")";

  // Step 2: Generate AIE dialect MLIR text.
  //
  // Architecture: npu2 (XDNA 2 / Strix Halo / AIE2PS)
  // Layout: num_cores columns, each with:
  //   - Shim tile (col, 0): host DMA interface
  //   - Mem tile (col, 1): L2 staging buffer (used when tiling)
  //   - Compute tile (col, 2): runs the kernel
  //
  // Data flow (tiled): shim → mem_tile → compute → mem_tile → shim
  // Data flow (single chunk): shim → compute → shim

  std::string aie_mlir;
  const std::string& ty = program.storage_type;
  std::string memref_ty = absl::StrFormat("memref<%dx%s>", chunk_size, ty);
  int start_col = caps.partition_start_column;
  int shim_row = caps.shim_row;
  int compute_row = caps.first_compute_row;

  // FIFO name helper: multi-core uses "c{col}_" prefix, single-core uses "".
  auto fifo_name = [&](int c, const std::string& base) -> std::string {
    return num_cores > 1 ? absl::StrFormat("c%d_%s", c, base) : base;
  };
  // FIFO prefix for core body (e.g., "c0_" or "").
  auto fifo_prefix = [&](int c) -> std::string {
    return num_cores > 1 ? absl::StrFormat("c%d_", c) : "";
  };

  absl::StrAppend(&aie_mlir, "module {\n");
  absl::StrAppend(&aie_mlir,
      absl::StrFormat("  aie.device(%s) {\n",
                      DeviceNameForColumns(num_cores)));

  // mlir-aie requires all tile declarations before ObjectFIFOs and cores.
  // Split generation into 3 passes: tiles, ObjectFIFOs, cores.

  // Pass 1: All tile declarations.
  if (use_distribute) {
    // Distribute: one shim + one mem tile, N compute tiles across columns.
    absl::StrAppend(&aie_mlir, absl::StrFormat(
        "    %%tile_%d_%d = aie.tile(%d, %d)\n",
        start_col, shim_row, start_col, shim_row));
    absl::StrAppend(&aie_mlir, absl::StrFormat(
        "    %%tile_%d_%d = aie.tile(%d, %d)\n",
        start_col, caps.mem_tile_row, start_col, caps.mem_tile_row));
    for (int c = 0; c < num_cores; c++) {
      int col = start_col + c;
      absl::StrAppend(&aie_mlir, absl::StrFormat(
          "    %%tile_%d_%d = aie.tile(%d, %d)\n",
          col, compute_row, col, compute_row));
    }
  } else {
    // Per-column independent: each column has its own shim/mem/compute.
    for (int c = 0; c < num_cores; c++) {
      int col = start_col + c;
      absl::StrAppend(&aie_mlir, absl::StrFormat(
          "    %%tile_%d_%d = aie.tile(%d, %d)\n",
          col, shim_row, col, shim_row));
      if (use_mem_tile) {
        absl::StrAppend(&aie_mlir, absl::StrFormat(
            "    %%tile_%d_%d = aie.tile(%d, %d)\n", col, caps.mem_tile_row,
            col, caps.mem_tile_row));
      }
      absl::StrAppend(&aie_mlir, absl::StrFormat(
          "    %%tile_%d_%d = aie.tile(%d, %d)\n",
          col, compute_row, col, compute_row));
    }
  }

  // Pass 2: All ObjectFIFOs and links.
  if (use_distribute) {
    // Distribute/join: one L2 FIFO fans out to N L1 FIFOs via mem tile.
    std::string shim_tile = absl::StrFormat("tile_%d_%d", start_col, shim_row);
    std::string mem_tile_str = absl::StrFormat("tile_%d_%d", start_col,
                                                caps.mem_tile_row);
    // L2 element = num_cores * chunk_size (combined chunk for all cores).
    std::string l2_memref_ty = absl::StrFormat("memref<%dx%s>",
        num_cores * chunk_size, ty);

    // Input L2 FIFOs: shim → mem (one per input, combined element size).
    for (int i = 0; i < program.num_inputs; i++) {
      absl::StrAppend(&aie_mlir, absl::StrFormat(
          "    aie.objectfifo @in%d_L2(%%%s, {%%%s}, 2 : i32) "
          ": !aie.objectfifo<%s>\n",
          i, shim_tile, mem_tile_str, l2_memref_ty));
    }
    // Output L2 FIFO: mem → shim (combined element size).
    absl::StrAppend(&aie_mlir, absl::StrFormat(
        "    aie.objectfifo @out0_L2(%%%s, {%%%s}, 2 : i32) "
        ": !aie.objectfifo<%s>\n",
        mem_tile_str, shim_tile, l2_memref_ty));

    // Per-core L1 input FIFOs: mem → compute (chunk_size per core).
    for (int c = 0; c < num_cores; c++) {
      int col = start_col + c;
      std::string compute_tile =
          absl::StrFormat("tile_%d_%d", col, compute_row);
      for (int i = 0; i < program.num_inputs; i++) {
        absl::StrAppend(&aie_mlir, absl::StrFormat(
            "    aie.objectfifo @c%d_in%d(%%%s, {%%%s}, 2 : i32) "
            ": !aie.objectfifo<%s>\n",
            c, i, mem_tile_str, compute_tile, memref_ty));
      }
    }
    // Per-core L1 output FIFOs: compute → mem (chunk_size per core).
    for (int c = 0; c < num_cores; c++) {
      int col = start_col + c;
      std::string compute_tile =
          absl::StrFormat("tile_%d_%d", col, compute_row);
      absl::StrAppend(&aie_mlir, absl::StrFormat(
          "    aie.objectfifo @c%d_out0(%%%s, {%%%s}, 2 : i32) "
          ": !aie.objectfifo<%s>\n",
          c, compute_tile, mem_tile_str, memref_ty));
    }

    // Distribute links: L2 input → per-core L1 inputs.
    for (int i = 0; i < program.num_inputs; i++) {
      std::string consumers, offsets;
      for (int c = 0; c < num_cores; c++) {
        if (c > 0) { consumers += ", "; offsets += ", "; }
        absl::StrAppendFormat(&consumers, "@c%d_in%d", c, i);
        absl::StrAppendFormat(&offsets, "%d", c * chunk_size);
      }
      absl::StrAppend(&aie_mlir, absl::StrFormat(
          "    aie.objectfifo.link [@in%d_L2] -> [%s] ([] [%s])\n",
          i, consumers, offsets));
    }
    // Join link: per-core L1 outputs → L2 output.
    {
      std::string producers, offsets;
      for (int c = 0; c < num_cores; c++) {
        if (c > 0) { producers += ", "; offsets += ", "; }
        absl::StrAppendFormat(&producers, "@c%d_out0", c);
        absl::StrAppendFormat(&offsets, "%d", c * chunk_size);
      }
      absl::StrAppend(&aie_mlir, absl::StrFormat(
          "    aie.objectfifo.link [%s] -> [@out0_L2] ([%s] [])\n",
          producers, offsets));
    }
  } else {
    // Per-column independent: each column has its own FIFO chain.
    for (int c = 0; c < num_cores; c++) {
      int col = start_col + c;
      std::string shim_tile = absl::StrFormat("tile_%d_%d", col, shim_row);
      std::string compute_tile =
          absl::StrFormat("tile_%d_%d", col, compute_row);
      std::string mem_tile_str = absl::StrFormat("tile_%d_%d", col,
                                                  caps.mem_tile_row);

      if (use_mem_tile) {
        for (int i = 0; i < program.num_inputs; i++) {
          absl::StrAppend(&aie_mlir, absl::StrFormat(
              "    aie.objectfifo @%s(%%%s, {%%%s}, 2 : i32) "
              ": !aie.objectfifo<%s>\n",
              fifo_name(c, absl::StrFormat("in%d_L2", i)),
              shim_tile, mem_tile_str, memref_ty));
        }
        absl::StrAppend(&aie_mlir, absl::StrFormat(
            "    aie.objectfifo @%s(%%%s, {%%%s}, 2 : i32) "
            ": !aie.objectfifo<%s>\n",
            fifo_name(c, "out0_L2"),
            mem_tile_str, shim_tile, memref_ty));
        for (int i = 0; i < program.num_inputs; i++) {
          absl::StrAppend(&aie_mlir, absl::StrFormat(
              "    aie.objectfifo @%s(%%%s, {%%%s}, 2 : i32) "
              ": !aie.objectfifo<%s>\n",
              fifo_name(c, absl::StrFormat("in%d", i)),
              mem_tile_str, compute_tile, memref_ty));
        }
        absl::StrAppend(&aie_mlir, absl::StrFormat(
            "    aie.objectfifo @%s(%%%s, {%%%s}, 2 : i32) "
            ": !aie.objectfifo<%s>\n",
            fifo_name(c, "out0"),
            compute_tile, mem_tile_str, memref_ty));
        for (int i = 0; i < program.num_inputs; i++) {
          absl::StrAppend(&aie_mlir, absl::StrFormat(
              "    aie.objectfifo.link [@%s] -> [@%s] ([] [])\n",
              fifo_name(c, absl::StrFormat("in%d_L2", i)),
              fifo_name(c, absl::StrFormat("in%d", i))));
        }
        absl::StrAppend(&aie_mlir, absl::StrFormat(
            "    aie.objectfifo.link [@%s] -> [@%s] ([] [])\n",
            fifo_name(c, "out0"), fifo_name(c, "out0_L2")));
      } else {
        for (int i = 0; i < program.num_inputs; i++) {
          absl::StrAppend(&aie_mlir, absl::StrFormat(
              "    aie.objectfifo @%s(%%%s, {%%%s}, 2 : i32) "
              ": !aie.objectfifo<%s>\n",
              fifo_name(c, absl::StrFormat("in%d", i)),
              shim_tile, compute_tile, memref_ty));
        }
        absl::StrAppend(&aie_mlir, absl::StrFormat(
            "    aie.objectfifo @%s(%%%s, {%%%s}, 2 : i32) "
            ": !aie.objectfifo<%s>\n",
            fifo_name(c, "out0"),
            compute_tile, shim_tile, memref_ty));
      }
    }
  }

  // Pass 3: All core bodies.
  for (int c = 0; c < num_cores; c++) {
    int col = start_col + c;
    std::string compute_tile = absl::StrFormat("tile_%d_%d", col, compute_row);

    absl::StrAppend(&aie_mlir, absl::StrFormat(
        "    %%core_%d_%d = aie.core(%%%s) {\n", col, compute_row,
        compute_tile));
    absl::StrAppend(&aie_mlir,
        GenerateCoreBody(program, chunk_size, num_chunks, fifo_prefix(c)));
    absl::StrAppend(&aie_mlir,
        "      aie.end\n"
        "    }\n");
  }

  // NPU instruction sequence.
  auto npu_seq_or = GenerateNpuSequence(program, per_core_elements,
                                         chunk_size, num_chunks,
                                         use_mem_tile, num_cores, start_col,
                                         use_distribute);
  if (!npu_seq_or.ok()) return npu_seq_or.status();
  absl::StrAppend(&aie_mlir, *npu_seq_or);

  absl::StrAppend(&aie_mlir, "  }\n");
  absl::StrAppend(&aie_mlir, "}\n");

  LOG(INFO) << "XDNA AIE lowering: generated AIE MLIR:\n" << aie_mlir;

  // Detect if kernel needs aievec pipeline or soft-float stubs.
  bool use_aievec = false;
  bool cvt_vector_to_aievec = false;
  bool needs_softfloat = false;

  if (program.single_op.has_value()) {
    const auto& info = *program.single_op;
    // Check if vectorization will use the aievec pipeline.
    // Same condition as GenerateCoreBody: vector_width > 0, chunk aligned.
    bool is_vectorized = (vector_width > 0 && chunk_size >= vector_width &&
                          chunk_size % vector_width == 0);
    if (is_vectorized && NeedsAievecPipeline(info.kernel_op)) {
      use_aievec = true;
      cvt_vector_to_aievec = true;
    }
    // Single-op max/min: f32 uses integer bitcast (no soft-float), bf16 scalar
    // uses native cmpf (no soft-float), bf16 vector uses aievec (no soft-float).
    // No soft-float stubs needed for single-op max/min.
  }
  if (program.is_multi_op() && program.generic_op) {
    // Multi-op body may emit arith.maximumf directly (not cmpf+select),
    // which lowers to llvm.intr.maximum → __unordsf2 for f32.
    // bf16 maximumf lowers to native bf16 compare — no soft-float needed.
    // Check operand type, not storage_type: extf/truncf patterns can have
    // bf16 storage with internal f32 maximumf.
    for (mlir::Operation& body_op :
         program.generic_op.getRegion().front().without_terminator()) {
      llvm::StringRef bname = body_op.getName().getStringRef();
      if (bname == "arith.maximumf" || bname == "arith.minimumf") {
        if (body_op.getNumOperands() > 0 &&
            body_op.getOperand(0).getType().isF32()) {
          needs_softfloat = true;
          break;
        }
      }
    }
  }

  return AieLoweringResult{aie_mlir, num_cores, use_aievec,
                           cvt_vector_to_aievec,
                           /*needs_matmul_workarounds=*/false,
                           needs_softfloat, use_distribute};
}

}  // namespace xla
