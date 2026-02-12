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

  bool is_multi_op() const {
    return !single_op.has_value() && generic_op;
  }
};

// Matmul program description for AIE lowering.
struct MatmulProgramInfo {
  int64_t M, K, N;
  std::string element_type;  // "f32" or "bf16"
};

// Matmul tile sizes for L1 tiling.
struct MatmulTileConfig {
  int64_t m, k, n;  // tile dimensions
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

// Analyzes a linalg.generic to build a LinalgProgramInfo.
// Returns nullopt if the generic is not a supported element-wise computation.
//
// Supports two patterns:
//   1. All-identity indexing maps (standard elementwise: add, sub, mul, neg)
//   2. Mixed identity + scalar-broadcast maps where broadcast inputs are
//      compile-time constants (e.g., relu = max(x, 0) where 0 is broadcast)
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

  // Classify indexing maps: identity (DMA input) or scalar broadcast (constant).
  auto maps = generic.getIndexingMapsArray();
  int num_total_inputs = static_cast<int>(generic.getInputs().size());
  std::vector<bool> is_broadcast(num_total_inputs, false);
  std::vector<std::string> broadcast_values(num_total_inputs);

  for (int i = 0; i < num_total_inputs; i++) {
    auto map = maps[i];
    if (map.isIdentity()) {
      // Identity map: could be a DMA input OR a constant splat tensor.
      // stablehlo→linalg materializes broadcast constants as linalg.fill
      // inside private helper functions. Trace through call boundaries.
      mlir::Value inp = generic.getInputs()[i];
      mlir::Value traced = TraceBlockArgToCallSite(inp, module);
      std::string constant = TryExtractScalarConstant(traced);
      if (!constant.empty()) {
        is_broadcast[i] = true;
        broadcast_values[i] = constant;
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
    } else {
      return std::nullopt;  // Unsupported indexing pattern.
    }
  }

  // Output map must be identity.
  auto output_map = maps[num_total_inputs];  // output is after all inputs
  if (!output_map.isIdentity()) return std::nullopt;

  // Get shape and element type from the first non-broadcast input.
  mlir::RankedTensorType input_type;
  for (int i = 0; i < num_total_inputs; i++) {
    if (is_broadcast[i]) continue;
    input_type = mlir::dyn_cast<mlir::RankedTensorType>(
        generic.getInputs()[i].getType());
    break;
  }
  if (!input_type) return std::nullopt;

  // Verify all non-broadcast inputs have the same shape.
  for (int i = 0; i < num_total_inputs; i++) {
    if (is_broadcast[i]) continue;
    auto ty = mlir::dyn_cast<mlir::RankedTensorType>(
        generic.getInputs()[i].getType());
    if (!ty || ty.getShape() != input_type.getShape()) return std::nullopt;
  }

  std::string storage_type = GetElementTypeStr(input_type.getElementType());

  // Reject f16: Peano's AIE2p backend treats f16 (half) as bf16 in its
  // fpext/fptrunc lowering (left-shift-by-16 and vconv.bf16.fp32), producing
  // incorrect results. All f16 ops are broken until Peano adds proper support.
  if (storage_type == "f16") return std::nullopt;

  // Verify output element type matches input element type.
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

  // Count DMA inputs (non-broadcast).
  int dma_inputs = 0;
  for (int i = 0; i < num_total_inputs; i++) {
    if (!is_broadcast[i]) dma_inputs++;
  }

  LinalgProgramInfo program;
  program.num_inputs = dma_inputs;
  program.num_elements = num_elements;
  program.storage_type = storage_type;

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
  bool use_vectorized = is_bf16 && (info.K % 8 == 0);

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
  // Walk the entire module (not just entry_func) because XLA may generate
  // private helper functions (e.g., @relu.1) containing the linalg ops.
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
      unsupported_str = absl::StrCat(
          " Unsupported ops found: ", absl::StrJoin(unsupported_ops, ", "),
          ".");
    }
    // Print the module for debugging.
    std::string module_str;
    llvm::raw_string_ostream os(module_str);
    module.print(os);
    return absl::UnimplementedError(absl::StrCat(
        "No supported linalg operation found in module.",
        unsupported_str,
        " Currently supported: elementwise (add, sub, mul, neg, max, min),"
        " matmul (dot product). "
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
    bool use_mem_tile, int num_cores, int start_col) {
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

  // Strides: set to 0 when the corresponding dim has size 1 (unused),
  // to satisfy the aie-opt verifier on rank-1 memrefs.
  int64_t s1 = (d0_reps > 1) ? d0_size : 0;
  int64_t s2 = (num_chunks > 1) ? chunk_size : 0;

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

  // DMA transfers for each column.
  // 4D addressing: [1, num_chunks, d0_reps, d0_size]
  //   d0: innermost contiguous transfer (≤ 1023 32-bit words)
  //   d1: sub-chunks within a tile chunk (stride = d0_size)
  //   d2: tile chunks (stride = chunk_size)
  // Each column's DMA starts at offset c * per_core_elements into the
  // host buffer, selecting that core's slice of the tensor.
  for (int c = 0; c < num_cores; c++) {
    int col = start_col + c;
    int64_t offset = c * per_core_elements;

    // Input DMAs for this column.
    // BD IDs are per-core (0..num_inputs), not global. Each column's shim
    // has its own BD space, so different columns can reuse the same IDs.
    // The hardware has 16 BDs per shim (4-bit ID field: 0-15).
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
  std::string b_memref = absl::StrFormat("memref<%dx%dx%s>", tile.k, tile.n, ty);
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

  if (is_bf16) {
    // bf16: native multiply, f32 accumulator.
    absl::StrAppend(&body, absl::StrFormat(
        "                %%a_val = memref.load %%elem_a[%%i, %%p] : %s\n",
        a_memref));
    absl::StrAppend(&body, absl::StrFormat(
        "                %%b_val = memref.load %%elem_b[%%p, %%j] : %s\n",
        b_memref));
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
        "                %%b_val = memref.load %%elem_b[%%p, %%j] : %s\n",
        b_memref));
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
  int64_t b_d0_words = tile.n * element_bytes / 4;
  int64_t b_d1 = tile.k;
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

      // B DMA: [Nt_per_core, Kt, k, n] strides [n, k*N, N, 1]
      //   d3=Nt_per_core: iterate over N-tiles (stride n)
      //   d2=Kt: iterate over K-tiles (stride k*N)
      //   d1=k: rows of tile (stride N = row stride in B)
      //   d0=n: elements per row (stride 1)
      // Each core reads a different column slice of B.
      int64_t b_offset = c * Nt_per_core * tile.n;
      absl::StrAppend(&seq, absl::StrFormat(
          "      aiex.npu.dma_memcpy_nd(%%arg1"
          "[0, 0, 0, %d][%d, %d, %d, %d][%d, %d, %d, 1]) "
          "{id = %d : i64, metadata = @%s, issue_token = true} : %s\n",
          b_offset,
          Nt_per_core, Kt, tile.k, tile.n,
          tile.n, tile.k * info.N, info.N,
          dma_id++, fifo_name(c, "inB_L2"), b_host_memref));

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
            << ") using " << num_cores << " core(s)";
  const std::string& ty = info.element_type;
  int start_col = caps.partition_start_column;
  int shim_row = caps.shim_row;
  int compute_row = caps.first_compute_row;

  std::string a_memref = absl::StrFormat("memref<%dx%dx%s>", tile.m, tile.k, ty);
  std::string b_memref = absl::StrFormat("memref<%dx%dx%s>", tile.k, tile.n, ty);
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
                           /*use_aievec=*/use_vectorized};
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

  // Count linalg ops: we accept exactly {fill, matmul} or just {matmul}.
  int fill_count = 0;
  int matmul_count = 0;
  int other_linalg_count = 0;
  mlir::Operation* matmul_op = nullptr;
  mlir::Operation* fill_op = nullptr;

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
    } else {
      other_linalg_count++;
    }
  });

  if (matmul_count != 1 || other_linalg_count != 0) return std::nullopt;
  if (fill_count > 1) return std::nullopt;  // At most one fill

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

  // Extract M, K, N from matmul operand shapes.
  // linalg.matmul ins(%A : tensor<MxKxTy>, %B : tensor<KxNxTy>)
  //               outs(%C : tensor<MxNxTy>)
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
  return info;
}

absl::StatusOr<AieLoweringResult> LowerLinalgToAie(
    mlir::ModuleOp linalg_module, const AieLoweringConfig& config,
    const TargetCaps& caps) {
  // Check for matmul before elementwise analysis.
  auto matmul_info = DetectMatmul(linalg_module);
  if (matmul_info.has_value()) {
    LOG(INFO) << "XDNA AIE lowering: detected matmul [" << matmul_info->M
              << "," << matmul_info->K << "]@[" << matmul_info->K << ","
              << matmul_info->N << "] " << matmul_info->element_type;
    return LowerMatmulToAieInternal(*matmul_info, caps, config.num_columns);
  }

  // Step 1: Analyze the linalg module (elementwise path).
  auto program_result = AnalyzeLinalgModule(linalg_module);
  if (!program_result.ok()) return program_result.status();
  LinalgProgramInfo program = std::move(*program_result);

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
  LOG(INFO) << "XDNA AIE lowering: tiling " << per_core_elements
            << " elements/core into " << num_chunks << " chunk(s) of "
            << chunk_size << " (L1 budget: " << l1_limit << "B"
            << (use_mem_tile ? ", mem tile staging" : ", direct") << ")";

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
  for (int c = 0; c < num_cores; c++) {
    int col = start_col + c;
    absl::StrAppend(&aie_mlir, absl::StrFormat(
        "    %%tile_%d_%d = aie.tile(%d, %d)\n", col, shim_row, col, shim_row));
    if (use_mem_tile) {
      absl::StrAppend(&aie_mlir, absl::StrFormat(
          "    %%tile_%d_%d = aie.tile(%d, %d)\n", col, caps.mem_tile_row,
          col, caps.mem_tile_row));
    }
    absl::StrAppend(&aie_mlir, absl::StrFormat(
        "    %%tile_%d_%d = aie.tile(%d, %d)\n", col, compute_row,
        col, compute_row));
  }

  // Pass 2: All ObjectFIFOs and links.
  for (int c = 0; c < num_cores; c++) {
    int col = start_col + c;
    std::string shim_tile = absl::StrFormat("tile_%d_%d", col, shim_row);
    std::string compute_tile = absl::StrFormat("tile_%d_%d", col, compute_row);
    std::string mem_tile_str = absl::StrFormat("tile_%d_%d", col,
                                                caps.mem_tile_row);

    if (use_mem_tile) {
      // Route through memory tile (L2 staging) for tiled execution.
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
      // L1 ObjectFIFOs: mem tile ↔ compute.
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
      // Link L2 ↔ L1 ObjectFIFOs through memory tile.
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
      // Direct shim ↔ compute (no memory tile for single-chunk execution).
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
                                         use_mem_tile, num_cores, start_col);
  if (!npu_seq_or.ok()) return npu_seq_or.status();
  absl::StrAppend(&aie_mlir, *npu_seq_or);

  absl::StrAppend(&aie_mlir, "  }\n");
  absl::StrAppend(&aie_mlir, "}\n");

  LOG(INFO) << "XDNA AIE lowering: generated AIE MLIR:\n" << aie_mlir;

  // Detect if kernel uses float comparison ops that need soft-float stubs.
  bool needs_softfloat = false;
  if (program.single_op.has_value()) {
    const auto& op = program.single_op->kernel_op;
    needs_softfloat = (op == "arith.maximumf" || op == "arith.minimumf");
  }

  return AieLoweringResult{aie_mlir, num_cores, /*use_aievec=*/false,
                           needs_softfloat};
}

}  // namespace xla
