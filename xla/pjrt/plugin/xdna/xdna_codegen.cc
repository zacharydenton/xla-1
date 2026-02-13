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

#include "xla/pjrt/plugin/xdna/xdna_codegen.h"

#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <regex>
#include <string>
#include <sys/wait.h>
#include <vector>

#include "absl/log/log.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/str_format.h"
#include "xla/tsl/platform/errors.h"

namespace xla {
namespace {

// ---------------------------------------------------------------------------
// Tool path helpers. Override via environment variables.
// ---------------------------------------------------------------------------

std::string GetAieOptPath() {
  const char* env = std::getenv("XDNA_AIE_OPT");
  return env ? env : "/opt/mlir-aie/bin/aie-opt";
}

std::string GetAieTranslatePath() {
  const char* env = std::getenv("XDNA_AIE_TRANSLATE");
  return env ? env : "/opt/mlir-aie/bin/aie-translate";
}

std::string GetPeanoClangPath() {
  const char* env = std::getenv("XDNA_PEANO_CLANG");
  return env ? env : "/opt/peano/bin/clang";
}

std::string GetPeanoOptPath() {
  const char* env = std::getenv("XDNA_PEANO_OPT");
  return env ? env : "/opt/peano/bin/opt";
}

std::string GetPeanoLlcPath() {
  const char* env = std::getenv("XDNA_PEANO_LLC");
  return env ? env : "/opt/peano/bin/llc";
}

std::string GetBootgenPath() {
  const char* env = std::getenv("XDNA_BOOTGEN");
  return env ? env : "bootgen";
}

std::string GetXclbinutilPath() {
  const char* env = std::getenv("XDNA_XCLBINUTIL");
  return env ? env : "/opt/xilinx/xrt/bin/xclbinutil";
}

// ---------------------------------------------------------------------------
// File I/O helpers.
// ---------------------------------------------------------------------------

int GetCompileTimeout() {
  const char* env = std::getenv("XDNA_COMPILE_TIMEOUT");
  return env ? std::atoi(env) : 120;
}

absl::Status RunCommand(const std::string& cmd,
                        const std::string& step_label = "") {
  int timeout_secs = GetCompileTimeout();
  auto start = std::chrono::steady_clock::now();

  LOG(INFO) << "XDNA codegen: " << step_label << ": " << cmd;

  // Use popen to capture stdout+stderr, with timeout command for safety.
  std::string timed_cmd = absl::StrCat(
      "timeout ", timeout_secs, " ", cmd, " 2>&1");

  FILE* pipe = popen(timed_cmd.c_str(), "r");
  if (!pipe) {
    return absl::InternalError(
        absl::StrCat("Failed to execute: ", step_label));
  }

  std::string output;
  char buffer[4096];
  while (fgets(buffer, sizeof(buffer), pipe) != nullptr) {
    output.append(buffer);
  }
  int status = pclose(pipe);

  auto elapsed = std::chrono::steady_clock::now() - start;
  double secs = std::chrono::duration<double>(elapsed).count();

  LOG(INFO) << absl::StrFormat("XDNA codegen: %s completed in %.1fs",
                               step_label, secs);

  if (status != 0) {
    int exit_code = WIFEXITED(status) ? WEXITSTATUS(status) : -1;

    // Truncate output if too long.
    if (output.size() > 2000) {
      output = "...(truncated)\n" + output.substr(output.size() - 2000);
    }

    if (exit_code == 124) {
      return absl::DeadlineExceededError(absl::StrFormat(
          "%s timed out after %ds.\nOutput:\n%s",
          step_label, timeout_secs,
          output.empty() ? "(no output)" : output));
    }

    return absl::InternalError(absl::StrFormat(
        "%s failed (exit code %d).\nCommand: %s\nOutput:\n%s",
        step_label, exit_code, cmd,
        output.empty() ? "(no output)" : output));
  }

  return absl::OkStatus();
}

absl::StatusOr<std::vector<uint8_t>> ReadBinaryFile(const std::string& path) {
  std::ifstream file(path, std::ios::binary | std::ios::ate);
  if (!file.is_open()) {
    return absl::NotFoundError(absl::StrCat("Cannot open file: ", path));
  }
  auto size = file.tellg();
  file.seekg(0, std::ios::beg);
  std::vector<uint8_t> data(size);
  file.read(reinterpret_cast<char*>(data.data()), size);
  return data;
}

// Parse a hex instruction text file (one uint32 per line) into a vector.
absl::StatusOr<std::vector<uint32_t>> ReadInstrFile(const std::string& path) {
  std::ifstream file(path);
  if (!file.is_open()) {
    return absl::NotFoundError(
        absl::StrCat("Cannot open instruction file: ", path));
  }
  std::vector<uint32_t> words;
  std::string line;
  while (std::getline(file, line)) {
    // Skip empty lines and comments.
    if (line.empty() || line[0] == '#' || line[0] == '/') continue;
    uint32_t word = std::stoul(line, nullptr, 16);
    words.push_back(word);
  }
  return words;
}

absl::Status WriteFile(const std::string& path, const std::string& content) {
  std::ofstream file(path);
  if (!file.is_open()) {
    return absl::InternalError(absl::StrCat("Cannot write file: ", path));
  }
  file << content;
  return absl::OkStatus();
}

// Patch lowered MLIR to strip core bodies and add elf_file attributes.
// mlir-aie v1.2.1 requires elf_file on aie.core ops for CDO generation.
absl::Status PatchCoreElfAttributes(const std::string& mlir_path) {
  std::ifstream in(mlir_path);
  if (!in.is_open()) {
    return absl::InternalError(
        absl::StrCat("Cannot read MLIR for patching: ", mlir_path));
  }
  std::string content((std::istreambuf_iterator<char>(in)),
                       std::istreambuf_iterator<char>());
  in.close();

  // Match patterns like: %core_X_Y = aie.core(%tile_X_Y) {
  std::regex core_re(R"((%\S+\s*=\s*aie\.core\(%(\S+)\)\s*)\{)");
  std::string result;
  std::smatch match;
  std::string::const_iterator search_start = content.cbegin();

  while (std::regex_search(search_start, content.cend(), match, core_re)) {
    // Append everything before the match.
    result.append(search_start, match[0].first);

    std::string prefix = match[1].str();
    std::string tile_name = match[2].str();

    // Extract col and row from tile name (tile_X_Y).
    int col = 0, row = 0;
    if (sscanf(tile_name.c_str(), "tile_%d_%d", &col, &row) != 2) {
      return absl::InternalError(
          absl::StrCat("Cannot parse tile name: ", tile_name));
    }

    std::string elf_name = absl::StrFormat("core_%d_%d.elf", col, row);

    // Find the matching closing brace by tracking brace depth.
    auto body_start = match[0].second;
    int depth = 1;
    auto pos = body_start;
    while (pos != content.cend() && depth > 0) {
      if (*pos == '{') depth++;
      else if (*pos == '}') depth--;
      if (depth > 0) ++pos;
    }

    if (depth != 0) {
      return absl::InternalError("Unmatched braces in aie.core op");
    }

    // Replace body with just aie.end and add elf_file attribute.
    result.append(prefix);
    result.append("{\n      aie.end\n    } {elf_file = \"");
    result.append(elf_name);
    result.append("\"}");

    search_start = pos + 1;
  }

  // Append remainder.
  result.append(search_start, content.cend());

  return WriteFile(mlir_path, result);
}

// ---------------------------------------------------------------------------
// xclbin metadata generation (matches aiecc.py's output).
// ---------------------------------------------------------------------------

std::string GenerateMemTopologyJson() {
  return R"({
  "mem_topology": {
    "m_count": "2",
    "m_mem_data": [
      {
        "m_type": "MEM_DRAM",
        "m_used": "1",
        "m_sizeKB": "0x10000",
        "m_tag": "HOST",
        "m_base_address": "0x4000000"
      },
      {
        "m_type": "MEM_DRAM",
        "m_used": "1",
        "m_sizeKB": "0xc000",
        "m_tag": "SRAM",
        "m_base_address": "0x4000000"
      }
    ]
  }
})";
}

std::string GenerateKernelsJson(const std::string& kernel_name,
                                int num_data_args) {
  // DPU kernel signature: opcode (scalar) + instr (global/SRAM) +
  // ninstr (scalar) + N data buffer args (global/HOST).
  std::string args = R"(
          {"name": "opcode", "address-qualifier": "SCALAR", "type": "uint64_t", "offset": "0x00"},
          {"name": "instr", "memory-connection": "SRAM", "address-qualifier": "GLOBAL", "type": "char *", "offset": "0x08"},
          {"name": "ninstr", "address-qualifier": "SCALAR", "type": "uint32_t", "offset": "0x10"})";

  // Data BO offsets start immediately after ninstr (uint32_t at 0x10).
  // XRT kernel arg binding uses these offsets; the NPU hardware handles
  // unaligned pointer args via its own DMA mechanism.
  int offset = 0x14;
  for (int i = 0; i < num_data_args; ++i) {
    absl::StrAppend(&args, absl::StrFormat(
        ",\n          {\"name\": \"bo%d\", \"memory-connection\": \"HOST\", "
        "\"address-qualifier\": \"GLOBAL\", \"type\": \"void*\", "
        "\"offset\": \"%s\"}",
        i, absl::StrFormat("0x%02x", offset)));
    offset += 0x08;
  }

  return absl::StrFormat(R"({
  "ps-kernels": {
    "kernels": [
      {
        "name": "%s",
        "type": "dpu",
        "extended-data": {
          "subtype": "DPU",
          "functional": "0",
          "dpu_kernel_id": "0x901"
        },
        "arguments": [%s
        ],
        "instances": [{"name": "%sInst"}]
      }
    ]
  }
})", kernel_name, args, kernel_name);
}

std::string GenerateAiePartitionJson(const std::string& pdi_path,
                                     const TargetCaps& caps,
                                     int num_cores) {
  // Use actual num_cores for column_width so the partition request matches
  // the columns allocated by the aie.device(npu2_Ncol) directive.
  int column_width = num_cores;
  return absl::StrFormat(R"({
  "aie_partition": {
    "name": "QoS",
    "operations_per_cycle": "2048",
    "inference_fingerprint": "23423",
    "pre_post_fingerprint": "12345",
    "partition": {
      "column_width": %d,
      "start_columns": [%d]
    },
    "PDIs": [
      {
        "uuid": "00000000-0000-0000-0000-000000000000",
        "file_name": "%s",
        "cdo_groups": [
          {
            "name": "DPU",
            "type": "PRIMARY",
            "pdi_id": "0x01",
            "dpu_kernel_ids": ["0x901"],
            "pre_cdo_groups": ["0xC1"]
          }
        ]
      }
    ]
  }
})", column_width, caps.partition_start_column, pdi_path);
}

std::string GenerateDesignBif(const std::string& workdir) {
  return absl::StrFormat(R"(all:
{
  id_code = 0x14ca8093
  extended_id_code = 0x01
  id = 0x2
  image
  {
    name=aie_image, id=0x1c000000
    partition
    {
      id=0x01
      type=cdo
      file=%s/main_aie_cdo_elfs.bin
      file=%s/main_aie_cdo_init.bin
      file=%s/main_aie_cdo_enable.bin
    }
  }
})", workdir, workdir, workdir);
}

// ---------------------------------------------------------------------------
// Check that all required tools exist.
// ---------------------------------------------------------------------------

absl::Status CheckTools(const std::string& aie_opt,
                        const std::string& aie_translate,
                        const std::string& peano_clang,
                        const std::string& bootgen,
                        const std::string& xclbinutil) {
  auto check = [](const std::string& tool, const std::string& name,
                   const std::string& env_var,
                   const std::string& install_hint) -> absl::Status {
    // For tools found via PATH (like bootgen), use `which` check.
    if (tool.find('/') == std::string::npos) {
      std::string cmd = absl::StrCat("which ", tool, " > /dev/null 2>&1");
      if (std::system(cmd.c_str()) != 0) {
        return absl::NotFoundError(absl::StrCat(
            name, " not found in PATH. Set ", env_var, " or ", install_hint));
      }
      return absl::OkStatus();
    }
    if (!std::filesystem::exists(tool)) {
      return absl::NotFoundError(absl::StrCat(
          name, " not found at ", tool, ". Set ", env_var, " or ",
          install_hint));
    }
    return absl::OkStatus();
  };

  TF_RETURN_IF_ERROR(check(aie_opt, "aie-opt", "XDNA_AIE_OPT",
                           "install mlir-aie to /opt/mlir-aie"));
  TF_RETURN_IF_ERROR(check(aie_translate, "aie-translate",
                           "XDNA_AIE_TRANSLATE",
                           "install mlir-aie to /opt/mlir-aie"));
  TF_RETURN_IF_ERROR(check(peano_clang, "Peano clang", "XDNA_PEANO_CLANG",
                           "install Peano to /opt/peano"));
  TF_RETURN_IF_ERROR(check(bootgen, "bootgen", "XDNA_BOOTGEN",
                           "build from https://github.com/Xilinx/bootgen"));
  TF_RETURN_IF_ERROR(check(xclbinutil, "xclbinutil", "XDNA_XCLBINUTIL",
                           "install XRT to /opt/xilinx/xrt"));
  return absl::OkStatus();
}

}  // namespace

absl::StatusOr<XdnaCodegenResult> GenerateXclbinFromAie(
    const std::string& aie_mlir, int num_data_args, const TargetCaps& caps,
    int num_cores, bool use_aievec, bool convert_vector_to_aievec,
    bool needs_matmul_workarounds, bool needs_softfloat_stubs) {
  // Create a temporary working directory.
  auto tmpdir = std::filesystem::temp_directory_path() / "xdna_codegen_XXXXXX";
  std::string tmpdir_str = tmpdir.string();
  char* tmpdir_cstr = tmpdir_str.data();
  if (!mkdtemp(tmpdir_cstr)) {
    return absl::InternalError("Failed to create temp directory.");
  }
  std::string workdir(tmpdir_cstr);
  LOG(INFO) << "XDNA codegen: working directory: " << workdir;

  // Resolve tool paths.
  std::string aie_opt = GetAieOptPath();
  std::string aie_translate = GetAieTranslatePath();
  std::string peano_clang = GetPeanoClangPath();
  std::string peano_opt = GetPeanoOptPath();
  std::string peano_llc = GetPeanoLlcPath();
  std::string bootgen = GetBootgenPath();
  std::string xclbinutil = GetXclbinutilPath();

  TF_RETURN_IF_ERROR(
      CheckTools(aie_opt, aie_translate, peano_clang, bootgen, xclbinutil));

  // -----------------------------------------------------------------------
  // Step 1: Write AIE MLIR to file.
  // -----------------------------------------------------------------------
  std::string input_mlir = workdir + "/input.mlir";
  TF_RETURN_IF_ERROR(WriteFile(input_mlir, aie_mlir));

  // -----------------------------------------------------------------------
  // Step 2: aie-opt lowering passes (INPUT_WITH_ADDRESSES_PIPELINE).
  // Matches aiecc.py's pipeline order. Key passes:
  //   - objectFifo-stateful-transform: lowers ObjectFIFOs to buffers + DMAs
  //   - assign-bd-ids: assigns buffer descriptor IDs (must be AFTER objFIFO)
  //   - generate-column-control-overlay: sets up tile controller routing
  //     (needed for CDO commands to reach compute tiles on NPU)
  //   - create-pathfinder-flows: routes logical flows through physical switches
  //   - assign-buffer-addresses: allocates memory in tile local stores
  // -----------------------------------------------------------------------
  // Step numbering: 1 shared + N per core + 5 shared = total.
  bool needs_rt_stubs = needs_matmul_workarounds || needs_softfloat_stubs;
  int steps_per_core = needs_rt_stubs ? 8 : 7;  // +1 for rt stubs
  int total_compile_steps = steps_per_core * num_cores + 6;
  int step = 1;

  std::string lowered_mlir = workdir + "/lowered.mlir";
  TF_RETURN_IF_ERROR(RunCommand(absl::StrCat(
      aie_opt,
      " --lower-affine"
      " --aie-canonicalize-device"
      " --aie-assign-lock-ids"
      " --aie-register-objectFifos"
      " --aie-objectFifo-stateful-transform"
      " --aie-assign-bd-ids"
      " --aie-lower-cascade-flows"
      " --aie-lower-broadcast-packet"
      " --aie-lower-multicast"
      " --aie-assign-tile-controller-ids"
      " \"--aie-generate-column-control-overlay=route-shim-to-tile-ctrl=false\""
      " --aie-create-pathfinder-flows"
      " --aie-assign-buffer-addresses"
      " --convert-scf-to-cf"
      " ", input_mlir, " -o ", lowered_mlir),
      absl::StrFormat("Step %d/%d: AIE lowering passes",
                      step++, total_compile_steps)));

  // -----------------------------------------------------------------------
  // Step 3: Per-core code compilation with Peano.
  // Extract each core's code → lower to LLVM → compile to ELF.
  // Loop over all compute columns used.
  // -----------------------------------------------------------------------

  for (int c = 0; c < num_cores; c++) {
    int col = caps.partition_start_column + c;
    int row = caps.first_compute_row;
    std::string core = absl::StrFormat("core_%d_%d", col, row);

    // 3a: Extract core code with AIE-specific lowering.
    std::string core_mlir = absl::StrFormat("%s/%s.mlir", workdir, core);
    TF_RETURN_IF_ERROR(RunCommand(absl::StrCat(
        aie_opt,
        " --aie-localize-locks"
        " --aie-normalize-address-spaces"
        " \"--aie-standard-lowering=tilecol=", col, " tilerow=", row, "\""
        " --aiex-standard-lowering"
        " ", lowered_mlir, " -o ", core_mlir),
        absl::StrFormat("Step %d/%d: Core (%d,%d) extraction",
                        step++, total_compile_steps, col, row)));

    // 3b: Lower to LLVM dialect.
    // aievec pass pipeline:
    //   --convert-vector-to-aievec: converts arith.maximumf/minimumf on vectors
    //     to aievec.max/min ops. Safe for elementwise (1D vector.load/store).
    //     NOT used for matmul (breaks 2D vector.transfer_read → aievec.upd).
    //   --convert-aievec-to-llvm: lowers aievec ops → xllvm intrinsics.
    //     Used by both matmul (matmul_aie2p → MAC intrinsics) and elementwise
    //     (max/min → VectorMaxLtBf16 intrinsics).
    std::string aievec_passes;
    if (convert_vector_to_aievec) {
      aievec_passes =
          " \"--convert-vector-to-aievec=aie-target=aie2p\""
          " \"--convert-aievec-to-llvm=aie-target=aie2p\"";
    } else if (use_aievec) {
      aievec_passes =
          " \"--convert-aievec-to-llvm=aie-target=aie2p\"";
    }
    std::string core_opt_mlir = absl::StrFormat("%s/%s.opt.mlir", workdir, core);
    TF_RETURN_IF_ERROR(RunCommand(absl::StrCat(
        aie_opt,
        " --canonicalize --cse",
        aievec_passes,
        " --convert-vector-to-llvm --expand-strided-metadata"
        " --lower-affine --convert-math-to-llvm --convert-index-to-llvm"
        " --arith-expand --convert-arith-to-llvm --finalize-memref-to-llvm"
        " \"--convert-func-to-llvm=use-bare-ptr-memref-call-conv\""
        " --convert-scf-to-cf --convert-cf-to-llvm"
        " --convert-ub-to-llvm --canonicalize --cse"
        " ", core_mlir, " -o ", core_opt_mlir),
        absl::StrFormat("Step %d/%d: Core (%d,%d) LLVM lowering",
                        step++, total_compile_steps, col, row)));

    // 3b.1: Fix ub.poison ops in the MLIR output.
    // convert-aievec-to-llvm generates ub.poison for padding shuffle masks,
    // but aie-translate doesn't have the UB dialect. Replace with llvm.mlir.poison.
    // For 1D vector types (already valid LLVM types) this is a direct substitution.
    // For 2D vector types like vector<4x4xf32>, we replace both the ub.poison
    // AND the unrealized_conversion_cast that converts it to the LLVM type.
    if (use_aievec) {
      std::ifstream mlir_in(core_opt_mlir);
      if (!mlir_in.is_open()) {
        return absl::InternalError(
            absl::StrCat("Cannot read MLIR: ", core_opt_mlir));
      }
      std::string mlir_content((std::istreambuf_iterator<char>(mlir_in)),
                               std::istreambuf_iterator<char>());
      mlir_in.close();

      // Replace "ub.poison : vector<NxTYPE>" → "llvm.mlir.poison : vector<NxTYPE>"
      // for 1D vector types (already valid LLVM types).
      {
        std::string from = "ub.poison : vector<";
        size_t pos = 0;
        while ((pos = mlir_content.find(from, pos)) != std::string::npos) {
          // Check if this is a 1D or 2D vector by looking for "x" count.
          // Find the closing ">" to get the full type.
          size_t type_start = pos + strlen("ub.poison : ");
          size_t gt = mlir_content.find('>', type_start);
          if (gt == std::string::npos) { pos += from.size(); continue; }
          std::string vec_type = mlir_content.substr(type_start,
                                                      gt + 1 - type_start);
          // Count 'x' chars to determine dimensionality.
          int x_count = 0;
          for (char c : vec_type) { if (c == 'x') x_count++; }

          if (x_count == 1) {
            // 1D vector: direct replacement.
            mlir_content.replace(pos, strlen("ub.poison"),
                                 "llvm.mlir.poison");
            pos += strlen("llvm.mlir.poison");
          } else {
            // 2D vector (e.g., vector<4x4xf32>): replace with poison of the
            // LLVM array type, and remove the unrealized_conversion_cast.
            // Extract the SSA name: "    %VAR = ub.poison : vector<4x4xf32>"
            size_t line_start = mlir_content.rfind('\n', pos);
            line_start = (line_start == std::string::npos) ? 0 : line_start + 1;
            size_t eq = mlir_content.find('=', line_start);
            if (eq == std::string::npos) { pos += from.size(); continue; }
            std::string ub_var = mlir_content.substr(line_start,
                                                      eq - line_start);
            while (!ub_var.empty() && ub_var.back() == ' ') ub_var.pop_back();
            size_t first = ub_var.find_first_not_of(' ');
            if (first != std::string::npos) ub_var = ub_var.substr(first);

            // Find the unrealized_conversion_cast that uses this variable.
            // Pattern: "%CAST = builtin.unrealized_conversion_cast %VAR :
            //           vector<4x4xf32> to !llvm.array<...>"
            std::string cast_marker = absl::StrCat(
                "builtin.unrealized_conversion_cast ", ub_var, " : ");
            size_t cast_pos = mlir_content.find(cast_marker, pos);
            if (cast_pos != std::string::npos) {
              // Find the LLVM type after "to ".
              size_t to_pos = mlir_content.find(" to ", cast_pos);
              if (to_pos != std::string::npos) {
                size_t llvm_type_start = to_pos + 4;
                size_t cast_line_end = mlir_content.find('\n', cast_pos);
                std::string llvm_type = mlir_content.substr(
                    llvm_type_start,
                    cast_line_end - llvm_type_start);

                // Get the cast result variable name.
                size_t cast_line_start = mlir_content.rfind('\n',
                    cast_pos > 0 ? cast_pos - 1 : 0);
                cast_line_start = (cast_line_start == std::string::npos)
                    ? 0 : cast_line_start + 1;
                size_t cast_eq = mlir_content.find('=', cast_line_start);
                std::string cast_var = mlir_content.substr(
                    cast_line_start, cast_eq - cast_line_start);
                while (!cast_var.empty() && cast_var.back() == ' ')
                  cast_var.pop_back();
                size_t cf = cast_var.find_first_not_of(' ');
                if (cf != std::string::npos) cast_var = cast_var.substr(cf);

                // Replace the cast line: "%CAST = llvm.mlir.poison : LLVM_TYPE"
                std::string new_cast = absl::StrFormat(
                    "    %s = llvm.mlir.poison : %s", cast_var, llvm_type);
                mlir_content.replace(cast_line_start,
                    cast_line_end - cast_line_start, new_cast);

                // Remove the original ub.poison line.
                size_t ub_line_end = mlir_content.find('\n', pos);
                if (ub_line_end == std::string::npos)
                  ub_line_end = mlir_content.size();
                mlir_content.erase(line_start,
                    ub_line_end - line_start + 1);
                pos = line_start;
                continue;
              }
            }
            // Fallback: just skip if we can't find the cast.
            pos += from.size();
          }
        }
      }

      // Fix poison-padded shufflevectors: replace -1 indices with zero padding.
      // convert-aievec-to-llvm pads MAC operands (4x8→8x8 LHS, 4x4→8x4 ACC)
      // using shufflevector with -1 (poison) mask indices. Peano reuses
      // registers for poison positions, so stale data from previous MAC
      // iterations leaks into results. Replace with explicit zero padding.
      {
        int zp_counter = 0;
        size_t pos = 0;
        while ((pos = mlir_content.find("llvm.shufflevector ", pos))
                != std::string::npos) {
          // Find the full line.
          size_t line_start = mlir_content.rfind('\n', pos);
          line_start = (line_start == std::string::npos) ? 0 : line_start + 1;
          size_t line_end = mlir_content.find('\n', pos);
          if (line_end == std::string::npos) line_end = mlir_content.size();

          std::string line = mlir_content.substr(line_start,
                                                  line_end - line_start);

          // Only process shufflevectors with poison (-1) indices.
          if (line.find("-1") == std::string::npos) {
            pos = line_end;
            continue;
          }

          // Parse: "%VAR = llvm.shufflevector %SRC, %SRC2 [...] : vector<NxT>"
          size_t bo = line.find('[');
          size_t bc = line.find(']');
          if (bo == std::string::npos || bc == std::string::npos) {
            pos = line_end;
            continue;
          }

          // Parse the indices.
          std::vector<int> indices;
          size_t ci = bo + 1;
          while (ci < bc) {
            while (ci < bc && (line[ci] == ' ' || line[ci] == ',')) ci++;
            if (ci >= bc) break;
            size_t num_start = ci;
            while (ci < bc && line[ci] != ',' && line[ci] != ' ') ci++;
            indices.push_back(std::stoi(line.substr(num_start, ci - num_start)));
          }

          // Count sequential non-poison elements = input vector width (N).
          int n_real = 0;
          for (int i = 0; i < (int)indices.size(); i++) {
            if (indices[i] == -1) break;
            if (indices[i] != i) { n_real = -1; break; }
            n_real++;
          }
          if (n_real <= 0) { pos = line_end; continue; }

          // Verify all remaining indices are -1.
          bool all_poison = true;
          for (int i = n_real; i < (int)indices.size(); i++) {
            if (indices[i] != -1) { all_poison = false; break; }
          }
          if (!all_poison) { pos = line_end; continue; }

          int total = indices.size();

          // Extract element type from ": vector<NxTYPE>".
          size_t colon = line.rfind(':');
          if (colon == std::string::npos) { pos = line_end; continue; }
          size_t vlt = line.find('<', colon);
          size_t vgt = line.find('>', vlt);
          if (vlt == std::string::npos || vgt == std::string::npos) {
            pos = line_end; continue;
          }
          std::string vec_inner = line.substr(vlt + 1, vgt - vlt - 1);
          size_t xp = vec_inner.find('x');
          std::string elem_type = vec_inner.substr(xp + 1);
          std::string input_vec_type = absl::StrFormat(
              "vector<%dx%s>", n_real, elem_type);

          // Extract %VAR (result name).
          size_t eq = line.find('=');
          std::string var = line.substr(0, eq);
          while (!var.empty() && var.back() == ' ') var.pop_back();
          size_t fnws = var.find_first_not_of(' ');
          if (fnws != std::string::npos) var = var.substr(fnws);

          // Extract %SRC (first source operand).
          size_t sv_end = line.find("llvm.shufflevector ")
                          + strlen("llvm.shufflevector ");
          size_t comma1 = line.find(',', sv_end);
          std::string src = line.substr(sv_end, comma1 - sv_end);
          while (!src.empty() && src.back() == ' ') src.pop_back();

          // Build new indices: [0..N-1] from data, rest from zero operand.
          // Clamp indices >= 2*N to valid second-operand positions to avoid
          // out-of-range shufflevector masks (e.g., ACC2048 widening 8→32).
          std::string new_indices;
          for (int i = 0; i < total; i++) {
            if (i > 0) absl::StrAppend(&new_indices, ", ");
            int idx = i;
            if (idx >= 2 * n_real) {
              idx = (idx % n_real) + n_real;
            }
            absl::StrAppend(&new_indices, idx);
          }

          // Build zero constant and new shufflevector.
          // Use integer zero for integer types, float zero for float types.
          bool is_int_type = (elem_type[0] == 'i');
          std::string zero_literal = is_int_type ? "0" : "0.000000e+00";
          std::string zero_var = absl::StrFormat("%%_zp_%d", zp_counter++);
          std::string zero_line = absl::StrFormat(
              "    %s = llvm.mlir.constant(dense<%s> : %s) : %s",
              zero_var, zero_literal, input_vec_type, input_vec_type);
          std::string new_sv = absl::StrFormat(
              "    %s = llvm.shufflevector %s, %s [%s] : %s",
              var, src, zero_var, new_indices, input_vec_type);

          std::string replacement = absl::StrCat(zero_line, "\n", new_sv);
          mlir_content.replace(line_start, line_end - line_start, replacement);
          pos = line_start + replacement.size();
        }
      }

      TF_RETURN_IF_ERROR(WriteFile(core_opt_mlir, mlir_content));
    }

    // 3c: Translate to LLVM IR.
    std::string core_ll = absl::StrFormat("%s/%s.ll", workdir, core);
    TF_RETURN_IF_ERROR(RunCommand(absl::StrCat(
        aie_translate, " --mlir-to-llvmir ", core_opt_mlir, " -o ", core_ll),
        absl::StrFormat("Step %d/%d: Core (%d,%d) LLVM IR translation",
                        step++, total_compile_steps, col, row)));

    // 3d: Fix intrinsic names for the target ISA.
    // mlir-aie emits "llvm.aie2.*" for objectfifo ops (acquire/release/etc.)
    // and "llvm.aie.*" for generic helpers (v16float.undef, upd.w, etc.).
    // Both must be renamed to "llvm.<isa_target>.*" for Peano.
    // Order matters: rename "llvm.aie2." first (skipping "llvm.aie2p."),
    // then "llvm.aie." (skipping already-renamed "llvm.aie2p.").
    {
      std::ifstream ll_in(core_ll);
      if (!ll_in.is_open()) {
        return absl::InternalError(
            absl::StrCat("Cannot read LLVM IR: ", core_ll));
      }
      std::string ll_content((std::istreambuf_iterator<char>(ll_in)),
                             std::istreambuf_iterator<char>());
      ll_in.close();

      std::string target_prefix = absl::StrCat("llvm.", caps.isa_target, ".");

      // Pass 1: "llvm.aie2." → target (skip "llvm.aie2p.").
      {
        std::string from = "llvm.aie2.";
        size_t pos = 0;
        while ((pos = ll_content.find(from, pos)) != std::string::npos) {
          if (pos + from.size() < ll_content.size() &&
              ll_content[pos + from.size()] == 'p') {
            pos += from.size();
            continue;
          }
          ll_content.replace(pos, from.size(), target_prefix);
          pos += target_prefix.size();
        }
      }

      // Pass 2: Replace aievec helper intrinsics with standard LLVM IR.
      // The aievec-to-llvm pass emits llvm.aie.{v16float,v32float}.undef()
      // and llvm.aie.upd.w.{v16float,v32float}.lo() for vector widening.
      // Peano doesn't support these; replace with shufflevector.
      //
      // Pattern (always consecutive lines):
      //   %X = call <16 x float> @llvm.aie.v16float.undef()
      //   %Y = call <16 x float> @llvm.aie.upd.w.v16float.lo(... %X, <8 x float> %Z)
      // → %Y = shufflevector <8 x float> %Z, <8 x float> poison, <16 x i32> <0..7,poison..>
      //
      // Similarly for <32 x bfloat>/<16 x bfloat> (v32float variant).
      {
        // Remove declare lines for these intrinsics.
        for (const auto& intrinsic : {"llvm.aie.v16float.undef",
                                       "llvm.aie.upd.w.v16float.lo",
                                       "llvm.aie.v32float.undef",
                                       "llvm.aie.upd.w.v32float.lo"}) {
          std::string marker = absl::StrCat("@", intrinsic);
          size_t pos = 0;
          while ((pos = ll_content.find(marker, pos)) != std::string::npos) {
            // Find start of this line.
            size_t line_start = ll_content.rfind('\n', pos);
            line_start = (line_start == std::string::npos) ? 0 : line_start + 1;
            // Only remove if it's a declare line.
            if (ll_content.substr(line_start, 7) == "declare") {
              size_t line_end = ll_content.find('\n', pos);
              if (line_end == std::string::npos) line_end = ll_content.size();
              ll_content.erase(line_start, line_end - line_start + 1);
              pos = line_start;
            } else {
              pos += marker.size();
            }
          }
        }

        // Replace undef+upd pairs with shufflevector.
        // Match: %VAR = call <TYPE> @llvm.aie.upd.w.vNTYPE.lo(<TYPE> %UNDEF, <HALF> %SRC)
        auto replace_upd = [&](const std::string& upd_name,
                                const std::string& full_type,
                                const std::string& half_type,
                                const std::string& shuffle_mask) {
          std::string call_marker = absl::StrCat("@", upd_name, "(");
          size_t pos = 0;
          while ((pos = ll_content.find(call_marker, pos)) != std::string::npos) {
            // Find the start of this call (the %VAR = call ... line).
            size_t line_start = ll_content.rfind('\n', pos);
            line_start = (line_start == std::string::npos) ? 0 : line_start + 1;

            // Find end of call line.
            size_t line_end = ll_content.find('\n', pos);
            if (line_end == std::string::npos) line_end = ll_content.size();

            std::string line = ll_content.substr(line_start,
                                                  line_end - line_start);

            // Extract result name (%VAR2 from "%VAR2 = call ...").
            size_t eq = line.find('=');
            if (eq == std::string::npos) { pos = line_end; continue; }
            std::string result_var = line.substr(0, eq);
            // Trim trailing spaces.
            while (!result_var.empty() && result_var.back() == ' ')
              result_var.pop_back();
            // Trim leading spaces.
            size_t first = result_var.find_first_not_of(' ');
            if (first != std::string::npos)
              result_var = result_var.substr(first);

            // Extract source operand: last argument in the call.
            // Pattern: ..., <HALF> %SRC)
            size_t last_pct = line.rfind('%');
            size_t last_paren = line.rfind(')');
            if (last_pct == std::string::npos || last_paren == std::string::npos) {
              pos = line_end; continue;
            }
            std::string src_var = line.substr(last_pct,
                                               last_paren - last_pct);

            // Also remove the preceding undef call line (should be right before).
            size_t prev_line_end = line_start > 0 ? line_start - 1 : 0;
            size_t prev_line_start = ll_content.rfind('\n', prev_line_end > 0 ? prev_line_end - 1 : 0);
            prev_line_start = (prev_line_start == std::string::npos) ? 0 : prev_line_start + 1;
            std::string prev_line = ll_content.substr(prev_line_start,
                prev_line_end - prev_line_start);
            bool remove_prev = (prev_line.find("undef()") != std::string::npos);

            // Build replacement.
            std::string replacement = absl::StrFormat(
                "  %s = shufflevector %s %s, %s poison, %s",
                result_var, half_type, src_var, half_type, shuffle_mask);

            size_t erase_start = remove_prev ? prev_line_start : line_start;
            size_t erase_end = line_end;
            ll_content.replace(erase_start, erase_end - erase_start,
                               replacement);
            pos = erase_start + replacement.size();
          }
        };

        // <8 x float> → <16 x float> (f32 accumulator widening)
        replace_upd("llvm.aie.upd.w.v16float.lo",
                     "<16 x float>", "<8 x float>",
                     "<16 x i32> <i32 0, i32 1, i32 2, i32 3, "
                     "i32 4, i32 5, i32 6, i32 7, "
                     "i32 poison, i32 poison, i32 poison, i32 poison, "
                     "i32 poison, i32 poison, i32 poison, i32 poison>");

        // <16 x bfloat> → <32 x bfloat> (bf16 input widening)
        replace_upd("llvm.aie.upd.w.v32float.lo",
                     "<32 x bfloat>", "<16 x bfloat>",
                     "<32 x i32> <i32 0, i32 1, i32 2, i32 3, "
                     "i32 4, i32 5, i32 6, i32 7, "
                     "i32 8, i32 9, i32 10, i32 11, "
                     "i32 12, i32 13, i32 14, i32 15, "
                     "i32 poison, i32 poison, i32 poison, i32 poison, "
                     "i32 poison, i32 poison, i32 poison, i32 poison, "
                     "i32 poison, i32 poison, i32 poison, i32 poison, "
                     "i32 poison, i32 poison, i32 poison, i32 poison>");
      }

      TF_RETURN_IF_ERROR(WriteFile(core_ll, ll_content));
    }

    // 3e: Peano opt.
    // Disable loop unrolling: Peano (Jan 2025) has a VLIW scheduling bug
    // where unrolled scalar loops produce incorrect results. The misscheduled
    // stores write stale register values (typically the first element of an
    // input buffer) to wrong output positions. This affects any loop with
    // >4 iterations when fully unrolled. Disabling unrolling keeps a clean
    // loop that Peano's backend schedules correctly.
    std::string core_opt_ll = absl::StrFormat("%s/%s.opt.ll", workdir, core);
    TF_RETURN_IF_ERROR(RunCommand(absl::StrCat(
        peano_opt,
        " '--passes=default<O0>,strip'"
        " -vectorize-slp=false"
        " -vectorize-loops=false"
        " -unroll-threshold=0"
        " -S ",
        core_ll, " -o ", core_opt_ll),
        absl::StrFormat("Step %d/%d: Core (%d,%d) Peano optimization",
                        step++, total_compile_steps, col, row)));

    // 3f: Peano llc.
    // Disable --aie-loop-aware for vectorized matmul: Peano's iterative loop
    // scheduler (Feb 2026) miscompiles single-block loops containing BFP MAC
    // intrinsics, producing hangs or wrong results depending on trip count.
    // Not needed for elementwise aievec (vmax.ltbf16 etc.) — simple ops, no loop issue.
    std::string llc_opt = "-O2";
    std::string llc_extra;
    if (needs_matmul_workarounds) {
      llc_extra = " --aie-loop-aware=false";
    }
    std::string core_obj = absl::StrFormat("%s/%s.o", workdir, core);
    TF_RETURN_IF_ERROR(RunCommand(absl::StrCat(
        peano_llc, " ", core_opt_ll,
        " ", llc_opt, llc_extra, " --march=", caps.isa_target,
        " --function-sections --filetype=obj"
        " -o ", core_obj),
        absl::StrFormat("Step %d/%d: Core (%d,%d) Peano codegen (%s%s)",
                        step++, total_compile_steps, col, row, llc_opt,
                        llc_extra)));

    // 3g: Generate linker script.
    std::string core_ldscript = absl::StrFormat("%s/%s.ld.script", workdir, core);
    TF_RETURN_IF_ERROR(RunCommand(absl::StrCat(
        aie_translate,
        " --aie-generate-ldscript --tilecol=", col, " --tilerow=", row, " ",
        lowered_mlir, " -o ", core_ldscript),
        absl::StrFormat("Step %d/%d: Core (%d,%d) linker script",
                        step++, total_compile_steps, col, row)));

    // 3h: Compile runtime stubs (soft-float and integer intrinsics).
    std::string rt_stubs_obj;
    if (needs_rt_stubs) {
      std::string rt_src = absl::StrFormat("%s/rt_stubs.c", workdir);
      rt_stubs_obj = absl::StrFormat("%s/rt_stubs.o", workdir);
      std::string rt_code;
      if (needs_matmul_workarounds) {
        // __muldi3: 64-bit integer multiply stub needed by vectorized matmul
        // (MAC intrinsics generate i64 multiplies for address calculation).
        // Not needed for elementwise aievec ops (vmax, etc.).
        absl::StrAppend(&rt_code,
            "typedef unsigned int su_int;\n"
            "typedef long long di_int;\n"
            "typedef unsigned long long du_int;\n"
            "di_int __muldi3(di_int a, di_int b) {\n"
            "    su_int al=(su_int)a, ah=(su_int)(a>>32);\n"
            "    su_int bl=(su_int)b, bh=(su_int)(b>>32);\n"
            "    du_int lo=(du_int)al*bl;\n"
            "    su_int hi=(su_int)(lo>>32)+ah*bl+al*bh;\n"
            "    return ((di_int)hi<<32)|(su_int)lo;\n"
            "}\n");
      }
      if (needs_softfloat_stubs) {
        // Soft-float comparison stubs for Peano's baremetal AIE2p target.
        // arith.cmpf lowers to fcmp which needs __gtsf2, __ltsf2, etc.
        // These follow GCC's soft-float ABI: return >0 if true, <=0 if false.
        absl::StrAppend(&rt_code,
            "typedef union { float f; unsigned int i; } fu_t;\n"
            "static int _fcmp(float a, float b) {\n"
            "    fu_t ua = {a}, ub = {b};\n"
            "    unsigned ea = (ua.i >> 23) & 0xFF;\n"
            "    unsigned eb = (ub.i >> 23) & 0xFF;\n"
            "    unsigned ma = ua.i & 0x7FFFFF;\n"
            "    unsigned mb = ub.i & 0x7FFFFF;\n"
            "    if ((ea == 0xFF && ma) || (eb == 0xFF && mb)) return -2;\n"
            "    int sa = ua.i >> 31, sb = ub.i >> 31;\n"
            "    if (ua.i == 0x80000000u) ua.i = 0;\n"
            "    if (ub.i == 0x80000000u) ub.i = 0;\n"
            "    if (!sa && !sb) return (ua.i > ub.i) ? 1 : (ua.i == ub.i) ? 0 : -1;\n"
            "    if (sa && sb) return (ua.i < ub.i) ? 1 : (ua.i == ub.i) ? 0 : -1;\n"
            "    if (sa) return -1;\n"
            "    return 1;\n"
            "}\n"
            "int __gtsf2(float a, float b) { return _fcmp(a, b); }\n"
            "int __gesf2(float a, float b) { return _fcmp(a, b); }\n"
            "int __ltsf2(float a, float b) { return _fcmp(a, b); }\n"
            "int __lesf2(float a, float b) { return _fcmp(a, b); }\n"
            "int __eqsf2(float a, float b) { return _fcmp(a, b); }\n"
            "int __nesf2(float a, float b) { return _fcmp(a, b); }\n"
            "int __unordsf2(float a, float b) {\n"
            "    fu_t ua = {a}, ub = {b};\n"
            "    unsigned ea = (ua.i >> 23) & 0xFF;\n"
            "    unsigned eb = (ub.i >> 23) & 0xFF;\n"
            "    unsigned ma = ua.i & 0x7FFFFF;\n"
            "    unsigned mb = ub.i & 0x7FFFFF;\n"
            "    return (ea == 0xFF && ma) || (eb == 0xFF && mb);\n"
            "}\n");
      }
      TF_RETURN_IF_ERROR(WriteFile(rt_src, rt_code));
      TF_RETURN_IF_ERROR(RunCommand(absl::StrCat(
          peano_clang,
          " --target=", caps.isa_target, "-none-elf -O0 -c ",
          rt_src, " -o ", rt_stubs_obj),
          absl::StrFormat("Step %d/%d: Core (%d,%d) runtime stubs",
                          step++, total_compile_steps, col, row)));
    }

    // 3i: Peano clang link.
    std::string core_elf = absl::StrFormat("%s/%s.elf", workdir, core);
    TF_RETURN_IF_ERROR(RunCommand(absl::StrCat(
        peano_clang,
        " -O2 --target=", caps.isa_target, "-none-elf -nostdlib"
        " -Wl,--gc-sections -Wl,--entry=", core, " ",
        core_obj, (rt_stubs_obj.empty() ? "" : " "),
        rt_stubs_obj,
        " -Wl,-T,", core_ldscript, " -o ", core_elf),
        absl::StrFormat("Step %d/%d: Core (%d,%d) Peano linking",
                        step++, total_compile_steps, col, row)));
  }

  // -----------------------------------------------------------------------
  // Step 3i: Patch lowered MLIR with elf_file attributes.
  // mlir-aie v1.2.1 requires elf_file on aie.core ops for CDO generation.
  // Strip core bodies (already extracted above) and add elf_file paths.
  // -----------------------------------------------------------------------
  TF_RETURN_IF_ERROR(PatchCoreElfAttributes(lowered_mlir));

  // -----------------------------------------------------------------------
  // Step 4: CDO generation.
  // Generate Configuration Data Objects from the lowered MLIR + core ELFs.
  // -----------------------------------------------------------------------
  TF_RETURN_IF_ERROR(RunCommand(absl::StrCat(
      aie_translate,
      " --aie-generate-cdo"
      " --work-dir-path=", workdir,
      " ", lowered_mlir),
      absl::StrFormat("Step %d/%d: CDO generation",
                      step++, total_compile_steps)));

  // -----------------------------------------------------------------------
  // Step 4b: Generate NPU instruction stream (DMA_TO_NPU pipeline).
  // -----------------------------------------------------------------------
  std::string npu_insts_mlir = workdir + "/npu_insts.mlir";
  TF_RETURN_IF_ERROR(RunCommand(absl::StrCat(
      aie_opt,
      " --aie-materialize-bd-chains"
      " --aie-substitute-shim-dma-allocations"
      " --aie-assign-runtime-sequence-bd-ids"
      " --aie-dma-tasks-to-npu"
      " --aie-dma-to-npu"
      " ", lowered_mlir, " -o ", npu_insts_mlir),
      absl::StrFormat("Step %d/%d: NPU DMA lowering",
                      step++, total_compile_steps)));

  std::string insts_txt = workdir + "/insts.txt";
  TF_RETURN_IF_ERROR(RunCommand(absl::StrCat(
      aie_translate,
      " --aie-npu-to-binary --aie-output-binary=false ",
      npu_insts_mlir, " -o ", insts_txt),
      absl::StrFormat("Step %d/%d: NPU instruction generation",
                      step++, total_compile_steps)));

  auto instr_words_or = ReadInstrFile(insts_txt);
  if (!instr_words_or.ok()) {
    return instr_words_or.status();
  }
  LOG(INFO) << "XDNA codegen: NPU instruction stream generated, "
            << instr_words_or->size() << " words.";

  // -----------------------------------------------------------------------
  // Step 5: bootgen — package CDOs into a PDI.
  // -----------------------------------------------------------------------
  std::string design_bif = workdir + "/design.bif";
  TF_RETURN_IF_ERROR(WriteFile(design_bif, GenerateDesignBif(workdir)));

  std::string design_pdi = workdir + "/design.pdi";
  TF_RETURN_IF_ERROR(RunCommand(absl::StrCat(
      bootgen,
      " -arch versal -image ", design_bif,
      " -o ", design_pdi, " -w"),
      absl::StrFormat("Step %d/%d: PDI packaging",
                      step++, total_compile_steps)));

  // -----------------------------------------------------------------------
  // Step 6: xclbinutil — assemble the xclbin.
  // -----------------------------------------------------------------------
  std::string kernel_name = "MLIR_AIE";
  int total_args = 3 + num_data_args;  // opcode + instr + ninstr + data args

  std::string mem_topology_json = workdir + "/mem_topology.json";
  std::string kernels_json = workdir + "/kernels.json";
  std::string aie_partition_json = workdir + "/aie_partition.json";

  TF_RETURN_IF_ERROR(WriteFile(mem_topology_json, GenerateMemTopologyJson()));
  TF_RETURN_IF_ERROR(
      WriteFile(kernels_json, GenerateKernelsJson(kernel_name, num_data_args)));
  TF_RETURN_IF_ERROR(WriteFile(
      aie_partition_json,
      GenerateAiePartitionJson(design_pdi, caps, num_cores)));

  std::string final_xclbin = workdir + "/final.xclbin";
  TF_RETURN_IF_ERROR(RunCommand(absl::StrCat(
      xclbinutil,
      " --add-replace-section MEM_TOPOLOGY:JSON:", mem_topology_json,
      " --add-kernel ", kernels_json,
      " --add-replace-section AIE_PARTITION:JSON:", aie_partition_json,
      " --force --quiet"
      " --output ", final_xclbin),
      absl::StrFormat("Step %d/%d: xclbin assembly",
                      step++, total_compile_steps)));

  // -----------------------------------------------------------------------
  // Read the final xclbin and clean up.
  // -----------------------------------------------------------------------
  auto xclbin_bytes_or = ReadBinaryFile(final_xclbin);
  if (!xclbin_bytes_or.ok()) {
    return xclbin_bytes_or.status();
  }

  if (!std::getenv("XDNA_KEEP_TEMP")) {
    std::filesystem::remove_all(workdir);
  } else {
    LOG(INFO) << "XDNA codegen: keeping temp directory: " << workdir;
  }

  LOG(INFO) << "XDNA codegen: xclbin generated, "
            << xclbin_bytes_or->size() << " bytes.";

  return XdnaCodegenResult{
      .xclbin_bytes = std::move(*xclbin_bytes_or),
      .kernel_name = kernel_name,
      .num_kernel_args = total_args,
      .instr_words = std::move(*instr_words_or),
  };
}

}  // namespace xla
