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
                                     const TargetCaps& caps) {
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
})", caps.partition_column_width, caps.partition_start_column, pdi_path);
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
      file=%s/aie_cdo_elfs.bin
      file=%s/aie_cdo_init.bin
      file=%s/aie_cdo_enable.bin
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
    const std::string& aie_mlir, int num_data_args, const TargetCaps& caps) {
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
      "Step 1/13: AIE lowering passes"));

  // -----------------------------------------------------------------------
  // Step 3: Per-core code compilation with Peano.
  // Extract core(0,2) code → lower to LLVM → compile to ELF.
  // -----------------------------------------------------------------------

  // 3a: Extract core code with AIE-specific lowering.
  std::string core_mlir = workdir + "/core_0_2.mlir";
  TF_RETURN_IF_ERROR(RunCommand(absl::StrCat(
      aie_opt,
      " --aie-localize-locks"
      " --aie-normalize-address-spaces"
      " \"--aie-standard-lowering=tilecol=0 tilerow=2\""
      " --aiex-standard-lowering"
      " ", lowered_mlir, " -o ", core_mlir),
      "Step 2/13: Core code extraction"));

  // 3b: Lower to LLVM dialect (matches aiecc.py LOWER_TO_LLVM_PIPELINE).
  std::string core_opt_mlir = workdir + "/core_0_2.opt.mlir";
  TF_RETURN_IF_ERROR(RunCommand(absl::StrCat(
      aie_opt,
      " --canonicalize --cse"
      " --convert-vector-to-llvm --expand-strided-metadata"
      " --lower-affine --convert-math-to-llvm --convert-index-to-llvm"
      " --arith-expand --convert-arith-to-llvm --finalize-memref-to-llvm"
      " \"--convert-func-to-llvm=use-bare-ptr-memref-call-conv\""
      " --convert-cf-to-llvm --canonicalize --cse"
      " ", core_mlir, " -o ", core_opt_mlir),
      "Step 3/13: LLVM dialect lowering"));

  // 3c: Translate to LLVM IR.
  std::string core_ll = workdir + "/core_0_2.ll";
  TF_RETURN_IF_ERROR(RunCommand(absl::StrCat(
      aie_translate, " --mlir-to-llvmir ", core_opt_mlir, " -o ", core_ll),
      "Step 4/13: LLVM IR translation"));

  // 3d: Fix intrinsic names (llvm.aie2.* → llvm.<isa_target>.* for target).
  // mlir-aie generates llvm.aie2.* intrinsics; Peano expects llvm.<target>.*.
  {
    std::ifstream ll_in(core_ll);
    if (!ll_in.is_open()) {
      return absl::InternalError(
          absl::StrCat("Cannot read LLVM IR: ", core_ll));
    }
    std::string ll_content((std::istreambuf_iterator<char>(ll_in)),
                           std::istreambuf_iterator<char>());
    ll_in.close();

    std::string from = "llvm.aie2.";
    std::string to = absl::StrCat("llvm.", caps.isa_target, ".");
    size_t pos = 0;
    while ((pos = ll_content.find(from, pos)) != std::string::npos) {
      ll_content.replace(pos, from.size(), to);
      pos += to.size();
    }
    TF_RETURN_IF_ERROR(WriteFile(core_ll, ll_content));
  }

  // 3e: Peano opt → llc → clang link.
  std::string core_opt_ll = workdir + "/core_0_2.opt.ll";
  // Disable SLP/loop vectorization: Peano's O2 creates small vectors
  // (e.g. <2 x bf16>) that are illegal on AIE2p — legal vector widths
  // are 32/64 elements. Scalar code is fine: Peano's legalizer
  // widens scalars to legal vector ops where needed.
  TF_RETURN_IF_ERROR(RunCommand(absl::StrCat(
      peano_opt,
      " '--passes=default<O2>,strip'"
      " -vectorize-slp=false"
      " -vectorize-loops=false"
      " -S ",
      core_ll, " -o ", core_opt_ll),
      "Step 5/13: Peano optimization"));

  std::string core_obj = workdir + "/core_0_2.o";
  TF_RETURN_IF_ERROR(RunCommand(absl::StrCat(
      peano_llc, " ", core_opt_ll,
      " -O2 --march=", caps.isa_target,
      " --function-sections --filetype=obj"
      " -o ", core_obj),
      "Step 6/13: Peano codegen"));

  std::string core_ldscript = workdir + "/core_0_2.ld.script";
  TF_RETURN_IF_ERROR(RunCommand(absl::StrCat(
      aie_translate,
      " --aie-generate-ldscript --tilecol=0 --tilerow=2 ",
      lowered_mlir, " -o ", core_ldscript),
      "Step 7/13: Linker script generation"));

  std::string core_elf = workdir + "/core_0_2.elf";
  TF_RETURN_IF_ERROR(RunCommand(absl::StrCat(
      peano_clang,
      " -O2 --target=", caps.isa_target, "-none-elf -nostdlib"
      " -Wl,--gc-sections -Wl,--entry=core_0_2 ",
      core_obj, " -Wl,-T,", core_ldscript, " -o ", core_elf),
      "Step 8/13: Peano linking"));

  // -----------------------------------------------------------------------
  // Step 4: CDO generation.
  // Generate Configuration Data Objects from the lowered MLIR + core ELFs.
  // -----------------------------------------------------------------------
  TF_RETURN_IF_ERROR(RunCommand(absl::StrCat(
      aie_translate,
      " --aie-generate-cdo"
      " --work-dir-path=", workdir,
      " ", lowered_mlir),
      "Step 9/13: CDO generation"));

  // -----------------------------------------------------------------------
  // Step 4b: Generate NPU instruction stream (DMA_TO_NPU pipeline).
  // Matches aiecc.py's DMA_TO_NPU pipeline:
  //   - materialize-bd-chains: materializes BD chain descriptors
  //   - substitute-shim-dma-allocations: resolves ObjectFIFO → DMA channel
  //   - assign-runtime-sequence-bd-ids: assigns BD IDs for runtime sequence
  //   - dma-tasks-to-npu: converts DMA task ops to NPU operations
  //   - dma-to-npu: final lowering to NPU write/sync operations
  // Then aie-npu-instgen translates to a binary instruction stream.
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
      "Step 10/13: NPU DMA lowering"));

  std::string insts_txt = workdir + "/insts.txt";
  TF_RETURN_IF_ERROR(RunCommand(absl::StrCat(
      aie_translate,
      " --aie-npu-instgen ",
      npu_insts_mlir, " -o ", insts_txt),
      "Step 11/13: NPU instruction generation"));

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
      "Step 12/13: PDI packaging"));

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
      GenerateAiePartitionJson(design_pdi, caps)));

  std::string final_xclbin = workdir + "/final.xclbin";
  TF_RETURN_IF_ERROR(RunCommand(absl::StrCat(
      xclbinutil,
      " --add-replace-section MEM_TOPOLOGY:JSON:", mem_topology_json,
      " --add-kernel ", kernels_json,
      " --add-replace-section AIE_PARTITION:JSON:", aie_partition_json,
      " --force --quiet"
      " --output ", final_xclbin),
      "Step 13/13: xclbin assembly"));

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
