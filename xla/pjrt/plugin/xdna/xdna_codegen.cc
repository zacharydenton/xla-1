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

#include <cstdint>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <string>
#include <vector>

#include "absl/log/log.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_cat.h"
#include "xla/tsl/platform/errors.h"

namespace xla {
namespace {

// Tool paths. These are compile-time defaults; override via environment
// variables XDNA_AIE_OPT, XDNA_AIE_TRANSLATE, XDNA_PEANO_CLANG,
// XDNA_AIEBU_ASM if installed elsewhere.
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

std::string GetAiebuAsmPath() {
  const char* env = std::getenv("XDNA_AIEBU_ASM");
  return env ? env : "/opt/xilinx/xrt/bin/aiebu-asm";
}

// Runs a shell command and returns an error if it fails.
absl::Status RunCommand(const std::string& cmd) {
  LOG(INFO) << "XDNA codegen: running: " << cmd;
  int ret = std::system(cmd.c_str());
  if (ret != 0) {
    return absl::InternalError(
        absl::StrCat("Command failed with exit code ", ret, ": ", cmd));
  }
  return absl::OkStatus();
}

// Reads a binary file into a byte vector.
absl::StatusOr<std::vector<uint8_t>> ReadBinaryFile(const std::string& path) {
  std::ifstream file(path, std::ios::binary | std::ios::ate);
  if (!file.is_open()) {
    return absl::NotFoundError(
        absl::StrCat("Cannot open file: ", path));
  }
  auto size = file.tellg();
  file.seekg(0, std::ios::beg);
  std::vector<uint8_t> data(size);
  file.read(reinterpret_cast<char*>(data.data()), size);
  return data;
}

// Writes a string to a file.
absl::Status WriteFile(const std::string& path, const std::string& content) {
  std::ofstream file(path);
  if (!file.is_open()) {
    return absl::InternalError(
        absl::StrCat("Cannot write file: ", path));
  }
  file << content;
  return absl::OkStatus();
}

}  // namespace

absl::StatusOr<std::vector<uint8_t>> GenerateElfFromAie(
    const std::string& aie_mlir) {
  // Create a temporary working directory.
  auto tmpdir = std::filesystem::temp_directory_path() / "xdna_codegen_XXXXXX";
  std::string tmpdir_str = tmpdir.string();
  char* tmpdir_cstr = tmpdir_str.data();
  if (!mkdtemp(tmpdir_cstr)) {
    return absl::InternalError("Failed to create temp directory.");
  }
  std::string workdir(tmpdir_cstr);
  LOG(INFO) << "XDNA codegen: working directory: " << workdir;

  // Check that required tools exist.
  std::string aie_opt = GetAieOptPath();
  std::string aie_translate = GetAieTranslatePath();
  std::string peano_clang = GetPeanoClangPath();

  if (!std::filesystem::exists(aie_opt)) {
    return absl::NotFoundError(absl::StrCat(
        "aie-opt not found at ", aie_opt,
        ". Set XDNA_AIE_OPT or install mlir-aie to /opt/mlir-aie."));
  }
  if (!std::filesystem::exists(aie_translate)) {
    return absl::NotFoundError(absl::StrCat(
        "aie-translate not found at ", aie_translate,
        ". Set XDNA_AIE_TRANSLATE or install mlir-aie to /opt/mlir-aie."));
  }
  if (!std::filesystem::exists(peano_clang)) {
    return absl::NotFoundError(absl::StrCat(
        "Peano clang not found at ", peano_clang,
        ". Set XDNA_PEANO_CLANG or install Peano to /opt/peano."));
  }

  // Step 1: Write AIE MLIR to file.
  std::string input_mlir = workdir + "/input.mlir";
  auto status = WriteFile(input_mlir, aie_mlir);
  if (!status.ok()) return status;

  // Step 2: Run aie-opt passes.
  // These passes lower ObjectFIFOs, route flows, assign resources.
  std::string lowered_mlir = workdir + "/lowered.mlir";
  std::string aie_opt_cmd = absl::StrCat(
      aie_opt,
      " --lower-affine"
      " --aie-canonicalize-device"
      " --aie-assign-lock-ids"
      " --aie-objectFifo-stateful-transform"
      " --aie-assign-bd-ids"
      " --aie-lower-cascade-flows"
      " --aie-create-pathfinder-flows"
      " --aie-assign-buffer-addresses"
      " --convert-scf-to-cf"
      " ", input_mlir, " -o ", lowered_mlir);
  TF_RETURN_IF_ERROR(RunCommand(aie_opt_cmd));

  // Step 3: Convert DMA tasks to NPU instructions and generate binary.
  // This matches aiecc.py's DMA_TO_NPU pipeline.
  std::string npu_ready_mlir = workdir + "/npu_ready.mlir";
  std::string dma_to_npu_cmd = absl::StrCat(
      aie_opt,
      " --aie-dma-tasks-to-npu"
      " --aie-assign-runtime-sequence-bd-ids"
      " ", lowered_mlir, " -o ", npu_ready_mlir);
  TF_RETURN_IF_ERROR(RunCommand(dma_to_npu_cmd));

  std::string npu_insts = workdir + "/npu_insts.bin";
  std::string npu_cmd = absl::StrCat(
      aie_translate,
      " --aie-npu-instgen"
      " --aie-output-binary"
      " ", npu_ready_mlir, " -o ", npu_insts);
  TF_RETURN_IF_ERROR(RunCommand(npu_cmd));

  // Step 4a: Extract core(0,2) code with AIE-specific lowering.
  // This converts aie.core regions to standalone functions and lowers
  // AIE/AIEX ops to standard LLVM intrinsic calls.
  std::string core_mlir = workdir + "/core_0_2.mlir";
  std::string core_extract_cmd = absl::StrCat(
      aie_opt,
      " --aie-localize-locks"
      " --aie-normalize-address-spaces"
      " \"--aie-standard-lowering=tilecol=0 tilerow=2\""
      " --aiex-standard-lowering"
      " ", lowered_mlir, " -o ", core_mlir);
  TF_RETURN_IF_ERROR(RunCommand(core_extract_cmd));

  // Step 4b: Lower to LLVM dialect.
  // Matches aiecc.py's LOWER_TO_LLVM_PIPELINE.
  std::string core_opt_mlir = workdir + "/core_0_2.opt.mlir";
  std::string core_llvm_cmd = absl::StrCat(
      aie_opt,
      " --canonicalize"
      " --cse"
      " --convert-vector-to-llvm"
      " --expand-strided-metadata"
      " --lower-affine"
      " --convert-math-to-llvm"
      " --convert-index-to-llvm"
      " --arith-expand"
      " --convert-arith-to-llvm"
      " --finalize-memref-to-llvm"
      " \"--convert-func-to-llvm=use-bare-ptr-memref-call-conv\""
      " --convert-cf-to-llvm"
      " --canonicalize"
      " --cse"
      " ", core_mlir, " -o ", core_opt_mlir);
  TF_RETURN_IF_ERROR(RunCommand(core_llvm_cmd));

  // Step 4c: Translate to LLVM IR.
  std::string core_ll = workdir + "/core_0_2.ll";
  std::string translate_cmd = absl::StrCat(
      aie_translate,
      " --mlir-to-llvmir"
      " ", core_opt_mlir, " -o ", core_ll);
  TF_RETURN_IF_ERROR(RunCommand(translate_cmd));

  // Step 4d: Fix intrinsic names for aie2p target.
  // mlir-aie generates llvm.aie2.* intrinsics but Peano's aie2p target
  // expects llvm.aie2p.* intrinsics.
  {
    std::ifstream ll_in(core_ll);
    if (!ll_in.is_open()) {
      return absl::InternalError(
          absl::StrCat("Cannot read LLVM IR: ", core_ll));
    }
    std::string ll_content((std::istreambuf_iterator<char>(ll_in)),
                           std::istreambuf_iterator<char>());
    ll_in.close();

    // Replace all llvm.aie2. with llvm.aie2p. — the dot after "aie2" means
    // existing "llvm.aie2p." won't be matched.
    std::string from = "llvm.aie2.";
    std::string to = "llvm.aie2p.";
    size_t pos = 0;
    while ((pos = ll_content.find(from, pos)) != std::string::npos) {
      ll_content.replace(pos, from.size(), to);
      pos += to.size();
    }

    TF_RETURN_IF_ERROR(WriteFile(core_ll, ll_content));
  }

  // Step 4e: Optimize LLVM IR with Peano opt.
  std::string peano_opt = GetPeanoOptPath();
  std::string peano_llc = GetPeanoLlcPath();
  std::string core_opt_ll = workdir + "/core_0_2.opt.ll";
  std::string opt_cmd = absl::StrCat(
      peano_opt,
      " '--passes=default<O2>,strip'"
      " -S ", core_ll, " -o ", core_opt_ll);
  TF_RETURN_IF_ERROR(RunCommand(opt_cmd));

  // Step 4e: Compile to object file with Peano llc.
  std::string core_obj = workdir + "/core_0_2.o";
  std::string llc_cmd = absl::StrCat(
      peano_llc,
      " ", core_opt_ll,
      " -O2"
      " --march=aie2p"
      " --function-sections"
      " --filetype=obj"
      " -o ", core_obj);
  TF_RETURN_IF_ERROR(RunCommand(llc_cmd));

  // Step 4f: Generate linker script for this core.
  std::string core_ldscript = workdir + "/core_0_2.ld.script";
  std::string ldscript_cmd = absl::StrCat(
      aie_translate,
      " --aie-generate-ldscript"
      " --tilecol=0 --tilerow=2"
      " ", lowered_mlir, " -o ", core_ldscript);
  TF_RETURN_IF_ERROR(RunCommand(ldscript_cmd));

  // Step 4g: Link to ELF with linker script.
  std::string core_elf = workdir + "/core_0_2.elf";
  std::string link_cmd = absl::StrCat(
      peano_clang,
      " -O2 --target=aie2p-none-elf"
      " -nostdlib"
      " -Wl,--gc-sections"
      " ", core_obj,
      " -Wl,-T,", core_ldscript,
      " -o ", core_elf);
  TF_RETURN_IF_ERROR(RunCommand(link_cmd));

  // Step 5: Generate transaction binary.
  // The transaction flow converts AIE device config (DMAs, locks, routes)
  // + core ELFs into a single transaction binary. This replaces CDO.
  std::string txn_mlir = workdir + "/txn.mlir";
  std::string txn_pass = absl::StrCat(
      "\"--pass-pipeline=builtin.module(aie.device(convert-aie-to-transaction"
      "{elf-dir=", workdir, "}))\"");
  std::string txn_cmd = absl::StrCat(
      aie_opt, " ", txn_pass, " ", lowered_mlir, " -o ", txn_mlir);
  TF_RETURN_IF_ERROR(RunCommand(txn_cmd));

  // Serialize transaction to binary.
  std::string txn_bin = workdir + "/txn.bin";
  std::string txn_serialize_cmd = absl::StrCat(
      aie_translate,
      " --aie-npu-instgen"
      " --aie-output-binary"
      " ", txn_mlir, " -o ", txn_bin);
  TF_RETURN_IF_ERROR(RunCommand(txn_serialize_cmd));

  // Step 6: Package into final ELF with aiebu-asm.
  std::string final_elf = workdir + "/final.elf";
  std::string aiebu_asm = GetAiebuAsmPath();

  if (std::filesystem::exists(aiebu_asm)) {
    std::string aiebu_cmd = absl::StrCat(
        aiebu_asm,
        " -t aie2txn"
        " -c ", txn_bin,
        " -o ", final_elf);
    TF_RETURN_IF_ERROR(RunCommand(aiebu_cmd));
  } else {
    LOG(WARNING) << "aiebu-asm not found at " << aiebu_asm
                 << ". Returning core ELF without packaging.";
    final_elf = core_elf;
  }

  // Read the final ELF.
  auto elf_result = ReadBinaryFile(final_elf);
  if (!elf_result.ok()) return elf_result.status();
  std::vector<uint8_t> elf_bytes = std::move(*elf_result);

  // Cleanup temp directory (keep for debugging if XDNA_KEEP_TEMP is set).
  if (!std::getenv("XDNA_KEEP_TEMP")) {
    std::filesystem::remove_all(workdir);
  } else {
    LOG(INFO) << "XDNA codegen: keeping temp directory: " << workdir;
  }

  return elf_bytes;
}

}  // namespace xla
