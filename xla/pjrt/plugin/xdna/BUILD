load("@rules_cc//cc:cc_library.bzl", "cc_library")
load("//xla:xla.default.bzl", "xla_cc_binary", "xla_cc_test")

package(
    # copybara:uncomment default_applicable_licenses = ["//tensorflow:license"],
    default_visibility = ["//visibility:public"],
    licenses = ["notice"],
)

cc_library(
    name = "xdna_pjrt_device",
    srcs = ["xdna_pjrt_device.cc"],
    hdrs = ["xdna_pjrt_device.h"],
    deps = [
        "//xla/pjrt:pjrt_client",
        "//xla/pjrt:pjrt_common",
        "//xla/pjrt:pjrt_device_description",
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/strings:string_view",
        "@com_google_absl//absl/types:span",
    ],
)

cc_library(
    name = "xdna_pjrt_buffer",
    srcs = ["xdna_pjrt_buffer.cc"],
    hdrs = ["xdna_pjrt_buffer.h"],
    deps = [
        "//xla:literal",
        "//xla:shape_util",
        "//xla/pjrt:pjrt_client",
        "//xla/pjrt:pjrt_future",
        "@com_google_absl//absl/functional:any_invocable",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/types:span",
    ],
)

cc_library(
    name = "xdna_pjrt_executable",
    srcs = ["xdna_pjrt_executable.cc"],
    hdrs = ["xdna_pjrt_executable.h"],
    deps = [
        ":xdna_pjrt_buffer",
        "//xla:shape_util",
        "//xla:util",
        "//xla/hlo/ir:hlo",
        "//xla/pjrt:pjrt_client",
        "//xla/pjrt:pjrt_executable",
        "//xla/pjrt:pjrt_future",
        "//xla/pjrt:pjrt_layout",
        "//xla/service:computation_placer",
        "@com_google_absl//absl/log",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/strings:string_view",
        "@com_google_absl//absl/types:span",
        "@xrt//:xrt_coreutil",
    ],
)

cc_library(
    name = "xdna_aie_lowering",
    srcs = ["xdna_aie_lowering.cc"],
    hdrs = ["xdna_aie_lowering.h"],
    deps = [
        "@com_google_absl//absl/log",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/strings:str_format",
        "@llvm-project//llvm:Support",
        "@llvm-project//mlir:FuncDialect",
        "@llvm-project//mlir:IR",
        "@llvm-project//mlir:LinalgDialect",
        "@tsl//tsl/platform:errors",
    ],
)

cc_library(
    name = "xdna_codegen",
    srcs = ["xdna_codegen.cc"],
    hdrs = ["xdna_codegen.h"],
    deps = [
        "@com_google_absl//absl/log",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/strings:str_format",
        "@tsl//tsl/platform:errors",
    ],
)

cc_library(
    name = "xdna_hlo_passes",
    srcs = ["xdna_hlo_passes.cc"],
    hdrs = ["xdna_hlo_passes.h"],
    deps = [
        "//xla/hlo/ir:hlo",
        "//xla/hlo/pass:hlo_pass_pipeline",
        "//xla/hlo/transforms/expanders:cholesky_expander",
        "//xla/hlo/transforms/expanders:convolution_type_canonicalizer",
        "//xla/hlo/transforms/expanders:dynamic_index_splitter",
        "//xla/hlo/transforms/expanders:eigh_expander",
        "//xla/hlo/transforms/expanders:qr_expander",
        "//xla/service:batchnorm_expander",
        "//xla/service:layout_assignment",
        "//xla/service:topk_rewriter",
        "//xla/service:triangular_solve_expander",
        "@com_google_absl//absl/status:statusor",
        "@tsl//tsl/platform:errors",
    ],
)

cc_library(
    name = "xdna_compiler",
    srcs = ["xdna_compiler.cc"],
    hdrs = ["xdna_compiler.h"],
    deps = [
        ":xdna_aie_lowering",
        ":xdna_codegen",
        "//xla:shape_util",
        "//xla/hlo/ir:hlo",
        "//xla/hlo/translate/hlo_to_mhlo:hlo_to_mlir_hlo",
        "//xla/pjrt:mlir_to_hlo",
        "@com_google_absl//absl/log",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings",
        "@llvm-project//llvm:Support",
        "@llvm-project//mlir:IR",
        "@llvm-project//mlir:Pass",
        "@stablehlo//:linalg_passes",
        "@tsl//tsl/platform:errors",
    ],
)

cc_library(
    name = "xdna_pjrt_client",
    srcs = ["xdna_pjrt_client.cc"],
    hdrs = ["xdna_pjrt_client.h"],
    deps = [
        ":xdna_codegen",
        ":xdna_compiler",
        ":xdna_hlo_passes",
        ":xdna_pjrt_buffer",
        ":xdna_pjrt_device",
        ":xdna_pjrt_executable",
        "//xla:shape_util",
        "//xla:xla_proto_cc",
        "//xla/client:executable_build_options",
        "//xla/hlo/builder:xla_computation",
        "//xla/hlo/ir:hlo",
        "//xla/pjrt:mlir_to_hlo",
        "//xla/pjrt:pjrt_client",
        "//xla/pjrt:pjrt_compiler",
        "//xla/pjrt:utils",
        "//xla/service:hlo_module_config",
        "//xla/service:hlo_module_util",
        "//xla/tsl/platform:errors",
        "//xla/tsl/platform:statusor",
        "@com_google_absl//absl/functional:any_invocable",
        "@com_google_absl//absl/log",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/strings:string_view",
        "@com_google_absl//absl/types:span",
        "@llvm-project//mlir:IR",
        "@tsl//tsl/platform:fingerprint",
        "@xrt//:xrt_coreutil",
    ],
)

cc_library(
    name = "xdna_c_pjrt_internal",
    srcs = ["xdna_c_pjrt_internal.cc"],
    hdrs = ["xdna_c_pjrt_internal.h"],
    deps = [
        ":xdna_pjrt_client",
        "//xla/pjrt:pjrt_client",
        "//xla/pjrt/c:pjrt_c_api_hdrs",
        "//xla/pjrt/c:pjrt_c_api_layouts_extension_hdrs",
        "//xla/pjrt/c:pjrt_c_api_wrapper_impl",
        "@com_google_absl//absl/status",
    ],
    alwayslink = 1,
)

cc_library(
    name = "xdna_c_pjrt",
    srcs = ["xdna_c_pjrt.cc"],
    hdrs = ["xdna_c_pjrt.h"],
    deps = [
        ":xdna_c_pjrt_internal",
        "//xla/pjrt/c:pjrt_c_api_hdrs",
    ],
)

xla_cc_test(
    name = "xdna_pjrt_client_test",
    srcs = ["xdna_pjrt_client_test.cc"],
    deps = [
        ":xdna_c_pjrt",
        ":xdna_pjrt_client",
        "//xla:literal_util",
        "//xla/pjrt:pjrt_client",
        "//xla/pjrt:pjrt_executable",
        "//xla/pjrt/c:pjrt_c_api_hdrs",
        "//xla/tests:xla_internal_test_main",
        "@com_google_googletest//:gtest",
    ],
)

xla_cc_test(
    name = "xdna_compiler_test",
    srcs = ["xdna_compiler_test.cc"],
    deps = [
        ":xdna_compiler",
        ":xdna_hlo_passes",
        ":xdna_pjrt_client",
        "//xla:literal_util",
        "//xla:shape_util",
        "//xla/hlo/builder:xla_builder",
        "//xla/hlo/builder:xla_computation",
        "//xla/hlo/ir:hlo",
        "//xla/pjrt:pjrt_client",
        "//xla/pjrt:pjrt_future",
        "//xla/tests:xla_internal_test_main",
        "@com_google_absl//absl/status:statusor",
        "@com_google_googletest//:gtest",
    ],
)

xla_cc_binary(
    name = "pjrt_c_api_xdna_plugin.so",
    linkopts = [
        "-Wl,--version-script,$(location :exported_symbols.lds)",
    ],
    linkshared = True,
    deps = [
        ":xdna_c_pjrt",
        ":exported_symbols.lds",
    ],
)
