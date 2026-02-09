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
        "//xla/pjrt:pjrt_client",
        "//xla/pjrt:pjrt_executable",
        "//xla/pjrt:pjrt_future",
        "//xla/service:computation_placer",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings:string_view",
        "@com_google_absl//absl/types:span",
    ],
)

cc_library(
    name = "xdna_pjrt_client",
    srcs = ["xdna_pjrt_client.cc"],
    hdrs = ["xdna_pjrt_client.h"],
    deps = [
        ":xdna_pjrt_buffer",
        ":xdna_pjrt_device",
        ":xdna_pjrt_executable",
        "//xla:shape_util",
        "//xla/pjrt:pjrt_client",
        "//xla/pjrt:pjrt_compiler",
        "@com_google_absl//absl/functional:any_invocable",
        "@com_google_absl//absl/log",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/strings:string_view",
        "@com_google_absl//absl/types:span",
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
        "//xla/pjrt/c:pjrt_c_api_hdrs",
        "//xla/tests:xla_internal_test_main",
        "@com_google_googletest//:gtest",
    ],
)

xla_cc_binary(
    name = "pjrt_c_api_xdna_plugin.so",
    linkshared = True,
    deps = [
        ":xdna_c_pjrt",
    ],
)
