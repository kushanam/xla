# Description:
#   XLA client libraries.

load("@tsl//tsl:tsl.default.bzl", "filegroup")
load("@tsl//tsl/platform:rules_cc.bzl", "cc_library")
load("//xla:xla.bzl", "xla_cc_test")

package(
    # copybara:uncomment default_applicable_licenses = ["//tensorflow:license"],
    default_visibility = ["//visibility:public"],
    licenses = ["notice"],
)

package_group(
    name = "friends",
    includes = [
        "//xla:friends",
    ],
)

# Filegroup used to collect source files for dependency checking.
filegroup(
    name = "c_srcs",
    data = glob([
        "**/*.cc",
        "**/*.h",
    ]),
)

cc_library(
    name = "global_data",
    srcs = ["global_data.cc"],
    hdrs = ["global_data.h"],
    deps = [
        "//xla:service_interface",
        "//xla:types",
        "//xla:xla_data_proto_cc",
        "//xla:xla_proto_cc",
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_absl//absl/types:span",
        "@tsl//tsl/platform:logging",
        "@tsl//tsl/platform:status",
    ],
)

cc_library(
    name = "padding",
    srcs = ["padding.cc"],
    hdrs = ["padding.h"],
    deps = [
        "//xla:statusor",
        "//xla:types",
        "//xla:util",
        "@com_google_absl//absl/types:span",
        "@tsl//tsl/lib/math:math_util",
        "@tsl//tsl/platform:logging",
    ],
)

xla_cc_test(
    name = "padding_test",
    srcs = ["padding_test.cc"],
    deps = [
        ":padding",
        "@tsl//tsl/platform:test",
        "@tsl//tsl/platform:test_main",
    ],
)

cc_library(
    name = "client",
    srcs = ["client.cc"],
    hdrs = ["client.h"],
    deps = [
        ":global_data",
        ":xla_computation",
        "//xla:debug_options_flags",
        "//xla:execution_options_util",
        "//xla:literal",
        "//xla:service_interface",
        "//xla:status_macros",
        "//xla:statusor",
        "//xla:types",
        "//xla:xla_data_proto_cc",
        "//xla:xla_proto_cc",
        "//xla/service:hlo_proto_cc",
        "@com_google_absl//absl/memory",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/types:span",
        "@tsl//tsl/platform:errors",
        "@tsl//tsl/platform:logging",
        "@tsl//tsl/platform:protobuf",
    ],
)

cc_library(
    name = "executable_build_options",
    srcs = ["executable_build_options.cc"],
    hdrs = ["executable_build_options.h"],
    deps = [
        "//xla:debug_options_flags",
        "//xla:execution_options_util",
        "//xla:shape_util",
        "//xla:xla_proto_cc",
        "//xla/pjrt:compile_options_proto_cc",
        "//xla/service:compilation_environments",
        "//xla/service:computation_placer",
        "@com_google_absl//absl/algorithm:container",
        "@com_google_absl//absl/container:inlined_vector",
        "@com_google_absl//absl/strings:str_format",
        "@tsl//tsl/platform:env",
        "@tsl//tsl/platform:statusor",
    ],
)

cc_library(
    name = "local_client",
    srcs = ["local_client.cc"],
    hdrs = ["local_client.h"],
    deps = [
        ":client",
        ":executable_build_options",
        ":xla_computation",
        "//xla:executable_run_options",
        "//xla:shape_tree",
        "//xla:statusor",
        "//xla:xla_data_proto_cc",
        "//xla/service:backend",
        "//xla/service:compiler",
        "//xla/service:dump",
        "//xla/service:executable",
        "//xla/service:hlo_proto_cc",
        "//xla/service:local_service",
        "//xla/service:maybe_owning_device_memory",
        "//xla/service:shaped_buffer",
        "//xla/service:source_map_util",
        "//xla/service:stream_pool",
        "//xla/stream_executor",
        "//xla/stream_executor:device_memory_allocator",
        "//xla/stream_executor:stream_executor_pimpl",  # fixdeps: keep
        "@com_google_absl//absl/types:span",
    ],
)

cc_library(
    name = "compile_only_client",
    srcs = ["compile_only_client.cc"],
    hdrs = ["compile_only_client.h"],
    deps = [
        ":client",
        ":xla_computation",
        "//xla:status_macros",
        "//xla:statusor",
        "//xla:xla_data_proto_cc",
        "//xla/service:compile_only_service",
        "//xla/service:compiler",
        "//xla/stream_executor",
        "@llvm-project//llvm:Support",
        "@llvm-project//llvm:TargetParser",
    ],
)

# This target is used to instantiate the XLA service in-process and create
# a client for it.
cc_library(
    name = "client_library",
    srcs = ["client_library.cc"],
    hdrs = ["client_library.h"],
    deps = [
        ":compile_only_client",
        ":local_client",
        "//xla:status_macros",
        "//xla:statusor",
        "//xla:types",
        "//xla:util",
        "//xla/service:backend",
        "//xla/service:compile_only_service",
        "//xla/service:local_service",
        "//xla/service:platform_util",
        "//xla/stream_executor",
        "//xla/stream_executor:device_memory_allocator",
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_absl//absl/memory",
        "@tsl//tsl/platform:logging",
    ],
)

cc_library(
    name = "sharding_builder",
    srcs = ["sharding_builder.cc"],
    hdrs = ["sharding_builder.h"],
    deps = [
        "//xla:array",
        "//xla:shape_tree",
        "//xla:shape_util",
        "//xla:types",
        "//xla:util",
        "//xla:xla_data_proto_cc",
    ],
)

cc_library(
    name = "xla_computation",
    srcs = ["xla_computation.cc"],
    hdrs = ["xla_computation.h"],
    visibility = ["//visibility:public"],
    deps = [
        "//xla:shape_util",
        "//xla:status_macros",
        "//xla:util",
        "//xla:xla_data_proto_cc",
        "//xla/service:hlo_proto_cc",
        "@com_google_absl//absl/memory",
    ],
)

cc_library(
    name = "value_inference",
    srcs = ["value_inference.cc"],
    hdrs = ["value_inference.h"],
    visibility = ["//visibility:public"],
    deps = [
        ":xla_builder",
        "//xla:comparison_util",
        "//xla:literal",
        "//xla:literal_util",
        "//xla:shape_util",
        "//xla:status_macros",
        "//xla:statusor",
        "//xla:util",
        "//xla:xla_data_proto_cc",
        "//xla/hlo/evaluator:hlo_evaluator",
        "//xla/hlo/ir:hlo",
        "//xla/service:hlo_proto_cc",
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_absl//absl/strings:str_format",
        "@com_google_absl//absl/types:span",
        "@tsl//tsl/platform:errors",
        "@tsl//tsl/platform:statusor",
    ],
)

cc_library(
    name = "xla_builder",
    srcs = ["xla_builder.cc"],
    hdrs = ["xla_builder.h"],
    visibility = ["//visibility:public"],
    deps = [
        ":padding",
        ":sharding_builder",
        ":xla_computation",
        "//xla:comparison_util",
        "//xla:literal",
        "//xla:literal_util",
        "//xla:permutation_util",
        "//xla:shape_util",
        "//xla:sharding_op_util",
        "//xla:status",
        "//xla:status_macros",
        "//xla:statusor",
        "//xla:util",
        "//xla:window_util",
        "//xla:xla_data_proto_cc",
        "//xla/hlo/ir:hlo",
        "//xla/service:hlo_proto_cc",
        "//xla/service:shape_inference",
        "@com_google_absl//absl/algorithm:container",
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_absl//absl/container:flat_hash_set",
        "@com_google_absl//absl/functional:function_ref",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/types:span",
        "@tsl//tsl/lib/core:bitmap",
        "@tsl//tsl/platform:errors",
        "@tsl//tsl/platform:stacktrace",
    ],
)

xla_cc_test(
    name = "xla_builder_test",
    srcs = ["xla_builder_test.cc"],
    deps = [
        ":sharding_builder",
        ":value_inference",
        ":xla_builder",
        ":xla_computation",
        "//xla:debug_options_flags",
        "//xla:shape_util",
        "//xla:test_helpers",
        "//xla:util",
        "//xla:xla_data_proto_cc",
        "//xla/hlo/ir:hlo",
        "//xla/service:pattern_matcher",
        "//xla/service:pattern_matcher_gmock",
        "//xla/tests:xla_internal_test_main",
        "@com_google_googletest//:gtest",
        "@tsl//tsl/platform:statusor",
    ],
)
