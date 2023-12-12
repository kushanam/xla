load("//xla:xla.bzl", "xla_cc_test")
load(
    "@tsl//tsl/platform:build_config_root.bzl",
    "tf_cuda_tests_tags",
)
load("@tsl//tsl/platform:rules_cc.bzl", "cc_library")
load(
    "@local_tsl//tsl/platform/default:cuda_build_defs.bzl",
    "if_cuda_is_configured",
)
package(
    default_visibility = ["//xla:internal"],
    licenses = ["notice"],
)

package_group(
    name = "friends",
    includes = [
        "//xla:friends",
        "//xla:internal",
    ],
    packages = [
        "//third_party/australis/...",
        "//third_party/py/jax/...",
    ],
)

cc_library(
    name = "testgpu_c_api_hdrs",
    hdrs = ["testgpu_c_api.h"],
    visibility = ["//visibility:public"],
    deps = [
        "//xla/hlo/ir:hlo",
        "//xla/hlo/utils:hlo_query",
        "//xla/service:collective_ops_utils",
        "//xla/service:hlo_pass",
    ],
)

cc_library(
    name = "testgpu_api",
    srcs = ["testgpu_api.cc"],
    hdrs = ["testgpu_api.h"],
    visibility = ["//visibility:public"],
    deps = [
        ":testgpu_c_api_hdrs",
        "//xla:status",
        "//xla:statusor",
        "@com_google_absl//absl/log",
        "@com_google_absl//absl/status",
        "@tsl//tsl/platform:env",
        "@tsl//tsl/platform:errors",
    ],
)

cc_library(
    name = "testgpu_library",
    srcs = ["testgpu_library.cc"],
    hdrs = ["testgpu_library.h"],
    deps = [
        ":testgpu_api",
        ":testgpu_c_api_hdrs",
        "//xla:status",
        "//xla:statusor",
    ],
)

cc_library(
    name = "testgpu_pass",
    srcs = ["testgpu_pass.cc"],
    hdrs = ["testgpu_pass.h"],
    visibility = ["//visibility:public"],
    deps = [
        ":testgpu_c_api_hdrs",
        ":testgpu_library",
        "//xla:status",
        "//xla:status_macros",
        "//xla:statusor",
        "//xla:types",
        "//xla:util",
        "//xla/hlo/ir:hlo",
        "//xla/service:hlo_pass",
        "@com_google_absl//absl/container:flat_hash_map",
        "@tsl//tsl/platform:errors",
        "@tsl//tsl/platform:logging",
    ],
)

xla_cc_test(
    name = "testgpu_pass_test",
    srcs = ["testgpu_pass_test.cc"],
    deps = [
        ":testgpu_c_api_hdrs",
        ":testgpu_pass",
        "//xla:statusor",
        "//xla/hlo/ir:hlo",
        "//xla/hlo/utils:hlo_matchers",
        "//xla/service:hlo_pass_pipeline",
        "//xla/service:pattern_matcher_gmock",
        "//xla/tests:hlo_test_base",
        "//xla/tests:xla_internal_test_main",
        "@com_google_absl//absl/algorithm:container",
        "@com_google_absl//absl/strings",
    ],
)

