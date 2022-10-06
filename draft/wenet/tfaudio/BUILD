load("@rules_cc//cc:defs.bzl", "cc_binary")


cc_binary(
    name = "tfaudio.so",
    srcs = [
        "cc/ops/resample.h",
        "cc/ops/resample.cc",
        "cc/ops/kaldi_fbank_kernels.cc",
        "cc/ops/kaldi_fbank_kernels.h",
        "cc/ops/speed_kernels.cc",
    ],
    linkshared = 1,
    deps = [
        "@local_config_tf//:tf_header_lib",
        "@local_config_tf//:libtensorflow_framework",
    ],
     copts = select({
        "//conditions:default": ["-pthread", "-std=c++17", "-D_GLIBCXX_USE_CXX11_ABI=0"],
    }),
   
)

