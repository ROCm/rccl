load("@fbcode//tools/build/buck:rocm_flags.bzl", "get_rocm_arch_args", "get_rocm_archs")
load("@fbcode_macros//build_defs:native_rules.bzl", "buck_genrule")
load("@fbcode_macros//build_defs/lib:rocm.bzl", "rocm")
load("@fbsource//tools/build_defs:selects.bzl", "selects")
load(":defs.bzl", "hip_toolchain_override")
load(":rccl_build_config.bzl", "rccl_third_party_cxx_library", "rccl_third_party_cxx_library_internal")

oncall("rccl")

# Note: we cannot name this library as rccl, because it's going to
# build librccl.so which will get confused with the librccl.so from
# tp2 (which we deploy on the host in /usr/local/fbcode). Depending
# on the rpath, the tp2 librccl.so may get used which makes us use
# unexpected version and cause performance regressions
rccl_third_party_cxx_library(
    name = "rcclx-dev",
    arname = "librcclx-dev.a",
)

rccl_third_party_cxx_library_internal(
    name = "rcclx-dev-internal",
    arname = "librcclx-dev-internal.a",
)

# Shared library version of rcclx for runtime dynamic loading
# This creates librccl.so.1 that contains rcclx functionality
buck_genrule(
    name = "rcclx-shared",
    out = "librccl.so.1",
    bash = "$(location fbsource//third-party/rocm:rocm_path)/llvm/bin/clang -shared -o $OUT -Wl,-soname,librccl.so.1 $(location :rcclx-dev)",
    visibility = ["PUBLIC"],
)

# rccl makes use of device code that needs to reference symbols from other
# translation units. The way to support this is to pass `-fgpu-rdc` to emit
# relocatable device code.
#
# However, if you want to emit a static library with _all_ the device code in it
# and link this library into a dependent binary _without_ using -fgpu-rdc, you
# need to create an archive using `--emit-static-lib` and some other flags.
# Only clang understands these flags, so we override the cxx_toolchain passed to
# the underlying rccl cxx_library rule so we can customize the static library
# archive creation action.
hip_toolchain_override(
    name = "cxx-platform010-amdclang-toolchain",
    # Make sure .cpp files compiled with this toolchain are done with the hip
    # processes.
    additional_cxx_compiler_flags = [
        "-x",
        "hip",
    ] + get_rocm_arch_args(),
    # Don't produce thin archives, we can't objcopy them afterwards.
    archive_contents = "normal",
    # Archive with amdclang instead of llvm-ar.
    archiver = selects.apply(
        rocm.get_platform_version_select(),
        lambda version: "fbsource//third-party/rocm/{}:amdclang".format(version) if version else None,
    ),
    archiver_flags = [
        "--hip-link",
        "-fgpu-rdc",
        "--emit-static-lib",
    ] + selects.apply(
        get_rocm_archs(),
        lambda p: ["--offload-arch={}".format(arch) for arch in p],
    ),
    archiver_type = "amdclang",
    base = "toolchains//:cxx",
    use_archiver_flags = True,
    visibility = ["PUBLIC"],
)
