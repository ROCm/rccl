load("@fbcode//comms/rcclx:utils.bzl", "generate_collectives", "hipify", "inject_faults")
load(
    "@fbcode//comms/rcclx/develop:def_build.bzl",
    "get_mscclpp_headers_develop",
    "get_mscclpp_nccl_headers_develop",
    "get_mscclpp_srcs_develop",
    "get_rccl_exported_headers_develop",
    "get_rccl_fbcode_exported_deps_develop",
    "get_rccl_generate_collectives_develop",
    "get_rccl_git_version_rule_develop",
    "get_rccl_headers_develop",
    "get_rccl_hipify_headers_develop",
    "get_rccl_hipify_srcs_develop",
)
load("@fbcode//tools/build/buck:rccl_deps.bzl", "use_rccl", "valid_rccl_versions")
load("@fbcode//tools/build/buck:rocm_flags.bzl", "get_rocm_arch_args")
load("@fbcode_macros//build_defs:cpp_library.bzl", "cpp_library")
load("@fbcode_macros//build_defs:native_rules.bzl", "buck_genrule")
load("@fbsource//tools/build_defs:buckconfig.bzl", "read_bool")
load("@fbsource//tools/build_defs:fb_native_wrapper.bzl", "fb_native")
load("@fbsource//tools/build_defs:glob_defs.bzl", "subdir_glob")
load("@fbsource//tools/build_defs:selects.bzl", "selects")

default_dev_version = "develop"

def get_rccl_dev_ver():
    if use_rccl in valid_rccl_versions:
        return use_rccl
    else:
        return default_dev_version

rccl_ver = get_rccl_dev_ver()

# Map version identifiers to actual source directories
# rocm-6.4 and rocm-7.0 use the same source as develop
def get_rccl_src_dir():
    version_to_dir = {
        "develop": "develop",
        "rocm-6.4": "develop",
        "rocm-7.0": "develop",
    }
    return version_to_dir.get(rccl_ver, "develop")

rccl_src_dir = get_rccl_src_dir()

COMMON_CONTACTS = ["oncall+hpc_comms_lib@xmail.facebook.com"]

COMMON_COMPILER_FLAGS = [
    "-fPIC",
    "-fvisibility=hidden",
    "-Wno-format-nonliteral",
    "-Wno-logical-op-parentheses",
    "-Wno-sometimes-uninitialized",
    "-Wno-missing-braces",
    "-Wno-reorder-ctor",
    "-Wno-unused-const-variable",
    "-Wno-unused-command-line-argument",
    "-Wno-shift-sign-overflow",
    "-Wno-unused-label",
    "-Wno-bool-operation",
    "-Wno-vla-cxx-extension",
    "-Wno-macro-redefined",
    "-Wno-unused-but-set-variable",
    "-Wno-format",
    "-Wno-implicit-fallthrough",
    "-Wno-unused-result",
    "-Wno-unused-variable",
    "-Wno-deprecated-copy-with-user-provided-copy",
    "-Wno-nontrivial-memcall",
    "-Wno-error=null-conversion",
    "-Wno-unused-result",
    "-Wno-cuda-compat",
]

COMMON_HIP_FLAGS = get_rocm_arch_args()
RDC_HIP_FLAGS = COMMON_HIP_FLAGS + ["-fgpu-rdc", "-mllvm", "--amdgpu-kernarg-preload-count=16"]
RDC_HIP_LINK_FLAGS = [
    # force static link at hip_link rule
    "--emit-static-lib",
    "-fgpu-rdc",
    "--hip-link",
    "-mllvm",
    "--amdgpu-kernarg-preload-count=16",
]

def get_npkit_compiler_flags():
    return select({
        "DEFAULT": [],
        "fbsource//third-party/rccl/constraints:enable-npkit": [
            # enable NPKIT
            "-DENABLE_NPKIT",
            "-DENABLE_NPKIT_EVENT_TIME_SYNC_GPU",
            "-DENABLE_NPKIT_EVENT_TIME_SYNC_CPU",
            "-DENABLE_NPKIT_EVENT_ALL_REDUCE_RING_ENTRY",
            "-DENABLE_NPKIT_EVENT_ALL_REDUCE_RING_EXIT",
            "-DENABLE_NPKIT_EVENT_ALL_REDUCE_TREE_UPDOWN_ENTRY",
            "-DENABLE_NPKIT_EVENT_ALL_REDUCE_TREE_UPDOWN_EXIT",
            "-DENABLE_NPKIT_EVENT_ALL_REDUCE_TREE_SPLIT_ENTRY",
            "-DENABLE_NPKIT_EVENT_ALL_REDUCE_TREE_SPLIT_EXIT",
            "-DENABLE_NPKIT_EVENT_COPY_SEND_ENTRY",
            "-DENABLE_NPKIT_EVENT_COPY_SEND_EXIT",
            "-DENABLE_NPKIT_EVENT_DIRECT_COPY_SEND_ENTRY",
            "-DENABLE_NPKIT_EVENT_DIRECT_COPY_SEND_EXIT",
            "-DENABLE_NPKIT_EVENT_DIRECT_RECV_ENTRY",
            "-DENABLE_NPKIT_EVENT_DIRECT_RECV_EXIT",
            "-DENABLE_NPKIT_EVENT_DIRECT_RECV_COPY_SEND_ENTRY",
            "-DENABLE_NPKIT_EVENT_DIRECT_RECV_COPY_SEND_EXIT",
            "-DENABLE_NPKIT_EVENT_DIRECT_RECV_REDUCE_COPY_SEND_ENTRY",
            "-DENABLE_NPKIT_EVENT_DIRECT_RECV_REDUCE_COPY_SEND_EXIT",
            "-DENABLE_NPKIT_EVENT_DIRECT_SEND_ENTRY",
            "-DENABLE_NPKIT_EVENT_DIRECT_SEND_EXIT",
            "-DENABLE_NPKIT_EVENT_DIRECT_SEND_FROM_OUTPUT_ENTRY",
            "-DENABLE_NPKIT_EVENT_DIRECT_SEND_FROM_OUTPUT_EXIT",
            "-DENABLE_NPKIT_EVENT_RECV_ENTRY",
            "-DENABLE_NPKIT_EVENT_RECV_EXIT",
            "-DENABLE_NPKIT_EVENT_RECV_COPY_SEND_ENTRY",
            "-DENABLE_NPKIT_EVENT_RECV_COPY_SEND_EXIT",
            "-DENABLE_NPKIT_EVENT_RECV_REDUCE_COPY_ENTRY",
            "-DENABLE_NPKIT_EVENT_RECV_REDUCE_COPY_EXIT",
            "-DENABLE_NPKIT_EVENT_RECV_REDUCE_COPY_SEND_ENTRY",
            "-DENABLE_NPKIT_EVENT_RECV_REDUCE_COPY_SEND_EXIT",
            "-DENABLE_NPKIT_EVENT_RECV_REDUCE_SEND_ENTRY",
            "-DENABLE_NPKIT_EVENT_RECV_REDUCE_SEND_EXIT",
            "-DENABLE_NPKIT_EVENT_SEND_ENTRY",
            "-DENABLE_NPKIT_EVENT_SEND_EXIT",
            "-DENABLE_NPKIT_EVENT_SEND_FROM_OUTPUT_ENTRY",
            "-DENABLE_NPKIT_EVENT_SEND_FROM_OUTPUT_EXIT",
            "-DENABLE_NPKIT_EVENT_PRIM_SIMPLE_WAIT_PEER_ENTRY",
            "-DENABLE_NPKIT_EVENT_PRIM_SIMPLE_WAIT_PEER_EXIT",
            "-DENABLE_NPKIT_EVENT_PRIM_SIMPLE_REDUCE_OR_COPY_MULTI_ENTRY",
            "-DENABLE_NPKIT_EVENT_PRIM_SIMPLE_REDUCE_OR_COPY_MULTI_EXIT",
            "-DENABLE_NPKIT_EVENT_PRIM_LL_WAIT_SEND_ENTRY",
            "-DENABLE_NPKIT_EVENT_PRIM_LL_WAIT_SEND_EXIT",
            "-DENABLE_NPKIT_EVENT_PRIM_LL_DATA_PROCESS_ENTRY",
            "-DENABLE_NPKIT_EVENT_PRIM_LL_DATA_PROCESS_EXIT",
            "-DENABLE_NPKIT_EVENT_PRIM_LL128_WAIT_SEND_ENTRY",
            "-DENABLE_NPKIT_EVENT_PRIM_LL128_WAIT_SEND_EXIT",
            "-DENABLE_NPKIT_EVENT_PRIM_LL128_DATA_PROCESS_ENTRY",
            "-DENABLE_NPKIT_EVENT_PRIM_LL128_DATA_PROCESS_EXIT",
            "-DENABLE_NPKIT_EVENT_NET_SEND_ENTRY",
            "-DENABLE_NPKIT_EVENT_NET_SEND_EXIT",
            "-DENABLE_NPKIT_EVENT_NET_TEST_ENTRY",
            "-DENABLE_NPKIT_EVENT_NET_TEST_EXIT",
            "-DENABLE_NPKIT_EVENT_NET_RECV_ENTRY",
            "-DENABLE_NPKIT_EVENT_NET_RECV_EXIT",
            "-DENABLE_NPKIT_EVENT_ALL_REDUCE_RING_SEND_ENTRY",
            "-DENABLE_NPKIT_EVENT_ALL_REDUCE_RING_SEND_EXIT",
            "-DENABLE_NPKIT_EVENT_ALL_REDUCE_RING_RECV_REDUCE_SEND_ENTRY",
            "-DENABLE_NPKIT_EVENT_ALL_REDUCE_RING_RECV_REDUCE_SEND_EXIT",
            "-DENABLE_NPKIT_EVENT_ALL_REDUCE_RING_DIRECT_RECV_REDUCE_COPY_SEND_ENTRY",
            "-DENABLE_NPKIT_EVENT_ALL_REDUCE_RING_DIRECT_RECV_REDUCE_COPY_SEND_EXIT",
            "-DENABLE_NPKIT_EVENT_ALL_REDUCE_RING_DIRECT_RECV_COPY_SEND_ENTRY",
            "-DENABLE_NPKIT_EVENT_ALL_REDUCE_RING_DIRECT_RECV_COPY_SEND_EXIT",
            "-DENABLE_NPKIT_EVENT_ALL_REDUCE_RING_DIRECT_RECV_ENTRY",
            "-DENABLE_NPKIT_EVENT_ALL_REDUCE_RING_DIRECT_RECV_EXIT",
            "-DENABLE_NPKIT_EVENT_ALL_REDUCE_TREE_UPDOWN_REDUCE_ENTRY",
            "-DENABLE_NPKIT_EVENT_ALL_REDUCE_TREE_UPDOWN_REDUCE_EXIT",
            "-DENABLE_NPKIT_EVENT_ALL_REDUCE_TREE_UPDOWN_BROADCAST_ENTRY",
            "-DENABLE_NPKIT_EVENT_ALL_REDUCE_TREE_UPDOWN_BROADCAST_EXIT",
            "-DENABLE_NPKIT_EVENT_ALL_REDUCE_TREE_SPLIT_REDUCE_BROADCAST_ENTRY",
            "-DENABLE_NPKIT_EVENT_ALL_REDUCE_TREE_SPLIT_REDUCE_BROADCAST_EXIT",
            "-DENABLE_NPKIT_EVENT_ALL_REDUCE_TREE_SPLIT_REDUCE_ENTRY",
            "-DENABLE_NPKIT_EVENT_ALL_REDUCE_TREE_SPLIT_REDUCE_EXIT",
            "-DENABLE_NPKIT_EVENT_ALL_REDUCE_TREE_SPLIT_BROADCAST_ENTRY",
            "-DENABLE_NPKIT_EVENT_ALL_REDUCE_TREE_SPLIT_BROADCAST_EXIT",
            "-DENABLE_NPKIT_EVENT_SEND_RECV_LOCAL_COPY_ENTRY",
            "-DENABLE_NPKIT_EVENT_SEND_RECV_LOCAL_COPY_EXIT",
            "-DENABLE_NPKIT_EVENT_SEND_RECV_SEND_ENTRY",
            "-DENABLE_NPKIT_EVENT_SEND_RECV_SEND_EXIT",
            "-DENABLE_NPKIT_EVENT_SEND_RECV_RECV_ENTRY",
            "-DENABLE_NPKIT_EVENT_SEND_RECV_RECV_EXIT",
            "-DENABLE_NPKIT_EVENT_ALL_GATHER_RING_ENTRY",
            "-DENABLE_NPKIT_EVENT_ALL_GATHER_RING_EXIT",
            "-DENABLE_NPKIT_EVENT_ALL_GATHER_RING_SEND_ENTRY",
            "-DENABLE_NPKIT_EVENT_ALL_GATHER_RING_SEND_EXIT",
            "-DENABLE_NPKIT_EVENT_ALL_GATHER_RING_RECV_COPY_SEND_ENTRY",
            "-DENABLE_NPKIT_EVENT_ALL_GATHER_RING_RECV_COPY_SEND_EXIT",
            "-DENABLE_NPKIT_EVENT_ALL_GATHER_RING_DIRECT_RECV_ENTRY",
            "-DENABLE_NPKIT_EVENT_ALL_GATHER_RING_DIRECT_RECV_EXIT",
            "-DENABLE_NPKIT_EVENT_MSCCL_GENERIC_OP_ENTRY",
            "-DENABLE_NPKIT_EVENT_MSCCL_GENERIC_OP_EXIT",
            "-DENABLE_NPKIT_EVENT_MSCCL_REDUCE_ENTRY",
            "-DENABLE_NPKIT_EVENT_MSCCL_REDUCE_EXIT",
            "-DENABLE_NPKIT_EVENT_MSCCL_SEND_ENTRY",
            "-DENABLE_NPKIT_EVENT_MSCCL_SEND_EXIT",
            "-DENABLE_NPKIT_EVENT_MSCCL_RECV_ENTRY",
            "-DENABLE_NPKIT_EVENT_MSCCL_RECV_EXIT",
            "-DENABLE_NPKIT_EVENT_MSCCL_RUN_ENTRY",
            "-DENABLE_NPKIT_EVENT_MSCCL_RUN_EXIT",
            "-DENABLE_NPKIT_EVENT_MSCCL_RECV_REDUCE_COPY_ENTRY",
            "-DENABLE_NPKIT_EVENT_MSCCL_RECV_REDUCE_COPY_EXIT",
            "-DENABLE_NPKIT_EVENT_MSCCL_INIT_ENTRY",
            "-DENABLE_NPKIT_EVENT_MSCCL_INIT_EXIT",
            "-DENABLE_NPKIT_EVENT_BROADCAST_RING_ENTRY",
            "-DENABLE_NPKIT_EVENT_BROADCAST_RING_EXIT",
            "-DENABLE_NPKIT_EVENT_REDUCE_SCATTER_RING_ENTRY",
            "-DENABLE_NPKIT_EVENT_REDUCE_SCATTER_RING_EXIT",
            "-DENABLE_NPKIT_EVENT_REDUCE_SCATTER_RING_SEND_ENTRY",
            "-DENABLE_NPKIT_EVENT_REDUCE_SCATTER_RING_SEND_EXIT",
            "-DENABLE_NPKIT_EVENT_REDUCE_SCATTER_RING_RECV_REDUCE_SEND_ENTRY",
            "-DENABLE_NPKIT_EVENT_REDUCE_SCATTER_RING_RECV_REDUCE_SEND_EXIT",
            "-DENABLE_NPKIT_EVENT_REDUCE_SCATTER_RING_RECV_REDUCE_COPY_ENTRY",
            "-DENABLE_NPKIT_EVENT_REDUCE_SCATTER_RING_RECV_REDUCE_COPY_EXIT",
            "-DENABLE_NPKIT_PRIM_COLLECT_DATA_PROCESS_TIME",
        ],
    })

def get_rdma_core_compiler_flags():
    return select({
        "DEFAULT": [],
        "fbsource//third-party/rccl/constraints:build_rdma_core": ["-DNCCL_BUILD_RDMA_CORE"],
    })

def get_host_uncahced_memory_flags():
    return select({
        "DEFAULT": ["-DHIP_HOST_UNCACHED_MEMORY"],
        "ovr_config//third-party/rocm/constraints:6.2.1": [],
        "ovr_config//third-party/rocm/constraints:6.4.0": [],
        "ovr_config//third-party/rocm/constraints:6.4.2": [],
        "ovr_config//third-party/rocm/constraints:7.0.0": [],
    })

COMMON_PRE_COMPILER_FLAGS = [
    # "-DENABLE_META_COLLTRACE",
    # "-DENABLE_COLLTRACE",
    # "-DHIP_EVENT_DISABLE_FENCE",
    "-DBUILD_META_INTERNAL",
    "-DNVTX_NO_IMPL",
    # This flag will trigger a codepath that caused 200us latency for any size collective
    # "-DUSE_INDIRECT_FUNCTION_CALL",
    "-D__HIP_PLATFORM_AMD__=1",
    "-D__HIP_PLATFORM_HCC__=1",
    # assume rocm/include/rocm_smi/rocm_smi64Config.h always exist; needed in rocm.5.1.0+ with rocm_smi
    # This header exists from rocm-4.5.2 which is the first stable tp2/rocm version. Thus, assuming
    # such a static configure is fine.
    "-DUSE_ROCM_SMI64CONFIG",
    # This flag is needed for MI300 which will hang w/o
    "-DHIP_UNCACHED_MEMORY",
    "-DROCM_VERSION=FB_ROCM_VERSION",
    # "-DCOMPILE_MSCCL_KERNEL",
    "-DENABLE_LL128",
    # Somehow upstream moved that, but we bindmount that to /etc instead, so keep it there
    "-DIBV_CONFIG_DIR=/etc/libibverbs.d",
    "-DVERBS_PROVIDER_DIR=/usr/lib64/libibverbs",
] + get_npkit_compiler_flags() + get_rdma_core_compiler_flags() + get_host_uncahced_memory_flags()

# Inject random warp delay in device code
if read_bool("rccl", "inject_faults", False):
    COMMON_PRE_COMPILER_FLAGS += ["-DENABLE_FAULT_INJECTION"]

#  dictionaries of srcs and headers from different rccl versions
get_rccl_hipify_srcs_func_dist = {
    "develop": get_rccl_hipify_srcs_develop,
    "rocm-6.4": get_rccl_hipify_srcs_develop,
    "rocm-7.0": get_rccl_hipify_srcs_develop,
}

get_rccl_hipify_headers_func_dist = {
    "develop": get_rccl_hipify_headers_develop,
    "rocm-6.4": get_rccl_hipify_headers_develop,
    "rocm-7.0": get_rccl_hipify_headers_develop,
}
get_rccl_headers_func_dist = {
    "develop": get_rccl_headers_develop,
    "rocm-6.4": get_rccl_headers_develop,
    "rocm-7.0": get_rccl_headers_develop,
}
get_exported_headers_func_dist = {
    "develop": get_rccl_exported_headers_develop,
    "rocm-6.4": get_rccl_exported_headers_develop,
    "rocm-7.0": get_rccl_exported_headers_develop,
}

get_rccl_fbcode_exported_deps_func_dist = {
    "develop": get_rccl_fbcode_exported_deps_develop,
    "rocm-6.4": get_rccl_fbcode_exported_deps_develop,
    "rocm-7.0": get_rccl_fbcode_exported_deps_develop,
}

get_generate_collectives_func_dist = {
    "develop": get_rccl_generate_collectives_develop,
    "rocm-6.4": get_rccl_generate_collectives_develop,
    "rocm-7.0": get_rccl_generate_collectives_develop,
}

get_rccl_git_version_rule_dist = {
    "develop": get_rccl_git_version_rule_develop,
    "rocm-6.4": get_rccl_git_version_rule_develop,
    "rocm-7.0": get_rccl_git_version_rule_develop,
}

get_mscclpp_headers_func_dist = {
    "develop": get_mscclpp_headers_develop,
    "rocm-6.4": get_mscclpp_headers_develop,
    "rocm-7.0": get_mscclpp_headers_develop,
}

get_mscclpp_srcs_func_dist = {
    "develop": get_mscclpp_srcs_develop,
    "rocm-6.4": get_mscclpp_srcs_develop,
    "rocm-7.0": get_mscclpp_srcs_develop,
}

get_mscclpp_nccl_headers_func_dist = {
    "develop": get_mscclpp_nccl_headers_develop,
    "rocm-6.4": get_mscclpp_nccl_headers_develop,
    "rocm-7.0": get_mscclpp_nccl_headers_develop,
}

def get_internal_deps():
    return [
        "fbcode//comms/rcclx/develop/meta/lib:rccl_logger",
        "fbcode//comms/rcclx/develop/meta/lib:colltrace_utils",
        "fbcode//comms/rcclx/develop/meta/scuba:rccl-scuba-logger",
        "fbcode//comms/common/algorithms:algo_factory",
        "//folly:dynamic",
        "fbcode//comms/utils:comm_utils",
    ]

def get_internal_exported_deps():
    return [
        "fbcode//comms/rcclx/develop/meta/lib:proxy_trace",
        "fbcode//comms/utils:str_utils",
    ]

def get_rccl_device_headers():
    device_folder = "{}/src/device".format(rccl_src_dir)
    device_headers = [(device_folder, "*.h"), (device_folder, "network/unpack/*.h")]
    return subdir_glob(device_headers)

def get_rccl_hipify_headers():
    return get_rccl_hipify_headers_func_dist[rccl_ver]()

def get_rccl_headers():
    all_headers = get_rccl_headers_func_dist[rccl_ver]()

    # overwrite with hipified headers
    for header_path, full_path in get_rccl_hipify_headers().items():
        all_headers[header_path] = ":hipify[{}]".format(full_path)

    # add device_table.h
    header_path = "device_table.h"
    all_headers[header_path] = ":generate_collectives[{}]".format(header_path)

    # Apply inject_faults rule after header files are hipified
    if read_bool("rccl", "inject_faults", False):
        inject_faults(
            "{}/cmake/scripts/add_faults.sh".format(rccl_src_dir),
            get_rccl_device_headers().values(),
            [":hipify[{}]".format(f) for f in get_rccl_device_headers().values()],
        )
        for header_path, full_path in get_rccl_device_headers().items():
            all_headers[header_path] = ":inject_faults[{}]".format(full_path)

    return all_headers

def get_rccl_exported_headers():
    return get_exported_headers_func_dist[rccl_ver]()

def get_rccl_hipify_srcs():
    return get_rccl_hipify_srcs_func_dist[rccl_ver]()

def get_rccl_fbcode_exported_deps():
    return get_rccl_fbcode_exported_deps_func_dist[rccl_ver]()

def get_rccl_generate_collectives():
    return get_generate_collectives_func_dist[rccl_ver]()

def get_rccl_git_version_rule(suffix = ""):
    return get_rccl_git_version_rule_dist[rccl_ver](suffix = suffix)

def get_mscclpp_headers():
    return get_mscclpp_headers_func_dist[rccl_ver]()

def get_mscclpp_srcs():
    return get_mscclpp_srcs_func_dist[rccl_ver]()

def get_mscclpp_nccl_headers():
    return get_mscclpp_nccl_headers_func_dist[rccl_ver]()

def build_rccl_objects(suffix = ""):
    get_rccl_git_version_rule(suffix = suffix)

    # We normaly rely on hipifying for .cu files. However, in this case:
    # 1. This .cu file doesn't actually use any nvidia-specific stuff.
    # 2. Hipifying with gpu_cpp_library doesn't wind up working due to header
    #    paths + messing with include directories.
    #
    # We also _need_ this to be named not-.cu because the cxx toolchain assumes
    # if you're compiling with a .cu extension you want to compile with nvcc
    # instead of amdclang.
    fb_native.export_file(
        name = "nccl{}.hip".format(suffix),
        src = "develop/ext-src/mscclpp/apps/nccl/src/nccl.cu",
    )

    mscclpp_public_headers, mscclpp_private_headers = get_mscclpp_headers()

    cpp_library(
        name = "mscclpp{}".format(suffix),
        srcs = get_mscclpp_srcs(),
        preprocessor_flags = [
            "-DUSE_ROCM",
            "-D__HIP_PLATFORM_AMD__",
        ],
        private_headers = mscclpp_private_headers,
        headers = mscclpp_public_headers,
        header_namespace = "",
        compiler_flags = get_rocm_arch_args() + [
            # nlohmann_json doesn't like C++20.
            "-std=c++17",
            "-Wno-option-ignored",
        ],
        external_deps = [
            ("rocm", None, "amdhip64-lazy"),
            ("numa", None, "numa"),
            "nlohmann_json",
        ],
        # Custom rocm toolchain which overwrites cxx_compiler_info with hip_compiler_info.
        _cxx_toolchain = "fbsource//third-party/rccl:cxx-platform010-amdclang-toolchain",
    )

    mscclpp_nccl_public_headers, mscclpp_nccl_private_headers = get_mscclpp_nccl_headers()

    cpp_library(
        name = "mscclpp_nccl{}".format(suffix),
        srcs = [
            ":nccl.hip",
        ],
        propagated_pp_flags = [
            "-DUSE_ROCM",
            "-D__HIP_PLATFORM_AMD__",
        ],
        # These are from third-party/rccl/develop/src/misc/mscclpp/mscclpp_nccl_syms.txt
        # mscclpp/apps/nccl has conflicting symbols with rccl, so we need to rename them.
        #
        # Reproduce with: grep -v '#' third-party/rccl/develop/src/misc/mscclpp/mscclpp_nccl_syms.txt | awk '{print "\"-D" $1 "=" $2 "\","}'
        # Note: these defines are not exported! We don't want them to leak to other libs.
        preprocessor_flags = [
            "-DncclAllGather=mscclpp_ncclAllGather",
            "-DncclAllReduce=mscclpp_ncclAllReduce",
            "-DncclAllToAll=mscclpp_ncclAllToAll",
            "-DncclBcast=mscclpp_ncclBcast",
            "-DncclBroadcast=mscclpp_ncclBroadcast",
            "-DncclCommAbort=mscclpp_ncclCommAbort",
            "-DncclCommCount=mscclpp_ncclCommCount",
            "-DncclCommCuDevice=mscclpp_ncclCommCuDevice",
            "-DncclCommDestroy=mscclpp_ncclCommDestroy",
            "-DncclCommFinalize=mscclpp_ncclCommFinalize",
            "-DncclCommGetAsyncError=mscclpp_ncclCommGetAsyncError",
            "-DncclCommInitAll=mscclpp_ncclCommInitAll",
            "-DncclCommInitRank=mscclpp_ncclCommInitRank",
            "-DncclCommInitRankConfig=mscclpp_ncclCommInitRankConfig",
            "-DncclCommSplit=mscclpp_ncclCommSplit",
            "-DncclCommUserRank=mscclpp_ncclCommUserRank",
            "-DncclGetErrorString=mscclpp_ncclGetErrorString",
            "-DncclGetLastError=mscclpp_ncclGetLastError",
            "-DncclGetUniqueId=mscclpp_ncclGetUniqueId",
            "-DncclGetVersion=mscclpp_ncclGetVersion",
            "-DncclGroupEnd=mscclpp_ncclGroupEnd",
            "-DncclGroupStart=mscclpp_ncclGroupStart",
            "-DncclRecv=mscclpp_ncclRecv",
            "-DncclRedOpCreatePreMulSum=mscclpp_ncclRedOpCreatePreMulSum",
            "-DncclRedOpDestroy=mscclpp_ncclRedOpDestroy",
            "-DncclReduce=mscclpp_ncclReduce",
            "-DncclReduceScatter=mscclpp_ncclReduceScatter",
            "-DncclSend=mscclpp_ncclSend",
            "-DncclCommRegister=mscclpp_ncclCommRegister",
            "-DncclCommDeregister=mscclpp_ncclCommDeregister",
            "-DncclMemAlloc=mscclpp_ncclMemAlloc",
            "-DncclMemFree=mscclpp_ncclMemFree",
        ],
        private_headers = mscclpp_nccl_private_headers,
        headers = mscclpp_nccl_public_headers,
        header_namespace = "",
        compiler_flags = get_rocm_arch_args() + [
            "-Wno-option-ignored",
        ],
        exported_deps = [":mscclpp{}".format(suffix)],
        external_deps = [
            ("rocm", None, "amdhip64-lazy"),
        ],
        _cxx_toolchain = "fbsource//third-party/rccl:cxx-platform010-amdclang-toolchain",
    )

    # Handle the Select object returned by get_rccl_generate_collectives()
    collective_sources = selects.apply(
        get_rccl_generate_collectives(),
        lambda filenames: [
            ":generate_collectives{}[{}]".format(suffix, filename)
            for filename in filenames
            if filename.endswith(".hip")
        ],
    )

    cpp_library(
        name = "rccl_objects{}".format(suffix),
        srcs = [":hipify[{}]".format(src) for src in get_rccl_hipify_srcs()] +
               # git_version needs to be treated special since it's a genrule.
               [":hipify_git_version{}[git_version.cpp]".format(suffix)] +
               collective_sources,
        headers = get_rccl_headers(),
        header_namespace = "",
        compiler_flags = COMMON_COMPILER_FLAGS,
        preprocessor_flags = COMMON_PRE_COMPILER_FLAGS,
        preferred_linkage = "static",
        hip_flags = COMMON_COMPILER_FLAGS + RDC_HIP_FLAGS,
        deps = get_internal_deps(),
        exported_deps = get_rccl_fbcode_exported_deps() + ["fbsource//third-party/fmt:fmt", ":mscclpp{}".format(suffix), ":mscclpp_nccl{}".format(suffix)] + get_internal_exported_deps(),
        exported_external_deps = [
            ("rocm", None, "amdhip64-lazy"),
        ],
    )
    return ":rccl_objects{}".format(suffix)

def _get_link_cmd(rdc_link_flags, objects):
    bash = " && ".join([
        "$(location fbsource//third-party/rocm:rocm_path)/llvm/bin/clang -o $OUT {0} `ar t $(location {1})`"
            .format(" ".join(rdc_link_flags + RDC_HIP_LINK_FLAGS), objects),
    ])
    return bash

def link_rccl_lib(objects, arname, suffix = ""):
    # Manually link with rocm clang to support -fgpu-rdc --hip-link
    bash = selects.apply(COMMON_HIP_FLAGS, native.partial(_get_link_cmd, objects = objects))
    buck_genrule(
        name = "hip_link{}".format(suffix),
        bash = bash,
        out = arname,
        exec_compatible_with = ["ovr_config//os:linux"],
    )
    return ":hip_link{}".format(suffix)

def rccl_third_party_cxx_library(name, arname):
    # When build with dev mode, the rccl lib generated is librccl.so, but it is still statically linked (see link_rccl_lib)
    # Here, we use a prebuilt_cxx_library to expose the static rccl lib to applications under both dev and opt modes
    # We do not have plan to maintain dynamic-link rccl lib for now due to the challenge of packaging and distribution.
    # Also, static build can be accelerated by selective codegen and device targets.

    # Declare hipify rule
    hipify(
        name = "hipify",
        srcs = get_rccl_hipify_srcs() + get_rccl_hipify_headers().values(),
    )
    hipify(
        name = "hipify_git_version",
        srcs = [":git_version"],
    )
    generate_collectives(
        "{}/src/device/generate.py".format(rccl_src_dir),
        "{}/src/device".format(rccl_src_dir),
        get_rccl_generate_collectives(),
    )

    # RCCL requires custom linkage thus all files are compiled in build_rccl_objects
    objects = build_rccl_objects()
    rccl_lib = link_rccl_lib(objects, arname)
    fb_native.prebuilt_cxx_library(
        name = name,
        header_dirs = ["{}/src".format(rccl_src_dir), "{}/src/include".format(rccl_src_dir), "{}".format(rccl_src_dir)],
        static_lib = rccl_lib,
        visibility = ["PUBLIC"],
        exported_headers = get_rccl_exported_headers(),
        exported_deps = [
            "fbsource//third-party/fmt:fmt",
            "fbsource//third-party/rocm:hsa-runtime64-lazy",
            ":mscclpp",
            ":mscclpp_nccl",
        ] + get_internal_deps() + get_internal_exported_deps() + get_rccl_fbcode_exported_deps(),
        header_namespace = "",
    )

def rccl_third_party_cxx_library_internal(name, arname):
    # Declare hipify rule
    hipify(
        name = "hipify_internal",
        srcs = get_rccl_hipify_srcs() + get_rccl_hipify_headers().values(),
    )
    hipify(
        name = "hipify_git_version_internal",
        srcs = [":git_version_internal"],
    )
    generate_collectives(
        "{}/src/device/generate.py".format(rccl_src_dir),
        "{}/src/device".format(rccl_src_dir),
        get_rccl_generate_collectives(),
        suffix = "_internal",
    )

    # RCCL requires custom linkage thus all files are compiled in build_rccl_objects
    objects = build_rccl_objects(suffix = "_internal")
    rccl_lib = link_rccl_lib(objects, arname, suffix = "_internal")
    fb_native.prebuilt_cxx_library(
        name = name,
        header_dirs = ["{}/src".format(rccl_src_dir), "{}/src/include".format(rccl_src_dir), "{}".format(rccl_src_dir)],
        static_lib = rccl_lib,
        visibility = ["PUBLIC"],
        exported_headers = get_rccl_headers(),
        exported_preprocessor_flags = COMMON_PRE_COMPILER_FLAGS,
        exported_deps = [
            "fbsource//third-party/fmt:fmt",
            "fbsource//third-party/rocm:hsa-runtime64-lazy",
            ":mscclpp{}".format("_internal"),
            ":mscclpp_nccl{}".format("_internal"),
        ] + get_internal_deps() + get_internal_exported_deps() + get_rccl_fbcode_exported_deps(),
        header_namespace = "",
    )
