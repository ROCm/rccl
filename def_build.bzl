load("@fbcode_macros//build_defs:native_rules.bzl", "buck_genrule")
load("@fbsource//tools/build_defs:glob_defs.bzl", "subdir_glob")
load("@fbsource//tools/build_defs:selects.bzl", "selects")
load("@prelude//:paths.bzl", "paths")
load(":METADATA.bzl", "METADATA")

rccl_ver = "develop"

# define dictionaries of srcs and headers
src_prefix = paths.join(rccl_ver, "src")
header_prefix = paths.join(rccl_ver, "src")
meta_prefix = paths.join(rccl_ver, "meta")

# rccl libs
rccl_LIBS_HIPIFY_SRCS_GLOB = [
    "commDumpMeta.cc",
    "bootstrap.cc",
    "channel.cc",
    "collectives.cc",
    "device/common.cu",
    "debug.cc",
    # Do not include enhcompat file because it provides empty implementation
    # of some new ROCm API to fool linker. It will make those new API call to
    # fail with random errors and debugging is very difficult
    # "enhcompat.cc",
    "enqueue.cc",
    "graph/connect.cc",
    "graph/paths.cc",
    "graph/rings.cc",
    "graph/rome_models.cc",
    "graph/search.cc",
    "graph/topo.cc",
    "graph/trees.cc",
    "graph/tuning.cc",
    "graph/xml.cc",
    "group.cc",
    "init.cc",
    "misc/alt_rsmi.cc",
    "misc/archinfo.cc",
    "misc/argcheck.cc",
    "misc/ibvsymbols.cc",
    "misc/ibvwrap.cc",
    "misc/ipcsocket.cc",
    "misc/msccl/msccl_lifecycle.cc",
    "misc/msccl/msccl_parser.cc",
    "misc/msccl/msccl_setup.cc",
    "misc/msccl/msccl_status.cc",
    "misc/mscclpp/mscclpp_nccl.cc",
    "misc/npkit.cc",
    "misc/nvmlwrap_stub.cc",
    "misc/param.cc",
    "misc/profiler.cc",
    "misc/rocm_smi_wrap.cc",
    "misc/rocmwrap.cc",
    "misc/shmutils.cc",
    "misc/signals.cc",
    "misc/socket.cc",
    "misc/strongstream.cc",
    "misc/api_trace.cc",
    "misc/tuner.cc",
    "misc/utils.cc",
    "misc/recorder.cc",
    "msccl.cc",
    "net.cc",
    "proxy.cc",
    "transport.cc",
    "transport/coll_net.cc",
    "transport/generic.cc",
    "transport/net.cc",
    "transport/net_ib.cc",
    "transport/net_socket.cc",
    "transport/nvls.cc",
    "transport/p2p.cc",
    "transport/shm.cc",
    # only add collectives that don't need to be expanded
    # expanded collectives are handled by get_rccl_expanded_coll_srcs_develop
    # "collectives/device/all_gather.cu",
    # "collectives/device/alltoall_pivot.cu",
    # "collectives/device/broadcast.cu",
    # "collectives/device/functions.cu",
    "device/onerank.cu",
    # "collectives/device/sendrecv.cu",
    "rccl_wrap.cc",
    "mnnvl.cc",
    "ras/client.cc",
    "ras/client_support.cc",
    "ras/collectives.cc",
    "ras/peers.cc",
    "ras/ras.cc",
    "ras/rasnet.cc",
    "register/coll_reg.cc",
    "register/register.cc",
    "register/sendrecv_reg.cc",
]

# headers that need to be hipified, check CMakeLists.txt
rccl_LIBS_HIPIFY_HEADERS_GLOB = [
    (header_prefix, "graph/*.h"),
    (header_prefix + "/device", "*.h"),
    (header_prefix + "/ras", "*.h"),
    (header_prefix + "/device", "network/unpack/*.h"),
    (header_prefix + "/include", "*.h"),
    (header_prefix + "/include", "*.hpp"),
    (header_prefix + "/include", "npkit/*.h"),
    (header_prefix + "/include", "nvtx3/*.h"),
    (header_prefix + "/include", "nvtx3/*.hpp"),
    (header_prefix + "/include", "nvtx3/nvtxDetail/*.h"),
    (header_prefix + "/include", "nvtx3/nvtxExtDetail/*.h"),
    (header_prefix + "/include", "msccl/*.h"),
    (rccl_ver + "/meta/colltrace", "*.h"),
    (rccl_ver + "/meta/ctran", "*.h"),
    (rccl_ver + "/meta/algorithms", "*.h"),
    (rccl_ver, "meta/lpcoll/*.h"),
]

rccl_LIBS_HEADERS_GLOB = rccl_LIBS_HIPIFY_HEADERS_GLOB + [
    (header_prefix, "*.h"),
]

rccl_LIBS_EXP_HEADERS_GLOB = [
    (header_prefix, "rccl.h"),
]

rccl_LIBS_FBCODE_EXPORTED_DEPS = [
    "fbcode//third-party-buck/projects/rocm:amd_smi-lazy",
] + select({
    "DEFAULT": [],
    "fbsource//third-party/rccl/constraints:adhoc_brcm": [
        "fbsource//third-party/rdma-core/adhoc/brcm:bnxt_re",
        "fbsource//third-party/rdma-core:ibverbs",
    ],
})

rccl_MSCCLPP_SRCS_GLOB = [
    paths.join(rccl_ver, "ext-src/mscclpp/src/**/*.cc"),
    paths.join(rccl_ver, "ext-src/mscclpp/src/**/*.cpp"),
    # This is included in MSCCL++'s CMakeLists.txt, but there's only one .cu
    # file and it's ifdef'd out. This would need to be hipified (or renamed to
    # .hip so we don't try to compile with nvcc).
    # paths.join(rccl_ver, "ext-src/mscclpp/src/**/*.cu"),
    paths.join(src_prefix, "src/misc/mscclpp/mscclpp_nccl.cc"),
]

def get_rccl_hipify_srcs_develop():
    return [paths.join(src_prefix, src) for src in rccl_LIBS_HIPIFY_SRCS_GLOB] + \
           subdir_glob([(rccl_ver + "/meta/colltrace", "*.cc")]).values() + \
           subdir_glob([(rccl_ver + "/meta/ctran", "*.cc")]).values() + \
           subdir_glob([(rccl_ver + "/meta/algorithms", "*.cc")]).values() + \
           subdir_glob([(rccl_ver + "/meta/lpcoll", "*.cc")]).values()

def get_rccl_hipify_headers_develop():
    return subdir_glob(rccl_LIBS_HIPIFY_HEADERS_GLOB)

def get_rccl_headers_develop():
    return subdir_glob(rccl_LIBS_HEADERS_GLOB)

def get_rccl_exported_headers_develop():
    return subdir_glob(rccl_LIBS_EXP_HEADERS_GLOB)

def get_rccl_fbcode_exported_deps_develop():
    return rccl_LIBS_FBCODE_EXPORTED_DEPS

def get_rccl_git_version_rule_develop(suffix = ""):
    buck_genrule(
        name = "git_version{}".format(suffix),
        out = "git_version.cpp",
        cmd = """
    echo 'const char *rcclGitHash = "{branch}:{hash}";' > $OUT
        """.format(
            branch = METADATA["version"],
            hash = METADATA["upstream_hash"][:7],
        ),
        visibility = ["PUBLIC"],
    )

# Returns a tuple of (public, private) headers because depending on the type
# they need a different header path.
def get_mscclpp_headers_develop():
    public = subdir_glob([
        (paths.join(rccl_ver, "ext-src/mscclpp/include"), "**/*.hpp"),
        (paths.join(rccl_ver, "ext-src/mscclpp/src/include"), "debug.h"),
        (paths.join(rccl_ver, "src/include"), "mscclpp/mscclpp_nccl.h"),
    ])
    private = subdir_glob(
        [
            (paths.join(rccl_ver, "ext-src/mscclpp/src/include"), "**/*.hpp"),
            (paths.join(rccl_ver, "ext-src/mscclpp/src/include"), "**/*.h"),
        ],
        # This is exposed as a public header for some reason...
        exclude = [paths.join(rccl_ver, "ext-src/mscclpp/src/include/debug.h")],
    )
    return public, private

def get_mscclpp_srcs_develop():
    return glob(rccl_MSCCLPP_SRCS_GLOB)

def get_mscclpp_nccl_headers_develop():
    public = subdir_glob([
        (paths.join(rccl_ver, "ext-src/mscclpp/apps/nccl/include"), "*.h"),
    ])
    private = subdir_glob([
        (paths.join(rccl_ver, "ext-src/mscclpp/apps/nccl/src"), "*.hpp"),
    ])
    return public, private

def _get_fast_build_filenames():
    # Fast build filenames - exact list provided for fast build mode
    # To genearte these filenames, run `python3 src/device/generate.py /tmp OFF OFF OFF ${ONLY_FUNCS}`
    fast_build_filenames = [
        "device_table.h",
        "host_table.cpp",
        "all_gather_sum_i8.cpp",
        "all_reduce_sum_u8.cpp",
        "all_reduce_sum_u32.cpp",
        "all_reduce_sum_f16.cpp",
        "all_reduce_sum_f32.cpp",
        "all_reduce_sum_bf16.cpp",
        "all_reduce_sum_f8e4m3.cpp",
        "all_reduce_sum_f8e5m2.cpp",
        "broadcast_sum_i8.cpp",
        "reduce_sum_u8.cpp",
        "reduce_sum_u32.cpp",
        "reduce_sum_f16.cpp",
        "reduce_sum_f32.cpp",
        "reduce_sum_bf16.cpp",
        "reduce_sum_f8e4m3.cpp",
        "reduce_sum_f8e5m2.cpp",
        "reduce_scatter_sum_u8.cpp",
        "reduce_scatter_sum_u32.cpp",
        "reduce_scatter_sum_f16.cpp",
        "reduce_scatter_sum_f32.cpp",
        "reduce_scatter_sum_bf16.cpp",
        "reduce_scatter_sum_f8e4m3.cpp",
        "reduce_scatter_sum_f8e5m2.cpp",
        "sendrecv_sum_i8.cpp",
    ]
    return fast_build_filenames

def _get_full_build_filenames():
    # Full build filenames - complete list for standard build
    return [
        "all_gather_sum_i8.cpp",
        "all_reduce_sum_f8e4m3.cpp",
        "reduce_premulsum_u32.cpp",
        "reduce_scatter_prod_f32.cpp",
        "all_reduce_minmax_bf16.cpp",
        "all_reduce_sum_f8e5m2.cpp",
        "reduce_premulsum_u64.cpp",
        "reduce_scatter_prod_f64.cpp",
        "all_reduce_minmax_f16.cpp",
        "all_reduce_sumpostdiv_u32.cpp",
        "reduce_premulsum_u8.cpp",
        "reduce_scatter_prod_f8e4m3.cpp",
        "all_reduce_minmax_f32.cpp",
        "all_reduce_sumpostdiv_u64.cpp",
        "reduce_prod_bf16.cpp",
        "reduce_scatter_prod_f8e5m2.cpp",
        "all_reduce_minmax_f64.cpp",
        "all_reduce_sumpostdiv_u8.cpp",
        "reduce_prod_f16.cpp",
        "reduce_scatter_prod_u32.cpp",
        "all_reduce_minmax_f8e4m3.cpp",
        "all_reduce_sum_u32.cpp",
        "reduce_prod_f32.cpp",
        "reduce_scatter_prod_u64.cpp",
        "all_reduce_minmax_f8e5m2.cpp",
        "all_reduce_sum_u64.cpp",
        "reduce_prod_f64.cpp",
        "reduce_scatter_prod_u8.cpp",
        "all_reduce_minmax_u32.cpp",
        "all_reduce_sum_u8.cpp",
        "reduce_prod_f8e4m3.cpp",
        "reduce_scatter_sum_bf16.cpp",
        "all_reduce_minmax_u64.cpp",
        "alltoall_pivot_sum_i8.cpp",
        "reduce_prod_f8e5m2.cpp",
        "reduce_scatter_sum_f16.cpp",
        "all_reduce_minmax_u8.cpp",
        "broadcast_sum_i8.cpp",
        "reduce_prod_u32.cpp",
        "reduce_scatter_sum_f32.cpp",
        "all_reduce_premulsum_bf16.cpp",
        "device_table.cpp",
        "reduce_prod_u64.cpp",
        "reduce_scatter_sum_f64.cpp",
        "all_reduce_premulsum_f16.cpp",
        "device_table.h",
        "reduce_prod_u8.cpp",
        "reduce_scatter_sum_f8e4m3.cpp",
        "all_reduce_premulsum_f32.cpp",
        "host_table.cpp",
        "reduce_scatter_minmax_bf16.cpp",
        "reduce_scatter_sum_f8e5m2.cpp",
        "all_reduce_premulsum_f64.cpp",
        "reduce_scatter_minmax_f16.cpp",
        "reduce_scatter_sumpostdiv_u32.cpp",
        "all_reduce_premulsum_f8e4m3.cpp",
        "reduce_scatter_minmax_f32.cpp",
        "reduce_scatter_sumpostdiv_u64.cpp",
        "all_reduce_premulsum_f8e5m2.cpp",
        "reduce_scatter_minmax_f64.cpp",
        "reduce_scatter_sumpostdiv_u8.cpp",
        "all_reduce_premulsum_u32.cpp",
        "reduce_scatter_minmax_f8e4m3.cpp",
        "reduce_scatter_sum_u32.cpp",
        "all_reduce_premulsum_u64.cpp",
        "reduce_minmax_bf16.cpp",
        "reduce_scatter_minmax_f8e5m2.cpp",
        "reduce_scatter_sum_u64.cpp",
        "all_reduce_premulsum_u8.cpp",
        "reduce_minmax_f16.cpp",
        "reduce_scatter_minmax_u32.cpp",
        "reduce_scatter_sum_u8.cpp",
        "all_reduce_prod_bf16.cpp",
        "reduce_minmax_f32.cpp",
        "reduce_scatter_minmax_u64.cpp",
        "reduce_sum_bf16.cpp",
        "all_reduce_prod_f16.cpp",
        "reduce_minmax_f64.cpp",
        "reduce_scatter_minmax_u8.cpp",
        "reduce_sum_f16.cpp",
        "all_reduce_prod_f32.cpp",
        "reduce_minmax_f8e4m3.cpp",
        "reduce_scatter_premulsum_bf16.cpp",
        "reduce_sum_f32.cpp",
        "all_reduce_prod_f64.cpp",
        "reduce_minmax_f8e5m2.cpp",
        "reduce_scatter_premulsum_f16.cpp",
        "reduce_sum_f64.cpp",
        "all_reduce_prod_f8e4m3.cpp",
        "reduce_minmax_u32.cpp",
        "reduce_scatter_premulsum_f32.cpp",
        "reduce_sum_f8e4m3.cpp",
        "all_reduce_prod_f8e5m2.cpp",
        "reduce_minmax_u64.cpp",
        "reduce_scatter_premulsum_f64.cpp",
        "reduce_sum_f8e5m2.cpp",
        "all_reduce_prod_u32.cpp",
        "reduce_minmax_u8.cpp",
        "reduce_scatter_premulsum_f8e4m3.cpp",
        "reduce_sumpostdiv_u32.cpp",
        "all_reduce_prod_u64.cpp",
        "reduce_premulsum_bf16.cpp",
        "reduce_scatter_premulsum_f8e5m2.cpp",
        "reduce_sumpostdiv_u64.cpp",
        "all_reduce_prod_u8.cpp",
        "reduce_premulsum_f16.cpp",
        "reduce_scatter_premulsum_u32.cpp",
        "reduce_sumpostdiv_u8.cpp",
        "all_reduce_sum_bf16.cpp",
        "reduce_premulsum_f32.cpp",
        "reduce_scatter_premulsum_u64.cpp",
        "reduce_sum_u32.cpp",
        "all_reduce_sum_f16.cpp",
        "reduce_premulsum_f64.cpp",
        "reduce_scatter_premulsum_u8.cpp",
        "reduce_sum_u64.cpp",
        "all_reduce_sum_f32.cpp",
        "reduce_premulsum_f8e4m3.cpp",
        "reduce_scatter_prod_bf16.cpp",
        "reduce_sum_u8.cpp",
        "all_reduce_sum_f64.cpp",
        "reduce_premulsum_f8e5m2.cpp",
        "reduce_scatter_prod_f16.cpp",
        "sendrecv_sum_i8.cpp",
    ]

def _convert_to_hip_filenames(gen_filenames):
    # generate.py from AMD generate .cpp files, need rename to .hip files to pass our buckified AMD toolchain
    ret = []
    for filename in gen_filenames:
        if filename.endswith(".cpp"):
            ret.append(filename.split(".")[0] + ".hip")
        else:
            ret.append(filename)
    return ret

# Get a list of filenames that generate.py would produce
# see src/device/generate.py
def get_rccl_generate_collectives_develop():
    # Use Buck constraints to determine which filenames to return
    return selects.if_(
        select({
            "DEFAULT": False,
            "fbsource//third-party/rccl/constraints:fast-build": True,
        }),
        _convert_to_hip_filenames(_get_fast_build_filenames()),
        _convert_to_hip_filenames(_get_full_build_filenames()),
    )
