load("@fbcode_macros//build_defs:native_rules.bzl", "buck_filegroup", "buck_genrule")
load("@fbcode_macros//build_defs:platform_utils.bzl", "platform_utils")
load("@fbsource//tools/build_defs:selects.bzl", "selects")
load("@prelude//:paths.bzl", "paths")

def _hipify_impl(ctx: AnalysisContext) -> list[Provider]:
    hipified = {}
    for src in ctx.attrs.srcs:
        # Make sure to update extension to .hip so we use the correct toolchain.
        path = src.short_path
        if paths.split_extension(path)[1] in (".cc", ".cpp", ".cu"):
            path += ".hip"

        output = ctx.actions.declare_output("hipify", path)
        ctx.actions.run(
            [
                ctx.attrs._hipify_exe[RunInfo],
                "-quiet-warnings",
                "-o",
                output.as_output(),
                src,
            ],
            category = "hipify_perl",
            identifier = src.short_path,
        )

        # Subtargets should be the original source name, not with .hip.
        hipified[src.short_path] = output

    return [
        DefaultInfo(
            sub_targets = {
                subtarget: [DefaultInfo(default_output = out)]
                for subtarget, out in hipified.items()
            },
        ),
    ]

_hipify = rule(
    impl = _hipify_impl,
    attrs = {
        "srcs": attrs.list(attrs.source()),
        "_hipify_exe": attrs.default_only(attrs.exec_dep(default = "fbsource//third-party/rocm:hipify-perl")),
    },
)

hipify = platform_utils.default_platform_decorator(_hipify)

# This is a trimmed set of kernels we want to generate based on Meta's internal workloads.
# Excluded data types: int64, fp64.
# Excluded unused collective AlltoAllV_Pivot.
# Unroll factor only supports 1/2, which is already true for gfx942 and gfx950
# Reduction operator only supports Sum
ONLY_FUNCS = "AllReduce * * Sum i8/u8/i32/u32/f16/f32/bf16/f8e4m3/f8e5m2 1/2|AllGather RING * Sum i8 1/2|Reduce RING * Sum i8/u8/i32/u32/f16/f32/bf16/f8e4m3/f8e5m2 1/2|Broadcast RING * Sum i8 1/2|ReduceScatter RING * Sum i8/u8/i32/u32/f16/f32/bf16/f8e4m3/f8e5m2 1/2|SendRecv RING SIMPLE Sum i8 1/2"

def _get_full_bash_cmd(script_file, gen_src, is_fast_build):
    if is_fast_build:
        generate_cmd = "python3 $SRCDIR/{} $OUT/{} OFF ON OFF \"{}\"".format(script_file, gen_src, ONLY_FUNCS)
    else:
        generate_cmd = "python3 $SRCDIR/{} $OUT/{} OFF ON OFF".format(script_file, gen_src)

    return " && ".join([
        "mkdir -p \"$OUT/{}\"".format(gen_src),
        generate_cmd,
        "rename .cpp .hip $OUT/{}/*".format(gen_src),
    ])

def generate_collectives(script_file, gen_src, collective_filenames_selector, suffix = ""):
    # generate .hip/h files for each collective to enable parallel compiling
    # output:
    #   generate files: $OUT/$gen_src/$CollType_$RedOp_$DataType.hip

    # collective_filenames_selector is always a Select object due to internal constraint logic
    outputs = selects.apply(
        collective_filenames_selector,
        lambda filenames: {filename: ["{}/{}".format(gen_src, filename)] for filename in filenames},
    )

    # Determine build mode using constraint selection for bash command
    build_mode_select = select({
        "DEFAULT": False,  # Standard build
        "fbsource//third-party/rccl/constraints:fast-build": True,  # Fast build
    })

    # Generate entire bash command based on build mode
    bash_cmd = selects.apply(build_mode_select, lambda is_fast: _get_full_bash_cmd(script_file, gen_src, is_fast))

    buck_genrule(
        name = "generate_collectives{}".format(suffix),
        srcs = [script_file],
        outs = outputs,
        bash = bash_cmd,
        default_outs = [],
        exec_compatible_with = ["ovr_config//os:linux"],
    )

# apply the add_fault script (from OSS) to device header files to inject warp-level random delay
def inject_faults(script_file, file_keys, hipified_files):
    buck_filegroup(
        name = "inject_faults_script",
        srcs = [script_file],
    )
    outputs = {}
    bash_cmds = []
    for filename in file_keys:
        outputs[filename] = [filename]
        bash_cmds.append("mkdir -p \"$OUT/\\$(dirname {})\"".format(filename))
        file_fullpath = "$OUT/{}".format(filename)
        bash_cmds.append("cp $SRCDIR/{} {}".format(filename, file_fullpath))
        bash_cmds.append("chmod 666 {}".format(file_fullpath))
        bash_cmds.append("$(location :inject_faults_script)/{} {}".format(script_file, file_fullpath))
    buck_genrule(
        name = "inject_faults",
        srcs = hipified_files,
        outs = outputs,
        bash = " && ".join(bash_cmds),
        exec_compatible_with = ["ovr_config//os:linux"],
    )
