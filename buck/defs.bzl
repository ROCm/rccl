load("@fbcode_macros//build_defs:platform_utils.bzl", "platform_utils")
load("@prelude//:asserts.bzl", "asserts")
load("@prelude//cxx:compile.bzl", "create_cmd_args")
load(
    "@prelude//cxx:cxx_toolchain_types.bzl",
    "CxxCompilerInfo",
    "CxxInternalTools",
    "CxxPlatformInfo",
    "CxxToolchainInfo",
    "LinkerInfo",
    "cxx_toolchain_infos",
)
load("@prelude//cxx:debug.bzl", "SplitDebugMode")
load("@prelude//utils:pick.bzl", _pick_and_add = "pick_and_add", _pick_bin = "pick_bin")
load("@prelude//utils:utils.bzl", "value_or")

def _override_info_kwargs(info: Provider, overrides: dict[str, typing.Any]) -> dict[str, typing.Any]:
    return {
        attr: value_or(overrides.get(attr), getattr(info, attr))
        for attr in dir(info)
    }

def _hip_toolchain_override_impl(ctx: AnalysisContext) -> list[Provider]:
    """
    Returns a toolchain with the cxx toolchain info replaced with hip toolchain
    info (+ a few extrat things that rccl needs).

    This is to facilitate better compilation of AMD-specific projects that need
    amdclang support (that we don't have in our own clang-17+ versions).
    """
    base_toolchain = ctx.attrs.base[CxxToolchainInfo]
    asserts.true(
        hasattr(base_toolchain, "hip_compiler_info"),
        "Expected base toolchain to have hip_compiler_info",
    )
    hip_info = base_toolchain.hip_compiler_info

    def mk_argsfile(filename: str, content) -> Artifact:
        artifact, _ = ctx.actions.write(filename, content, allow_args = True)
        return artifact

    def create_cxx_compiler_info_override() -> CxxCompilerInfo:
        # Override cxx compiler info with hip info.
        compiler_flags = _pick_and_add(ctx.attrs.cxx_compiler_flags, ctx.attrs.additional_cxx_compiler_flags, hip_info.compiler_flags)

        args_list = [base_toolchain.hip_compiler_info.preprocessor_flags, compiler_flags]
        argsfile = mk_argsfile(
            "cxxflags",
            create_cmd_args(
                False,  # is_nasm
                False,  # is_xcode_argsfile
                args_list,
            ),
        )
        argsfile_xcode = mk_argsfile(
            "cxxflags_xcode",
            create_cmd_args(
                False,  # is_nasm
                True,  # is_xcode_argsfile
                args_list,
            ),
        )
        return CxxCompilerInfo(
            **_override_info_kwargs(
                base_toolchain.hip_compiler_info,
                dict(compiler_flags = compiler_flags, argsfile = argsfile, argsfile_xcode = argsfile_xcode),
            )
        )

    cxx_info = create_cxx_compiler_info_override()

    base_linker_info = base_toolchain.linker_info
    linker_info = LinkerInfo(
        **_override_info_kwargs(
            base_toolchain.linker_info,
            dict(
                archiver = _pick_bin(ctx.attrs.archiver, base_linker_info.archiver),
                archiver_flags = value_or(ctx.attrs.archiver_flags, base_linker_info.archiver_flags),
                archiver_type = value_or(ctx.attrs.archiver_type, base_linker_info.archiver_type),
                archive_contents = value_or(ctx.attrs.archive_contents, base_linker_info.archive_contents),
                use_archiver_flags = value_or(ctx.attrs.use_archiver_flags, base_linker_info.use_archiver_flags),
            ),
        )
    )

    return [
        DefaultInfo(),
    ] + cxx_toolchain_infos(
        internal_tools = ctx.attrs._internal_tools[CxxInternalTools],
        platform_name = ctx.attrs.base[CxxPlatformInfo].name,
        linker_info = linker_info,
        as_compiler_info = base_toolchain.as_compiler_info,
        asm_compiler_info = base_toolchain.asm_compiler_info,
        binary_utilities_info = base_toolchain.binary_utilities_info,
        bolt_enabled = base_toolchain.bolt_enabled,
        c_compiler_info = base_toolchain.c_compiler_info,
        # Note: this is where the cxx --> hip override happens!
        cxx_compiler_info = cxx_info,
        llvm_link = base_toolchain.llvm_link,
        cuda_compiler_info = base_toolchain.cuda_compiler_info,
        # Keep hip_compiler info correct if we actually process any .hip files.
        hip_compiler_info = base_toolchain.hip_compiler_info,
        header_mode = base_toolchain.header_mode,
        headers_as_raw_headers_mode = base_toolchain.headers_as_raw_headers_mode,
        use_dep_files = base_toolchain.use_dep_files,
        clang_remarks = base_toolchain.clang_remarks,
        clang_llvm_statistics = base_toolchain.clang_llvm_statistics,
        gcno_files = base_toolchain.gcno_files,
        clang_trace = base_toolchain.clang_trace,
        object_format = base_toolchain.object_format,
        strip_flags_info = base_toolchain.strip_flags_info,
        pic_behavior = base_toolchain.pic_behavior,
        # amdclang doesn't support split debug mode.
        split_debug_mode = SplitDebugMode("none"),
        target_sdk_version = base_toolchain.target_sdk_version,
    )

_hip_toolchain_override = rule(
    impl = _hip_toolchain_override_impl,
    attrs = {
        "additional_cxx_compiler_flags": attrs.option(attrs.list(attrs.arg()), default = None),
        "archive_contents": attrs.option(attrs.string(), default = None),
        "archiver": attrs.option(attrs.exec_dep(providers = [RunInfo]), default = None),
        "archiver_flags": attrs.option(attrs.list(attrs.arg()), default = None),
        "archiver_type": attrs.option(attrs.string(), default = None),
        "base": attrs.toolchain_dep(providers = [CxxToolchainInfo]),
        "cxx_compiler_flags": attrs.option(attrs.list(attrs.arg()), default = None),
        "use_archiver_flags": attrs.option(attrs.bool(), default = None),
        "_internal_tools": attrs.default_only(attrs.exec_dep(providers = [CxxInternalTools], default = "prelude//cxx/tools:internal_tools")),
    },
    is_toolchain_rule = True,
)

hip_toolchain_override = platform_utils.default_platform_decorator(_hip_toolchain_override)
