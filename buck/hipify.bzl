load("@fbcode_macros//build_defs:native_rules.bzl", "buck_filegroup", "buck_genrule")

def hipify(name, cu_srcs, headers):
    # Expose source files to buck so that the hipify rule can find the sources
    buck_filegroup(
        name = "hipify_src_path",
        srcs = cu_srcs,
    )

    # Hipify rule
    # - Add .hip extension for specified source files
    # - Add header namespace in generated files otherwise it cannot find the header.
    #   Cannot wrap headers in cpp_library with header_namespace="" because header_namespace is not allowed in fbcode (T15633682)
    outputs = {}
    cmd = ["mkdir -p \"$OUT\""]
    for src in cu_srcs:
        outputs[src] = [src + ".hip"]
        cmd.append("mkdir -p \"$OUT/\\$(dirname {})\"".format(src))
        cmd.append("cp \"$(location :hipify_src_path)/{}\" \"$OUT/{}.hip\"".format(src, src))
        for h in headers:
            # Extract just the filename part (after the last slash)
            h_parts = h.split("/")
            h_filename = h_parts[-1]
            cmd.append("sed -i 's/\"{}\"/\"comms\\/rcclx\\/develop\\/test\\/common\\/{}\"/g' \"$OUT/{}.hip\"".format(h_filename, h_filename, src))

    buck_genrule(
        name = name,
        outs = outputs,
        default_outs = [],
        cmd = " && ".join(cmd),
    )
