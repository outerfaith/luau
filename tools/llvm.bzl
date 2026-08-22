def _llvm_repo_impl(ctx):
    llvm_config = ctx.which("llvm-config")

    if llvm_config == None:
        fail("llvm-config not found in PATH")

    def run(args):
        result = ctx.execute([llvm_config] + args)

        if result.return_code != 0:
            fail(
                "llvm-config {} failed:\n{}".format(
                    " ".join(args),
                    result.stderr,
                ),
            )

        return result.stdout.strip()

    includedir = run(["--includedir"])
    libdir = run(["--libdir"])
    libs = run(["--libs"])
    system_libs = run(["--system-libs"])

    linkopts = ["-L" + libdir]

    linkopts += [
        flag
        for flag in libs.split(" ")
        if flag
    ]

    linkopts += [
        flag
        for flag in system_libs.split(" ")
        if flag
    ]

    ctx.file(
        "BUILD.bazel",
        """\
load("@rules_cc//cc:cc_library.bzl", "cc_library")

package(default_visibility = ["//visibility:public"])

cc_library(
    name = "llvm",
    includes = [{includedir}],
    linkopts = {linkopts},
)
""".format(
            includedir = repr(includedir),
            linkopts = repr(linkopts),
        ),
    )

llvm_repository = repository_rule(
    implementation = _llvm_repo_impl,
    local = True,
)
