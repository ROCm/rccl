#!/usr/bin/env python3

#SBATCH --job-name=rccl-run
#SBATCH --output=rccl-run-%j.out
#SBATCH --error=rccl-run-%j.out
#SBATCH --time=60
#SBATCH --nodes=2
#SBATCH --exclusive
#SBATCH --partition=gt


"""
Run rccl-tests over an env-var matrix, collect .profraw files, merge with
llvm-profdata, and render HTML coverage with llvm-cov.

Now also sets MPI_ROOT / PATH / LD_LIBRARY_PATH / RCCL_INSTALL_DIR from config.yml.

Requirements: PyYAML (pip install pyyaml)
"""

import argparse
import itertools
import os
import sys
import time
import shutil
import subprocess
from pathlib import Path

try:
    import yaml
except Exception:
    print("ERROR: PyYAML is required. Install with: pip install pyyaml", file=sys.stderr)
    sys.exit(2)


subprocess.run(['bash','-lc','source /etc/profile.d/lmod.sh'], check=True)
subprocess.run(['bash','-lc','module load rocm/6.4.1'], check=True)

def which(cmd, extra_dir=None):
    if extra_dir:
        cand = Path(extra_dir) / cmd
        if cand.exists():
            return str(cand)
    return shutil.which(cmd)


def read_config(path):
    with open(path, "r") as f:
        return yaml.safe_load(f)


def ensure_dir(p: Path):
    p.mkdir(parents=True, exist_ok=True)
    return p


def prepend_env_path(var: str, new_path: str):
    if not new_path:
        return
    cur = os.environ.get(var, "")
    parts = [p for p in new_path.split(os.pathsep) if p]
    if cur:
        parts += [p for p in cur.split(os.pathsep) if p]
    # de-dupe while preserving order
    seen = set()
    merged = []
    for p in parts:
        if p not in seen:
            seen.add(p)
            merged.append(p)
    os.environ[var] = os.pathsep.join(merged)


def apply_env_from_config(cfg):
    """
    Sets:
      - os.environ['MPI_ROOT'] and prepends its bin/lib to PATH/LD_LIBRARY_PATH
      - os.environ['RCCL_INSTALL_DIR'] and prepends it to LD_LIBRARY_PATH
    """
    mpi_cfg = (cfg.get("mpi") or {})
    mpi_root = mpi_cfg.get("root") or cfg.get("mpi_root")  # support either key
    if mpi_root:
        mpi_root = str(Path(mpi_root).resolve())
        os.environ["MPI_ROOT"] = mpi_root
        prepend_env_path("PATH", str(Path(mpi_root) / "bin"))
        prepend_env_path("LD_LIBRARY_PATH", str(Path(mpi_root) / "lib"))
        print(f"LD_LIBRARY_PATH: {os.environ['LD_LIBRARY_PATH']}")

    rccl_install_dir = cfg.get("rccl_install_dir")
    if rccl_install_dir:
        rccl_install_dir = str(Path(rccl_install_dir).resolve())
        os.environ["RCCL_INSTALL_DIR"] = rccl_install_dir
        prepend_env_path("LD_LIBRARY_PATH", rccl_install_dir)


def tee_run(cmd, log_path: Path, env=None, cwd=None):
    ensure_dir(log_path.parent)
    with log_path.open("wb") as logf:
        print(f"Executing command: {cmd}")
        proc = subprocess.Popen(
            cmd, cwd=str(cwd) if cwd else None, env=env,
            stdout=subprocess.PIPE, stderr=subprocess.STDOUT, bufsize=1, text=False
        )
        for chunk in iter(lambda: proc.stdout.readline(), b""):
            sys.stdout.buffer.write(chunk); logf.write(chunk); sys.stdout.flush()
        proc.stdout.close()
        return proc.wait()


def combo_key(kv_pairs):
    return "baseline" if not kv_pairs else "__".join([f"{k}_{v}" for k, v in kv_pairs])


def main():
    ap = argparse.ArgumentParser(description="RCCL tests coverage runner")
    ap.add_argument("--config", required=True, help="Path to config.yml")
    ap.add_argument("--run-name", default=None, help="Optional results/<run-name>")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--verbose", action="store_true")
    args = ap.parse_args()

    cfg = read_config(args.config)

    # ---- Apply env from config (MPI_ROOT/PATH/LD_LIBRARY_PATH/RCCL_INSTALL_DIR) ----
    apply_env_from_config(cfg)

    # ---- Core paths from config ----
    rccl_tests_dir = Path(cfg["rccl_tests_dir"]).resolve()
    rccl_install_dir = Path(os.environ.get("RCCL_INSTALL_DIR", cfg["rccl_install_dir"])).resolve()
    results_root = Path(cfg["results_root"]).resolve()

    rccl_test_args = list(map(str, cfg.get("rccl_test_args", [])))
    collectives = cfg.get("collectives", [])
    dtypes = cfg.get("dtypes", [])

    # ---- MPI launch config ----
    mpi = cfg.get("mpi", {}) or {}
    mpi_extra = list(map(str, mpi.get("extra_args", ["--mca", "pml", "ucx", "--mca", "btl", "^vader,openib"])))
    # Export base: now default includes MPI_ROOT and RCCL_INSTALL_DIR too.
    mpi_export_base = list(map(str, mpi.get("export_base", [
        "PATH", "LD_LIBRARY_PATH", "MPI_ROOT", "RCCL_INSTALL_DIR",
        "NCCL_DEBUG=VERSION", "NCCL_SOCKET_IFNAME=eth1",
        "NCCL_DMABUF_ENABLE=1", "RCCL_GDR_FLUSH_GPU_MEM_NO_RELAXED_ORDERING=0",
        "HSA_NO_SCRATCH_RECLAIM=1"
    ])))
    

    llvm_cfg = cfg.get("llvm", {}) or {}
    llvm_tools_dir = llvm_cfg.get("tools_dir")  # optional; else rely on PATH
    llvm_profdata = which("llvm-profdata", llvm_tools_dir) or which("llvm-profdata")
    llvm_cov = which("llvm-cov", llvm_tools_dir) or which("llvm-cov")
    if not llvm_profdata:
        print("ERROR: llvm-profdata not found; load rocm/6.4.1 or set llvm.tools_dir.", file=sys.stderr)
    if not llvm_cov:
        print("ERROR: llvm-cov not found; load rocm/6.4.1 or set llvm.tools_dir.", file=sys.stderr)

    ignore_regex = llvm_cfg.get("ignore_filename_regex", r"ext-src/*")
    project_title = llvm_cfg.get("project_title", "RCCL_Lib_Coverage_Report")
    out_dir_name = llvm_cfg.get("output_dir", "coverage_html")

    # Resolve librccl.so
    librccl = rccl_install_dir / "librccl.so"
    if not librccl.exists():
        cand = list(rccl_install_dir.rglob("librccl.so"))
        if cand: librccl = cand[0]
    if not librccl.exists():
        print(f"ERROR: librccl.so not found under {rccl_install_dir}", file=sys.stderr); sys.exit(6)

    # ---- Env var matrix ----
    env_matrix = cfg.get("env_matrix", {}) or {}
    matrix_keys = list(env_matrix.keys())
    matrix_values = [env_matrix[k] for k in matrix_keys] if matrix_keys else [[]]

    timestamp = time.strftime("%Y%m%d_%H%M%S")
    run_name = args.run_name or f"rccl_coverage_{timestamp}"
    run_root = ensure_dir(results_root / run_name)

    def export_args(items):
        out = []
        for item in items:
            if "=" in item:
                out += ["-x", item]
            elif item in os.environ:
                out += ["-x", item]
        return out

    mpirun = which("mpirun") or which("mpiexec")
    if not mpirun:
        print("ERROR: mpirun/mpiexec not found in PATH (after applying MPI_ROOT).", file=sys.stderr)
        sys.exit(7)

    failures = 0
    total = 0

    if args.verbose:
        print("== ENV after config ==")
        print("MPI_ROOT:", os.environ.get("MPI_ROOT", ""))
        print("RCCL_INSTALL_DIR:", os.environ.get("RCCL_INSTALL_DIR", ""))
        print("PATH:", os.environ.get("PATH", ""))
        print("LD_LIBRARY_PATH:", os.environ.get("LD_LIBRARY_PATH", ""))

    for values in (itertools.product(*matrix_values) if matrix_keys else [()]):
        combo_pairs = list(zip(matrix_keys, values)) if matrix_keys else []
        key = combo_key(combo_pairs)
        combo_root = ensure_dir(run_root / key)

        for coll in collectives:
            coll_root = ensure_dir(combo_root / coll)
            test_bin = rccl_tests_dir / f"{coll}_perf"
            if not test_bin.exists():
                print(f"WARNING: {test_bin} not found; skipping {coll}", file=sys.stderr)
                continue

            for dtype in dtypes:
                leaf = ensure_dir(coll_root / dtype)
                profraw_dir = ensure_dir(leaf / "profraw")
                logs_dir = ensure_dir(leaf / "logs")

                llvm_profile_pattern = str(profraw_dir / "rccl_%h_%p_%m.profraw")

                cmd = [mpirun, "-np", "8"]
                cmd += mpi_extra
                cmd += export_args(mpi_export_base)
                for k, v in combo_pairs:
                    cmd += ["-x", f"{k}={v}"]
                cmd += ["-x", "LLVM_PROFILE_FILE"]
                cmd.append(str(test_bin))
                cmd += rccl_test_args
                cmd += ["-d", dtype]

                env = os.environ.copy()
                env["LLVM_PROFILE_FILE"] = llvm_profile_pattern

                log_path = logs_dir / f"rccl_{key}_{coll}_{dtype}.log"
                total += 1
                print(f"\n=== Running [{key}] {coll} dtype={dtype} ===")
                if args.verbose:
                    print("CWD:", leaf)
                    print("CMD:", " ".join(cmd))
                    print("LLVM_PROFILE_FILE:", env["LLVM_PROFILE_FILE"])
                if args.dry_run:
                    continue
                rc = tee_run(cmd, log_path, env=env, cwd=leaf)
                if rc != 0:
                    failures += 1
                    print(f"!! FAILED (rc={rc}) [{key}] {coll} {dtype}", file=sys.stderr)

    if args.dry_run:
        print("\n[DRY-RUN] Skipping coverage merge.")
        return 0

    # ---- Gather & merge coverage ----
    prof_list_path = run_root / "rawprofiles.list"
    profs = [str(p.resolve()) for p in run_root.rglob("*.profraw")]
    with prof_list_path.open("w") as f:
        f.write("\n".join(profs))
    print(f"\nFound {len(profs)} .profraw files.")

    merged = run_root / "merged.profdata"
    cmd_merge = [llvm_profdata, "merge", "--sparse", "--input-files", str(prof_list_path), "--output", str(merged)]
    print("\n== llvm-profdata merge ==")
    print(" ".join(cmd_merge))
    rc = subprocess.run(cmd_merge).returncode
    if rc != 0:
        print("ERROR: llvm-profdata merge failed.", file=sys.stderr)
        return rc

    cov_out = ensure_dir(run_root / (cfg.get("llvm", {}).get("output_dir", "coverage_html")))
    cmd_show = [
        (which("llvm-cov", llvm_cfg.get("tools_dir")) or "llvm-cov"),
        "show", "--instr-profile", str(merged),
        "--format=html", "--output-dir", str(cov_out),
        "--project-title", cfg.get("llvm", {}).get("project_title", "RCCL_Lib_Coverage_Report"),
        "--ignore-filename-regex", cfg.get("llvm", {}).get("ignore_filename_regex", r"ext-src/*"),
        str(librccl),
    ]
    print("\n== llvm-cov show (HTML) ==")
    print(" ".join(cmd_show))
    rc = subprocess.run(cmd_show).returncode
    if rc != 0:
        print("ERROR: llvm-cov show failed.", file=sys.stderr)
        return rc

    print(f"\nDONE. Runs: {total}, Failures: {failures}")
    print(f"HTML coverage: {cov_out}")
    print(f"Merged profdata: {merged}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

