#!/usr/bin/env bash
# Full-matrix rccl-tests stress run for the IB net-checksum feature.
# Exercises every collective binary in ${RCCL_TESTS_BIN_DIR} from -b 8 (bytes)
# to -e 1G with factor 2 so each call iterates ~28 sizes, with -w warmup + -n
# perf iters per size and rccl-tests' built-in validation (-c 1, the default)
# turning every iteration into a correctness check. Designed to land
# kernel-side mismatch warnings ("RCCL: net recv csum mismatch ...") and
# rccl-tests #wrong counts in the same log so a single pass surfaces both
# data corruption and checksum-impl regressions.
#
# Usage:
#   tools/run_csum_stress.sh                  # auto-pick ~/csum_stress_<N>/
#   tools/run_csum_stress.sh ~/my_run         # use that dir (suffixed _1, _2,
#                                             # ... if it already exists)
#   ONLY="all_reduce_perf alltoall_perf" \
#     tools/run_csum_stress.sh                # restrict to a subset
#   SKIP="hypercube_perf" tools/run_csum_stress.sh
#
# Env overrides (any of these is forwarded to mpirun as -env, so they reach
# every rank; the defaults match tools/run_csum_debug.sh so a stress run uses
# the same NIC / TC / topology config as the single-shot debug run):
#   HOSTS=host1:N,host2:N,...                 # auto-derived from SLURM_NODELIST
#                                             # (+ SLURM_GPUS_ON_NODE /
#                                             # SLURM_NTASKS_PER_NODE) inside an
#                                             # sbatch/salloc/srun allocation;
#                                             # falls back to the historical
#                                             # useocpm2m-097-026:8,032:8 pair
#                                             # when SLURM env is absent.
#   NP=16                                     # defaults to sum of :N suffixes
#                                             # in HOSTS (total GPUs across all
#                                             # nodes), or 16 if HOSTS has none.
#   DTYPE=bfloat16                            # rccl-tests -d
#   MIN_BYTES=8                               # rccl-tests -b
#   MAX_BYTES=1G                              # rccl-tests -e
#   STEP_FACTOR=2                             # rccl-tests -f
#   WARMUP=5                                  # rccl-tests -w
#   ITERS=20                                  # rccl-tests -n
#   GPUS_PER_THREAD=1                         # rccl-tests -g
#   RCCL_IB_RDMA_CHECKSUM=1                   # gate kernel XOR + IB IMM csum
#   RCCL_IB_RDMA_CHECKSUM_BYTES=0             # per-slot byte cap (0=no cap,
#                                             # N>0 skips XOR + publishes
#                                             # NCCL_IB_CHECKSUM_NONE for any
#                                             # slot whose payload > N bytes)
#   NCCL_DEBUG=WARN                           # bump to INFO if you want the
#                                             # init "Kernel net checksum:"
#                                             # line in every log
#   NCCL_DEBUG_SUBSYS=INIT                    # WARN+NET would also surface
#                                             # the kernel mismatch printfs
#   RCCL_TESTS_BIN_DIR=${HOME}/rocm-systems/projects/rccl-tests/build
#   RCCL_BUILD_DIR=${HOME}/rccl/build/release # librccl.so load path
#   MPIRUN_BIN=${HOME}/mpich/install/bin/mpirun
#
# The per-collective rccl-tests stdout/stderr is teed into
#   ${LOG_DIR}/<binary>.log
# and a final summary table is printed (and appended to
# ${LOG_DIR}/SUMMARY.txt) reporting, per collective:
#   * exit status (0 = mpirun returned cleanly)
#   * total rccl-tests #wrong (sum across all sizes, in-place + out-of-place)
#   * count of "RCCL: net recv csum mismatch" kernel printfs
#   * count of NCCL WARN / ERROR lines
# Any non-zero in any column is highlighted with a "FAIL" marker so a quick
# grep ^FAIL ${LOG_DIR}/SUMMARY.txt tells you what to look at first.

set -uo pipefail

# ---- log directory selection (mirror run_csum_debug.sh's suffix rule) ------
if [[ -n "${1:-}" ]]; then
  LOG_DIR="$1"
elif [[ -n "${LOG_DIR:-}" ]]; then
  :  # honour pre-set LOG_DIR
else
  next=1
  shopt -s nullglob
  for d in "${HOME}"/csum_stress_[0-9]*; do
    [[ -d "${d}" ]] || continue
    base="${d##*/csum_stress_}"
    if [[ "${base}" =~ ^[0-9]+$ ]] && (( base + 1 > next )); then
      next=$(( base + 1 ))
    fi
  done
  shopt -u nullglob
  LOG_DIR="${HOME}/csum_stress_${next}"
fi
if [[ -e "${LOG_DIR}" ]]; then
  n=1
  while [[ -e "${LOG_DIR}_${n}" ]]; do n=$(( n + 1 )); done
  LOG_DIR="${LOG_DIR}_${n}"
fi
mkdir -p "${LOG_DIR}"

# ---- defaults (overridable via env) ----------------------------------------
# Derive HOSTS / NP from SLURM env when available (sbatch/salloc/srun); explicit
# HOSTS / NP env (or values inherited from the caller) still win.
#   nodelist  <- SLURM_JOB_NODELIST | SLURM_NODELIST, expanded via `scontrol`
#   per_node  <- SLURM_GPUS_ON_NODE | SLURM_GPUS_PER_NODE | SLURM_NTASKS_PER_NODE | 8
#   NP        = sum of the :N suffixes in HOSTS (i.e. total GPUs across nodes)
if [[ -z "${HOSTS:-}" ]]; then
  _nodelist="${SLURM_JOB_NODELIST:-${SLURM_NODELIST:-}}"
  if [[ -n "${_nodelist}" ]] && command -v scontrol >/dev/null 2>&1; then
    _per_node="${SLURM_GPUS_ON_NODE:-${SLURM_GPUS_PER_NODE:-${SLURM_NTASKS_PER_NODE:-8}}}"
    HOSTS=$(scontrol show hostname "${_nodelist}" | sed "s/$/:${_per_node}/" | paste -sd, -)
  fi
fi
: "${HOSTS:=useocpm2m-097-026:8,useocpm2m-097-032:8}"
# NP defaults to total GPU count implied by HOSTS (sum of :N suffixes).
# A bare `host` (no :N) contributes 1 by mpirun's convention. Falls back to 16
# only if HOSTS can't be parsed for any positive count.
if [[ -z "${NP:-}" ]]; then
  NP=0
  IFS=, read -r -a _hostlist <<< "${HOSTS}"
  for _h in "${_hostlist[@]}"; do
    _n="${_h##*:}"
    if [[ "${_h}" == *:* && "${_n}" =~ ^[0-9]+$ ]]; then
      NP=$(( NP + _n ))
    else
      NP=$(( NP + 1 ))
    fi
  done
  (( NP > 0 )) || NP=16
fi
: "${DTYPE:=bfloat16}"
: "${MIN_BYTES:=8}"
: "${MAX_BYTES:=1G}"
: "${STEP_FACTOR:=2}"
: "${WARMUP:=5}"
: "${ITERS:=20}"
: "${GPUS_PER_THREAD:=1}"
: "${RCCL_TESTS_BIN_DIR:=${HOME}/rocm-systems/projects/rccl-tests/build}"
: "${RCCL_BUILD_DIR:=${HOME}/rccl/build/release}"
: "${MPIRUN_BIN:=${HOME}/mpich/install/bin/mpirun}"

: "${NCCL_DEBUG:=VERSION}"
: "${NCCL_DEBUG_SUBSYS:=INIT}"
: "${NCCL_IB_HCA:=mlx5_0,mlx5_2,mlx5_3,mlx5_4,mlx5_5,mlx5_7,mlx5_8,mlx5_9}"
: "${NCCL_IB_TC:=41}"
: "${NCCL_IGNORE_CPU_AFFINITY:=1}"
: "${HSA_NO_SCRATCH_RECLAIM:=1}"
: "${RCCL_IB_RDMA_CHECKSUM:=1}"
# Per-step byte-count cap for the kernel-side XOR (see RCCL_PARAM in
# src/transport/net_ib.cc). 0 (default) means "no cap" -- every eligible
# slot is checksummed, matching the historical behaviour. A positive value
# caps the XOR work to slots <= that many bytes; oversize slots publish
# NCCL_IB_CHECKSUM_NONE so the proxy/IB plugin/receiver kernel all skip
# the verify together. Useful for stress runs that want to keep checksum
# coverage on small/medium messages while bounding per-step latency on the
# 128MB / 1GB sizes at the tail of the sweep.
: "${RCCL_IB_RDMA_CHECKSUM_BYTES:=0}"
: "${RCCL_IB_RDMA_CHECKSUM_TRACE:=0}"

MPI_LIB_DIR="${HOME}/mpich/install/lib"
MPI_BIN_DIR="${HOME}/mpich/install/bin"

MPIRUN_ENV=(
  -env PATH="${MPI_BIN_DIR}:${PATH}"
  -env LD_LIBRARY_PATH="${RCCL_BUILD_DIR}:${MPI_LIB_DIR}:${LD_LIBRARY_PATH:-}"
  -env NCCL_DEBUG="${NCCL_DEBUG}"
  -env NCCL_DEBUG_SUBSYS="${NCCL_DEBUG_SUBSYS}"
  -env NCCL_IB_HCA="${NCCL_IB_HCA}"
  -env NCCL_IB_TC="${NCCL_IB_TC}"
  -env NCCL_IGNORE_CPU_AFFINITY="${NCCL_IGNORE_CPU_AFFINITY}"
  -env HSA_NO_SCRATCH_RECLAIM="${HSA_NO_SCRATCH_RECLAIM}"
  -env RCCL_IB_RDMA_CHECKSUM="${RCCL_IB_RDMA_CHECKSUM}"
  -env RCCL_IB_RDMA_CHECKSUM_BYTES="${RCCL_IB_RDMA_CHECKSUM_BYTES}"
  -env RCCL_IB_RDMA_CHECKSUM_TRACE="${RCCL_IB_RDMA_CHECKSUM_TRACE}"
)

# ---- collective list --------------------------------------------------------
# Default: every *_perf binary in the rccl-tests build dir. The user can pass
# ONLY="bin1 bin2" to restrict, or SKIP="bin3" to drop noisy ones.
if [[ -n "${ONLY:-}" ]]; then
  read -r -a COLLECTIVES <<< "${ONLY}"
else
  COLLECTIVES=()
  shopt -s nullglob
  for f in "${RCCL_TESTS_BIN_DIR}"/*_perf; do
    [[ -x "${f}" ]] || continue
    COLLECTIVES+=( "$(basename "${f}")" )
  done
  shopt -u nullglob
fi
if [[ -n "${SKIP:-}" ]]; then
  read -r -a SKIP_ARR <<< "${SKIP}"
  KEEP=()
  for c in "${COLLECTIVES[@]}"; do
    drop=0
    for s in "${SKIP_ARR[@]}"; do [[ "${c}" == "${s}" ]] && drop=1; done
    (( drop == 0 )) && KEEP+=( "${c}" )
  done
  COLLECTIVES=( "${KEEP[@]}" )
fi

if (( ${#COLLECTIVES[@]} == 0 )); then
  echo "[run_csum_stress] no collectives selected (RCCL_TESTS_BIN_DIR=${RCCL_TESTS_BIN_DIR})" >&2
  exit 2
fi

# ---- header ----------------------------------------------------------------
SUMMARY="${LOG_DIR}/SUMMARY.txt"
{
  echo "# run_csum_stress @ $(date -u +%Y-%m-%dT%H:%M:%SZ)"
  echo "# log_dir         : ${LOG_DIR}"
  echo "# hosts           : ${HOSTS}  (np=${NP})"
  echo "# rccl_build_dir  : ${RCCL_BUILD_DIR}"
  echo "# rccl_tests_bin  : ${RCCL_TESTS_BIN_DIR}"
  echo "# dtype/size sweep: -d ${DTYPE} -b ${MIN_BYTES} -e ${MAX_BYTES} -f ${STEP_FACTOR}"
  echo "# iters           : -w ${WARMUP} -n ${ITERS} -g ${GPUS_PER_THREAD}"
  echo "# checksum gate   : RCCL_IB_RDMA_CHECKSUM=${RCCL_IB_RDMA_CHECKSUM} RCCL_IB_RDMA_CHECKSUM_BYTES=${RCCL_IB_RDMA_CHECKSUM_BYTES}"
  echo "# collectives     : ${COLLECTIVES[*]}"
  echo "#"
  printf "# %-22s %6s %8s %12s %8s %4s\n" \
    "collective" "exit" "#wrong" "csum_mm" "warn" "ok?"
} | tee "${SUMMARY}"

# ---- main loop -------------------------------------------------------------
overall_fail=0
for col in "${COLLECTIVES[@]}"; do
  bin="${RCCL_TESTS_BIN_DIR}/${col}"
  log="${LOG_DIR}/${col}.log"
  if [[ ! -x "${bin}" ]]; then
    printf "  %-22s %6s %8s %12s %8s %4s\n" \
      "${col}" "SKIP" "-" "-" "-" "miss" | tee -a "${SUMMARY}"
    continue
  fi
  echo "[run_csum_stress] === ${col} ===" >&2
  set +e
  "${MPIRUN_BIN}" -np "${NP}" \
    --hosts "${HOSTS}" \
    --bind-to numa \
    "${MPIRUN_ENV[@]}" \
    "${bin}" \
      -b "${MIN_BYTES}" -e "${MAX_BYTES}" -f "${STEP_FACTOR}" \
      -g "${GPUS_PER_THREAD}" -d "${DTYPE}" \
      -w "${WARMUP}" -n "${ITERS}" \
      -c 1 \
    > "${log}" 2>&1
  rc=$?
  set -e

  # Sum the two #wrong columns rccl-tests emits per size row. The data rows
  # start with leading whitespace + digits (size in bytes); the #wrong column
  # is the 13th whitespace-separated field for the out-of-place report and
  # the last field for the in-place report. awk picks both.
  wrong=$(awk '
    /^[[:space:]]*[0-9]+[[:space:]]/ {
      # field 9 = oop_wrong, field 13 = ip_wrong in rccl-tests output
      if (NF >= 13) { w += $9 + $13 } else if (NF >= 9) { w += $9 }
    }
    END { print w+0 }
  ' "${log}")
  # grep -c exits 1 on zero matches even though it prints "0", so swallow the
  # status with `|| true` instead of `|| echo 0` (which would prepend a stray
  # "0" line and break the arithmetic / printf %d below).
  csum_mm=$(grep -c "RCCL: net recv csum mismatch" "${log}" 2>/dev/null || true)
  warn=$(grep -cE "NCCL (WARN|ERROR)" "${log}" 2>/dev/null || true)
  csum_mm=${csum_mm:-0}
  warn=${warn:-0}

  status="OK"
  if (( rc != 0 || wrong != 0 || csum_mm != 0 )); then
    status="FAIL"
    overall_fail=1
  fi
  printf "  %-22s %6d %8d %12d %8d %4s\n" \
    "${col}" "${rc}" "${wrong}" "${csum_mm}" "${warn}" "${status}" \
    | tee -a "${SUMMARY}"
done

echo "# done -> ${LOG_DIR}" | tee -a "${SUMMARY}"
echo "# overall: $([[ ${overall_fail} -eq 0 ]] && echo PASS || echo FAIL)" \
  | tee -a "${SUMMARY}"

exit "${overall_fail}"
