#!/usr/bin/env bash
# Single-node variant of run_csum_stress.sh for exercising the IB net-checksum
# feature inside one box. Sets both RCCL_ENABLE_INTRANET=1 (keeps NET nodes in
# the single-node topology, graph/paths.cc:916) and RCCL_P2P_NET_DISABLE=0
# (flips the rcclParamP2pNetDisable gate at init.cc:1843 so comm->useIntraNet
# gets set), which together let intra-node pairs land on IB and exercise the
# kernel-side XOR + IB IMM checksum path. Size sweep, validation, and
# per-collective summary logic are otherwise identical to run_csum_stress.sh.
#
# Usage:
#   tools/run_csum_single.sh                       # auto-pick ~/csum_single_<N>/
#   tools/run_csum_single.sh ~/my_run              # use that dir (suffixed _1, _2,
#                                                  # ... if it already exists)
#   ONLY="all_reduce_perf alltoall_perf" \
#     tools/run_csum_single.sh                     # restrict to a subset
#   SKIP="all_reduce_bias_perf" tools/run_csum_single.sh
#
# Default collective set:
#   all_reduce_perf all_gather_perf alltoall_perf alltoallv_perf
# This matches the librccl build's ONLY_FUNCS narrowing
# ("AllReduce * * * bf16|AllGather|SendRecv"): every other binary needs
# Broadcast/Reduce/ReduceScatter kernels that aren't in this librccl, so
# they'd hit "ncclDevFuncId not found" at runtime.
#
# Env overrides:
#   NP=8                                           # local GPU count
#   DTYPE=bfloat16                                 # rccl-tests -d
#   MIN_BYTES=8                                    # rccl-tests -b
#   MAX_BYTES=1G                                   # rccl-tests -e
#   STEP_FACTOR=2                                  # rccl-tests -f
#   WARMUP=5                                       # rccl-tests -w
#   ITERS=20                                       # rccl-tests -n
#   GPUS_PER_THREAD=1                              # rccl-tests -g
#   RCCL_ENABLE_INTRANET=1                         # << keep NET in the
#                                                  # single-node topology
#   RCCL_P2P_NET_DISABLE=0                         # << flip the useIntraNet
#                                                  # gate so intra-node pairs
#                                                  # actually route over NET
#   RCCL_IB_RDMA_CHECKSUM=1                        # gate kernel XOR + IB IMM csum
#   RCCL_IB_RDMA_CHECKSUM_BYTES=0                  # per-slot byte cap (0=no cap)
#   RCCL_IB_RDMA_CHECKSUM_ITERS=2                  # recv-verify passes [1,4]
#                                                  # (matches library default;
#                                                  # the 2nd pass with
#                                                  # threadfence_system detects
#                                                  # late buffer mutations)
#   RCCL_IB_RDMA_CHECKSUM_SEND_VERIFY=1            # post-completion CPU XOR
#                                                  # re-check on the send buffer
#                                                  # (matches library default;
#                                                  # set to 0 only for clean
#                                                  # bandwidth comparisons)
#   NCCL_DEBUG=INFO                                # default INFO so the
#                                                  # P2P-over-NET gate
#                                                  # decisions appear in
#                                                  # every per-collective log
#   NCCL_DEBUG_SUBSYS=INIT,GRAPH                   # INIT for the
#                                                  # "RCCL enabled / force
#                                                  # disabled same node P2P
#                                                  # over network" line, GRAPH
#                                                  # for the per-channel
#                                                  # "RING/N -> ... via NET/IB"
#                                                  # routing summary
#   RCCL_TESTS_BIN_DIR=${HOME}/rocm-systems/projects/rccl-tests/build
#   RCCL_BUILD_DIR=${HOME}/rccl/build/release      # librccl.so load path
#   MPIRUN_BIN=${HOME}/mpich/install/bin/mpirun
#
# Per-collective stdout/stderr is teed into
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

# ---- log directory selection (mirror run_csum_stress.sh's suffix rule) -----
if [[ -n "${1:-}" ]]; then
  LOG_DIR="$1"
elif [[ -n "${LOG_DIR:-}" ]]; then
  :  # honour pre-set LOG_DIR
else
  next=1
  shopt -s nullglob
  for d in "${HOME}"/csum_single_[0-9]*; do
    [[ -d "${d}" ]] || continue
    base="${d##*/csum_single_}"
    if [[ "${base}" =~ ^[0-9]+$ ]] && (( base + 1 > next )); then
      next=$(( base + 1 ))
    fi
  done
  shopt -u nullglob
  LOG_DIR="${HOME}/csum_single_${next}"
fi
if [[ -e "${LOG_DIR}" ]]; then
  n=1
  while [[ -e "${LOG_DIR}_${n}" ]]; do n=$(( n + 1 )); done
  LOG_DIR="${LOG_DIR}_${n}"
fi
mkdir -p "${LOG_DIR}"

# ---- defaults (overridable via env) ----------------------------------------
# This script is single-node by design: mpirun launches all ranks on the
# local machine, so we don't pass --hosts. SELF_HOST is informational only
# (printed in the summary).
SELF_HOST=$(hostname -s 2>/dev/null || echo "unknown")
: "${NP:=8}"
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

# Default verbosity is bumped to INFO+GRAPH (vs the multi-node stress
# script's VERSION+INIT) because the single-node use case here is to
# *debug* whether RCCL_ENABLE_INTRANET=1 actually drove intra-node pairs
# onto the NET transport. The relevant per-collective log evidence:
#   * grep for "via NET/" -- non-zero count means at least one channel
#     landed on IB and the kernel-side XOR + IB IMM checksum path is
#     being exercised. Zero means everything went P2P/XGMI and the run
#     is not testing the checksum feature at all.
#   * grep for "via P2P/" -- ratio with the NET count tells you which
#     transports the topology + algo picker chose.
# The GRAPH subsystem gives the per-channel ring/tree construction
# decisions that lead up to that transport selection.
: "${NCCL_DEBUG:=INFO}"
: "${NCCL_DEBUG_SUBSYS:=INIT,GRAPH}"
: "${NCCL_IB_HCA:=mlx5_0,mlx5_2,mlx5_3,mlx5_4,mlx5_5,mlx5_7,mlx5_8,mlx5_9}"
: "${NCCL_IB_TC:=41}"
: "${NCCL_IGNORE_CPU_AFFINITY:=1}"
: "${HSA_NO_SCRATCH_RECLAIM:=1}"
: "${RCCL_IB_RDMA_CHECKSUM:=1}"
# Per-step byte-count cap for the kernel-side XOR (see RCCL_PARAM in
# src/transport/net_ib.cc). 0 (default) means "no cap" -- every eligible
# slot is checksummed, matching the historical behaviour.
: "${RCCL_IB_RDMA_CHECKSUM_BYTES:=0}"
: "${RCCL_IB_RDMA_CHECKSUM_TRACE:=0}"
# Receive-side verify iteration count (clamped to [1, 4] in init.cc).
# 2 matches the library default; pinned here so the stress script keeps
# exercising the multi-pass path even if a future library bump moves
# the default. The second pass with __threadfence_system() between
# iters 1 and 2 catches late buffer mutations (DMA tail writes,
# peer-incoherent stores) that a single pass would miss.
: "${RCCL_IB_RDMA_CHECKSUM_ITERS:=2}"
# Send-side post-completion CPU verify. 1 matches the library default;
# pinned here for the same reason as ITERS above -- this script's
# purpose is to exercise every checksum path the library can run, so
# we lock both verify gates on regardless of any future default flip.
# Per-step OK trace lines need RCCL_IB_RDMA_CHECKSUM_TRACE=1 too;
# mismatches always WARN.
: "${RCCL_IB_RDMA_CHECKSUM_SEND_VERIFY:=1}"
# Single-node-specific NET enable. Two gates must both be flipped for
# intra-node pairs to actually land on the IB transport and exercise the
# kernel-side XOR + IB IMM checksum path:
#
#   RCCL_ENABLE_INTRANET=1   keeps the NET nodes in the single-node
#                            topology and tags RCCL_TOPO_FORCE_INTRA
#                            (graph/paths.cc:916).
#   RCCL_P2P_NET_DISABLE=0   flips the rcclParamP2pNetDisable() gate at
#                            init.cc:1843 so comm->useIntraNet gets set
#                            and the NCCL_CONN_IDX_P2P_NET connections
#                            are actually opened.
#
# With only the first knob set, RCCL keeps NET in topology but never
# routes any pair over it, defeating the purpose of this script.
: "${RCCL_ENABLE_INTRANET:=1}"
: "${RCCL_P2P_NET_DISABLE:=0}"

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
  -env RCCL_IB_RDMA_CHECKSUM_ITERS="${RCCL_IB_RDMA_CHECKSUM_ITERS}"
  -env RCCL_IB_RDMA_CHECKSUM_SEND_VERIFY="${RCCL_IB_RDMA_CHECKSUM_SEND_VERIFY}"
  -env RCCL_ENABLE_INTRANET="${RCCL_ENABLE_INTRANET}"
  -env RCCL_P2P_NET_DISABLE="${RCCL_P2P_NET_DISABLE}"
)

# ---- collective list --------------------------------------------------------
# Default to the subset that has kernels in this librccl build. The
# build's ONLY_FUNCS narrowing
#   AllReduce * * * bf16 | AllGather | SendRecv
# leaves Broadcast/Reduce/ReduceScatter rows in the dispatch table as
# {key, -1}, which crashes rccl-tests at first iteration with
# "ncclDevFuncId not found". AllToAll(v) are built on top of SendRecv
# primitives, so they work even though there's no dedicated "AllToAll"
# entry in ONLY_FUNCS.
if [[ -n "${ONLY:-}" ]]; then
  read -r -a COLLECTIVES <<< "${ONLY}"
else
  COLLECTIVES=( all_reduce_perf all_gather_perf alltoall_perf alltoallv_perf )
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
  echo "[run_csum_single] no collectives selected (RCCL_TESTS_BIN_DIR=${RCCL_TESTS_BIN_DIR})" >&2
  exit 2
fi

# ---- header ----------------------------------------------------------------
SUMMARY="${LOG_DIR}/SUMMARY.txt"
{
  echo "# run_csum_single @ $(date -u +%Y-%m-%dT%H:%M:%SZ)"
  echo "# log_dir         : ${LOG_DIR}"
  echo "# host            : ${SELF_HOST}  (np=${NP}, single-node)"
  echo "# rccl_build_dir  : ${RCCL_BUILD_DIR}"
  echo "# rccl_tests_bin  : ${RCCL_TESTS_BIN_DIR}"
  echo "# dtype/size sweep: -d ${DTYPE} -b ${MIN_BYTES} -e ${MAX_BYTES} -f ${STEP_FACTOR}"
  echo "# iters           : -w ${WARMUP} -n ${ITERS} -g ${GPUS_PER_THREAD}"
  echo "# checksum gate   : RCCL_IB_RDMA_CHECKSUM=${RCCL_IB_RDMA_CHECKSUM} RCCL_IB_RDMA_CHECKSUM_BYTES=${RCCL_IB_RDMA_CHECKSUM_BYTES} RCCL_IB_RDMA_CHECKSUM_ITERS=${RCCL_IB_RDMA_CHECKSUM_ITERS} RCCL_IB_RDMA_CHECKSUM_SEND_VERIFY=${RCCL_IB_RDMA_CHECKSUM_SEND_VERIFY}"
  echo "# intranet gate   : RCCL_ENABLE_INTRANET=${RCCL_ENABLE_INTRANET} RCCL_P2P_NET_DISABLE=${RCCL_P2P_NET_DISABLE}"
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
  echo "[run_csum_single] === ${col} ===" >&2
  set +e
  "${MPIRUN_BIN}" -np "${NP}" \
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

  wrong=$(awk '
    /^[[:space:]]*[0-9]+[[:space:]]/ {
      if (NF >= 13) { w += $9 + $13 } else if (NF >= 9) { w += $9 }
    }
    END { print w+0 }
  ' "${log}")
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
