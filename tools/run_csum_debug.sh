#!/usr/bin/env bash
# Run the cross-node ring AllReduce that exercises the IB net-checksum path
# with the debug-build RCCL from ~/rccl/build/. This is the companion to
# tools/build_csum_debug.sh: build_csum_debug.sh produces the binary,
# this script exercises it.
#
# Usage:
#   tools/run_csum_debug.sh                       # auto-bumps to ~/ar_op_<N>.log
#   tools/run_csum_debug.sh ~/ar_op_5.log         # custom log path (no bump)
#   LOG_PATH=~/foo.log tools/run_csum_debug.sh    # same, via env (no bump)
#
# Auto-bump rule (default, no argument and no LOG_PATH env):
#   scan ${HOME} for files matching ar_op_<digits>.log, pick the highest
#   <digits> + 1, and write to ~/ar_op_<that>.log. Starts at 1 if none.
#
# RCCL_IB_RDMA_CHECKSUM_TRACE is left at 0 by default. Flip to 1 if you
# need the IB plugin's per-isend TRACE for wire-level visibility on a
# suspected isend.

set -euo pipefail

if [[ -n "${1:-}" ]]; then
  LOG_PATH="$1"
elif [[ -n "${LOG_PATH:-}" ]]; then
  :  # honour pre-set LOG_PATH
else
  next=1
  shopt -s nullglob
  for f in "${HOME}"/ar_op_[0-9]*.log; do
    base="${f##*/ar_op_}"
    num="${base%.log}"
    if [[ "${num}" =~ ^[0-9]+$ ]] && (( num + 1 > next )); then
      next=$(( num + 1 ))
    fi
  done
  shopt -u nullglob
  LOG_PATH="${HOME}/ar_op_${next}.log"
fi

# Derive HOSTS / NP from SLURM env when available (sbatch/salloc/srun), so the
# script "just works" inside a SLURM allocation without editing the host list.
# Explicit HOSTS / NP env (or values inherited from the caller) always win.
#   HOSTS  : --hosts arg for mpirun ("host1:N,host2:N,...")
#   NP     : -np arg for mpirun (total ranks = sum of the :N in HOSTS)
# SLURM mapping:
#   nodelist  <- SLURM_JOB_NODELIST | SLURM_NODELIST, expanded via `scontrol`
#   per_node  <- SLURM_GPUS_ON_NODE | SLURM_GPUS_PER_NODE | SLURM_NTASKS_PER_NODE | 8
if [[ -z "${HOSTS:-}" ]]; then
  _nodelist="${SLURM_JOB_NODELIST:-${SLURM_NODELIST:-}}"
  if [[ -n "${_nodelist}" ]] && command -v scontrol >/dev/null 2>&1; then
    _per_node="${SLURM_GPUS_ON_NODE:-${SLURM_GPUS_PER_NODE:-${SLURM_NTASKS_PER_NODE:-8}}}"
    HOSTS=$(scontrol show hostname "${_nodelist}" | sed "s/$/:${_per_node}/" | paste -sd, -)
  fi
fi
: "${HOSTS:=useocpm2m-097-026:8,useocpm2m-097-032:8}"
# NP defaults to the total GPU count implied by HOSTS (sum of the :N suffixes).
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

: "${NCCL_DEBUG:=VERSION}"
: "${NCCL_DEBUG_SUBSYS:=NET}"
: "${NCCL_IB_HCA:=mlx5_0,mlx5_2,mlx5_3,mlx5_4,mlx5_5,mlx5_7,mlx5_8,mlx5_9}"
: "${NCCL_IB_TC:=41}"
: "${NCCL_IGNORE_CPU_AFFINITY:=1}"
: "${HSA_NO_SCRATCH_RECLAIM:=1}"
: "${NCCL_PROTO:=Simple}"
: "${NCCL_ALGO:=Ring}"
: "${RCCL_IB_RDMA_CHECKSUM_TRACE:=0}"

MPI_LIB_DIR="${HOME}/mpich/install/lib"
MPI_BIN_DIR="${HOME}/mpich/install/bin"

MPIRUN_ENV=(
  -env PATH="${MPI_BIN_DIR}:${PATH}"
  -env LD_LIBRARY_PATH="${HOME}/rccl/build/release:${MPI_LIB_DIR}:${LD_LIBRARY_PATH:-}"
  -env NCCL_DEBUG="${NCCL_DEBUG}"
  -env NCCL_DEBUG_SUBSYS="${NCCL_DEBUG_SUBSYS}"
  -env NCCL_IB_HCA="${NCCL_IB_HCA}"
  -env NCCL_IB_TC="${NCCL_IB_TC}"
  -env NCCL_IGNORE_CPU_AFFINITY="${NCCL_IGNORE_CPU_AFFINITY}"
  -env HSA_NO_SCRATCH_RECLAIM="${HSA_NO_SCRATCH_RECLAIM}"
  -env NCCL_PROTO="${NCCL_PROTO}"
  -env NCCL_ALGO="${NCCL_ALGO}"
  -env RCCL_IB_RDMA_CHECKSUM_TRACE="${RCCL_IB_RDMA_CHECKSUM_TRACE}"
)

echo "[run_csum_debug] LOG_PATH=${LOG_PATH}"
echo "[run_csum_debug] HOSTS=${HOSTS} NP=${NP}"

MPIRUN_BIN="${MPIRUN_BIN:-${HOME}/mpich/install/bin/mpirun}"

"${MPIRUN_BIN}" -np "${NP}" \
  --hosts "${HOSTS}" \
  --bind-to numa \
  "${MPIRUN_ENV[@]}" \
  "${HOME}/rocm-systems/projects/rccl-tests/build/all_reduce_perf" \
    -b 64M -e 64M -f 2 -g 1 -d bfloat16 -w 0 -n 1 -A 1 -O 1 \
  | tee "${LOG_PATH}"
