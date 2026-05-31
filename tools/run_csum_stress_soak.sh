#!/usr/bin/env bash
# Thin soak wrapper around tools/run_csum_stress.sh: runs the full-collective
# stress sweep 10 times (override with CYCLES) under each
# RCCL_IB_RDMA_CHECKSUM_BYTES configuration in CAPS (default "4096 0") to
# surface intermittent net-checksum mismatches a single sweep might miss.
#
# Unlike tools/run_csum_soak.sh this never auto-selects a single-node sweeper:
# it always drives the multi-node tools/run_csum_stress.sh against that
# script's hard-coded HOSTS list (override via the HOSTS env, forwarded
# straight through). Each cycle gets its own log dir so a failing run can be
# inspected after the fact; the wrapper does NOT abort on a failing cycle --
# the point is to count how many of the N cycles per cap stay clean.
#
# Usage:
#   tools/run_csum_stress_soak.sh                 # 10 cycles each of 4096, 0
#   CYCLES=20 CAPS="4096 0 1024" tools/run_csum_stress_soak.sh
#   HOSTS=hostA:8,hostB:8 tools/run_csum_stress_soak.sh
#
# Env:
#   CYCLES=10                              # cycles per cap configuration
#   CAPS="4096 0"                          # RCCL_IB_RDMA_CHECKSUM_BYTES values
#   SOAK_DIR=~/csum_stress_soak_<utc-ts>   # parent output dir
# Any other RCCL_*/NCCL_* env (HOSTS, NP, NCCL_IB_HCA, ...) is passed straight
# through to tools/run_csum_stress.sh.

set -uo pipefail

REPO_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
SWEEP_BIN="${REPO_ROOT}/tools/run_csum_stress.sh"
: "${CYCLES:=10}"
: "${CAPS:=4096 0}"
: "${SOAK_DIR:=${HOME}/csum_stress_soak_$(date -u +%Y%m%dT%H%M%SZ)}"

if [[ ! -x "${SWEEP_BIN}" ]]; then
  echo "[run_csum_stress_soak] sweeper not found/executable: ${SWEEP_BIN}" >&2
  exit 2
fi
mkdir -p "${SOAK_DIR}"

AGG="${SOAK_DIR}/AGGREGATE.txt"
read -r -a CAP_ARR <<< "${CAPS}"

{
  echo "# run_csum_stress_soak @ $(date -u +%Y-%m-%dT%H:%M:%SZ)"
  echo "# soak_dir : ${SOAK_DIR}"
  echo "# cycles   : ${CYCLES} per cap"
  echo "# caps     : ${CAPS}"
  echo "# sweeper  : ${SWEEP_BIN}"
  echo "# hosts    : ${HOSTS:-<run_csum_stress.sh default>}"
  echo "#"
  printf "# %-6s %-7s %-6s %10s %10s %s\n" "cap" "cycle" "result" "tot_wrong" "tot_csum_mm" "log_dir"
} | tee "${AGG}"

overall_fail=0
declare -A pass_count
declare -A mm_total

for cap in "${CAP_ARR[@]}"; do
  pass_count[$cap]=0
  mm_total[$cap]=0
  mkdir -p "${SOAK_DIR}/cap${cap}"
  for (( c=1; c<=CYCLES; c++ )); do
    cyc=$(printf "%02d" "${c}")
    log_dir="${SOAK_DIR}/cap${cap}/cycle${cyc}"
    # NOTE: do NOT pre-create ${log_dir}. run_csum_stress.sh bumps the path
    # with a _N suffix if it already exists, which would silently route logs to
    # cycleNN_1 and leave the aggregator (which globs ${log_dir}/*.log) reading
    # an empty cycleNN dir.

    RCCL_IB_RDMA_CHECKSUM_BYTES="${cap}" \
      "${SWEEP_BIN}" "${log_dir}" >/dev/null 2>&1
    rc=$?

    # Aggregate across this cycle's per-collective logs.
    tot_wrong=0
    tot_mm=0
    if compgen -G "${log_dir}/*.log" >/dev/null; then
      tot_mm=$(grep -ch "net recv csum mismatch" "${log_dir}"/*.log 2>/dev/null | awk '{s+=$1} END{print s+0}')
      tot_wrong=$(awk '
        /^[[:space:]]*[0-9]+[[:space:]]/ {
          if (NF >= 13) { w += $9 + $13 } else if (NF >= 9) { w += $9 }
        }
        END { print w+0 }
      ' "${log_dir}"/*.log)
    fi

    result="PASS"
    if (( rc != 0 || tot_wrong != 0 || tot_mm != 0 )); then
      result="FAIL"
      overall_fail=1
    else
      pass_count[$cap]=$(( pass_count[$cap] + 1 ))
    fi
    mm_total[$cap]=$(( mm_total[$cap] + tot_mm ))

    printf "  %-6s %-7s %-6s %10d %10d %s\n" \
      "${cap}" "${cyc}/${CYCLES}" "${result}" "${tot_wrong}" "${tot_mm}" "${log_dir}" \
      | tee -a "${AGG}"
  done
done

{
  echo "#"
  echo "# ==== per-cap totals ===="
  for cap in "${CAP_ARR[@]}"; do
    printf "# cap=%-6s clean_cycles=%d/%d  total_csum_mm=%d\n" \
      "${cap}" "${pass_count[$cap]}" "${CYCLES}" "${mm_total[$cap]}"
  done
  echo "# overall: $([[ ${overall_fail} -eq 0 ]] && echo PASS || echo FAIL)"
  echo "# done -> ${SOAK_DIR}"
} | tee -a "${AGG}"

exit "${overall_fail}"
