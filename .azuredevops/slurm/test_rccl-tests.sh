#!/bin/bash
#SBATCH --job-name=rccl-tests
#SBATCH --output=%x-%j.out
#SBATCH --error=%x-%j.out
#SBATCH --time=72:00:00
#SBATCH --nodes=2
#SBATCH --exclusive
#SBATCH --partition=gt

short_id=$(hostname | cut -d'.' -f1 | cut -d'-' -f3-)
echo "Node identifier: $short_id"

source /etc/profile.d/lmod.sh
module load rocm/6.4.1

if [ "$ENABLE_COVERAGE" = "true" ]; then
    cd $RCCL_TEST_INFRA_DIR || exit
    TEST_INFRA_WORK_DIR=$BINARIES_DIR CODE_COV=1 ENABLE_MSCCLPP=0 ./run.sh -c config/"$INFRA_TEST_CONFIG".json -B -C -O --skip-build --use-slurm --slurm-time="72:00:00" --slurm-partition=gt --slurm-nodes=2
    cd slurm_runs_"${INFRA_TEST_CONFIG}_"* || exit

    FILE="rawprofiles.list"
    MAX_WAIT_SECS=259200 # 72 hours, these tests can take a long time
    WAIT_SECS=0
    echo "Waiting for ${FILE} (timeout: ${MAX_WAIT_SECS}s)"

    while [ ! -e "${FILE}" ]; do
      sleep 5
      WAIT_SECS=$((WAIT_SECS + 5))
      if [ "$WAIT_SECS" -ge "$MAX_WAIT_SECS" ]; then
          echo "Timed out waiting for ${FILE}"
          exit 1
      fi
    done
    echo "File ${FILE} found, now building coverage report"
    /opt/rocm/lib/llvm/bin/llvm-profdata merge --sparse --input-files=$FILE --output=merged.profdata
    /opt/rocm/lib/llvm/bin/llvm-cov show --instr-profile="merged.profdata" --format=html --output-dir=report --project-title=RCCL_Lib_Coverage_Report --ignore-filename-regex="ext-src\/*" "${BINARIES_DIR}"/rccl/build/release/librccl.so
    exit 0
fi

cd ${PIPELINE_WORKSPACE}/TestResults
mkdir -p ${PIPELINE_WORKSPACE}/TestResults/rccl-tests_logs
export WORKDIR=${PIPELINE_WORKSPACE}/TestResults/rccl-tests_logs

export PATH="$BINARIES_DIR/bin:$PATH"
export LD_LIBRARY_PATH="$BINARIES_DIR/lib:$LD_LIBRARY_PATH"

## create hostlist
nodelist=($(scontrol show hostnames))
echo "SLURM nodes:"
echo ${nodelist[@]}
echo ""

hosts_8ppn=()
for node in "${nodelist[@]}"
do
    hosts_8ppn+=("${node}:8")
done
echo ${hosts_8ppn[@]}

## Run multi- and single-node RCCL-Tests
for n in 2 1
do
    total=$((n*8))
    h_8ppn=`echo ${hosts_8ppn[@]:0:${n}} | tr ' ' ','`

    for coll in all_reduce all_gather reduce_scatter alltoall alltoallv broadcast gather reduce scatter sendrecv
    do
        for dtype in float bfloat16 half fp8_e5m2
        do
            out_filename="${WORKDIR}/rccl-tests_${coll}_1KB-16GB_nodes${n}_gpus${total}_${dtype}.log"
            cmd="${MPI_HOME}/bin/mpirun -np ${total} --host ${h_8ppn} -mca oob_tcp_if_exclude docker,lo -mca btl_tcp_if_exclude docker,lo -x HSA_ENABLE_IPC_MODE_LEGACY=1 -x HIP_FORCE_DEV_KERNARG=1 -mca pml ob1 -mca btl ^openib -x PATH -x LD_LIBRARY_PATH -x NCCL_DEBUG=VERSION -x NCCL_IB_HCA=bnxt_re0,bnxt_re1,bnxt_re2,bnxt_re3,bnxt_re4,bnxt_re5,bnxt_re6,bnxt_re7 -x NCCL_IGNORE_CPU_AFFINITY=1 -x HSA_NO_SCRATCH_RECLAIM=1 ${BINARIES_DIR}/bin/${coll}_perf -b 1K -e 16G -f 2 -g 1 -n 100 -w 50 -d ${dtype} -Z json -x ${WORKDIR}/rccl-tests_${coll}_nodes${n}_gpus${total}_${dtype}.json"

            echo "Running ${coll}" 2>&1 | tee ${out_filename}
            echo "Run cmd: ${cmd}" 2>&1 | tee -a ${out_filename}
            eval ${cmd} 2>&1 | tee -a ${out_filename}

            sleep 2
        done
    done
done

## To add
### Summarize results
### Convert to junit
