#!/bin/bash

short_id=$(hostname | cut -d'.' -f1 | cut -d'-' -f3-)
echo "Node identifier: $short_id"

source /etc/profile.d/lmod.sh
module load rocm/6.4.1

if [ "$ENABLE_COVERAGE" = "true" ]; then
    cd $RCCL_TEST_INFRA_DIR || exit
    TEST_INFRA_WORK_DIR=$BINARIES_DIR CODE_COV=1 ENABLE_MSCCLPP=0 ./run.sh -c config/"$INFRA_TEST_CONFIG".json -B -C -O --skip-build --use-slurm --slurm-time="01:00:00" --slurm-partition=gt --slurm-nodes=2
    echo "Tests have finished running, collecting coverage data..."
    cd slurm_runs_"${INFRA_TEST_CONFIG}_"* || exit

    /opt/rocm/lib/llvm/bin/llvm-profdata merge --sparse --input-files=$FILE --output=merged.profdata
    /opt/rocm/lib/llvm/bin/llvm-cov show --instr-profile="merged.profdata" --format=html --output-dir=report --project-title=RCCL_Lib_Coverage_Report --ignore-filename-regex="ext-src\/*" "${BINARIES_DIR}"/rccl/build/release/librccl.so
    exit 0
fi

cd ${PIPELINE_WORKSPACE}/TestResults
mkdir -p ${PIPELINE_WORKSPACE}/TestResults/rccl-tests_logs
export WORKDIR=${PIPELINE_WORKSPACE}/TestResults/rccl-tests_logs

