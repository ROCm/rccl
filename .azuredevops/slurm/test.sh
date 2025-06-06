#!/bin/bash
#SBATCH --job-name=rccl-test
#SBATCH --output=rccl-test-%j.out
#SBATCH --error=rccl-test-%j.err
#SBATCH --time=60
#SBATCH --partition=gt

set -e
if [[ -n "$MODULES_INIT_SCRIPT" && -f "$MODULES_INIT_SCRIPT" ]]; then
  source "$MODULES_INIT_SCRIPT"
  module load rocm/6.3.0
else
  echo "ERROR: Modules init script '$MODULES_INIT_SCRIPT' not found in job environment."
  exit 1
fi
cd "${SLURM_SUBMIT_DIR:-$PWD}/build/test"
./rccl-UnitTests --gtest_output=xml:./test_output.xml --gtest_color=yes
