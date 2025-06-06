#!/bin/bash
#SBATCH --job-name=rccl-test
#SBATCH --output=rccl-test-%j.out
#SBATCH --error=rccl-test-%j.err
#SBATCH --time=60
#SBATCH --partition=gt

set -e
module load rocm/6.3.0
cd "${SLURM_SUBMIT_DIR:-$PWD}/build/test"
./rccl-UnitTests --gtest_output=xml:./test_output.xml --gtest_color=yes
