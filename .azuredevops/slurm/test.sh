#!/bin/bash
#SBATCH --job-name=rccl-test
#SBATCH --output=rccl-test-%j.out
#SBATCH --error=rccl-test-%j.err
#SBATCH --time=60
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --partition=gt

source /etc/profile.d/lmod.sh
module load rocm/6.4.0
module list
cd "${SLURM_SUBMIT_DIR:-$PWD}/build/test"
./rccl-UnitTests --gtest_output=xml:./test_output.xml --gtest_color=yes
