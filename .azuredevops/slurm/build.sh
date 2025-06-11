#!/bin/bash
#SBATCH --job-name=rccl-build
#SBATCH --output=rccl-build-%j.out
#SBATCH --error=rccl-build-%j.err
#SBATCH --time=60
#SBATCH --partition=gt

module list
cd "${SLURM_SUBMIT_DIR:-$PWD}"
mkdir -p build
cd build
cmake -DCMAKE_BUILD_TYPE=Release -DBUILD_TESTS=ON ..
cmake --build . -- -j $SLURM_CPUS_ON_NODE
cmake --build . --target install
