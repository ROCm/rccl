#!/bin/bash
#SBATCH --job-name=rccl-build
#SBATCH --output=rccl-build-%j.out
#SBATCH --error=rccl-build-%j.err
#SBATCH --time=60
#SBATCH --partition=compute

set -e
cd "${SLURM_SUBMIT_DIR:-$PWD}"
mkdir -p build
cd build
cmake -DCMAKE_BUILD_TYPE=Release -DBUILD_TESTS=ON ..
cmake --build . -- -j32
cmake --build . --target install
