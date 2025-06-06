#!/bin/bash
#SBATCH --job-name=rccl-build
#SBATCH --output=rccl-build-%j.out
#SBATCH --error=rccl-build-%j.err
#SBATCH --time=60
#SBATCH --partition=gt

set -e
source /etc/profile.d/modules.sh
module load rocm/6.3.0
cd "${SLURM_SUBMIT_DIR:-$PWD}"
mkdir -p build
cd build
cmake -DCMAKE_BUILD_TYPE=Release -DBUILD_TESTS=ON ..
cmake --build . -- -j32
cmake --build . --target install
