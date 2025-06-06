#!/bin/bash
#SBATCH --job-name=rccl-build
#SBATCH --output=rccl-build-%j.out
#SBATCH --error=rccl-build-%j.err
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
cd "${SLURM_SUBMIT_DIR:-$PWD}"
mkdir -p build
cd build
cmake -DCMAKE_BUILD_TYPE=Release -DBUILD_TESTS=ON ..
cmake --build . -- -j32
cmake --build . --target install
