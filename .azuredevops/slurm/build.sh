#!/bin/bash
#SBATCH --job-name=rccl-build
#SBATCH --output=rccl-build-%j.out
#SBATCH --error=rccl-build-%j.out
#SBATCH --time=60
#SBATCH --nodes=1
#SBATCH --exclusive
#SBATCH --partition=gt

short_id=$(hostname | cut -d'.' -f1 | cut -d'-' -f3-)
echo "Node identifier: $short_id"

source /etc/profile.d/lmod.sh
module load rocm/6.4.0

# Setup local binary path
export PATH="$HOME/.local/bin:$PATH"
mkdir -p "$HOME/.local/bin"

# Install Ninja if not already available
if ! command -v ninja &>/dev/null; then
  echo "Ninja not found. Installing locally..."
  wget -q https://github.com/ninja-build/ninja/releases/download/v1.11.1/ninja-linux.zip -O /tmp/ninja.zip
  unzip -q /tmp/ninja.zip -d "$HOME/.local/bin"
  chmod +x "$HOME/.local/bin/ninja"
fi

echo "Using Ninja at: $(which ninja)"
ninja --version

cd "${SLURM_SUBMIT_DIR:-$PWD}"
mkdir -p build
cd build
cmake -G Ninja -DCMAKE_INSTALL_PREFIX="$BINARIES_DIR" -DCMAKE_BUILD_TYPE=Release -DGPU_TARGETS=gfx942 -DBUILD_TESTS=ON -DROCM_PATH="$ROCM_PATH" ..
cmake --build .
cmake --build . --target install
