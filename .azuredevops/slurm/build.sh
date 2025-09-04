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
module load rocm/6.4.1

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

# Define GPU target
export GPU_TARGETS="gfx942"

cd "${SLURM_SUBMIT_DIR:-$PWD}"

if [ "$ENABLE_COVERAGE" = "true" ]; then
    echo "Coverage build enabled"
    CXX=${ROCM_PATH}/bin/hipcc cmake --trace-expand \
                               -DCMAKE_PREFIX_PATH="${ROCM_PATH}/llvm;${ROCM_PATH};${ROCM_PATH}/share/rocm/cmake/" \
                               -DHIP_COMPILER=clang \
                               -DCMAKE_CXX_FLAGS="-Xarch_host -fprofile-instr-generate -Xarch_host -fcoverage-mapping" \
                               -DCMAKE_SHARED_LINKER_FLAGS="-fprofile-generate" \
                               -DCMAKE_EXE_LINKER_FLAGS="-fprofile-generate" \
                               -DCMAKE_SHARED_LINKER_FLAGS_INIT="-Wl,--enable-new-dtags,--build-id=sha1,--rpath,$ORIGIN" \
                               -DCMAKE_EXE_LINKER_FLAGS_INIT="-Wl,--enable-new-dtags,--build-id=sha1,--rpath,$ORIGIN/../lib" \
                               -DCMAKE_VERBOSE_MAKEFILE=ON \
                               -DCMAKE_FIND_DEBUG_MODE=ON \
                               -DCMAKE_BUILD_TYPE=RelWithDebInfo \
                               -DCMAKE_INSTALL_RPATH_USE_LINK_PATH=FALSE \
                               -DCMAKE_INSTALL_PREFIX="${BINARIES_DIR}" \
                               -DBUILD_FILE_REORG_BACKWARD_COMPATIBILITY=OFF \
                               -DROCM_SYMLINK_LIBS=OFF \
                               -DROCM_DISABLE_LDCONFIG=ON \
                               -DCPACK_SET_DESTDIR=OFF \
                               -DCPACK_RPM_PACKAGE_RELOCATABLE=ON \
                               -DROCM_PATH="${ROCM_PATH}" \
                               -DAMDGPU_TARGETS=${GPU_TARGETS} \
                               -DGPU_TARGETS=${GPU_TARGETS} \
                               -DCMAKE_HIP_ARCHITECTURES=${GPU_TARGETS} \
                               -DCPACK_DEBIAN_DEBUGINFO_PACKAGE=TRUE \
                               -DCPACK_RPM_DEBUGINFO_PACKAGE=TRUE \
                               -DCPACK_RPM_INSTALL_WITH_EXEC=TRUE \
                               -DCMAKE_CXX_FLAGS_RELWITHDEBINFO="-Xarch_host -fprofile-instr-generate -Xarch_host -fcoverage-mapping -O3 -g -DNDEBUG" \
                               -DCPACK_GENERATOR=DEB \
                               -DROCM_PATCH_VERSION="${ROCM_PATCH_VERSION}" \
                               -DBUILD_ADDRESS_SANITIZER=OFF \
                               -DBUILD_TESTS=OFF \
                               ..
    make -j${PROC} package 2>&1 | tee -a ../rccl_build_log.txt
    make install
    exit 0
else
  cmake -G Ninja -DCMAKE_INSTALL_PREFIX="$BINARIES_DIR" -DCMAKE_BUILD_TYPE=Release -DGPU_TARGETS=${GPU_TARGETS} -DBUILD_TESTS=ON -DROCM_PATH="$ROCM_PATH" ..
fi

## Building RCCL
cmake --build .
cmake --build . --target install


cd "${SLURM_SUBMIT_DIR:-$PWD}"
## Building RCCL-Tests
git clone https://github.com/ROCm/rccl-tests
cd rccl-tests
mkdir -p build
cd build
cmake -DCMAKE_PREFIX_PATH="$BINARIES_DIR;$MPI_HOME" -DUSE_MPI=ON -DCMAKE_INSTALL_PREFIX="$BINARIES_DIR" -DCMAKE_BUILD_TYPE=Release -DGPU_TARGETS=${GPU_TARGETS} -DROCM_PATH="$ROCM_PATH" ..
cmake --build .
cmake --build . --target install
