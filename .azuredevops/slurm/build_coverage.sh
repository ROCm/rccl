#!/bin/bash
#SBATCH --job-name=rccl-build
#SBATCH --output=rccl-build-%j.out
#SBATCH --error=rccl-build-%j.out
#SBATCH --time=60
#SBATCH --nodes=1
#SBATCH --exclusive
#SBATCH --partition=gt



source /etc/profile.d/lmod.sh
module load rocm/6.4.1

set -e
set -x


export WORKDIR=`pwd`

ROCM_VERSION="6.4.1"
ROCM_PATCH_VERSION=6.4.1
echo $ROCM_PATCH_VERSION
export ROCM_PATH=/opt/rocm-6.4.1
echo $ROCM_PATH

# RCCL local install dir
RCCL_INSTALL_DIR=$WORKDIR/rccl/install

# Delete the existing folder
# Set the env variables
export HIPCC_COMPILE_FLAGS_APPEND="-g -Wno-format-nonliteral -Xarch_host -fprofile-instr-generate -Xarch_host -fcoverage-mapping -parallel-jobs=16"
export HIPCC_LINK_FLAGS_APPEND="-fprofile-instr-generate -fcoverage-mapping -parallel-jobs=16"
export LLVM_PROFILE_FILE=rccl_tests_%9999m.profraw
export HSA_NO_SCRATCH_RECLAIM=1

# Clone and build RCCL

cd $WORKDIR/rccl

mkdir -p build && cd build
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
                            -DCMAKE_INSTALL_PREFIX=${RCCL_INSTALL_DIR} \
                            -DCMAKE_PACKAGING_INSTALL_PREFIX=${RCCL_INSTALL_DIR} \
                            -DBUILD_FILE_REORG_BACKWARD_COMPATIBILITY=OFF \
                            -DROCM_SYMLINK_LIBS=OFF \
                            -DCPACK_PACKAGING_INSTALL_PREFIX=${RCCL_INSTALL_DIR} \
                            -DROCM_DISABLE_LDCONFIG=ON \
                            -DCPACK_SET_DESTDIR=OFF \
                            -DCPACK_RPM_PACKAGE_RELOCATABLE=ON \
                            -DROCM_PATH=${ROCM_PATH} \
                            -DAMDGPU_TARGETS="gfx90a:xnack-;gfx90a:xnack+;gfx942:xnack-;gfx942:xnack+" \
                            -DGPU_TARGETS="gfx90a:xnack-;gfx90a:xnack+;gfx942:xnack-;gfx942:xnack+" \
                            -DCMAKE_HIP_ARCHITECTURES="gfx90a:xnack-;gfx90a:xnack+;gfx942:xnack-;gfx942:xnack+" \
                            -DCPACK_DEBIAN_DEBUGINFO_PACKAGE=TRUE \
                            -DCPACK_RPM_DEBUGINFO_PACKAGE=TRUE \
                            -DCPACK_RPM_INSTALL_WITH_EXEC=TRUE \
                            -DCMAKE_CXX_FLAGS_RELWITHDEBINFO="-Xarch_host -fprofile-instr-generate -Xarch_host -fcoverage-mapping -O3 -g -DNDEBUG" \
                            -DCPACK_GENERATOR=DEB \
                            -DROCM_PATCH_VERSION=${ROCM_PATCH_VERSION} \
                            -DBUILD_ADDRESS_SANITIZER=OFF \
                            -DBUILD_TESTS=OFF \
                            ..

make -j${PROC} package 2>&1 | tee -a ../rccl_build_log.txt
make install 2>&1 | tee -a ../rccl_build_log.txt

cd $WORKDIR


OMPMPI_UCX_LIB_DIR=/opt/ompi/4.1.5-ucx1.18.0-rocm6.4.1/lib/
MPI_RUN=/opt/ompi/4.1.5-ucx1.18.0-rocm6.4.1/bin/
MPI_INSTALL_PREFIX=/opt/ompi/4.1.5-ucx1.18.0-rocm6.4.1

export LD_LIBRARY_PATH=$OMPMPI_UCX_LIB_DIR/lib:$LD_LIBRARY_PATH
export PATH=$MPI_RUN:$PATH

git clone https://github.com/ROCm/rccl-tests
cd rccl-tests
make MPI=1 MPI_HOME=${MPI_INSTALL_PREFIX} NCCL_HOME=${RCCL_INSTALL_DIR} -j





