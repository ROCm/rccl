#!/usr/bin/env bash
# Configure + build RCCL with the IB / socket net-checksum feature enabled
# (the default since the feature commit). The compile-time gate
# RCCL_IB_CHECKSUM_DEVICE_ENABLED is exposed at cmake-configure time as the
# ENABLE_IB_NET_CHECKSUM option (default ON). This script just nails it to
# ON explicitly so a stale CMakeCache cannot accidentally disable it.
#
# Runtime knobs (set in the environment when launching the test):
#   NCCL_DEBUG=INFO NCCL_DEBUG_SUBSYS=NET        # surface plugin NET traces
#   RCCL_IB_RDMA_CHECKSUM=1                      # (default 1) keep IB csum on
#   RCCL_IB_RDMA_CHECKSUM_TRACE=1                # OPTIONAL: per-isend TRACE
#                                                # in the IB plugin (verbose).
#
# Other CMake flags chosen for fast turnaround on this branch:
#   BUILD_LOCAL_GPU_TARGET_ONLY=ON               # only the local gfx target
#   ONLY_FUNCS="AllReduce * * * bf16|AllGather|SendRecv"
#                                                # restrict the kernel matrix
#   ENABLE_MSCCL_KERNEL=OFF / ENABLE_MSCCLPP=OFF # skip the MSCCL paths
#   CMAKE_BUILD_TYPE=Debug / TRACE=1
#
# This script wipes the build/ directory (so a struct-layout change in
# ncclComm / ncclDevComm forces a full rebuild) and then re-runs cmake +
# make from the build/ directory. Run from any cwd.

set -euo pipefail

REPO_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
BUILD_DIR="${REPO_ROOT}/build/release"

mkdir -p "${BUILD_DIR}"
cd "${BUILD_DIR}"

rm -rf -- "${BUILD_DIR:?}/"*

CXX=/opt/rocm/bin/hipcc cmake \
  -DCMAKE_PREFIX_PATH=/opt/rocm/ \
  -DBUILD_LOCAL_GPU_TARGET_ONLY=ON \
  -DONLY_FUNCS="AllReduce * * * bf16|AllGather|SendRecv" \
  -DENABLE_MSCCL_KERNEL=OFF \
  -DENABLE_MSCCLPP=OFF \
  -DENABLE_IB_NET_CHECKSUM=ON \
  -DCMAKE_BUILD_TYPE=Debug \
  -DTRACE=1 \
  ../..

make -j
