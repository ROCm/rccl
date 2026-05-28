#!/usr/bin/env bash
# Same build as tools/build_csum_debug.sh but with the RCCL net checksum
# feature compiled OUT (-DENABLE_IB_NET_CHECKSUM=OFF). Used to validate
# that the entire feature is reachable through the compile-time gate
# RCCL_IB_CHECKSUM_DEVICE_ENABLED and that a feature-off build still
# links and runs correctly.

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
  -DENABLE_IB_NET_CHECKSUM=OFF \
  -DCMAKE_BUILD_TYPE=Debug \
  -DTRACE=1 \
  ../..

make -j
