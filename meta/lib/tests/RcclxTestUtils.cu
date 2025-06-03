// (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

#include "comms/rcclx/develop/meta/lib/tests/RcclxTestUtils.h"

namespace meta::rcclx {

// RcclxBaseTestFixture
void RcclxBaseTestFixture::SetUp() {
  meta::comms::MpiBaseTestFixture::SetUp();

  // set default logging to WARN
  setenv("NCCL_DEBUG", "WARN", 0);

  // disable MSCCLPP
  setenv("RCCL_MSCCLPP_ENABLE", "0", 0);

  CUDA_CHECK(cudaSetDevice(localRank));

  // broadcast commId
  if (globalRank == 0) {
    NCCL_CHECK(ncclGetUniqueId(&commId));
  }

  MPI_Bcast(&commId, sizeof(commId), MPI_BYTE, 0, MPI_COMM_WORLD);
}

void RcclxBaseTestFixture::TearDown() {
  meta::comms::MpiBaseTestFixture::TearDown();
}

} // namespace meta::rcclx
