// (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

#include <cuda_runtime.h>
#include <folly/init/Init.h>
#include <folly/logging/xlog.h>
#include <gtest/gtest.h>

#include "comms/rcclx/develop/meta/lib/tests/RcclxTestUtils.h"
#include "comms/utils/CudaRAII.h"
#include "rccl.h" // @manual

using namespace meta::rcclx;
using namespace meta::comms;

TEST_F(RcclxBaseTestFixture, HelloWorld) {
  int device{-1};
  cudaGetDevice(&device);
  XLOG(INFO) << fmt::format(
      "Hello from global-rank: {}, local-rank: {}, num-ranks: {}, device: {}",
      globalRank,
      localRank,
      numRanks,
      device);

  ncclComm_t comm{nullptr};
  ncclConfig_t config = NCCL_CONFIG_INITIALIZER;
  NCCL_CHECK(
      ncclCommInitRankConfig(&comm, numRanks, commId, globalRank, &config));
  XLOG(INFO) << fmt::format("rank {} init done.", globalRank);

  const size_t count{1204};
  auto sendbuff = DeviceBuffer(count * sizeof(int32_t));
  auto recvbuff = DeviceBuffer(count * sizeof(int32_t));
  auto stream = CudaStream();

  NCCL_CHECK(ncclAllReduce(
      sendbuff.get(),
      recvbuff.get(),
      count,
      ncclInt32,
      ncclSum,
      comm,
      stream.get()));
  CUDA_CHECK(cudaStreamSynchronize(stream.get()));
  XLOG(INFO) << fmt::format("rank {} allreduce done.", globalRank);

  NCCL_CHECK(ncclCommDestroy(comm));
  XLOG(INFO) << fmt::format("rank {} destroy done.", globalRank);
}

int main(int argc, char* argv[]) {
  ::testing::InitGoogleTest(&argc, argv);
  ::testing::AddGlobalTestEnvironment(new MPIEnvironmentBase);
  folly::Init init(&argc, &argv);
  return RUN_ALL_TESTS();
}
