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

// Use Fusekernel API, but since buffer is nullptr, expect plain all reduce
TEST_F(RcclxBaseTestFixture, FusekernelTestWithoutAccBuf) {
  int device{-1};
  CUDA_CHECK(cudaGetDevice(&device));
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

  const size_t count{2};
  auto sendbuff = DeviceBuffer(count * sizeof(int32_t));
  auto recvbuff = DeviceBuffer(count * sizeof(int32_t));
  auto accbuff = DeviceBuffer(count * sizeof(int32_t));
  auto sendptr = reinterpret_cast<int32_t*>(sendbuff.get());
  *sendptr = globalRank;
  *(sendptr + 1) = 5;
  auto stream = CudaStream();

  NCCL_CHECK(ncclAllReduceWithBias(
      sendbuff.get(),
      recvbuff.get(),
      count,
      ncclInt32,
      ncclSum,
      comm,
      stream.get(),
      nullptr));
  CUDA_CHECK(cudaStreamSynchronize(stream.get()));
  auto recvptr = reinterpret_cast<int32_t*>(recvbuff.get());
  XLOG(INFO) << fmt::format("rank {} allreduce done", globalRank);
  EXPECT_EQ(*recvptr, 28);
  EXPECT_EQ(*(recvptr + 1), 40);

  NCCL_CHECK(ncclCommDestroy(comm));
  XLOG(INFO) << fmt::format("rank {} destroy done.", globalRank);
}

TEST_F(RcclxBaseTestFixture, FusekernelTestWithAccBufSum) {
  int device{-1};
  CUDA_CHECK(cudaGetDevice(&device));
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

  const size_t count{2};
  auto sendbuff = DeviceBuffer(count * sizeof(int32_t));
  auto recvbuff = DeviceBuffer(count * sizeof(int32_t));
  auto accbuff = DeviceBuffer(count * sizeof(int32_t));
  auto sendptr = reinterpret_cast<int32_t*>(sendbuff.get());
  *sendptr = globalRank;
  *(sendptr + 1) = 5;
  auto accptr = reinterpret_cast<int32_t*>(accbuff.get());
  *accptr = globalRank + 1;
  *(accptr + 1) = 10;
  auto stream = CudaStream();

  NCCL_CHECK(ncclAllReduceWithBias(
      sendbuff.get(),
      recvbuff.get(),
      count,
      ncclInt32,
      ncclSum,
      comm,
      stream.get(),
      accbuff.get()));
  CUDA_CHECK(cudaStreamSynchronize(stream.get()));
  auto recvptr = reinterpret_cast<int32_t*>(recvbuff.get());
  XLOG(INFO) << fmt::format("rank {} allreduce done", globalRank);
  // For first value, rank 0 has value 0, rank 1 has value 1, ...,
  // Total is 28 = (1 + 2 + ... + 7)
  // Since accbuff is provided, after reduce, it is apply
  // reduce results to acc so rank0 final value is 28 + acc value
  // (set to be globalRank + 1 for globalRank)
  EXPECT_EQ(*recvptr, 28 + globalRank + 1);
  // For second value, all has const value of 5, 8 ranks have value 40.
  // Since accbuff is provided, after reduce, it is 40 + 10 (acc value)
  EXPECT_EQ(*(recvptr + 1), 50);

  NCCL_CHECK(ncclCommDestroy(comm));
  XLOG(INFO) << fmt::format("rank {} destroy done.", globalRank);
}

TEST_F(RcclxBaseTestFixture, FusekernelTestWithAccBufMax) {
  int device{-1};
  CUDA_CHECK(cudaGetDevice(&device));
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

  const size_t count{2};
  auto sendbuff = DeviceBuffer(count * sizeof(int32_t));
  auto recvbuff = DeviceBuffer(count * sizeof(int32_t));
  auto accbuff = DeviceBuffer(count * sizeof(int32_t));
  auto sendptr = reinterpret_cast<int32_t*>(sendbuff.get());
  *sendptr = globalRank;
  *(sendptr + 1) = 5;
  auto accptr = reinterpret_cast<int32_t*>(accbuff.get());
  *accptr = globalRank + 1;
  *(accptr + 1) = 10;
  auto stream = CudaStream();

  NCCL_CHECK(ncclAllReduceWithBias(
      sendbuff.get(),
      recvbuff.get(),
      count,
      ncclInt32,
      ncclMax,
      comm,
      stream.get(),
      accbuff.get()));
  CUDA_CHECK(cudaStreamSynchronize(stream.get()));
  auto recvptr = reinterpret_cast<int32_t*>(recvbuff.get());
  XLOG(INFO) << fmt::format("rank {} allreduce done", globalRank);
  // For first value, rank 0 has value 0, rank 1 has value 1, ...,
  // Max is 7
  // Since accbuff is provided, after reduce, it is apply
  // reduce results to acc so rank0 final value is
  // max(7, acc value) (set to be globalRank + 1 for globalRank)
  EXPECT_EQ(*recvptr, max(7, globalRank + 1));
  // For second value, all has const value of 5, 8 ranks have
  // max value of 5.
  // Since accbuff is provided, after reduce, it is max(5, 10)
  // (acc value)
  EXPECT_EQ(*(recvptr + 1), 10);

  NCCL_CHECK(ncclCommDestroy(comm));
  XLOG(INFO) << fmt::format("rank {} destroy done.", globalRank);
}

int main(int argc, char* argv[]) {
  ::testing::InitGoogleTest(&argc, argv);
  ::testing::AddGlobalTestEnvironment(new MPIEnvironmentBase);
  folly::Init init(&argc, &argv);
  return RUN_ALL_TESTS();
}
