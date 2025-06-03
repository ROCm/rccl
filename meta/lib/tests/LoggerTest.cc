// (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

#include <folly/init/Init.h>
#include <gtest/gtest.h>
#include "comms/rcclx/develop/meta/lib/ScubaLogger.h"

TEST(ScubaLoggerTest, getSingleton) {
  EXPECT_NE(facebook::rcclx::ScubaLogger::getInstance(), nullptr);
}

TEST(ScubaLoggerTest, getCurrentTimestamp) {
  auto ts = facebook::rcclx::ScubaLogger::getInstance()->getCurrentTimestamp();
  EXPECT_GE(ts, 1733176553720000);
}

TEST(ScubaLoggerTest, getHostnamePid) {
  auto rs1 = facebook::rcclx::ScubaLogger::getInstance()->getHostnamePid();
  auto rs2 = facebook::rcclx::ScubaLogger::getInstance()->getHostnamePid();
  EXPECT_NE(rs1, "");
  EXPECT_NE(rs2, "");
  EXPECT_EQ(rs1, rs2) << "rs1: " << rs1 << " rs2: " << rs2;
  std::cout << rs1 << std::endl;
}

TEST(ScubaLoggerTest, logCollEnqueueTraceFunc) {
  int dummy_send = 0, dummy_recv = 0, stream = 0;
  uint64_t comm = 1111111111;
  facebook::rcclx::ScubaLogger::getInstance()->logCollEnqueueTrace(
      &dummy_send, // sendbuff
      {}, // sendCounts
      {}, // sendDispls
      &dummy_recv, // recvbuff
      {}, // recvcounts
      {}, // recvDispls
      10, // count
      1, // datatype
      1, // op
      0, // root
      3, // peer
      1, // cudaDev
      5, // globalRank
      8, // nranks
      21, // opCount
      22, // taskId
      // This unittest will be scheduled to run nightly and log to scuba as
      // well, so using a different name to avoid confusion
      "AllGatherUnitTest", // collName
      comm, // commHash
      &stream // hipStream
  );
}

TEST(ScubaLoggerTest, logCollEnqueueTracePayload) {
  int dummy_send = 0, dummy_recv = 0, stream = 0;
  uint64_t comm = 1111111111;
  facebook::rcclx::RcclCollTraceLogPayload payload = {
      .red_op = 0,
      .data_type = 1,
      .global_rank = 3,
      .cuda_dev = 6,
      .root = 2,
      .n_ranks = 10,
      .task_id = 20,
      .op_count = 30,
      .count = 40,
      .sendbuff = &dummy_send,
      .recvbuff = &dummy_recv,
      .comm_hash = comm,
      .hip_stream = &stream,
      // This unittest will be scheduled to run nightly and log to scuba as
      // well, so using a different name to avoid confusion
      .coll_name = "AllReduceUnitTest"};
  EXPECT_NE(payload.sendbuff, nullptr);
  EXPECT_NE(payload.recvbuff, nullptr);
  EXPECT_NE(payload.hip_stream, nullptr);
  facebook::rcclx::ScubaLogger::getInstance()->logCollEnqueueTrace(
      std::move(payload));
}

int main(int argc, char** argv) {
  ::testing::InitGoogleTest(&argc, argv);
  folly::Init init(&argc, &argv);
  return RUN_ALL_TESTS();
}
