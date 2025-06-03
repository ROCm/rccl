// (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

#include "comms/rcclx/develop/meta/lib/CollTraceUtils.h"
#include "comms/rcclx/develop/meta/lib/EventQueue.h"

#include <gtest/gtest.h>
#include <thread>

using meta::colltrace::CollStats;
using meta::colltrace::CollTraceInfo;
using meta::colltrace::EventQueue;
using meta::colltrace::getSizeMb;

TEST(CollTraceUtilsTest, aggregateResultsTest) {
  std::deque<std::unique_ptr<CollTraceInfo>> results;
  // Host has 2 ranks, Ring buffer has 3 records
  results.emplace_back(
      std::make_unique<CollTraceInfo>(123, "allReduce", "float32", 10));
  results.emplace_back(
      std::make_unique<CollTraceInfo>(127, "allReduce", "int64", 20));
  results.emplace_back(
      std::make_unique<CollTraceInfo>(200, "allGather", "float32", 50));

  std::vector<float> latencyAllGather = {10, 20, 50, 15, 21, 45};
  auto stats = aggregateResults(
      results,
      latencyAllGather,
      2 /* ranks per host */,
      3 /* element per rank */);
  EXPECT_EQ(3, stats.size());
  std::vector<CollStats> expected = {
      CollStats{
          .collId = 123,
          .percent = 50,
          .minLatencyUs = 10000,
          .maxLatencyUs = 15000,
          .opName = "allReduce",
          .dataType = "int64"},
      CollStats{
          .collId = 127,
          .percent = 5,
          .minLatencyUs = 20000,
          .maxLatencyUs = 21000,
          .opName = "allReduce",
          .dataType = "int64"},
      CollStats{
          .collId = 200,
          .percent = 11,
          .minLatencyUs = 45000,
          .maxLatencyUs = 50000,
          .opName = "allReduce",
          .dataType = "int64"}};

  for (int i = 0; i < 3; i++) {
    EXPECT_EQ(stats[i].collId, expected[i].collId);
    EXPECT_EQ(stats[i].percent, expected[i].percent);
    EXPECT_EQ(stats[i].minLatencyUs, expected[i].minLatencyUs);
    EXPECT_EQ(stats[i].maxLatencyUs, expected[i].maxLatencyUs);
  }
}

TEST(CollTraceUtilsTest, EventQueueOperationTest) {
  EventQueue<int> q;
  q.push(std::make_unique<int>(5));
  q.push(std::make_unique<int>(100));
  auto res1 = q.waitPop();
  EXPECT_EQ(*res1, 5);
  auto res2 = q.waitPop();
  EXPECT_EQ(*res2, 100);
}

void producer(EventQueue<std::string>& q, const std::string& str) {
  q.push(std::make_unique<std::string>(str));
}

void consumer(EventQueue<std::string>& q, std::vector<std::string>& results) {
  results.clear();
  auto res1 = q.waitPop();
  results.push_back(*res1);
  auto res2 = q.waitPop();
  results.push_back(*res2);
}

TEST(CollTraceUtilsTest, EventQueueMultiThreadTest) {
  EventQueue<std::string> q;
  std::vector<std::string> results;
  std::thread t0(consumer, std::ref(q), std::ref(results));
  std::thread t1(producer, std::ref(q), "hello");
  std::thread t2(producer, std::ref(q), "world");
  t0.join();
  t1.join();
  t2.join();
  EXPECT_TRUE(results[0] == "hello" || results[0] == "world");
  EXPECT_TRUE(results[1] == "hello" || results[1] == "world");
}

TEST(CollTraceUtilsTest, getSizeMbTest) {
  std::unordered_map<int, std::vector<std::string>> bytesToTypes = {
      {1, {"ncclInt8", "ncclFp8E4M3", "ncclFp8E5M2"}},
      {2, {"ncclFloat16", "ncclBfloat16"}},
      {4, {"ncclInt32", "ncclUint32", "ncclFloat32"}},
      {8, {"ncclInt64", "ncclUint64", "ncclFloat64"}}};
  for (const auto& it : bytesToTypes) {
    auto types = it.second;
    for (const auto& type : types) {
      auto mb = getSizeMb(type, 1024 * 1024);
      EXPECT_NEAR(mb, it.first, 0.01);
    }
  }
}

int main(int argc, char** argv) {
  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
