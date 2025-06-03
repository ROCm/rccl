// (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

#include "comms/rcclx/develop/meta/lib/ProxyTrace.h"
#include <gtest/gtest.h>

#define NCCL_STEPS 8

// Copied over from proxy.h to avoid dependency on the rccl library
struct ncclProxySubArgs {
  struct ncclProxyConnection* connection;
  int reg;
  // p2p mhandle
  void* mhandle;
  // collnet handles
  void* sendMhandle;
  void* recvMhandle;
  uint8_t* sendbuff;
  uint8_t* recvbuff;
  size_t offset;
  int channelId;
  int nsteps;
  ssize_t nbytes;
  int peer;

  int groupSize; // Number of consecutive sub operations sharing the same
                 // recvComm
  uint64_t base;
  uint64_t posted;
  uint64_t received;
  uint64_t flushed;
  uint64_t transmitted;
  uint64_t done;
  uint64_t end;
  void* requests[NCCL_STEPS];
  void* profilingEvents[NCCL_STEPS];
  void* recvRequestsCache[NCCL_STEPS];
  int recvRequestsSubCount;
  facebook::rcclx::ProxyTraceRecordKey traceKey;
  facebook::rcclx::ProxyTraceExtraInfo traceInfo;
};

// Mock ncclProxyState struct
struct ncclProxyState {
  std::unique_ptr<facebook::rcclx::ProxyTrace> proxyTrace;
};

class ProxyTraceTestFixture : public ::testing::Test {
 public:
  ncclProxyState* proxyState;
  ncclProxySubArgs *sub1, *sub2;
  uint64_t commHash = 123456789;
  int64_t opCount = 31;
  int nSteps = 10;
  void SetUp() override {
    proxyState = new ncclProxyState();
    META_INIT_PROXY_TRACE(proxyState, 0, commHash);
    EXPECT_NE(proxyState->proxyTrace, nullptr);
    sub1 = new ncclProxySubArgs();
    sub2 = new ncclProxySubArgs();
    sub1->traceKey = {commHash, opCount, -1};
    sub2->traceKey = {commHash, opCount, -1};
    sub1->nsteps = nSteps;
    sub2->nsteps = nSteps;
  }
  void TearDown() override {
    delete sub1;
    delete sub2;
    delete proxyState;
  }
};

TEST_F(ProxyTraceTestFixture, nonEmptySingleton) {
  const auto& tracer = proxyState->proxyTrace;
  EXPECT_NE(tracer, nullptr);
}

TEST_F(ProxyTraceTestFixture, addTraceOp) {
  auto& tracer = proxyState->proxyTrace;
  EXPECT_EQ(
      tracer->getCreateProxyOpId(
          sub1->traceKey.commHash, sub1->traceKey.opCount),
      0);
  META_PROXY_ADD_NEW_TRACE_OP(
      proxyState, facebook::rcclx::ProxyOpType::SEND, sub1);
  EXPECT_EQ(sub1->traceKey.proxyOpId, 0);
  META_PROXY_ADD_NEW_TRACE_OP(
      proxyState, facebook::rcclx::ProxyOpType::RECV, sub2);
  EXPECT_EQ(sub2->traceKey.proxyOpId, 1);
  EXPECT_EQ(
      tracer->getCreateProxyOpId(
          sub1->traceKey.commHash, sub1->traceKey.opCount),
      2);
  auto traceRecordPtr = tracer->getProxyTraceOpPtr(sub1->traceKey);
  EXPECT_EQ(traceRecordPtr->opType, facebook::rcclx::ProxyOpType::SEND);
}

TEST_F(ProxyTraceTestFixture, getMapSizeMB) {
  auto& tracer = proxyState->proxyTrace;

  META_PROXY_ADD_NEW_TRACE_OP(
      proxyState, facebook::rcclx::ProxyOpType::SEND, sub1);
  auto size1 = tracer->getMapSizeMB();
  EXPECT_GT(size1, 0);
  META_PROXY_ADD_NEW_TRACE_OP(
      proxyState, facebook::rcclx::ProxyOpType::RECV, sub2);
  auto size2 = tracer->getMapSizeMB();
  EXPECT_GT(size2, size1);
  // finish sub1
  META_PROXY_TRACE_INFO_COPY(sub1->done, nSteps);
  META_PROXY_TRACE_SET_COUNTER(proxyState, sub1->traceKey, done, sub1->done);
  // sub1 is now serialized and should be moved from activeOps to finishedOps
  auto size3 = tracer->getMapSizeMB();
  EXPECT_GT(size3, size1);
}

TEST_F(ProxyTraceTestFixture, updateTraceOp) {
  auto& tracer = proxyState->proxyTrace;
  META_PROXY_ADD_NEW_TRACE_OP(
      proxyState, facebook::rcclx::ProxyOpType::SEND, sub1);
  sub1->posted += 3;
  sub1->transmitted += 2;
  META_PROXY_TRACE_SET_COUNTER(proxyState, sub1->traceKey, kernelCopyReady, 1);
  META_PROXY_TRACE_SET_COUNTER(proxyState, sub1->traceKey, posted, 3);
  META_PROXY_TRACE_SET_COUNTER(proxyState, sub1->traceKey, transmitted, 2);

  auto traceRecordPtr = tracer->getProxyTraceOpPtr(sub1->traceKey);
  EXPECT_NE(traceRecordPtr, nullptr);
  EXPECT_EQ(traceRecordPtr->posted, 3);
  EXPECT_EQ(traceRecordPtr->transmitted, 2);
  EXPECT_EQ(traceRecordPtr->kernelCopyReady, 1);
  EXPECT_GE(traceRecordPtr->lastUpdateTs, traceRecordPtr->startTs);
}

TEST_F(ProxyTraceTestFixture, infoCopy) {
  auto& tracer = proxyState->proxyTrace;
  META_PROXY_ADD_NEW_TRACE_OP(
      proxyState, facebook::rcclx::ProxyOpType::SEND, sub1);
  int64_t rand = 123456789;
  META_PROXY_TRACE_INFO_COPY(sub1->posted, rand);
  META_PROXY_TRACE_SET_COUNTER(
      proxyState, sub1->traceKey, posted, sub1->posted);
  auto traceRecordPtr = tracer->getProxyTraceOpPtr(sub1->traceKey);
  EXPECT_EQ(traceRecordPtr->posted, rand);
}

TEST_F(ProxyTraceTestFixture, memoryReclaim) {
  auto& tracer = proxyState->proxyTrace;
  tracer->resetAll();
  META_PROXY_ADD_NEW_TRACE_OP(
      proxyState, facebook::rcclx::ProxyOpType::SEND, sub1);
  META_PROXY_TRACE_INFO_COPY(sub1->done, nSteps);
  META_PROXY_TRACE_SET_COUNTER(proxyState, sub1->traceKey, done, sub1->done);
  auto traceRecordPtr = tracer->getProxyTraceOpPtr(sub1->traceKey);
  EXPECT_EQ(traceRecordPtr, nullptr);
  EXPECT_GT(tracer->getMapSizeMB(), 0);
}
