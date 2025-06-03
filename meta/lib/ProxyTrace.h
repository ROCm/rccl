// (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

#pragma once

#include <fmt/format.h>
#include <chrono>
#include <cstdint>
#include <deque>
#include <memory>
#include <unordered_map>

namespace facebook::rcclx {

enum class ProxyOpStepStatus {
  INIT,
  POSTING,
  SENDING,
  RECEIVING,
  WAITING_GPU,
  FLUSHING,
  DONE,
  UNINITIALIZED,
  NUM_STATUS,
};

enum class ProxyOpType { SEND, RECV };
// ProxyTraceRecordKey and ProxyTraceExtraInfo is used to pass arguments to
// proxy thread (see ncclProxyOp and ncclProxySubArgs in proxy.h)
struct ProxyTraceRecordKey {
  uint64_t commHash{0};
  int64_t opCount{-1}; // opCount is a unique id for a given collective/p2p
  int64_t proxyOpId{-1}; // id of a proxyOp in an given comm and grouped
                         // collective/p2p (identified as commHash:opCount),
                         // assigned when creating ProxyTraceOp entry
  inline std::string str() const {
    return "<" + std::to_string(commHash) + ":" + std::to_string(opCount) +
        ":" + std::to_string(proxyOpId) + ">";
  }
};

struct ProxyTraceExtraInfo {
  int32_t funcIdx{-1};
  int32_t protocol{-1};
  int32_t pattern{-1};
  uint32_t totalBytes{0};
  uint32_t chunkSize{0};
  inline std::string str() const {
    return fmt::format(
        "[fu,pr,pa,tb,ck]:{},{},{},{},{}",
        funcIdx,
        protocol,
        pattern,
        totalBytes,
        chunkSize);
  }
};

// record progress state per comm per collective per proxyOp
struct ProxyTraceOp {
  ProxyTraceRecordKey traceKey;
  ProxyTraceExtraInfo extraInfo;
  int32_t channelId{-1};
  int32_t nSteps{-1};
  uint32_t nbytes{0};
  int32_t myRank{-1};
  int32_t peerRank{-1};
  int32_t posted{0};
  int32_t kernelCopyReady{0};
  int32_t rtrRecvd{0}; // ready-to-receive signal received from remote
  int32_t rtsSent{0}; // ready-to-send signal sent to remote
  int32_t received{0};
  int32_t transmitted{0};
  int32_t flushed{0};
  int32_t done{0};
  // following four counters are used to debug kernel hang
  uint64_t recvTail{
      0}; // tail pointer of data ready to be sent, updated by kernel
  uint64_t tailOrHead{
      0}; // tail for sender, head for receiver, updated by kernel
  int fifoSzOrHeadCache{
      -1}; // for sender this is data size of D2D copy; for receiver this is
           // head cache, i.e., sub->base + sub->done
  std::string lastUpdatingCounterName;
  ProxyOpType opType{ProxyOpType::SEND};
  ProxyOpStepStatus status{ProxyOpStepStatus::UNINITIALIZED};
  std::chrono::time_point<std::chrono::high_resolution_clock> startTs{};
  std::chrono::time_point<std::chrono::high_resolution_clock> lastUpdateTs{};
  std::chrono::time_point<std::chrono::high_resolution_clock> postTs{};
  std::chrono::time_point<std::chrono::high_resolution_clock> lastSendTs{};
  void computeStatus();
  // str the entry to a string
  std::string str();
};

struct ncclProxyArgs;
struct ncclProxySubArgs;

using ProxyActiveOpMap = std::unordered_map<
    uint64_t /* commHash*/,
    std::unordered_map<
        int64_t /* opCount*/,
        /* proxyOpId : op */
        std::unordered_map<int64_t, ProxyTraceOp>>>;

using ProxyActiveOpIdTracker = std::unordered_map<
    uint64_t /* commHash*/,
    std::unordered_map<int64_t /* opCount*/, int64_t>>;

class ProxyTrace {
 public:
  ProxyTrace(int32_t rank) : myRank(rank){};
  ProxyTrace(const ProxyTrace&) = delete;
  ProxyTrace& operator=(const ProxyTrace&) = delete;
  bool initialized{false};
  void checkOpCompleted(const ProxyTraceRecordKey& key);

  void addNewProxyTraceOp(
      const ProxyTraceRecordKey& key,
      const ProxyTraceExtraInfo& extraInfo,
      ProxyOpType opType,
      int channelId,
      int nSteps,
      uint32_t nbytes,
      int peerRank);

  // Get a unique proxyOpId for a given commHash:opCount
  // If the opCount is not found, create a new entry for it and return 0
  int64_t getCreateProxyOpId(uint64_t commHash, uint64_t opCount);

  // Dump all trace for a given communicator
  std::string dump(uint64_t commHash);

  // Dump all active ops
  std::string dump();

  // check if an active send/recv operation exists for a given commHash:opCount
  bool checkActiveOpExist(
      uint64_t commHash,
      uint64_t opCount,
      uint32_t proxyOpId) const;

  //[TODO] Aoid risk of exposing internal data structures (useful for
  // META_PROXY_TRACE_SET_COUNTER for now)
  ProxyTraceOp* getProxyTraceOpPtr(const ProxyTraceRecordKey& traceKey);
  float getMapSizeMB() const;
  void resetAll();

 private:
  int myRank{-1};

  // Current active send/recv operations.
  // Use map to quickly find the record with commHash:opCount:proxyOpId during
  // active progress. Note that each op may not complete in order, e.g.,
  // proxyOpId 1 may finish before proxyOpId 0 if they are to different peers.
  // Thus, the inner-most layer has to still be a map for searching by
  // proxyOpId, no matter other ops are completed or not.
  ProxyActiveOpMap activeOps;
  ProxyActiveOpIdTracker activeOpIdTracker;

  // keep track of the recent completed ops;
  // A record is a pair of traceKey.str() and ProxyTraceOp.str()
  std::deque<std::pair<std::string, std::string>> finishedOps;
};

void proxyTraceInit(
    std::unique_ptr<ProxyTrace>& proxyTraceObj,
    int32_t rank,
    uint64_t commHash);

} // namespace facebook::rcclx

template <typename T1, typename T2>
inline void META_PROXY_TRACE_INFO_COPY(T1& dst, const T2& src) {
  dst = src;
}

#define META_PROXY_TRACE_SET_COUNTER(proxyStatePtr, traceKey, counter, val)    \
  do {                                                                         \
    auto traceOpPtr = proxyStatePtr->proxyTrace->getProxyTraceOpPtr(traceKey); \
    if (traceOpPtr) {                                                          \
      traceOpPtr->counter = val;                                               \
      traceOpPtr->lastUpdateTs = std::chrono::high_resolution_clock::now();    \
      traceOpPtr->lastUpdatingCounterName = #counter;                          \
      if (strcmp(#counter, "posted") == 0)                                     \
        traceOpPtr->postTs = std::chrono::high_resolution_clock::now();        \
      proxyStatePtr->proxyTrace->checkOpCompleted(traceKey);                   \
    }                                                                          \
  } while (0)

#define META_PROXY_ADD_NEW_TRACE_OP(proxyStatePtr, opType, subArg) \
  do {                                                             \
    auto& tracer = proxyStatePtr->proxyTrace;                      \
    auto opId = tracer->getCreateProxyOpId(                        \
        subArg->traceKey.commHash, subArg->traceKey.opCount);      \
    subArg->traceKey.proxyOpId = opId;                             \
    tracer->addNewProxyTraceOp(                                    \
        subArg->traceKey,                                          \
        subArg->traceInfo,                                         \
        opType,                                                    \
        subArg->channelId,                                         \
        subArg->nsteps,                                            \
        subArg->nbytes,                                            \
        subArg->peer);                                             \
  } while (0)

#define META_INIT_PROXY_TRACE(proxyStatePtr, rank, commHash) \
  do {                                                       \
    proxyStatePtr->proxyTrace =                              \
        std::make_unique<facebook::rcclx::ProxyTrace>(rank); \
    facebook::rcclx::proxyTraceInit(                         \
        proxyStatePtr->proxyTrace, rank, commHash);          \
  } while (0)
