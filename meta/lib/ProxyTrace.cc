// (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

#include "ProxyTrace.h"

#include <fmt/format.h>
#include <folly/logging/xlog.h>
#include <iomanip>
#include <iostream>
#include <map>
#include <string>
#include <unordered_map>

#include "Common.h"

constexpr int32_t kFinishedProxyOpItems = 32;
static std::map<facebook::rcclx::ProxyOpStepStatus, std::string>
    proxyStepStatusStrMap = {
        {facebook::rcclx::ProxyOpStepStatus::INIT, "INIT"},
        {facebook::rcclx::ProxyOpStepStatus::POSTING, "POSTING"},
        {facebook::rcclx::ProxyOpStepStatus::SENDING, "SENDING"},
        {facebook::rcclx::ProxyOpStepStatus::RECEIVING, "RECEIVING"},
        {facebook::rcclx::ProxyOpStepStatus::WAITING_GPU, "WAITING_GPU"},
        {facebook::rcclx::ProxyOpStepStatus::FLUSHING, "FLUSHING"},
        {facebook::rcclx::ProxyOpStepStatus::DONE, "DONE"},
        {facebook::rcclx::ProxyOpStepStatus::UNINITIALIZED, "ILLEGAL"},
};

void facebook::rcclx::ProxyTrace::resetAll() {
  activeOps.clear();
  activeOpIdTracker.clear();
  myRank = -1;
  initialized = false;
}

bool facebook::rcclx::ProxyTrace::checkActiveOpExist(
    uint64_t commHash,
    uint64_t opCount,
    uint32_t proxyOpId) const {
  return (
      activeOps.find(commHash) != activeOps.end() &&
      activeOps.at(commHash).find(opCount) != activeOps.at(commHash).end() &&
      activeOps.at(commHash).at(opCount).find(proxyOpId) !=
          activeOps.at(commHash).at(opCount).end());
}

// Get a unique proxyOpId for a given commHash:opCount
// If the opCount is not found, create a new entry for it and return 0
int64_t facebook::rcclx::ProxyTrace::getCreateProxyOpId(
    uint64_t commHash,
    uint64_t opCount) {
  return activeOpIdTracker[commHash][opCount];
}

facebook::rcclx::ProxyTraceOp* facebook::rcclx::ProxyTrace::getProxyTraceOpPtr(
    const ProxyTraceRecordKey& key) {
  if (checkActiveOpExist(key.commHash, key.opCount, key.proxyOpId)) {
    return &(activeOps.at(key.commHash).at(key.opCount).at(key.proxyOpId));
  }
  return nullptr;
}

void facebook::rcclx::ProxyTrace::checkOpCompleted(
    const ProxyTraceRecordKey& key) {
  if (checkActiveOpExist(key.commHash, key.opCount, key.proxyOpId)) {
    auto& traceOp = activeOps[key.commHash][key.opCount][key.proxyOpId];
    // Remove finished proxyOp or colls to avoid memory leak
    if (traceOp.done == traceOp.nSteps) {
      traceOp.status = ProxyOpStepStatus::DONE;
      if (finishedOps.size() >= kFinishedProxyOpItems) {
        finishedOps.pop_front();
      }
      finishedOps.push_back({key.str(), traceOp.str()});
      activeOps[key.commHash][key.opCount].erase(key.proxyOpId);
      XLOG(DBG) << "[proxy-debug] ProxyTraceOp done " << key.str();
      if (activeOps[key.commHash][key.opCount].empty()) {
        activeOps[key.commHash].erase(key.opCount);
        activeOpIdTracker[key.commHash].erase(key.opCount);
        XLOG(DBG) << "[proxy] opCount <" << key.commHash << "," << key.opCount
                  << "> done" << ", mapSizeMB:" << std::fixed
                  << std::setprecision(2) << getMapSizeMB();
      }
    }
  } else {
    XLOG(WARN) << "[proxy] ProxyTraceOp not found " << key.str();
  }
}

void facebook::rcclx::ProxyTrace::addNewProxyTraceOp(
    const ProxyTraceRecordKey& key,
    const ProxyTraceExtraInfo& extraInfo,
    ProxyOpType opType,
    int channelId,
    int nSteps,
    uint32_t nbytes,
    int peerRank) {
  if (nSteps > 0 &&
      !checkActiveOpExist(key.commHash, key.opCount, key.proxyOpId)) {
    auto traceOp = facebook::rcclx::ProxyTraceOp();
    traceOp.opType = opType;
    traceOp.traceKey = key;
    traceOp.extraInfo = extraInfo;
    traceOp.channelId = channelId;
    traceOp.nSteps = nSteps;
    traceOp.nbytes = nbytes;
    traceOp.myRank = this->myRank;
    traceOp.peerRank = peerRank;
    traceOp.startTs = std::chrono::high_resolution_clock::now();
    traceOp.status = ProxyOpStepStatus::INIT;
    activeOpIdTracker[key.commHash][key.opCount]++;
    XLOG(DBG) << "[proxy] add " << traceOp.traceKey.str() << " ,"
              << traceOp.extraInfo.str() << " ,opType:"
              << (traceOp.opType == ProxyOpType::SEND ? "S" : "R")
              << " ,chan:" << traceOp.channelId << " ,nSteps:" << traceOp.nSteps
              << " ,nbytes:" << traceOp.nbytes << " ,peer:" << traceOp.peerRank
              << " ,mapSizeMB:" << std::fixed << std::setprecision(2)
              << getMapSizeMB();
    activeOps[key.commHash][key.opCount].emplace(
        key.proxyOpId, std::move(traceOp));
  } else if (nSteps == 0) {
    XLOG(WARN) << "nSteps is 0, ignored << " << key.str();
  }
}

#define NCCL_STEPS 8
void facebook::rcclx::ProxyTraceOp::computeStatus() {
  ProxyOpStepStatus newStatus;
  if (opType == ProxyOpType::RECV) {
    if (posted < nSteps && posted < done + NCCL_STEPS)
      newStatus = ProxyOpStepStatus::POSTING; // Init
    else if (received < posted)
      newStatus = ProxyOpStepStatus::RECEIVING; // Receiving
    else if (received < transmitted)
      newStatus = ProxyOpStepStatus::RECEIVING; // Receiving
    else if (transmitted < received)
      newStatus = ProxyOpStepStatus::FLUSHING; // Flushing
    else if (done < transmitted)
      newStatus = ProxyOpStepStatus::WAITING_GPU; // Waiting on GPU
    else
      newStatus = ProxyOpStepStatus::DONE; // Done
  } else {
    if (posted < nSteps && posted < done + NCCL_STEPS)
      newStatus = ProxyOpStepStatus::POSTING; // Init
    else if (transmitted < posted)
      newStatus = ProxyOpStepStatus::WAITING_GPU; // Waiting on GPU
    else if (done < transmitted)
      newStatus = ProxyOpStepStatus::SENDING; // Sending
    else
      newStatus = ProxyOpStepStatus::DONE; // Done
  }
  this->status = newStatus;
}
#undef NCCL_STEPS

std::string facebook::rcclx::ProxyTrace::dump(uint64_t commHash) {
  std::string result = fmt::format("commDump for commHash:{}\n", commHash);
  std::map<std::string, std::string> sortedDumpStrMap;
  for (auto& opCountMap : activeOps.at(commHash)) {
    for (auto& proxyOpMap : opCountMap.second) {
      ProxyTraceRecordKey traceKey = {
          commHash, opCountMap.first, proxyOpMap.first};
      proxyOpMap.second.computeStatus();
      sortedDumpStrMap[traceKey.str()] = proxyOpMap.second.str();
    }
  }
  for (const auto& [keyStr, proxyOpStr] : sortedDumpStrMap) {
    result += proxyOpStr;
  }
  return result;
}

std::string facebook::rcclx::ProxyTrace::dump() {
  std::string result = "commDump for all active ops ";
  result += fmt::format("mapSizeMB:{:.2f}\n", getMapSizeMB());

  // maps serialized key to serliazed proxyOp; sorted by key
  std::map<std::string, std::string> sortedDumpStrMap;
  for (auto& [commHash, opCountMap] : activeOps) {
    for (auto& [opCount, proxyOpMap] : opCountMap) {
      for (auto& [opId, opEntry] : proxyOpMap) {
        ProxyTraceRecordKey traceKey = {commHash, opCount, opId};
        opEntry.computeStatus();
        sortedDumpStrMap[traceKey.str()] = opEntry.str();
      }
    }
  }

  // add the recent finished ops as well
  for (const auto& [keyStr, proxyOpStr] : finishedOps) {
    sortedDumpStrMap[keyStr] = proxyOpStr;
  }
  for (const auto& [keyStr, proxyOpStr] : sortedDumpStrMap) {
    result += proxyOpStr;
  }
  return result;
}

std::string facebook::rcclx::ProxyTraceOp::str() {
  computeStatus();
  std::string ret = fmt::format(
      "createT:{}, lastT:{}, pot:{}, sendT:{}, cntNm:{}, {}, {}, {}->{}({}), chan:{}, status:{}, ns:{}, nb:{}, po:{}, ke:{}, tail/h:{}, recvT:{}, connSz/h:{}, trans:{}, flushed:{}, recvd:{}, done:{}\n",
      std::chrono::duration_cast<std::chrono::milliseconds>(
          startTs.time_since_epoch())
          .count(),
      std::chrono::duration_cast<std::chrono::milliseconds>(
          lastUpdateTs.time_since_epoch())
          .count(),
      std::chrono::duration_cast<std::chrono::milliseconds>(
          postTs.time_since_epoch())
          .count(),
      std::chrono::duration_cast<std::chrono::milliseconds>(
          lastSendTs.time_since_epoch())
          .count(),
      lastUpdatingCounterName.substr(0, 3),
      traceKey.str(),
      extraInfo.str(),
      myRank,
      peerRank,
      opType == ProxyOpType::SEND ? "S" : "R",
      channelId,
      proxyStepStatusStrMap[status],
      nSteps,
      nbytes,
      posted,
      kernelCopyReady,
      tailOrHead,
      recvTail,
      fifoSzOrHeadCache,
      transmitted,
      flushed,
      received,
      done);
  return ret;
}

float facebook::rcclx::ProxyTrace::getMapSizeMB() const {
  float size = 0;
  for (const auto& [commHash, opCountMap] : activeOps) {
    for (const auto& [opCount, proxyOpMap] : opCountMap) {
      size += proxyOpMap.size() *
          (sizeof(ProxyTraceOp) +
           sizeof(std::unique_ptr<facebook::rcclx::ProxyTraceOp>));
    }
  }
  for (const auto& [keyStr, proxyOpStr] : finishedOps) {
    size += keyStr.size() + proxyOpStr.size();
  }
  return size / 1024.0 / 1024.0;
}

void facebook::rcclx::proxyTraceInit(
    std::unique_ptr<facebook::rcclx::ProxyTrace>& proxyTrace,
    int32_t rank,
    uint64_t commHash) {
  if (!proxyTrace) {
    XLOG(CRITICAL) << "Failed to get ProxyTrace, rank: " << rank
                   << ", commHash: " << commHash;
    return;
  }
  if (!proxyTrace->initialized) {
    META_INTERNAL_INIT();
    XLOG(INFO) << "Initializing ProxyTrace, rank: " << rank
               << ", commHash: " << commHash;
    proxyTrace->initialized = true;
  } else {
    XLOG(INFO) << "ProxyTrace already initialized, rank: " << rank
               << ", commHash: " << commHash;
  }
}
