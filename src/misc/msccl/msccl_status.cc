/*************************************************************************
 * Copyright (c) Microsoft Corporation.
 * Licensed under the MIT License.
 ************************************************************************/

#include "msccl/msccl_status.h"
#include "msccl/msccl_struct.h"

#include <memory>
#include <mutex>
#include <unordered_map>

struct mscclThreadState {
  bool initialized;
  mscclStatus status;
  mscclThreadLocalStatus threadLocalStatus;
  mscclSavedProxyArgs savedProxyArgs;

  mscclThreadState() : initialized(false), status(), threadLocalStatus(), savedProxyArgs() {}
  mscclThreadState(const mscclThreadState&) = delete;
};

static std::mutex threadStatesMutex;
static std::unordered_map<int, std::shared_ptr<mscclThreadState>> threadStates;

static thread_local std::shared_ptr<mscclThreadState> threadState;
static thread_local int threadLocalRank = -1;

void mscclSetThreadRank(int rank) {
  if (rank < 0 || threadLocalRank == rank) {
    return;
  }

  threadLocalRank = rank;

  std::lock_guard<std::mutex> lock(threadStatesMutex);

  auto threadStateIt = threadStates.find(threadLocalRank);
  if (threadStateIt == threadStates.end()) {
    if (!threadState) {
      threadState = std::make_shared<mscclThreadState>();
    }
    threadStates.insert(std::make_pair(threadLocalRank, threadState));
  }
  else {
    threadState = threadStateIt->second;
  }
}

static inline mscclThreadState& mscclGetThreadState() {
  if (!threadState) {
    threadState = std::make_shared<mscclThreadState>();
  }
  return *threadState;
}

bool& mscclInitialized() {
  return mscclGetThreadState().initialized;
}

mscclStatus& mscclGetStatus() {
  //static thread_local mscclStatus status;
  return mscclGetThreadState().status;
}

mscclThreadLocalStatus& mscclGetThreadLocalStatus() {
  //static thread_local mscclThreadLocalStatus threadLocalStatus;
  return mscclGetThreadState().threadLocalStatus;
}

mscclSavedProxyArgs& mscclGetSavedProxyArgs() {
  //static thread_local mscclSavedProxyArgs savedProxyArgs;
  return mscclGetThreadState().savedProxyArgs;
}
