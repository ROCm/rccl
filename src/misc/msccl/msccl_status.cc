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
  mscclSavedProxyArgs savedProxyArgs;

  mscclThreadState() : initialized(false), status(), savedProxyArgs() {}
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

static inline mscclThreadState& mscclGetThreadState(int rank) {
  mscclSetThreadRank(rank);
  if (!threadState) {
    threadState = std::make_shared<mscclThreadState>();
  }
  return *threadState;
}

bool& mscclInitialized(int rank) {
  return mscclGetThreadState(rank).initialized;
}

mscclStatus& mscclGetStatus(int rank) {
  return mscclGetThreadState(rank).status;
}

mscclThreadLocalStatus& mscclGetThreadLocalStatus() {
  static thread_local mscclThreadLocalStatus threadLocalStatus;
  return threadLocalStatus;
}

mscclSavedProxyArgs& mscclGetSavedProxyArgs(int rank) {
  return mscclGetThreadState(rank).savedProxyArgs;
}
