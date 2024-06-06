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
  mscclStatus status;
  mscclThreadLocalStatus threadLocalStatus;
  mscclSavedProxyArgs savedProxyArgs;

  mscclThreadState() : status(), threadLocalStatus(), savedProxyArgs() {}
  mscclThreadState(const mscclThreadState&) = delete;
};

static std::mutex threadStatesMutex;
static std::unordered_map<struct ncclComm*, std::shared_ptr<mscclThreadState>> threadStates;

static thread_local std::shared_ptr<mscclThreadState> threadState;
static thread_local struct ncclComm* threadLocalComm = nullptr;

void mscclSetThreadLocalComm(struct ncclComm* comm) {
  if (threadLocalComm == comm) {
    return;
  }

  threadLocalComm = comm;

  std::lock_guard<std::mutex> lock(threadStatesMutex);

  auto threadStateIt = threadStates.find(threadLocalComm);
  if (threadStateIt == threadStates.end()) {
    if (!threadState) {
      threadState = std::make_shared<mscclThreadState>();
    }
    threadStates.insert(std::make_pair(threadLocalComm, threadState));
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
