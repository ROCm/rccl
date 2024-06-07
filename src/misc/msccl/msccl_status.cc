/*************************************************************************
 * Copyright (c) Microsoft Corporation.
 * Licensed under the MIT License.
 ************************************************************************/

#include "msccl/msccl_status.h"
#include "msccl/msccl_struct.h"

#include "debug.h"

#include <memory>
#include <mutex>
#include <unordered_map>
using namespace std;

struct mscclRankState {
  int rank;
  bool initialized;
  mscclStatus status;
  mscclSavedProxyArgs savedProxyArgs;

  mscclRankState() : rank(-1), initialized(false), status(), savedProxyArgs() {}
  mscclRankState(const mscclRankState&) = delete;
};

static inline mscclRankState& mscclGetRankState(int rank) {
  static mutex rankStatesMutex;
  static unordered_map<int, shared_ptr<mscclRankState>> rankStates;

  static thread_local shared_ptr<mscclRankState> threadRankState = make_shared<mscclRankState>();

  if (rank < 0 || threadRankState->rank == rank) {
    return *threadRankState;
  }
  if (threadRankState->rank >= 0) {
    WARN("Changing rank %d to rank %d", threadRankState->rank, rank);
  }

  lock_guard<mutex> lock(rankStatesMutex);

  auto rankStateIt = rankStates.find(rank);
  if (rankStateIt == rankStates.end()) {
    threadRankState->rank = rank;
    rankStates.insert(make_pair(rank, threadRankState));
  }
  else {
    threadRankState = rankStateIt->second;
  }
  return *threadRankState;
}

bool mscclInitialized(int rank) {
  return mscclGetRankState(rank).initialized;
}

void mscclSetInitialized(int rank, bool initialized) {
  auto& state = mscclGetRankState(rank);
  assert(!initialized || !state.initialized);
  state.initialized = initialized;
}

mscclStatus& mscclGetStatus(int rank) {
  return mscclGetRankState(rank).status;
}

mscclThreadLocalStatus& mscclGetThreadLocalStatus() {
  static thread_local mscclThreadLocalStatus threadLocalStatus;
  return threadLocalStatus;
}

mscclSavedProxyArgs& mscclGetSavedProxyArgs(int rank) {
  return mscclGetRankState(rank).savedProxyArgs;
}
