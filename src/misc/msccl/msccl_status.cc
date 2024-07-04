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
  explicit mscclRankState(const mscclRankState&) = default;
};

static mutex rankStatesMutex;
static unordered_map<int, shared_ptr<mscclRankState>> rankStates;

static inline mscclRankState& mscclGetRankState(int rank) {
  static thread_local shared_ptr<mscclRankState> threadRankState = make_shared<mscclRankState>();

  if (rank < 0) {
    return *threadRankState;
  }

  lock_guard<mutex> lock(rankStatesMutex);

  auto rankStateIt = rankStates.find(rank);
  if (rankStateIt == rankStates.end()) {
    rankStateIt = rankStates.insert(make_pair(rank, make_shared<mscclRankState>(*threadRankState))).first;
    rankStateIt->second->rank = rank;
  }
  return *(rankStateIt->second);
}

bool mscclInitialized(int rank) {
  return mscclGetRankState(rank).initialized;
}

void mscclSetInitialized(int rank, bool initialized) {
  auto& state = mscclGetRankState(rank);
  assert(!initialized || !state.initialized);
  state.initialized = initialized;
}

void mscclRemoveRank(int rank) {
  lock_guard<mutex> lock(rankStatesMutex);
  rankStates.erase(rank);
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
