/*************************************************************************
 * Copyright (c) Microsoft Corporation.
 * Licensed under the MIT License.
 ************************************************************************/

#include "msccl/msccl_status.h"
#include "msccl/msccl_struct.h"

#include "debug.h"

#include <memory>
#include <mutex>
#include <vector>
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
static vector<shared_ptr<mscclRankState>> rankStates;

static inline mscclRankState& mscclGetRankState(int rank, int rankCount = -1) {
  static thread_local shared_ptr<mscclRankState> threadRankState;

  if (rankCount > 0) {
    lock_guard<mutex> lock(rankStatesMutex);
    if (rankStates.size() < rankCount) {
      rankStates.resize((size_t)rankCount);
    }
  }

  if (rank < 0 || rank >= rankStates.size()) {
    if (!threadRankState) {
      threadRankState.reset(new mscclRankState());
    }
    return *threadRankState;
  }

  if (!rankStates[rank]) {
    if (!threadRankState) {
      threadRankState.reset(new mscclRankState());
    }
    rankStates[rank] = threadRankState;
  }

  if (!threadRankState) {
    threadRankState = rankStates[rank];
  }

  return *rankStates[rank];
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
  if (rank < rankStates.size()) {
    rankStates[rank].reset();
  }
}

mscclStatus& mscclGetStatus(int rank, int rankCount) {
  return mscclGetRankState(rank, rankCount).status;
}

mscclThreadLocalStatus& mscclGetThreadLocalStatus() {
  static thread_local mscclThreadLocalStatus threadLocalStatus;
  return threadLocalStatus;
}

mscclSavedProxyArgs& mscclGetSavedProxyArgs(int rank) {
  return mscclGetRankState(rank).savedProxyArgs;
}
