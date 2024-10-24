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

  // Calling code can allocate states for the number of ranks at an appropriate time.
  // It is assumed that all threads will call this function simultaneously with the
  // same rankCount, which would avoid race conditions later in the function.
  if (rankCount > 0) {
    lock_guard<mutex> lock(rankStatesMutex);
    if (rankStates.size() < rankCount) {
      rankStates.resize((size_t)rankCount);
    }
  }

  if (rank < 0 || rank >= rankStates.size()) {
    // threadRankState is used when no rank state can be returned (rank<0 or rank not in rankStates)
    if (!threadRankState) {
      threadRankState.reset(new mscclRankState());
    }
    return *threadRankState;
  }

  if (!rankStates[rank]) {
    // When no state is yet assigned to a rank, use the current thread's threadRankState.
    if (!threadRankState) {
      threadRankState.reset(new mscclRankState());
    }
    rankStates[rank] = threadRankState;
  }

  if (!threadRankState) {
    // Cache this rank's state in threadRankState in case this thread calls with rank<0 later.
    // NOTE: When multiple ranks share a thread, only the first rank in will be used for rank<0.
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
