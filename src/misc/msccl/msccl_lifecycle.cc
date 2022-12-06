/*************************************************************************
 * Copyright (c) Microsoft Corporation.
 * Licensed under the MIT License.
 ************************************************************************/

#include "alloc.h"
#include "checks.h"

#include "msccl/msccl_lifecycle.h"
#include "msccl/msccl_parser.h"
#include "msccl/msccl_setup.h"
#include "msccl/msccl_status.h"

RCCL_PARAM(MscclEnabled, "MSCCL_ENABLED", 0);
const char* mscclAlgoFilePathEnv = "MSCCL_ALGO_FILE_PATH";

bool mscclEnabled() {
  return rcclParamMscclEnabled();
}

ncclResult_t mscclInit(int rank) {
  mscclStatus& status = mscclGetStatus();
  status.scratchBuffer = nullptr;
  status.scratchBufferSize = 0;
  status.rank = rank;
  status.workIndex = 1;
  status.freeAlgoHandles.resize(MSCCL_MAX_NUM_ALGOS);
  for (int i = 0; i < MSCCL_MAX_NUM_ALGOS; i++) {
    status.freeAlgoHandles[i] = MSCCL_MAX_NUM_ALGOS - i - 1;
  }
  NCCLCHECK(ncclCudaCalloc(&status.syncFlags, MSCCL_MAX_NUM_THREAD_BLOCKS));
  return ncclSuccess;
}

ncclResult_t mscclScheduler(
    const void* sendBuff, const size_t sendCounts[], const size_t sDisPls[],
    void* recvBuff, const size_t recvCounts[], const size_t rDisPls[],
    size_t count, ncclDataType_t dataType, int root, int peer, ncclRedOp_t op,
    mscclFunc_t mscclFunc, bool* mscclScheduled,
    ncclComm_t comm, hipStream_t stream) {
  static bool mscclAlgoTriedLoad = false;
  static bool mscclAlgoAvailable = false;
  static mscclAlgoHandle_t mscclAlgoHandle;
  static mscclFunc_t mscclAlgoFunc;

  *mscclScheduled = false;

  if (!mscclAlgoTriedLoad) {
    mscclAlgoTriedLoad = true;
    const char* mscclAlgoFilePath = getenv(mscclAlgoFilePathEnv);
    if (mscclAlgoFilePath != nullptr) {
      NCCLCHECK(mscclLoadAlgo(mscclAlgoFilePath, &mscclAlgoHandle));
      mscclStatus& status = mscclGetStatus();
      mscclAlgoFunc = status.hostAlgos[mscclAlgoHandle]->func;
      mscclAlgoAvailable = true;
    }
  }
  if (mscclAlgoAvailable && mscclAlgoFunc == mscclFunc) {
    NCCLCHECK(mscclRunAlgo(
      sendBuff, nullptr, nullptr,
      recvBuff, nullptr, nullptr,
      count, dataType, 0, 0, op,
      mscclAlgoHandle, comm, stream));
    *mscclScheduled = true;
  }
  return ncclSuccess;
}

ncclResult_t mscclTeardown() {
  mscclStatus& status = mscclGetStatus();
  for (auto &p : status.hostAlgos) {
    free(p.second);
    status.freeAlgoHandles.push_back(p.first);
  }
  for (auto &p : status.devAlgos) {
    CUDACHECK(hipFree(p.second));
  }
  CUDACHECK(hipFree(status.scratchBuffer));
  CUDACHECK(hipFree(status.syncFlags));
  status.hostAlgos.clear();
  status.devAlgos.clear();
  status.freeAlgoHandles.clear();
  status.scratchBuffer = nullptr;
  status.scratchBufferSize = 0;
  status.workIndex = 1;
  return ncclSuccess;
}
