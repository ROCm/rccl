/*************************************************************************
 * Copyright (c) Microsoft Corporation.
 * Licensed under the MIT License.
 ************************************************************************/

#include <atomic>

#include "alloc.h"
#include "checks.h"

#include "msccl/msccl_lifecycle.h"
#include "msccl/msccl_parser.h"
#include "msccl/msccl_setup.h"
#include "msccl/msccl_status.h"

RCCL_PARAM(MscclEnabled, "MSCCL_ENABLE", 0);
static const char* mscclAlgoFilePathEnv = "MSCCL_ALGO_FILE_PATH";
static std::atomic<bool> mscclInitialized;
static bool mscclSchedulerTriedLoadAlgo = false;

bool mscclEnabled() {
  return rcclParamMscclEnabled();
}

static bool mscclIsCallerFlag = false;

void mscclSetIsCallerFlag() {
  mscclIsCallerFlag = true;
}

void mscclClearIsCallerFlag() {
  mscclIsCallerFlag = false;
}

bool mscclIsCaller() {
  return mscclIsCallerFlag;
}

bool mscclAvailable() {
  return mscclEnabled() && mscclInitialized.load(std::memory_order_acquire);
}

ncclResult_t mscclInit(ncclComm_t comm) {
  if (comm->intraRanks > 1) {
    mscclInitialized.store(false, std::memory_order_release);
    INFO(NCCL_INIT, "MSCCL doesn't support multiple GPUs in one process and is not available");
    return ncclSuccess;
  } else {
    mscclInitialized.store(true, std::memory_order_release);
  }

  mscclStatus& status = mscclGetStatus();
  status.scratchBuffer = nullptr;
  status.scratchBufferSize = 0;
  status.rank = comm->rank;
  status.workIndex = 1;
  status.freeAlgoHandles.resize(MSCCL_MAX_NUM_ALGOS);
  for (int i = 0; i < MSCCL_MAX_NUM_ALGOS; i++) {
    status.freeAlgoHandles[i] = MSCCL_MAX_NUM_ALGOS - i - 1;
  }
  NCCLCHECK(ncclCudaCalloc(&status.syncFlags, MSCCL_MAX_NUM_THREAD_BLOCKS));
  status.groupStatus = mscclNoGroup;
  status.groupDepth = 0;
  mscclSchedulerTriedLoadAlgo = false;
  return ncclSuccess;
}

ncclResult_t mscclGroupStart() {
  mscclStatus& status = mscclGetStatus();
  status.groupDepth++;
  if (status.groupStatus == mscclNoGroup) {
    status.groupStatus = mscclGroupSupportedOp;
  }
  return ncclSuccess;
}

static ncclResult_t mscclScheduler(struct mscclSchedulerParam* param) {
  static bool algoAvailable = false;
  static mscclAlgoHandle_t loadedAlgoHandle;
  static mscclAlgo* loadedHostAlgo = nullptr;

  param->scheduled = false;

  if (!mscclSchedulerTriedLoadAlgo) {
    mscclSchedulerTriedLoadAlgo = true;
    const char* mscclAlgoFilePath = getenv(mscclAlgoFilePathEnv);
    if (mscclAlgoFilePath != nullptr) {
      NCCLCHECK(mscclLoadAlgo(mscclAlgoFilePath, &loadedAlgoHandle));
      mscclStatus& status = mscclGetStatus();
      loadedHostAlgo = status.hostAlgos[loadedAlgoHandle];
      algoAvailable = true;
    }
  }
  if (!algoAvailable) {
    return ncclSuccess;
  }

  bool mscclAlgoFuncIsValid = loadedHostAlgo->func == param->func;
  if (!mscclAlgoFuncIsValid) {
    return ncclSuccess;
  }

  bool numGpusIsValid = loadedHostAlgo->nRanks == param->comm->nRanks;
  if (!numGpusIsValid) {
    return ncclSuccess;
  }

  size_t nBytes = param->count * ncclTypeSize(param->dataType) * loadedHostAlgo->sizeMultiplier;
  bool msgSizeIsValid =
    param->count > 0 && (param->count % loadedHostAlgo->nChunksPerLoop) == 0 &&
    nBytes >= loadedHostAlgo->minBytes &&
    (loadedHostAlgo->maxBytes == 0 || nBytes <= loadedHostAlgo->maxBytes);
  if (!msgSizeIsValid) {
    return ncclSuccess;
  }

  bool isInPlace = false;
  if (param->func == mscclFuncReduce ||
      param->func == mscclFuncBroadcast ||
      param->func == mscclFuncAllReduce ||
      param->func == mscclFuncAllToAll ||
      param->func == mscclFuncAllToAllv) {
    isInPlace = param->sendBuff == param->recvBuff;
  } else if (param->func == mscclFuncAllGather ||
             param->func == mscclFuncGather) {
    isInPlace = (char*)param->sendBuff == (char*)param->recvBuff + param->comm->rank * param->count * ncclTypeSize(param->dataType);
  } else if (param->func == mscclFuncReduceScatter ||
             param->func == mscclFuncScatter) {
    isInPlace = (char*)param->recvBuff == (char*)param->sendBuff + param->comm->rank * param->count * ncclTypeSize(param->dataType);
  }
  bool inPlaceOutOfPlaceIsValid = isInPlace ? loadedHostAlgo->inPlace : loadedHostAlgo->outOfPlace;
  if (!inPlaceOutOfPlaceIsValid) {
    return ncclSuccess;
  }

  param->handle = loadedAlgoHandle;
  param->scheduled = true;
  return ncclSuccess;
}

static ncclResult_t mscclSetSchedulerParam(
  const void* sendBuff, const size_t sendCounts[], const size_t sDisPls[],
  void* recvBuff, const size_t recvCounts[], const size_t rDisPls[],
  size_t count, ncclDataType_t dataType, int root, int peer, ncclRedOp_t op,
  mscclFunc_t func, ncclComm_t comm, hipStream_t stream,
  struct mscclSchedulerParam* param) {
  param->sendBuff = sendBuff;
  param->sendCounts = sendCounts;
  param->sDisPls = sDisPls;
  param->recvBuff = recvBuff;
  param->recvCounts = recvCounts;
  param->rDisPls = rDisPls;
  param->count = count;
  param->dataType = dataType;
  param->root = root;
  param->peer = peer;
  param->op = op;
  param->func = func;
  param->comm = comm;
  param->stream = stream;
  return ncclSuccess;
}

static ncclResult_t mscclSaveCountsAndDispls(struct mscclSchedulerParam* param) {
  if (param->sendCounts) {
    param->savedSendCounts.assign(param->sendCounts, param->sendCounts + param->comm->nRanks);
    param->sendCounts = param->savedSendCounts.data();
    param->savedSDisPls.assign(param->sDisPls, param->sDisPls + param->comm->nRanks);
    param->sDisPls = param->savedSDisPls.data();
    param->savedRecvCounts.assign(param->recvCounts, param->recvCounts + param->comm->nRanks);
    param->recvCounts = param->savedRecvCounts.data();
    param->savedRDisPls.assign(param->rDisPls, param->rDisPls + param->comm->nRanks);
    param->rDisPls = param->savedRDisPls.data();
  }
  return ncclSuccess;
}

static ncclResult_t mscclRunSavedParams() {
  mscclStatus& status = mscclGetStatus();
  for (auto& param : status.savedSchedulerParams) {
    NCCLCHECK(mscclRunAlgo(
      param.sendBuff, param.sendCounts, param.sDisPls,
      param.recvBuff, param.recvCounts, param.rDisPls,
      param.count, param.dataType, param.root, param.peer, param.op, param.handle, param.comm, param.stream));
  }
  status.savedSchedulerParams.clear();
  return ncclSuccess;
}

static ncclResult_t mscclFallBackSavedParams() {
  mscclStatus& status = mscclGetStatus();
  mscclSetIsCallerFlag();
  for (auto& param : status.savedSchedulerParams) {
    switch (param.func) {
      case mscclFuncReduce:
        NCCLCHECK(ncclReduce(param.sendBuff, param.recvBuff, param.count, param.dataType,
          param.op, param.root, param.comm, param.stream));
        break;
      case mscclFuncBroadcast:
        NCCLCHECK(ncclBroadcast(param.sendBuff, param.recvBuff, param.count, param.dataType,
          param.root, param.comm, param.stream));
        break;
      case mscclFuncAllReduce:
        NCCLCHECK(ncclAllReduce(param.sendBuff, param.recvBuff, param.count, param.dataType,
          param.op, param.comm, param.stream));
        break;
      case mscclFuncReduceScatter:
        NCCLCHECK(ncclReduceScatter(param.sendBuff, param.recvBuff, param.count, param.dataType,
          param.op, param.comm, param.stream));
        break;
      case mscclFuncAllGather:
        NCCLCHECK(ncclAllGather(param.sendBuff, param.recvBuff, param.count, param.dataType,
          param.comm, param.stream));
        break;
      case mscclFuncSend:
        NCCLCHECK(ncclSend(param.sendBuff, param.count, param.dataType,
          param.peer, param.comm, param.stream));
        break;
      case mscclFuncRecv:
        NCCLCHECK(ncclRecv(param.recvBuff, param.count, param.dataType,
          param.peer, param.comm, param.stream));
        break;
      case mscclFuncGather:
        NCCLCHECK(ncclGather(param.sendBuff, param.recvBuff, param.count, param.dataType,
          param.root, param.comm, param.stream));
        break;
      case mscclFuncScatter:
        NCCLCHECK(ncclScatter(param.sendBuff, param.recvBuff, param.count, param.dataType,
          param.root, param.comm, param.stream));
        break;
      case mscclFuncAllToAll:
        NCCLCHECK(ncclAllToAll(param.sendBuff, param.recvBuff, param.count, param.dataType,
          param.comm, param.stream));
        break;
      case mscclFuncAllToAllv:
        NCCLCHECK(ncclAllToAllv(
          param.sendBuff, param.sendCounts, param.sDisPls,
          param.recvBuff, param.recvCounts, param.rDisPls,
          param.dataType, param.comm, param.stream));
        break;
      default:
        WARN("Invalid MSCCL function type in saved parameter");
        return ncclInvalidUsage;
    }
  }
  mscclClearIsCallerFlag();
  status.savedSchedulerParams.clear();
  return ncclSuccess;
}

ncclResult_t mscclEnqueueCheck(
    const void* sendBuff, const size_t sendCounts[], const size_t sDisPls[],
    void* recvBuff, const size_t recvCounts[], const size_t rDisPls[],
    size_t count, ncclDataType_t dataType, int root, int peer, ncclRedOp_t op,
    mscclFunc_t func, ncclComm_t comm, hipStream_t stream) {
  mscclStatus& status = mscclGetStatus();
  hipStreamCaptureStatus captureStatus;
  unsigned long long pid;

  status.savedSchedulerParams.push_back({});
  NCCLCHECK(mscclSetSchedulerParam(
    sendBuff, sendCounts, sDisPls, recvBuff, recvCounts, rDisPls,
    count, dataType, root, peer, op, func, comm, stream,
    &status.savedSchedulerParams.back()));

  switch (status.groupStatus) {
    case mscclNoGroup:
      CUDACHECK(hipStreamGetCaptureInfo(stream, &captureStatus, &pid));
      if (captureStatus == hipStreamCaptureStatusNone) {
        NCCLCHECK(mscclScheduler(&status.savedSchedulerParams.back()));
        if (status.savedSchedulerParams.back().scheduled) {
          NCCLCHECK(mscclRunSavedParams());
          break;
        }
      }
      NCCLCHECK(mscclFallBackSavedParams());
      break;
    case mscclGroupSupportedOp:
      CUDACHECK(hipStreamGetCaptureInfo(stream, &captureStatus, &pid));
      if (captureStatus == hipStreamCaptureStatusNone) {
        NCCLCHECK(mscclScheduler(&status.savedSchedulerParams.back()));
        if (status.savedSchedulerParams.back().scheduled) {
          // Only save counts and displs when there is suitable MSCCL algorithm for this
          NCCLCHECK(mscclSaveCountsAndDispls(&status.savedSchedulerParams.back()));
          break;
        }
      }
      NCCLCHECK(mscclFallBackSavedParams());
      break;
    case mscclGroupUnsupportedOp:
      NCCLCHECK(mscclFallBackSavedParams());
      break;
    default:
      return ncclInvalidUsage;
  }
  return ncclSuccess;
}

ncclResult_t mscclGroupEnd() {
  mscclStatus& status = mscclGetStatus();
  status.groupDepth--;
  if (status.groupDepth == 0) {
    if (status.groupStatus == mscclGroupSupportedOp) {
      NCCLCHECK(mscclRunSavedParams());
    }
    status.groupStatus = mscclNoGroup;
  }
  return ncclSuccess;
}

ncclResult_t mscclTeardown() {
  if (!mscclInitialized.load(std::memory_order_acquire)) {
    return ncclSuccess;
  }
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
  mscclInitialized.store(false, std::memory_order_release);
  return ncclSuccess;
}
