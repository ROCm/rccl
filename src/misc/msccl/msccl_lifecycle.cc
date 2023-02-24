/*************************************************************************
 * Copyright (c) Microsoft Corporation.
 * Licensed under the MIT License.
 ************************************************************************/

#include <atomic>
#include <map>
#include <mutex>
#include <set>

#include <dlfcn.h>

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
static std::mutex mscclLifecycleMutex;

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

static bool mscclCommCompatible(ncclComm_t comm) {
  std::map<uint64_t, std::set<uint64_t>> hostHashToPidHashes;
  for (int i = 0; i < comm->nRanks; i++) {
    uint64_t hostHash = comm->peerInfo[i].hostHash;
    uint64_t pidHash = comm->peerInfo[i].pidHash;
    if (hostHashToPidHashes.find(hostHash) != hostHashToPidHashes.end()) {
      auto& pidHashSet = hostHashToPidHashes[hostHash];
      if (pidHashSet.find(pidHash) != pidHashSet.end()) {
        return false;
      }
    }
    hostHashToPidHashes[hostHash].insert(pidHash);
  }
  return true;
}

ncclResult_t mscclSchedulerInit() {
  mscclStatus& status = mscclGetStatus();

  status.mscclSchedulerLib = dlopen("libmsccl-scheduler.so", RTLD_NOW | RTLD_LOCAL);
  if (status.mscclSchedulerLib == nullptr) {
    if (errno == ENOENT) {
      INFO(NCCL_INIT, "MSCCL: No scheduler found, using internal implementation");
    } else {
      INFO(NCCL_INIT, "MSCCL: Scheduler load returned %d : %s. Using internal implementation", errno, dlerror());
    }
    return ncclSuccess;
  }

  status.mscclSchedulerPtr = (mscclSchedulerInterface *)dlsym(status.mscclSchedulerLib, "mscclScheduler");
  if (status.mscclSchedulerPtr == nullptr) {
    INFO(NCCL_INIT, "MSCCL: Failed to find mscclScheduler symbol, using internal implementation");
    return ncclSuccess;
  }
  NCCLCHECK(status.mscclSchedulerPtr->init());

  return ncclSuccess;
}

ncclResult_t mscclInit(ncclComm_t comm) {
  std::lock_guard<std::mutex> lock(mscclLifecycleMutex);

  if (mscclInitialized.load(std::memory_order_acquire)) {
    return ncclSuccess;
  }

  mscclStatus& status = mscclGetStatus();
  status.scratchBuffer = nullptr;
  status.scratchBufferSize = 0;
  status.workIndex = 1;
  status.freeAlgoHandles.resize(MSCCL_MAX_NUM_ALGOS);
  for (int i = 0; i < MSCCL_MAX_NUM_ALGOS; i++) {
    status.freeAlgoHandles[i] = MSCCL_MAX_NUM_ALGOS - i - 1;
  }
  NCCLCHECK(ncclCudaCalloc(&status.syncFlags, MSCCL_MAX_NUM_THREAD_BLOCKS));
  status.groupStatus = mscclNoGroup;
  status.groupDepth = 0;
  status.lastStream = nullptr;
  mscclSchedulerTriedLoadAlgo = false;

  if (!mscclCommCompatible(comm)) {
    status.fallbackComms.insert(comm);
  }

  NCCLCHECK(mscclSchedulerInit());

  mscclInitialized.store(true, std::memory_order_release);

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

static ncclResult_t mscclExternalScheduler(struct mscclSavedSchedulerParam* param) {
  mscclStatus& status = mscclGetStatus();
  NCCLCHECK(status.mscclSchedulerPtr->selectAlgo(&(param->p)));
  return ncclSuccess;
}

static ncclResult_t mscclInternalScheduler(struct mscclSavedSchedulerParam* param) {
  static bool algoAvailable = false;
  static mscclAlgoHandle_t loadedAlgoHandle;
  static mscclAlgo* loadedHostAlgo = nullptr;

  param->p.scheduled = false;

  if (!mscclSchedulerTriedLoadAlgo) {
    mscclSchedulerTriedLoadAlgo = true;
    const char* mscclAlgoFilePath = getenv(mscclAlgoFilePathEnv);
    if (mscclAlgoFilePath != nullptr) {
      NCCLCHECK(mscclLoadAlgo(mscclAlgoFilePath, &loadedAlgoHandle, param->p.rank));
      mscclStatus& status = mscclGetStatus();
      loadedHostAlgo = status.hostAlgos[loadedAlgoHandle];
      algoAvailable = true;
    }
  }
  if (!algoAvailable) {
    return ncclSuccess;
  }

  bool mscclAlgoFuncIsValid = loadedHostAlgo->func == param->p.func;
  if (!mscclAlgoFuncIsValid) {
    return ncclSuccess;
  }

  bool numGpusIsValid = loadedHostAlgo->nRanks == param->p.nRanks;
  if (!numGpusIsValid) {
    return ncclSuccess;
  }

  size_t nBytes = param->p.count * ncclTypeSize(param->p.dataType) * loadedHostAlgo->sizeMultiplier;
  bool msgSizeIsValid =
    param->p.count > 0 && (param->p.count % loadedHostAlgo->nChunksPerLoop) == 0 &&
    nBytes >= loadedHostAlgo->minBytes &&
    (loadedHostAlgo->maxBytes == 0 || nBytes <= loadedHostAlgo->maxBytes);
  if (!msgSizeIsValid) {
    return ncclSuccess;
  }

  bool isInPlace = false;
  if (param->p.func == mscclFuncReduce ||
      param->p.func == mscclFuncBroadcast ||
      param->p.func == mscclFuncAllReduce ||
      param->p.func == mscclFuncAllToAll ||
      param->p.func == mscclFuncAllToAllv) {
    isInPlace = param->p.sendBuff == param->p.recvBuff;
  } else if (param->p.func == mscclFuncAllGather ||
             param->p.func == mscclFuncGather) {
    isInPlace = (char*)param->p.sendBuff == (char*)param->p.recvBuff + param->p.rank * param->p.count * ncclTypeSize(param->p.dataType);
  } else if (param->p.func == mscclFuncReduceScatter ||
             param->p.func == mscclFuncScatter) {
    isInPlace = (char*)param->p.recvBuff == (char*)param->p.sendBuff + param->p.rank * param->p.count * ncclTypeSize(param->p.dataType);
  }
  bool inPlaceOutOfPlaceIsValid = isInPlace ? loadedHostAlgo->inPlace : loadedHostAlgo->outOfPlace;
  if (!inPlaceOutOfPlaceIsValid) {
    return ncclSuccess;
  }

  param->p.handle = loadedAlgoHandle;
  param->p.scheduled = true;
  return ncclSuccess;
}

static ncclResult_t mscclCallScheduler(struct mscclSavedSchedulerParam* param) {
  mscclStatus& status = mscclGetStatus();
  if (status.mscclSchedulerPtr) {
    NCCLCHECK(mscclExternalScheduler(param));
  } else {
    NCCLCHECK(mscclInternalScheduler(param));
  }
  return ncclSuccess;
}

static ncclResult_t mscclSetSavedSchedulerParam(
  const void* sendBuff, const size_t sendCounts[], const size_t sDisPls[],
  void* recvBuff, const size_t recvCounts[], const size_t rDisPls[],
  size_t count, ncclDataType_t dataType, int root, int peer, ncclRedOp_t op,
  mscclFunc_t func, ncclComm_t comm, hipStream_t stream,
  struct mscclSavedSchedulerParam* param) {
  param->p.sendBuff = sendBuff;
  param->p.sendCounts = sendCounts;
  param->p.sDisPls = sDisPls;
  param->p.recvBuff = recvBuff;
  param->p.recvCounts = recvCounts;
  param->p.rDisPls = rDisPls;
  param->p.count = count;
  param->p.dataType = dataType;
  param->p.root = root;
  param->p.peer = peer;
  param->p.op = op;
  param->p.func = func;
  param->p.rank = comm->rank;
  param->p.nRanks = comm->nRanks;
  param->comm = comm;
  param->stream = stream;
  return ncclSuccess;
}

static ncclResult_t mscclSaveCountsAndDispls(struct mscclSavedSchedulerParam* param) {
  if (param->p.sendCounts) {
    param->savedSendCounts.assign(param->p.sendCounts, param->p.sendCounts + param->p.nRanks);
    param->p.sendCounts = param->savedSendCounts.data();
    param->savedSDisPls.assign(param->p.sDisPls, param->p.sDisPls + param->p.nRanks);
    param->p.sDisPls = param->savedSDisPls.data();
    param->savedRecvCounts.assign(param->p.recvCounts, param->p.recvCounts + param->p.nRanks);
    param->p.recvCounts = param->savedRecvCounts.data();
    param->savedRDisPls.assign(param->p.rDisPls, param->p.rDisPls + param->p.nRanks);
    param->p.rDisPls = param->savedRDisPls.data();
  }
  return ncclSuccess;
}

static ncclResult_t mscclRunSavedParams() {
  mscclStatus& status = mscclGetStatus();
  for (auto& param : status.savedSchedulerParams) {
    NCCLCHECK(mscclRunAlgo(
      param.p.sendBuff, param.p.sendCounts, param.p.sDisPls,
      param.p.recvBuff, param.p.recvCounts, param.p.rDisPls,
      param.p.count, param.p.dataType, param.p.root, param.p.peer, param.p.op, param.p.handle, param.comm, param.stream));
  }
  status.savedSchedulerParams.clear();
  return ncclSuccess;
}

static ncclResult_t mscclFallBackSavedParams() {
  mscclStatus& status = mscclGetStatus();
  mscclSetIsCallerFlag();
  for (auto& param : status.savedSchedulerParams) {
    switch (param.p.func) {
      case mscclFuncReduce:
        NCCLCHECK(ncclReduce(param.p.sendBuff, param.p.recvBuff, param.p.count, param.p.dataType,
          param.p.op, param.p.root, param.comm, param.stream));
        break;
      case mscclFuncBroadcast:
        NCCLCHECK(ncclBroadcast(param.p.sendBuff, param.p.recvBuff, param.p.count, param.p.dataType,
          param.p.root, param.comm, param.stream));
        break;
      case mscclFuncAllReduce:
        NCCLCHECK(ncclAllReduce(param.p.sendBuff, param.p.recvBuff, param.p.count, param.p.dataType,
          param.p.op, param.comm, param.stream));
        break;
      case mscclFuncReduceScatter:
        NCCLCHECK(ncclReduceScatter(param.p.sendBuff, param.p.recvBuff, param.p.count, param.p.dataType,
          param.p.op, param.comm, param.stream));
        break;
      case mscclFuncAllGather:
        NCCLCHECK(ncclAllGather(param.p.sendBuff, param.p.recvBuff, param.p.count, param.p.dataType,
          param.comm, param.stream));
        break;
      case mscclFuncSend:
        NCCLCHECK(ncclSend(param.p.sendBuff, param.p.count, param.p.dataType,
          param.p.peer, param.comm, param.stream));
        break;
      case mscclFuncRecv:
        NCCLCHECK(ncclRecv(param.p.recvBuff, param.p.count, param.p.dataType,
          param.p.peer, param.comm, param.stream));
        break;
      case mscclFuncGather:
        NCCLCHECK(ncclGather(param.p.sendBuff, param.p.recvBuff, param.p.count, param.p.dataType,
          param.p.root, param.comm, param.stream));
        break;
      case mscclFuncScatter:
        NCCLCHECK(ncclScatter(param.p.sendBuff, param.p.recvBuff, param.p.count, param.p.dataType,
          param.p.root, param.comm, param.stream));
        break;
      case mscclFuncAllToAll:
        NCCLCHECK(ncclAllToAll(param.p.sendBuff, param.p.recvBuff, param.p.count, param.p.dataType,
          param.comm, param.stream));
        break;
      case mscclFuncAllToAllv:
        NCCLCHECK(ncclAllToAllv(
          param.p.sendBuff, param.p.sendCounts, param.p.sDisPls,
          param.p.recvBuff, param.p.recvCounts, param.p.rDisPls,
          param.p.dataType, param.comm, param.stream));
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
  NCCLCHECK(mscclSetSavedSchedulerParam(
    sendBuff, sendCounts, sDisPls, recvBuff, recvCounts, rDisPls,
    count, dataType, root, peer, op, func, comm, stream,
    &status.savedSchedulerParams.back()));

  switch (status.groupStatus) {
    case mscclNoGroup:
      if (status.fallbackComms.find(comm) == status.fallbackComms.end()) {
        CUDACHECK(hipStreamGetCaptureInfo(stream, &captureStatus, &pid));
        if (captureStatus == hipStreamCaptureStatusNone) {
          NCCLCHECK(mscclCallScheduler(&status.savedSchedulerParams.back()));
          if (status.savedSchedulerParams.back().p.scheduled) {
            NCCLCHECK(mscclRunSavedParams());
            break;
          }
        }
      }
      NCCLCHECK(mscclFallBackSavedParams());
      break;
    case mscclGroupSupportedOp:
      if (status.fallbackComms.find(comm) == status.fallbackComms.end()) {
        CUDACHECK(hipStreamGetCaptureInfo(stream, &captureStatus, &pid));
        if (captureStatus == hipStreamCaptureStatusNone) {
          NCCLCHECK(mscclCallScheduler(&status.savedSchedulerParams.back()));
          if (status.savedSchedulerParams.back().p.scheduled) {
            // Only save counts and displs when there is suitable MSCCL algorithm for this
            NCCLCHECK(mscclSaveCountsAndDispls(&status.savedSchedulerParams.back()));
            break;
          }
        }
      }
      status.groupStatus = mscclGroupUnsupportedOp;
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
  std::lock_guard<std::mutex> lock(mscclLifecycleMutex);
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
  status.savedSchedulerParams.clear();
  status.connectedAlgos.clear();
  status.fallbackComms.clear();
  if (status.mscclSchedulerPtr) {
    NCCLCHECK(status.mscclSchedulerPtr->tearDown());
    status.mscclSchedulerPtr = nullptr;
    dlclose(status.mscclSchedulerLib);
    status.mscclSchedulerLib = nullptr;
  }
  mscclInitialized.store(false, std::memory_order_release);
  return ncclSuccess;
}
