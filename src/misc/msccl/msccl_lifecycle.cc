/*************************************************************************
 * Copyright (c) Microsoft Corporation.
 * Licensed under the MIT License.
 ************************************************************************/

#include <atomic>
#include <map>
#include <mutex>
#include <set>

#include <dirent.h>
#include <dlfcn.h>
#include <error.h>
#include <link.h>

#include "alloc.h"
#include "checks.h"
#include "graph/topo.h"

#include "msccl/msccl_lifecycle.h"
#include "msccl/msccl_parser.h"
#include "msccl/msccl_setup.h"
#include "msccl/msccl_status.h"

#ifdef ENABLE_MSCCLPP
#include "mscclpp/mscclpp_nccl.h"
#endif

RCCL_PARAM(MscclEnabled, "MSCCL_ENABLE", 1);
RCCL_PARAM(MscclForceEnabled, "MSCCL_FORCE_ENABLE", 0);
static const char* mscclAlgoFilePathEnv = "MSCCL_ALGO_FILE_PATH";

bool mscclEnabled() {
#ifdef COMPILE_MSCCL_KERNEL
  return rcclParamMscclEnabled();
#else
  return false;
#endif
}

bool mscclForceEnabled() {
#ifdef COMPILE_MSCCL_KERNEL
  return rcclParamMscclForceEnabled();
#else
  return false;
#endif
}

void mscclSetIsCallerFlag() {
  mscclGetThreadLocalStatus().mscclIsCallerFlag = true;
}

void mscclClearIsCallerFlag() {
  mscclGetThreadLocalStatus().mscclIsCallerFlag = false;
}

bool mscclIsCaller() {
  return mscclGetThreadLocalStatus().mscclIsCallerFlag;
}

bool mscclAvailable(int rank) {
  return mscclEnabled() && mscclInitialized(rank);
}

static bool mscclCommCompatible(ncclComm_t comm) {
  // MSCCL is always compatible now. No need to guard against multi-thread.
  return true;
}

const char *mscclFuncNames[] = {
            "mscclFuncReduce",
            "mscclFuncBroadcast",
            "mscclFuncAllReduce",
            "mscclFuncReduceScatter",
            "mscclFuncAllGather",
            "mscclFuncSend",
            "mscclFuncRecv",
            "mscclFuncGather",
            "mscclFuncScatter",
            "mscclFuncAllToAll",
            "mscclFuncAllToAllv",
          };

static const char* mscclSchedulerPathEnv = "MSCCL_SCHEDULER";
static const char* mscclSchedulerDefaultPath = "libmsccl-scheduler.so";
static const char* mscclAlgoDirEnv = "MSCCL_ALGO_DIR";
static const char* mscclAlgoDefaultDir = "msccl-algorithms";
extern "C" bool mscclUnitTestMode() __attribute__((__weak__));
static const char* mscclUnitTestAlgoDefaultDir = "msccl-unit-test-algorithms";
static const char* mscclAlgoShareDirPath = "../share/rccl/msccl-algorithms";
static const char* mscclUnitTestAlgoShareDirPath = "../share/rccl/msccl-unit-test-algorithms";

static ncclResult_t mscclInternalSchedulerInit(ncclComm_t comm, int* numChannelsRequired) {
  static thread_local bool mscclAlgoMetaLoaded = false;
  mscclStatus& status = mscclGetStatus(comm->rank);

  int maxNchannels = *numChannelsRequired;
  *numChannelsRequired = 0;
  // Query numChannelsRequired from loaded algorithm metas
  if (mscclAlgoMetaLoaded) {
    for (auto& m : status.algoMetas) {
      if (comm->nRanks == m.nRanks) {
        if(m.nChannels <= maxNchannels) {
          *numChannelsRequired = std::max(*numChannelsRequired, m.nChannels);
        } else {
          WARN("NCCL_MAX_NCHANNELS:%d is lesser than number of channels required by MSCCL:%d, so disabling MSCCL for %s between minBytes:%ld and maxBytes:%ld from file: %s", \
                maxNchannels, m.nChannels, mscclFuncNames[m.func], m.minBytes, m.maxBytes, m.filePath.c_str());
        }
      }
    }
    return ncclSuccess;
  }

  const char* mscclAlgoDir = getenv(mscclAlgoDirEnv);
  const char* mscclAlgoShareDir = nullptr;
  std::string mscclAlgoDirStr;
  std::string mscclAlgoShareDirStr;
  const char *fullDirPath = nullptr;
  if (mscclAlgoDir == nullptr) {
    // Try to find default algorithm directory based on librccl.so path
    Dl_info dl_info;
    struct link_map *link_map_ptr = nullptr;
    if (!dladdr1((void *)mscclInternalSchedulerInit, &dl_info, (void **)&link_map_ptr, RTLD_DL_LINKMAP)) {
      WARN("MSCCL Internal Scheduler: dladdr1 failed");
      return ncclInvalidUsage;
    }
    std::string selfLibPath = link_map_ptr->l_name;
    mscclAlgoDirStr = selfLibPath.substr(0, selfLibPath.find_last_of("/\\") + 1);
    mscclAlgoDirStr += (mscclUnitTestMode && mscclUnitTestMode()) ? mscclUnitTestAlgoDefaultDir : mscclAlgoDefaultDir;
    mscclAlgoDir = mscclAlgoDirStr.c_str();
    // Get share Directory Paths
    mscclAlgoShareDirStr = selfLibPath.substr(0, selfLibPath.find_last_of("/\\") + 1);
    mscclAlgoShareDirStr += (mscclUnitTestMode && mscclUnitTestMode()) ? mscclUnitTestAlgoShareDirPath : mscclAlgoShareDirPath;
    mscclAlgoShareDir = mscclAlgoShareDirStr.c_str();
  }
  struct dirent *entry = nullptr;
  DIR *dp = nullptr;
  dp = opendir(mscclAlgoDir);
  if (dp == nullptr) {
    //Try to find the algorithm directory under share folder based on librccl.so path
    dp = opendir(mscclAlgoShareDir);
    if (dp == nullptr) {
      WARN("MSCCL Internal Scheduler:  Unable to find algorithm files in %s. Please ensure that RCCL has been installed properly" , mscclAlgoShareDir);
      return ncclInvalidUsage;
    }
    fullDirPath = mscclAlgoShareDir;
  } else {
    fullDirPath = mscclAlgoDir;
  }
  INFO(NCCL_INIT, "Using MSCCL files from %s", fullDirPath);
  while ((entry = readdir(dp))) {
    if (entry->d_type != DT_LNK && entry->d_type != DT_REG) {
      continue;
    }
    status.algoMetas.emplace_back();
    std::string fullPath = fullDirPath;
    fullPath += "/";
    fullPath += entry->d_name;
    NCCLCHECK(mscclGetAlgoMetaFromXmlFile(fullPath.c_str(), &(status.algoMetas.back())));
    if (status.algoMetas.back().nRanks == comm->nRanks) {
      if(status.algoMetas.back().nChannels <= maxNchannels) {
        *numChannelsRequired = std::max(*numChannelsRequired, status.algoMetas.back().nChannels);
      } else {
        WARN("NCCL_MAX_NCHANNELS:%d is lesser than number of channels required by MSCCL:%d, so disabling MSCCL for %s between minBytes:%ld and maxBytes:%ld from file: %s", \
	      maxNchannels, status.algoMetas.back().nChannels, mscclFuncNames[status.algoMetas.back().func], status.algoMetas.back().minBytes, status.algoMetas.back().maxBytes, \
	      status.algoMetas.back().filePath.c_str());
        status.algoMetas.pop_back();
      }
    }
  }
  if (closedir(dp)) {
    WARN("MSCCL Internal Scheduler: closedir failed, error %d", errno);
    return ncclInvalidUsage;
  }
  status.rankToAlgoHandles.resize(status.algoMetas.size());
  mscclAlgoMetaLoaded = true;
  return ncclSuccess;
}

ncclResult_t mscclSchedulerInit(ncclComm_t comm, int* numChannelsRequired) {
  comm->mscclCompatible = mscclCommCompatible(comm);
  if (!comm->mscclCompatible) {
    return ncclSuccess;
  }

  mscclStatus& status = mscclGetStatus(comm->rank);
  bool useInternalScheduler = false;

  const char* mscclSchedulerPath = getenv(mscclSchedulerPathEnv);
  if (mscclSchedulerPath) {
    status.mscclSchedulerLib = dlopen(mscclSchedulerPath, RTLD_NOW | RTLD_LOCAL);
  } else {
    status.mscclSchedulerLib = dlopen(mscclSchedulerDefaultPath, RTLD_NOW | RTLD_LOCAL);
  }
  if (status.mscclSchedulerLib == nullptr) {
    INFO(NCCL_INIT, "MSCCL: No external scheduler found, using internal implementation");
    useInternalScheduler = true;
  } else {
    status.mscclSchedulerPtr = (mscclSchedulerInterface *)dlsym(status.mscclSchedulerLib, "mscclScheduler");
    if (status.mscclSchedulerPtr == nullptr) {
      INFO(NCCL_INIT, "MSCCL: Failed to find mscclScheduler symbol, using internal implementation");
      useInternalScheduler = true;
    }
  }

  if (useInternalScheduler) {
    NCCLCHECK(mscclInternalSchedulerInit(comm, numChannelsRequired));
  } else {
    NCCLCHECK(status.mscclSchedulerPtr->init());
    *numChannelsRequired = MAXCHANNELS;
  }

  return ncclSuccess;
}

ncclResult_t mscclInit(ncclComm_t comm) {
  {
    mscclStatus& status = mscclGetStatus(comm->rank);

    // freeAlgoHandles and needsProxy are initialized globally once and before algorithm pre-processing and connection
    if (!mscclInitialized(comm->rank)) {
      status.freeAlgoHandles.resize(MSCCL_MAX_NUM_ALGOS);
      for (int i = 0; i < MSCCL_MAX_NUM_ALGOS; i++) {
        status.freeAlgoHandles[i] = MSCCL_MAX_NUM_ALGOS - i - 1;
      }
      status.needsProxy = false;
    }

    // Pre-process all algorithms for internal scheduler and for different comms.
    // This is a temp fix to bypass the issue that stream cannot be synchronized during HIP graph capturing,
    // should use dynamic loading approach after the issue is fixed.
    if (comm->mscclCompatible && !status.mscclSchedulerPtr) {
      for (size_t i = 0; i < status.algoMetas.size(); i++) {
        auto &m = status.algoMetas[i];
        if (m.nRanks == comm->nRanks) {
          // Load algorithms
          if (status.rankToAlgoHandles[i].find(comm->rank) == status.rankToAlgoHandles[i].end()) {
            static std::mutex loadAlgoMutex;
            std::lock_guard<std::mutex> lock(loadAlgoMutex);
            NCCLCHECK(mscclLoadAlgo(m.filePath.c_str(), &(status.rankToAlgoHandles[i][comm->rank]), comm->rank));
          }
          // Connect algorithms
          mscclAlgoHandle_t mscclAlgoHandle = status.rankToAlgoHandles[i][comm->rank];
          if (status.connectedAlgos[comm].find(mscclAlgoHandle) == status.connectedAlgos[comm].end()) {
            NCCLCHECK(mscclSetupConnections(status.hostAlgos[mscclAlgoHandle], comm));
            status.connectedAlgos[comm].insert(mscclAlgoHandle);
          }
        }
      }
    }

    if (mscclInitialized(comm->rank)) {
      return ncclSuccess;
    }

    status.workIndex = 1;
    NCCLCHECK(ncclCudaCalloc(&status.syncFlags, MSCCL_MAX_NUM_THREAD_BLOCKS));
    status.lastStream = nullptr;
    NCCLCHECK(mscclInitWorkFifoStatus(&(status.defaultWorkFifoStatus)));

    mscclSetInitialized(comm->rank);
  }

  INFO(NCCL_INIT, "MSCCL: Initialization finished, localSize %ld", mscclKernMaxLocalSize());
  return ncclSuccess;
}

ncclResult_t mscclGroupStart() {
  mscclThreadLocalStatus& threadLocalStatus = mscclGetThreadLocalStatus();
  threadLocalStatus.groupDepth++;
  if (threadLocalStatus.groupStatus == mscclNoGroup) {
    threadLocalStatus.groupStatus = mscclGroupSupportedOp;
  }
  return ncclSuccess;
}

static ncclResult_t mscclInternalSchedulerSelectAlgo(int rank, struct mscclSchedulerParam* param) {
  mscclStatus& status = mscclGetStatus(rank);
  param->scheduled = false;

  // Current MSCCL doesn't support pre/post op
  if (param->op >= ncclAvg) {
    return ncclSuccess;
  }

  // Whether the algorithm is in-place
  bool isInPlace = false;
  if (param->func == mscclFuncReduce ||
      param->func == mscclFuncBroadcast ||
      param->func == mscclFuncAllReduce ||
      param->func == mscclFuncAllToAll ||
      param->func == mscclFuncAllToAllv) {
    isInPlace = param->sendBuff == param->recvBuff;
  } else if (param->func == mscclFuncAllGather ||
             param->func == mscclFuncGather) {
    isInPlace = (char*)param->sendBuff == (char*)param->recvBuff + param->rank * param->count * ncclTypeSize(param->dataType);
  } else if (param->func == mscclFuncReduceScatter ||
             param->func == mscclFuncScatter) {
    isInPlace = (char*)param->recvBuff == (char*)param->sendBuff + param->rank * param->count * ncclTypeSize(param->dataType);
  }

  // Search suitable algorithms
  for (size_t i = 0; i < status.algoMetas.size(); i++) {
    auto &m = status.algoMetas[i];
    size_t nBytes = param->count * ncclTypeSize(param->dataType) * m.sizeMultiplier;
    bool msgSizeIsValid =
      param->count > 0 && (param->count % m.nChunksPerLoop) == 0 &&
      nBytes >= m.minBytes && (m.maxBytes == 0 || nBytes <= m.maxBytes);
    if (msgSizeIsValid &&
        m.nRanks == param->nRanks &&
        m.func == param->func &&
        (isInPlace ? m.inPlace : m.outOfPlace)) {
      param->handle = status.rankToAlgoHandles[i][param->rank];
      param->scheduled = true;
      return ncclSuccess;
    }
  }

  return ncclSuccess;
}

static ncclResult_t mscclSchedulerSelectAlgo(struct mscclSavedSchedulerParam* param) {
  mscclStatus& status = mscclGetStatus(param->comm->rank);
  if (status.mscclSchedulerPtr) {
    NCCLCHECK(status.mscclSchedulerPtr->selectAlgo(&(param->p)));
  } else {
    // Disable MSCCL algorithms if machine type is not matching
    if (param->comm->topo->mscclEnabled || mscclForceEnabled()) {
      NCCLCHECK(mscclInternalSchedulerSelectAlgo(param->comm->rank, &(param->p)));
    } else {
      param->p.scheduled = false;
    }
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
  param->p.opCount = comm->opCount;
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
  mscclThreadLocalStatus& threadLocalStatus = mscclGetThreadLocalStatus();
  for (auto& param : threadLocalStatus.savedSchedulerParams) {
    INFO(NCCL_COLL,"%s: opCount %lx sendbuff %p recvbuff %p count %zi datatype %d op %d root %d comm %p [nranks=%d] stream %p task %d globalrank %d",
    mscclFuncNames[param.p.func], param.p.opCount, param.p.sendBuff, param.p.recvBuff, param.p.count,
    param.p.dataType, param.p.op, param.p.root, param.comm, param.p.nRanks, param.stream, param.comm->tasks.nTasksP2p + param.comm->tasks.nTasksColl, param.comm->localRankToRank[param.comm->localRank]);

    NCCLCHECK(mscclRunAlgo(
      param.p.sendBuff, param.p.sendCounts, param.p.sDisPls,
      param.p.recvBuff, param.p.recvCounts, param.p.rDisPls,
      param.p.count, param.p.dataType, param.p.root, param.p.peer, param.p.op, param.p.handle, param.comm, param.stream));
  }
  threadLocalStatus.savedSchedulerParams.clear();
  return ncclSuccess;
}

static ncclResult_t mscclFallBackSavedParams() {
  mscclThreadLocalStatus& threadLocalStatus = mscclGetThreadLocalStatus();
  mscclSetIsCallerFlag();
  for (auto& param : threadLocalStatus.savedSchedulerParams) {
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
  threadLocalStatus.savedSchedulerParams.clear();
  return ncclSuccess;
}

ncclResult_t mscclEnqueueCheck(
    const void* sendBuff, const size_t sendCounts[], const size_t sDisPls[],
    void* recvBuff, const size_t recvCounts[], const size_t rDisPls[],
    size_t count, ncclDataType_t dataType, int root, int peer, ncclRedOp_t op,
    mscclFunc_t func, ncclComm_t comm, hipStream_t stream) {
  mscclThreadLocalStatus& threadLocalStatus = mscclGetThreadLocalStatus();

  threadLocalStatus.savedSchedulerParams.push_back({});
  NCCLCHECK(mscclSetSavedSchedulerParam(
    sendBuff, sendCounts, sDisPls, recvBuff, recvCounts, rDisPls,
    count, dataType, root, peer, op, func, comm, stream,
    &threadLocalStatus.savedSchedulerParams.back()));

  size_t nBytes = count * ncclTypeSize(dataType);

  switch (threadLocalStatus.groupStatus) {
    case mscclNoGroup:
#ifdef ENABLE_MSCCLPP
      if (comm->mscclppCompatible) {
        /* check if one rank per GPU and graph mode is enabled */
        if ((threadLocalStatus.captureStatus != mscclNoCapture) && comm->mscclCompatible) {
          if (func == mscclFuncAllReduce && nBytes <= comm->mscclpp_threshold && (nBytes & 31) == 0) {
            INFO(NCCL_COLL,"%s: opCount %lx sendbuff %p recvbuff %p count %zi datatype %d op %d root %d comm %p [nranks=%d] stream %p",
              "mscclpp_ncclAllReduce", comm->opCount, sendBuff, recvBuff, count, dataType, op, root, comm, comm->nRanks, stream);
            NCCLCHECK(mscclpp_ncclAllReduce(sendBuff, recvBuff, count, dataType, op, comm->mscclpp_comm, stream));
            threadLocalStatus.savedSchedulerParams.clear();
            break;
          }
          else if (func == mscclFuncAllGather && nBytes * comm->nRanks <= comm->mscclpp_threshold) {
            INFO(NCCL_COLL,"%s: opCount %lx sendbuff %p recvbuff %p count %zi datatype %d op %d root %d comm %p [nranks=%d] stream %p",
              "mscclpp_ncclAllGather", comm->opCount, sendBuff, recvBuff, count, dataType, op, root, comm, comm->nRanks, stream);
            NCCLCHECK(mscclpp_ncclAllGather(sendBuff, recvBuff, count, dataType, comm->mscclpp_comm, stream));
            threadLocalStatus.savedSchedulerParams.clear();
            break;
          }
        }
      }
#endif
      if (comm->mscclCompatible) {
          NCCLCHECK(mscclSchedulerSelectAlgo(&threadLocalStatus.savedSchedulerParams.back()));
          if (threadLocalStatus.savedSchedulerParams.back().p.scheduled) {
            NCCLCHECK(mscclRunSavedParams());
            break;
          }
      }
      NCCLCHECK(mscclFallBackSavedParams());
      break;
    case mscclGroupSupportedOp:
#ifdef ENABLE_MSCCLPP
      if (comm->mscclppCompatible) {
        /* check if one rank per GPU and graph mode is enabled */
        if ((threadLocalStatus.captureStatus != mscclNoCapture) && comm->mscclCompatible) {
          if (func == mscclFuncAllReduce && nBytes <= comm->mscclpp_threshold && (nBytes & 31) == 0) {
            INFO(NCCL_COLL,"%s: opCount %lx sendbuff %p recvbuff %p count %zi datatype %d op %d root %d comm %p [nranks=%d] stream %p",
              "mscclpp_ncclAllReduce", comm->opCount, sendBuff, recvBuff, count, dataType, op, root, comm, comm->nRanks, stream);
            NCCLCHECK(mscclpp_ncclAllReduce(sendBuff, recvBuff, count, dataType, op, comm->mscclpp_comm, stream));
            threadLocalStatus.savedSchedulerParams.clear();
            break;
          }
          else if (func == mscclFuncAllGather && nBytes * comm->nRanks <= comm->mscclpp_threshold) {
            INFO(NCCL_COLL,"%s: opCount %lx sendbuff %p recvbuff %p count %zi datatype %d op %d root %d comm %p [nranks=%d] stream %p",
              "mscclpp_ncclAllGather", comm->opCount, sendBuff, recvBuff, count, dataType, op, root, comm, comm->nRanks, stream);
            NCCLCHECK(mscclpp_ncclAllGather(sendBuff, recvBuff, count, dataType, comm->mscclpp_comm, stream));
            threadLocalStatus.savedSchedulerParams.clear();
            break;
          }
        }
      }
#endif
      if (comm->mscclCompatible) {
          NCCLCHECK(mscclSchedulerSelectAlgo(&threadLocalStatus.savedSchedulerParams.back()));
          if (threadLocalStatus.savedSchedulerParams.back().p.scheduled) {
            // Only save counts and displs when there is suitable MSCCL algorithm for this
            NCCLCHECK(mscclSaveCountsAndDispls(&threadLocalStatus.savedSchedulerParams.back()));
            break;
          }
        }
      threadLocalStatus.groupStatus = mscclGroupUnsupportedOp;
      NCCLCHECK(mscclFallBackSavedParams());
    case mscclGroupUnsupportedOp:
      NCCLCHECK(mscclFallBackSavedParams());
      break;
    default:
      return ncclInvalidUsage;
  }
  return ncclSuccess;
}

ncclResult_t mscclGroupEnd() {
  mscclThreadLocalStatus& threadLocalStatus = mscclGetThreadLocalStatus();
  threadLocalStatus.groupDepth--;
  if (threadLocalStatus.groupDepth == 0) {
    if (threadLocalStatus.groupStatus == mscclGroupSupportedOp) {
      NCCLCHECK(mscclRunSavedParams());
    }
    threadLocalStatus.groupStatus = mscclNoGroup;
  }
  return ncclSuccess;
}

static ncclResult_t mscclInternalUnloadAlgo(int rank, mscclAlgoHandle_t mscclAlgoHandle) {
  mscclStatus& status = mscclGetStatus(rank);

  free(status.hostAlgos[mscclAlgoHandle]);
  status.hostAlgos.erase(mscclAlgoHandle);

  NCCLCHECK(ncclCudaFree(status.devAlgos[mscclAlgoHandle]));
  status.devAlgos.erase(mscclAlgoHandle);

  status.freeAlgoHandles.push_back(mscclAlgoHandle);

  for (auto &s : status.connectedAlgos) {
    s.second.erase(mscclAlgoHandle);
  }

  return ncclSuccess;
}

static ncclResult_t mscclInternalSchedulerTeardown(int rank) {
  ncclResult_t ret = ncclSuccess, tmpRet = ncclSuccess;
  mscclStatus& status = mscclGetStatus(rank);
  for (auto &m : status.rankToAlgoHandles) {
    for (auto &p : m) {
      tmpRet = mscclInternalUnloadAlgo(rank, p.second);
      if (ret == ncclSuccess) {
        ret = tmpRet;
      }
    }
  }
  status.algoMetas.clear();
  status.rankToAlgoHandles.clear();
  return ret;
}

ncclResult_t mscclTeardown(int rank) {
  {
    if (!mscclInitialized(rank)) {
      mscclRemoveRank(rank);
      return ncclSuccess;
    }
    mscclStatus& status = mscclGetStatus(rank);
    for (auto &p : status.hostAlgos) {
      free(p.second);
      status.freeAlgoHandles.push_back(p.first);
    }
    for (auto &p : status.devAlgos) {
      CUDACHECK(hipFree(p.second));
    }
    CUDACHECK(hipFree(status.syncFlags));
    status.hostAlgos.clear();
    status.devAlgos.clear();
    status.freeAlgoHandles.clear();
    for (auto &p : status.scratchBuffers) {
      CUDACHECK(hipFree(p.second));
    }
    status.scratchBuffers.clear();
    status.connectedAlgos.clear();
    if (status.mscclSchedulerPtr) {
      NCCLCHECK(status.mscclSchedulerPtr->teardown());
      status.mscclSchedulerPtr = nullptr;
      dlclose(status.mscclSchedulerLib);
      status.mscclSchedulerLib = nullptr;
    } else {
      NCCLCHECK(mscclInternalSchedulerTeardown(rank));
    }
    NCCLCHECK(mscclDestroyWorkFifoStatus(&(status.defaultWorkFifoStatus)));
    for (auto &p : status.graphWorkFifoStatus) {
      NCCLCHECK(mscclDestroyWorkFifoStatus(&(p.second)));
    }
    mscclSetInitialized(rank, false);
    mscclRemoveRank(rank);
  }

  INFO(NCCL_INIT, "MSCCL: Teardown finished");
  return ncclSuccess;
}
