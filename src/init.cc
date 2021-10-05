/*************************************************************************
 * Copyright (c) 2015-2021, NVIDIA CORPORATION. All rights reserved.
 * Modifications Copyright (c) 2019-2021 Advanced Micro Devices, Inc. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

#include "nccl.h"
#include "channel.h"
#include "nvmlwrap.h"
#include "gdrwrap.h"
#include "bootstrap.h"
#include "transport.h"
#include "group.h"
#include "net.h"
#include "coll_net.h"
#include "enqueue.h"
#include "graph.h"
#include "argcheck.h"
#include <fcntl.h>
#include <unistd.h>
#include <hip/hip_runtime.h>
#include <string.h>
#include <errno.h>
#include <assert.h>
#include <dlfcn.h>
#include <sys/types.h>
#include <sys/stat.h>
#include <unistd.h>
#include "graph/topo.h"

// [RCCL]
#include "clique/CliqueManager.h"
#include <hsa/hsa_ext_amd.h>
// [/RCCL]

#define STR2(v) #v
#define STR(v) STR2(v)

#ifdef ENABLE_TRACE
std::chrono::high_resolution_clock::time_point ncclEpoch;
#endif

#if CUDART_VERSION >= 9020 || defined(__HIP_PLATFORM_HCC__) || defined(__HCC__) || defined(__HIPCC__)
#define NCCL_GROUP_CUDA_STREAM 0 // CGMD: CUDA 9.2,10.X Don't need to use an internal CUDA stream
#else
#define NCCL_GROUP_CUDA_STREAM 1 // CGMD: CUDA 9.0,9.1 Need to use an internal CUDA stream
#endif

const char* ncclFuncStr[NCCL_NUM_FUNCTIONS+1] = { "Broadcast", "Reduce", "AllGather", "ReduceScatter", "AllReduce", "SendRecv" };
const char* ncclAlgoStr[NCCL_NUM_ALGORITHMS] = { "Tree", "Ring", "CollNet" };
const char* ncclProtoStr[NCCL_NUM_PROTOCOLS] = { "LL", "LL128", "Simple" };
const char* ncclRedOpStr[ncclNumOps] = { "Sum", "Prod", "Max", "Min" };
const char *ncclTypeStr[ncclNumTypes] = {"_i8", "_u8", "_i32", "_u32", "_i64", "_u64", "_f16", "_f32", "_f64", "_b16"};

NCCL_PARAM(GroupCudaStream, "GROUP_CUDA_STREAM", NCCL_GROUP_CUDA_STREAM);

NCCL_PARAM(CheckPointers, "CHECK_POINTERS", 0);

ncclNet_t* ncclNet = NULL;
ncclCollNet_t* ncclCollNet = NULL;

struct allocationTracker allocTracker[MAX_ALLOC_TRACK_NGPU] = {};

// Returns ncclInternalError if anything fails, causing that network to be ignored.
ncclResult_t initNet(ncclNet_t* net) {
  int ndev;
  if (net->init(ncclDebugLog) != ncclSuccess) return ncclInternalError;
  if (net->devices(&ndev) != ncclSuccess) return ncclInternalError;
  if (ndev <= 0) return ncclSystemError;
  return ncclSuccess;
}

ncclResult_t initCollNet(ncclCollNet_t* collnet) {
  int ndev;
  if (collnet->init(ncclDebugLog) != ncclSuccess) return ncclInternalError;
  if (collnet->devices(&ndev) != ncclSuccess) return ncclInternalError;
  if (ndev <= 0) return ncclSystemError;
  return ncclSuccess;
}

ncclResult_t initNetPlugin(ncclNet_t** net, ncclCollNet_t** collnet) {
  void* netPluginLib = dlopen("librccl-net.so", RTLD_NOW | RTLD_LOCAL);
  if (netPluginLib == NULL) {
    // dlopen does not guarantee to set errno, but dlerror only gives us a
    // string, so checking errno doesn't hurt to try to provide a better
    // error message
    if (errno == ENOENT) {
      INFO(NCCL_INIT|NCCL_NET, "NET/Plugin : No plugin found (librccl-net.so), using internal implementation");
    } else {
      INFO(NCCL_INIT|NCCL_NET, "NET/Plugin : Plugin load returned %d : %s.", errno, dlerror());
    }
    return ncclSuccess;
  }
  *net = (ncclNet_t*) dlsym(netPluginLib, STR(NCCL_PLUGIN_SYMBOL));
  if (*net == NULL) {
    INFO(NCCL_INIT|NCCL_NET, "NET/Plugin: Failed to find " STR(NCCL_PLUGIN_SYMBOL) " symbol.");
    if (netPluginLib != NULL) dlclose(netPluginLib);
    return ncclSuccess;
  }
  // Check for CollNet
  *collnet = (ncclCollNet_t*) dlsym(netPluginLib, STR(NCCL_COLLNET_PLUGIN_SYMBOL));
  if (*collnet == NULL) {
    INFO(NCCL_INIT|NCCL_NET, "NET/Plugin: Failed to find " STR(NCCL_COLLNET_PLUGIN_SYMBOL) " symbol.");
  }
  return ncclSuccess;
}

ncclResult_t initNet() {
  // Always initialize bootstrap network
  NCCLCHECK(bootstrapNetInit());

  // Initialize main communication network
  ncclNet_t* nets[3] = { NULL, &ncclNetIb, &ncclNetSocket };
  ncclCollNet_t* collNets[3] = { NULL, NULL, NULL };
  NCCLCHECK(initNetPlugin(nets+0, collNets+0));
  char* netName = getenv("NCCL_NET");

  for (int i=0; i<3; i++) {
    if (nets[i] == NULL) continue;
    if (netName && strcmp(netName, nets[i]->name) != 0) continue;
    // net plugin is already initialized
    if (initNet(nets[i]) != ncclSuccess) continue;
    ncclNet = nets[i];
    if (collNets[i] && initCollNet(collNets[i]) == ncclSuccess) {
      ncclCollNet = collNets[i];
    }
    break;
  }

  if (ncclNet == NULL) {
    WARN("Error: network %s not found.", netName ? netName : "");
    return ncclInvalidUsage;
  }
  return ncclSuccess;
}

// GDRCOPY support: Off by default
NCCL_PARAM(GdrCopyEnable, "GDRCOPY_ENABLE", 0);

// GDRCOPY support
gdr_t ncclGdrCopy = NULL;

ncclResult_t initGdrCopy() {
  if (ncclParamGdrCopyEnable() == 1) {
    ncclGdrCopy = ncclGdrInit();
  }
  return ncclSuccess;
}

NCCL_PARAM(CollNetEnable, "COLLNET_ENABLE", 0);

pthread_mutex_t initLock = PTHREAD_MUTEX_INITIALIZER;
static bool initialized = false;
static size_t maxLocalSizeBytes = 0;
static ncclResult_t ncclInit() {
  if (initialized) return ncclSuccess;
  pthread_mutex_lock(&initLock);
  if (!initialized) {
    initEnv();
    initGdrCopy();
    maxLocalSizeBytes = ncclKernMaxLocalSize();
    NCCLCHECK(initNet());
    INFO(NCCL_INIT, "Using network %s", ncclNetName());
    initialized = true;
  }
  pthread_mutex_unlock(&initLock);
  return ncclSuccess;
}

NCCL_API(ncclResult_t, ncclGetVersion, int* version);
ncclResult_t ncclGetVersion(int* version) {
  if (version == NULL) return ncclInvalidArgument;
  *version = NCCL_VERSION_CODE;
  return ncclSuccess;
}

NCCL_API(ncclResult_t, ncclGetUniqueId, ncclUniqueId* out);
ncclResult_t ncclGetUniqueId(ncclUniqueId* out) {
  NCCLCHECK(ncclInit());
  NCCLCHECK(PtrCheck(out, "GetUniqueId", "out"));
  return bootstrapGetUniqueId(out);
}

// Prevent compiler from optimizing out these operations
#ifdef __clang__
#define NCCL_NO_OPTIMIZE __attribute__((optnone))
#else
#define NCCL_NO_OPTIMIZE __attribute__((optimize("O0")))
#endif

void NCCL_NO_OPTIMIZE commPoison(ncclComm_t comm) {
  comm->rank = comm->cudaDev = comm->busId = comm->nRanks = -1;
}

RCCL_PARAM(KernelCollTraceEnable, "KERNEL_COLL_TRACE_ENABLE", 0);

#ifdef ENABLE_COLLTRACE
void *ncclCommThreadMain(void *arg) {
  ncclComm_t comm = (ncclComm_t)arg;
  int head = comm->hostDevComm.collTraceHead;
  #define MAX_NAME_LENGTH 32
  char* func_names = (char *)malloc(MAX_NAME_LENGTH*(FUNC_INDEX_P2P+1));
  for (int func = 0; func < NCCL_NUM_FUNCTIONS; func++) {
    for (int al = 0; al < NCCL_NUM_ALGORITHMS; al++) {
      for (int type = 0; type < ncclNumTypes; type++) {
        for (int pr = 0; pr < NCCL_NUM_PROTOCOLS; pr++) {
          for (int redop = 0; redop < ncclNumOps; redop++) {
            char* line = func_names+MAX_NAME_LENGTH*FUNC_INDEX(func, redop, type, al, pr);
            sprintf(line, "%s%s%s%s%s", ncclFuncStr[func], ncclAlgoStr[al], ncclProtoStr[pr],
              ncclRedOpStr[redop], ncclTypeStr[type]);
          }
        }
      }
    }
  }
  char* line = func_names+MAX_NAME_LENGTH*FUNC_INDEX_P2P;
  sprintf(line, "%s", ncclFuncStr[NCCL_NUM_FUNCTIONS]);
  do {
    int tail = LOAD(comm->hostDevComm.collTraceTail)%COLLTRACE_NUM_ITEMS;
    int count;
    if (head <= tail)
      count = tail - head;
    else
      count = COLLTRACE_NUM_ITEMS + head - tail;
    if (!count) {
      if(LOAD(&comm->hostDevComm.collTraceExit))
        break;
      else {
        usleep(1000); //sleep 1ms
        continue;
      }
    }
    for (int i = 0; i < count; i++) {
      struct ncclCollTrace *td = comm->hostDevComm.collTrace+head;
      uint8_t type = LOAD(&(td->type));
      if (type == ncclCollTraceNotReady)
        break;
      char line[1024];
      int offset = 0;
      uint16_t fIdx = td->funcIndex;
      #define VEGA_GPU_RTC_FREQUENCY 2.5E7
      if (type == ncclCollTraceDataType) {
        sprintf(line, "## [%12.6f] [%02d:%02d] L:%04d DT %08x %016lx %016lx",
          (double)(td->timeStamp)/VEGA_GPU_RTC_FREQUENCY, comm->rank, td->bid,
          fIdx, td->data_0, td->opCount, td->data_1);
      } else {
        sprintf(line, "## [%12.6f] [%02d:%02d] %06lx",
          (double)(td->timeStamp)/VEGA_GPU_RTC_FREQUENCY, comm->rank, td->bid, td->opCount);
        offset = strlen(line);
        switch (type) {
          case ncclCollTraceKernelLaunchType:
            sprintf(line+offset, " KL HWID %8x %s ",
              td->data_0, func_names+MAX_NAME_LENGTH*fIdx);
            offset = strlen(line);
            if (fIdx > FUNC_INDEX_P2P)
              sprintf(line+offset, "ERROR bad function index %d", fIdx);
            else if (fIdx == FUNC_INDEX_P2P)
              sprintf(line+offset, "nt %d dt %d busId %lx nRanks %d", td->p2p.nThreads, td->p2p.delta, comm->busId, comm->nRanks);
            else
              sprintf(line+offset, "nt %d bi %d nc %d busId %lx nRanks %d", td->coll.nThreads, td->coll.bid, td->coll.nChannels, comm->busId, comm->nRanks);
            break;
          case ncclCollTraceCollEndType:
            if (fIdx != 0xffff) {
              sprintf(line+offset, " CE %s ", func_names+MAX_NAME_LENGTH*fIdx);
              offset = strlen(line);
              if (fIdx > FUNC_INDEX_P2P)
                sprintf(line+offset, "ERROR bad function index %d", fIdx);
              else if (fIdx == FUNC_INDEX_P2P)
                sprintf(line+offset, "nt %d dt %d busId %lx nRanks %d", td->p2p.nThreads, td->p2p.delta, comm->busId, comm->nRanks);
              else
                sprintf(line+offset, "nt %d bi %d nc %d busId %lx nRanks %d", td->coll.nThreads, td->coll.bid, td->coll.nChannels, comm->busId, comm->nRanks);
            }
            else
              sprintf(line+offset, " KE busId %lx nRanks %d", comm->busId, comm->nRanks);
            break;
          case ncclCollTraceAbortType:
            sprintf(line+offset, " Abort");
            break;
          default:
            sprintf(line+offset, " unknown collective trace data type");
            break;
        }
      }
      INFO(NCCL_COLL, "%s", line);
      STORE(&(td->type), ncclCollTraceNotReady);
      head ++;
      head %= COLLTRACE_NUM_ITEMS;
    }
  } while(1);
  free(func_names);
  comm->hostDevComm.collTraceHead = head;
  pthread_exit(NULL);
}
#endif

#undef NCCL_NO_OPTIMIZE

static ncclResult_t commFree(ncclComm_t comm) {
  if (comm == NULL)
    return ncclSuccess;
  free(comm->connectSend);
  free(comm->connectRecv);
  for (int peer=0; peer<comm->nRanks; peer++) {
    delete comm->p2pSends[peer];
    delete comm->p2pRecvs[peer];
  }
  free(comm->p2pSends);
  free(comm->p2pRecvs);
  free(comm->asyncOps);

#ifdef ENABLE_PROFILING
  struct ncclProf* prof = (struct ncclProf*)malloc(sizeof(struct ncclProf));
  CUDACHECK(hipMemcpy(prof, comm->hostDevComm.devProf, sizeof(struct ncclProf), hipMemcpyDeviceToHost));
  uint64_t total_cycle = 0, wait_cycle = 0, send_cycle = 0, directSend_cycle = 0, recv_cycle = 0, \
    directRecv_cycle = 0, copySend_cycle = 0, directCopySend_cycle = 0, recvCopySend_cycle = 0, directRecvCopySend_cycle = 0, \
    recvReduceCopy_cycle = 0, recvReduceSend_cycle = 0, recvReduceCopySend_cycle = 0, directRecvReduceCopySend_cycle = 0, \
    send_byte = 0, directSend_byte = 0, recv_byte = 0, directRecv_byte = 0, copySend_byte = 0, directCopySend_byte = 0, \
    recvCopySend_byte = 0, directRecvCopySend_byte = 0, recvReduceCopy_byte = 0, recvReduceSend_byte = 0, \
    recvReduceCopySend_byte = 0, directRecvReduceCopySend_byte = 0;
  for (int chan=0; chan<comm->nChannels; chan++) {
    total_cycle += prof->elems[chan].total_cycle;
    wait_cycle += prof->elems[chan].wait_cycle;
    send_cycle += prof->elems[chan].send_cycle;
    directSend_cycle += prof->elems[chan].directSend_cycle;
    recv_cycle += prof->elems[chan].recv_cycle;
    directRecv_cycle += prof->elems[chan].directRecv_cycle;
    copySend_cycle += prof->elems[chan].copySend_cycle;
    directCopySend_cycle += prof->elems[chan].directCopySend_cycle;
    recvCopySend_cycle += prof->elems[chan].recvCopySend_cycle;
    directRecvCopySend_cycle += prof->elems[chan].directRecvCopySend_cycle;
    recvReduceCopy_cycle += prof->elems[chan].recvReduceCopy_cycle;
    recvReduceSend_cycle += prof->elems[chan].recvReduceSend_cycle;
    recvReduceCopySend_cycle += prof->elems[chan].recvReduceCopySend_cycle;
    directRecvReduceCopySend_cycle += prof->elems[chan].directRecvReduceCopySend_cycle;
    send_byte += prof->elems[chan].send_byte;
    directSend_byte += prof->elems[chan].directSend_byte;
    recv_byte += prof->elems[chan].recv_byte;
    directRecv_byte += prof->elems[chan].directRecv_byte;
    copySend_byte += prof->elems[chan].copySend_byte;
    directCopySend_byte += prof->elems[chan].directCopySend_byte;
    recvCopySend_byte += prof->elems[chan].recvCopySend_byte;
    directRecvCopySend_byte += prof->elems[chan].directRecvCopySend_byte;
    recvReduceCopy_byte += prof->elems[chan].recvReduceCopy_byte;
    recvReduceSend_byte += prof->elems[chan].recvReduceSend_byte;
    recvReduceCopySend_byte += prof->elems[chan].recvReduceCopySend_byte;
    directRecvReduceCopySend_byte += prof->elems[chan].directRecvReduceCopySend_byte;
  }
  #define VEGA_GPU_RTC_FREQUENCY 2.5E7
  if (comm->rank == 0) {
    INFO(NCCL_INIT, "# %4s %6s %6s %6s %6s %7s %6s %6s %6s %6s %6s", "Rank", "total", "  wait", "send", "rcRdS", "dRcRdCS", "dRcCS", "dRc", "cS", "rc", "rcCS");
    INFO(NCCL_INIT, "# %4s %6s %6s %6s %6s %7s %6s %6s %6s %6s %6s", "", "(s)", "(s)", "(GB/s)", "(GB/s)", "(GB/s)", "(GB/s)", "(GB/s)", "(GB/s)", "(GB/s)", "(GB/s)");
  }
  INFO(NCCL_INIT, "# %4d %6.4f %6.4f %6.2f %6.2f %7.2f %6.2f %6.2f %6.2f %6.2f %6.2f",
    comm->rank, (double)total_cycle/VEGA_GPU_RTC_FREQUENCY/comm->nChannels,
    (double)wait_cycle/VEGA_GPU_RTC_FREQUENCY/comm->nChannels,
    (send_cycle) ? (double)send_byte*comm->nChannels/((double)send_cycle/VEGA_GPU_RTC_FREQUENCY*1.0E9) : 0,
    (recvReduceSend_cycle) ? (double)recvReduceSend_byte*comm->nChannels/((double)recvReduceSend_cycle/VEGA_GPU_RTC_FREQUENCY*1.0E9) : 0,
    (directRecvReduceCopySend_cycle) ? (double)directRecvReduceCopySend_byte*comm->nChannels/((double)directRecvReduceCopySend_cycle/VEGA_GPU_RTC_FREQUENCY*1.0E9) : 0,
    (directRecvCopySend_cycle) ? (double)directRecvCopySend_byte*comm->nChannels/((double)directRecvCopySend_cycle/VEGA_GPU_RTC_FREQUENCY*1.0E9) : 0,
    (directRecv_cycle) ? (double)directRecv_byte*comm->nChannels/((double)directRecv_cycle/VEGA_GPU_RTC_FREQUENCY*1.0E9) : 0,
    (copySend_cycle) ? (double)copySend_byte*comm->nChannels/((double)copySend_cycle/VEGA_GPU_RTC_FREQUENCY*1.0E9) : 0,
    (recv_cycle) ? (double)recv_byte*comm->nChannels/((double)recv_cycle/VEGA_GPU_RTC_FREQUENCY*1.0E9) : 0,
    (recvCopySend_cycle) ? (double)recvCopySend_byte*comm->nChannels/((double)recvCopySend_cycle/VEGA_GPU_RTC_FREQUENCY*1.0E9) : 0);
  free(prof);
  CUDACHECK(hipFree(comm->hostDevComm.devProf));

  for (int channel=0; channel<std::max(comm->nChannels, comm->p2pnChannels); channel++) {
    if (comm->channels[channel].send_byte) INFO(NCCL_INIT, "# [%03d:%02d] Proxy Send %6.2f GB/s (%ld bytes %d measurements)",
      comm->rank, channel, (comm->channels[channel].bw_count) ?
      (float)comm->channels[channel].bw_cumulative/comm->channels[channel].bw_count : 0,
      comm->channels[channel].send_byte, comm->channels[channel].bw_count);
    if (comm->channels[channel].recv_byte) INFO(NCCL_INIT, "# [%03d:%02d] Proxy Recv %6.2f GB/s (%ld bytes %d measurements)",
      comm->rank, channel, (comm->channels[channel].bw_count) ?
      (float)comm->channels[channel].bw_cumulative/comm->channels[channel].bw_count : 0,
      comm->channels[channel].recv_byte, comm->channels[channel].bw_count);
  }
#endif

#ifdef ENABLE_COLLTRACE
  STORE(&comm->hostDevComm.collTraceExit, 1);
  if (comm->hostDevComm.collTraceThread) pthread_join(comm->hostDevComm.collTraceThread, NULL);
  NCCLCHECK(ncclCudaHostFree((void *)comm->hostDevComm.collTrace));
  NCCLCHECK(ncclCudaHostFree((void *)comm->hostDevComm.collTraceTail));
#endif

  free(comm->peerInfo);
  ncclTopoFree(comm->topo);

  if (comm->bootstrap)
    NCCLCHECK(bootstrapClose(comm->bootstrap));

  CUDACHECK(hipFree((ncclDevCommAndChannels*)comm->devComm));

  for (int channel=0; channel<MAXCHANNELS; channel++)
    NCCLCHECK(freeChannel(comm->channels+channel, comm->nRanks));

  if (comm->doneEvent != NULL)
    CUDACHECK(hipEventDestroy(comm->doneEvent));

  if (comm->intDoneEvent != NULL)
    CUDACHECK(hipEventDestroy(comm->intDoneEvent));

  if (comm->launchMode == ncclComm::GROUP) {
    CUDACHECK(hipStreamDestroy(comm->groupStream));
  }

  ncclDestroyQueueInfo(comm->enqueueInfo);

  // Last rank frees shared resources between threads
  int isLast;
  NCCLCHECK(ncclCpuBarrierIn(comm, &isLast));
  if (isLast) {
    free(comm->intraBarrier);
    free(comm->intraParams);
    free(comm->intraCudaDevs);
    free(comm->intraCGMode);
    free(comm->intraCC);
  }
  NCCLCHECK(ncclCudaHostFree((void *)comm->abortFlag));

  // Poison comm to try and catch a double free
  commPoison(comm);

  free(comm);
  return ncclSuccess;
}

RCCL_PARAM(CliqueIgnoreTopo, "CLIQUE_IGNORE_TOPO", 0);
RCCL_PARAM(P2pNetDisable, "P2P_NET_DISABLE", 0);
NCCL_PARAM(AggChannelSize, "AGG_CHANNEL_SIZE", -2);

static ncclResult_t commAlloc(ncclComm_t* comret, int ndev, int rank) {
  if (ndev < 1) {
    WARN("invalid device count (%d) requested", ndev);
    return ncclInvalidArgument;
  }
  if (rank >= ndev || rank < 0) {
    WARN("rank %d exceeds ndev=%d", rank, ndev);
    return ncclInvalidArgument;
  }

  // Try to create a CUDA object right away. If there is something wrong with
  // the device we're on (failure cause #1) , better know it early.
  hipEvent_t doneEvent;
  CUDACHECK(hipEventCreateWithFlags(&doneEvent, hipEventDisableTiming));
  hipEvent_t intDoneEvent;
  CUDACHECK(hipEventCreateWithFlags(&intDoneEvent, hipEventDisableTiming));

  struct ncclComm* comm;
  NCCLCHECK(ncclCalloc(&comm, 1));

  comm->rank = comm->hostDevComm.rank = rank;
  comm->nRanks = comm->hostDevComm.nRanks = ndev;
  hipGetDevice(&comm->cudaDev);
  NCCLCHECK(getBusId(comm->cudaDev, &comm->busId));
  TRACE(NCCL_INIT,"comm %p rank %d nranks %d cudaDev %d busId %lx", comm, rank, ndev, comm->cudaDev, comm->busId);

  comm->doneEvent = doneEvent;
  comm->intDoneEvent = intDoneEvent;
  comm->checkPointers = ncclParamCheckPointers() == 1 ? true : false;
#if CUDART_VERSION >= 9020 || defined(__HIP_PLATFORM_HCC__) || defined(__HCC__) || defined(__HIPCC__)
  comm->groupCudaStream = ncclParamGroupCudaStream();
#else
  // Don't allow the user to overload the default setting in older CUDA builds
  comm->groupCudaStream = NCCL_GROUP_CUDA_STREAM;
#endif
  comm->fatalError = ncclSuccess;

  NCCLCHECK(ncclCudaHostCalloc((uint32_t**)&comm->abortFlag, 1));
  comm->hostDevComm.abortFlag = comm->abortFlag;
  *comm->abortFlag = 0;

  comm->collOpCount = 0;
  comm->p2pOpCount = 0x8000;

  comm->argsptr = &comm->args;
#ifdef ENABLE_PROFILING
  NCCLCHECK(ncclCudaCalloc(&comm->hostDevComm.devProf, 1));
#endif

#ifdef ENABLE_COLLTRACE
  NCCLCHECK(ncclCudaHostCalloc(&comm->hostDevComm.collTraceTail, 1));
  NCCLCHECK(ncclCudaHostCalloc(&comm->hostDevComm.collTrace, COLLTRACE_NUM_ITEMS));
  memset(comm->hostDevComm.collTrace, 0, sizeof(struct ncclCollTrace) * COLLTRACE_NUM_ITEMS);
  comm->hostDevComm.collTraceExit = comm->hostDevComm.collTraceHead = *comm->hostDevComm.collTraceTail = 0;
  if ((ncclDebugLevel >= NCCL_LOG_INFO) && rcclParamKernelCollTraceEnable())
    pthread_create(&comm->hostDevComm.collTraceThread, NULL, ncclCommThreadMain, (void *)comm);
  else
    comm->hostDevComm.collTraceThread = 0;
#endif
  comm->collNetSupport = 0;

  NCCLCHECK(ncclCalloc(&comm->asyncOps, NCCL_MAX_OPS));
  comm->asyncOpCount = 0;
  comm->asyncTotalSize = 0;
  comm->channelSize = ncclParamAggChannelSize();
  comm->asyncAllocMode = ncclComm::ROUND_ROBIN;
  char* str = getenv("NCCL_AGG_ALLOC_MODE");
  if (str) INFO(NCCL_ENV, "NCCL_AGG_ALLOC_MODE set by environment to %s", str);
  if (str && strcmp(str, "SHORTEST_QUEUE") == 0) {
    comm->asyncAllocMode = ncclComm::SHORTEST_QUEUE;
  }

  NCCLCHECK(ncclCreateQueueInfo(&comm->enqueueInfo, comm));
  comm->lastSetupNode = NULL;
  comm->lastCudaGraphId = -1;

  CUDACHECK(hipDriverGetVersion(&comm->driverVersion));

  static_assert(MAXCHANNELS <= sizeof(*comm->connectSend)*8, "comm->connectSend must have enough bits for all channels");
  static_assert(MAXCHANNELS <= sizeof(*comm->connectRecv)*8, "comm->connectRecv must have enough bits for all channels");
  NCCLCHECK(ncclCalloc(&comm->connectSend, comm->nRanks*NCCL_MAX_CONNS));
  NCCLCHECK(ncclCalloc(&comm->connectRecv, comm->nRanks*NCCL_MAX_CONNS));

  comm->p2pSendCount = comm->p2pRecvCount = 0;
  NCCLCHECK(ncclCalloc(&comm->p2pSends, comm->nRanks));
  NCCLCHECK(ncclCalloc(&comm->p2pRecvs, comm->nRanks));

  // Mark channels as non initialized.
  for (int c=0; c<MAXCHANNELS; c++) comm->channels[c].id = -1;

  *comret = comm;
  return ncclSuccess;
}

static ncclResult_t devCommSetup(ncclComm_t comm) {
  ncclDevCommAndChannels *devCommAndChans;
  NCCLCHECK(ncclCudaCalloc(&devCommAndChans, 1));
  comm->devComm = &devCommAndChans->comm;
  comm->hostDevComm.channels = devCommAndChans->channels;

  // Duplicate the channels on the device
  int nChannels = std::max(comm->nChannels, comm->p2pnChannels);
  NCCLCHECK(ncclCudaMemcpy(comm->hostDevComm.channels, comm->channels, nChannels));

  // Copy userRanks and peers
  for (int r=0; r<comm->nChannels; r++) {
    NCCLCHECK(ncclCudaMemcpy(comm->channels[r].ring.devUserRanks, comm->channels[r].ring.userRanks, comm->nRanks));
  }

  // Duplicate the dev comm on the device
  NCCLCHECK(ncclCudaMemcpy(comm->devComm, &comm->hostDevComm, 1));
  return ncclSuccess;
}

// Pre-process the string so that running "strings" on the lib can quickly reveal the version.
#if defined(__HIP_PLATFORM_HCC__) || defined(__HCC__) || defined(__HIPCC__)
#define VERSION_STRING "RCCL version " STR(NCCL_MAJOR) "." STR(NCCL_MINOR) "." STR(NCCL_PATCH) NCCL_SUFFIX "+hip" STR(HIP_VERSION_MAJOR) "." STR(HIP_VERSION_MINOR)
#else
#define VERSION_STRING "NCCL version " STR(NCCL_MAJOR) "." STR(NCCL_MINOR) "." STR(NCCL_PATCH) NCCL_SUFFIX "+cuda" STR(CUDA_MAJOR) "." STR(CUDA_MINOR)
#endif
static void showVersion() {
  static int shown = 0;
  if (shown == 0 && ncclDebugLevel >= NCCL_LOG_VERSION) {
    printf("%s\n", VERSION_STRING);
    fflush(stdout);
    if (ncclDebugFile != stdout)
      INFO(NCCL_ALL,"%s", VERSION_STRING); // Also log NCCL version in one of the files
    shown = 1;
  }
}

static ncclResult_t fillInfo(struct ncclComm* comm, struct ncclPeerInfo* info, uint64_t commHash) {
  info->rank = comm->rank;
  CUDACHECK(hipGetDevice(&info->cudaDev));
  info->hostHash=getHostHash()+commHash;
  info->pidHash=getPidHash()+commHash;

  // Get the device MAJOR:MINOR of /dev/shm so we can use that
  // information to decide whether we can use SHM for inter-process
  // communication in a container environment
  struct stat statbuf;
  SYSCHECK(stat("/dev/shm", &statbuf), "stat");
  info->shmDev = statbuf.st_dev;

  info->busId = comm->busId;

  NCCLCHECK(ncclGpuGdrSupport(&info->gdrSupport));
  return ncclSuccess;
}

static ncclResult_t setupChannel(struct ncclComm* comm, int channelId, int rank, int nranks, int* ringRanks) {
  TRACE(NCCL_INIT, "rank %d nranks %d", rank, nranks);
  NCCLCHECK(initChannel(comm, channelId));

  struct ncclRing* ring = &comm->channels[channelId].ring;
  // Find our ring-distance from rank zero and reorganize ranks to start with rank.
  int ixZero=0, ixRank=0;
  for (int i=0; i < nranks; i++) {
    if (ringRanks[i] == 0) ixZero = i;
    if (ringRanks[i] == rank) ixRank = i;
  }
  ring->index = (ixRank-ixZero + nranks)%nranks;
  for (int i=0; i<nranks; i++) {
    ring->userRanks[i] = ringRanks[(i+ixRank)%nranks];
  }
  return ncclSuccess;
}

void* waitForNonNullPtr(void* p) {
  volatile void** ptr = (volatile void**) p;
  while (*ptr == NULL) sched_yield();
  return (void*)*ptr;
}

ncclResult_t initParams(struct ncclComm* comm) {
  hipLaunchParams* params = comm->myParams = comm->intraParams+comm->intraRank;
  params->args = (void **)&comm->argsptr;
  params->stream = NULL;
  params->sharedMem = 0;
  params->blockDim.x = 0; params->blockDim.y = params->blockDim.z = 1;
  params->gridDim.x = 0; params->gridDim.y = params->gridDim.z = 1;
  return ncclSuccess;
}

// Allocate/Set Intra Process Structures and set CG options
ncclResult_t ncclCommSetIntraProc(struct ncclComm* comm, int rank, int ranks, struct ncclComm* comm0) {
  comm->intraRank = rank;
  comm->intraRanks = ranks;
  comm->intraPhase = 0;

  // Alloc shared structures
  if (rank == 0) {
    assert(comm == comm0);
    int* bar;
    NCCLCHECK(ncclCalloc(&bar, 2));
    bar[0] = bar[1] = 0;
    comm->intraBarrier = bar;
    NCCLCHECK(ncclCalloc(&comm->intraParams, comm->intraRanks));
    NCCLCHECK(ncclCalloc(&comm->intraCudaDevs, comm->intraRanks));
    int* CGMode;
    NCCLCHECK(ncclCalloc(&CGMode, 1));
    *CGMode = 0x11;
    comm->intraCGMode = CGMode;
    int* CC;
    NCCLCHECK(ncclCalloc(&CC, 1));
    *CC = ncclCudaCompCap();
    comm->intraCC = CC;
  } else {
    comm->intraBarrier = (int*)waitForNonNullPtr(&comm0->intraBarrier);
    comm->intraParams = (hipLaunchParams*)waitForNonNullPtr(&comm0->intraParams);
    comm->intraCudaDevs = (int*)waitForNonNullPtr(&comm0->intraCudaDevs);
    comm->intraCGMode = (int*)waitForNonNullPtr(&comm0->intraCGMode);
    comm->intraCC = (int*)waitForNonNullPtr(&comm0->intraCC);
  }
  comm->intraCudaDevs[comm->intraRank] = comm->cudaDev;
  NCCLCHECK(initParams(comm));

  int cgMdLaunch = 1;

  // Set CG Mode
  comm->launchMode = ncclComm::PARALLEL;
  char* str = getenv("NCCL_LAUNCH_MODE");
  if (str) INFO(NCCL_ENV, "NCCL_LAUNCH_MODE set by environment to %s", str);
  if (str && strcmp(str, "GROUP") == 0) {
    comm->launchMode = ncclComm::GROUP;
  }
  if (comm->launchMode == ncclComm::GROUP) {
    CUDACHECK(hipStreamCreateWithFlags(&comm->groupStream, hipStreamNonBlocking));
    if (*comm->intraCC && (ncclCudaCompCap() == *comm->intraCC)) {
      // Check whether the GPU supports Cooperative Group Multi Device Launch
      hipError_t ret = hipDeviceGetAttribute(&cgMdLaunch, hipDeviceAttributeCooperativeMultiDeviceLaunch, comm->cudaDev);
      if (ret != hipSuccess) {
        INFO(NCCL_INIT, "hipDeviceGetAttribute(hipDeviceAttributeCooperativeMultiDeviceLaunch, %d) failed with %s",
                        comm->cudaDev, hipGetErrorString(ret));
        return ncclInternalError;
      }
      if (!cgMdLaunch) {
        INFO(NCCL_INIT, "Multi-GPU cooperative launch support not available for device %d", comm->cudaDev);
      }
    }
  }

  // Disable cgMdLaunch if any rank does not support it
  if (cgMdLaunch == 0) {
    *comm->intraCGMode = 0x10;
  }
  return ncclSuccess;
}

#define DEFAULT_LL_BUFFSIZE (NCCL_LL_LINES_PER_THREAD*NCCL_LL_MAX_NTHREADS*NCCL_STEPS*sizeof(union ncclLLFifoLine))
#define DEFAULT_LL128_BUFFSIZE (NCCL_LL128_ELEMS_PER_THREAD*NCCL_LL128_MAX_NTHREADS*NCCL_STEPS*sizeof(uint64_t))
#define DEFAULT_BUFFSIZE (1 << 22) /* 4MiB */
#define DEFAULT_BUFFSIZE_ARM (1 << 20) /* 1MiB */
NCCL_PARAM(BuffSize, "BUFFSIZE", -2);
NCCL_PARAM(LlBuffSize, "LL_BUFFSIZE", -2);
NCCL_PARAM(Ll128BuffSize, "LL128_BUFFSIZE", -2);

static ncclResult_t computeBuffSizes(struct ncclComm* comm) {
  int cpuArch, cpuVendor, cpuModel;
  NCCLCHECK(ncclTopoCpuType(comm->topo, &cpuArch, &cpuVendor, &cpuModel));

  int64_t envs[NCCL_NUM_PROTOCOLS] = { ncclParamLlBuffSize(), ncclParamLl128BuffSize(), ncclParamBuffSize() };
  int defaults[NCCL_NUM_PROTOCOLS] = { DEFAULT_LL_BUFFSIZE, DEFAULT_LL128_BUFFSIZE, DEFAULT_BUFFSIZE };

  if (cpuArch == NCCL_TOPO_CPU_ARCH_ARM) defaults[NCCL_PROTO_SIMPLE] = DEFAULT_BUFFSIZE_ARM;

  for (int p=0; p<NCCL_NUM_PROTOCOLS; p++) {
    comm->buffSizes[p] = comm->hostDevComm.buffSizes[p] = envs[p] != -2 ? envs[p] : defaults[p];
  }
  return ncclSuccess;
}

NCCL_PARAM(CrossNic, "CROSS_NIC", 2);
NCCL_PARAM(GraphDumpFileRank, "GRAPH_DUMP_FILE_RANK", 0);
NCCL_PARAM(CollNetNodeThreshold, "COLLNET_NODE_THRESHOLD", 2);
NCCL_PARAM(NvbPreconnect, "NVB_PRECONNECT", 1);

static ncclResult_t initTransportsRank(struct ncclComm* comm, ncclUniqueId* commId) {
  // We use 2 AllGathers
  // 1. { peerInfo, comm, compCap}
  // 2. { nChannels, graphInfo, topoRanks }

  int rank = comm->rank;
  int nranks = comm->nRanks;
  uint64_t commHash = getHash(commId->internal, NCCL_UNIQUE_ID_BYTES);
  TRACE(NCCL_INIT, "comm %p, commHash %lx, rank %d nranks %d - BEGIN", comm, commHash, rank, nranks);
  // [RCCL] Collect the PID of the root
  int rootPid;
  NCCLCHECK(bootstrapInit(commId, rank, nranks, &comm->bootstrap, &rootPid));
  // [/RCCL]

  // AllGather1 - begin
  struct {
    struct ncclPeerInfo peerInfo;
    struct ncclComm* comm;
    int cudaCompCap;
  } *allGather1Data;

  NCCLCHECK(ncclCalloc(&allGather1Data, nranks));
  allGather1Data[rank].comm = comm;
  allGather1Data[rank].cudaCompCap = ncclCudaCompCap();
  struct ncclPeerInfo* myInfo = &allGather1Data[rank].peerInfo;
  NCCLCHECK(fillInfo(comm, myInfo, commHash));
  NCCLCHECK(bootstrapAllGather(comm->bootstrap, allGather1Data, sizeof(*allGather1Data)));

  NCCLCHECK(ncclCalloc(&comm->peerInfo, nranks+1)); // Extra rank to represent CollNet root
  for (int i = 0; i < nranks; i++) {
    memcpy(comm->peerInfo+i, &allGather1Data[i].peerInfo, sizeof(struct ncclPeerInfo));
    if ((i != rank) && (comm->peerInfo[i].hostHash == myInfo->hostHash) && (comm->peerInfo[i].busId == myInfo->busId)) {
      WARN("Duplicate GPU detected : rank %d and rank %d both on CUDA device %lx", rank, i, myInfo->busId);
      return ncclInvalidUsage;
    }
  }

  // Compute intra ranks and minimum CUDA Compute capabilities of intra-node GPUs and all GPUs
  int intraProcRank0 = -1, intraProcRank = -1, intraProcRanks = 0;
  int intraNodeRank0 = -1, intraNodeRank = -1, intraNodeRanks = 0;
  int myCompCap = allGather1Data[rank].cudaCompCap;
  int minCompCap = myCompCap, maxCompCap = myCompCap;
  int intraNodeGlobalRanks[256];
  for (int i = 0; i < nranks; i++) {
    if (allGather1Data[i].peerInfo.hostHash == allGather1Data[rank].peerInfo.hostHash) {
      // Rank is on same node
      if (intraNodeRanks == 0) intraNodeRank0 = i;
      if (i == rank) intraNodeRank = intraNodeRanks;
      intraNodeGlobalRanks[intraNodeRanks++] = i;
      if (allGather1Data[i].peerInfo.pidHash == allGather1Data[rank].peerInfo.pidHash) {
        // Rank is in same process
        if (intraProcRanks == 0) intraProcRank0 = i;
        if (i == rank) intraProcRank = intraProcRanks;
        intraProcRanks++;
      }
    }
    minCompCap = std::min(allGather1Data[i].cudaCompCap, minCompCap);
    maxCompCap = std::max(allGather1Data[i].cudaCompCap, maxCompCap);
  }
  TRACE(NCCL_INIT,"hostHash[%d] %lx intraNodeRank %d intraNodeRanks %d intraNodeRank0 %d",
        rank, allGather1Data[rank].peerInfo.hostHash, intraNodeRank, intraNodeRanks, intraNodeRank0);
  TRACE(NCCL_INIT,"pidHash[%d] %lx intraProcRank %d intraProcRanks %d intraProcRank0 %d",
        rank, allGather1Data[rank].peerInfo.pidHash, intraProcRank, intraProcRanks, intraProcRank0);
  if (intraProcRank == -1 || intraProcRank0 == -1 || allGather1Data[intraProcRank0].comm == NULL) {
    WARN("Failed to determine intra proc ranks rank %d hostHash %lx pidHash %lx intraProcRank %d intraProcRanks %d intraProcRank0 %d",
         rank, allGather1Data[rank].peerInfo.hostHash, allGather1Data[rank].peerInfo.pidHash,
         intraProcRank, intraProcRanks, intraProcRank0);
    return ncclInternalError;
  }
  if (intraNodeRank == -1 || intraNodeRank0 == -1 || intraNodeRanks == 0) {
    WARN("Failed to determine intra node ranks rank %d hostHash %lx pidHash %lx intraNodeRank %d intraNodeRanks %d intraNodeRank0 %d",
         rank, allGather1Data[rank].peerInfo.hostHash, allGather1Data[rank].peerInfo.pidHash,
         intraNodeRank, intraNodeRanks, intraNodeRank0);
    return ncclInternalError;
  }
  struct ncclComm* intraProcRank0Comm = allGather1Data[intraProcRank0].comm;
  uint64_t intraNodeRank0pidHash = allGather1Data[intraNodeRank0].peerInfo.pidHash;

  // AllGather1 - end

  // Topo detection / System graph creation
  NCCLCHECK(ncclTopoGetSystem(comm, &comm->topo));
  // save nRanks to ncclTopoSystem as indicator of multi-node
  comm->topo->nRanks = comm->nRanks;
  // Compute paths between GPUs and NICs
  NCCLCHECK(ncclTopoComputePaths(comm->topo, comm->peerInfo));
  // Remove inaccessible GPUs and unused NICs
  NCCLCHECK(ncclTopoTrimSystem(comm->topo, comm));
  // Recompute paths after trimming
  NCCLCHECK(ncclTopoComputePaths(comm->topo, comm->peerInfo));
  // Init search
  NCCLCHECK(ncclTopoSearchInit(comm->topo));
  // Print final topology
  NCCLCHECK(ncclTopoPrint(comm->topo));

  // Get rings and trees
  struct ncclTopoGraph ringGraph;
  ringGraph.id = 0;
  ringGraph.pattern = NCCL_TOPO_PATTERN_RING;
  ringGraph.crossNic = ncclParamCrossNic();
  ringGraph.collNet = 0;
  ringGraph.minChannels = 1;
  ringGraph.maxChannels = MAXCHANNELS/2;
  NCCLCHECK(ncclTopoCompute(comm->topo, &ringGraph));
  NCCLCHECK(ncclTopoPrintGraph(comm->topo, &ringGraph));

  struct ncclTopoGraph treeGraph;
  treeGraph.id = 1;
  treeGraph.pattern = NCCL_TOPO_PATTERN_BALANCED_TREE;
  treeGraph.crossNic = ncclParamCrossNic();
  treeGraph.collNet = 0;
  treeGraph.minChannels = comm->topo->nodes[NET].count != 0 ? 1 : ringGraph.nChannels;
  treeGraph.maxChannels = ringGraph.nChannels;
  NCCLCHECK(ncclTopoCompute(comm->topo, &treeGraph));
  NCCLCHECK(ncclTopoPrintGraph(comm->topo, &treeGraph));

  struct ncclTopoGraph collNetGraph;
  collNetGraph.id = 2;
  collNetGraph.pattern = NCCL_TOPO_PATTERN_TREE;
  collNetGraph.collNet = 1;
  collNetGraph.crossNic = ncclParamCrossNic();
  collNetGraph.minChannels = 1;
  collNetGraph.maxChannels = ringGraph.nChannels;
  NCCLCHECK(ncclTopoCompute(comm->topo, &collNetGraph));
  NCCLCHECK(ncclTopoPrintGraph(comm->topo, &collNetGraph));

  bool allXgmi = true;
  { // [RCCL] Check if clique-based kernels can be enabled and initialize CliqueManager
    CliqueManager::cliqueMode_t cliqueMode = CliqueManager::CLIQUE_DISABLED;
    if (comm->localRanks == comm->nRanks && comm->topo->nodes[GPU].nodes[0].gpu.gcn != 910)
    {
      // Check that all the GPUs have peer access to one another and are XGMI connected
      bool hasPeerAccess = true;
      for (int i = 0; i < nranks && hasPeerAccess; i++)
      {
        int cudaDev1 = allGather1Data[i].peerInfo.cudaDev;
        for (int j = 0; j < nranks; j++)
        {
          if (i == j) continue;
          int cudaDev2 = allGather1Data[j].peerInfo.cudaDev;
          int p2p;
          if (hipDeviceCanAccessPeer(&p2p, cudaDev1, cudaDev2) != hipSuccess || !p2p)
          {
            hasPeerAccess = false;
            break;
          }

          bool isXGMI;
          // Limit to single intermediate GPU for enabling clique
          NCCLCHECK(ncclTopoGetLinkType(comm->topo, i, j, &isXGMI, 1));
          allXgmi &= isXGMI;
        }
      }
      if (hasPeerAccess)
      {
        if (intraProcRanks == nranks)
          cliqueMode = CliqueManager::CLIQUE_SINGLE_PROCESS;
        else
          cliqueMode = CliqueManager::CLIQUE_SINGLE_NODE;
      }

      // For now, only enable clique-based kernels on nodes where all GPUs are XGMI connected
      if (!allXgmi && !rcclParamCliqueIgnoreTopo())
      {
        INFO(NCCL_INIT, "Disabling clique-based kernels due to topology (ignore with RCCL_CLIQUE_IGNORE_TOPO)");
        cliqueMode = CliqueManager::CLIQUE_DISABLED;
      }
    }
    comm->cliqueManager = new CliqueManager(rank, nranks, cliqueMode);
    NCCLCHECK(comm->cliqueManager->Init(commId, rootPid));
  } // [/RCCL]

  if (comm->rank == ncclParamGraphDumpFileRank()) {
    struct ncclTopoGraph* graphs[3] = { &ringGraph, &treeGraph, &collNetGraph };
    NCCLCHECK(ncclTopoDumpGraphs(comm->topo, 3, graphs));
  }

  // Determine local CollNet support before all-gather
  if (ncclParamCollNetEnable() == 1 && collNetSupport() == 1 && collNetGraph.nChannels > 0) comm->collNetSupport = 1;
  if (intraNodeRanks > 8) {
    if (comm->collNetSupport == 1) WARN("CollNet currently only supports up to 8 GPUs per node");
    comm->collNetSupport = 0;
  }

  if ((comm->topo->type & RCCL_TOPO_4P2H_ROME) && (comm->topo->type & RCCL_TOPO_GDR_ALL)) {
    if (rcclParamP2pNetDisable() == 0) {
      comm->p2pNet = 1;
      INFO(NCCL_INIT, "RCCL enabled same node P2P over network");
    }
    else
      INFO(NCCL_INIT, "RCCL force disabled same node P2P over network");
  }
  // AllGather3 - begin
  struct ncclGraphInfo {
    int pattern;
    int nChannels;
    int sameChannels;
    float speedIntra;
    float speedInter;
    int typeIntra;
    int typeInter;
  };

  struct {
    int collNetSupport;
    int nc;
    struct ncclGraphInfo tree;
    struct ncclGraphInfo ring;
    struct ncclGraphInfo collNet;
    struct ncclTopoRanks topoRanks;
  } *allGather3Data;

  NCCLCHECK(ncclCalloc(&allGather3Data, nranks));
  int idx;
  NCCLCHECK(ncclTopoIdToIndex(comm->topo, GPU, myInfo->busId, &idx));
  allGather3Data[rank].nc = 2;
  if (comm->topo->nodes[GPU].count == comm->topo->nRanks && comm->topo->nodes[GPU].nodes[idx].gpu.gcn == 906 && allXgmi)
    allGather3Data[rank].nc = 4;
  if (comm->topo->nodes[GPU].nodes[idx].gpu.gcn == 908)
    allGather3Data[rank].nc = std::max(4/ringGraph.nChannels, 2);
  if (comm->topo->nodes[GPU].count == comm->topo->nRanks && (comm->topo->type & RCCL_TOPO_CR8G))
    allGather3Data[rank].nc = 4;
  if (comm->topo->nodes[GPU].count == comm->topo->nRanks && comm->topo->nodes[GPU].nodes[idx].gpu.gcn == 910)
    allGather3Data[rank].nc = 4;
  allGather3Data[rank].tree.pattern = treeGraph.pattern;
  allGather3Data[rank].tree.nChannels = treeGraph.nChannels;
  allGather3Data[rank].tree.sameChannels = treeGraph.sameChannels;
  allGather3Data[rank].tree.speedIntra = treeGraph.speedIntra;
  allGather3Data[rank].tree.speedInter = treeGraph.speedInter;
  allGather3Data[rank].tree.typeIntra = treeGraph.typeIntra;
  allGather3Data[rank].tree.typeInter = treeGraph.typeInter;
  allGather3Data[rank].ring.pattern = ringGraph.pattern;
  allGather3Data[rank].ring.nChannels = ringGraph.nChannels;
  allGather3Data[rank].ring.sameChannels = ringGraph.sameChannels;
  allGather3Data[rank].ring.speedIntra = ringGraph.speedIntra;
  allGather3Data[rank].ring.speedInter = ringGraph.speedInter;
  allGather3Data[rank].ring.typeIntra = ringGraph.typeIntra;
  allGather3Data[rank].ring.typeInter = ringGraph.typeInter;
  allGather3Data[rank].collNet.pattern = collNetGraph.pattern;
  allGather3Data[rank].collNet.nChannels = collNetGraph.nChannels;
  allGather3Data[rank].collNet.sameChannels = collNetGraph.sameChannels;
  allGather3Data[rank].collNet.speedIntra = collNetGraph.speedIntra;
  allGather3Data[rank].collNet.speedInter = collNetGraph.speedInter;
  allGather3Data[rank].collNet.typeIntra = collNetGraph.typeIntra;
  allGather3Data[rank].collNet.typeInter = collNetGraph.typeInter;
  allGather3Data[rank].collNetSupport = comm->collNetSupport;

  comm->nChannels = (comm->topo->nodes[GPU].count != comm->topo->nRanks && comm->topo->nodes[NET].count)
    ? std::min(treeGraph.nChannels, ringGraph.nChannels) : ringGraph.nChannels;
  NCCLCHECK(ncclTopoPreset(comm, &treeGraph, &ringGraph, &allGather3Data[rank].topoRanks));

  NCCLCHECK(bootstrapAllGather(comm->bootstrap, allGather3Data, sizeof(*allGather3Data)));

  // Determine nNodes, firstRanks, ...
  int *nodesFirstRank, *nodesTreePatterns;
  NCCLCHECK(ncclCalloc(&nodesFirstRank, nranks));
  NCCLCHECK(ncclCalloc(&nodesTreePatterns, nranks));
  for (int i=0; i<nranks; i++) {
    int node = -1;
    int firstRank = allGather3Data[i].topoRanks.ringRecv[0];
    for (int n=0; n<comm->nNodes; n++) {
      if (nodesFirstRank[n] == firstRank) node = n;
    }
    if (node == -1) {
      node = comm->nNodes++;
      nodesFirstRank[node] = firstRank;
      // Record tree pattern of each node as they can be different depending on sm arch
      nodesTreePatterns[node] = allGather3Data[i].tree.pattern;
    }
    if (i == comm->rank) comm->node = node;
  }

  int nChannelsOrig = comm->nChannels;
  struct ncclTopoRanks** allTopoRanks;
  NCCLCHECK(ncclCalloc(&allTopoRanks, comm->nRanks));
  int nc = allGather3Data[0].nc;
  for (int i=0; i<nranks; i++) {
    allTopoRanks[i] = &allGather3Data[i].topoRanks;
    nc = std::min(allGather3Data[i].nc, nc);
    // Make sure we align all ranks so that the tuning is consistent across ranks
    treeGraph.nChannels = std::min(allGather3Data[i].tree.nChannels, treeGraph.nChannels);
    treeGraph.sameChannels = std::min(allGather3Data[i].tree.sameChannels, treeGraph.sameChannels);
    treeGraph.speedIntra = std::min(allGather3Data[i].tree.speedIntra, treeGraph.speedIntra);
    treeGraph.speedInter = std::min(allGather3Data[i].tree.speedInter, treeGraph.speedInter);
    treeGraph.typeIntra = std::min(allGather3Data[i].tree.typeIntra, treeGraph.typeIntra);
    treeGraph.typeInter = std::min(allGather3Data[i].tree.typeInter, treeGraph.typeInter);
    ringGraph.nChannels = std::min(allGather3Data[i].ring.nChannels, ringGraph.nChannels);
    ringGraph.sameChannels = std::min(allGather3Data[i].ring.sameChannels, ringGraph.sameChannels);
    ringGraph.speedIntra = std::min(allGather3Data[i].ring.speedIntra, ringGraph.speedIntra);
    ringGraph.speedInter = std::min(allGather3Data[i].ring.speedInter, ringGraph.speedInter);
    ringGraph.typeIntra = std::min(allGather3Data[i].ring.typeIntra, ringGraph.typeIntra);
    ringGraph.typeInter = std::min(allGather3Data[i].ring.typeInter, ringGraph.typeInter);
    collNetGraph.nChannels = std::min(allGather3Data[i].collNet.nChannels, collNetGraph.nChannels);
    collNetGraph.sameChannels = std::min(allGather3Data[i].collNet.sameChannels, collNetGraph.sameChannels);
    collNetGraph.speedIntra = std::min(allGather3Data[i].collNet.speedIntra, collNetGraph.speedIntra);
    collNetGraph.speedInter = std::min(allGather3Data[i].collNet.speedInter, collNetGraph.speedInter);
    collNetGraph.typeIntra = std::min(allGather3Data[i].collNet.typeIntra, collNetGraph.typeIntra);
    collNetGraph.typeInter = std::min(allGather3Data[i].collNet.typeInter, collNetGraph.typeInter);
    comm->collNetSupport = std::min(allGather3Data[i].collNetSupport, comm->collNetSupport);
  }

  comm->nChannels = treeGraph.nChannels = ringGraph.nChannels =
    (comm->topo->nodes[GPU].count != comm->topo->nRanks && comm->topo->nodes[NET].count)
    ? std::min(treeGraph.nChannels, ringGraph.nChannels) : ringGraph.nChannels;
  if (comm->nChannels < nChannelsOrig) {
    // We started duplicating channels during Preset(), so we need to move the
    // duplicated channels since we have removed some.
    for (int i=0; i<comm->nChannels; i++) memcpy(comm->channels+comm->nChannels+i, comm->channels+nChannelsOrig+i, sizeof(struct ncclChannel));
  }

  // Determine CollNet support after all-gather now that we know nNodes
  int collNetNodeThreshold = ncclParamCollNetNodeThreshold();
  if (comm->nNodes < collNetNodeThreshold) {
    if (comm->collNetSupport == 1)
      INFO(NCCL_INIT, "Communicator has %d nodes which is less than CollNet node threshold %d, disabling CollNet", comm->nNodes, collNetNodeThreshold);
    comm->collNetSupport = 0;
  }

  int *rings;
  NCCLCHECK(ncclCalloc(&rings, nranks*MAXCHANNELS));
  NCCLCHECK(ncclTopoPostset(comm, nodesFirstRank, nodesTreePatterns, allTopoRanks, rings, &collNetGraph, nc));

  free(allTopoRanks);
  free(nodesTreePatterns);
  free(nodesFirstRank);
  free(allGather1Data);
  free(allGather3Data);

  // AllGather3 - end

  TRACE(NCCL_INIT, "rank %d nranks %d - BUILT %d TREES/RINGS", rank, nranks, comm->nChannels);

  char line[1024];
  line[0]='\0';
  for (int c=0; c<comm->nChannels; c++) {
    struct ncclTree* tree = &comm->channels[c].tree;
    snprintf(line+strlen(line), 1023-strlen(line), " [%d] %d/%d/%d->%d->%d",
        c, tree->down[0], tree->down[1], tree->down[2], rank, tree->up);
    INFO(NCCL_GRAPH, "Ring %d : %d -> %d -> %d comm %p nRanks %02d busId %lx", c, comm->channels[c].ring.prev, 
         comm->rank, comm->channels[c].ring.next, comm, comm->nRanks, comm->busId);
  }
  line[1023] = '\0';
  INFO(NCCL_INIT, "Trees%s comm %p nRanks %02d busId %lx", line, comm, comm->nRanks, comm->busId);

  // Set Affinity to a CPU local the our GPU, so that all memory we allocate
  // on the host is local.
  NCCLCHECK(ncclTopoGetCpuAffinity(comm->topo, comm->rank, &comm->cpuAffinity));
  cpu_set_t affinitySave;
  if (CPU_COUNT(&comm->cpuAffinity)) {
    sched_getaffinity(0, sizeof(cpu_set_t), &affinitySave);
    sched_setaffinity(0, sizeof(cpu_set_t), &comm->cpuAffinity);
  }
  ncclResult_t ret;

  NCCLCHECK(computeBuffSizes(comm));

  // Connect with prev/next for each ring
  for (int c=0; c<comm->nChannels; c++) {
    struct ncclChannel* channel = comm->channels+c;
    NCCLCHECKGOTO(setupChannel(comm, c, rank, nranks, rings+c*nranks), ret, affinity_restore);
    if (comm->nRanks == 1) continue;
    NCCLCHECKGOTO(ncclTransportP2pConnect(comm, channel, 1, &channel->ring.prev, 1, &channel->ring.next, 0), ret, affinity_restore);
  }
  NCCLCHECKGOTO(ncclTransportP2pSetup(comm, &ringGraph, 0), ret, affinity_restore);
  if (ringGraph.nIntraChannels && rcclParamP2pNetDisable() == 0) {
    comm->useIntraNet = 1;
    // Connect NET for intranode use
    for (int c=0; c<comm->nChannels; c++) {
      struct ncclChannel* channel = comm->channels+c;
      if (comm->nRanks == 1) continue;
      NCCLCHECKGOTO(ncclTransportP2pConnect(comm, channel, 1, &channel->ring.prev, 1, &channel->ring.next, NCCL_CONN_IDX_P2P_NET), ret, affinity_restore);
    }
    NCCLCHECKGOTO(ncclTransportP2pSetup(comm, &ringGraph, NCCL_CONN_IDX_P2P_NET), ret, affinity_restore);
  }
  free(rings);
  INFO(NCCL_INIT, "Connected all rings comm %p nRanks %02d busId %lx", comm, comm->nRanks, comm->busId);

  // Connect Trees
  for (int c=0; c<comm->nChannels; c++) {
    struct ncclChannel* channel = comm->channels+c;
    if (comm->nRanks == 1) continue;
    NCCLCHECKGOTO(ncclTransportP2pConnect(comm, channel, NCCL_MAX_TREE_ARITY, channel->tree.down, 1, &channel->tree.up, 0), ret, affinity_restore);
    NCCLCHECKGOTO(ncclTransportP2pConnect(comm, channel, 1, &channel->tree.up, NCCL_MAX_TREE_ARITY, channel->tree.down, 0), ret, affinity_restore);
  }
  NCCLCHECKGOTO(ncclTransportP2pSetup(comm, &treeGraph, 0), ret, affinity_restore);
  INFO(NCCL_INIT, "Connected all trees comm %p nRanks %02d busId %lx", comm, comm->nRanks, comm->busId);

  // Check if we can setup CollNet
  if (comm->collNetSupport > 0) {
    int collNetSetupFail = 0;
    // Find all head ranks
    int nHeads = collNetGraph.nChannels;
    int *heads;
    NCCLCHECK(ncclCalloc(&heads, nHeads));
    // Head GPU index is always 0
    for (int c=0; c<nHeads; c++) {
      heads[c] = collNetGraph.intra[c*comm->localRanks+0];
    }
    for (int c=0; c<comm->nChannels; c++) {
      struct ncclChannel* channel = comm->channels+c;
      for (int h=0; h<nHeads; h++) {
        const int head = heads[h];
        collNetSetupFail = ncclTransportCollNetSetup(comm, &collNetGraph, channel, head, head, h, collNetRecv);
        if (!collNetSetupFail) collNetSetupFail = ncclTransportCollNetSetup(comm, &collNetGraph, channel, head, head, h, collNetSend);
      }
      // Verify CollNet setup across ranks after trying the first channel
      if (c == 0) {
        NCCLCHECKGOTO(ncclTransportCollNetCheck(comm, collNetSetupFail), ret, collnet_cleanup);
      }
    }
    // Verify CollNet setup across ranks after trying all channels
    NCCLCHECKGOTO(ncclTransportCollNetCheck(comm, collNetSetupFail), ret, collnet_cleanup);
    TRACE(NCCL_INIT, "rank %d Connected inter-node CollNet", rank);

    // Connect intra-node CollNet
    for (int c=0; c<comm->nChannels; c++) {
      struct ncclChannel* channelRecv = comm->channels+c;
      NCCLCHECKGOTO(ncclTransportP2pConnect(comm, channelRecv, NCCL_MAX_DIRECT_ARITY, channelRecv->collTree.up, NCCL_MAX_DIRECT_ARITY, channelRecv->collTree.down, 0), ret, collnet_cleanup);
    }
    NCCLCHECKGOTO(ncclTransportP2pSetup(comm, &collNetGraph, 0), ret, collnet_cleanup);
    for (int c=0; c<comm->nChannels; c++) {
      struct ncclChannel* channelSend = comm->channels+c;
      NCCLCHECKGOTO(ncclTransportP2pConnect(comm, channelSend, NCCL_MAX_DIRECT_ARITY, channelSend->collTree.down, NCCL_MAX_DIRECT_ARITY, channelSend->collTree.up, 1), ret, collnet_cleanup);
    }
    NCCLCHECKGOTO(ncclTransportP2pSetup(comm, &collNetGraph, 1), ret, collnet_cleanup);
    INFO(NCCL_INIT, "rank %d Connected CollNet comm %p nRanks %02d", rank, comm, comm->nRanks);

collnet_cleanup:
    free(heads);
    if (ret != ncclSuccess) {
      NCCLCHECK(ncclTransportCollNetFree(comm));
      comm->collNetSupport = 0;
      ret = ncclSuccess;
    }
  }
  TRACE(NCCL_INIT, "rank %d nranks %d - CONNECTED %d RINGS AND TREES", rank, nranks, comm->nChannels);

  // Compute time models for algorithm and protocol combinations
  NCCLCHECK(ncclTopoTuneModel(comm, minCompCap, maxCompCap, &treeGraph, &ringGraph, &collNetGraph, comm->topo->nodes[GPU].nodes[0].gpu.gcn));

  // Compute nChannels per peer for p2p
  NCCLCHECK(ncclTopoComputeP2pChannels(comm));

  if (ncclParamNvbPreconnect()) {
    // Connect p2p when using NVB path
    int nvbNpeers;
    int* nvbPeers;
    NCCLCHECK(ncclTopoGetNvbGpus(comm->topo, comm->rank, &nvbNpeers, &nvbPeers));
    for (int r=0; r<nvbNpeers; r++) {
      int peer = nvbPeers[r];
      int delta = (comm->nRanks + (comm->rank-peer)) % comm->nRanks;
      for (int c=0; c<comm->p2pnChannelsPerPeer; c++) {
        int channelId = (delta+comm->p2pChannels[c]) % comm->p2pnChannels;
        if (comm->channels[channelId].peers[peer].recv[0].connected == 0) { // P2P uses only 1 connector
          comm->connectRecv[peer] |= (1<<channelId);
        }
      }
      delta = (comm->nRanks - (comm->rank-peer)) % comm->nRanks;
      for (int c=0; c<comm->p2pnChannelsPerPeer; c++) {
        int channelId = (delta+comm->p2pChannels[c]) % comm->p2pnChannels;
        if (comm->channels[channelId].peers[peer].send[0].connected == 0) { // P2P uses only 1 connector
          comm->connectSend[peer] |= (1<<channelId);
        }
      }
    }
    NCCLCHECK(ncclTransportP2pSetup(comm, NULL, 0));
    free(nvbPeers);
  }

  NCCLCHECK(ncclCommSetIntraProc(comm, intraProcRank, intraProcRanks, intraProcRank0Comm));

  /* Local intra-node barrier */
  NCCLCHECK(bootstrapBarrier(comm->bootstrap, intraNodeGlobalRanks, (int)intraNodeRank0pidHash, intraNodeRank, intraNodeRanks));

  if (comm->nNodes) NCCLCHECK(ncclProxyCreate(comm));

  // We should have allocated all buffers, collective fifos, ... we can
  // restore the affinity.
affinity_restore:
  if (CPU_COUNT(&comm->cpuAffinity)) sched_setaffinity(0, sizeof(cpu_set_t), &affinitySave);
  if (ret != ncclSuccess) return ret;

  TRACE(NCCL_INIT, "rank %d nranks %d - DONE", rank, nranks);
  return ncclSuccess;
}

NCCL_PARAM(SetStackSize, "SET_STACK_SIZE", 0);

ncclResult_t ncclCommInitRankSync(ncclComm_t* newcomm, int nranks, ncclUniqueId commId, int myrank, int cudaDev) {
  ncclResult_t res;

  CUDACHECK(hipSetDevice(cudaDev));
  // Set the maximum kernel stack size of all kernels to avoid
  // a CUDA memory reconfig on load (c.f. NVSHMEM issue)
  //if (maxLocalSizeBytes > 0 && ncclParamSetStackSize() == 1) {
  //  TRACE(NCCL_INIT, "Setting hipLimitStackSize to %zi", maxLocalSizeBytes);
  //  CUDACHECKIGNORE(hipDeviceSetLimit(hipLimitStackSize, maxLocalSizeBytes));
  //}
  NCCLCHECKGOTO(commAlloc(newcomm, nranks, myrank), res, cleanup);
  NCCLCHECKGOTO(initTransportsRank(*newcomm, &commId), res, cleanup);
  NCCLCHECKGOTO(devCommSetup(*newcomm), res, cleanup);

  INFO(NCCL_INIT,"comm %p rank %d nranks %d cudaDev %d busId %lx used %ld bytes - Init COMPLETE", *newcomm, myrank, nranks, (*newcomm)->cudaDev, (*newcomm)->busId, allocTracker[(*newcomm)->cudaDev].totalAllocSize);

  return ncclSuccess;
cleanup:
  if ((*newcomm) && (*newcomm)->bootstrap) bootstrapAbort((*newcomm)->bootstrap);
  *newcomm = NULL;
  return res;
}

static ncclResult_t ncclCommInitRankDev(ncclComm_t* newcomm, int nranks, ncclUniqueId commId, int myrank, int cudaDev) {
  ncclResult_t res;
  char* env = getenv("NCCL_COMM_ID");
  if (env && myrank == 0) {
    INFO(NCCL_ENV, "NCCL_COMM_ID set by environment to %s", env);
    NCCLCHECKGOTO(bootstrapCreateRoot(&commId, true), res, end);
  }

  NCCLCHECKGOTO(ncclInit(), res, end);
  if (myrank == 0) showVersion();

  memset(allocTracker+cudaDev, 0, sizeof(struct allocationTracker));
  // Make sure the CUDA runtime is initialized.
  CUDACHECKGOTO(hipFree(NULL), res, end);

  NCCLCHECKGOTO(PtrCheck(newcomm, "CommInitRank", "newcomm"), res, end);
  if (nranks < 1 || myrank < 0 || myrank >= nranks) {
    WARN("Invalid rank requested : %d/%d", myrank, nranks);
    res = ncclInvalidArgument;
    goto end;
  }

  if (ncclAsyncMode()) {
    NCCLCHECKGOTO(ncclAsyncInit(ncclCommInitRankSync, newcomm, nranks, commId, myrank, cudaDev), res, end);
  } else {
    NCCLCHECKGOTO(ncclCommInitRankSync(newcomm, nranks, commId, myrank, cudaDev), res, end);
  }

end:
  if (ncclAsyncMode()) return ncclAsyncErrCheck(res);
  else return res;
}

NCCL_API(ncclResult_t, ncclCommInitRank, ncclComm_t* newcomm, int nranks, ncclUniqueId commId, int myrank);
ncclResult_t ncclCommInitRank(ncclComm_t* newcomm, int nranks, ncclUniqueId commId, int myrank) {
  NVTX3_FUNC_RANGE_IN(nccl_domain);
  int cudaDev;
  CUDACHECK(hipGetDevice(&cudaDev));
  NCCLCHECK(ncclCommInitRankDev(newcomm, nranks, commId, myrank, cudaDev));
  return ncclSuccess;
}

NCCL_API(ncclResult_t, ncclCommInitAll, ncclComm_t* comms, int ndev, const int* devlist);
ncclResult_t ncclCommInitAll(ncclComm_t* comms, int ndev, const int* devlist) {
  NVTX3_FUNC_RANGE_IN(nccl_domain);
  NCCLCHECK(PtrCheck(comms, "CommInitAll", "comms"));
  if (ndev < 0) {
    WARN("Invalid device count requested : %d", ndev);
    return ncclInvalidArgument;
  }

  ncclUniqueId uniqueId;
  NCCLCHECK(ncclGetUniqueId(&uniqueId));
  NCCLCHECK(ncclGroupStart());
  for (int i=0; i<ndev; i++) {
    // Ignore return codes .. we need to call ncclGroupEnd to clean up anyway
    ncclCommInitRankDev(comms+i, ndev, uniqueId, i, devlist ? devlist[i] : i);
  }
  NCCLCHECK(ncclGroupEnd());
  return ncclSuccess;
}

static ncclResult_t commDestroy(ncclComm_t comm) {
  int savedDevice;
#ifdef ENABLE_TRACE
  int rank = comm->rank;
#endif
  CUDACHECK(hipGetDevice(&savedDevice));
  int commDevice = comm->cudaDev;

  if (savedDevice != commDevice) {
    CUDACHECK(hipSetDevice(commDevice));
  }

  TRACE(NCCL_INIT, "Destroying comm %p rank %d abortFlag %d fatalError %d", comm, comm->rank, LOAD(comm->abortFlag), comm->fatalError);

  CUDACHECK(hipStreamSynchronize(comm->groupStream));
  NCCLCHECK(ncclProxyDestroy(comm));
  NCCLCHECK(commFree(comm));

  if (savedDevice != commDevice)
    CUDACHECK(hipSetDevice(savedDevice));

  TRACE(NCCL_INIT, "Destroyed comm %p rank %d", comm, rank);

  return ncclSuccess;
}

NCCL_API(ncclResult_t, ncclCommDestroy, ncclComm_t comm);
ncclResult_t ncclCommDestroy(ncclComm_t comm) {
  NVTX3_FUNC_RANGE_IN(nccl_domain);
  if (comm == NULL)
    return ncclSuccess;

  TRACE(NCCL_INIT, "comm %p rank %d nRanks %d cudaDev %d busId %lx", comm, comm->rank, comm->nRanks, comm->cudaDev, comm->busId);

  // Try and prevent a double free of the comm struct (user error)
  if (comm->rank == -1 || comm->nRanks <= 0 || comm->cudaDev == -1 || comm->busId == -1) {
    WARN("comm %p has already been destroyed", comm);
    return ncclInvalidArgument;
  }

  // [RCCL] Delete CliqueManager if it exists
  if (comm->cliqueManager) delete comm->cliqueManager;
  // [/RCCL]

  return commDestroy(comm);
}

NCCL_API(ncclResult_t, ncclCommAbort, ncclComm_t comm);
ncclResult_t ncclCommAbort(ncclComm_t comm) {
  NVTX3_FUNC_RANGE_IN(nccl_domain);
  if (comm == NULL)
    return ncclSuccess;

  // Ask anything that might still be running on the device to quit
  *comm->abortFlag = 1;

  // do not destroy comm because kernel maybe still running
  // return commDestroy(comm);
  return ncclSuccess;
}

NCCL_API(const char*, ncclGetErrorString, ncclResult_t code);
const char* ncclGetErrorString(ncclResult_t code) {
  switch (code) {
    case ncclSuccess                : return "no error";
    case ncclUnhandledCudaError     : return "unhandled cuda error";
    case ncclSystemError            : return "unhandled system error";
    case ncclInternalError          : return "internal error";
    case ncclInvalidArgument        : return "invalid argument";
    case ncclInvalidUsage           : return "invalid usage";
    default                         : return "unknown result code";
  }
}

NCCL_API(ncclResult_t, ncclCommGetAsyncError, ncclComm_t comm, ncclResult_t *asyncError);
ncclResult_t ncclCommGetAsyncError(ncclComm_t comm, ncclResult_t *asyncError) {
  NCCLCHECK(PtrCheck(comm, "ncclGetAsyncError", "comm"));
  NCCLCHECK(PtrCheck(asyncError, "ncclGetAsyncError", "asyncError"));
  *asyncError = comm->fatalError;
  return ncclSuccess;
}

NCCL_API(ncclResult_t, ncclCommCount, const ncclComm_t comm, int* count);
ncclResult_t ncclCommCount(const ncclComm_t comm, int* count) {
  NVTX3_FUNC_RANGE_IN(nccl_domain);
  NCCLCHECK(PtrCheck(comm, "CommCount", "comm"));
  NCCLCHECK(PtrCheck(count, "CommCount", "count"));
  *count = comm->nRanks;
  return ncclSuccess;
}

NCCL_API(ncclResult_t, ncclCommCuDevice, const ncclComm_t comm, int* devid);
ncclResult_t ncclCommCuDevice(const ncclComm_t comm, int* devid) {
  NVTX3_FUNC_RANGE_IN(nccl_domain);
  NCCLCHECK(PtrCheck(comm, "CommCuDevice", "comm"));
  NCCLCHECK(PtrCheck(devid, "CommCuDevice", "devid"));
  *devid = comm->cudaDev;
  return ncclSuccess;
}

NCCL_API(ncclResult_t, ncclCommUserRank, const ncclComm_t comm, int* rank);
ncclResult_t ncclCommUserRank(const ncclComm_t comm, int* rank) {
  NVTX3_FUNC_RANGE_IN(nccl_domain);
  NCCLCHECK(PtrCheck(comm, "CommUserRank", "comm"));
  NCCLCHECK(PtrCheck(rank, "CommUserRank", "rank"));
  *rank = comm->rank;
  return ncclSuccess;
}
