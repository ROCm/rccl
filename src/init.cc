/*************************************************************************
 * Copyright (c) 2015-2022, NVIDIA CORPORATION. All rights reserved.
 * Modifications Copyright (c) 2019-2022 Advanced Micro Devices, Inc. All rights reserved.
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
#if defined(ENABLE_NPKIT)
#include "npkit/npkit.h"
#endif
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
#include "git_version.h"
#include "rccl_vars.h"
//#include "clique/CliqueManager.h"
//#include <hsa/hsa_ext_amd.h>
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

const char* ncclFuncStr[NCCL_NUM_FUNCTIONS+2] = { "Broadcast", "Reduce", "AllGather", "ReduceScatter", "AllReduce", "SendRecv", "AllToAllPivot" };
const char* ncclAlgoStr[NCCL_NUM_ALGORITHMS] = { "Tree", "Ring", "CollNet" };
const char* ncclProtoStr[NCCL_NUM_PROTOCOLS] = { "LL", "LL128", "Simple" };
const char* ncclDevRedOpStr[ncclNumDevRedOps] = { "Sum", "Prod", "Max", "Min", "PreMulSum", "SumPostDiv" };
const char *ncclTypeStr[ncclNumTypes] = {"_i8", "_u8", "_i32", "_u32", "_i64", "_u64", "_f16", "_f32", "_f64", "_b16"};

NCCL_PARAM(GroupCudaStream, "GROUP_CUDA_STREAM", NCCL_GROUP_CUDA_STREAM);

NCCL_PARAM(CheckPointers, "CHECK_POINTERS", 0);
struct allocationTracker allocTracker[MAX_ALLOC_TRACK_NGPU] = {};

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


NCCL_PARAM(L1SharedMemoryCarveout, "L1_SHARED_MEMORY_CARVEOUT", 0);

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
    int carveout = ncclParamL1SharedMemoryCarveout();
    if (carveout) ncclKernSetSharedMemoryCarveout(carveout);
    NCCLCHECK(ncclNetInit());
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
  #define MAX_NAME_LENGTH 64
  char* func_names = (char *)malloc(MAX_NAME_LENGTH*(FUNC_INDEX_P2P+2));
  for (int func = 0; func < NCCL_NUM_FUNCTIONS; func++) {
    for (int al = 0; al < NCCL_NUM_ALGORITHMS; al++) {
      for (int type = 0; type < ncclNumTypes; type++) {
        for (int pr = 0; pr < NCCL_NUM_PROTOCOLS; pr++) {
          for (int devredop = 0; devredop < ncclNumDevRedOps; devredop++) {
            char* line = func_names+MAX_NAME_LENGTH*FUNC_INDEX(func, devredop, type, al, pr);
            sprintf(line, "%s%s%s%s%s", ncclFuncStr[func], ncclAlgoStr[al], ncclProtoStr[pr],
              ncclDevRedOpStr[devredop], ncclTypeStr[type]);
          }
        }
      }
    }
  }
  for (int type = 0; type < ncclNumTypes; type++) {
    char* line = func_names+MAX_NAME_LENGTH*(FUNC_INDEX_P2P-ncclNumTypes+type);
    sprintf(line, "OneRankReducePreMulSum%s", ncclTypeStr[type]);
  }
  char* line = func_names+MAX_NAME_LENGTH*FUNC_INDEX_P2P;
  sprintf(line, "SendRecvRingSimpleSum_i8");
  line += MAX_NAME_LENGTH;
  sprintf(line, "AllToAllPivotRingSimpleSum_i8");
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
        if (fIdx == FUNC_INDEX_P2P || type == ncclCollTraceP2pElemType)
          sprintf(line, "## [%12.6f] [%02d:%02d] %06x-%06x", (double)(td->timeStamp)/VEGA_GPU_RTC_FREQUENCY, comm->rank, td->bid, td->p2pOpCount[0], td->p2pOpCount[1]);
        else
          sprintf(line, "## [%12.6f] [%02d:%02d] %06lx", (double)(td->timeStamp)/VEGA_GPU_RTC_FREQUENCY, comm->rank, td->bid, td->opCount);
        offset = strlen(line);
        if (type == ncclCollTraceCollElemType) {
          sprintf(line+offset, " CE %s nw %d bi %d nc %d busId %lx nRanks %d", func_names+MAX_NAME_LENGTH*fIdx, td->coll.nWarps, td->coll.bid, td->coll.nChannels, comm->busId, comm->nRanks);
        } else if (type == ncclCollTraceP2pElemType) {
          sprintf(line+offset, " PE %s %d -> %d/%d/%d/%d conn/nw/ws/ng %d/%d/%d/%d -> %d busId %lx nRanks %d", func_names+MAX_NAME_LENGTH*fIdx,
            td->p2p[0].peer, td->p2p[0].connIndex, td->p2p[0].nWarps, td->p2p[0].warpStart, td->p2p[0].ngroups,
            td->p2p[1].connIndex, td->p2p[1].nWarps, td->p2p[1].warpStart, td->p2p[1].ngroups, td->p2p[1].peer, comm->busId, comm->nRanks);
        } else {
          switch (type&0xf) {
            case ncclCollTraceKernelLaunchType:
            case ncclCollTraceCollLaunchType:
              if ((type&0xf) == ncclCollTraceKernelLaunchType)
                sprintf(line+offset, " KL HWID %8x %s", td->data_0, func_names+MAX_NAME_LENGTH*fIdx);
              else if ((type&0xf) == ncclCollTraceCollLaunchType)
                sprintf(line+offset, " CL %s", func_names+MAX_NAME_LENGTH*fIdx);
              offset = strlen(line);
              if ((type&0xf0) == ncclCollTraceCollElemType)
                sprintf(line+offset, " nw %d bi %d nc %d busId %lx nRanks %d", td->coll.nWarps, td->coll.bid, td->coll.nChannels, comm->busId, comm->nRanks);
              else if ((type&0xf0) == ncclCollTraceP2pElemType)
                sprintf(line+offset, " %d -> %d/%d/%d/%d conn/nw/ws/ng %d/%d/%d/%d -> %d busId %lx nRanks %d",
                  td->p2p[0].peer, td->p2p[0].connIndex, td->p2p[0].nWarps, td->p2p[0].warpStart, td->p2p[0].ngroups,
                  td->p2p[1].connIndex, td->p2p[1].nWarps, td->p2p[1].warpStart, td->p2p[1].ngroups, td->p2p[1].peer, comm->busId, comm->nRanks);
              break;
            case ncclCollTraceKernelEndType:
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
#define PROFILE_USE_TIME

static ncclResult_t commFree(ncclComm_t comm) {
  if (comm == NULL)
    return ncclSuccess;

  // First stop all threads before we free anything.
  NCCLCHECK(ncclProxyDestroy(comm));

  delete[] comm->userRedOps;

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
  struct ncclProf prof;
  prof.elems = (struct ncclProfElem*)malloc(sizeof(struct ncclProfElem)*PROFILE_NUM_ITEMS);
  CUDACHECK(hipMemcpy(prof.elems, comm->hostDevComm.devProf.elems, sizeof(struct ncclProfElem)*PROFILE_NUM_ITEMS, hipMemcpyDeviceToHost));
  #define VEGA_GPU_RTC_FREQUENCY 2.5E7
  if (comm->rank == 0) {
    INFO(NCCL_INIT, "# %7s %4s %6s %6s %6s %6s %6s %7s %6s %6s %6s %6s %6s", "Rank:Ch", "opCt", "total", "  prim", "  wait", "send", "rcRdS", "dRcRdCS", "dRcCS", "dRc", "cS", "rc", "rcCS");
#ifdef PROFILE_USE_TIME
    INFO(NCCL_INIT, "# %7s %4s %6s %6s %6s %6s %6s %7s %6s %6s %6s %6s %6s", "", "", "  (us)", "  (us)", "  (us)", "  (us)", "  (us)", "  (us)", "  (us)", "  (us)", "  (us)", "  (us)", "  (us)");
#else
    INFO(NCCL_INIT, "# %7s %4s %6s %6s %6s %6s %7s %6s %6s %6s %6s %6s", "", "", "   (s)", "   (s)", "(GB/s)", "(GB/s)", "(GB/s)", "(GB/s)", "(GB/s)", "(GB/s)", "(GB/s)", "(GB/s)");
#endif
  }
  for (int i = 1; i < PROFILE_NUM_ITEMS; i++) {
    int valid = 0;
    for (int chan=0; chan<comm->nChannels; chan++) {
      struct ncclProfElem *elem = prof.elems+i;
      if (elem->elem[chan].opCount == 0)
        continue;
      valid++;
#ifdef PROFILE_USE_TIME
      INFO(NCCL_INIT, "# [%02d:%02d] %04d %6.2f %6.2f %6.2f %6.2f %6.2f %7.2f %6.2f %6.2f %6.2f %6.2f %6.2f",
        comm->rank, chan, (uint32_t)elem->elem[chan].opCount, (double)elem->elem[chan].total_cycle/VEGA_GPU_RTC_FREQUENCY*1.0E6,
        (double)elem->elem[chan].prim_cycle/VEGA_GPU_RTC_FREQUENCY*1.0E6, (double)elem->elem[chan].wait_cycle/VEGA_GPU_RTC_FREQUENCY*1.0E6,
        (elem->elem[chan].send_cycle) ? ((double)elem->elem[chan].send_cycle/VEGA_GPU_RTC_FREQUENCY*1.0E6) : 0,
        (elem->elem[chan].recvReduceSend_cycle) ? ((double)elem->elem[chan].recvReduceSend_cycle/VEGA_GPU_RTC_FREQUENCY*1.0E6) : 0,
        (elem->elem[chan].directRecvReduceCopySend_cycle) ? ((double)elem->elem[chan].directRecvReduceCopySend_cycle/VEGA_GPU_RTC_FREQUENCY*1.0E6) : 0,
        (elem->elem[chan].directRecvCopySend_cycle) ? ((double)elem->elem[chan].directRecvCopySend_cycle/VEGA_GPU_RTC_FREQUENCY*1.0E6) : 0,
        (elem->elem[chan].directRecv_cycle) ? ((double)elem->elem[chan].directRecv_cycle/VEGA_GPU_RTC_FREQUENCY*1.0E6) : 0,
        (elem->elem[chan].copySend_cycle) ? ((double)elem->elem[chan].copySend_cycle/VEGA_GPU_RTC_FREQUENCY*1.0E6) : 0,
        (elem->elem[chan].recv_cycle) ? ((double)elem->elem[chan].recv_cycle/VEGA_GPU_RTC_FREQUENCY*1.0E6) : 0,
        (elem->elem[chan].recvCopySend_cycle) ? ((double)elem->elem[chan].recvCopySend_cycle/VEGA_GPU_RTC_FREQUENCY*1.0E6) : 0);
#else
      INFO(NCCL_INIT, "# [%02d:%02d] %04d %6.4f %6.4f %6.2f %6.2f %7.2f %6.2f %6.2f %6.2f %6.2f %6.2f",
        comm->rank, chan, (uint32_t)elem->elem[chan].opCount, (double)elem->elem[chan].total_cycle/VEGA_GPU_RTC_FREQUENCY,
        (double)elem->elem[chan].wait_cycle/VEGA_GPU_RTC_FREQUENCY,
        (elem->elem[chan].send_cycle) ? (double)elem->elem[chan].send_byte/((double)elem->elem[chan].send_cycle/VEGA_GPU_RTC_FREQUENCY*1.0E9) : 0,
        (elem->elem[chan].recvReduceSend_cycle) ? (double)elem->elem[chan].recvReduceSend_byte/((double)elem->elem[chan].recvReduceSend_cycle/VEGA_GPU_RTC_FREQUENCY*1.0E9) : 0,
        (elem->elem[chan].directRecvReduceCopySend_cycle) ? (double)elem->elem[chan].directRecvReduceCopySend_byte/((double)elem->elem[chan].directRecvReduceCopySend_cycle/VEGA_GPU_RTC_FREQUENCY*1.0E9) : 0,
        (elem->elem[chan].directRecvCopySend_cycle) ? (double)elem->elem[chan].directRecvCopySend_byte/((double)elem->elem[chan].directRecvCopySend_cycle/VEGA_GPU_RTC_FREQUENCY*1.0E9) : 0,
        (elem->elem[chan].directRecv_cycle) ? (double)elem->elem[chan].directRecv_byte/((double)elem->elem[chan].directRecv_cycle/VEGA_GPU_RTC_FREQUENCY*1.0E9) : 0,
        (elem->elem[chan].copySend_cycle) ? (double)elem->elem[chan].copySend_byte/((double)elem->elem[chan].copySend_cycle/VEGA_GPU_RTC_FREQUENCY*1.0E9) : 0,
        (elem->elem[chan].recv_cycle) ? (double)elem->elem[chan].recv_byte/((double)elem->elem[chan].recv_cycle/VEGA_GPU_RTC_FREQUENCY*1.0E9) : 0,
        (elem->elem[chan].recvCopySend_cycle) ? (double)elem->elem[chan].recvCopySend_byte/((double)elem->elem[chan].recvCopySend_cycle/VEGA_GPU_RTC_FREQUENCY*1.0E9) : 0);
#endif
    }
    if (valid == 0)
      break;
  }
  free(prof.elems);
  CUDACHECK(hipFree(comm->hostDevComm.devProf.elems));

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
  for (int n=0; n<comm->nNodes; n++) free(comm->nodeRanks[n].localRankToRank);
  free(comm->nodeRanks);
  free(comm->rankToNode);
  free(comm->rankToLocalRank);

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

  CUDACHECK(hipStreamDestroy(comm->sideStream));

  // Last rank frees shared resources between threads
  int isLast;
  NCCLCHECK(ncclCpuBarrierIn(comm, &isLast));
  if (isLast) {
    // Wait for all service threads to be done. We could not
    // do it earlier because it could have blocked and prevented
    // other ranks in the process to call ncclCommDestroy
    for (int i=0; i<comm->intraRanks; i++) {
      void* ret;
      if (comm->intraThreads[i]) pthread_join(comm->intraThreads[i], &ret);
    }
    free(comm->intraBarrier);
    free(comm->intraParams);
    free(comm->intraThreads);
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
RCCL_PARAM(PivotAlltoallEnable, "PIVOT_ALLTOALL_ENABLE", 1);
RCCL_PARAM(LL128ForceEnable, "LL128_FORCE_ENABLE", 0);
NCCL_PARAM(AggChannelSize, "AGG_CHANNEL_SIZE", -2);
NCCL_PARAM(DisableGraphHelper, "GRAPH_HELPER_DISABLE", 0);
NCCL_PARAM(GraphRegister, "GRAPH_REGISTER", 0);

static ncclResult_t commAlloc(ncclComm_t* comret, int ndev, int rank, int virtualId) {
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
  comm->virtualId = virtualId;
  hipGetDevice(&comm->cudaDev);
  NCCLCHECK(getBusId(comm->cudaDev, &comm->busId));
  TRACE(NCCL_INIT,"comm %p rank %d nranks %d cudaDev %d busId %lx", comm, rank, ndev, comm->cudaDev, comm->busId);

  // RCCL: create persistent stream for calloc
  CUDACHECK(hipStreamCreateWithFlags(&comm->sideStream, hipStreamNonBlocking));

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
  comm->p2pOpCount = 0;

  comm->argsptrs[0] = &comm->devComm;
#ifdef ENABLE_PROFILING
  NCCLCHECK(ncclCudaCalloc(&comm->hostDevComm.devProf.elems, PROFILE_NUM_ITEMS, comm->sideStream));
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
  comm->asyncAllocMode = ncclComm::SHORTEST_QUEUE;
  char* str = getenv("NCCL_AGG_ALLOC_MODE");
  if (str) INFO(NCCL_ENV, "NCCL_AGG_ALLOC_MODE set by environment to %s", str);
  if (str && strcmp(str, "ROUND_ROBIN") == 0) {
    comm->asyncAllocMode = ncclComm::ROUND_ROBIN;
  }

  CUDACHECK(hipDriverGetVersion(&comm->driverVersion));

  NCCLCHECK(ncclCreateQueueInfo(&comm->enqueueInfo, comm));
  comm->lastSetupNode = NULL;
  comm->lastCudaGraphId = -1;
  comm->disableGraphHelper = ncclParamDisableGraphHelper();
  comm->graphRegister = ncclParamGraphRegister();

  if (rcclParamEnableHipGraph())
  {
    NCCLCHECK(ncclCalloc(&comm->graphHelperResources, 1));
    comm->graphHelperResources->comm = comm;
    comm->pfnCuMemGetAddressRange = hipMemGetAddressRange;
  }

  static_assert(MAXCHANNELS <= sizeof(*comm->connectSend)*8, "comm->connectSend must have enough bits for all channels");
  static_assert(MAXCHANNELS <= sizeof(*comm->connectRecv)*8, "comm->connectRecv must have enough bits for all channels");
  NCCLCHECK(ncclCalloc(&comm->connectSend, comm->nRanks*NCCL_MAX_CONNS));
  NCCLCHECK(ncclCalloc(&comm->connectRecv, comm->nRanks*NCCL_MAX_CONNS));

  comm->p2pSendCount = comm->p2pRecvCount = 0;
  NCCLCHECK(ncclCalloc(&comm->p2pSends, comm->nRanks));
  NCCLCHECK(ncclCalloc(&comm->p2pRecvs, comm->nRanks));

  // Mark channels as non initialized.
  for (int c=0; c<MAXCHANNELS; c++) comm->channels[c].id = -1;

  CUDACHECK(hipDeviceGetAttribute(&comm->WarpSize, hipDeviceAttributeWarpSize, comm->cudaDev));
  *comret = comm;
  return ncclSuccess;
}

static ncclResult_t devCommSetup(ncclComm_t comm) {
  ncclDevCommAndChannels *devCommAndChans;
  NCCLCHECK(ncclCudaCalloc(&devCommAndChans, 1, comm->sideStream));
  comm->devComm = &devCommAndChans->comm;
  comm->hostDevComm.channels = devCommAndChans->channels;

  // Duplicate the channels on the device
  int nChannels = std::max(comm->nChannels, comm->p2pnChannels);
  NCCLCHECK(ncclCudaMemcpy(comm->hostDevComm.channels, comm->channels, nChannels));

  // Copy userRanks and peers
  for (int r=0; r<comm->nChannels; r++) {
    NCCLCHECK(ncclCudaMemcpy(comm->channels[r].ring.devUserRanks, comm->channels[r].ring.userRanks, comm->nRanks));
  }

#if defined(ENABLE_NPKIT)
  // Init NPKit
  NCCLCHECK(NpKit::Init(comm->rank));
  comm->hostDevComm.npKitEventCollectContexts = NpKit::GetGpuEventCollectContexts();
  comm->hostDevComm.cpuTimestamp = NpKit::GetCpuTimestamp();
#endif

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
    printf("%s %s\n", VERSION_STRING, rcclGitHash);
    fflush(stdout);
    if (ncclDebugFile != stdout)
      INFO(NCCL_ALL,"%s %s", VERSION_STRING, rcclGitHash); // Also log NCCL version in one of the files
    shown = 1;
  }
}

static ncclResult_t fillInfo(struct ncclComm* comm, struct ncclPeerInfo* info, uint64_t commHash) {
  info->rank = comm->rank;
  info->virtualId = comm->virtualId;
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

  // detect if fine grained memory is available on this GPU
  int *ptr;
  if (hipExtMallocWithFlags((void**)&ptr, sizeof(int), hipDeviceMallocFinegrained) == hipSuccess) {
    CUDACHECK(hipFree(ptr));
    info->hasFineGrain = true;
    NCCLCHECK(ncclGpuGdrSupport(&info->gdrSupport));
  }
  else {
    info->hasFineGrain = false;
    info->gdrSupport = 0;
  }

  info->comm = comm;
  info->cudaCompCap = ncclCudaCompCap();
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
  params->args = (void **)&comm->argsptrs;
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
    NCCLCHECK(ncclCalloc(&comm->intraThreads, comm->intraRanks));
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
    comm->intraThreads = (pthread_t*)waitForNonNullPtr(&comm0->intraThreads);
    comm->intraCudaDevs = (int*)waitForNonNullPtr(&comm0->intraCudaDevs);
    comm->intraCGMode = (int*)waitForNonNullPtr(&comm0->intraCGMode);
    comm->intraCC = (int*)waitForNonNullPtr(&comm0->intraCC);
  }
  comm->intraCudaDevs[comm->intraRank] = comm->cudaDev;
  comm->intraThreads[comm->intraRank] = comm->proxyState.thread;
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

NCCL_PARAM(GraphDumpFileRank, "GRAPH_DUMP_FILE_RANK", 0);
NCCL_PARAM(CollNetNodeThreshold, "COLLNET_NODE_THRESHOLD", 2);
NCCL_PARAM(NvbPreconnect, "NVB_PRECONNECT", 0);

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
  NCCLCHECK(bootstrapInit(commId, comm));
  // [/RCCL]

  // AllGather1 - begin
  NCCLCHECK(ncclCalloc(&comm->peerInfo, nranks+1)); // Extra rank to represent CollNet root
  NCCLCHECK(fillInfo(comm, comm->peerInfo+rank, commHash));
  NCCLCHECK(bootstrapAllGather(comm->bootstrap, comm->peerInfo, sizeof(struct ncclPeerInfo)));

  //If virtualId == -1 multiRank support has not been requested by user, using original interface
  if (comm->virtualId == -1) {
    for (int i = 0; i < nranks; i++) {
      if ((i != rank) && (comm->peerInfo[i].hostHash == comm->peerInfo[rank].hostHash) && (comm->peerInfo[i].busId == comm->peerInfo[rank].busId)) {
	WARN("Duplicate GPU detected : rank %d and rank %d both on CUDA device %lx", rank, i, comm->peerInfo[rank].busId);
	return ncclInvalidUsage;
      }
    }
  }
  else {
    //Multiple ranks can use the same device, but need to have different virtualId's.
    for (int i = 0; i < nranks; i++) {
      for (int j=0; j < nranks; j++) {
	if (j==i) continue;
	if((comm->peerInfo[i].hostHash  == comm->peerInfo[j].hostHash)  &&
	   (comm->peerInfo[i].busId     == comm->peerInfo[j].busId)     &&
	   (comm->peerInfo[i].virtualId == comm->peerInfo[j].virtualId)) {
	  WARN("Duplicate virtualId detected : rank %d and rank %d both on GPU device %lx virtualId %d",
	       i, j, comm->peerInfo[rank].busId, comm->peerInfo[i].virtualId);
	  return ncclInvalidUsage;
	}
      }
    }
  }
  // AllGather1 - end

  // Topo detection / System graph creation
  NCCLCHECK(ncclTopoGetSystem(comm, &comm->topo));
  // save nRanks to ncclTopoSystem as indicator of multi-node
  comm->topo->nRanks = comm->nRanks;
  // init netGdrLevel
  comm->topo->netGdrLevel = -2;
  // init Pivot A2A related fields
  comm->topo->pivotA2AEnabled = false;
  comm->topo->pivotA2ANumBiRings = 0;
  // LL128
  comm->topo->ll128Enabled = false;
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

  // Set Affinity to a CPU local the our GPU, so that all memory we allocate
  // on the host is local.
  NCCLCHECK(ncclTopoGetCpuAffinity(comm->topo, comm->rank, &comm->cpuAffinity));
  cpu_set_t affinitySave;
  if (CPU_COUNT(&comm->cpuAffinity)) {
    sched_getaffinity(0, sizeof(cpu_set_t), &affinitySave);
    sched_setaffinity(0, sizeof(cpu_set_t), &comm->cpuAffinity);
  }
  ncclResult_t ret;

  // Launch proxy service thread
  NCCLCHECK(ncclProxyCreate(comm));

  // Get rings and trees
  struct ncclTopoGraph ringGraph;
  ringGraph.id = 0;
  ringGraph.pattern = NCCL_TOPO_PATTERN_RING;
  ringGraph.collNet = 0;
  ringGraph.minChannels = 1;
  ringGraph.maxChannels = MAXCHANNELS/2;
  NCCLCHECK(ncclTopoCompute(comm->topo, &ringGraph));
  NCCLCHECK(ncclTopoPrintGraph(comm->topo, &ringGraph));

  struct ncclTopoGraph treeGraph;
  treeGraph.id = 1;
  treeGraph.pattern = NCCL_TOPO_PATTERN_BALANCED_TREE;
  treeGraph.collNet = 0;
  treeGraph.minChannels = comm->topo->nodes[NET].count != 0 ? 1 : ringGraph.nChannels;
  treeGraph.maxChannels = ringGraph.nChannels;
  NCCLCHECK(ncclTopoCompute(comm->topo, &treeGraph));
  NCCLCHECK(ncclTopoPrintGraph(comm->topo, &treeGraph));

  struct ncclTopoGraph collNetGraph;
  collNetGraph.id = 2;
  collNetGraph.pattern = NCCL_TOPO_PATTERN_TREE;
  collNetGraph.collNet = 1;
  collNetGraph.minChannels = collNetGraph.maxChannels = ringGraph.nChannels;
  NCCLCHECK(ncclTopoCompute(comm->topo, &collNetGraph));
  NCCLCHECK(ncclTopoPrintGraph(comm->topo, &collNetGraph));

  bool allXgmi = true, hasPeerAccess = true;
  // Check that all the GPUs have peer access to one another and are XGMI connected
  for (int i = 0; i < nranks && hasPeerAccess; i++) {
    int cudaDev1 = comm->peerInfo[i].cudaDev;
    for (int j = 0; j < nranks; j++) {
      if (i == j) continue;
      int cudaDev2 = comm->peerInfo[j].cudaDev;
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

#if 0
  { // [RCCL] Check if clique-based kernels can be enabled and initialize CliqueManager
    CliqueManager::cliqueMode_t cliqueMode = CliqueManager::CLIQUE_DISABLED;
    if (comm->localRanks == comm->nRanks && comm->topo->nodes[GPU].nodes[0].gpu.gcn != 910)
    {
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
#endif

  if (comm->rank == ncclParamGraphDumpFileRank()) {
    struct ncclTopoGraph* graphs[3] = { &ringGraph, &treeGraph, &collNetGraph };
    NCCLCHECK(ncclTopoDumpGraphs(comm->topo, 3, graphs));
  }

  // Determine local CollNet support before all-gather
  if (collNetSupport()) {
    char *collNetEnable = getenv("NCCL_COLLNET_ENABLE");
    if (collNetEnable != NULL) {
      INFO(NCCL_ALL, "NCCL_COLLNET_ENABLE set by environment to %s.", collNetEnable);
      if (strcmp(collNetEnable, "1") == 0) {
        comm->collNetSupport = 1;
      }
    }
  }
  if (comm->collNetSupport == 1 && collNetGraph.nChannels <= 0) comm->collNetSupport = 0;

  if ((comm->topo->type & RCCL_TOPO_4P2H_ROME) && (comm->topo->type & RCCL_TOPO_GDR_ALL)) {
    if (rcclParamP2pNetDisable() == 0) {
      if (!(comm->topo->type & RCCL_TOPO_FORCE_INTRA)) comm->p2pNet = 1;
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
    int netDev;
    int collNetSupport;
    int nc;
    struct ncclGraphInfo tree;
    struct ncclGraphInfo ring;
    struct ncclGraphInfo collNet;
    struct ncclTopoRanks topoRanks;
    bool pivotA2AEnabled;
    bool ll128Enabled;
  } *allGather3Data;

  NCCLCHECK(ncclCalloc(&allGather3Data, nranks));
  int idx;
  NCCLCHECK(ncclTopoIdToIndex(comm->topo, GPU, comm->busId, &idx));
  allGather3Data[rank].nc = 2;
  if (comm->topo->nodes[GPU].count == comm->topo->nRanks && comm->topo->nodes[GPU].nodes[idx].gpu.gcn == 906 && allXgmi)
    allGather3Data[rank].nc = 4;
  if (comm->topo->nodes[GPU].nodes[idx].gpu.gcn == 908)
    allGather3Data[rank].nc = std::max(4/ringGraph.nChannels, 2);
  if (comm->topo->nodes[GPU].count == comm->topo->nRanks && (comm->topo->type & RCCL_TOPO_CR8G))
    allGather3Data[rank].nc = 4;
  if (comm->topo->nodes[GPU].count == comm->topo->nRanks && comm->topo->nodes[GPU].nodes[idx].gpu.gcn == 910)
    allGather3Data[rank].nc = 4;
  if (comm->topo->nodes[GPU].nodes[idx].gpu.gcn == 910)
    allGather3Data[rank].nc = std::max(allGather3Data[rank].nc, 4/ringGraph.nChannels);
  if (ringGraph.nChannels > MAXCHANNELS/2)
    allGather3Data[rank].nc = 1;
  NCCLCHECK(ncclTopoGetLocalNet(comm->topo, rank, &allGather3Data[rank].netDev));
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
  allGather3Data[rank].pivotA2AEnabled = comm->topo->pivotA2AEnabled && rcclParamPivotAlltoallEnable();
  comm->topo->ll128Enabled =  comm->topo->ll128Enabled || rcclParamLL128ForceEnable();
  allGather3Data[rank].ll128Enabled = comm->topo->ll128Enabled;

  comm->nChannels = (comm->topo->nodes[GPU].count != comm->topo->nRanks && comm->topo->nodes[NET].count)
    ? std::min(treeGraph.nChannels, ringGraph.nChannels) : ringGraph.nChannels;
  NCCLCHECK(ncclTopoPreset(comm, &treeGraph, &ringGraph, &allGather3Data[rank].topoRanks));

  NCCLCHECK(bootstrapAllGather(comm->bootstrap, allGather3Data, sizeof(*allGather3Data)));

  // Determine nNodes, firstRanks, ...
  int *nodesFirstRank, *nodesTreePatterns;
  NCCLCHECK(ncclCalloc(&nodesFirstRank, nranks));
  NCCLCHECK(ncclCalloc(&nodesTreePatterns, nranks));
  NCCLCHECK(ncclCalloc(&comm->rankToNode, comm->nRanks));
  for (int r=0; r<nranks; r++) {
    int node;
    int firstRank = allGather3Data[r].topoRanks.ringRecv[0];
    for (node=0; node<comm->nNodes && nodesFirstRank[node] != firstRank; node++);
    if (node == comm->nNodes) {
      comm->nNodes++;
      nodesFirstRank[node] = firstRank;
      // Record tree pattern of each node as they can be different depending on sm arch
      nodesTreePatterns[node] = allGather3Data[r].tree.pattern;
    }
    comm->rankToNode[r] = node;
  }
  // Now that we know nNodes, alloc nodeRanks and compute localRanks for each node
  NCCLCHECK(ncclCalloc(&comm->nodeRanks, comm->nNodes));
  NCCLCHECK(ncclCalloc(&comm->rankToLocalRank, comm->nRanks));
  for (int r=0; r<comm->nRanks; r++) {
    int node = comm->rankToNode[r];
    comm->rankToLocalRank[r] = comm->nodeRanks[node].localRanks;
    comm->nodeRanks[node].localRanks++;
  }
  // Allocate ranks arrays for each node
  for (int n=0; n<comm->nNodes; n++) {
    NCCLCHECK(ncclCalloc(&comm->nodeRanks[n].localRankToRank, comm->nodeRanks[n].localRanks));
    comm->maxLocalRanks = std::max(comm->maxLocalRanks, comm->nodeRanks[n].localRanks);
    comm->nodeRanks[n].localRanks = 0;
  }
  // And fill the ranks arrays
  for (int r=0; r<comm->nRanks; r++) {
    int node = comm->rankToNode[r];
    comm->nodeRanks[node].localRankToRank[comm->nodeRanks[node].localRanks++] = r;
  }
  comm->node = comm->rankToNode[rank];
  comm->localRankToRank = comm->nodeRanks[comm->node].localRankToRank;
  comm->localRank = comm->rankToLocalRank[rank];
  comm->localRanks = comm->nodeRanks[comm->node].localRanks;

  TRACE(NCCL_INIT,"hostHash[%d] %lx localRank %d localRanks %d localRank0 %d",
        rank, comm->peerInfo[rank].hostHash, comm->localRank, comm->localRanks, comm->localRankToRank[0]);
  if (comm->localRank == -1 || comm->localRankToRank[0] == -1 || comm->localRanks == 0) {
    WARN("Failed to determine local ranks rank %d hostHash %lx pidHash %lx localRank %d localRanks %d localRank0 %d",
         rank, comm->peerInfo[rank].hostHash, comm->peerInfo[rank].pidHash,
         comm->localRank, comm->localRanks, comm->localRankToRank[0]);
    return ncclInternalError;
  }

  int nChannelsOrig = comm->nChannels;
  struct ncclTopoRanks** allTopoRanks;
  NCCLCHECK(ncclCalloc(&allTopoRanks, comm->nRanks));
  int nc = allGather3Data[0].nc;
  for (int i=0; i<nranks; i++) {
    comm->peerInfo[i].netDev = allGather3Data[i].netDev;
    allTopoRanks[i] = &allGather3Data[i].topoRanks;
    nc = std::min(allGather3Data[i].nc, nc);
    // Make sure we align all ranks so that the tuning is consistent across ranks
    treeGraph.nChannels = std::min(allGather3Data[i].tree.nChannels, treeGraph.nChannels);
    treeGraph.sameChannels = std::min(allGather3Data[i].tree.sameChannels, treeGraph.sameChannels);
    treeGraph.speedIntra = std::min(allGather3Data[i].tree.speedIntra, treeGraph.speedIntra);
    treeGraph.speedInter = std::min(allGather3Data[i].tree.speedInter, treeGraph.speedInter);
    treeGraph.typeIntra = std::max(allGather3Data[i].tree.typeIntra, treeGraph.typeIntra);
    treeGraph.typeInter = std::max(allGather3Data[i].tree.typeInter, treeGraph.typeInter);
    ringGraph.nChannels = std::min(allGather3Data[i].ring.nChannels, ringGraph.nChannels);
    ringGraph.sameChannels = std::min(allGather3Data[i].ring.sameChannels, ringGraph.sameChannels);
    ringGraph.speedIntra = std::min(allGather3Data[i].ring.speedIntra, ringGraph.speedIntra);
    ringGraph.speedInter = std::min(allGather3Data[i].ring.speedInter, ringGraph.speedInter);
    ringGraph.typeIntra = std::max(allGather3Data[i].ring.typeIntra, ringGraph.typeIntra);
    ringGraph.typeInter = std::max(allGather3Data[i].ring.typeInter, ringGraph.typeInter);
    collNetGraph.nChannels = std::min(allGather3Data[i].collNet.nChannels, collNetGraph.nChannels);
    collNetGraph.sameChannels = std::min(allGather3Data[i].collNet.sameChannels, collNetGraph.sameChannels);
    collNetGraph.speedIntra = std::min(allGather3Data[i].collNet.speedIntra, collNetGraph.speedIntra);
    collNetGraph.speedInter = std::min(allGather3Data[i].collNet.speedInter, collNetGraph.speedInter);
    collNetGraph.typeIntra = std::max(allGather3Data[i].collNet.typeIntra, collNetGraph.typeIntra);
    collNetGraph.typeInter = std::max(allGather3Data[i].collNet.typeInter, collNetGraph.typeInter);
    comm->collNetSupport = std::min(allGather3Data[i].collNetSupport, comm->collNetSupport);
    comm->topo->pivotA2AEnabled = comm->topo->pivotA2AEnabled && allGather3Data[i].pivotA2AEnabled;
    comm->topo->ll128Enabled = comm->topo->ll128Enabled && allGather3Data[i].ll128Enabled;
  }

  comm->nChannels = treeGraph.nChannels = ringGraph.nChannels =
    (comm->topo->nodes[GPU].count != comm->topo->nRanks && comm->topo->nodes[NET].count)
    ? std::min(treeGraph.nChannels, ringGraph.nChannels) : ringGraph.nChannels;
  if (comm->nChannels < nChannelsOrig) {
    // We started duplicating channels during Preset(), so we need to move the
    // duplicated channels since we have removed some.
    for (int i=0; i<comm->nChannels; i++) memcpy(comm->channels+comm->nChannels+i, comm->channels+nChannelsOrig+i, sizeof(struct ncclChannel));
  }

  // Determine CollNet support after all-gather now that we know nNodes and each node localRanks
  if (comm->collNetSupport == 1) {
    int collNetNodeThreshold = ncclParamCollNetNodeThreshold();
    if (comm->nNodes < collNetNodeThreshold) {
      INFO(NCCL_INIT, "Communicator has %d nodes which is less than CollNet node threshold %d, disabling CollNet", comm->nNodes, collNetNodeThreshold);
      comm->collNetSupport = 0;
    }
    for (int n=0; n<comm->nNodes; n++) {
      if (comm->nodeRanks[n].localRanks > NCCL_MAX_DIRECT_ARITY+1) {
        WARN("CollNet currently only supports up to %d GPUs per node, disabling CollNet", NCCL_MAX_DIRECT_ARITY+1);
        comm->collNetSupport = 0;
        break;
      }
    }
  }

  int *rings;
  NCCLCHECK(ncclCalloc(&rings, nranks*MAXCHANNELS));
  NCCLCHECK(ncclTopoPostset(comm, nodesFirstRank, nodesTreePatterns, allTopoRanks, rings, &collNetGraph, nc));

  if (comm->topo->pivotA2ANumBiRings == 3) NCCLCHECK(ncclTreeBasePostset(comm, &treeGraph));

  free(allTopoRanks);
  free(nodesTreePatterns);
  free(nodesFirstRank);
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
    int highestTypes[NCCL_MAX_LOCAL_RANKS] = {TRANSPORT_P2P};
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
        collNetSetupFail += ncclTransportCollNetSetup(comm, &collNetGraph, channel, head, head, h, collNetSend);
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
    int highestTransportType0, highestTransportType1;
    for (int c=0; c<comm->nChannels; c++) {
      struct ncclChannel* channelRecv = comm->channels+c;
      NCCLCHECKGOTO(ncclTransportP2pConnect(comm, channelRecv, NCCL_MAX_DIRECT_ARITY, channelRecv->collTree.up, NCCL_MAX_DIRECT_ARITY, channelRecv->collTree.down, 0), ret, collnet_cleanup);
    }
    NCCLCHECKGOTO(ncclTransportP2pSetup(comm, &collNetGraph, 0, &highestTransportType0), ret, collnet_cleanup);
    for (int c=0; c<comm->nChannels; c++) {
      struct ncclChannel* channelSend = comm->channels+c;
      NCCLCHECKGOTO(ncclTransportP2pConnect(comm, channelSend, NCCL_MAX_DIRECT_ARITY, channelSend->collTree.down, NCCL_MAX_DIRECT_ARITY, channelSend->collTree.up, 1), ret, collnet_cleanup);
    }
    NCCLCHECKGOTO(ncclTransportP2pSetup(comm, &collNetGraph, 1, &highestTransportType1), ret, collnet_cleanup);

    // Exchange highest intra-node transport type among ranks
    // because we need to know whether all ranks can p2p each other to determine whether we can directly read/write registered user buffer
    comm->intraHighestTransportType = highestTypes[comm->localRank] = highestTransportType0 > highestTransportType1 ? highestTransportType0 : highestTransportType1;
    NCCLCHECK(bootstrapIntraNodeAllGather(comm->bootstrap, comm->localRankToRank, comm->localRank, comm->localRanks, highestTypes, sizeof(int)));
    for (int i=0; i<comm->localRanks; i++) {
      if (highestTypes[i] > comm->intraHighestTransportType)
        comm->intraHighestTransportType = highestTypes[i];
    }
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
  do {
    int myCompCap = comm->peerInfo[rank].cudaCompCap;
    int minCompCap = myCompCap, maxCompCap = myCompCap;
    for (int i = 0; i < nranks; i++) {
      minCompCap = std::min(comm->peerInfo[i].cudaCompCap, minCompCap);
      maxCompCap = std::max(comm->peerInfo[i].cudaCompCap, maxCompCap);
    }
    NCCLCHECK(ncclTopoTuneModel(comm, minCompCap, maxCompCap, &treeGraph, &ringGraph, &collNetGraph));
  } while(0);

  // Compute nChannels per peer for p2p
  NCCLCHECK(ncclTopoComputeP2pChannels(comm));

  if (ncclParamNvbPreconnect()) {
    // Connect p2p when using NVB path
    int nvbNpeers;
    int* nvbPeers;
    NCCLCHECK(ncclTopoGetNvbGpus(comm->topo, comm->rank, &nvbNpeers, &nvbPeers));
    for (int r=0; r<nvbNpeers; r++) {
      int peer = nvbPeers[r];
      int channelId;
      for (int c=0; c<comm->p2pnChannelsPerPeer; c++) {
        NCCLCHECK(ncclChannelCompute(comm, peer, c, ncclFuncSend, &channelId));
        if (comm->channels[channelId].peers[peer].send[1].connected == 0) {
          comm->connectSend[peer] |= (1<<channelId);
        }
      }
      for (int c=0; c<comm->p2pnChannelsPerPeer; c++) {
        NCCLCHECK(ncclChannelCompute(comm, peer, c, ncclFuncRecv, &channelId));
        if (comm->channels[channelId].peers[peer].recv[1].connected == 0) {
          comm->connectRecv[peer] |= (1<<channelId);
        }
      }
    }
    NCCLCHECK(ncclTransportP2pSetup(comm, NULL, 1));
    free(nvbPeers);
  }

  // Connect to local net proxy
  struct ncclProxyConnector proxyConn;
  NCCLCHECK(ncclProxyConnect(comm, TRANSPORT_NET, 1, comm->rank, &proxyConn));
  NCCLCHECK(ncclProxyCall(&proxyConn, ncclProxyMsgSharedInit, &comm->p2pnChannels, sizeof(int), NULL, 0));

  // Then to remote ones when using PXN
  if (ncclPxnDisable() == 0) {
    int nranks;
    int* pxnPeers;
    NCCLCHECK(ncclTopoGetPxnRanks(comm, &pxnPeers, &nranks));
    for (int r=0; r<nranks; r++) {
      NCCLCHECK(ncclProxyConnect(comm, TRANSPORT_NET, 1, pxnPeers[r], &proxyConn));
      NCCLCHECK(ncclProxyCall(&proxyConn, ncclProxyMsgSharedInit, &comm->p2pnChannels, sizeof(int), NULL, 0));
    }
    free(pxnPeers);
  }

  do {
    // Compute intra-process ranks
    int intraProcRank0 = -1, intraProcRank = -1, intraProcRanks = 0;
    for (int i = 0; i < nranks; i++) {
      if ((comm->peerInfo[i].hostHash == comm->peerInfo[rank].hostHash)
          && (comm->peerInfo[i].pidHash == comm->peerInfo[rank].pidHash)) {
        // Rank is in same process
        if (intraProcRanks == 0) intraProcRank0 = i;
        if (i == rank) intraProcRank = intraProcRanks;
        intraProcRanks++;
      }
    }
    TRACE(NCCL_INIT,"pidHash[%d] %lx intraProcRank %d intraProcRanks %d intraProcRank0 %d",
        rank, comm->peerInfo[rank].pidHash, intraProcRank, intraProcRanks, intraProcRank0);
    if (intraProcRank == -1 || intraProcRank0 == -1 || comm->peerInfo[intraProcRank0].comm == NULL) {
      WARN("Failed to determine intra proc ranks rank %d hostHash %lx pidHash %lx intraProcRank %d intraProcRanks %d intraProcRank0 %d",
          rank, comm->peerInfo[rank].hostHash, comm->peerInfo[rank].pidHash,
          intraProcRank, intraProcRanks, intraProcRank0);
      return ncclInternalError;
    }
    NCCLCHECK(ncclCommSetIntraProc(comm, intraProcRank, intraProcRanks, comm->peerInfo[intraProcRank0].comm));
  } while(0);

  /* Local intra-node barrier */
  NCCLCHECK(bootstrapBarrier(comm->bootstrap, comm->localRankToRank, comm->localRank, comm->localRanks, comm->localRankToRank[0]));

  // Unlink proxy shm to make sure it will be properly cleaned up.
  NCCLCHECK(ncclProxyShmUnlink(comm));

  // We should have allocated all buffers, collective fifos, ... we can
  // restore the affinity.
affinity_restore:
  if (CPU_COUNT(&comm->cpuAffinity)) sched_setaffinity(0, sizeof(cpu_set_t), &affinitySave);
  if (ret != ncclSuccess) return ret;

  TRACE(NCCL_INIT, "rank %d nranks %d - DONE", rank, nranks);
  return ncclSuccess;
}

NCCL_PARAM(SetStackSize, "SET_STACK_SIZE", 0);

ncclResult_t ncclCommInitRankSync(ncclComm_t* newcomm, int nranks, ncclUniqueId commId, int myrank, int cudaDev, int virtualId) {
  ncclResult_t res;

  CUDACHECK(hipSetDevice(cudaDev));
  // Set the maximum kernel stack size of all kernels to avoid
  // a CUDA memory reconfig on load (c.f. NVSHMEM issue)
  if (maxLocalSizeBytes > 0 && ncclParamSetStackSize() == 1) {
    TRACE(NCCL_INIT, "Setting cudaLimitStackSize to %zi", maxLocalSizeBytes);
    //CUDACHECKIGNORE(hipDeviceSetLimit(hipLimitStackSize, maxLocalSizeBytes));
  }
  *newcomm = NULL;
  NCCLCHECKGOTO(commAlloc(newcomm, nranks, myrank, virtualId), res, cleanup);
  NCCLCHECKGOTO(initTransportsRank(*newcomm, &commId), res, cleanup);
  NCCLCHECKGOTO(devCommSetup(*newcomm), res, cleanup);

  INFO(NCCL_INIT,"comm %p rank %d nranks %d cudaDev %d busId %lx localSize %ld used %ld bytes - Init COMPLETE", *newcomm, myrank, nranks, (*newcomm)->cudaDev, (*newcomm)->busId, ncclKernLocalSize(ncclGetKernelIndex(*newcomm)), allocTracker[(*newcomm)->cudaDev].totalAllocSize);

  return ncclSuccess;
cleanup:
  if ((*newcomm) && (*newcomm)->bootstrap) bootstrapAbort((*newcomm)->bootstrap);
  *newcomm = NULL;
  return res;
}

static ncclResult_t ncclCommInitRankDev(ncclComm_t* newcomm, int nranks, ncclUniqueId commId, int myrank, int cudaDev, int virtualId) {
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
    NCCLCHECKGOTO(ncclAsyncInit(ncclCommInitRankSync, newcomm, nranks, commId, myrank, cudaDev, virtualId), res, end);
  } else {
    NCCLCHECKGOTO(ncclCommInitRankSync(newcomm, nranks, commId, myrank, cudaDev, virtualId), res, end);
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
  NCCLCHECK(ncclCommInitRankDev(newcomm, nranks, commId, myrank, cudaDev, -1));
  return ncclSuccess;
}

NCCL_API(ncclResult_t, ncclCommInitRankMulti, ncclComm_t* newcomm, int nranks, ncclUniqueId commId, int myrank, int virtualId);
ncclResult_t ncclCommInitRankMulti(ncclComm_t* newcomm, int nranks, ncclUniqueId commId, int myrank, int virtualId) {
  NVTX3_FUNC_RANGE_IN(nccl_domain);
  int cudaDev;
  CUDACHECK(hipGetDevice(&cudaDev));
  NCCLCHECK(ncclCommInitRankDev(newcomm, nranks, commId, myrank, cudaDev, virtualId));
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
    ncclCommInitRankDev(comms+i, ndev, uniqueId, i, devlist ? devlist[i] : i, -1);
  }
  NCCLCHECK(ncclGroupEnd());
  return ncclSuccess;
}

static ncclResult_t ncclGraphHelperDestroy(ncclComm* comm) {
  auto res = comm->graphHelperResources;
  if (comm->graphHelperThread && res) {
    pthread_mutex_lock(&res->threadLock);
    res->threadState = ThreadStop;
    pthread_cond_signal(&res->threadCond);
    pthread_mutex_unlock(&res->threadLock);
    pthread_join(comm->graphHelperThread, NULL);
  }
  if (res) {
    free(res);
    res = NULL;
  }
  return ncclSuccess;
}

static ncclResult_t commDestroy(ncclComm_t comm) {
  // Try and prevent a double free of the comm struct (user error)
  if (comm->rank == -1 || comm->nRanks <= 0 || comm->cudaDev == -1 || comm->busId == -1) {
    WARN("comm %p has already been destroyed", comm);
    return ncclInvalidArgument;
  }

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

  ncclDestroyQueueInfo(comm->enqueueInfo);

  if (rcclParamEnableHipGraph())
    NCCLCHECK(ncclGraphHelperDestroy(comm));

  INFO(NCCL_COLL, "Created %d queue info, destroyed %d", comm->nQueueInfoCreated, comm->nQueueInfoDestroyed);

  NCCLCHECK(commFree(comm));

  if (savedDevice != commDevice)
    CUDACHECK(hipSetDevice(savedDevice));

#if defined(ENABLE_NPKIT)
  // Dump NPKit events and shutdown
  const char* npkitDumpDir = getenv("NPKIT_DUMP_DIR");
  if (npkitDumpDir == nullptr) {
    WARN("NPKIT_DUMP_DIR is empty");
  } else {
    NCCLCHECK(NpKit::Dump(npkitDumpDir));
  }
  NCCLCHECK(NpKit::Shutdown());
#endif

  return ncclSuccess;
}

NCCL_API(ncclResult_t, ncclCommDestroy, ncclComm_t comm);
ncclResult_t ncclCommDestroy(ncclComm_t comm) {
  NVTX3_FUNC_RANGE_IN(nccl_domain);
  if (comm == NULL)
    return ncclSuccess;

  int rank = comm->rank, nranks = comm->nRanks, cudaDev = comm->cudaDev;
  int64_t busId = comm->busId;
  TRACE(NCCL_INIT, "comm %p rank %d nRanks %d cudaDev %d busId %lx", comm, rank, nranks, cudaDev, busId);

  // [RCCL] Delete CliqueManager if it exists
  //if (comm->cliqueManager) delete comm->cliqueManager;
  // [/RCCL]

  NCCLCHECK(commDestroy(comm));
  INFO(NCCL_INIT,"comm %p rank %d nranks %d cudaDev %d busId %lx - Destroy COMPLETE", comm, rank, nranks, cudaDev, busId);
  return ncclSuccess;
}

NCCL_API(ncclResult_t, ncclCommAbort, ncclComm_t comm);
ncclResult_t ncclCommAbort(ncclComm_t comm) {
  NVTX3_FUNC_RANGE_IN(nccl_domain);
  if (comm == NULL)
    return ncclSuccess;

  int rank = comm->rank, nranks = comm->nRanks, cudaDev = comm->cudaDev;
  int64_t busId = comm->busId;
  TRACE(NCCL_INIT, "comm %p rank %d nRanks %d cudaDev %d busId %lx", comm, rank, nranks, cudaDev, busId);

  // Ask anything that might still be running on the device to quit
  *comm->abortFlag = 1;

  // do not destroy comm because kernel maybe still running
  // return commDestroy(comm);
  INFO(NCCL_INIT,"comm %p rank %d nranks %d cudaDev %d busId %lx - Abort COMPLETE", comm, rank, nranks, cudaDev, busId);
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
