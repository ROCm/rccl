/*************************************************************************
 * Copyright (c) 2015-2022, NVIDIA CORPORATION. All rights reserved.
 * Modifications Copyright (c) 2019-2022 Advanced Micro Devices, Inc. All rights reserved.
 * Modifications Copyright (c) Microsoft Corporation. Licensed under the MIT License.
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

#include "msccl/msccl_lifecycle.h"

#define STR2(v) #v
#define STR(v) STR2(v)

#if CUDART_VERSION >= 9020 || defined(__HIP_PLATFORM_HCC__) || defined(__HCC__) || defined(__HIPCC__)
#define NCCL_GROUP_CUDA_STREAM 0 // CGMD: CUDA 9.2,10.X Don't need to use an internal CUDA stream
#else
#define NCCL_GROUP_CUDA_STREAM 1 // CGMD: CUDA 9.0,9.1 Need to use an internal CUDA stream
#endif

const char* ncclFuncStr[NCCL_NUM_FUNCTIONS+2] = { "Broadcast", "Reduce", "AllGather", "ReduceScatter", "AllReduce", "SendRecv", "AllToAllPivot" };
const char* ncclAlgoStr[NCCL_NUM_ALGORITHMS] = { "Tree", "Ring", "CollNetDirect", "CollNetChain" };
const char* ncclProtoStr[NCCL_NUM_PROTOCOLS] = { "LL", "LL128", "Simple" };
const char* ncclDevRedOpStr[ncclNumDevRedOps] = { "Sum", "Prod", "Max", "Min", "PreMulSum", "SumPostDiv" };
const char *ncclTypeStr[ncclNumTypes] = {"_i8", "_u8", "_i32", "_u32", "_i64", "_u64", "_f16", "_f32", "_f64", "_b16"};

NCCL_PARAM(GroupCudaStream, "GROUP_CUDA_STREAM", NCCL_GROUP_CUDA_STREAM);

NCCL_PARAM(CheckPointers, "CHECK_POINTERS", 0);
NCCL_PARAM(CommBlocking, "COMM_BLOCKING", 0);

struct allocationTracker allocTracker[MAX_ALLOC_TRACK_NGPU] = {};

static uint64_t hashUniqueId(ncclUniqueId const &id) {
  char const *bytes = (char const*)&id;
  uint64_t h = 0xdeadbeef;
  for(int i=0; i < (int)sizeof(ncclUniqueId); i++) {
    h ^= h >> 32;
    h *= 0x8db3db47fa2994ad;
    h += bytes[i];
  }
  return h;
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


NCCL_PARAM(L1SharedMemoryCarveout, "L1_SHARED_MEMORY_CARVEOUT", 0);

pthread_mutex_t initLock = PTHREAD_MUTEX_INITIALIZER;
static bool initialized = false;
static size_t maxLocalSizeBytes = 0;

static ncclResult_t ncclInit() {
  if (__atomic_load_n(&initialized, __ATOMIC_ACQUIRE)) return ncclSuccess;
  pthread_mutex_lock(&initLock);
  if (!initialized) {
    initEnv();
    initGdrCopy();
    maxLocalSizeBytes = ncclKernMaxLocalSize();
    int carveout = ncclParamL1SharedMemoryCarveout();
    if (carveout) ncclKernSetSharedMemoryCarveout(carveout);
    // Always initialize bootstrap network
    NCCLCHECK(bootstrapNetInit());
    NCCLCHECK(ncclNetPluginInit());

    __atomic_store_n(&initialized, true, __ATOMIC_RELEASE);
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
  ncclResult_t res = bootstrapGetUniqueId(out);
  TRACE_CALL("ncclGetUniqueId(0x%llx)", (unsigned long long)hashUniqueId(*out));
  return res;
}

// Prevent compiler from optimizing out these operations
#ifdef __clang__
#define NCCL_NO_OPTIMIZE __attribute__((optnone))
#else
#define NCCL_NO_OPTIMIZE __attribute__((optimize("O0")))
#endif

void NCCL_NO_OPTIMIZE commPoison(ncclComm_t comm) {
  // Important that this does not trash intraComm0 & intraRefs.
  comm->rank = comm->cudaDev = comm->busId = comm->nRanks = -1;
}

RCCL_PARAM(KernelCollTraceEnable, "KERNEL_COLL_TRACE_ENABLE", 0);

#ifdef ENABLE_COLLTRACE
void *ncclCommThreadMain(void *arg) {
  ncclComm_t comm = (ncclComm_t)arg;
  int head = 0;
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
    int tail = (*comm->collTraceTail)%COLLTRACE_NUM_ITEMS;
    int count;
    if (head <= tail)
      count = tail - head;
    else
      count = COLLTRACE_NUM_ITEMS + head - tail;
    if (!count) {
      usleep(1000); //sleep 1ms
      continue;
    }
    for (int i = 0; i < count; i++) {
      volatile struct ncclCollTrace *td = comm->collTrace+head;
      uint8_t type = td->type;
      if (type == ncclCollTraceNotReady)
        break;
      char line[1024];
      int offset = 0;
      uint16_t fIdx = td->funcIndex;
      #define VEGA_GPU_RTC_FREQUENCY 2.5E7
      if (type == ncclCollTraceDataType) {
        sprintf(line, "## [%012.6f] [%02d:%02d] L:%04d DT %08x %016lx %016lx",
          (double)(td->timeStamp)/VEGA_GPU_RTC_FREQUENCY, comm->rank, td->bid,
          fIdx, td->data_0, td->opCount, td->data_1);
      } else {
        if (fIdx == FUNC_INDEX_P2P || type == ncclCollTraceP2pElemType)
          sprintf(line, "## [%012.6f] [%02d:%02d] %06x-%06x", (double)(td->timeStamp)/VEGA_GPU_RTC_FREQUENCY, comm->rank, td->bid, td->p2pOpCount[0], td->p2pOpCount[1]);
        else
          sprintf(line, "## [%012.6f] [%02d:%02d] %06lx", (double)(td->timeStamp)/VEGA_GPU_RTC_FREQUENCY, comm->rank, td->bid, td->opCount);
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
      td->type = ncclCollTraceNotReady;
      head ++;
      head %= COLLTRACE_NUM_ITEMS;
    }
  } while(!comm->collTraceExit);
  free(func_names);
  pthread_exit(NULL);
}
#endif

#undef NCCL_NO_OPTIMIZE


static ncclResult_t ncclDestructorFnFree(struct ncclDestructor* dtor) {
  free(dtor->obj);
  return ncclSuccess;
}
void ncclCommPushFree(struct ncclComm* comm, void* obj) {
  struct ncclDestructor* dtor = ncclMemoryStackAlloc<struct ncclDestructor>(&comm->memPermanent);
  dtor->fn = ncclDestructorFnFree;
  dtor->obj = obj;
  dtor->next = comm->destructorHead;
  comm->destructorHead = dtor;
}

static ncclResult_t ncclDestructorFnCudaFree(struct ncclDestructor* dtor) {
  CUDACHECK(cudaFree(dtor->obj));
  return ncclSuccess;
}
void ncclCommPushCudaFree(struct ncclComm* comm, void* obj) {
  struct ncclDestructor* dtor = ncclMemoryStackAlloc<struct ncclDestructor>(&comm->memPermanent);
  dtor->fn = ncclDestructorFnCudaFree;
  dtor->obj = obj;
  dtor->next = comm->destructorHead;
  comm->destructorHead = dtor;
}

static ncclResult_t ncclDestructorFnCudaHostFree(struct ncclDestructor* dtor) {
  CUDACHECK(cudaFreeHost(dtor->obj));
  return ncclSuccess;
}
void ncclCommPushCudaHostFree(struct ncclComm* comm, void* obj) {
  struct ncclDestructor* dtor = ncclMemoryStackAlloc<struct ncclDestructor>(&comm->memPermanent);
  dtor->fn = ncclDestructorFnCudaHostFree;
  dtor->obj = obj;
  dtor->next = comm->destructorHead;
  comm->destructorHead = dtor;
}

static ncclResult_t ncclDestructorFnCudaGdrFree(struct ncclDestructor* dtor) {
  NCCLCHECK(ncclGdrCudaFree(dtor->obj));
  return ncclSuccess;
}
void ncclCommPushCudaGdrFree(struct ncclComm* comm, void* handle) {
  struct ncclDestructor* dtor = ncclMemoryStackAlloc<struct ncclDestructor>(&comm->memPermanent);
  dtor->fn = ncclDestructorFnCudaGdrFree;
  dtor->obj = handle;
  dtor->next = comm->destructorHead;
  comm->destructorHead = dtor;
}

static ncclResult_t commFree(ncclComm_t comm) {
  if (comm == NULL)
    return ncclSuccess;

  // Stop all threads before we free anything.
  NCCLCHECK(ncclProxyDestroy(comm));

  delete[] comm->userRedOps;

  free(comm->connectSend);
  free(comm->connectRecv);

#ifdef ENABLE_PROFILING
  struct ncclProf *prof, *prof_seq;
  prof = (struct ncclProf*)malloc(sizeof(struct ncclProf)*MAXCHANNELS*PROFILE_NUM_LAUNCHES);
  CUDACHECK(hipMemcpy(prof, comm->devComm->devProf, sizeof(struct ncclProf)*MAXCHANNELS*PROFILE_NUM_LAUNCHES, hipMemcpyDeviceToHost));
  #define VEGA_GPU_RTC_FREQUENCY 2.5E7
  for (int i=0; i<comm->nChannels; i++) {
    for (int s=0; s<prof[MAXCHANNELS*i].seq; s++) {
      if (prof[MAXCHANNELS*s+i].count == 0) continue;
      for (int j=0; j<prof[MAXCHANNELS*s+i].count; j++) {
        INFO(NCCL_INIT, "# [%02d:%02d] %02d-%02d L:%04u %6.2fus", comm->rank, i, s, j, prof[MAXCHANNELS*s+i].elem[j].line, (prof[MAXCHANNELS*s+i].elem[j].timeStamp-prof[MAXCHANNELS*s+i].elem[0].timeStamp)/VEGA_GPU_RTC_FREQUENCY*1.0E6);
      }
    }
  }
  free(prof);
  CUDACHECK(hipFree(comm->devComm->devProf));
#endif

#ifdef ENABLE_COLLTRACE
  comm->collTraceExit = 1;
  if (comm->collTraceThread) pthread_join(comm->collTraceThread, NULL);
  NCCLCHECK(ncclCudaHostFree((void *)comm->collTrace));
  NCCLCHECK(ncclCudaHostFree((void *)comm->collTraceTail));
#endif

  free(comm->peerInfo);
  if (comm->topo)
    ncclTopoFree(comm->topo);
  if (comm->nodeRanks) {
    for (int n=0; n<comm->nNodes; n++) free(comm->nodeRanks[n].localRankToRank);
    free(comm->nodeRanks);
  }
  free(comm->rankToNode);
  free(comm->rankToLocalRank);

  if (comm->bootstrap)
    NCCLCHECK(bootstrapClose(comm->bootstrap));

  for (int channel=0; channel<MAXCHANNELS; channel++)
    NCCLCHECK(freeChannel(comm->channels+channel, comm->nRanks));

  if (comm->doneEvent != NULL)
    CUDACHECK(hipEventDestroy(comm->doneEvent));

  if (comm->initState == ncclSuccess) {
    NCCLCHECK(ncclStrongStreamDestruct(&comm->hostStream));
    NCCLCHECK(ncclStrongStreamDestruct(&comm->deviceStream));
  }

  struct ncclDestructor* dtor = comm->destructorHead;
  while (dtor != nullptr) {
    NCCLCHECK(dtor->fn(dtor));
    dtor = dtor->next;
  }
  CUDACHECK(hipStreamDestroy(comm->sideStream));

  ncclMemoryStackDestruct(&comm->memScoped);
  ncclMemoryStackDestruct(&comm->memPermanent);

  commPoison(comm); // Important that this does not interfere with anything used below.

  if (comm->initState == ncclSuccess) {
    struct ncclComm* intraComm0 = comm->intraComm0;
    if (0 == ncclAtomicRefCountDecrement(&intraComm0->intraRefs)) {
      // Wait for all service threads to be done. We could not
      // do it earlier because it could have blocked and prevented
      // other ranks in the process to call ncclCommDestroy
      comm = intraComm0;
      while (comm != nullptr) {
        if (comm->proxyState.thread) pthread_join(comm->proxyState.thread, nullptr);
        struct ncclComm* next = comm->intraNext;
        free(comm);
        comm = next;
      }
    }
  } else if (comm->proxyState.thread) {
    pthread_join(comm->proxyState.thread, nullptr);
    ncclCudaHostFree((void *)comm->abortFlag);
    free(comm);
  } else {
    ncclCudaHostFree((void *)comm->abortFlag);
    free(comm);
  }

  return ncclSuccess;
}

RCCL_PARAM(CliqueIgnoreTopo, "CLIQUE_IGNORE_TOPO", 0);
RCCL_PARAM(P2pNetDisable, "P2P_NET_DISABLE", 0);
RCCL_PARAM(PivotAlltoallEnable, "PIVOT_ALLTOALL_ENABLE", 1);
RCCL_PARAM(LL128ForceEnable, "LL128_FORCE_ENABLE", 0);
NCCL_PARAM(AggChannelSize, "AGG_CHANNEL_SIZE", -2);
NCCL_PARAM(DisableGraphHelper, "GRAPH_HELPER_DISABLE", 0);
// GDRCOPY support: FIFO_ENABLE when enabled locates a workFifo in CUDA memory
NCCL_PARAM(GdrCopyFifoEnable, "GDRCOPY_FIFO_ENABLE", 1);
NCCL_PARAM(WorkFifoDepth, "WORK_FIFO_DEPTH", 64<<10);
enum ncclLaunchMode ncclParamLaunchMode;

NCCL_PARAM(DmaBufEnable, "DMABUF_ENABLE", 0);

// Detect DMA-BUF support
static ncclResult_t dmaBufSupported(struct ncclComm* comm) {
  if (ncclParamDmaBufEnable() == 0 || comm->ncclNet->regMrDmaBuf == NULL) return ncclInternalError;
#if CUDA_VERSION >= 11070
  int flag = 0;
  CUdevice dev;
  int cudaDriverVersion;
  CUCHECK(cuDriverGetVersion(&cudaDriverVersion));
  if (cudaDriverVersion < 11070) return ncclInternalError;
  CUCHECK(cuDeviceGet(&dev, comm->cudaDev));
  // Query device to see if DMA-BUF support is available
  (void) CUPFN(cuDeviceGetAttribute(&flag, CU_DEVICE_ATTRIBUTE_DMA_BUF_SUPPORTED, dev));
  if (flag == 0) return ncclInternalError;
  INFO(NCCL_INIT, "DMA-BUF is available on GPU device %d", comm->cudaDev);
  return ncclSuccess;
#else
  return pfn_hsa_amd_portable_export_dmabuf != NULL ? ncclSuccess : ncclInternalError;
#endif
  return ncclInternalError;
}

ncclResult_t ncclCommEnsureReady(ncclComm_t comm) {
  /* comm must be ready, or error will be reported */
  ncclResult_t ret = ncclSuccess;

  if (*comm->abortFlag) {
    ncclGroupJobAbort();
  } else {
    NCCLCHECK(ncclCommGetAsyncError(comm, &ret));
    if (ret != ncclSuccess) {
      /* if ret is not ncclInProgress, we just keep it. */
      WARN("Attempt to use communicator before the previous operation returned ncclSuccess\n");
      if (ret == ncclInProgress) ret = ncclInvalidArgument;
      goto exit;
    }
  }

exit:
  return ret;
}

static ncclResult_t commAlloc(ncclComm_t* comret, int ndev, int rank, int virtualId) {
  if (ndev < 1) {
    WARN("invalid device count (%d) requested", ndev);
    return ncclInvalidArgument;
  }
  if (rank >= ndev || rank < 0) {
    WARN("rank %d exceeds ndev=%d", rank, ndev);
    return ncclInvalidArgument;
  }

  struct ncclComm* comm;
  /* Cuurently we calloc comm in ncclCommInitRankDev for async function support.
   * This 'if' structure is designed to consider the case where commAlloc is called
   * in other cases except ncclCommInitRankDev. */
  if (*comret == NULL) {
    /* user requests a new communicator */
    NCCLCHECK(ncclCalloc(&comm, 1));
    NCCLCHECK(ncclCudaHostCalloc((uint32_t**)&comm->abortFlag, 1));
    NCCLCHECK(ncclCommSetAsyncError(comm, ncclInProgress));
  } else {
    /* We already allocated a communicator in ncclCommInitRankDev. */
    comm = *comret;
  }

  ncclMemoryStackConstruct(&comm->memPermanent);
  ncclMemoryStackConstruct(&comm->memScoped);
  comm->destructorHead = nullptr;
  comm->rank = rank;
  comm->nRanks = ndev;

  NCCLCHECK(ncclNetInit(comm));
  INFO(NCCL_INIT, "Using network %s", ncclNetName(comm));

  // Try to create a CUDA object right away. If there is something wrong with
  // the device we're on (failure cause #1) , better know it early.
  hipEvent_t doneEvent;
  CUDACHECK(hipEventCreateWithFlags(&doneEvent, hipEventDisableTiming));

  NCCLCHECK(ncclStrongStreamConstruct(&comm->deviceStream));
  NCCLCHECK(ncclStrongStreamConstruct(&comm->hostStream));

  comm->doneEvent = doneEvent;
  comm->lastStream = nullptr;
  comm->virtualId = virtualId;
  cudaGetDevice(&comm->cudaDev);
  NCCLCHECK(getBusId(comm->cudaDev, &comm->busId));
  TRACE(NCCL_INIT,"comm %p rank %d nranks %d cudaDev %d busId %lx", comm, rank, ndev, comm->cudaDev, comm->busId);

  // RCCL: create persistent stream for calloc
  CUDACHECK(hipStreamCreateWithFlags(&comm->sideStream, hipStreamNonBlocking));
  comm->checkPointers = ncclParamCheckPointers() == 1 ? true : false;
  comm->dmaBufSupport = (dmaBufSupported(comm) == ncclSuccess) ? true : false;

#ifdef ENABLE_COLLTRACE
  NCCLCHECK(ncclCudaHostCalloc((uint32_t **)&comm->collTraceTail, 1));
  NCCLCHECK(ncclCudaHostCalloc(&comm->collTrace, COLLTRACE_NUM_ITEMS));
  memset(comm->collTrace, 0, sizeof(struct ncclCollTrace) * COLLTRACE_NUM_ITEMS);
  comm->collTraceExit = *comm->collTraceTail = 0;
  if ((ncclDebugLevel >= NCCL_LOG_INFO) && rcclParamKernelCollTraceEnable())
    pthread_create(&comm->collTraceThread, NULL, ncclCommThreadMain, (void *)comm);
  else
    comm->collTraceThread = 0;
#endif
  comm->collNetSupport = 0;

  ncclMemoryPoolConstruct(&comm->memPool_ncclKernelPlan);
  ncclMemoryPoolConstruct(&comm->memPool_ncclProxyOp);
  ncclMemoryPoolConstruct(&comm->memPool_ncclPointerList);

  comm->groupNext = reinterpret_cast<struct ncclComm*>(0x1);
  comm->preconnectNext = reinterpret_cast<struct ncclComm*>(0x1);
  comm->channelSize = ncclParamAggChannelSize();

  static_assert(MAXCHANNELS <= sizeof(*comm->connectSend)*8, "comm->connectSend must have enough bits for all channels");
  static_assert(MAXCHANNELS <= sizeof(*comm->connectRecv)*8, "comm->connectRecv must have enough bits for all channels");
  NCCLCHECK(ncclCalloc(&comm->connectSend, comm->nRanks*NCCL_MAX_CONNS));
  NCCLCHECK(ncclCalloc(&comm->connectRecv, comm->nRanks*NCCL_MAX_CONNS));

  // Mark channels as non initialized.
  for (int c=0; c < MAXCHANNELS; c++) comm->channels[c].id = -1;

  ncclIntruQueueMpscConstruct(&comm->callbackQueue);

  CUDACHECK(hipDeviceGetAttribute(&comm->WarpSize, hipDeviceAttributeWarpSize, comm->cudaDev));
  *comret = comm;
  return ncclSuccess;
}

static ncclResult_t devCommSetup(ncclComm_t comm) {

  NCCLCHECK(ncclStrongStreamAcquireUncaptured(&comm->deviceStream));

  int nRanks = comm->nRanks;
  struct ncclDevCommAndChannels *devCommAndChans, tmpCommAndChans;
  NCCLCHECK(ncclCudaCallocAsync(&devCommAndChans, 1, comm->deviceStream.cudaStream));
  ncclCommPushCudaFree(comm, devCommAndChans);
  comm->devComm = &devCommAndChans->comm;
  tmpCommAndChans.comm.rank = comm->rank;
  tmpCommAndChans.comm.nRanks = nRanks;
  tmpCommAndChans.comm.abortFlag = comm->abortFlag;
  for (int p=0; p < NCCL_NUM_PROTOCOLS; p++) {
    tmpCommAndChans.comm.buffSizes[p] = comm->buffSizes[p];
  }
  tmpCommAndChans.comm.channels = &devCommAndChans->channels[0];

  comm->workFifoDepth = ncclParamWorkFifoDepth();
  if (0 != (comm->workFifoDepth & (comm->workFifoDepth-1))) {
    WARN("NCCL_WORK_FIFO_DEPTH=%d is being ignored because it is not a power of 2.", comm->workFifoDepth);
    comm->workFifoDepth = 64<<10;
  }
  tmpCommAndChans.comm.workFifoDepth = comm->workFifoDepth;

  if (ncclGdrCopy != NULL && ncclParamGdrCopyFifoEnable() == 1) {
    // The workFifoHeap lives in GDR mapped CUDA memory.
    NCCLCHECK(ncclGdrCudaCalloc(&comm->workFifoHeap, &comm->devWorkFifoHeap, comm->workFifoDepth, &comm->workFifoHeapGdrHandle));
    ncclCommPushCudaGdrFree(comm, comm->workFifoHeapGdrHandle);
  } else {
    // The workFifoHeap lives in cudaHost memory.
    comm->workFifoHeapGdrHandle = nullptr;
    NCCLCHECK(ncclCudaHostCalloc(&comm->workFifoHeap, comm->workFifoDepth));
    ncclCommPushCudaHostFree(comm, comm->workFifoHeap);
    comm->devWorkFifoHeap = comm->workFifoHeap;
  }
  tmpCommAndChans.comm.workFifoHeap = comm->devWorkFifoHeap;

  NCCLCHECK(ncclCudaHostCalloc(&comm->workFifoDone, MAXCHANNELS));
  ncclCommPushCudaHostFree(comm, comm->workFifoDone);
  comm->workFifoSent = 0;
  comm->workFifoAckdMin = 0;

  for (int c=0; c < MAXCHANNELS; c++) {
    tmpCommAndChans.channels[c].peers = comm->channels[c].devPeers;
    tmpCommAndChans.channels[c].ring = comm->channels[c].ring;
    tmpCommAndChans.channels[c].ring.userRanks = comm->channels[c].devRingUserRanks;
    tmpCommAndChans.channels[c].tree = comm->channels[c].tree;
    tmpCommAndChans.channels[c].collnetChain = comm->channels[c].collnetChain;
    tmpCommAndChans.channels[c].collnetDirect = comm->channels[c].collnetDirect;
    tmpCommAndChans.channels[c].binTree = comm->channels[c].binTree;
    tmpCommAndChans.channels[c].workFifoDone = &comm->workFifoDone[c];

    if (comm->channels[c].ring.userRanks != nullptr) {
      NCCLCHECK(ncclCudaMemcpyAsync(tmpCommAndChans.channels[c].ring.userRanks, comm->channels[c].ring.userRanks, nRanks, comm->deviceStream.cudaStream));
    }
  }

#ifdef ENABLE_COLLTRACE
  tmpCommAndChans.comm.collTrace = comm->collTrace;
  tmpCommAndChans.comm.collTraceTail = comm->collTraceTail;
  tmpCommAndChans.comm.collTraceThread = comm->collTraceThread;
#endif

#if defined(ENABLE_NPKIT)
  // Init NPKit
  NCCLCHECK(NpKit::Init(comm->rank));
  tmpCommAndChans.comm.npKitEventCollectContexts = NpKit::GetGpuEventCollectContexts();
  tmpCommAndChans.comm.cpuTimestamp = NpKit::GetCpuTimestamp();
#endif

#ifdef ENABLE_PROFILING
  NCCLCHECK(ncclCudaCalloc(&tmpCommAndChans.comm.devProf, MAXCHANNELS*PROFILE_NUM_LAUNCHES), comm->sideStream);
#endif

  if (mscclEnabled()) {
    NCCLCHECK(mscclInit(comm));
  }

  NCCLCHECK(ncclCudaMemcpyAsync(devCommAndChans, &tmpCommAndChans, 1, comm->deviceStream.cudaStream));
  CUDACHECK(cudaStreamSynchronize(comm->deviceStream.cudaStream));
  NCCLCHECK(ncclStrongStreamRelease(ncclCudaGraphNone(), &comm->deviceStream));
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
  CUDACHECK(cudaGetDevice(&info->cudaDev));
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
    NCCLCHECK(ncclGpuGdrSupport(comm, &info->gdrSupport));
  }
  else {
    info->hasFineGrain = false;
    info->gdrSupport = 0;
  }
  comm->hasFineGrain = info->hasFineGrain;

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

#define DEFAULT_LL_BUFFSIZE (NCCL_LL_LINES_PER_THREAD*NCCL_LL_MAX_NTHREADS*NCCL_STEPS*sizeof(union ncclLLFifoLine))
#define DEFAULT_LL128_BUFFSIZE (NCCL_LL128_ELEMS_PER_THREAD*NCCL_LL128_MAX_NTHREADS*NCCL_STEPS*sizeof(uint64_t))
#define DEFAULT_BUFFSIZE (1 << 22) /* 4MiB */
#define DEFAULT_BUFFSIZE_ARM (1 << 20) /* 1MiB */
NCCL_PARAM(BuffSize, "BUFFSIZE", -2);
NCCL_PARAM(LlBuffSize, "LL_BUFFSIZE", -2);
NCCL_PARAM(Ll128BuffSize, "LL128_BUFFSIZE", -2);

NCCL_PARAM(P2pNetChunkSize, "P2P_NET_CHUNKSIZE", (1 << 17)); /* 128 kB */

static ncclResult_t computeBuffSizes(struct ncclComm* comm) {
  int cpuArch, cpuVendor, cpuModel;
  NCCLCHECK(ncclTopoCpuType(comm->topo, &cpuArch, &cpuVendor, &cpuModel));

  int64_t envs[NCCL_NUM_PROTOCOLS] = { ncclParamLlBuffSize(), ncclParamLl128BuffSize(), ncclParamBuffSize() };
  int defaults[NCCL_NUM_PROTOCOLS] = { DEFAULT_LL_BUFFSIZE, DEFAULT_LL128_BUFFSIZE, DEFAULT_BUFFSIZE };

  if (cpuArch == NCCL_TOPO_CPU_ARCH_ARM) defaults[NCCL_PROTO_SIMPLE] = DEFAULT_BUFFSIZE_ARM;

  for (int p=0; p<NCCL_NUM_PROTOCOLS; p++) {
    comm->buffSizes[p] = envs[p] != -2 ? envs[p] : defaults[p];
  }

  comm->p2pNetChunkSize = ncclParamP2pNetChunkSize();
  return ncclSuccess;
}

NCCL_PARAM(GraphDumpFileRank, "GRAPH_DUMP_FILE_RANK", 0);
NCCL_PARAM(CollNetNodeThreshold, "COLLNET_NODE_THRESHOLD", 2);
NCCL_PARAM(NvbPreconnect, "NVB_PRECONNECT", 0);
NCCL_PARAM(AllocP2pNetLLBuffers, "NCCL_ALLOC_P2P_NET_LL_BUFFERS", 0);

static ncclResult_t initTransportsRank(struct ncclComm* comm, ncclUniqueId* commId) {
  // We use 2 AllGathers
  // 1. { peerInfo, comm, compCap}
  // 2. { nChannels, graphInfo, topoRanks }

  int rank = comm->rank;
  int nranks = comm->nRanks;
  uint64_t commHash = getHash(commId->internal, NCCL_UNIQUE_ID_BYTES);
  TRACE(NCCL_INIT, "comm %p, commHash %lx, rank %d nranks %d - BEGIN", comm, commHash, rank, nranks);
  NCCLCHECK(bootstrapInit(commId, comm));

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
  NCCLCHECK(ncclTopoComputePaths(comm->topo, comm));
  // Remove inaccessible GPUs and unused NICs
  NCCLCHECK(ncclTopoTrimSystem(comm->topo, comm));
  // Recompute paths after trimming
  NCCLCHECK(ncclTopoComputePaths(comm->topo, comm));
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
  // Initialize num P2P LL buffers for this communicator
  comm->allocP2pNetLLBuffers = ncclParamAllocP2pNetLLBuffers() == 1;

  if (comm->rank == ncclParamGraphDumpFileRank()) {
    struct ncclTopoGraph* graphs[3] = { &ringGraph, &treeGraph, &collNetGraph };
    NCCLCHECK(ncclTopoDumpGraphs(comm->topo, 3, graphs));
  }

  // Determine local CollNet support before all-gather
  if (collNetSupport(comm)) {
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
    float bwIntra;
    float bwInter;
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
  if ( ((comm->topo->nodes[GPU].count == comm->topo->nRanks && comm->virtualId == -1)  ||
	(comm->topo->nodes[GPU].count <= comm->topo->nRanks && comm->virtualId != -1)) &&
       comm->topo->nodes[GPU].nodes[idx].gpu.gcn == 906 && allXgmi)
    allGather3Data[rank].nc = 4;
  if (comm->topo->nodes[GPU].nodes[idx].gpu.gcn == 908)
    allGather3Data[rank].nc = std::max(4/ringGraph.nChannels, 2);
  if ( ((comm->topo->nodes[GPU].count == comm->topo->nRanks && comm->virtualId == -1)  ||
	(comm->topo->nodes[GPU].count <= comm->topo->nRanks && comm->virtualId != -1)) &&
       (comm->topo->type & RCCL_TOPO_CR8G))
    allGather3Data[rank].nc = 4;
  if (((comm->topo->nodes[GPU].count == comm->topo->nRanks && comm->virtualId == -1)  ||
       (comm->topo->nodes[GPU].count <= comm->topo->nRanks && comm->virtualId != -1)) &&
      comm->topo->nodes[GPU].nodes[idx].gpu.gcn == 910)
    allGather3Data[rank].nc = 4;
  if (comm->topo->nodes[GPU].nodes[idx].gpu.gcn == 910)
    allGather3Data[rank].nc = std::max(allGather3Data[rank].nc, 4/ringGraph.nChannels);
  if (ringGraph.nChannels > MAXCHANNELS/2)
    allGather3Data[rank].nc = 1;
  NCCLCHECK(ncclTopoGetLocalNet(comm->topo, rank, &allGather3Data[rank].netDev));
  allGather3Data[rank].tree.pattern = treeGraph.pattern;
  allGather3Data[rank].tree.nChannels = treeGraph.nChannels;
  allGather3Data[rank].tree.sameChannels = treeGraph.sameChannels;
  allGather3Data[rank].tree.bwIntra = treeGraph.bwIntra;
  allGather3Data[rank].tree.bwInter = treeGraph.bwInter;
  allGather3Data[rank].tree.typeIntra = treeGraph.typeIntra;
  allGather3Data[rank].tree.typeInter = treeGraph.typeInter;
  allGather3Data[rank].ring.pattern = ringGraph.pattern;
  allGather3Data[rank].ring.nChannels = ringGraph.nChannels;
  allGather3Data[rank].ring.sameChannels = ringGraph.sameChannels;
  allGather3Data[rank].ring.bwIntra = ringGraph.bwIntra;
  allGather3Data[rank].ring.bwInter = ringGraph.bwInter;
  allGather3Data[rank].ring.typeIntra = ringGraph.typeIntra;
  allGather3Data[rank].ring.typeInter = ringGraph.typeInter;
  allGather3Data[rank].collNet.pattern = collNetGraph.pattern;
  allGather3Data[rank].collNet.nChannels = collNetGraph.nChannels;
  allGather3Data[rank].collNet.sameChannels = collNetGraph.sameChannels;
  allGather3Data[rank].collNet.bwIntra = collNetGraph.bwIntra;
  allGather3Data[rank].collNet.bwInter = collNetGraph.bwInter;
  allGather3Data[rank].collNet.typeIntra = collNetGraph.typeIntra;
  allGather3Data[rank].collNet.typeInter = collNetGraph.typeInter;
  allGather3Data[rank].collNetSupport = comm->collNetSupport;
  allGather3Data[rank].pivotA2AEnabled = comm->topo->pivotA2AEnabled && rcclParamPivotAlltoallEnable();
  comm->topo->ll128Enabled =  comm->topo->ll128Enabled || rcclParamLL128ForceEnable();
  allGather3Data[rank].ll128Enabled = comm->topo->ll128Enabled;

  comm->nChannels = (comm->topo->nodes[GPU].count != comm->topo->nRanks && comm->topo->nodes[NET].count)
    ? std::min(treeGraph.nChannels, ringGraph.nChannels) : ringGraph.nChannels;
  NCCLCHECK(ncclTopoPreset(comm, &treeGraph, &ringGraph, &collNetGraph, &allGather3Data[rank].topoRanks));

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
    treeGraph.bwIntra = std::min(allGather3Data[i].tree.bwIntra, treeGraph.bwIntra);
    treeGraph.bwInter = std::min(allGather3Data[i].tree.bwInter, treeGraph.bwInter);
    treeGraph.typeIntra = std::max(allGather3Data[i].tree.typeIntra, treeGraph.typeIntra);
    treeGraph.typeInter = std::max(allGather3Data[i].tree.typeInter, treeGraph.typeInter);
    ringGraph.nChannels = std::min(allGather3Data[i].ring.nChannels, ringGraph.nChannels);
    ringGraph.sameChannels = std::min(allGather3Data[i].ring.sameChannels, ringGraph.sameChannels);
    ringGraph.bwIntra = std::min(allGather3Data[i].ring.bwIntra, ringGraph.bwIntra);
    ringGraph.bwInter = std::min(allGather3Data[i].ring.bwInter, ringGraph.bwInter);
    ringGraph.typeIntra = std::max(allGather3Data[i].ring.typeIntra, ringGraph.typeIntra);
    ringGraph.typeInter = std::max(allGather3Data[i].ring.typeInter, ringGraph.typeInter);
    collNetGraph.nChannels = std::min(allGather3Data[i].collNet.nChannels, collNetGraph.nChannels);
    collNetGraph.sameChannels = std::min(allGather3Data[i].collNet.sameChannels, collNetGraph.sameChannels);
    collNetGraph.bwIntra = std::min(allGather3Data[i].collNet.bwIntra, collNetGraph.bwIntra);
    collNetGraph.bwInter = std::min(allGather3Data[i].collNet.bwInter, collNetGraph.bwInter);
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

  if (comm->topo->pivotA2ANumBiRings == 3) {
    NCCLCHECK(ncclTreeBasePostset(comm, &treeGraph));
    if (comm->virtualId == -1) {
      NCCLCHECK(ncclBinaryTreeHayabusaPostset(comm, &treeGraph));
    } else {
      NCCLCHECK(ncclBinaryTreePostset(comm, &treeGraph));
    }
  }

  free(allTopoRanks);
  free(nodesTreePatterns);
  free(nodesFirstRank);
  free(allGather3Data);

  // AllGather3 - end

  TRACE(NCCL_INIT, "rank %d nranks %d - BUILT %d TREES/RINGS", rank, nranks, comm->nChannels);

  char line[1024], binline[1024];
  line[0]='\0';
  binline[0]='\0';
  for (int c=0; c<comm->nChannels; c++) {
    struct ncclTree* tree = &comm->channels[c].tree;
    struct ncclTree* binTree = &comm->channels[c].binTree;
    snprintf(line+strlen(line), 1023-strlen(line), " [%d] %d/%d/%d->%d->%d",
        c, tree->down[0], tree->down[1], tree->down[2], rank, tree->up);
    if (comm->topo->pivotA2ANumBiRings == 3)
      snprintf(binline+strlen(binline), 1023-strlen(binline), " [%d] %d/%d/%d->%d->%d",
	       c, binTree->down[0], binTree->down[1], binTree->down[2], rank, binTree->up);
    INFO(NCCL_GRAPH, "Ring %d : %d -> %d -> %d comm %p nRanks %02d busId %lx", c, comm->channels[c].ring.prev,
         comm->rank, comm->channels[c].ring.next, comm, comm->nRanks, comm->busId);
  }
  line[1023] = '\0';
  INFO(NCCL_INIT, "Trees%s comm %p nRanks %02d busId %lx", line, comm, comm->nRanks, comm->busId);
  if (comm->topo->pivotA2ANumBiRings == 3) {
    binline[1023] = '\0';
    INFO(NCCL_INIT, "BinTrees%s comm %p nRanks %02d busId %lx", binline, comm, comm->nRanks, comm->busId);
  }

  NCCLCHECK(computeBuffSizes(comm));

  // Connect with prev/next for each ring
  for (int c=0; c<comm->nChannels; c++) {
    struct ncclChannel* channel = comm->channels+c;
    NCCLCHECKGOTO(setupChannel(comm, c, rank, nranks, rings+c*nranks), ret, affinity_restore);
    if (comm->nRanks == 1) continue;
    NCCLCHECKGOTO(ncclTransportP2pConnect(comm, c, 1, &channel->ring.prev, 1, &channel->ring.next, 0), ret, affinity_restore);
  }
  NCCLCHECKGOTO(ncclTransportP2pSetup(comm, &ringGraph, 0), ret, affinity_restore);
  if (ringGraph.nIntraChannels && rcclParamP2pNetDisable() == 0) {
    comm->useIntraNet = 1;
    // Connect NET for intranode use
    for (int c=0; c<comm->nChannels; c++) {
      struct ncclChannel* channel = comm->channels+c;
      if (comm->nRanks == 1) continue;
      NCCLCHECKGOTO(ncclTransportP2pConnect(comm, c, 1, &channel->ring.prev, 1, &channel->ring.next, NCCL_CONN_IDX_P2P_NET), ret, affinity_restore);
    }
    NCCLCHECKGOTO(ncclTransportP2pSetup(comm, &ringGraph, NCCL_CONN_IDX_P2P_NET), ret, affinity_restore);
  }
  free(rings);
  INFO(NCCL_INIT, "Connected all rings comm %p nRanks %02d busId %lx", comm, comm->nRanks, comm->busId);

  // Connect Trees
  for (int c=0; c<comm->nChannels; c++) {
    struct ncclChannel* channel = comm->channels+c;
    if (comm->nRanks == 1) continue;
    NCCLCHECKGOTO(ncclTransportP2pConnect(comm, c, NCCL_MAX_TREE_ARITY, channel->tree.down, 1, &channel->tree.up, 0), ret, affinity_restore);
    NCCLCHECKGOTO(ncclTransportP2pConnect(comm, c, 1, &channel->tree.up, NCCL_MAX_TREE_ARITY, channel->tree.down, 0), ret, affinity_restore);
    // RCCL: need to connect binTree as well
    if (comm->topo->pivotA2ANumBiRings == 3) {
      NCCLCHECKGOTO(ncclTransportP2pConnect(comm, c, NCCL_MAX_TREE_ARITY, channel->binTree.down, 1, &channel->binTree.up, 0), ret, affinity_restore);
      NCCLCHECKGOTO(ncclTransportP2pConnect(comm, c, 1, &channel->binTree.up, NCCL_MAX_TREE_ARITY, channel->binTree.down, 0), ret, affinity_restore);
    }
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

    char line[1024];
    line[0]='\0';
    for (int c=0; c<comm->nChannels; c++) {
      struct ncclTree* chain = &comm->channels[c].collnetChain;
      snprintf(line+strlen(line), 1023-strlen(line), " [%d] %d->%d->%d",
          c, chain->down[0], rank, chain->up);
    }
    line[1023] = '\0';
    INFO(NCCL_INIT, "Collnet Chains %s", line);
    // Connect Collnet + chain
    for (int c=0; c<comm->nChannels; c++) {
      struct ncclChannel* channel = comm->channels+c;
      NCCLCHECKGOTO(ncclTransportP2pConnect(comm, c, 1, &channel->collnetChain.up, 1, channel->collnetChain.down, 0), ret, collnet_cleanup);
    }
    NCCLCHECKGOTO(ncclTransportP2pSetup(comm, &collNetGraph, 0), ret, collnet_cleanup);
    for (int c=0; c<comm->nChannels; c++) {
      struct ncclChannel* channel = comm->channels+c;
      NCCLCHECKGOTO(ncclTransportP2pConnect(comm, c, 1, channel->collnetChain.down, 1, &channel->collnetChain.up, 1), ret, collnet_cleanup);
    }
    NCCLCHECKGOTO(ncclTransportP2pSetup(comm, &collNetGraph, 1), ret, collnet_cleanup);
    INFO(NCCL_INIT, "Connected collnet + chain");

    // Connect intra-node CollNet + Direct
    int highestTransportType0, highestTransportType1;
    for (int c=0; c<comm->nChannels; c++) {
      struct ncclChannel* channelRecv = comm->channels+c;
      NCCLCHECKGOTO(ncclTransportP2pConnect(comm, c, NCCL_MAX_DIRECT_ARITY, channelRecv->collnetDirect.up, NCCL_MAX_DIRECT_ARITY, channelRecv->collnetDirect.down, 0), ret, collnet_cleanup);
    }
    NCCLCHECKGOTO(ncclTransportP2pSetup(comm, &collNetGraph, 0, &highestTransportType0), ret, collnet_cleanup);
    for (int c=0; c<comm->nChannels; c++) {
      struct ncclChannel* channelSend = comm->channels+c;
      NCCLCHECKGOTO(ncclTransportP2pConnect(comm, c, NCCL_MAX_DIRECT_ARITY, channelSend->collnetDirect.down, NCCL_MAX_DIRECT_ARITY, channelSend->collnetDirect.up, 1), ret, collnet_cleanup);
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

  do { // Setup p2p structures in comm->tasks
    struct ncclTasks* tasks = &comm->tasks;
    int nRanks = comm->nRanks;
    int node = comm->node;
    int nNodes = comm->nNodes;
    struct ncclNodeRanks *nodeRanks = comm->nodeRanks;
    int localRank = comm->localRank;
    tasks->peers = ncclMemoryStackAlloc<ncclTasks::Peer>(&comm->memPermanent, nRanks);
    tasks->p2pSendOrder = ncclMemoryStackAlloc<int>(&comm->memPermanent, nRanks);
    tasks->p2pRecvOrder = ncclMemoryStackAlloc<int>(&comm->memPermanent, nRanks);
    int s=0, r=0;
    // schedule delta 0, +1, -1, +2, -2, ...
    // also make sure we don't do 0 twice, nor +n/2 and -n/2 if n is even.
    for (int d=0; d <= nNodes/4; d++) {
      int deltas[4] = { d, (nNodes-d)%nNodes, nNodes/2-d, (nNodes-(nNodes/2-d))%nNodes };
      int index = 0;
      int delta = deltas[index];
    sched_delta:
      int recvNode = (node+nNodes-delta)%nNodes;
      int sendNode = (node+delta)%nNodes;
      int steps = comm->maxLocalRanks;
      for (int step=0; step < steps; step++) {
        int recvIndex = (localRank-step+steps)%steps;
        if (recvIndex < nodeRanks[recvNode].localRanks) {
          tasks->p2pRecvOrder[r] = nodeRanks[recvNode].localRankToRank[recvIndex];
          r++;
        }
        int sendIndex = (localRank+step)%steps;
        if (sendIndex < nodeRanks[sendNode].localRanks) {
          tasks->p2pSendOrder[s] = nodeRanks[sendNode].localRankToRank[sendIndex];
          s++;
        }
      }
      index++;
      if (index == 1 && deltas[1] == deltas[0]) index++;
      if (index == 2 && deltas[2] == deltas[0]) index++;
      if (index == 3 && deltas[3] == deltas[2]) index++;
      if (index == 3 && deltas[3] == deltas[1]) index++;
      if (index < 4) {
        delta = deltas[index];
        goto sched_delta;
      }
    }
    assert(s == nRanks && r == nRanks);
  } while (0);

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
          comm->connectSend[peer] |= (1UL<<channelId);
        }
      }
      for (int c=0; c<comm->p2pnChannelsPerPeer; c++) {
        NCCLCHECK(ncclChannelCompute(comm, peer, c, ncclFuncRecv, &channelId));
        if (comm->channels[channelId].peers[peer].recv[1].connected == 0) {
          comm->connectRecv[peer] |= (1UL<<channelId);
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
  if (ncclPxnDisable(comm) == 0) {
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
        if (intraProcRank0 == rank && rank != i) {
          comm->peerInfo[i].comm->intraNext = comm->intraNext;
          comm->intraNext = comm->peerInfo[i].comm;
        }
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
    struct ncclComm* comm0 = comm->peerInfo[intraProcRank0].comm;
    assert(intraProcRank==0 ? comm==comm0 : true);
    comm->intraComm0 = comm0;
    comm->intraRefs = intraProcRank==0 ? intraProcRanks : 0;
    comm->intraRank = intraProcRank;
    comm->intraRanks = intraProcRanks;
    comm->intraBarrierPhase = 0;
    comm->intraBarrierCounter = 0;
    comm->intraBarrierGate = 0;
  } while(0);

  if (comm->intraRank == 0) { // Load ncclParamLaunchMode
    char* str = getenv("NCCL_LAUNCH_MODE");
    enum ncclLaunchMode mode, modeOld;
    if (str && strcasecmp(str, "GROUP") == 0) {
      mode = ncclLaunchModeGroup;
    } else {
      mode = ncclLaunchModeParallel;
    }
    // In theory we could be racing with other communicators not associated with
    // this one if the user is connecting to multiple ncclUniqueId's concurrently.
    modeOld = __atomic_exchange_n(&ncclParamLaunchMode, mode, __ATOMIC_RELAXED);
    if (modeOld == ncclLaunchModeInvalid && str && str[0]!='\0') {
      INFO(NCCL_ENV, "NCCL_LAUNCH_MODE set by environment to %s", mode == ncclLaunchModeParallel ? "PARALLEL" : "GROUP");
    }
  }

  NCCLCHECKGOTO(devCommSetup(comm), ret, affinity_restore);

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

struct ncclCommInitRankAsyncJob {
  struct ncclAsyncJob base;
  ncclComm_t* newcomm;
  int nranks, myrank;
  ncclUniqueId commId;
  int cudaDev;
  int virtualId;
};

struct ncclCommFinalizeAsyncJob {
  struct ncclAsyncJob base;
  ncclComm_t comm;
};

static ncclResult_t ncclCommInitRankFunc(struct ncclAsyncJob* job_) {
  struct ncclCommInitRankAsyncJob* job = (struct ncclCommInitRankAsyncJob*)job_;
  ncclComm_t* newcomm = job->newcomm;
  ncclComm_t comm = *newcomm;
  int nranks = job->nranks;
  ncclUniqueId commId = job->commId; // C++ struct assignment
  int myrank = job->myrank;
  int cudaDev = job->cudaDev;
  int virtualId = job->virtualId;
  ncclResult_t res = ncclSuccess;

  CUDACHECK(cudaSetDevice(cudaDev));
  // Set the maximum kernel stack size of all kernels to avoid
  // a CUDA memory reconfig on load (c.f. NVSHMEM issue)
  if (maxLocalSizeBytes > 0 && ncclParamSetStackSize() == 1) {
    TRACE(NCCL_INIT, "Setting cudaLimitStackSize to %zi", maxLocalSizeBytes);
    //CUDACHECKIGNORE(hipDeviceSetLimit(hipLimitStackSize, maxLocalSizeBytes));
  }
  NCCLCHECKGOTO(commAlloc(newcomm, nranks, myrank, virtualId), res, cleanup);
  NCCLCHECKGOTO(initTransportsRank(*newcomm, &commId), res, cleanup);

  // update communicator state
  comm->initState = ncclSuccess;

  INFO(NCCL_INIT,"comm %p rank %d nranks %d cudaDev %d busId %lx localSize %ld used %ld bytes - Init COMPLETE", *newcomm, myrank, nranks, (*newcomm)->cudaDev, (*newcomm)->busId, ncclKernLocalSize(ncclGetKernelIndex(*newcomm)), allocTracker[(*newcomm)->cudaDev].totalAllocSize);
  TRACE_CALL("ncclCommInitRank(%p,%d,0x%llx,%d,%d)", *newcomm, nranks, (unsigned long long)hashUniqueId(commId), myrank, (*newcomm)->cudaDev);
  return ncclSuccess;
cleanup:
  comm->initState = res;
  return res;
}

static ncclResult_t parseCommConfig(ncclComm_t comm, ncclConfig_t *config) {
  ncclResult_t ret = ncclSuccess;

  /* first set configuration */
  if (config) {
    comm->blocking = config->blocking;
  } else {
    /* default setting of communicator */
    comm->blocking = 1;
  }

  return ret;
}

static void ncclCommInitRankUndo(struct ncclAsyncJob* job_) {
  struct ncclCommInitRankAsyncJob* job = (struct ncclCommInitRankAsyncJob*)job_;
  ncclCommDestroy(*job->newcomm);
  *job->newcomm = nullptr;
}

static ncclResult_t ncclCommInitRankDev(ncclComm_t* newcomm, int nranks, ncclUniqueId commId, int myrank, int cudaDev, ncclConfig_t *config, int virtualId) {
  ncclResult_t res;
  ncclComm_t comm = NULL;
  struct ncclCommInitRankAsyncJob *job = NULL;
  char* env = getenv("NCCL_COMM_ID");
  if (env && myrank == 0) {
    INFO(NCCL_ENV, "NCCL_COMM_ID set by environment to %s", env);
    NCCLCHECKGOTO(bootstrapCreateRoot(&commId, true), res, fail);
  }

  NCCLCHECKGOTO(ncclInit(), res, fail);
  if (myrank == 0) showVersion();

  memset(allocTracker+cudaDev, 0, sizeof(struct allocationTracker));
  // Make sure the CUDA runtime is initialized.
  CUDACHECKGOTO(cudaFree(NULL), res, fail);

  NCCLCHECKGOTO(PtrCheck(newcomm, "CommInitRank", "newcomm"), res, fail);
  if (nranks < 1 || myrank < 0 || myrank >= nranks) {
    WARN("Invalid rank requested : %d/%d", myrank, nranks);
    res = ncclInvalidArgument;
    goto fail;
  }

  NCCLCHECKGOTO(ncclCalloc(&comm, 1), res, fail);
  NCCLCHECKGOTO(ncclCudaHostCalloc((uint32_t**)&comm->abortFlag, 1), res, fail);
  // set up comm state and abortFlag only
  *comm->abortFlag = 0;
  NCCLCHECKGOTO(parseCommConfig(comm, config), res, fail);
  /* start with ncclInternalError and will be changed to ncclSuccess if init succeeds. */
  comm->initState = ncclInternalError;
  *newcomm = comm;

  NCCLCHECKGOTO(ncclCalloc(&job, 1), res, fail);
  job->newcomm = newcomm;
  job->nranks = nranks;
  job->commId = commId; // C++ struct assignment
  job->myrank = myrank;
  job->cudaDev = cudaDev;
  job->virtualId = virtualId;
  NCCLCHECKGOTO(ncclAsyncLaunch(&job->base, ncclCommInitRankFunc, NULL, free, comm), res, fail);

exit:
  return ncclGroupErrCheck(res);
fail:
  goto exit;
}

NCCL_API(ncclResult_t, ncclCommInitRank, ncclComm_t* newcomm, int nranks, ncclUniqueId commId, int myrank);
ncclResult_t ncclCommInitRank(ncclComm_t* newcomm, int nranks, ncclUniqueId commId, int myrank) {
  NVTX3_FUNC_RANGE_IN(nccl_domain);

  // Load the CUDA driver and dlsym hooks (can fail on old drivers)
  if (ncclParamDmaBufEnable()) rocmLibraryInit();

  int cudaDev;
  CUDACHECK(cudaGetDevice(&cudaDev));
  NCCLCHECK(ncclCommInitRankDev(newcomm, nranks, commId, myrank, cudaDev, NULL, -1));
  return ncclSuccess;
}

NCCL_API(ncclResult_t, ncclCommInitRankMulti, ncclComm_t* newcomm, int nranks, ncclUniqueId commId, int myrank, int virtualId);
ncclResult_t ncclCommInitRankMulti(ncclComm_t* newcomm, int nranks, ncclUniqueId commId, int myrank, int virtualId) {
  NVTX3_FUNC_RANGE_IN(nccl_domain);
  int cudaDev;
  CUDACHECK(hipGetDevice(&cudaDev));
  NCCLCHECK(ncclCommInitRankDev(newcomm, nranks, commId, myrank, cudaDev, NULL, virtualId));
  return ncclSuccess;
}


NCCL_API(ncclResult_t, ncclCommInitAll, ncclComm_t* comms, int ndev, const int* devlist);
ncclResult_t ncclCommInitAll(ncclComm_t* comms, int ndev, const int* devlist) {
  NVTX3_FUNC_RANGE_IN(nccl_domain);
  ncclResult_t ret = ncclSuccess;
  int totalnDev;
  int *gpuFlags = NULL;
  // Load the CUDA driver and dlsym hooks (can fail on old drivers)
  if (ncclParamDmaBufEnable()) (void) rocmLibraryInit();

  NCCLCHECKGOTO(PtrCheck(comms, "CommInitAll", "comms"), ret, fail);
  if (ndev < 0) {
    WARN("Invalid device count requested : %d", ndev);
    ret = ncclInvalidArgument;
    goto fail;
  }

  CUDACHECKGOTO(cudaGetDeviceCount(&totalnDev), ret, fail);
  if (devlist) {
    NCCLCHECKGOTO(ncclCalloc(&gpuFlags, totalnDev), ret, fail);
    for (int i = 0; i < ndev; ++i) {
      /* invalid device check. */
      if (devlist[i] < 0 || devlist[i] >= totalnDev) {
        ret = ncclUnhandledCudaError;
        goto fail;
      }

      /* duplicate device check. */
      if (gpuFlags[devlist[i]] != 0) {
        ret = ncclInvalidUsage;
        goto fail;
      }

      gpuFlags[devlist[i]] = 1;
    }
    free(gpuFlags);
    gpuFlags = nullptr;
  }

  ncclUniqueId uniqueId;
  NCCLCHECKGOTO(ncclGetUniqueId(&uniqueId), ret, fail);
  NCCLCHECKGOTO(ncclGroupStart(), ret, fail);
  for (int i=0; i<ndev; i++) {
    // Ignore return codes .. we need to call ncclGroupEnd to clean up anyway
    ncclCommInitRankDev(comms+i, ndev, uniqueId, i, devlist ? devlist[i] : i, NULL, -1);
  }
  NCCLCHECKGOTO(ncclGroupEnd(), ret, fail);

fail:
  free(gpuFlags);
  return ret;
}

ncclResult_t ncclCommSetAsyncError(ncclComm_t comm, ncclResult_t nextState) {
  if (nextState < 0 || nextState >= ncclNumResults || comm == NULL) {
    WARN("ncclCommSetAsyncError: error comm %p sets state %d", comm, nextState);
    return ncclInvalidArgument;
  }

  __atomic_store_n(&comm->asyncResult, nextState, __ATOMIC_RELEASE);
  return ncclSuccess;
}

NCCL_API(ncclResult_t, ncclCommInitRankConfig, ncclComm_t* comm, int nranks, ncclUniqueId commId, int myrank, ncclConfig_t *config);
ncclResult_t ncclCommInitRankConfig(ncclComm_t *newcomm, int nranks, ncclUniqueId commId, int myrank, ncclConfig_t *config) {
  NVTX3_FUNC_RANGE_IN(nccl_domain);
  int cudaDev;
  ncclResult_t ret = ncclSuccess;
  ncclConfig_t internalConfig = NCCL_CONFIG_INITIALIZER;
  ncclConfig_t *internalConfigPtr;
  size_t realSize;
  int blockingEnv;

  NCCLCHECK(ncclGroupStartInternal());
  internalConfigPtr = &internalConfig;
  if (config) {
    memcpy((void*)&realSize, (void*)config, sizeof(size_t));
    realSize = realSize > sizeof(ncclConfig_t) ? sizeof(ncclConfig_t) : realSize;
    memcpy((void*)internalConfigPtr, (void*)config, realSize);
    if (internalConfigPtr->magic != 0xcafebeef) {
      WARN("ncclConfig_t argument not initialized via NCCL_CONFIG_INITIALIZER");
      ret = ncclInvalidArgument;
      goto exit;
    }
  }

  /* check input config attributes */
  if (internalConfigPtr->blocking != 0 && internalConfigPtr->blocking != 1) {
    WARN("Invalid config blocking attribute value %d", internalConfigPtr->blocking);
    ret = ncclInvalidArgument;
    goto exit;
  }

  /* overwrite configuration from env variable. */
  blockingEnv = ncclParamCommBlocking();
  if (blockingEnv != 0 && blockingEnv != 1) {
    WARN("Invalid NCCL_COMM_BLOCKING value %d", blockingEnv);
  }
  if (blockingEnv == 1) internalConfigPtr->blocking = blockingEnv;

  if (ncclParamDmaBufEnable()) (void) rocmLibraryInit();
  CUDACHECKGOTO(cudaGetDevice(&cudaDev), ret, exit);
  NCCLCHECKGOTO(ncclCommInitRankDev(newcomm, nranks, commId, myrank, cudaDev, internalConfigPtr, -1), ret, fail);

exit:
  ncclGroupErrCheck(ret);
  NCCLCHECK(ncclGroupEndInternal());
  if (newcomm && *newcomm && !(*newcomm)->blocking) (void) ncclCommGetAsyncError(*newcomm, &ret);
  return ret;
fail:
  if (newcomm && *newcomm && !(*newcomm)->blocking) (void) ncclCommSetAsyncError(*newcomm, ret);
  goto exit;
}

static ncclResult_t commDestroySync(struct ncclAsyncJob* job_) {
  struct ncclCommFinalizeAsyncJob* job = (struct ncclCommFinalizeAsyncJob*) job_;
  ncclComm_t comm = job->comm;
  int savedDevice;
#ifdef ENABLE_TRACE
  int rank = comm->rank;
#endif
  CUDACHECK(cudaGetDevice(&savedDevice));
  int commDevice = comm->cudaDev;
  ncclResult_t ret;

  CUDACHECKGOTO(cudaGetDevice(&savedDevice), ret, fail);
  if (savedDevice != commDevice) {
    CUDACHECKGOTO(cudaSetDevice(commDevice), ret, fail);
  }

  TRACE(NCCL_INIT, "Destroying comm %p rank %d abortFlag %d asyncResult %d", comm, comm->rank, *comm->abortFlag, comm->asyncResult);

  if (comm->initState == ncclSuccess) {
    NCCLCHECKGOTO(ncclStrongStreamSynchronize(&comm->hostStream), ret, fail);
    NCCLCHECKGOTO(ncclStrongStreamSynchronize(&comm->deviceStream), ret, fail);
  }
  NCCLCHECKGOTO(ncclCommPollCallbacks(comm, false), ret, fail);
  // And keep polling until all graphs referencing us die.
  while (comm->persistentRefs != 0) {
    NCCLCHECKGOTO(ncclCommPollCallbacks(comm, /*waitSome=*/true), ret, fail);
  }

  if (savedDevice != commDevice) {
    CUDACHECKGOTO(cudaSetDevice(savedDevice), ret, fail);
  }

exit:
  return ret;
fail:
  goto exit;
}

static ncclResult_t commCleanup(ncclComm_t comm) {
  int savedDevice;
  int commDevice = comm->cudaDev;

  CUDACHECK(cudaGetDevice(&savedDevice));
  if (savedDevice != commDevice) {
    CUDACHECK(cudaSetDevice(commDevice));
  }

  NCCLCHECK(commFree(comm));

  if (savedDevice != commDevice) {
    CUDACHECK(cudaSetDevice(savedDevice));
  }

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

  if (mscclEnabled()) {
    NCCLCHECK(mscclTeardown());
  }

  return ncclSuccess;
}

static ncclResult_t commFinalize(ncclComm_t comm, bool userCalled) {
  ncclResult_t ret = ncclSuccess;
  struct ncclCommFinalizeAsyncJob *job = NULL;

  comm->finalizeCalled = true;
  /* launch async thread to finalize comm. */
  NCCLCHECKGOTO(ncclCalloc(&job, 1), ret, fail);
  job->comm = comm;

  if (userCalled) {
    NCCLCHECKGOTO(ncclAsyncLaunch(&job->base, commDestroySync, NULL, free, comm), ret, fail);
  } else {
    NCCLCHECKGOTO(commDestroySync(&job->base), ret, fail);
    free(job);
  }

exit:
  return ncclGroupErrCheck(ret);
fail:
  if (job) free(job);
  goto exit;
}

NCCL_API(ncclResult_t, ncclCommFinalize, ncclComm_t comm);
ncclResult_t ncclCommFinalize(ncclComm_t comm) {
  NVTX3_FUNC_RANGE_IN(nccl_domain);
  ncclResult_t ret = ncclSuccess;

  NCCLCHECK(ncclGroupStartInternal());
  if (comm == NULL) goto exit;

  /* wait comm ready before finalize. */
  NCCLCHECKGOTO(ncclCommEnsureReady(comm), ret, fail);

  /* prevent double finalize. */
  if (comm->finalizeCalled) {
    ret = ncclInvalidArgument;
    goto fail;
  }

  /* finalize comm. */
  ret = commFinalize(comm, true);

exit:
  ncclGroupErrCheck(ret);
  NCCLCHECK(ncclGroupEndInternal());
  if (comm && !comm->blocking) { NCCLCHECK(ncclCommGetAsyncError(comm, &ret)) };
  return ret;
fail:
  if (comm && !comm->blocking) (void) ncclCommSetAsyncError(comm, ret);
  goto exit;
}

static ncclResult_t commReclaim(ncclComm_t comm) {
  ncclResult_t ret = ncclSuccess;
  ncclResult_t state;
  int curRank; /* Debug info */

  NCCLCHECKGOTO(ncclCommGetAsyncError(comm, &state), ret, fail);
  TRACE(NCCL_INIT, "commReclaim: reclaim comm %p rank %d state %d", comm, comm->rank, state);
  if (state == ncclSuccess && *comm->abortFlag == 0 && comm->finalizeCalled == false) {
    /* user does not call ncclCommFinalize and this is a normal comm destroy. ncclCommDestroy
     * should be nonblocking until last call of ncclCommDestroy. */
    NCCLCHECKGOTO(commFinalize(comm, false), ret, fail);
  }

  if (comm->initState != ncclSuccess) {
    /* if init errors happen, no finalize thread should have been launched. Main thread can reclaim
     * everything since no NCCL kernel was issued. */
    struct ncclCommFinalizeAsyncJob job;

    job.comm = comm;
    curRank = comm->rank;
    /* comm aborts, commDestroySync should not be blocked. */
    if ((ret = commDestroySync((struct ncclAsyncJob*) &job)) != ncclSuccess) {
      WARN("commReclaim: comm %p (rank = %d) in abort, error %d", comm, curRank, ret);
    }

    if ((ret = commCleanup(comm)) != ncclSuccess) {
      WARN("commReclaim: cleanup comm %p rank %d failed in destroy/abort, error %d", comm, curRank, ret);
    }
  } else {
    int curRankCnt;
    int intraRanks = comm->intraRanks;
    ncclComm_t intracomm0 = comm->intraComm0;
    int *finalizeRankCnt = &intracomm0->finalizeRankCnt;

    assert(intracomm0 != NULL && finalizeRankCnt != NULL);
    curRankCnt = __atomic_add_fetch(finalizeRankCnt, 1, __ATOMIC_ACQ_REL);
    if (curRankCnt == intraRanks) {
      ncclComm_t curIntraComm;
      ncclComm_t nextIntraComm = intracomm0;

      while (nextIntraComm) {
        curIntraComm = nextIntraComm;
        curRank = curIntraComm->rank;
        nextIntraComm = nextIntraComm->intraNext;

        if (comm->finalizeCalled == false) {
          struct ncclCommFinalizeAsyncJob job;
          job.comm = curIntraComm;
          /* every comm aborts, commDestroySync should not be blocked. */
          if ((ret = commDestroySync((struct ncclAsyncJob*) &job)) != ncclSuccess)
            WARN("commReclaim: comm %p (rank = %d) in abort, error %d", curIntraComm, curRank, ret);
        }

        if ((ret = commCleanup(curIntraComm)) != ncclSuccess) {
          WARN("commReclaim: cleanup comm %p rank %d failed in destroy/abort, error %d", curIntraComm, curRank, ret);
        }
      }
    }
  }

exit:
  return ret;
fail:
  goto exit;
}

NCCL_API(ncclResult_t, ncclCommDestroy, ncclComm_t comm);
ncclResult_t ncclCommDestroy(ncclComm_t comm) {
  NVTX3_FUNC_RANGE_IN(nccl_domain);
  if (comm == NULL)
    return ncclSuccess;

  int rank = comm->rank, nranks = comm->nRanks, cudaDev = comm->cudaDev;
  int64_t busId = comm->busId;
  TRACE(NCCL_INIT, "comm %p rank %d nRanks %d cudaDev %d busId %lx", comm, rank, nranks, cudaDev, busId);
  // Try and prevent a double free of the comm struct (user error)
  if (comm->rank == -1 || comm->nRanks == -1 || comm->cudaDev == -1 || comm->busId == -1) {
    WARN("comm %p has already been destroyed", comm);
    return ncclInvalidArgument;
  }

  /* init thread must be joined before we destroy the comm. */
  NCCLCHECK(ncclCommEnsureReady(comm));

  NCCLCHECK(commReclaim(comm));
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
  /* init thread must be joined before we destroy the comm,
   * and we should ignore the init error here. */
  ncclCommEnsureReady(comm);

  (void) commReclaim(comm);
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
    case ncclRemoteError            : return "remote process exited or there was a network error";
    case ncclInProgress             : return "NCCL operation in progress";
    default                         : return "unknown result code";
  }
}

/* Returns a human-readable message of the last error that occurred.
 * comm is currently unused and can be set to NULL
 */
NCCL_API(const char*, ncclGetLastError, const ncclComm_t comm);
const char* ncclGetLastError(ncclComm_t comm) {
  return ncclLastError;
}

NCCL_API(ncclResult_t, ncclCommGetAsyncError, ncclComm_t comm, ncclResult_t *asyncError);
ncclResult_t ncclCommGetAsyncError(ncclComm_t comm, ncclResult_t *asyncError) {
  NCCLCHECK(PtrCheck(comm, "ncclGetAsyncError", "comm"));
  NCCLCHECK(PtrCheck(asyncError, "ncclGetAsyncError", "asyncError"));

  *asyncError = __atomic_load_n(&comm->asyncResult, __ATOMIC_ACQUIRE);
  return ncclSuccess;
}

NCCL_API(ncclResult_t, ncclCommCount, const ncclComm_t comm, int* count);
ncclResult_t ncclCommCount(const ncclComm_t comm, int* count) {
  NVTX3_FUNC_RANGE_IN(nccl_domain);

  NCCLCHECK(PtrCheck(comm, "CommCount", "comm"));
  NCCLCHECK(PtrCheck(count, "CommCount", "count"));

  /* init thread must be joined before we access the attributes of comm. */
  NCCLCHECK(ncclCommEnsureReady(comm));

  *count = comm->nRanks;
  return ncclSuccess;
}

NCCL_API(ncclResult_t, ncclCommCuDevice, const ncclComm_t comm, int* devid);
ncclResult_t ncclCommCuDevice(const ncclComm_t comm, int* devid) {
  NVTX3_FUNC_RANGE_IN(nccl_domain);

  NCCLCHECK(PtrCheck(comm, "CommCuDevice", "comm"));
  NCCLCHECK(PtrCheck(devid, "CommCuDevice", "devid"));

  NCCLCHECK(ncclCommEnsureReady(comm));

  *devid = comm->cudaDev;
  return ncclSuccess;
}

NCCL_API(ncclResult_t, ncclCommUserRank, const ncclComm_t comm, int* rank);
ncclResult_t ncclCommUserRank(const ncclComm_t comm, int* rank) {
  NVTX3_FUNC_RANGE_IN(nccl_domain);

  NCCLCHECK(PtrCheck(comm, "CommUserRank", "comm"));
  NCCLCHECK(PtrCheck(rank, "CommUserRank", "rank"));

  NCCLCHECK(ncclCommEnsureReady(comm));

  *rank = comm->rank;
  return ncclSuccess;
}
