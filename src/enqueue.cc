/*************************************************************************
 * Copyright (c) 2017-2021, NVIDIA CORPORATION. All rights reserved.
 * Modifications Copyright (c) 2019-2021 Advanced Micro Devices, Inc. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

#include "enqueue.h"
#include "argcheck.h"
#include "coll_net.h"
#include "graph/topo.h"
#include <hip/hip_runtime.h>
#include <hip/hip_ext.h>
#include "gdrwrap.h"

// Only generate inline kernels for LL
#define NCCL_FUNC5(func, algo, redop, dtype) \
  NCCL_KERN_NAME(func, algo, LL, redop, dtype), \
  NCCL_KERN_NAME(func, algo, LL, redop, dtype), \
  NCCL_KERN_NAME(func, algo, LL, redop, dtype)

#define NCCL_FUNC4(func, redop, type) \
  NCCL_FUNC5(func, TREE,    redop, type), \
  NCCL_FUNC5(func, RING,    redop, type), \
  NCCL_FUNC5(func, COLLNET, redop, type)

// Must be consistent with ncclDataType_t
#define NCCL_FUNCS3A(func, redop) \
  NCCL_FUNC4(func, redop, int8_t), \
  NCCL_FUNC4(func, redop, uint8_t), \
  NCCL_FUNC4(func, redop, int32_t), \
  NCCL_FUNC4(func, redop, uint32_t), \
  NCCL_FUNC4(func, redop, int64_t), \
  NCCL_FUNC4(func, redop, uint64_t), \
  NCCL_FUNC4(func, redop, half), \
  NCCL_FUNC4(func, redop, float), \
  NCCL_FUNC4(func, redop, double), \
  NCCL_FUNC4(func, redop, rccl_bfloat16)
#define NCCL_FUNCS3B(func, redop) \
  NCCL_FUNC4(func, redop, int8_t), \
  NCCL_FUNC4(func, redop, int8_t), \
  NCCL_FUNC4(func, redop, int8_t), \
  NCCL_FUNC4(func, redop, int8_t), \
  NCCL_FUNC4(func, redop, int8_t), \
  NCCL_FUNC4(func, redop, int8_t), \
  NCCL_FUNC4(func, redop, int8_t), \
  NCCL_FUNC4(func, redop, int8_t), \
  NCCL_FUNC4(func, redop, int8_t), \
  NCCL_FUNC4(func, redop, int8_t)

// Must be consistent with ncclRedOp_t -- but we only generate kernel for sums.
#define NCCL_FUNCS2A(func) \
  NCCL_FUNCS3A(func, Sum), \
  NCCL_FUNCS3A(func, Sum), \
  NCCL_FUNCS3A(func, Sum), \
  NCCL_FUNCS3A(func, Sum)
#define NCCL_FUNCS2B(func) \
  NCCL_FUNCS3B(func, Sum), \
  NCCL_FUNCS3B(func, Sum), \
  NCCL_FUNCS3B(func, Sum), \
  NCCL_FUNCS3B(func, Sum)

typedef void(*ncclKern_t)(struct ncclWorkElem first);
// Must be consistent with the ncclFuncSet enum
static ncclKern_t const ncclKerns[1] = {
  NCCL_KERN_NAME(SendRecv, RING, SIMPLE, Sum, int8_t),
};

// Determine the maximum kernel stack size of all CUDA kernels
size_t ncclKernMaxLocalSize() {
  ncclResult_t res = ncclSuccess;
  int numNcclKerns = sizeof(ncclKerns)/sizeof(ncclKerns[0]);
  hipFuncAttributes attr = {0};
  size_t max = 0;
  for (int i = 0; i < numNcclKerns; i++) {
    CUDACHECKGOTO(hipFuncGetAttributes(&attr, (const void*)(ncclKerns[i])), res, error);
    if (attr.localSizeBytes > max) max = attr.localSizeBytes;
  }

error:
  return (res != ncclSuccess) ? 0 : max;
}

/*****************************************************************************/
/*       Launch system : synchronization and CUDA kernel launch              */
/*****************************************************************************/

ncclResult_t ncclLaunchCooperativeKernelMultiDevice(hipLaunchParams *paramsList, int* cudaDevs, int numDevices, int cgMode) {
  if (cgMode & 0x01) {
    CUDACHECK(hipExtLaunchMultiKernelMultiDevice(paramsList, numDevices,
      // These flags are to reduce the latency of using this API
      hipCooperativeLaunchMultiDeviceNoPreSync|hipCooperativeLaunchMultiDeviceNoPostSync));
    return ncclSuccess;
  }
  int savedDev;
  CUDACHECK(hipGetDevice(&savedDev));
  for (int i = 0; i < numDevices; i++) {
    hipLaunchParams* params = paramsList+i;
    CUDACHECK(hipSetDevice(cudaDevs[i]));
    hipLaunchKernelGGL(((void (*)(struct ncclWorkElem))params->func), params->gridDim, params->blockDim, params->sharedMem, params->stream, **((struct ncclWorkElem**)params->args));
  }
  CUDACHECK(hipSetDevice(savedDev));
  return ncclSuccess;
}

static ncclResult_t getNextOp(struct ncclChannel* channel, struct ncclWork** work, struct ncclWorkElem* base) {
  if (channel->workCount == NCCL_MAX_OPS) {
    WARN("Too many aggregated operations on channel %d (%d max)", channel->id, NCCL_MAX_OPS);
    return ncclInvalidUsage;
  }
  int opIndex = channel->workFifoTail%NCCL_MAX_OPS;
  struct ncclWork* w = channel->workFifo+opIndex;
  struct ncclWorkElem* e = w->elems;
  volatile uint8_t* activePtr = (volatile uint8_t*)&e->active;
  while (LOAD(activePtr) != 0) sched_yield();
  memset(w, 0, sizeof(struct ncclWork));
  // Initialize with work elem if provided
  if (base) memcpy(e, base, sizeof(struct ncclWorkElem));
  STORE(&e->active, 1);
  e->index = opIndex;
  channel->workFifoTail++;
  channel->workCount++;
  if (work) *work = w;
  return ncclSuccess;
}

static ncclResult_t setupLaunch(struct ncclQueueInfo* eqInfo, int usingCudaGraph) {
  ncclComm_t comm = eqInfo->comm;
  hipLaunchParams* params = comm->myParams;

  // Only launch blocks where we have work to do.
  // This is not supported when we are in cudaGraph mode.
  // Because in cudaGraph mode the launch param needs to be determined
  // at capture time instead of launch time.
  if (!usingCudaGraph) {
    int nChannels = std::max(comm->nChannels, comm->p2pnChannels);
    for (int c=0; c<nChannels; c++) {
      if (comm->channels[c].workCount) params->gridDim.x = c+1;
    }
    eqInfo->maxChannels = params->gridDim.x;
  }

  // Set active = 2 for the last operation and add a no-op on empty channels (p2p case).
  for (int c=0; c<eqInfo->maxChannels; c++) {
    struct ncclChannel* channel = comm->channels+c;
    if (channel->workCount == 0) {
      struct ncclWork* w;
      NCCLCHECK(getNextOp(channel, &w, NULL));
      struct ncclWorkElem* e = w->elems;
      e->comm = comm->devComm;
      e->funcIndex = FUNC_INDEX_P2P;
      e->p2p.nThreads = 0;
    }
    STORE(&channel->workFifo[(channel->workFifoTail-1)%NCCL_MAX_OPS].elems[0].active, 2);

    if (c == 0) {
      // Find the first operation, choose the kernel accordingly and pass it as the first argument.
      // Note that changing cuda launch argument after capture is not supported by cudaGraph
      struct ncclWork* work = channel->workFifo+((channel->workFifoTail-channel->workCount)%NCCL_MAX_OPS);
      struct ncclWorkElem* elem = work->elems;
      if (!usingCudaGraph) {
        params->func = (void *)ncclKerns[0];
        memcpy(&comm->args, elem, sizeof(struct ncclWorkElem));
      }
      // As we inline the first coll directly, we can free it immediately.
      if (elem->funcIndex != FUNC_INDEX_P2P) elem->active = 0;
    }

    if (channel->gdrMemDesc) {
      // GDRCOPY support
      uint64_t first = (channel->workFifoTail-channel->workCount)%NCCL_MAX_OPS;
      uint64_t nelems = channel->workCount;
      TRACE(NCCL_INIT, "GDRCOPY : copy workFifo %p to %p first %ld nelems %zi",
            channel->workFifo, channel->workFifoGdr, first, nelems);

      for (int i = 0; i < nelems; i++) {
        int elem = (first+i) % NCCL_MAX_OPS;
        // Copy Host workFifo to CUDA workFifo via the GDRCOPY mapping
        NCCLCHECK(ncclGdrCudaCopy(channel->gdrMemDesc, channel->workFifoGdr+elem, channel->workFifo+elem, 1));
      }
    }
  }

  return ncclSuccess;
}

ncclResult_t ncclCpuBarrierIn(struct ncclComm* comm, int* isLast) {
  volatile int* ptr = (volatile int*)(comm->intraBarrier+comm->intraPhase);
  int val = LOAD(ptr);
  bool done = false;
  while (done == false) {
    if (val >= comm->intraRanks) {
      WARN("Trying to launch too many work elements, max is %d", NCCL_MAX_OPS);
      return ncclInvalidUsage;
    }
    if (val+1 == comm->intraRanks) {
      // Reset the barrier.
      comm->intraBarrier[comm->intraPhase^1] = 0;
      *isLast = 1;
      return ncclSuccess;
    }
    done = __sync_bool_compare_and_swap(ptr, val, val+1);
    val++;
  }
  *isLast = 0;
  return ncclSuccess;
}

ncclResult_t ncclCpuBarrierLast(struct ncclComm* comm) {
  volatile int* ptr = (volatile int*)(comm->intraBarrier+comm->intraPhase);
  int val = LOAD(ptr);
  if (__sync_bool_compare_and_swap(ptr, val, val+1) != true) {
    WARN("Trying to launch too many work elements, max is %d", NCCL_MAX_OPS);
    return ncclInternalError;
  }
  return ncclSuccess;
}

ncclResult_t ncclCpuBarrierOut(struct ncclComm* comm) {
  volatile int* ptr = (volatile int*)(comm->intraBarrier+comm->intraPhase);
  while (LOAD(ptr) < comm->intraRanks) pthread_yield();
  comm->intraPhase ^= 1;
  return ncclSuccess;
}

ncclResult_t ncclLaunchBarrier(struct ncclComm* comm) {
  hipLaunchParams* params = comm->myParams;
  if (params->gridDim.x == 0) return ncclSuccess;

  // Use internal NCCL stream for CGMD/GROUP launch if required or if the user stream is NULL
  if (comm->launchMode == ncclComm::GROUP &&
      (comm->groupCudaStream ||
       comm->userStream == hipStreamDefault/* ||
       comm->userStream == hipStreamLegacy ||
       comm->userStream == hipStreamPerThread*/)) {
    // Enqueue event in user stream
    CUDACHECK(hipEventRecord(comm->intDoneEvent, comm->userStream));
    // Create dependency between user stream and internal NCCL stream
    CUDACHECK(hipStreamWaitEvent(comm->groupStream, comm->intDoneEvent, 0));
    params->stream = comm->groupStream;
  } else {
    if (comm->userStream != params->stream && !comm->usingCudaGraph) {
      // Stream changed from last call, create dependency against last NCCL kernel launch
      CUDACHECK(hipStreamWaitEvent(comm->userStream, comm->doneEvent, 0));
    }
    params->stream = comm->userStream;
  }

  if (comm->launchMode == ncclComm::GROUP) {
    int isLast = 0;
    NCCLCHECK(ncclCpuBarrierIn(comm, &isLast));
    if (isLast) {
      // I'm the last. Launch all operations.
      NCCLCHECK(ncclLaunchCooperativeKernelMultiDevice(comm->intraParams, comm->intraCudaDevs, comm->intraRanks, *comm->intraCGMode));
      NCCLCHECK(ncclCpuBarrierLast(comm));
    }
  }
  return ncclSuccess;
}

ncclResult_t ncclLaunchKernel(ncclComm_t comm) {
  hipLaunchParams *params = comm->myParams;
  if (params->gridDim.x == 0) return ncclSuccess;

  // We can't print the CG mode before the first barrier happened.
  if (comm->rank == 0 && *comm->intraCGMode & 0x10) {
    *comm->intraCGMode ^= 0x10;
    INFO(NCCL_INIT,"Launch mode %s%s%s",
        comm->launchMode == ncclComm::GROUP ? "Group" : "Parallel",
        *comm->intraCGMode ? "/CGMD" : "",
        (comm->launchMode == ncclComm::GROUP && comm->groupCudaStream) ? "/Stream" : "");
  }

  if (comm->launchMode == ncclComm::GROUP) {
    NCCLCHECK(ncclCpuBarrierOut(comm));
  } else {
    CUDACHECK(hipLaunchKernel(params->func, params->gridDim, params->blockDim, params->args, params->sharedMem, params->stream));
  }

  return ncclSuccess;
}

static ncclResult_t ncclLaunchProxy(struct ncclQueueInfo* eqInfo) {
  // Start the network proxies as soon as the kernel has been launched. We can't
  // perform any CUDA call between the two or having a cudaFree between the CUDA
  // launch and the ncclProxyStart call could cause a deadlock.
  // Also, starting the proxies after the CUDA launch seems to be better for
  // performance (latency).
  ncclComm_t comm = eqInfo->comm;
  if (eqInfo->maxChannels == 0) return ncclSuccess;

  for (int r=0; r<eqInfo->maxChannels; r++) {
    struct ncclChannel* channel = comm->channels+r;
    channel->workCount = 0;
  }
  comm->lastChannel = 0;
  NCCLCHECK(ncclProxyStart(comm));
  return ncclSuccess;
}

ncclResult_t ncclRecordEvents(ncclComm_t comm) {
  hipLaunchParams *params = comm->myParams;

  // Enqueue event after NCCL kernel (only in non-graph mode)
  if (!comm->usingCudaGraph) CUDACHECK(hipEventRecord(comm->doneEvent, params->stream));
  // Use internal NCCL stream for CGMD/GROUP launch if required or if the user stream is NULL
  if (comm->launchMode == ncclComm::GROUP &&
      (comm->groupCudaStream ||
       comm->userStream == hipStreamDefault/* ||
       comm->userStream == hipStreamLegacy ||
       comm->userStream == hipStreamPerThread*/)) {
    CUDACHECK(hipEventRecord(comm->intDoneEvent, params->stream));
    // Create dependency between NCCL internal stream and user stream
    CUDACHECK(hipStreamWaitEvent(comm->userStream, comm->intDoneEvent, 0));
  }
  return ncclSuccess;
}

ncclResult_t ncclLaunchReset(ncclComm_t comm) {
  comm->userStreamSet = false;

  // We are finishing capture of the current launch
  // But we need to keep the current enqueue info for CUDA graph
  // Thus we need to creating a new enqueue info for the next run
  if (comm->usingCudaGraph) {
    NCCLCHECK(ncclCalloc(&comm->enqueueInfo, 1));
    comm->enqueueInfo->comm = comm;
  } else {
    // If not in CUDA graph mode, we reuse the same info space
    NCCLCHECK(ncclResetQueueInfo(comm->enqueueInfo));
  }

  hipLaunchParams *params = comm->myParams;
  params->gridDim.x = params->blockDim.x = 0;
  params->func = NULL;

  // Reset launch mode to GROUP if changed
  if (comm->launchMode == ncclComm::GROUP_GRAPH) comm->launchMode = ncclComm::GROUP;
  comm->usingCudaGraph = 0;

  return ncclSuccess;
}

/*****************************************************************************/
/* Enqueueing system : computation of kernel and proxy operations parameters */
/*****************************************************************************/
RCCL_PARAM(SharpThreshold, "SHARP_THRESHOLD", 16384);

static ncclResult_t getAlgoInfo(struct ncclInfo* info) {
  struct ncclComm* comm = info->comm;
  float minTime = 3600000000.0; // Hopefully no operation will take an hour to complete.
  // Find algorithm / protocol.
  info->algorithm = -1;
  info->protocol = -1;
  int nAlgos = NCCL_NUM_ALGORITHMS;

  // Check collNet support
  int collNetTypeSupport = 0;
  if (info->comm->collNetSupport > 0 && info->nBytes < rcclParamSharpThreshold())
    NCCLCHECK(collNetReduceSupport(info->datatype, info->op, &collNetTypeSupport));
  for (int a=0; a<nAlgos; a++) {
    if (a == NCCL_ALGO_COLLNET && collNetTypeSupport != 1) continue;
    for (int p=0; p<NCCL_NUM_PROTOCOLS; p++) {
      float time;
      NCCLCHECK(ncclTopoGetAlgoTime(info, a, p, &time));
      if (time >= 0 && time < minTime) {
        info->algorithm = a;
        info->protocol = p;
        minTime = time;
      }
    }
  }
  if (info->algorithm == -1 || info->protocol == -1) {
    WARN("Error : no algorithm/protocol available");
    return ncclInternalError;
  }
  //if (comm->rank == 0) INFO(NCCL_TUNING, "%ld Bytes -> Algo %d proto %d time %f", info->nBytes, info->algorithm, info->protocol, minTime);
  TRACE(NCCL_COLL, "%ld Bytes -> Algo %d proto %d time %f", info->nBytes, info->algorithm, info->protocol, minTime);

  int nc = (info->nChannels > 0) ? info->nChannels : comm->nChannels;
  int nt = comm->maxThreads[info->algorithm][info->protocol];
  int threadThreshold = comm->threadThresholds[info->algorithm][info->protocol];
  if (info->algorithm == NCCL_ALGO_COLLNET) {
    int ncSwitch = 16;
    bool flag = true;
    while (ncSwitch >= 1 && flag) {
      while ((flag = info->nBytes < nc*nt*info->comm->channels[0].collTree.nHeads*threadThreshold) && nc > ncSwitch) {
        if (nc == ncSwitch+ncSwitch/2) threadThreshold /= 2;
        nc--;
      }
      ncSwitch /= 2;
    }
  } else {
    while (info->nBytes < nc*nt*threadThreshold) {
      if (nc >= 2) nc--;
#if defined(__HIP_PLATFORM_HCC__) || defined(__HCC__) || defined(__HIPCC__)
      // do not reduce threads count on VEGA
#else
      else if ((nt % 128) == 0) nt/=2;
#endif
      else break;
    }
  }
#if defined(__HIP_PLATFORM_HCC__) || defined(__HCC__) || defined(__HIPCC__)
#else
  if (info->protocol == NCCL_PROTO_SIMPLE) {
    nt += WARP_SIZE; // Extra warp for sync
    if (info->algorithm == NCCL_ALGO_TREE) nt += WARP_SIZE;
    if (info->algorithm == NCCL_ALGO_COLLNET) nt += 3*WARP_SIZE;
  }
#endif
  info->nChannels = nc;
  info->nThreads = nt;
  return ncclSuccess;
}

static ncclResult_t getPatternInfo(struct ncclInfo* info) {
  switch (info->coll) {
    case ncclFuncBroadcast:
      info->pattern = info->algorithm == NCCL_ALGO_TREE ? ncclPatternTreeDown : ncclPatternPipelineFrom; break;
    case ncclFuncReduce:
      info->pattern = info->algorithm == NCCL_ALGO_TREE ? ncclPatternTreeUp : ncclPatternPipelineTo; break;
    case ncclFuncReduceScatter:
    case ncclFuncAllGather:
      info->pattern = ncclPatternRing; break;
    case ncclFuncAllReduce:
      info->pattern = info->algorithm == NCCL_ALGO_COLLNET ? ncclPatternCollTreeUpDown : info->algorithm == NCCL_ALGO_TREE ? ncclPatternTreeUpDown : ncclPatternRingTwice; break;
    default:
      WARN("Unknown pattern for collective %d algorithm %d", info->coll, info->algorithm);
      return ncclInternalError;
  }
  return ncclSuccess;
}

static ncclResult_t getLoopInfo(struct ncclInfo* info) {
  switch (info->pattern) {
    case ncclPatternTreeUp:
    case ncclPatternTreeDown:
    case ncclPatternTreeUpDown:
    case ncclPatternPipelineFrom:
    case ncclPatternPipelineTo:
      info->nstepsPerLoop = info-> nchunksPerLoop = 1; break;
    case ncclPatternCollTreeUpDown:
      info->nstepsPerLoop = 1; info->nchunksPerLoop = info->comm->channels[0].collTree.nHeads; break;
    case ncclPatternRing:
      info->nstepsPerLoop = info->comm->nRanks-1; info->nchunksPerLoop = info->comm->nRanks; break;
    case ncclPatternRingTwice:
      info->nstepsPerLoop = 2*(info->comm->nRanks-1); info->nchunksPerLoop = info->comm->nRanks; break;
    default:
      WARN("Unknown pattern %d", info->pattern);
      return ncclInternalError;
  }
  return ncclSuccess;
}

RCCL_PARAM(IntraNetThreshold, "RCCL_INTRANET_THRESHOLD", 8388608);

static ncclResult_t computeColl(struct ncclInfo* info /* input */, struct ncclWorkElem* work, struct ncclProxyArgs* proxyArgs /* output */) {
  work->comm = info->comm->devComm;

  // Set nstepsPerLoop and nchunksPerLoop
  NCCLCHECK(getAlgoInfo(info));
  NCCLCHECK(getPatternInfo(info));
  NCCLCHECK(getLoopInfo(info));

  work->op.opCount = info->comm->collOpCount;
  work->sendbuff = info->sendbuff;
  work->recvbuff = info->recvbuff;
  work->coll.root = info->root;
  work->coll.count = info->count;
  work->coll.nChannels = info->nChannels;
  work->nThreads = info->nThreads;

  work->funcIndex = FUNC_INDEX(info->coll, info->op, info->datatype, info->algorithm, info->protocol);

  work->coll.connIndex = 0;
  proxyArgs->connIndex = 0;
  if (info->protocol == NCCL_PROTO_SIMPLE && info->algorithm == NCCL_ALGO_RING) {
    if (info->comm->useIntraNet && info->nBytes > rcclParamIntraNetThreshold()) {
      work->coll.connIndex = NCCL_CONN_IDX_P2P_NET;
      proxyArgs->connIndex = NCCL_CONN_IDX_P2P_NET;
    }
  }

  { // [RCCL] Check for clique-based kernel support
    if (info->comm->cliqueManager->IsSupported(info->coll,
                                               info->count,
                                               info->datatype,
                                               info->op))
    {
      info->algorithm = NCCL_ALGO_RING;
      info->protocol = NCCL_PROTO_CLIQUE;
      // Determine the number of channels to use for clique-kernel
      NCCLCHECK(info->comm->cliqueManager->GetNumChannelsToUse(info->coll,
                                                               info->count,
                                                               info->datatype,
                                                               info->op,
                                                               info->comm->nChannels,
                                                               &work->clique.nChannels));
      work->clique.count = info->count;
      work->funcIndex = FUNC_INDEX(info->coll, info->op, info->datatype, info->algorithm, info->protocol);

      // Setup pointers to where all the input/output pointers will be
      NCCLCHECK(info->comm->cliqueManager->WaitForPointers(work));
      return ncclSuccess;
    }
  } // [RCCL]

  int stepSize   = info->comm->buffSizes[info->protocol]/NCCL_STEPS;
  int chunkSteps = (info->protocol == NCCL_PROTO_SIMPLE && info->algorithm == NCCL_ALGO_RING) ? info->chunkSteps : 1;
  int sliceSteps = (info->protocol == NCCL_PROTO_SIMPLE && info->algorithm == NCCL_ALGO_RING) ? info->sliceSteps : 1;
  int chunkSize  = stepSize*chunkSteps;

  // Compute lastChunkSize
  if (info->algorithm == NCCL_ALGO_TREE && info->protocol == NCCL_PROTO_SIMPLE) {
    if (info->pattern == ncclPatternTreeUpDown) {
      // Optimize chunkSize / nSteps
      while (info->nBytes / (info->nChannels*chunkSize) < info->comm->channels[0].tree.depth*8 && chunkSize > 131072) chunkSize /= 2;
      while (info->nBytes / (info->nChannels*chunkSize) < info->comm->channels[0].tree.depth*4 && chunkSize > 65536) chunkSize /= 2;
      while (info->nBytes / (info->nChannels*chunkSize) < info->comm->channels[0].tree.depth && chunkSize > 32768) chunkSize /= 2;
    }
    // Use lastChunkSize as chunkSize
    work->coll.lastChunkSize = chunkSize / ncclTypeSize(info->datatype);
  } else if (info->algorithm == NCCL_ALGO_COLLNET && info->protocol == NCCL_PROTO_SIMPLE) {
    // Optimize chunkSize / nSteps
    while (info->nBytes / (info->nChannels*info->comm->channels[0].collTree.nHeads*chunkSize) < info->comm->channels[0].collTree.depth*32 && chunkSize > 262144) chunkSize /= 2;
    while (info->nBytes / (info->nChannels*info->comm->channels[0].collTree.nHeads*chunkSize) < info->comm->channels[0].collTree.depth*16 && chunkSize > 131072) chunkSize /= 2;
    while (info->nBytes / (info->nChannels*info->comm->channels[0].collTree.nHeads*chunkSize) < info->comm->channels[0].collTree.depth*8 && chunkSize > 32768) chunkSize /= 2;
    while (info->nBytes / (info->nChannels*info->comm->channels[0].collTree.nHeads*chunkSize) < info->comm->channels[0].collTree.depth/2 && chunkSize > 16384) chunkSize /= 2;
    // Use lastChunkSize as chunkSize
    work->coll.lastChunkSize = chunkSize / ncclTypeSize(info->datatype);
  } else if (info->protocol == NCCL_PROTO_LL) {
    const ssize_t sliceSize = stepSize*sizeof(uint64_t)/sizeof(union ncclLLFifoLine);
    const ssize_t loopSize = info->nChannels*info->nchunksPerLoop*(ssize_t)sliceSize;
    work->coll.lastChunkSize = DIVUP((info->nBytes-(info->nBytes/loopSize)*loopSize), info->nChannels*info->nchunksPerLoop);
    ALIGN_SIZE(work->coll.lastChunkSize, info->nThreads*sizeof(uint64_t));
    work->coll.lastChunkSize /= ncclTypeSize(info->datatype);
  } else if (info->algorithm == NCCL_ALGO_TREE && info->protocol == NCCL_PROTO_LL128) {
    int nNodes = info->comm->nNodes;
    float ppn = info->comm->nRanks / (float)nNodes;
    float nstepsLL128 = 1+log2i(nNodes) + 0.1*ppn;
    while (info->nBytes / (info->nChannels*chunkSize) < nstepsLL128*64/ppn && chunkSize > 131072) chunkSize /= 2;
    while (info->nBytes / (info->nChannels*chunkSize) < nstepsLL128*16/ppn && chunkSize > 32768) chunkSize /= 2;
    // Use lastChunkSize as chunkSize
    work->coll.lastChunkSize = chunkSize*NCCL_LL128_DATAELEMS/(NCCL_LL128_LINEELEMS*ncclTypeSize(info->datatype));
  }

  // Compute nSteps for proxies
  int chunkEffectiveSize = chunkSize;
  if (info->protocol == NCCL_PROTO_LL) chunkEffectiveSize /= 2;
  if (info->protocol == NCCL_PROTO_LL128) chunkEffectiveSize = (chunkSize / NCCL_LL128_LINEELEMS) * NCCL_LL128_DATAELEMS;
  //if (info->comm->rank == 0) printf("Coll %d, size %ld -> %dx%d, chunkSize %d (algo %d proto%d)\n", info->coll, info->nBytes, info->nChannels, info->nThreads, chunkSize, info->algorithm, info->protocol);
  int nLoops = (int)(DIVUP(info->nBytes, (((size_t)(info->nChannels))*info->nchunksPerLoop*chunkEffectiveSize)));
  proxyArgs->subs[0].nsteps = info->nstepsPerLoop * nLoops * chunkSteps;
  proxyArgs->sliceSteps = sliceSteps;
  proxyArgs->chunkSteps = chunkSteps;
  proxyArgs->chunkSize = chunkSize;
  proxyArgs->protocol = info->protocol;
  proxyArgs->dtype = info->datatype;
  proxyArgs->redOp = (info->algorithm == NCCL_ALGO_COLLNET) ? info->op : ncclNumOps;  // Only set redOp when using CollNet
  proxyArgs->pattern = info->pattern;
  proxyArgs->root = info->root;
  // This is used by P2P to reduce the receive buffer size. We don't use it in collectives
  // because some protocols need to transmit more than the total size, plus they sometimes
  // round up
  proxyArgs->subs[0].recvbytes = stepSize*proxyArgs->sliceSteps;

  TRACE(NCCL_COLL,"opCount %lx slicesteps %d spl %d cpl %d nbytes %zi -> protocol %d nchannels %d nthreads %d, nloops %d nsteps %d chunksize %d comm %p",
      proxyArgs->opCount, sliceSteps, info->nstepsPerLoop, info->nchunksPerLoop, info->nBytes, info->protocol, info->nChannels, info->nThreads,
      nLoops, proxyArgs->subs[0].nsteps, chunkSize, info->comm);
  return ncclSuccess;
}

static ncclResult_t checkSetStream(struct ncclInfo* info) {
 if (info->comm->userStreamSet == false) {
    info->comm->userStream = info->stream;
    info->comm->userStreamSet = true;
  } else if (info->stream != info->comm->userStream) {
    WARN("Error : mixing different streams within a group call is not supported.");
    return ncclInvalidUsage;
  }
  return ncclSuccess;
}

// Compute enqueue element, save it in list
// Compute CUDA launch parameters
// Capture time code in view of CUDA graph
static ncclResult_t ncclSetupCollKernel(struct ncclInfo* info) {
  ncclComm_t comm = info->comm;
  if (comm->nRanks == 1) {
    if (info->sendbuff != info->recvbuff)
      CUDACHECK(hipMemcpyAsync(info->recvbuff, info->sendbuff, info->nBytes, hipMemcpyDeviceToDevice, info->stream));
    return ncclSuccess;
  }

  // Compute cuda kernel arg and proxy arg templates
  struct ncclQueueElem* eqElem;
  NCCLCHECK(ncclAddQueueElem(comm->enqueueInfo, &eqElem));
  struct ncclWorkElem* work = &eqElem->work;
  eqElem->proxyArgs.nsubs = 1;
  NCCLCHECK(computeColl(info, work, &eqElem->proxyArgs));

  // Determine grid size
  hipLaunchParams* params = comm->myParams;
  params->gridDim.x += info->nChannels;
  params->gridDim.x = std::min<unsigned>(params->gridDim.x, comm->nChannels);
  params->blockDim.x = std::max<unsigned>(params->blockDim.x, info->nThreads);
  comm->enqueueInfo->maxChannels = params->gridDim.x;  // params may be varied by a second graph hence we need to capture it here

  // Inline the first kernel
  if (params->func == NULL) {
    params->func = (void *)ncclKerns[0];
    memcpy(&comm->args, work, sizeof(struct ncclWorkElem));
    comm->args.coll.bid = 0;  // Only inline for channel 0
    comm->args.active = 2;    // I am so far the last element; may be changed later in aggregation mode
  }

  return ncclSuccess;
}

// Dynamic enqueue code
static ncclResult_t ncclEnqueueCollKernel(ncclComm_t comm, struct ncclQueueElem* eqElem) {
  struct ncclWorkElem* work = &eqElem->work;
  struct ncclProxyArgs* proxyArgs = &eqElem->proxyArgs;

  int nChannels = work->coll.nChannels;
  for (int bid=0; bid<nChannels; bid++) {
    int channelId = comm->lastChannel % comm->nChannels;
    struct ncclChannel* channel = comm->channels+channelId;

    // Proxy
    proxyArgs->subs[0].channel = channel;
    proxyArgs->opCount = comm->collOpCount;
    proxyArgs->commOpCount = comm->opCount;

    if (proxyArgs->subs[0].nsteps) NCCLCHECK(ncclProxySaveColl(proxyArgs, comm->nRanks));

    comm->lastChannel++;
    work->coll.bid = bid % nChannels;
    NCCLCHECK(getNextOp(channel, NULL, work));
    //INFO(NCCL_COLL, "Host enqueue: bid %d channel %d index %ld nThreads %d funcIndex %d count %ld nChannels %d",
    //      work->coll.bid, channelId, channel->workFifoTail, work->nThreads, work->funcIndex, work->coll.count, work->coll.nChannels);
  }
  comm->collOpCount++;
  return ncclSuccess;
}

#define NCCL_MIN_CHANNEL_SIZE (NCCL_LL_THREAD_THRESHOLD*64)
#define NCCL_AGG_CHANNEL_SIZE (1LL << 21) /* 2 MiB, ideal per-channel size to fully utilize bandwidth */

ncclResult_t ncclSetupAsyncKernels(ncclComm_t comm) {
  if (comm->asyncOpCount == 0) {
    return ncclSuccess;
  } else if (comm->asyncOpCount == 1) {
    // No aggregation
    struct ncclInfo* info = comm->asyncOps;
    info->nChannels = 0;
    NCCLCHECK(ncclSetupCollKernel(info));
  } else {
    // Aggregation
    size_t channelSize = NCCL_AGG_CHANNEL_SIZE * comm->nRanks;  // scale channel size based on nranks as latency increases
    // Reduce the per-channel size if we cannot fully utilize the channels
    while (comm->asyncTotalSize < channelSize * comm->nChannels && channelSize > NCCL_MIN_CHANNEL_SIZE) channelSize /= 2;
    int channelUsed = 0;
    for (int c = 0; c < comm->asyncOpCount; c++) {
      struct ncclInfo* info = comm->asyncOps+c;
      info->nChannels = std::min((int)DIVUP(info->nBytes, channelSize), comm->nChannels); // assign number of channels
      channelUsed += info->nChannels;
      NCCLCHECK(ncclSetupCollKernel(info));
    }
    // If we wrap around on channels, then the inlined op on channel 0 is not the last one on this channel
    // Then we need to change active from 2 to 1
    if (channelUsed > comm->nChannels) comm->args.active = 1;
  }
  // Reset counters
  comm->asyncOpCount = 0;
  comm->asyncTotalSize = 0;
  return ncclSuccess;
}

static ncclResult_t ncclSaveAsyncColl(struct ncclInfo* info) {
  ncclComm_t comm = info->comm;
  if (comm->asyncOpCount >= NCCL_MAX_OPS) {
    WARN("Too many async operations in progress, max is %d", NCCL_MAX_OPS);
    return ncclInvalidUsage;
  }
  memcpy(comm->asyncOps+comm->asyncOpCount, info, sizeof(struct ncclInfo));
  comm->asyncOpCount++;
  comm->asyncTotalSize += info->nBytes;
  return ncclSuccess;
}

// Save p2p operations in comm->p2pSends and p2pRecvs. Operations will be posted to channels
// during ncclGroupEnd()
static ncclResult_t ncclSaveP2p(struct ncclInfo* info) {
  struct ncclComm* comm = info->comm;
  int peer = info->root;
  ssize_t nBytes = info->count*ncclTypeSize(info->datatype);
  if (info->opName[0] == 'S') { // Send
    if (peer != comm->rank) {
      int delta = (comm->nRanks - (comm->rank-peer)) % comm->nRanks;
      for (int c=0; c<comm->p2pnChannelsPerPeer; c++) {
        int channelId = (delta+comm->p2pChannels[c]) % comm->p2pnChannels;
        if (comm->channels[channelId].peers[peer].send[0].connected == 0) {
          comm->connectSend[peer] |= (1<<channelId);
          comm->connect[0] = 1;
        }
        if (comm->p2pNet && comm->channels[channelId].peers[peer].send[NCCL_CONN_IDX_P2P_NET].connected == 0) {
          comm->connectSend[peer+comm->nRanks*NCCL_CONN_IDX_P2P_NET] |= (1<<channelId);
          comm->connect[NCCL_CONN_IDX_P2P_NET] = 1;
        }
      }
    }
    NCCLCHECK(enqueueP2pInfo(comm->p2pSends+info->root, (void*)info->sendbuff, nBytes));
    comm->p2pSendCount++;
  } else {
    if (peer != comm->rank) {
      int delta = (comm->nRanks + (comm->rank-peer)) % comm->nRanks;
      for (int c=0; c<comm->p2pnChannelsPerPeer; c++) {
        int channelId = (delta+comm->p2pChannels[c]) % comm->p2pnChannels;
        if (comm->channels[channelId].peers[peer].recv[0].connected == 0) {
          comm->connectRecv[peer] |= (1<<channelId);
          comm->connect[0] = 1;
        }
        if (comm->p2pNet && comm->channels[channelId].peers[peer].recv[NCCL_CONN_IDX_P2P_NET].connected == 0) {
          comm->connectRecv[peer+comm->nRanks*NCCL_CONN_IDX_P2P_NET] |= (1<<channelId);
          comm->connect[NCCL_CONN_IDX_P2P_NET] = 1;
        }
      }
    }
    NCCLCHECK(enqueueP2pInfo(comm->p2pRecvs+info->root, info->recvbuff, nBytes));
    comm->p2pRecvCount++;
  }
  return ncclSuccess;
}

static int getSegment(int delta, struct ncclWork* work) {
  for (int s=0; s<NCCL_MAX_WORK_ELEMENTS && work->elems[s].p2p.delta != delta; s++) {
    if (work->elems[s].p2p.nThreads == 0) return s;
  }
  return -1;
}

static ncclResult_t computeP2pWorkElem(struct ncclInfo* info /* input */, struct ncclWorkElem* elem /* output */) {
  elem->comm = info->comm->devComm;
  elem->funcIndex = FUNC_INDEX_P2P;
  elem->nThreads = NCCL_MAX_NTHREADS;
  elem->sendbuff = info->sendbuff;
  elem->recvbuff = info->recvbuff;
  elem->op.opCount = info->comm->p2pOpCount;
  elem->p2p.sendCount = info->sendbytes;
  elem->p2p.recvCount = info->recvbytes;
  elem->p2p.sendChunkSize = info->sendChunkSize;
  elem->p2p.recvChunkSize = info->recvChunkSize;
  elem->p2p.delta = info->delta;
  return ncclSuccess;
}

static ncclResult_t enqueueP2pOp(struct ncclWorkElem* elem /* input */, struct ncclWork* work, int s) {
  // Copy element into corresponding segment of ncclWork
  memcpy(work->elems+s, elem, sizeof(struct ncclWorkElem));

  // Determine nThreads at dynamic time
  const int nsegments = s+1;
  int nThreads = 512;
  while (nsegments*nThreads > 256) nThreads /= 2;
  //if (nThreads >= 128) nThreads += WARP_SIZE;
  for (int i=0; i<nsegments; i++) work->elems[i].p2p.nThreads = nThreads;

  return ncclSuccess;
}

ncclResult_t ncclEnqueueP2pKernel(struct ncclComm* comm, struct ncclQueueElem* eqElem) {
  struct ncclWorkElem* workElem = &eqElem->work;
  struct ncclProxyArgs* proxyArgs = &eqElem->proxyArgs;

  // Try to reuse last p2p operation if not full yet
  struct ncclChannel* channel = proxyArgs->subs[0].channel;
  int opIndex = (channel->workFifoTail-1+NCCL_MAX_OPS)%NCCL_MAX_OPS;
  struct ncclWork* w = channel->workFifo+opIndex;
  int segment = -1;
  if (channel->workCount && w->elems[0].funcIndex == FUNC_INDEX_P2P && w->elems[NCCL_MAX_WORK_ELEMENTS-1].p2p.nThreads == 0) {
    // Try to pack more segments into a single operation
    segment = getSegment(workElem->p2p.delta, w);
  }
  if (segment == -1) {
    NCCLCHECK(getNextOp(channel, &w, NULL));
    segment = 0;
  }

  // store work element into FIFO
  NCCLCHECK(ncclProxySaveP2p(comm, proxyArgs));
  NCCLCHECK(enqueueP2pOp(workElem, w, segment));
  comm->p2pOpCount++;
  return ncclSuccess;
}

ncclResult_t ncclSetupP2pKernel(struct ncclInfo* info) {
  ncclComm* comm = info->comm;
  // Compute cuda kernel arg and proxy arg templates
  struct ncclQueueElem* eqElem;
  NCCLCHECK(ncclAddQueueElem(comm->enqueueInfo, &eqElem));

  // The proxy code will set and tune the send/recv chunk size, make sure to run it first.
  NCCLCHECK(ncclProxyComputeP2p(info, &eqElem->proxyArgs));
  NCCLCHECK(computeP2pWorkElem(info, &eqElem->work));

  eqElem->proxyArgs.sendIdx = info->sendIdx;
  eqElem->proxyArgs.recvIdx = info->recvIdx;
  eqElem->work.p2p.sendIdx = info->sendIdx;
  eqElem->work.p2p.recvIdx = info->recvIdx;

  int channelId = info->channelId;
  hipLaunchParams* params = comm->myParams;
  params->gridDim.x = std::max<unsigned>(params->gridDim.x, channelId+1);
  params->blockDim.x = std::max<unsigned>(params->blockDim.x, eqElem->work.nThreads);
  comm->enqueueInfo->maxChannels = params->gridDim.x;  // params may be varied by a second graph hence we need to capture it here

  // Record the first kernel to launch
  // Just for CUDA kernel to know this is a P2P operation
  // The CUDA kernel does not use the inlined first work element as fastpath argument
  if (params->func == NULL) {
    params->func = (void *)ncclKerns[0];
    memcpy(&comm->args, &eqElem->work, sizeof(struct ncclWorkElem));
  }
  return ncclSuccess;
}

template<int USING_CUDA_GRAPH>
void HIPRT_CB ncclEnqueueHostSetup(void* arg) {
  ncclResult_t ret;
  struct ncclQueueInfo* eqInfo = (struct ncclQueueInfo*)arg;
  ncclComm_t comm = eqInfo->comm;

  // Iterate through the element list
  struct ncclQueueElem* eqElem = eqInfo->elemList.head;
  while (eqElem != eqInfo->elemList.tail) { // The queue always has one extra element
    if (eqElem->work.funcIndex == FUNC_INDEX_P2P) {
      NCCLCHECKGOTO(ncclEnqueueP2pKernel(comm, eqElem), ret, cb_end);
    } else {
      NCCLCHECKGOTO(ncclEnqueueCollKernel(comm, eqElem), ret, cb_end);
    }
    eqElem = eqElem->next;
  }

  NCCLCHECKGOTO(setupLaunch(eqInfo, USING_CUDA_GRAPH), ret, cb_end);
  NCCLCHECKGOTO(ncclLaunchProxy(eqInfo), ret, cb_end);

cb_end:
  if (ret != ncclSuccess) {
    WARN("Failure in host setup : %s", ncclGetErrorString(ret));
  }
  eqInfo->ret = ret;
}

template void HIPRT_CB ncclEnqueueHostSetup<0>(void*);
template void HIPRT_CB ncclEnqueueHostSetup<1>(void*);

ncclResult_t ncclGetCudaGraph(ncclComm_t comm, cudaGraph_t* graph) {
  comm->usingCudaGraph = 0;
#if CUDART_VERSION >= 11030
  cudaStreamCaptureStatus captureStatus;
  unsigned long long cudaGraphId;
  if (comm->driverVersion < 11030) {
    CUDACHECK(cudaStreamIsCapturing(comm->userStream, &captureStatus));
    if (captureStatus != cudaStreamCaptureStatusNone) {
      WARN("The installed CUDA driver is older than the minimum version (R465) required for NCCL's CUDA Graphs support");
      return ncclInvalidUsage;
    }
    return ncclSuccess;
  }
  CUDACHECK(cudaStreamGetCaptureInfo_v2(comm->userStream, &captureStatus, &cudaGraphId, graph, NULL, NULL));
  if (captureStatus == cudaStreamCaptureStatusActive) {
    if (cudaGraphId != comm->lastCudaGraphId) {
      INFO(NCCL_COLL, "stream is being captured by a new graph, id %llu", cudaGraphId);
      // We are in a new graph, hence need to forget the last setup node so that
      // the first setup node in the new graph will not have a dependency
      comm->lastCudaGraphId = cudaGraphId;
      comm->lastSetupNode = NULL;
    }
    if (comm->launchMode == ncclComm::GROUP) comm->launchMode = ncclComm::GROUP_GRAPH;
    comm->usingCudaGraph = 1;
  }
#endif
  return ncclSuccess;
}

ncclResult_t ncclCudaGraphHostSetup(ncclComm_t comm, cudaGraph_t graph) {
#if CUDART_VERSION >= 11030
  struct ncclQueueInfo* eqInfo = comm->enqueueInfo;
  // Create a CUDA object to wrap around the argument space
  // which CUDA graph would manage lifetime of
  cudaUserObject_t object;
  CUDACHECK(cudaUserObjectCreate(&object, eqInfo, ncclDestroyQueueInfo, 1/*initialRefcount*/, cudaUserObjectNoDestructorSync));
  CUDACHECK(cudaGraphRetainUserObject(graph, object, 1, cudaGraphUserObjectMove));

  cudaHostFn_t fn = ncclEnqueueHostSetup<1>;
  // Add a CPU node to the graph
  cudaGraphNode_t setupNode;
  cudaHostNodeParams setupNodeParams = {fn, eqInfo};
  int numDependencies = comm->lastSetupNode == NULL ? 0 : 1;
  CUDACHECK(cudaGraphAddHostNode(&setupNode, graph, &comm->lastSetupNode, numDependencies, &setupNodeParams));
  CUDACHECK(cudaStreamUpdateCaptureDependencies(comm->userStream, &setupNode, 1, cudaStreamAddCaptureDependencies));
  comm->lastSetupNode = setupNode;
  return ncclSuccess;
#else
  WARN("NCCL does not support this CUDA version for CUDA graph feature");
  return ncclInternalError;
#endif
}

ncclResult_t ncclEnqueueCheck(struct ncclInfo* info) {
  // [RCCL] Check for clique-based kernel support
  {
    if (info->comm->cliqueManager->IsSupported(info->coll,
                                               info->count,
                                               info->datatype,
                                               info->op))
    {
      // Declare the input / output pointers being used (to exchange via IPC with other ranks)
      // This is done immediately, and does not block
      NCCLCHECK(info->comm->cliqueManager->DeclarePointers(info->sendbuff, info->recvbuff));
    }
  }
  // [/RCCL]

  // Launch asynchronously if needed
  if (ncclAsyncMode()) {
    ncclResult_t ret = ncclSuccess;
    int savedDev = -1;
    // Check arguments
    NCCLCHECK(PtrCheck(info->comm, info->opName, "comm"));
    if (info->comm->checkPointers) {
      CUDACHECKGOTO(hipGetDevice(&savedDev), ret, end);
      CUDACHECKGOTO(hipSetDevice(info->comm->cudaDev), ret, end);
    }
    NCCLCHECKGOTO(ArgsCheck(info), ret, end);
    // Always register comm even in case of error to make sure ncclGroupEnd
    // cleans it up.
    NCCLCHECKGOTO(ncclAsyncColl(info->comm), ret, end);
    NCCLCHECKGOTO(checkSetStream(info), ret, end);

    INFO(NCCL_COLL,"%s: opCount %lx sendbuff %p recvbuff %p count %zi datatype %d op %d root %d comm %p [nranks=%d] stream %p",
        info->opName, info->coll == ncclFuncSendRecv ? info->comm->p2pOpCount : info->comm->collOpCount, info->sendbuff, info->recvbuff, info->count,
        info->datatype, info->op, info->root, info->comm, info->comm->nRanks, info->stream);

    if (info->coll == ncclFuncSendRecv) { //p2p stored separately
      NCCLCHECKGOTO(ncclSaveP2p(info), ret, end);
    } else {
      NCCLCHECKGOTO(ncclSaveAsyncColl(info), ret, end);
    }

end:
    if (savedDev != -1) CUDACHECK(hipSetDevice(savedDev));
    ncclAsyncErrCheck(ret);
    return ret;
  } else {
    NCCLCHECK(PtrCheck(info->comm, info->opName, "comm"));
    NCCLCHECK(ArgsCheck(info));
    NCCLCHECK(checkSetStream(info));

    INFO(NCCL_COLL,"%s: opCount %lx sendbuff %p recvbuff %p count %zi datatype %d op %d root %d comm %p [nranks=%d] stream %p",
        info->opName, info->comm->collOpCount, info->sendbuff, info->recvbuff, info->count,
        info->datatype, info->op, info->root, info->comm, info->comm->nRanks, info->stream);

    // Check whether we are in cuda graph mode
    cudaGraph_t graph;
    ncclComm_t comm = info->comm;
    NCCLCHECK(ncclGetCudaGraph(comm, &graph));

    // Common part between graph mode and non-graph mode
    NCCLCHECK(ncclSetupCollKernel(info));

    // Host setup
    if (comm->usingCudaGraph) {
      NCCLCHECK(ncclCudaGraphHostSetup(comm, graph));
    } else {
      ncclEnqueueHostSetup<0>(comm->enqueueInfo);
      NCCLCHECK(comm->enqueueInfo->ret);
    }

    // Common part between graph mode and non-graph mode
    NCCLCHECK(ncclLaunchBarrier(comm));
    NCCLCHECK(ncclLaunchKernel(comm));
    NCCLCHECK(ncclRecordEvents(comm));
    NCCLCHECK(ncclLaunchReset(comm));
    return ncclSuccess;
  }
}
