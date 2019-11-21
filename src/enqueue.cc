/*************************************************************************
 * Copyright (c) 2017-2019, NVIDIA CORPORATION. All rights reserved.
 * Modifications Copyright (c) 2019 Advanced Micro Devices, Inc. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

#include "enqueue.h"
#include "argcheck.h"

// Only generate inline kernels for LL
#define NCCL_FUNC5(coll, op, dtype) \
  NCCL_KERN_NAME(coll##LL, op, dtype), \
  NCCL_KERN_NAME(coll##LL, op, dtype), \
  NCCL_KERN_NAME(coll##LL, op, dtype)

#define NCCL_FUNC4(coll, op, dtype) \
  NCCL_FUNC5(coll##Tree, op, dtype), \
  NCCL_FUNC5(coll##Ring, op, dtype)

// Must be consistent with ncclDataType_t
#define NCCL_FUNCS3A(coll, op) \
  NCCL_FUNC4(coll, op,  i8), \
  NCCL_FUNC4(coll, op,  u8), \
  NCCL_FUNC4(coll, op, i32), \
  NCCL_FUNC4(coll, op, u32), \
  NCCL_FUNC4(coll, op, i64), \
  NCCL_FUNC4(coll, op, u64), \
  NCCL_FUNC4(coll, op, f16), \
  NCCL_FUNC4(coll, op, f32), \
  NCCL_FUNC4(coll, op, f64), \
  NCCL_FUNC4(coll, op, b16)
#define NCCL_FUNCS3B(coll, op) \
  NCCL_FUNC4(coll, op,  i8), \
  NCCL_FUNC4(coll, op,  i8), \
  NCCL_FUNC4(coll, op,  i8), \
  NCCL_FUNC4(coll, op,  i8), \
  NCCL_FUNC4(coll, op,  i8), \
  NCCL_FUNC4(coll, op,  i8), \
  NCCL_FUNC4(coll, op,  i8), \
  NCCL_FUNC4(coll, op,  i8), \
  NCCL_FUNC4(coll, op,  i8), \
  NCCL_FUNC4(coll, op,  i8)

// Must be consistent with ncclRedOp_t -- but we only generate kernel for sums.
#define NCCL_FUNCS2A(coll) \
  NCCL_FUNCS3A(coll, sum), \
  NCCL_FUNCS3A(coll, sum), \
  NCCL_FUNCS3A(coll, sum), \
  NCCL_FUNCS3A(coll, sum)
#define NCCL_FUNCS2B(coll) \
  NCCL_FUNCS3B(coll, copy), \
  NCCL_FUNCS3B(coll, copy), \
  NCCL_FUNCS3B(coll, copy), \
  NCCL_FUNCS3B(coll, copy)

typedef void(*ncclKern_t)(struct ncclColl);
// Must be consistent with the ncclFuncSet enum
static ncclKern_t const ncclKerns[NCCL_NUM_FUNCTIONS*ncclNumOps*ncclNumTypes*NCCL_NUM_ALGORITHMS*NCCL_NUM_PROTOCOLS] = {
  NCCL_FUNCS2B(ncclBroadcast),
  NCCL_FUNCS2A(ncclReduce),
  NCCL_FUNCS2B(ncclAllGather),
  NCCL_FUNCS2A(ncclReduceScatter),
  NCCL_FUNCS2A(ncclAllReduce)
};

/*****************************************************************************/
/*       Launch system : synchronization and CUDA kernel launch              */
/*****************************************************************************/

ncclResult_t ncclLaunchCooperativeKernelMultiDevice(hipLaunchParams *paramsList, int* cudaDevs, int numDevices, int cgMode) {
  if (cgMode & 0x01) {
    CUDACHECK(hipExtLaunchMultiKernelMultiDevice(paramsList, numDevices,
            // These flags are to reduce the latency of using this API
            0));
    return ncclSuccess;
  }
  int savedDev;
  CUDACHECK(hipGetDevice(&savedDev));
  for (int i = 0; i < numDevices; i++) {
    hipLaunchParams* params = paramsList+i;
    CUDACHECK(hipSetDevice(cudaDevs[i]));
    hipLaunchKernelGGL(((void (*)(struct ncclColl))params->func), params->gridDim, params->blockDim, params->sharedMem, params->stream, **((struct ncclColl **)(params->args)));
  }
  CUDACHECK(hipSetDevice(savedDev));
  return ncclSuccess;
}

ncclResult_t setupLaunch(struct ncclComm* comm, hipLaunchParams* params) {
  params->gridDim.x = std::min<unsigned>(params->gridDim.x, comm->nChannels);

  // Set active = 2 for the last operation
  for (int r=0; r<params->gridDim.x; r++) {
    struct ncclChannel* channel = comm->channels+r;
    STORE(&channel->collectives[(channel->collStart+channel->collCount-1)%NCCL_MAX_OPS].active, 2);
  }

  // Find the first operation, choose the kernel accordingly and pass it
  // as the first argument.
  struct ncclColl* coll = comm->channels[0].collectives+comm->channels[0].collStart;
  memcpy(&comm->args, coll, sizeof(struct ncclColl));
  // As we pass that coll directly, we can free it immediately.
  STORE(&coll->active, 0);

  params->func = (void *)ncclKerns[coll->funcIndex];
  return ncclSuccess;
}

ncclResult_t ncclCpuBarrierIn(struct ncclComm* comm, int* isLast) {
  volatile int* ptr = (volatile int*)(comm->intraBarrier+comm->intraPhase);
  int val = LOAD(ptr);
  bool done = false;
  while (done == false) {
    if (val >= comm->intraRanks) {
      WARN("Trying to launch too many collectives");
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
    WARN("Trying to launch too many collectives");
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

ncclResult_t ncclBarrierEnqueue(struct ncclComm* comm) {
  if (comm->nRanks == 1) return ncclSuccess;
  hipLaunchParams* params = comm->myParams;

  NCCLCHECK(setupLaunch(comm, params));

  // Use internal NCCL stream for CGMD/GROUP launch if required or if the user stream is NULL
  if (comm->launchMode == ncclComm::GROUP && (comm->groupCudaStream || comm->userStream == NULL)) {
    // Enqueue event in user stream
    CUDACHECK(hipEventRecord(comm->doneEvent, comm->userStream));
    // Create dependency between user stream and internal NCCL stream
    CUDACHECK(hipStreamWaitEvent(comm->groupStream, comm->doneEvent, 0));
    params->stream = comm->groupStream;
  } else {
    if (comm->userStream != params->stream) {
      // Stream changed from last call, create dependency against last NCCL kernel launch
      CUDACHECK(hipStreamWaitEvent(comm->userStream, comm->doneEvent, 0));
    }
    params->stream = comm->userStream;
  }

  int isLast = 0;
  NCCLCHECK(ncclCpuBarrierIn(comm, &isLast));

  if (isLast) {
    if (comm->launchMode == ncclComm::GROUP) {
      // I'm the last. Launch all operations.
      NCCLCHECK(ncclLaunchCooperativeKernelMultiDevice(comm->intraParams, comm->intraCudaDevs, comm->intraRanks, *comm->intraCGMode));
    }
    NCCLCHECK(ncclCpuBarrierLast(comm));
  }
  return ncclSuccess;
}

ncclResult_t ncclBarrierEnqueueWait(ncclComm_t comm) {
  if (comm->nRanks == 1) return ncclSuccess;
  // We can't print the CG mode before the first barrier happened.
  if (comm->rank == 0 && *comm->intraCGMode & 0x10) {
    *comm->intraCGMode ^= 0x10;
    INFO(NCCL_INIT,"Launch mode %s%s%s",
        comm->launchMode == ncclComm::GROUP ? "Group" : "Parallel",
        *comm->intraCGMode ? "/CGMD" : "",
        (comm->launchMode == ncclComm::GROUP && comm->groupCudaStream) ? "/Stream" : "");
  }

  NCCLCHECK(ncclCpuBarrierOut(comm));

  hipLaunchParams *params = comm->myParams;
  if (comm->launchMode == ncclComm::PARALLEL) {
    hipLaunchKernelGGL(((void (*)(struct ncclColl))params->func), params->gridDim, params->blockDim, params->sharedMem, params->stream, **((struct ncclColl **)(params->args)));
  }
  // Start the network proxies as soon as the kernel has been launched. We can't
  // perform any CUDA call between the two or having a hipFree between the CUDA
  // launch and the transportStartProxy call could cause a deadlock.
  // Also, starting the proxies after the CUDA launch seems to be better for
  // performance (latency).
  for (int r=0; r<params->gridDim.x; r++) {
    struct ncclChannel* channel = comm->channels+r;
    channel->collStart = channel->collFifoTail;
    channel->collCount = 0;
  }
  params->gridDim.x = params->blockDim.x = 0;
  comm->lastOpCount = comm->opCount;
  NCCLCHECK(transportStartProxy(comm));
  return ncclSuccess;
}

ncclResult_t ncclEnqueueEvents(ncclComm_t comm) {
  hipLaunchParams *params = comm->myParams;
  // Enqueue event after NCCL kernel
  CUDACHECK(hipEventRecord(comm->doneEvent, params->stream));
  // Use internal NCCL stream for CGMD/GROUP launch if required or if the user stream is NULL
  if (comm->launchMode == ncclComm::GROUP && (comm->groupCudaStream || comm->userStream == NULL)) {
    // Create dependency between NCCL internal stream and user stream
    CUDACHECK(hipStreamWaitEvent(comm->userStream, comm->doneEvent, 0));
  }
  comm->userStreamSet = false;
  return ncclSuccess;
}

/*****************************************************************************/
/* Enqueueing system : computation of kernel and proxy operations parameters */
/*****************************************************************************/

// Trees are not perfectly sticking to the model for medium sizes. Applying a static correction
// factor is not ideal but works quite well. Powers of two, 64 B to 1 GB.
static float treeCorrectionFactor[NCCL_NUM_PROTOCOLS][22] = {
  { 1.0, 1.0, 1.0, 1.0,  .9,  .8,  .7,  .7,  .7,  .7,  .6,  .5,  .5,  .5,  .6,  .7,  .8,  .9,  .9, 1.0, 1.0, 1.0 },
  { 1.0, 1.0, 1.0, 1.0, 1.0,  .9,  .8,  .8,  .8,  .8,  .7,  .7,  .7,  .6,  .6,  .7,  .7,  .8,  .8,  .9,  .9, 1.0 },
  {  .9,  .9,  .9,  .9,  .9,  .9,  .9,  .8,  .7,  .6,  .6,  .5,  .5,  .5,  .5,  .5,  .5,  .6,  .6,  .7,  .8,  .9 }
};

static ncclResult_t getAlgoInfo(struct ncclInfo* info) {
  struct ncclComm* comm = info->comm;
  float minTime = 3600000.0; // Hopefully no operation will take an hour to complete.
  // Find algorithm / protocol.
  info->algorithm = -1;
  info->protocol = -1;
  for (int a=0; a<NCCL_NUM_ALGORITHMS; a++) {
    for (int p=0; p<NCCL_NUM_PROTOCOLS; p++) {
      float bw = comm->bandwidths[info->coll][a][p];
      if (bw == 0) continue;
      int logSize = log2i(info->nBytes>>6);
      if (a == NCCL_ALGO_TREE && logSize < 22) bw *= treeCorrectionFactor[p][logSize];
      float time = comm->latencies[info->coll][a][p] + (info->nBytes) / (1000 * bw);
      if (time < minTime) {
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

  if (comm->rank == 0) INFO(NCCL_COLL, "%ld Bytes -> Algo %d proto %d time %d", info->nBytes, info->algorithm, info->protocol, minTime);
  TRACE(NCCL_COLL, "%ld Bytes -> Algo %d proto %d time %f", info->nBytes, info->algorithm, info->protocol, minTime);

  int nc = comm->nChannels;
  int nt = comm->maxThreads[info->protocol];
  int threadThreshold = comm->threadThresholds[info->algorithm][info->protocol];
  while (info->nBytes < nc*nt*threadThreshold) {
    if (nc >= 2) nc--;
#if defined(__HIP_PLATFORM_HCC__) || defined(__HCC__) || defined(__HIPCC__)
    // do not reduce threads count on VEGA
#else
    else if ((nt % 128) == 0) nt/=2;
#endif
    else break;
  }
#if defined(__HIP_PLATFORM_HCC__) || defined(__HCC__) || defined(__HIPCC__)
#else
  if (info->protocol == NCCL_PROTO_SIMPLE) nt += WARP_SIZE; // Extra warp for sync
#endif
  info->nChannels = nc;
  info->nThreads = nt;
  return ncclSuccess;
}

static ncclResult_t getPatternInfo(struct ncclInfo* info) {
  switch (info->coll) {
    case ncclCollBroadcast:
      info->pattern = info->algorithm == NCCL_ALGO_TREE ? ncclPatternTreeDown : ncclPatternPipelineFrom; break;
    case ncclCollReduce:
      info->pattern = info->algorithm == NCCL_ALGO_TREE ? ncclPatternTreeUp : ncclPatternPipelineTo; break;
    case ncclCollReduceScatter:
    case ncclCollAllGather:
      info->pattern = ncclPatternRing; break;
    case ncclCollAllReduce:
      info->pattern = info->algorithm == NCCL_ALGO_TREE ? ncclPatternTreeUpDown : ncclPatternRingTwice; break;
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
    case ncclPatternRing:
      info->nstepsPerLoop = info->comm->nRanks-1; info->nchunksPerLoop = info->comm->nRanks; break;
    case ncclPatternRingTwice:
      info->nstepsPerLoop = 2*(info->comm->nRanks-1); info->nchunksPerLoop = info->comm->nRanks; break;
    default:
      WARN("Unknown pattern %d\n", info->pattern);
      return ncclInternalError;
  }
  return ncclSuccess;
}

static ncclResult_t computeColl(struct ncclInfo* info /* input */, struct ncclColl* coll, struct ncclProxyArgs* proxyArgs /* output */) {
  // Set nstepsPerLoop and nchunksPerLoop
  NCCLCHECK(getAlgoInfo(info));
  NCCLCHECK(getPatternInfo(info));
  NCCLCHECK(getLoopInfo(info));

  coll->args.root = info->root;
  coll->args.N = info->count;
  coll->args.ThisInput = info->sendbuff;
  coll->args.ThisOutput = info->recvbuff;
  coll->args.comm = info->comm->devComm;
  coll->args.opCount = info->comm->opCount;
  coll->args.nChannels = info->nChannels;
  coll->args.nThreads = info->nThreads;

  coll->funcIndex = FUNC_INDEX(info->coll, info->op, info->datatype, info->algorithm, info->protocol);

  int stepSize   = (info->protocol == NCCL_PROTO_LL ? NCCL_LL_BUFF_SIZE : info->protocol == NCCL_PROTO_LL128 ? NCCL_LL128_BUFF_SIZE : info->comm->channels[0].buffSize ) / NCCL_STEPS;
  int chunkSteps = (info->protocol == NCCL_PROTO_SIMPLE && info->algorithm == NCCL_ALGO_RING) ? info->chunkSteps : 1;
  int sliceSteps = (info->protocol == NCCL_PROTO_SIMPLE && info->algorithm == NCCL_ALGO_RING) ? info->sliceSteps : 1;
  int chunkSize  = stepSize*chunkSteps;

  // Compute lastChunkSize
  if (info->algorithm == NCCL_ALGO_TREE && info->protocol == NCCL_PROTO_SIMPLE) {
    if (info->pattern == ncclPatternTreeUpDown) {
      // Optimize chunkSize / nSteps
      while (info->nBytes / (info->nChannels*chunkSize) < info->comm->channels[0].treeUp.depth*8 && chunkSize > 131072) chunkSize /= 2;
      while (info->nBytes / (info->nChannels*chunkSize) < info->comm->channels[0].treeUp.depth*4 && chunkSize > 65536) chunkSize /= 2;
      while (info->nBytes / (info->nChannels*chunkSize) < info->comm->channels[0].treeUp.depth && chunkSize > 32768) chunkSize /= 2;
    }
    // Use lastChunkSize as chunkSize
    coll->args.lastChunkSize = chunkSize / ncclTypeSize(info->datatype);
  } else if (info->protocol == NCCL_PROTO_LL) {
    int sliceSize = NCCL_LL_SLICE_LINES * sizeof(uint64_t);
    const ssize_t loopSize = info->nChannels*info->nchunksPerLoop*(ssize_t)sliceSize;
    coll->args.lastChunkSize = DIVUP((info->nBytes-(info->nBytes/loopSize)*loopSize), info->nChannels*info->nchunksPerLoop);
    ALIGN_SIZE(coll->args.lastChunkSize, info->nThreads*sizeof(uint64_t));
    coll->args.lastChunkSize /= ncclTypeSize(info->datatype);
  } else if (info->algorithm == NCCL_ALGO_TREE && info->protocol == NCCL_PROTO_LL128) {
    int nstepsInter = 1+log2i(info->comm->nNodes);
    while (info->nBytes / (info->nChannels*chunkSize) < nstepsInter*4 && chunkSize > 32768) chunkSize /= 2;
    // Use lastChunkSize as chunkSize
    coll->args.lastChunkSize = chunkSize*NCCL_LL128_DATAELEMS/(NCCL_LL128_LINEELEMS*ncclTypeSize(info->datatype));
  }

  // Compute nSteps for proxies
  int chunkEffectiveSize = chunkSize;
  if (info->protocol == NCCL_PROTO_LL) chunkEffectiveSize /= 2;
  if (info->protocol == NCCL_PROTO_LL128) chunkEffectiveSize = (chunkSize / NCCL_LL128_LINEELEMS) * NCCL_LL128_DATAELEMS;
  //if (info->comm->rank == 0) printf("Coll %d, size %ld -> %dx%d, chunkSize %d (algo %d proto%d)\n", info->coll, info->nBytes, info->nChannels, info->nThreads, chunkSize, info->algorithm, info->protocol);
  int nLoops = (int)(DIVUP(info->nBytes, (((size_t)(info->nChannels))*info->nchunksPerLoop*chunkEffectiveSize)));
  proxyArgs->nsteps = info->nstepsPerLoop * nLoops * chunkSteps;
  proxyArgs->sliceSteps = sliceSteps;
  proxyArgs->chunkSteps = chunkSteps;
  proxyArgs->protocol = info->protocol;
  proxyArgs->opCount = info->comm->opCount;
  TRACE(NCCL_NET,"opCount %lx slicesteps %d spl %d cpl %d nbytes %zi -> protocol %d nchannels %d nthreads %d, nloops %d nsteps %d comm %p",
      coll->args.opCount, proxyArgs->sliceSteps, info->nstepsPerLoop, info->nchunksPerLoop, info->nBytes, info->protocol, info->nChannels, info->nThreads,
      nLoops, proxyArgs->nsteps, info->comm);
  return ncclSuccess;
}

static ncclResult_t saveKernel(struct ncclInfo* info) {
  if (info->comm->nRanks == 1) {
    if (info->sendbuff != info->recvbuff)
      CUDACHECK(hipMemcpyAsync(info->recvbuff, info->sendbuff, info->nBytes, hipMemcpyDeviceToDevice, info->stream));
    return ncclSuccess;
  }

  struct ncclColl coll;
  struct ncclProxyArgs proxyArgs;
  memset(&proxyArgs, 0, sizeof(struct ncclProxyArgs));
  NCCLCHECK(computeColl(info, &coll, &proxyArgs));

  info->comm->myParams->blockDim.x = std::max<unsigned>(info->comm->myParams->blockDim.x, coll.args.nThreads);
  if (info->comm->userStreamSet == false) {
    info->comm->userStream = info->stream;
    info->comm->userStreamSet = true;
  } else if (info->stream != info->comm->userStream) {
    WARN("Error : mixing different streams within a group call is not supported.");
    return ncclInvalidUsage;
  }
  for (int bid=0; bid<coll.args.nChannels; bid++) {
    struct ncclChannel* channel = info->comm->channels+(info->comm->myParams->gridDim.x % info->comm->nChannels);

    if (channel->collCount == NCCL_MAX_OPS) {
      WARN("Too many aggregated operations (%d max)", NCCL_MAX_OPS);
      return ncclInvalidUsage;
    }

    // Proxy
    proxyArgs.channel = channel;
    NCCLCHECK(transportSaveProxies(&proxyArgs, info->pattern, info->root, info->comm->nRanks));

    info->comm->myParams->gridDim.x++;

    int opIndex = channel->collFifoTail;
    struct ncclColl* c = channel->collectives+opIndex;
    volatile uint8_t* activePtr = (volatile uint8_t*)&c->active;
    while (LOAD(activePtr) != 0) sched_yield();

    memcpy(c, &coll, sizeof(struct ncclColl));

    c->args.bid = bid;
    STORE(&c->active, 1);
    opIndex = (opIndex+1)%NCCL_MAX_OPS;
    c->nextIndex = opIndex;
    channel->collFifoTail = opIndex;
    channel->collCount++;
  }
  info->comm->opCount++;
  return ncclSuccess;
}


ncclResult_t ncclEnqueueCheck(struct ncclInfo* info) {
  if (info->comm == NULL) return ncclInvalidArgument;

  INFO(NCCL_COLL,"%s: opCount %lx sendbuff %p recvbuff %p count %zi datatype %d op %d root %d comm %p [nranks=%d] stream %p",
       info->opName, info->comm->opCount, info->sendbuff, info->recvbuff, info->count,
       info->datatype, info->op, info->root, info->comm, info->comm->nRanks, info->stream);

  // Launch asynchronously if needed
  if (ncclAsyncMode()) {
    ncclResult_t ret = ncclSuccess;
    int savedDev = -1;
    if (info->comm->checkPointers) {
      CUDACHECKGOTO(hipGetDevice(&savedDev), ret, end);
      CUDACHECKGOTO(hipSetDevice(info->comm->cudaDev), ret, end);
    }
    // Check arguments
    NCCLCHECKGOTO(ArgsCheck(info), ret, end);
    // Always register comm even in case of error to make sure ncclGroupEnd
    // cleans it up.
    NCCLCHECKGOTO(ncclAsyncColl(info->comm), ret, end);
    NCCLCHECKGOTO(saveKernel(info), ret, end);
end:
    if (savedDev != -1) CUDACHECK(hipSetDevice(savedDev));
    ncclAsyncErrCheck(ret);
    return ret;
  } else {
    NCCLCHECK(ArgsCheck(info));
    NCCLCHECK(saveKernel(info));
    NCCLCHECK(ncclBarrierEnqueue(info->comm));
    NCCLCHECK(ncclBarrierEnqueueWait(info->comm));
    NCCLCHECK(ncclEnqueueEvents(info->comm));
    return ncclSuccess;
  }
}
