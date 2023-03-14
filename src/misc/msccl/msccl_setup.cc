/*************************************************************************
 * Copyright (c) Microsoft Corporation.
 * Licensed under the MIT License.
 ************************************************************************/

#include "checks.h"
#include "collectives.h"
#include "proxy.h"
#include "transport.h"

#include "msccl/msccl_lifecycle.h"
#include "msccl/msccl_kernel.h"
#include "msccl/msccl_setup.h"
#include "msccl/msccl_status.h"

ncclResult_t mscclSetupCount(struct mscclAlgo* hostAlgo, ncclComm_t comm, size_t count, ncclDataType_t dataType) {
  mscclStatus& status = mscclGetStatus();
  status.stepSize = comm->buffSizes[hostAlgo->protocol] / NCCL_STEPS;
  status.chunkSteps = hostAlgo->protocol == NCCL_PROTO_SIMPLE ? hostAlgo->chunkSteps : 1;
  status.sliceSteps = hostAlgo->protocol == NCCL_PROTO_SIMPLE ? hostAlgo->sliceSteps : 1;
  status.chunkSize  = status.stepSize * status.chunkSteps;
  status.chunkEffectiveSize = status.chunkSize;
  if (hostAlgo->protocol == NCCL_PROTO_LL) status.chunkEffectiveSize /= 2;
  if (hostAlgo->protocol == NCCL_PROTO_LL128) status.chunkEffectiveSize = (status.chunkSize / NCCL_LL128_LINEELEMS) * NCCL_LL128_DATAELEMS;
  status.dataType = dataType;
  status.nBytes = count * ncclTypeSize(status.dataType) * hostAlgo->sizeMultiplier;
  status.maxAllowedCount = std::max((uint32_t)1, (uint32_t)(status.chunkEffectiveSize / DIVUP(status.nBytes, (size_t)(hostAlgo->nChunksPerLoop))));
  if (status.maxAllowedCount == 0){
    WARN("MSCCL: something went wrong. Max allowed count is 0\n");
    return ncclInternalError;
  }
  if (status.maxAllowedCount >= MSCCL_MAX_COUNT) {
    status.maxAllowedCount = MSCCL_MAX_COUNT - 1;
  }
  return ncclSuccess;
}

ncclResult_t mscclSetupScratch(struct mscclAlgo* hostAlgo, hipStream_t stream) {
  mscclStatus& status = mscclGetStatus();
  size_t sizeNeeded = (status.nBytes * (size_t)(hostAlgo->nScratchChunks)) / (size_t)(hostAlgo->nChunksPerLoop);
  if (sizeNeeded > status.scratchBufferSize){
    CUDACHECK(hipStreamSynchronize(stream));
    CUDACHECK(hipFree(status.scratchBuffer));
    NCCLCHECK(ncclCudaCalloc((char**)&status.scratchBuffer, sizeNeeded));
    status.scratchBufferSize = sizeNeeded;
  }
  return ncclSuccess;
}

ncclResult_t mscclSetupSyncFlags(hipStream_t stream) {
  mscclStatus& status = mscclGetStatus();
  if (status.workIndex > (1ULL << (8*sizeof(status.workIndex))) - 2 * NCCL_MAX_OPS - 1) {
    CUDACHECK(hipMemsetAsync(status.syncFlags, 0, sizeof(struct mscclFlag) * MSCCL_MAX_NUM_THREAD_BLOCKS, stream));
    status.workIndex = 1; // setting the workIndex back to 1 for next iterations
  }
  return ncclSuccess;
}

ncclResult_t mscclSetupConnections(struct mscclAlgo* hostAlgo, ncclComm_t comm) {
  // Check whether there is enough channels
  if (hostAlgo->nChannels > comm->nChannels) {
    WARN("MSCCL: number of channels available (%d) less than required (%d)", comm->nChannels, hostAlgo->nChannels);
    return ncclInvalidUsage;
  }

  // Flag MSCCL connections
  for (int i = 0; i < hostAlgo->nChannels; i++) {
    struct mscclChannelInfo* mCh = hostAlgo->mscclChannels + i;

    int sendPeers[MSCCL_MAX_NUM_THREAD_BLOCKS_PER_CHANNEL];
    for (int p = 0; p < mCh->nSendPeers; p++) {
      sendPeers[p] = mCh->sendPeerInfo[p].peer;
    }

    int recvPeers[MSCCL_MAX_NUM_THREAD_BLOCKS_PER_CHANNEL];
    for (int p = 0; p < mCh->nRecvPeers; p++) {
      recvPeers[p] = mCh->recvPeerInfo[p].peer;
    }

    NCCLCHECK(ncclTransportP2pConnect(comm, i, mCh->nRecvPeers, recvPeers, mCh->nSendPeers, sendPeers, 0 /*connIndex*/));
  }

  // Connect MSCCL connections
  mscclSetIsCallerFlag();
  NCCLCHECK(ncclTransportP2pSetup(comm, NULL, 0));
  mscclClearIsCallerFlag();

  return ncclSuccess;
}

ncclResult_t mscclSetupProxy(struct mscclAlgo* hostAlgo, ncclComm_t comm) {
  mscclStatus& status = mscclGetStatus();
  struct ncclProxyOp proxyOp = {};
  proxyOp.connIndex = 0;
  proxyOp.sliceSteps = status.sliceSteps;
  proxyOp.chunkSteps = status.chunkSteps;
  proxyOp.chunkSize = status.chunkSize;
  proxyOp.protocol = hostAlgo->protocol;
  proxyOp.dtype = status.dataType;
  proxyOp.redOp = 0;
  proxyOp.pattern = 0;
  proxyOp.root = 0;
  proxyOp.nbytes = status.stepSize*proxyOp.sliceSteps;
  proxyOp.opCount = comm->collOpCount;
  int nLoops = (int)(DIVUP(status.nBytes, (size_t)((size_t)hostAlgo->nChunksPerLoop*(size_t)status.chunkEffectiveSize)));
  int nLoopsChunkSteps = nLoops * status.chunkSteps;
  for (int ch = 0; ch < hostAlgo->nChannels; ch++) {
    proxyOp.channelId = ch;
    struct mscclChannelInfo* mscclChannel = hostAlgo->mscclChannels + ch;
    struct ncclChannel* ncclChannel = comm->channels + ch;
    for (int i = 0; i < mscclChannel->nRecvPeers; i++){
      struct mscclChannelPeerInfo* recvPeer = mscclChannel->recvPeerInfo + i;
      int nRecvs = 0;
      for (int j = 0; j < recvPeer->nExistingCounts; j++){
        int c = recvPeer->existingCounts[j];
        int nStepsInCount = DIVUP(c+1, status.maxAllowedCount);
        nRecvs += recvPeer->nTransmissionsOfCount[c] * nStepsInCount;
      }
      proxyOp.nsteps = nLoopsChunkSteps * nRecvs;
      if (proxyOp.nsteps > 0) {
        NCCLCHECK(mscclSaveProxy(ncclChannel, proxyRecv, recvPeer->peer, &proxyOp, 0));
      }
    }
    for (int i=0; i<mscclChannel->nSendPeers; i++){
      struct mscclChannelPeerInfo* sendPeer = &mscclChannel->sendPeerInfo[i];
      int nSends = 0;
      for (int j = 0; j < sendPeer->nExistingCounts; j++){
        int c = sendPeer->existingCounts[j];
        int nStepsInCount = DIVUP(c+1, status.maxAllowedCount);
        nSends += sendPeer->nTransmissionsOfCount[c] * nStepsInCount;
      }
      proxyOp.nsteps = nLoopsChunkSteps * nSends;
      if (proxyOp.nsteps > 0) {
        NCCLCHECK(mscclSaveProxy(ncclChannel, proxySend, sendPeer->peer, &proxyOp, 0));
      }
    }
  }
  NCCLCHECK(ncclProxyStart(comm));
  comm->collOpCount++;
  return ncclSuccess;
}

static ncclResult_t hostToDevRedOp(
    ncclDevRedOpFull *opFull, ncclRedOp_t op, ncclDataType_t datatype, ncclComm *comm
  ) {
  union {
    int8_t i8;
    uint8_t u8;
    int32_t i32;
    uint32_t u32;
    int64_t i64;
    uint64_t u64;
    half f16;
    #if defined(RCCL_BFLOAT16)
      rccl_bfloat16 bf16;
    #endif
    float f32;
    double f64;
    void *ptr;
  };
  u64 = 0;
  opFull->scalarArgIsPtr = false;
  switch (int(op)) {
  case ncclSum:  opFull->op = ncclDevSum;  break;
  case ncclProd: opFull->op = ncclDevProd; break;
  case ncclMax:  opFull->op = ncclDevMax;  break;
  case ncclMin:  opFull->op = ncclDevMin;  break;
  case ncclAvg:
    switch ((int)datatype) {
    case ncclInt8:  case ncclInt32:  case ncclInt64:
    case ncclUint8: case ncclUint32: case ncclUint64:
      opFull->op = ncclDevSumPostDiv;
      u64 = comm->nRanks;
      break;
    case ncclFloat16:
      opFull->op = ncclDevPreMulSum;
      f16 = __float2half(float(1.0/comm->nRanks)); // __double2half not supported pre CUDA 11.x
      break;
    #if defined(RCCL_BFLOAT16)
    case ncclBfloat16:
      opFull->op = ncclDevPreMulSum;
      bf16 = (rccl_bfloat16)(float(1.0/comm->nRanks));
      break;
    #endif
    case ncclFloat32:
      opFull->op = ncclDevPreMulSum;
      f32 = float(1.0/comm->nRanks);
      break;
    case ncclFloat64:
      opFull->op = ncclDevPreMulSum;
      f64 = 1.0/comm->nRanks;
      break;
    }
    opFull->scalarArgIsPtr = false;
    opFull->scalarArg = u64;
    break;
  default: // user created
    int ix = int(ncclUserRedOpMangle(comm, op)) - int(ncclNumOps);
    ncclUserRedOp *user = &comm->userRedOps[ix];
    if (datatype != user->datatype) {
      WARN("Data type supplied to user-created ncclRedOp_t does not match type "
           "given to reduction operation");
      return ncclInvalidArgument;
    }
    *opFull = user->opFull;
    break;
  }
  return ncclSuccess;
}

#define MSCCL_KERNEL_ENTRY_DEVREDOP_NULL() \
  nullptr, \
  nullptr, \
  nullptr

#define MSCCL_KERNEL_ENTRY_DEVREDOP_TYPE(devredop, type) \
  (void *)MSCCL_KERNEL_ENTRY_NAME(devredop, type, LL), \
  (void *)MSCCL_KERNEL_ENTRY_NAME(devredop, type, LL128), \
  (void *)MSCCL_KERNEL_ENTRY_NAME(devredop, type, Simple)

#define MSCCL_KERNEL_ENTRY_DEVREDOP(devredop) \
  MSCCL_KERNEL_ENTRY_DEVREDOP_TYPE(devredop, int8_t), \
  MSCCL_KERNEL_ENTRY_DEVREDOP_TYPE(devredop, uint8_t), \
  MSCCL_KERNEL_ENTRY_DEVREDOP_TYPE(devredop, int32_t), \
  MSCCL_KERNEL_ENTRY_DEVREDOP_TYPE(devredop, uint32_t), \
  MSCCL_KERNEL_ENTRY_DEVREDOP_TYPE(devredop, int64_t), \
  MSCCL_KERNEL_ENTRY_DEVREDOP_TYPE(devredop, uint64_t), \
  MSCCL_KERNEL_ENTRY_DEVREDOP_TYPE(devredop, half), \
  MSCCL_KERNEL_ENTRY_DEVREDOP_TYPE(devredop, float), \
  MSCCL_KERNEL_ENTRY_DEVREDOP_TYPE(devredop, double), \
  MSCCL_KERNEL_ENTRY_DEVREDOP_TYPE(devredop, rccl_bfloat16)

#define MSCCL_KERNEL_ENTRY_DEVREDOP_NOFLOAT(devredop) \
  MSCCL_KERNEL_ENTRY_DEVREDOP_TYPE(devredop, int8_t), \
  MSCCL_KERNEL_ENTRY_DEVREDOP_TYPE(devredop, uint8_t), \
  MSCCL_KERNEL_ENTRY_DEVREDOP_TYPE(devredop, int32_t), \
  MSCCL_KERNEL_ENTRY_DEVREDOP_TYPE(devredop, uint32_t), \
  MSCCL_KERNEL_ENTRY_DEVREDOP_TYPE(devredop, int64_t), \
  MSCCL_KERNEL_ENTRY_DEVREDOP_TYPE(devredop, uint64_t), \
  MSCCL_KERNEL_ENTRY_DEVREDOP_NULL(), \
  MSCCL_KERNEL_ENTRY_DEVREDOP_NULL(), \
  MSCCL_KERNEL_ENTRY_DEVREDOP_NULL(), \
  MSCCL_KERNEL_ENTRY_DEVREDOP_NULL()

#define MSCCL_KERNEL_ENTRY() \
  MSCCL_KERNEL_ENTRY_DEVREDOP(Sum), \
  MSCCL_KERNEL_ENTRY_DEVREDOP(Prod), \
  MSCCL_KERNEL_ENTRY_DEVREDOP(Min), \
  MSCCL_KERNEL_ENTRY_DEVREDOP(Max), \
  MSCCL_KERNEL_ENTRY_DEVREDOP(PreMulSum), \
  MSCCL_KERNEL_ENTRY_DEVREDOP_NOFLOAT(SumPostDiv)

void* mscclKernelEntries[ncclNumDevRedOps * ncclNumTypes * NCCL_NUM_PROTOCOLS] = {
  MSCCL_KERNEL_ENTRY()
};

ncclResult_t mscclSetupKernel(const void* sendBuff, void* recvBuff, size_t count,
    ncclDataType_t dataType, ncclRedOp_t op, struct mscclAlgo* hostAlgo, struct mscclAlgo* devAlgo,
    ncclComm_t comm, hipStream_t stream) {
  mscclStatus& status = mscclGetStatus();

  if (status.lastStream != stream && status.lastStream != nullptr) {
    CUDACHECK(hipStreamWaitEvent(stream, comm->doneEvent, 0));
  }

  dim3 grid = {(uint32_t)hostAlgo->nBlocks, 1, 1};
  dim3 block = {NCCL_MAX_NTHREADS, 1, 1};
  ncclDevRedOpFull opFull;
  NCCLCHECK(hostToDevRedOp(&opFull, op, dataType, comm));

  mscclWork work;
  work.syncFlags = status.syncFlags;
  work.scratchBuffer = status.scratchBuffer;
  work.sendBuff = sendBuff;
  work.recvBuff = recvBuff;
  work.count = count * hostAlgo->sizeMultiplier; // count is sum of all ranks in MSCCL kernel
  work.redOpArg = opFull.scalarArg;
  work.workIndex = status.workIndex;
  work.nChunksPerLoop = hostAlgo->nChunksPerLoop;
  work.maxAllowedCount = status.maxAllowedCount;
  work.hasReduce = hostAlgo->hasReduce;
  work.redOpArgIsPtr = opFull.scalarArgIsPtr;

  void *args[3] = {&comm->devComm, &devAlgo, &work};
  void *func = mscclKernelEntries[(opFull.op * ncclNumTypes + dataType) * NCCL_NUM_PROTOCOLS + hostAlgo->protocol];
  CUDACHECK(hipExtLaunchKernel(func, grid, block, args, 0, stream, NULL, comm->doneEvent, 0));
  status.workIndex++;
  status.lastStream = stream;
  return ncclSuccess;
}
