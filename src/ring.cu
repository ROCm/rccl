/*************************************************************************
 * Copyright (c) 2015-2018, NVIDIA CORPORATION. All rights reserved.
 * Modifications Copyright (c) 2019 Advanced Micro Devices, Inc. All rights reserved. 
 *
 * See LICENSE.txt for license information
 ************************************************************************/

#include "ring.h"
#include "param.h"

NCCL_PARAM(Buffsize, "BUFFSIZE", DEFAULT_BUFFER_SIZE_BYTES);

ncclResult_t initRing(struct ncclComm* comm, int ringid) {
  struct ncclRing* ring = comm->rings+ringid;
  ring->id = ringid;

  // Setup intermediate buffering
  ring->buffSize = ncclParamBuffsize();

  // attempt to allocate buffers in fine grain
  const int sendSize = ring->devMemSendSize = sizeof(struct ncclSendMem);
  struct ncclSendMem* sendMem;
  ncclCudaCalloc((char**)&sendMem, sendSize, true);
  ring->devMemSend = sendMem;

  const int recvSize = ring->devMemRecvSize = offsetof(struct ncclRecvMem, buff)+ring->buffSize;
  struct ncclRecvMem* recvMem;
  ncclCudaCalloc((char**)&recvMem, recvSize, true);
  ring->devMemRecv = recvMem;

  TRACE(NCCL_INIT,"sendMem %p size %d recvMem %p size %d", sendMem, sendSize, recvMem, recvSize);

  // Pre-configure send/recv pointers. Those are the default, they may change later.
  if (recvMem){
    ring->recv.conn.buff = recvMem->buff;
    ring->recv.conn.llBuff = recvMem->llBuff;
    ring->recv.conn.tail = &recvMem->tail;
    ring->recv.conn.opCount = &recvMem->opCount;
  } else {
    ring->recv.conn.buff = 0;
    ring->recv.conn.llBuff = 0;
    ring->recv.conn.tail = 0;
    ring->recv.conn.opCount = 0;
  }
  ring->recv.conn.direct = 0;

  if (sendMem) {
    ring->send.conn.head = &sendMem->head;
    ring->send.conn.llHead = &sendMem->llHead;
  } else {
    ring->send.conn.head = 0;
    ring->send.conn.llHead = 0;
  }
  ring->send.conn.direct = 0;
  ring->send.conn.llStep = 0;
  ring->send.conn.llLastCleaning = 0;

  // Ring index to user rank table.
  NCCLCHECK(ncclCudaCalloc(&ring->devUserRanks, comm->nRanks));
  NCCLCHECK(ncclCalloc(&ring->userRanks, comm->nRanks));

  // Per-ring operation list.
  NCCLCHECK(ncclCudaHostAlloc((void**)&ring->collectives, (void**)&ring->devCollectives, sizeof(struct ncclColl)*NCCL_MAX_OPS));
  return ncclSuccess;
}

ncclResult_t freeRing(struct ncclRing* ring) {
  // Intermediate buffering
  CUDACHECK(hipFree(ring->devMemSend));
  CUDACHECK(hipFree(ring->devMemRecv));

  // Index to rank table
  free(ring->userRanks);
  CUDACHECK(hipFree(ring->devUserRanks));

  // Operation list
  NCCLCHECK(ncclCudaHostFree(ring->collectives));

  // Free transport proxy resources
  if (ring->send.transportResources) NCCLCHECK(ring->send.transport->send.free(ring->send.transportResources));
  NCCLCHECK(transportDestroyProxy(&ring->send));
  if (ring->recv.transportResources) NCCLCHECK(ring->recv.transport->recv.free(ring->recv.transportResources));
  NCCLCHECK(transportDestroyProxy(&ring->recv));
  return ncclSuccess;
}
