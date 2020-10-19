/*************************************************************************
 * Copyright (c) 2015-2019, NVIDIA CORPORATION. All rights reserved.
 * Modifications Copyright (c) 2019-2020 Advanced Micro Devices, Inc. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

#include "devcomm.h"
#include "primitives.h"
#include "collectives.h"

template<int UNROLL, class FUNC, typename T>
__attribute__((noinline))
__device__ void ncclAllToAllvKernel(struct CollectiveArgs* args) {
  const int tid = threadIdx.x;
  const int nthreads = args->a2av.nThreads;
  const int nChannels = args->a2av.nChannels;
  struct ncclDevComm* comm = args->comm;
  struct ncclChannel* channel = comm->channels+blockIdx.x;
  struct ncclRing* ring = &channel->ring;
  const ssize_t typesize = args->a2av.count;
  const int nranks = comm->nRanks;
  const int bid = args->a2av.bid;
  const int rank = ring->devUserRanks[0];
  const int stepSize = comm->buffSizes[NCCL_PROTO_SIMPLE] / (sizeof(T)*NCCL_STEPS);
  const int chunkSize = stepSize * ALLTOALLV_CHUNKSTEPS;
  const int peersPerChan = DIVUP(nranks, nChannels);
  const ssize_t loopSize = (peersPerChan == 1 ? (nChannels/nranks)*(ssize_t)chunkSize : (ssize_t)chunkSize);

  // Compute pointers
  const T * __restrict__ thisInput = (const T*)args->sendbuff;
  T * __restrict__ thisOutput = (T*)args->recvbuff;

  size_t *sendcounts = args->a2av.extra;
  size_t *sdispls = args->a2av.extra + nranks;
  size_t *recvcounts = args->a2av.extra + nranks*2;
  size_t *rdispls = args->a2av.extra + nranks*3;
  ssize_t size = sendcounts[rank]*typesize;
  for (ssize_t gridOffset = 0; gridOffset < size; gridOffset += loopSize) {
    for (int i = 0; i < peersPerChan; i++) {
      if ((peersPerChan == 1 && blockIdx.x >= (nChannels/nranks)*nranks) ||
        (peersPerChan > 1 && blockIdx.x*peersPerChan+i >= nranks))
        continue;
      int realChunkSize = min(chunkSize, DIVUP(size-gridOffset, (peersPerChan == 1 ? (nChannels/nranks) : 1)));
      ALIGN_SIZE(realChunkSize, nthreads*sizeof(uint64_t)/sizeof(T));
      ssize_t chunkOffset = gridOffset + (peersPerChan == 1 ? (bid/nranks)*realChunkSize : 0);
      int nelem = min(realChunkSize, size-chunkOffset);
      if ((blockIdx.x*peersPerChan+i)%nranks == 0) {
        if (tid < nthreads && thisInput != thisOutput) {
          const T* sendbuff = thisInput+chunkOffset+sdispls[rank]*typesize;
          T* recvbuff = thisOutput+chunkOffset+rdispls[rank]*typesize;
          // local copy
          ReduceOrCopyMulti<UNROLL, FUNC, T, 1, 1, 1, 1>(tid, nthreads, 1, &sendbuff, 1, &recvbuff, nelem);
        }
      }
    }
  }

  for (int i = 0; i < peersPerChan; i++) {
    if ((peersPerChan == 1 && blockIdx.x >= (nChannels/nranks)*nranks) ||
      (peersPerChan > 1 && blockIdx.x*peersPerChan+i >= nranks))
      continue;
    if ((blockIdx.x*peersPerChan+i)%nranks != 0) {
      int nthreadsSplit = nthreads/2;
      int peerNone[2] = {-1,-1};
      if (tid < nthreadsSplit ) {
        int peerSend = (rank+(blockIdx.x*peersPerChan)+i)%nranks;
        ncclPrimitives<UNROLL, ALLTOALLV_CHUNKSTEPS/ALLTOALLV_SLICESTEPS, ALLTOALLV_SLICESTEPS, T, 2, 1, 0, FUNC>
          prims(tid, nthreadsSplit, peerNone, &peerSend, NULL, stepSize, channel, comm);
        size = sendcounts[peerSend]*typesize;
        for (ssize_t gridOffset = 0; gridOffset < size; gridOffset += loopSize) {
          int realChunkSize = min(chunkSize, DIVUP(size-gridOffset, (peersPerChan == 1 ? (nChannels/nranks) : 1)));
          ALIGN_SIZE(realChunkSize, nthreadsSplit*sizeof(uint64_t)/sizeof(T));
          ssize_t chunkOffset = gridOffset + (peersPerChan == 1 ? (bid/nranks)*realChunkSize : 0);
          int nelem = min(realChunkSize, size-chunkOffset);
          ssize_t send_offset = chunkOffset + sdispls[peerSend]*typesize;
          prims.send(thisInput+send_offset, nelem);
        }
      } else {
        int peerRecv = (2*nranks+rank-((blockIdx.x*peersPerChan)%nranks)-(i%nranks))%nranks;
        ncclPrimitives<UNROLL, ALLTOALLV_CHUNKSTEPS/ALLTOALLV_SLICESTEPS, ALLTOALLV_SLICESTEPS, T, 1, 2, 0, FUNC>
          prims(tid-nthreadsSplit, nthreads-nthreadsSplit, &peerRecv, peerNone, NULL, stepSize, channel, comm);
        size = recvcounts[peerRecv]*typesize;
        for (ssize_t gridOffset = 0; gridOffset < size; gridOffset += loopSize) {
          int realChunkSize = min(chunkSize, DIVUP(size-gridOffset, (peersPerChan == 1 ? (nChannels/nranks) : 1)));
          ALIGN_SIZE(realChunkSize, (nthreads-nthreadsSplit)*sizeof(uint64_t)/sizeof(T));
          ssize_t chunkOffset = gridOffset + (peersPerChan == 1 ? (bid/nranks)*realChunkSize : 0);
          int nelem = min(realChunkSize, size-chunkOffset);
          ssize_t recv_offset = chunkOffset + rdispls[peerRecv]*typesize;
          prims.recv(thisOutput+recv_offset, nelem);
        }
      }
    }
  }
}
