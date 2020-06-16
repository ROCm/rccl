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
__device__ void ncclAllToAllKernel(struct CollectiveArgs* args) {
  const int tid = threadIdx.x;
  const int nthreads = args->coll.nThreads;
  const int nChannels = args->coll.nChannels;
  struct ncclDevComm* comm = args->comm;
  struct ncclChannel* channel = comm->channels+blockIdx.x;
  struct ncclRing* ring = &channel->ring;
  const ssize_t size = args->coll.count;
  const int nranks = comm->nRanks;
  const int bid = args->coll.bid;
  const int rank = ring->devUserRanks[0];
  const int stepSize = comm->buffSizes[NCCL_PROTO_SIMPLE] / (sizeof(T)*NCCL_STEPS);
  const int chunkSize = stepSize * ALLTOALL_CHUNKSTEPS;
  const int peersPerChan = (nChannels >= nranks ? 1 : DIVUP(nranks, nChannels));
  const ssize_t loopSize = (peersPerChan == 1 ? (nChannels/nranks)*(ssize_t)chunkSize : (ssize_t)chunkSize);

  // Compute pointers
  const T * __restrict__ thisInput = (const T*)args->sendbuff;
  T * __restrict__ thisOutput = (T*)args->recvbuff;

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
          const T* sendbuff = thisInput+chunkOffset+rank*size;
          T* recvbuff = thisOutput+chunkOffset+rank*size;
          // local copy
          ReduceOrCopyMulti<UNROLL, FUNC, T, 1, 1, 1, 1>(tid, nthreads, 1, &sendbuff, 1, &recvbuff, nelem);
        }
      }
      else {
        int peerSend = (rank+(blockIdx.x*peersPerChan)+i)%nranks;
        int peerRecv = (2*nranks+rank-((blockIdx.x*peersPerChan)%nranks)-(i%nranks))%nranks;
        ncclPrimitives<UNROLL, ALLTOALL_CHUNKSTEPS/ALLTOALL_SLICESTEPS, ALLTOALL_SLICESTEPS, T, 1, 1, 0, FUNC>
          prims(tid, nthreads, &peerRecv, &peerSend, NULL, stepSize, channel, comm, args->opCount);

        ssize_t send_offset = chunkOffset + peerSend*size;
        ssize_t recv_offset = chunkOffset + peerRecv*size;
        prims.send(thisInput+send_offset, nelem);
        prims.recv(thisOutput+recv_offset, nelem);
      }
    }
  }
}
