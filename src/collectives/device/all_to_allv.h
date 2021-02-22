/*************************************************************************
 * Copyright (c) 2015-2019, NVIDIA CORPORATION. All rights reserved.
 * Modifications Copyright (c) 2019-2020 Advanced Micro Devices, Inc. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

#include "devcomm.h"
#include "primitives.h"
#include "collectives.h"

template<class FUNC, typename T, int UNROLL>
class ncclFunction<ncclFuncAllToAllv, NCCL_ALGO_RING, NCCL_PROTO_SIMPLE, FUNC, T, UNROLL> {
  public:
  __device__ __attribute__((noinline)) void run(struct ncclWorkElem* args) {
    const int tid = threadIdx.x;
    const int nthreads = args->nThreads;
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

    size_t* params = channel->a2avParams + nranks*4*args->index;
    size_t *sendcounts = params;
    size_t *sdispls = params + nranks;
    size_t *recvcounts = params + nranks*2;
    size_t *rdispls = params + nranks*3;
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
        if (tid < nthreadsSplit ) {
          int peerSend = (rank+(blockIdx.x*peersPerChan)+i)%nranks;
          ncclPrimitives<UNROLL, ALLTOALLV_CHUNKSTEPS/ALLTOALLV_SLICESTEPS, ALLTOALLV_SLICESTEPS, T, 0, 1, 0, FUNC>
            prims(tid, nthreadsSplit, NULL, &peerSend, NULL, stepSize, channel, comm, ncclShmem->ptrs, 0);
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
          ncclPrimitives<UNROLL, ALLTOALLV_CHUNKSTEPS/ALLTOALLV_SLICESTEPS, ALLTOALLV_SLICESTEPS, T, 1, 0, 0, FUNC>
            prims(tid-nthreadsSplit, nthreads-nthreadsSplit, &peerRecv, NULL, NULL, stepSize, channel, comm, ncclShmem->ptrs, 1);
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
};