/*************************************************************************
 * Copyright (c) 2015-2022, NVIDIA CORPORATION. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

#include "devcomm.h"
#include "collectives.h"
#include "primitives.h"

namespace {
  template<typename T, typename RedOp, typename Proto>
  __device__ __attribute__((noinline)) void runRing(ncclWorkElem *args) {
    const int tid = threadIdx.x;
    const int nthreads = args->header.nWarps*WARP_SIZE;
    const int bid = args->bid;
    const int nChannels = args->nChannels;
    ncclRing *ring = &ncclShmem->channel.ring;
    const ssize_t chunkSize = int(Proto::calcBytePerStep()/sizeof(T) * (Proto::Id == NCCL_PROTO_SIMPLE ? BROADCAST_CHUNKSTEPS : 1));
    const ssize_t minChunkSizeLL128 = int(nthreads*(Proto::calcBytePerGrain()/sizeof(T)));
    const ssize_t loopSize = nChannels*chunkSize;
    const ssize_t size = args->count;
    const int rank = ring->devUserRanks[0];
    const int nextRank = ring->devUserRanks[1];
#ifdef ENABLE_PROFILING
    auto devProf = ncclShmem->comm.devProf;
    uint64_t clk, t0 = 0ULL, ws;
    if (tid == 0) clk = __builtin_amdgcn_s_memrealtime();
#endif
    const int root = args->root;

    T *inputBuf = (T*)args->sendbuff;
    T *outputBuf = (T*)args->recvbuff;
    Primitives<T, RedOp, FanSymmetric<1>, 0, Proto, 0>
      prims(tid, nthreads, &ring->prev, &ring->next, inputBuf, outputBuf, args->redOpArg, args->connIndex << 16);

    for (ssize_t gridOffset = 0; gridOffset < size; gridOffset += loopSize) {
      ssize_t realChunkSize;
      if (Proto::Id == NCCL_PROTO_SIMPLE) {
        realChunkSize = min(chunkSize, divUp(size-gridOffset, nChannels));
        realChunkSize = roundUp(realChunkSize, nthreads*sizeof(uint64_t)/sizeof(T));
      }
      else if (Proto::Id == NCCL_PROTO_LL)
        realChunkSize = size-gridOffset < loopSize ? args->lastChunkSize : chunkSize;
      else if (Proto::Id == NCCL_PROTO_LL128)
        realChunkSize = min(chunkSize, divUp(size-gridOffset, nChannels*minChunkSizeLL128)*minChunkSizeLL128);
      realChunkSize = int(realChunkSize);

      ssize_t offset = gridOffset + int(bid*realChunkSize);
      int nelem = min(realChunkSize, size-offset);

      if (rank == root) {
        if (inputBuf == outputBuf) {
          INIT_COUNTER;
          prims.send(offset, nelem);
          ACCUMULATE_COUNTER(send);
        } else {
          INIT_COUNTER;
          prims.copySend(offset, offset, nelem);
          ACCUMULATE_COUNTER(copySend);
        }
      } else if (nextRank == root) {
        INIT_COUNTER;
        prims.recv(offset, nelem);
        ACCUMULATE_COUNTER(recv);
      } else {
        INIT_COUNTER;
        prims.recvCopySend(offset, nelem);
        ACCUMULATE_COUNTER(recvCopySend);
      }
    }
#ifdef ENABLE_PROFILING
    if (tid == 0) {
      struct ncclProfElem *elem = devProf.elems+args->opCount;
      elem->elem[blockIdx.x].total_cycle += (__builtin_amdgcn_s_memrealtime() - clk);
    }
#endif
  }
}

template<typename T, typename RedOp>
struct RunWorkElement<ncclFuncBroadcast, T, RedOp, NCCL_ALGO_RING, NCCL_PROTO_SIMPLE> {
  __device__ __attribute__((noinline)) void run(ncclWorkElem *args) {
    using Proto = ProtoSimple<BROADCAST_CHUNKSTEPS/BROADCAST_SLICESTEPS, BROADCAST_SLICESTEPS>;
    runRing<T, RedOp, Proto>(args);
  }
};

template<typename T, typename RedOp>
struct RunWorkElement<ncclFuncBroadcast, T, RedOp, NCCL_ALGO_RING, NCCL_PROTO_LL> {
  __device__ __attribute__((noinline)) void run(ncclWorkElem *args) {
    runRing<T, RedOp, ProtoLL>(args);
  }
};

template<typename T, typename RedOp>
struct RunWorkElement<ncclFuncBroadcast, T, RedOp, NCCL_ALGO_RING, NCCL_PROTO_LL128> {
  __device__ __attribute__((noinline)) void run(ncclWorkElem *args) {
    runRing<T, RedOp, ProtoLL128>(args);
  }
};
