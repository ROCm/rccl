/*************************************************************************
 * Copyright (c) 2015-2022, NVIDIA CORPORATION. All rights reserved.
 * Modifications Copyright (c) 2019-2022 Advanced Micro Devices, Inc. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

#include "devcomm.h"
#include "collectives.h"
#include "primitives.h"

namespace {
  template<typename T, typename RedOp, typename Proto>
#if defined(USE_INDIRECT_FUNCTION_CALL) && !defined(__gfx940__) && !defined(__gfx941__) && !defined(__gfx942__)
  __device__ void runRing(ncclWorkElem *args) {
#else
  __device__ __attribute__((noinline)) void runRing(ncclWorkElem *args) {
#endif
    const int tid = threadIdx.x;
    const int nthreads = args->nWarps*WARP_SIZE;
    const int bid = args->bid;
    const int nChannels = args->nChannels;
    ncclRing *ring = &ncclShmem.channel.ring;
    const int *ringRanks = ring->userRanks;
    const ssize_t chunkSize = int(Proto::calcBytePerStep()/sizeof(T) * (Proto::Id == NCCL_PROTO_SIMPLE ? ALLGATHER_CHUNKSTEPS : 1));
    // We should not need the final /2 but it makes performance much, much smoother. Might be a bug somewhere.
    const ssize_t minChunkSizeLL128 = int(nthreads*(Proto::calcBytePerGrain()/sizeof(T))/2);
    const int nranks = ncclShmem.comm.nRanks;
    const ssize_t loopSize = nChannels*int(chunkSize);
    const ssize_t size = args->count;

#if defined(ENABLE_NPKIT)
    int npKitCtxIdx = bid;
#endif

#if defined(ENABLE_NPKIT) && defined(ENABLE_NPKIT_EVENT_TIME_SYNC_CPU)
    if (tid == 0) {
      uint64_t* cpuTimestamp = ncclShmem.comm.cpuTimestamp;
      NpKit::CollectGpuEvent(NPKIT_EVENT_TIME_SYNC_CPU, 0, 0, *cpuTimestamp,
          ncclShmem.comm.npKitEventCollectContexts + npKitCtxIdx);
    }
#endif

#if defined(ENABLE_NPKIT) && defined(ENABLE_NPKIT_EVENT_TIME_SYNC_GPU)
    if (tid == 0) {
      NpKit::CollectGpuEvent(NPKIT_EVENT_TIME_SYNC_GPU, 0, 0, NPKIT_GET_GPU_TIMESTAMP(),
          ncclShmem.comm.npKitEventCollectContexts + npKitCtxIdx);
    }
#endif

#if defined(ENABLE_NPKIT) && defined(ENABLE_NPKIT_EVENT_ALL_GATHER_RING_ENTRY)
    if (tid == 0) {
      NpKit::CollectGpuEvent(NPKIT_EVENT_ALL_GATHER_RING_ENTRY, size*sizeof(T), 0, NPKIT_GET_GPU_TIMESTAMP(),
          ncclShmem.comm.npKitEventCollectContexts + npKitCtxIdx);
    }
#endif

    T *inputBuf = (T*)args->sendbuff;
    T *outputBuf = (T*)args->recvbuff;
    Primitives<T, RedOp, FanSymmetric<1>, 0, Proto, 0> prims
      (tid, nthreads, &ring->prev, &ring->next, inputBuf, outputBuf, args->redOpArg, 0, args->connIndex, args->connIndex);

#if defined(ENABLE_NPKIT)
    if (tid == 0) {
      prims.npKitCtxIdx = npKitCtxIdx;
    }
#endif

    for (ssize_t gridOffset = 0; gridOffset < size; gridOffset += loopSize) {
      ssize_t realChunkSize;
      if (Proto::Id == NCCL_PROTO_SIMPLE) {
        realChunkSize = min(chunkSize, divUp(size-gridOffset,nChannels));
        realChunkSize = roundUp(realChunkSize, nthreads*sizeof(uint64_t)/sizeof(T));
      }
      else if (Proto::Id == NCCL_PROTO_LL)
        realChunkSize = size-gridOffset < loopSize ? args->lastChunkSize : chunkSize;
      else if (Proto::Id == NCCL_PROTO_LL128)
        realChunkSize = min(chunkSize, divUp(size-gridOffset, nChannels*minChunkSizeLL128)*minChunkSizeLL128);
      realChunkSize = int(realChunkSize);

      ssize_t chunkOffset = gridOffset + int(bid*realChunkSize);

      /////////////// begin AllGather steps ///////////////
      ssize_t offset;
      int nelem = min(realChunkSize, size-chunkOffset);
      int rankDest;

      // step 0: push data to next GPU
      rankDest = ringRanks[0];
      offset = chunkOffset + rankDest * size;

#if defined(ENABLE_NPKIT) && defined(ENABLE_NPKIT_EVENT_ALL_GATHER_RING_SEND_ENTRY)
      if (tid == 0) {
        NpKit::CollectGpuEvent(NPKIT_EVENT_ALL_GATHER_RING_SEND_ENTRY, nelem*sizeof(T), 0, NPKIT_GET_GPU_TIMESTAMP(),
            ncclShmem.comm.npKitEventCollectContexts + npKitCtxIdx);
        prims.npKitDataProcessTotalTime = 0;
      }
#endif

      if (inputBuf + chunkOffset == outputBuf + offset) { // In place
        prims.directSend(chunkOffset, offset, nelem);
      } else {
        prims.directCopySend(chunkOffset, offset, nelem);
      }

#if defined(ENABLE_NPKIT) && defined(ENABLE_NPKIT_EVENT_ALL_GATHER_RING_SEND_EXIT)
      if (tid == 0) {
        NpKit::CollectGpuEvent(NPKIT_EVENT_ALL_GATHER_RING_SEND_EXIT, nelem*sizeof(T), prims.npKitDataProcessTotalTime, NPKIT_GET_GPU_TIMESTAMP(),
            ncclShmem.comm.npKitEventCollectContexts + npKitCtxIdx);
      }
#endif

#if defined(ENABLE_NPKIT) && defined(ENABLE_NPKIT_EVENT_ALL_GATHER_RING_RECV_COPY_SEND_ENTRY)
      if (tid == 0 && nranks > 2) {
        NpKit::CollectGpuEvent(NPKIT_EVENT_ALL_GATHER_RING_RECV_COPY_SEND_ENTRY, nelem*(nranks-2)*sizeof(T), 0, NPKIT_GET_GPU_TIMESTAMP(),
            ncclShmem.comm.npKitEventCollectContexts + npKitCtxIdx);
        prims.npKitDataProcessTotalTime = 0;
      }
#endif

      // k-2 steps: copy to next GPU
      for (int j=1; j<nranks-1; ++j) {
        rankDest = ringRanks[nranks-j];
        offset = chunkOffset + rankDest * size;

        prims.directRecvCopySend(offset, nelem);
      }

#if defined(ENABLE_NPKIT) && defined(ENABLE_NPKIT_EVENT_ALL_GATHER_RING_RECV_COPY_SEND_EXIT)
      if (tid == 0 && nranks > 2) {
        NpKit::CollectGpuEvent(NPKIT_EVENT_ALL_GATHER_RING_RECV_COPY_SEND_EXIT, nelem*(nranks-2)*sizeof(T), prims.npKitDataProcessTotalTime, NPKIT_GET_GPU_TIMESTAMP(),
            ncclShmem.comm.npKitEventCollectContexts + npKitCtxIdx);
      }
#endif

      // Make final copy from buffer to dest.
      rankDest = ringRanks[1];
      offset = chunkOffset + rankDest * size;

#if defined(ENABLE_NPKIT) && defined(ENABLE_NPKIT_EVENT_ALL_GATHER_RING_DIRECT_RECV_ENTRY)
      if (tid == 0) {
        NpKit::CollectGpuEvent(NPKIT_EVENT_ALL_GATHER_RING_DIRECT_RECV_ENTRY, nelem*sizeof(T), 0, NPKIT_GET_GPU_TIMESTAMP(),
            ncclShmem.comm.npKitEventCollectContexts + npKitCtxIdx);
        prims.npKitDataProcessTotalTime = 0;
      }
#endif
      // Final wait/copy.
      prims.directRecv(offset, nelem);

#if defined(ENABLE_NPKIT) && defined(ENABLE_NPKIT_EVENT_ALL_GATHER_RING_DIRECT_RECV_EXIT)
      if (tid == 0) {
        NpKit::CollectGpuEvent(NPKIT_EVENT_ALL_GATHER_RING_DIRECT_RECV_EXIT, nelem*sizeof(T), prims.npKitDataProcessTotalTime, NPKIT_GET_GPU_TIMESTAMP(),
            ncclShmem.comm.npKitEventCollectContexts + npKitCtxIdx);
      }
#endif



    }
#if defined(ENABLE_NPKIT) && defined(ENABLE_NPKIT_EVENT_ALL_GATHER_RING_EXIT)
    if (tid == 0) {
      NpKit::CollectGpuEvent(NPKIT_EVENT_ALL_GATHER_RING_EXIT, size*sizeof(T), 0, NPKIT_GET_GPU_TIMESTAMP(),
          ncclShmem.comm.npKitEventCollectContexts + npKitCtxIdx);
    }
#endif
  }
}

template<typename T, typename RedOp>
struct RunWorkElement<ncclFuncAllGather, T, RedOp, NCCL_ALGO_RING, NCCL_PROTO_SIMPLE> {
  __device__ __forceinline__ void run(ncclWorkElem *args) {
    using Proto = ProtoSimple<ALLGATHER_CHUNKSTEPS/ALLGATHER_SLICESTEPS, ALLGATHER_SLICESTEPS>;
    runRing<T, RedOp, Proto>(args);
  }
};

template<typename T, typename RedOp>
struct RunWorkElement<ncclFuncAllGather, T, RedOp, NCCL_ALGO_RING, NCCL_PROTO_LL> {
  __device__ __forceinline__ void run(ncclWorkElem *args) {
    runRing<T, RedOp, ProtoLL>(args);
  }
};

template<typename T, typename RedOp>
struct RunWorkElement<ncclFuncAllGather, T, RedOp, NCCL_ALGO_RING, NCCL_PROTO_LL128> {
  __device__ __forceinline__ void run(ncclWorkElem *args) {
    runRing<T, RedOp, ProtoLL128>(args);
  }
};

template<typename T, typename RedOp>
struct RunWorkElement<ncclFuncAllGather, T, RedOp, NCCL_ALGO_NVLS, NCCL_PROTO_SIMPLE> {
  __device__ __forceinline__ void run(ncclWorkElem *args) {
    const int tid = threadIdx.x;
    const int bid = args->bid;
    const int nChannels = args->nChannels;
    struct ncclNvls* nvls = &ncclShmem.channel.nvls;
    const ssize_t chunkSize = int(args->lastChunkSize);
    const ssize_t size = args->count;
    const ssize_t loopSize = nChannels*chunkSize;

    const int nThreadsGather = 128;
    const int nThreadsBcast = 384 + WARP_SIZE;
    const int tidEndGather = nThreadsGather;
    const int tidEndBcast = tidEndGather + nThreadsBcast;

    using Proto = ProtoSimple<1, 1>;

    if (tid < tidEndGather) {
      // Gather
      Primitives<T, RedOp, FanAsymmetric<NCCL_MAX_NVLS_ARITY, 0>, /*Direct=*/0, Proto, 0>
        prims(tid, nThreadsGather, nvls->up, NULL, NULL, args->recvbuff,
           args->redOpArg, 0*Proto::MaxGroupWidth, 0, 0);
      for (ssize_t gridOffset = 0; gridOffset < size; gridOffset += loopSize) {
        ssize_t offset = gridOffset + bid*chunkSize;
        int nelem = min(chunkSize, size-offset);
        prims.gather(offset, nvls->nHeads*size, nelem, size, -1, 0);
      }
    } else if (tid < tidEndBcast) {
      // Bcast through NVLS
      Primitives<T, RedOp, FanAsymmetric<0, 1>, /*Direct=*/0, Proto, 0>
        prims(tid-tidEndGather, nThreadsBcast, NULL, &nvls->down, args->sendbuff, NULL,
           args->redOpArg, 3*Proto::MaxGroupWidth, 1, 1);
      for (ssize_t gridOffset = 0; gridOffset < size; gridOffset += loopSize) {
        ssize_t offset = gridOffset + bid*chunkSize;
        int nelem = min(chunkSize, size-offset);
        prims.send(offset, nelem);
      }
    }
  }
};
