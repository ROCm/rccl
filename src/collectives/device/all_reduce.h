/*************************************************************************
 * Copyright (c) 2015-2022, NVIDIA CORPORATION. All rights reserved.
 * Modifications Copyright (c) 2019-2022 Advanced Micro Devices, Inc. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

#include "devcomm.h"
#include "collectives.h"
#include "primitives.h"

#if defined(ENABLE_NPKIT)
#include "npkit/npkit.h"
#endif

namespace {
  template<typename T, typename RedOp, typename Proto>
  __device__ __attribute__((noinline)) void runRing(ncclWorkElem *args) {
    const int tid = threadIdx.x;
    const int nthreads = args->nWarps*WARP_SIZE;
    const int bid = args->bid;
    const int nChannels = args->nChannels;
    ncclRing *ring = &ncclShmem.channel.ring;
    int ringIx = ring->index;
    const ssize_t chunkSize = int(Proto::calcBytePerStep()/sizeof(T) * (Proto::Id == NCCL_PROTO_SIMPLE ? ALLREDUCE_CHUNKSTEPS : 1));
    const int nranks = ncclShmem.comm.nRanks;
    const ssize_t loopSize = nChannels*nranks*chunkSize;
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
      NpKit::CollectGpuEvent(NPKIT_EVENT_TIME_SYNC_GPU, 0, 0, __builtin_amdgcn_s_memrealtime(),
          ncclShmem.comm.npKitEventCollectContexts + npKitCtxIdx);
    }
#endif

#if defined(ENABLE_NPKIT) && defined(ENABLE_NPKIT_EVENT_ALL_REDUCE_RING_ENTRY)
    if (tid == 0) {
      NpKit::CollectGpuEvent(NPKIT_EVENT_ALL_REDUCE_RING_ENTRY, size*sizeof(T), 0, __builtin_amdgcn_s_memrealtime(),
          ncclShmem.comm.npKitEventCollectContexts + npKitCtxIdx);
    }
#endif

    int minChunkSize;
    if (Proto::Id == NCCL_PROTO_LL)
      minChunkSize = nthreads*(Proto::calcBytePerGrain()/sizeof(T));
    if (Proto::Id == NCCL_PROTO_LL128) {
      // We should not need the final /2 but it makes performance much, much smoother. Might be a bug somewhere.
      minChunkSize = nthreads*(Proto::calcBytePerGrain()/sizeof(T))/2;
    }

    Primitives<T, RedOp, FanSymmetric<1>, 0, Proto, 0> prims
      (tid, nthreads, &ring->prev, &ring->next, args->sendbuff, args->recvbuff, args->redOpArg, args->connIndex << 16);

#if defined(ENABLE_NPKIT)
    if (tid == 0) {
      prims.npKitCtxIdx = npKitCtxIdx;
    }
#endif

    for (ssize_t gridOffset = 0; gridOffset < size; gridOffset += loopSize) {
      ssize_t realChunkSize;
      if (Proto::Id == NCCL_PROTO_SIMPLE) {
        realChunkSize = min(chunkSize, divUp(size-gridOffset, nChannels*nranks));
        realChunkSize = roundUp(realChunkSize, nthreads*sizeof(uint64_t)/sizeof(T));
      }
      else
        realChunkSize = min(chunkSize, divUp(size-gridOffset, nChannels*nranks*minChunkSize)*minChunkSize);
      realChunkSize = int(realChunkSize);

      auto calcOffset = [&]__device__(int chunk)->ssize_t {
        if (Proto::Id == NCCL_PROTO_SIMPLE)
          return gridOffset + bid*nranks*realChunkSize + chunk*realChunkSize;
        else
          return gridOffset + (chunk*nChannels + bid)*realChunkSize;
      };
      auto modRanks = [&]__device__(int r)->int {
        return r - (r >= nranks ? nranks : 0);
      };

      ssize_t offset;
      int nelem;
      int chunk;

      // step 0: push data to next GPU
      chunk = modRanks(ringIx + nranks-1);
      offset = calcOffset(chunk);
      nelem = min(realChunkSize, size-offset);

#if defined(ENABLE_NPKIT) && defined(ENABLE_NPKIT_EVENT_ALL_REDUCE_RING_SEND_ENTRY)
      if (tid == 0) {
        NpKit::CollectGpuEvent(NPKIT_EVENT_ALL_REDUCE_RING_SEND_ENTRY, nelem*sizeof(T), 0, __builtin_amdgcn_s_memrealtime(),
            ncclShmem.comm.npKitEventCollectContexts + npKitCtxIdx);
        prims.npKitDataProcessTotalTime = 0;
      }
#endif

      prims.send(offset, nelem);

#if defined(ENABLE_NPKIT) && defined(ENABLE_NPKIT_EVENT_ALL_REDUCE_RING_SEND_EXIT)
      if (tid == 0) {
        NpKit::CollectGpuEvent(NPKIT_EVENT_ALL_REDUCE_RING_SEND_EXIT, nelem*sizeof(T), prims.npKitDataProcessTotalTime, __builtin_amdgcn_s_memrealtime(),
            ncclShmem.comm.npKitEventCollectContexts + npKitCtxIdx);
      }
#endif

      // k-2 steps: reduce and copy to next GPU

#if defined(ENABLE_NPKIT) && defined(ENABLE_NPKIT_EVENT_ALL_REDUCE_RING_RECV_REDUCE_SEND_ENTRY)
      if (tid == 0 && nranks > 2) {
        NpKit::CollectGpuEvent(NPKIT_EVENT_ALL_REDUCE_RING_RECV_REDUCE_SEND_ENTRY, nelem*(nranks-2)*sizeof(T), 0, __builtin_amdgcn_s_memrealtime(),
            ncclShmem.comm.npKitEventCollectContexts + npKitCtxIdx);
        prims.npKitDataProcessTotalTime = 0;
      }
#endif

      for (int j=2; j<nranks; ++j) {
        chunk = modRanks(ringIx + nranks-j);
        offset = calcOffset(chunk);
        nelem = min(realChunkSize, size-offset);
        prims.recvReduceSend(offset, nelem);
      }

#if defined(ENABLE_NPKIT) && defined(ENABLE_NPKIT_EVENT_ALL_REDUCE_RING_RECV_REDUCE_SEND_EXIT)
      if (tid == 0 && nranks > 2) {
        NpKit::CollectGpuEvent(NPKIT_EVENT_ALL_REDUCE_RING_RECV_REDUCE_SEND_EXIT, nelem*(nranks-2)*sizeof(T), prims.npKitDataProcessTotalTime, __builtin_amdgcn_s_memrealtime(),
            ncclShmem.comm.npKitEventCollectContexts + npKitCtxIdx);
      }
#endif

      // step k-1: reduce this buffer and data, which will produce the final
      // result that we store in this data and push to the next GPU
      chunk = ringIx + 0;
      offset = calcOffset(chunk);
      nelem = min(realChunkSize, size-offset);

#if defined(ENABLE_NPKIT) && defined(ENABLE_NPKIT_EVENT_ALL_REDUCE_RING_DIRECT_RECV_REDUCE_COPY_SEND_ENTRY)
      if (tid == 0) {
        NpKit::CollectGpuEvent(NPKIT_EVENT_ALL_REDUCE_RING_DIRECT_RECV_REDUCE_COPY_SEND_ENTRY, nelem*sizeof(T), 0, __builtin_amdgcn_s_memrealtime(),
            ncclShmem.comm.npKitEventCollectContexts + npKitCtxIdx);
        prims.npKitDataProcessTotalTime = 0;
      }
#endif

      prims.directRecvReduceCopySend(offset, offset, offset, nelem, /*postOp=*/true);

#if defined(ENABLE_NPKIT) && defined(ENABLE_NPKIT_EVENT_ALL_REDUCE_RING_DIRECT_RECV_REDUCE_COPY_SEND_EXIT)
      if (tid == 0) {
        NpKit::CollectGpuEvent(NPKIT_EVENT_ALL_REDUCE_RING_DIRECT_RECV_REDUCE_COPY_SEND_EXIT, nelem*sizeof(T), prims.npKitDataProcessTotalTime, __builtin_amdgcn_s_memrealtime(),
            ncclShmem.comm.npKitEventCollectContexts + npKitCtxIdx);
      }
#endif

#if defined(ENABLE_NPKIT) && defined(ENABLE_NPKIT_EVENT_ALL_REDUCE_RING_DIRECT_RECV_COPY_SEND_ENTRY)
      if (tid == 0 && nranks > 2) {
        NpKit::CollectGpuEvent(NPKIT_EVENT_ALL_REDUCE_RING_DIRECT_RECV_COPY_SEND_ENTRY, nelem*(nranks-2)*sizeof(T), 0, __builtin_amdgcn_s_memrealtime(),
            ncclShmem.comm.npKitEventCollectContexts + npKitCtxIdx);
        prims.npKitDataProcessTotalTime = 0;
      }
#endif

      // k-2 steps: copy to next GPU
      for (int j=1; j<nranks-1; ++j) {
        chunk = modRanks(ringIx + nranks-j);
        offset = calcOffset(chunk);
        nelem = min(realChunkSize, size-offset);
        prims.directRecvCopySend(offset, offset, nelem);
      }

#if defined(ENABLE_NPKIT) && defined(ENABLE_NPKIT_EVENT_ALL_REDUCE_RING_DIRECT_RECV_COPY_SEND_EXIT)
      if (tid == 0 && nranks > 2) {
        NpKit::CollectGpuEvent(NPKIT_EVENT_ALL_REDUCE_RING_DIRECT_RECV_COPY_SEND_EXIT, nelem*(nranks-2)*sizeof(T), prims.npKitDataProcessTotalTime, __builtin_amdgcn_s_memrealtime(),
            ncclShmem.comm.npKitEventCollectContexts + npKitCtxIdx);
      }
#endif

#if defined(ENABLE_NPKIT) && defined(ENABLE_NPKIT_EVENT_ALL_REDUCE_RING_DIRECT_RECV_ENTRY)
      if (tid == 0) {
        NpKit::CollectGpuEvent(NPKIT_EVENT_ALL_REDUCE_RING_DIRECT_RECV_ENTRY, nelem*sizeof(T), 0, __builtin_amdgcn_s_memrealtime(),
            ncclShmem.comm.npKitEventCollectContexts + npKitCtxIdx);
        prims.npKitDataProcessTotalTime = 0;
      }
#endif

      // Make final copy from buffer to dest.
      chunk = modRanks(ringIx + 1);
      offset = calcOffset(chunk);
      nelem = min(realChunkSize, size-offset);
      prims.directRecv(offset, nelem);

#if defined(ENABLE_NPKIT) && defined(ENABLE_NPKIT_EVENT_ALL_REDUCE_RING_DIRECT_RECV_EXIT)
      if (tid == 0) {
        NpKit::CollectGpuEvent(NPKIT_EVENT_ALL_REDUCE_RING_DIRECT_RECV_EXIT, nelem*sizeof(T), prims.npKitDataProcessTotalTime, __builtin_amdgcn_s_memrealtime(),
            ncclShmem.comm.npKitEventCollectContexts + npKitCtxIdx);
      }
#endif

    }

#if defined(ENABLE_NPKIT) && defined(ENABLE_NPKIT_EVENT_ALL_REDUCE_RING_EXIT)
    if (tid == 0) {
      NpKit::CollectGpuEvent(NPKIT_EVENT_ALL_REDUCE_RING_EXIT, size*sizeof(T), 0, __builtin_amdgcn_s_memrealtime(),
          ncclShmem.comm.npKitEventCollectContexts + npKitCtxIdx);
    }
#endif

  }

  template<typename T, typename RedOp, typename Proto>
  __device__ __attribute__((noinline)) void runTreeUpDown(ncclWorkElem *args) {
    const int tid = threadIdx.x;
    const int nthreads = args->nWarps*WARP_SIZE;
    const int bid = args->bid;
    const int nChannels = args->nChannels;
    ncclTree *tree = &ncclShmem.channel.tree;
    ssize_t chunkSize = int(
      Proto::Id == NCCL_PROTO_SIMPLE ? args->lastChunkSize
                   /* LL & LL128 */  : Proto::calcBytePerStep()/sizeof(T));
    const ssize_t minChunkSize = int(
      Proto::Id == NCCL_PROTO_SIMPLE ? nthreads*8*(sizeof(uint64_t)/sizeof(T))
                   /* LL & LL128 */  : nthreads*(Proto::calcBytePerGrain()/sizeof(T)));
    const ssize_t loopSize = int(nChannels*chunkSize);
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
      NpKit::CollectGpuEvent(NPKIT_EVENT_TIME_SYNC_GPU, 0, 0, __builtin_amdgcn_s_memrealtime(),
          ncclShmem.comm.npKitEventCollectContexts + npKitCtxIdx);
    }
#endif

#if defined(ENABLE_NPKIT) && defined(ENABLE_NPKIT_EVENT_ALL_REDUCE_TREE_UPDOWN_ENTRY)
    if (tid == 0) {
      NpKit::CollectGpuEvent(NPKIT_EVENT_ALL_REDUCE_TREE_UPDOWN_ENTRY, size*sizeof(T), 0, __builtin_amdgcn_s_memrealtime(),
          ncclShmem.comm.npKitEventCollectContexts + npKitCtxIdx);
    }
#endif

    if (loopSize > size)
      chunkSize = divUp((int)size, int(nChannels*minChunkSize))*int(minChunkSize);

    { // Reduce : max number of recv is 3, max number of send is 1 (binary tree + local)
      Primitives<T, RedOp, FanAsymmetric<NCCL_MAX_DEV_ARITY, 1>, /*Direct=*/0, Proto, 0> prims
        (tid, nthreads, tree->down, &tree->up, args->sendbuff, args->recvbuff, args->redOpArg);

#if defined(ENABLE_NPKIT)
      if (tid == 0) {
        prims.npKitCtxIdx = npKitCtxIdx;
      }
#endif

#if defined(ENABLE_NPKIT) && defined(ENABLE_NPKIT_EVENT_ALL_REDUCE_TREE_UPDOWN_REDUCE_ENTRY)
      if (tid == 0) {
        NpKit::CollectGpuEvent(NPKIT_EVENT_ALL_REDUCE_TREE_UPDOWN_REDUCE_ENTRY, size*sizeof(T), 0, __builtin_amdgcn_s_memrealtime(),
            ncclShmem.comm.npKitEventCollectContexts + npKitCtxIdx);
        prims.npKitDataProcessTotalTime = 0;
      }
#endif

      if (tree->up == -1) {
        for (ssize_t gridOffset = 0; gridOffset < size; gridOffset += loopSize) {
          ssize_t offset = gridOffset + bid*int(chunkSize);
          int nelem = min(chunkSize, size-offset);
          prims.recvReduceCopy(offset, offset, nelem, /*postOp=*/true);
        }
      }
      else if (tree->down[0] == -1) {
        for (ssize_t gridOffset = 0; gridOffset < size; gridOffset += loopSize) {
          ssize_t offset = gridOffset + bid*int(chunkSize);
          int nelem = min(chunkSize, size-offset);
          prims.send(offset, nelem);
        }
      }
      else {
        for (ssize_t gridOffset = 0; gridOffset < size; gridOffset += loopSize) {
          ssize_t offset = gridOffset + bid*int(chunkSize);
          int nelem = min(chunkSize, size-offset);
          prims.recvReduceSend(offset, nelem);
        }
      }

#if defined(ENABLE_NPKIT) && defined(ENABLE_NPKIT_EVENT_ALL_REDUCE_TREE_UPDOWN_REDUCE_EXIT)
      if (tid == 0) {
        NpKit::CollectGpuEvent(NPKIT_EVENT_ALL_REDUCE_TREE_UPDOWN_REDUCE_EXIT, size*sizeof(T), prims.npKitDataProcessTotalTime, __builtin_amdgcn_s_memrealtime(),
            ncclShmem.comm.npKitEventCollectContexts + npKitCtxIdx);
      }
#endif

    }

    { // Broadcast : max number of recv is 1, max number of send is 3 (binary tree + local)
      Primitives<T, RedOp, FanAsymmetric<1, NCCL_MAX_DEV_ARITY>, /*Direct=*/0, Proto, 0> prims
        (tid, nthreads, &tree->up, tree->down, args->sendbuff, args->recvbuff, args->redOpArg);

#if defined(ENABLE_NPKIT)
      if (tid == 0) {
        prims.npKitCtxIdx = npKitCtxIdx;
      }
#endif

#if defined(ENABLE_NPKIT) && defined(ENABLE_NPKIT_EVENT_ALL_REDUCE_TREE_UPDOWN_BROADCAST_ENTRY)
      if (tid == 0) {
        NpKit::CollectGpuEvent(NPKIT_EVENT_ALL_REDUCE_TREE_UPDOWN_BROADCAST_ENTRY, size*sizeof(T), 0, __builtin_amdgcn_s_memrealtime(),
            ncclShmem.comm.npKitEventCollectContexts + npKitCtxIdx);
        prims.npKitDataProcessTotalTime = 0;
      }
#endif

      if (tree->up == -1) {
        for (ssize_t gridOffset = 0; gridOffset < size; gridOffset += loopSize) {
          ssize_t offset = gridOffset + bid*int(chunkSize);
          int nelem = min(chunkSize, size-offset);
          prims.directSendFromOutput(offset, offset, nelem);
        }
      }
      else if (tree->down[0] == -1) {
        for (ssize_t gridOffset = 0; gridOffset < size; gridOffset += loopSize) {
          ssize_t offset = gridOffset + bid*int(chunkSize);
          int nelem = min(chunkSize, size-offset);
          prims.directRecv(offset, nelem);
        }
      }
      else {
        for (ssize_t gridOffset = 0; gridOffset < size; gridOffset += loopSize) {
          ssize_t offset = gridOffset + bid*int(chunkSize);
          int nelem = min(chunkSize, size-offset);
          prims.directRecvCopySend(offset, offset, nelem);
        }
      }

#if defined(ENABLE_NPKIT) && defined(ENABLE_NPKIT_EVENT_ALL_REDUCE_TREE_UPDOWN_BROADCAST_EXIT)
      if (tid == 0) {
        NpKit::CollectGpuEvent(NPKIT_EVENT_ALL_REDUCE_TREE_UPDOWN_BROADCAST_EXIT, size*sizeof(T), prims.npKitDataProcessTotalTime, __builtin_amdgcn_s_memrealtime(),
            ncclShmem.comm.npKitEventCollectContexts + npKitCtxIdx);
      }
#endif

    }

#if defined(ENABLE_NPKIT) && defined(ENABLE_NPKIT_EVENT_ALL_REDUCE_TREE_UPDOWN_EXIT)
    if (tid == 0) {
      NpKit::CollectGpuEvent(NPKIT_EVENT_ALL_REDUCE_TREE_UPDOWN_EXIT, size*sizeof(T), 0, __builtin_amdgcn_s_memrealtime(),
          ncclShmem.comm.npKitEventCollectContexts + npKitCtxIdx);
    }
#endif

  }

  template<typename T, typename RedOp, typename Proto>
  __device__ __attribute__((noinline)) void runTreeSplit(ncclWorkElem *args) {
    const int tid = threadIdx.x;
    const int nthreads = args->nWarps*WARP_SIZE;
    const int bid = args->bid;
    const int nChannels = args->nChannels;
    ncclTree *tree = (args->pad_0 == 2) ? &ncclShmem.channel.binTree : &ncclShmem.channel.tree;
    ssize_t chunkSize = int(
      Proto::Id != NCCL_PROTO_LL ? args->lastChunkSize
                                 : Proto::calcBytePerStep()/sizeof(T));
    const ssize_t minChunkSize = int(
      Proto::Id == NCCL_PROTO_SIMPLE ? nthreads*8*(sizeof(uint64_t)/sizeof(T)) :
      Proto::Id == NCCL_PROTO_LL     ? nthreads*(Proto::calcBytePerGrain()/sizeof(T))
                   /* LL128 */       : nthreads*(Proto::calcBytePerGrain()/sizeof(T))/8);
    const ssize_t loopSize = int(nChannels*chunkSize);
    const ssize_t size = args->count;
    int nthreadsSplit;
    if (Proto::Id == NCCL_PROTO_SIMPLE) {
      nthreadsSplit = nthreads/2;
      if (nthreadsSplit >= 256) nthreadsSplit += 64;
    } else { // LL & LL128
      // Receiving from up to 3 sources is more compute intensive than sending
      // to 3 dests. Use 70% for reduce and 30% for bcast.
      nthreadsSplit = (nthreads*7/(10*WARP_SIZE))*WARP_SIZE;
    }

#if defined(ENABLE_NPKIT)
    bool isNpKitThread = false;
    int npKitCtxIdx = 0;
    if (threadIdx.x == 0) {
      isNpKitThread = true;
      npKitCtxIdx = bid * 2;
    } else if (tree->up != -1 && threadIdx.x == nthreadsSplit) {
      isNpKitThread = true;
      npKitCtxIdx = bid * 2 + 1;
    }
#endif

#if defined(ENABLE_NPKIT) && defined(ENABLE_NPKIT_EVENT_TIME_SYNC_CPU)
    if (isNpKitThread) {
      uint64_t* cpuTimestamp = ncclShmem.comm.cpuTimestamp;
      NpKit::CollectGpuEvent(NPKIT_EVENT_TIME_SYNC_CPU, 0, 0, *cpuTimestamp,
          ncclShmem.comm.npKitEventCollectContexts + npKitCtxIdx);
    }
#endif

#if defined(ENABLE_NPKIT) && defined(ENABLE_NPKIT_EVENT_TIME_SYNC_GPU)
    if (isNpKitThread) {
      NpKit::CollectGpuEvent(NPKIT_EVENT_TIME_SYNC_GPU, 0, 0, __builtin_amdgcn_s_memrealtime(),
          ncclShmem.comm.npKitEventCollectContexts + npKitCtxIdx);
    }
#endif

#if defined(ENABLE_NPKIT) && defined(ENABLE_NPKIT_EVENT_ALL_REDUCE_TREE_SPLIT_ENTRY)
    if (isNpKitThread) {
      NpKit::CollectGpuEvent(NPKIT_EVENT_ALL_REDUCE_TREE_SPLIT_ENTRY, size*sizeof(T), 0, __builtin_amdgcn_s_memrealtime(),
          ncclShmem.comm.npKitEventCollectContexts + npKitCtxIdx);
    }
#endif

    if (loopSize > size)
      chunkSize = divUp((int)size, nChannels*int(minChunkSize))*int(minChunkSize);

    if (tree->up == -1) {
      // Reduce and broadcast. Max number of recv is 3, max number of send is 3
      Primitives<T, RedOp, FanSymmetric<NCCL_MAX_DEV_ARITY>, /*Direct=*/0, Proto, 0>
        prims(tid, nthreads, tree->down, tree->down, args->sendbuff, args->recvbuff, args->redOpArg);

#if defined(ENABLE_NPKIT)
      if (isNpKitThread) {
        prims.npKitCtxIdx = npKitCtxIdx;
      }
#endif

#if defined(ENABLE_NPKIT) && defined(ENABLE_NPKIT_EVENT_ALL_REDUCE_TREE_SPLIT_REDUCE_BROADCAST_ENTRY)
      if (isNpKitThread) {
        NpKit::CollectGpuEvent(NPKIT_EVENT_ALL_REDUCE_TREE_SPLIT_REDUCE_BROADCAST_ENTRY, size*sizeof(T), 0, __builtin_amdgcn_s_memrealtime(),
            ncclShmem.comm.npKitEventCollectContexts + npKitCtxIdx);
        prims.npKitDataProcessTotalTime = 0;
      }
#endif

      for (ssize_t gridOffset = 0; gridOffset < size; gridOffset += loopSize) {
        ssize_t offset = gridOffset + bid*int(chunkSize);
        int nelem = min(chunkSize, size-offset);
        prims.directRecvReduceCopySend(offset, offset, offset, nelem, /*doPost=*/true);
      }

#if defined(ENABLE_NPKIT) && defined(ENABLE_NPKIT_EVENT_ALL_REDUCE_TREE_SPLIT_REDUCE_BROADCAST_EXIT)
      if (isNpKitThread) {
        NpKit::CollectGpuEvent(NPKIT_EVENT_ALL_REDUCE_TREE_SPLIT_REDUCE_BROADCAST_EXIT, size*sizeof(T), prims.npKitDataProcessTotalTime, __builtin_amdgcn_s_memrealtime(),
            ncclShmem.comm.npKitEventCollectContexts + npKitCtxIdx);
      }
#endif

    }
    else if (tid < nthreadsSplit) {
      /* Reduce up. Max number of recv is 3, max number of send is 1 (binary tree + local).
       * Why Direct=1????
       * Answer: Because despite not performing any direct operations, the ctor
       * must assume Direct so that it can exchange direct pointers with remote ctors
       * that are Direct, otherwise it hangs. A cleaner solution would be to seperate
       * into DirectRecv and DirectSend capabilities, this ctor would have both=0,
       * but the ctor above for tree roots would be DirectRecv=0 DirectSend=1.
       */
      Primitives<T, RedOp, FanAsymmetric<NCCL_MAX_DEV_ARITY, 1>, /*Direct=*/0, Proto, 0>
        prims(tid, nthreadsSplit, tree->down, &tree->up, args->sendbuff, args->recvbuff, args->redOpArg, 0*Proto::MaxGroupWidth);

#if defined(ENABLE_NPKIT)
      if (isNpKitThread) {
        prims.npKitCtxIdx = npKitCtxIdx;
      }
#endif

#if defined(ENABLE_NPKIT) && defined(ENABLE_NPKIT_EVENT_ALL_REDUCE_TREE_SPLIT_REDUCE_ENTRY)
      if (isNpKitThread) {
        NpKit::CollectGpuEvent(NPKIT_EVENT_ALL_REDUCE_TREE_SPLIT_REDUCE_ENTRY, size*sizeof(T), 0, __builtin_amdgcn_s_memrealtime(),
            ncclShmem.comm.npKitEventCollectContexts + npKitCtxIdx);
        prims.npKitDataProcessTotalTime = 0;
      }
#endif

      if (tree->down[0] == -1) {
        for (ssize_t gridOffset = 0; gridOffset < size; gridOffset += loopSize) {
          ssize_t offset = gridOffset + bid*int(chunkSize);
          int nelem = min(chunkSize, size-offset);
          prims.send(offset, nelem);
        }
      }
      else {
        for (ssize_t gridOffset = 0; gridOffset < size; gridOffset += loopSize) {
          ssize_t offset = gridOffset + bid*int(chunkSize);
          int nelem = min(chunkSize, size-offset);
          prims.recvReduceSend(offset, nelem);
        }
      }

#if defined(ENABLE_NPKIT) && defined(ENABLE_NPKIT_EVENT_ALL_REDUCE_TREE_SPLIT_REDUCE_EXIT)
      if (isNpKitThread) {
        NpKit::CollectGpuEvent(NPKIT_EVENT_ALL_REDUCE_TREE_SPLIT_REDUCE_EXIT, size*sizeof(T), prims.npKitDataProcessTotalTime, __builtin_amdgcn_s_memrealtime(),
            ncclShmem.comm.npKitEventCollectContexts + npKitCtxIdx);
      }
#endif

    }
    else {
      // Broadcast down. Max number of recv is 1, max number of send is 3 (binary tree + local)
      Primitives<T, RedOp, FanAsymmetric<1, NCCL_MAX_DEV_ARITY>, /*Direct=*/0, Proto, 0>
        prims(tid-nthreadsSplit, nthreads-nthreadsSplit, &tree->up, tree->down, args->sendbuff, args->recvbuff, args->redOpArg, 1*Proto::MaxGroupWidth);

#if defined(ENABLE_NPKIT)
      if (isNpKitThread) {
        prims.npKitCtxIdx = npKitCtxIdx;
      }
#endif

#if defined(ENABLE_NPKIT) && defined(ENABLE_NPKIT_EVENT_ALL_REDUCE_TREE_SPLIT_BROADCAST_ENTRY)
      if (isNpKitThread) {
        NpKit::CollectGpuEvent(NPKIT_EVENT_ALL_REDUCE_TREE_SPLIT_BROADCAST_ENTRY, size*sizeof(T), 0, __builtin_amdgcn_s_memrealtime(),
            ncclShmem.comm.npKitEventCollectContexts + npKitCtxIdx);
        prims.npKitDataProcessTotalTime = 0;
      }
#endif

      if (tree->down[0] == -1) {
        for (ssize_t gridOffset = 0; gridOffset < size; gridOffset += loopSize) {
          ssize_t offset = gridOffset + bid*int(chunkSize);
          int nelem = min(chunkSize, size-offset);
          prims.directRecv(offset, nelem);
        }
      }
      else {
        for (ssize_t gridOffset = 0; gridOffset < size; gridOffset += loopSize) {
          ssize_t offset = gridOffset + bid*int(chunkSize);
          int nelem = min(chunkSize, size-offset);
          prims.directRecvCopySend(offset, offset, nelem);
        }
      }

#if defined(ENABLE_NPKIT) && defined(ENABLE_NPKIT_EVENT_ALL_REDUCE_TREE_SPLIT_BROADCAST_EXIT)
      if (isNpKitThread) {
        NpKit::CollectGpuEvent(NPKIT_EVENT_ALL_REDUCE_TREE_SPLIT_BROADCAST_EXIT, size*sizeof(T), prims.npKitDataProcessTotalTime, __builtin_amdgcn_s_memrealtime(),
            ncclShmem.comm.npKitEventCollectContexts + npKitCtxIdx);
      }
#endif

    }

#if defined(ENABLE_NPKIT) && defined(ENABLE_NPKIT_EVENT_ALL_REDUCE_TREE_SPLIT_EXIT)
    if (isNpKitThread) {
      NpKit::CollectGpuEvent(NPKIT_EVENT_ALL_REDUCE_TREE_SPLIT_EXIT, size*sizeof(T), 0, __builtin_amdgcn_s_memrealtime(),
          ncclShmem.comm.npKitEventCollectContexts + npKitCtxIdx);
    }
#endif

  }
}

template<typename T, typename RedOp>
struct RunWorkElement<ncclFuncAllReduce, T, RedOp, NCCL_ALGO_RING, NCCL_PROTO_SIMPLE> {
  __device__ __forceinline__ void run(ncclWorkElem *args) {
    using Proto = ProtoSimple<ALLREDUCE_CHUNKSTEPS/ALLREDUCE_SLICESTEPS, ALLREDUCE_SLICESTEPS>;
    runRing<T, RedOp, Proto>(args);
  }
};

template<typename T, typename RedOp>
struct RunWorkElement<ncclFuncAllReduce, T, RedOp, NCCL_ALGO_TREE, NCCL_PROTO_SIMPLE> {
  __device__ __forceinline__ void run(ncclWorkElem *args) {
    runTreeUpDown<T, RedOp, ProtoSimple<1, 1>>(args);
  }
};

template<typename T, typename RedOp>
struct RunWorkElement<ncclFuncAllReduce, T, RedOp, NCCL_ALGO_COLLNET, NCCL_PROTO_SIMPLE> {
  __device__ __attribute__((noinline)) void run(ncclWorkElem *args) {
    static constexpr int COLLNET_COPY_THREADS = 64;
    const int tid = threadIdx.x;
    const int bid = args->bid;
    const int nChannels = args->nChannels;
    struct ncclDirect* tree = &ncclShmem.channel.collTree;
    const ssize_t chunkSize = int(args->lastChunkSize);
    const ssize_t size = args->count;
    const ssize_t loopSize = nChannels*tree->nHeads*chunkSize;

    const int hasUp = (tree->up[0] >= 0) ? 1 : 0;
    const int hasDn = (tree->down[0] >= 0) ? 1 : 0;
    const int nThreadsScatter = ((hasUp && hasDn) ? COLLNET_COPY_THREADS : hasUp ? 2*COLLNET_COPY_THREADS : 0);
    const int nThreadsGather  =             ((hasUp && hasDn) ? COLLNET_COPY_THREADS : hasUp ? 1*COLLNET_COPY_THREADS : 0);
    const int nThreadsBcast   = ((hasUp && hasDn) ? COLLNET_COPY_THREADS : hasUp ? 0 : 1*COLLNET_COPY_THREADS);
    const int nThreadsReduce = args->nWarps*WARP_SIZE - nThreadsScatter - nThreadsGather - nThreadsBcast;
    const int tidStartBcast = nThreadsGather;
    const int tidStartScatter = tidStartBcast + nThreadsBcast;
    const int tidStartReduce = tidStartScatter + nThreadsScatter;

    using Proto = ProtoSimple<1, 1>;

    if (tid >= tidStartScatter && tid < tidStartReduce && hasUp) {
      // Scatter
      int group = (2*Proto::MaxGroupWidth) | (1<<16);
      Primitives<T, RedOp, FanAsymmetric<0, NCCL_MAX_DIRECT_ARITY>, /*Direct=*/0, Proto, 0>
        prims(tid-tidStartScatter, nThreadsScatter, NULL, tree->up, args->sendbuff, args->recvbuff, args->redOpArg, group, args);
      for (ssize_t gridOffset = 0; gridOffset < size; gridOffset += loopSize) {
        ssize_t offset = gridOffset + bid*tree->nHeads*chunkSize;
        int nelem = min(tree->nHeads*chunkSize, size-offset);
        if (args->regUsed) {
          prims.directScatter(offset, nelem, chunkSize, tree->headRank, tree->shift);
        } else {
          prims.scatter(offset, nelem, chunkSize, tree->headRank, tree->shift);
        }
      }
    } else if (tid >= tidStartReduce && tree->out != -1) {
      int group = (3*Proto::MaxGroupWidth) | (1<<16);
      if (hasDn) {
        // Reduce, send to network
        Primitives<T, RedOp, FanAsymmetric<NCCL_MAX_DIRECT_ARITY, 1>, /*Direct=*/0, Proto, 0>
          prims(tid-tidStartReduce, nThreadsReduce, tree->down, &tree->out, args->sendbuff, args->recvbuff, args->redOpArg, group, args);
        for (ssize_t gridOffset = 0; gridOffset < size; gridOffset += loopSize) {
          ssize_t offset = gridOffset + (bid*tree->nHeads+tree->headRank)*chunkSize;
          int nelem = min(chunkSize, size-offset);
          if (args->regUsed) {
            prims.directRecvReduceSend(offset, offset, nelem);
          } else {
            prims.recvReduceSend(offset, nelem);
          }
        }
      } else {
        // Directly send to network
        Primitives<T, RedOp, FanAsymmetric<0, 1>, /*Direct=*/0, Proto, 0>
          prims(tid-tidStartReduce, nThreadsReduce, nullptr, &tree->out, args->sendbuff, args->recvbuff, args->redOpArg, group);
        for (ssize_t gridOffset = 0; gridOffset < size; gridOffset += loopSize) {
          ssize_t offset = gridOffset + (bid*tree->nHeads+tree->headRank)*chunkSize;
          int nelem = min(chunkSize, size-offset);
          prims.send(offset, nelem);
        }
      }
    } else if (tid < tidStartBcast && hasUp) {
      // Gather
      int group = (0*Proto::MaxGroupWidth) | (0<<16);
      Primitives<T, RedOp, FanAsymmetric<NCCL_MAX_DIRECT_ARITY, 0>, /*Direct=*/0, Proto, 0>
        prims(tid, nThreadsGather, tree->up, NULL, args->sendbuff, args->recvbuff, args->redOpArg, group, args);
      for (ssize_t gridOffset = 0; gridOffset < size; gridOffset += loopSize) {
        ssize_t offset = gridOffset + bid*tree->nHeads*chunkSize;
        int nelem = min(tree->nHeads*chunkSize, size-offset);
        prims.directGather(offset, nelem, chunkSize, tree->headRank, tree->shift);
      }
    } else if (tid >= tidStartBcast && tid < tidStartScatter && tree->out != -1) {
      int group = (1*Proto::MaxGroupWidth) | (0<<16);
      if (hasDn) {
        // Recv from network, broadcast
        Primitives<T, RedOp, FanAsymmetric<1, NCCL_MAX_DIRECT_ARITY>, /*Direct=*/0, Proto, 0>
          prims(tid-tidStartBcast, nThreadsBcast, &tree->out, tree->down, args->sendbuff, args->recvbuff, args->redOpArg, group, args);
        for (ssize_t gridOffset = 0; gridOffset < size; gridOffset += loopSize) {
          ssize_t offset = gridOffset + (bid*tree->nHeads+tree->headRank)*chunkSize;
          int nelem = min(chunkSize, size-offset);
          prims.recvCopyDirectSend(offset, offset, nelem, /*postOp=*/true);
        }
      } else {
        // Recv from network (no post thread needed)
        Primitives<T, RedOp, FanAsymmetric<1, 0>, /*Direct=*/0, Proto, 0>
          prims(tid-tidStartBcast, nThreadsBcast, &tree->out, nullptr, args->sendbuff, args->recvbuff, args->redOpArg, group);
        for (ssize_t gridOffset = 0; gridOffset < size; gridOffset += loopSize) {
          ssize_t offset = gridOffset + (bid*tree->nHeads+tree->headRank)*chunkSize;
          int nelem = min(chunkSize, size-offset);
          prims.recv(offset, nelem, /*postOp=*/true);
        }
      }
    }
  }
};

template<typename T, typename RedOp>
struct RunWorkElement<ncclFuncAllReduce, T, RedOp, NCCL_ALGO_RING, NCCL_PROTO_LL> {
  __device__ __forceinline__ void run(ncclWorkElem *args) {
    runRing<T, RedOp, ProtoLL>(args);
  }
};

template<typename T, typename RedOp>
struct RunWorkElement<ncclFuncAllReduce, T, RedOp, NCCL_ALGO_TREE, NCCL_PROTO_LL> {
  __device__ __forceinline__ void run(ncclWorkElem *args) {
    runTreeSplit<T, RedOp, ProtoLL>(args);
  }
};

template<typename T, typename RedOp>
struct RunWorkElement<ncclFuncAllReduce, T, RedOp, NCCL_ALGO_RING, NCCL_PROTO_LL128> {
  __device__ __forceinline__ void run(ncclWorkElem *args) {
    runRing<T, RedOp, ProtoLL128>(args);
    //LAUNCH_CLIQUE_KERNEL(AllReduceCliqueSplitKernel, RedOp, T, args);
  }
};

template<typename T, typename RedOp>
struct RunWorkElement<ncclFuncAllReduce, T, RedOp, NCCL_ALGO_TREE, NCCL_PROTO_LL128> {
  __device__ __forceinline__ void run(ncclWorkElem *args) {
    runTreeSplit<T, RedOp, ProtoLL128>(args);
    //LAUNCH_CLIQUE_KERNEL(AllReduceCliqueSplitKernel, RedOp, T, args);
  }
};
