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

template<typename T, typename RedOp>
struct RunWork<ncclFuncSendRecv, T, RedOp, NCCL_ALGO_RING, NCCL_PROTO_SIMPLE> {
  __device__ __forceinline__ void runSend(const int tid, const int nthreads, const int group, struct ncclWorkElemP2p* args) {

#if defined(ENABLE_NPKIT)
    bool isNpKitThread = (tid == 0);
    int npKitCtxIdx = blockIdx.x * NCCL_MAX_WORK_ELEMENTS_P2P;
#endif

#if defined(ENABLE_NPKIT) && defined(ENABLE_NPKIT_EVENT_TIME_SYNC_CPU)
    if (isNpKitThread) {
      uint64_t* cpuTimestamp = ncclShmem->comm.cpuTimestamp;
      NpKit::CollectGpuEvent(NPKIT_EVENT_TIME_SYNC_CPU, 0, 0, *cpuTimestamp,
          ncclShmem->comm.npKitEventCollectContexts + npKitCtxIdx);
    }
#endif

#if defined(ENABLE_NPKIT) && defined(ENABLE_NPKIT_EVENT_TIME_SYNC_GPU)
    if (isNpKitThread) {
      NpKit::CollectGpuEvent(NPKIT_EVENT_TIME_SYNC_GPU, 0, 0, __builtin_amdgcn_s_memrealtime(),
          ncclShmem->comm.npKitEventCollectContexts + npKitCtxIdx);
    }
#endif

    if (args->peer == ncclShmem->comm.rank) {
      struct ncclWorkElemP2p* recvArgs = args-1;
      if (args->buff != recvArgs->buff) {

#if defined(ENABLE_NPKIT) && defined(ENABLE_NPKIT_EVENT_SEND_RECV_LOCAL_COPY_ENTRY)
        if (isNpKitThread) {
          NpKit::CollectGpuEvent(NPKIT_EVENT_SEND_RECV_LOCAL_COPY_ENTRY, args->count*sizeof(T), 0, __builtin_amdgcn_s_memrealtime(),
              ncclShmem->comm.npKitEventCollectContexts + npKitCtxIdx);
        }
#endif

#if defined(ENABLE_NPKIT) && defined(ENABLE_NPKIT_EVENT_PRIM_SIMPLE_REDUCE_OR_COPY_MULTI_ENTRY)
        if (isNpKitThread) {
          NpKit::CollectGpuEvent(NPKIT_EVENT_PRIM_SIMPLE_REDUCE_OR_COPY_MULTI_ENTRY, args->count*sizeof(T), 0, __builtin_amdgcn_s_memrealtime(),
              ncclShmem->comm.npKitEventCollectContexts + npKitCtxIdx);
        }
#endif

        ReduceOrCopyMulti<COLL_UNROLL, RedOp, T, 1, 1, 1, 1, 0>(tid, nthreads, nullptr, false, 1, (const T**)&args->buff, 1, (T**)&recvArgs->buff, args->count);

#if defined(ENABLE_NPKIT) && defined(ENABLE_NPKIT_EVENT_PRIM_SIMPLE_REDUCE_OR_COPY_MULTI_EXIT)
        if (isNpKitThread) {
          NpKit::CollectGpuEvent(NPKIT_EVENT_PRIM_SIMPLE_REDUCE_OR_COPY_MULTI_EXIT, args->count*sizeof(T), 0, __builtin_amdgcn_s_memrealtime(),
              ncclShmem->comm.npKitEventCollectContexts + npKitCtxIdx);
        }
#endif

#if defined(ENABLE_NPKIT) && defined(ENABLE_NPKIT_EVENT_SEND_RECV_LOCAL_COPY_EXIT)
        if (isNpKitThread) {
          NpKit::CollectGpuEvent(NPKIT_EVENT_SEND_RECV_LOCAL_COPY_EXIT, args->count*sizeof(T), 0, __builtin_amdgcn_s_memrealtime(),
              ncclShmem->comm.npKitEventCollectContexts + npKitCtxIdx);
        }
#endif

      }
    } else {
      using Proto = ProtoSimple<1, 1>;
      ssize_t const count = args->count;
      int const chunkSize = args->chunkSize/sizeof(T);
      int const peer = args->peer;
      Primitives<T, RedOp, FanAsymmetric<0, 1>, 0, Proto, 1> prims
        (tid, nthreads, nullptr, &peer, args->buff, nullptr, /*redOpArg(ignored)=*/0, group);

#if defined(ENABLE_NPKIT)
      if (isNpKitThread) {
        prims.npKitCtxIdx = npKitCtxIdx;
      }
#endif

#if defined(ENABLE_NPKIT) && defined(ENABLE_NPKIT_EVENT_SEND_RECV_SEND_ENTRY)
      if (isNpKitThread) {
        NpKit::CollectGpuEvent(NPKIT_EVENT_SEND_RECV_SEND_ENTRY, count*sizeof(T), 0, __builtin_amdgcn_s_memrealtime(),
            ncclShmem->comm.npKitEventCollectContexts + npKitCtxIdx);
        prims.npKitDataProcessTotalTime = 0;
      }
#endif

      ssize_t offset = 0;
      do {
        int nelem = min(chunkSize, count-offset);
        prims.directSend(offset, offset, nelem);
        offset += nelem;
      } while(offset < count);

#if defined(ENABLE_NPKIT) && defined(ENABLE_NPKIT_EVENT_SEND_RECV_SEND_EXIT)
      if (isNpKitThread) {
        NpKit::CollectGpuEvent(NPKIT_EVENT_SEND_RECV_SEND_EXIT, count*sizeof(T), prims.npKitDataProcessTotalTime, __builtin_amdgcn_s_memrealtime(),
            ncclShmem->comm.npKitEventCollectContexts + npKitCtxIdx);
      }
#endif

    }
  }

  __device__ __forceinline__ void runRecv(const int tid, const int nthreads, const int group, struct ncclWorkElemP2p* args) {
#if defined(ENABLE_NPKIT)
    bool isNpKitThread = (tid == 0);
    int npKitCtxIdx = blockIdx.x * NCCL_MAX_WORK_ELEMENTS_P2P + 1;
#endif

#if defined(ENABLE_NPKIT) && defined(ENABLE_NPKIT_EVENT_TIME_SYNC_CPU)
    if (isNpKitThread) {
      uint64_t* cpuTimestamp = ncclShmem->comm.cpuTimestamp;
      NpKit::CollectGpuEvent(NPKIT_EVENT_TIME_SYNC_CPU, 0, 0, *cpuTimestamp,
          ncclShmem->comm.npKitEventCollectContexts + npKitCtxIdx);
    }
#endif

#if defined(ENABLE_NPKIT) && defined(ENABLE_NPKIT_EVENT_TIME_SYNC_GPU)
    if (isNpKitThread) {
      NpKit::CollectGpuEvent(NPKIT_EVENT_TIME_SYNC_GPU, 0, 0, __builtin_amdgcn_s_memrealtime(),
          ncclShmem->comm.npKitEventCollectContexts + npKitCtxIdx);
    }
#endif

    if (args->peer != ncclShmem->comm.rank) {
      using Proto = ProtoSimple<1, 1>;
      ssize_t const count = args->count;
      int const chunkSize = args->chunkSize/sizeof(T);
      int const peer = args->peer;
      Primitives<T, RedOp, FanAsymmetric<1, 0>, 0, Proto, 1> prims
        (tid, nthreads, &peer, nullptr, nullptr, args->buff, /*redOpArg(ignored)=*/0, group);

#if defined(ENABLE_NPKIT)
      if (isNpKitThread) {
        prims.npKitCtxIdx = npKitCtxIdx;
      }
#endif

#if defined(ENABLE_NPKIT) && defined(ENABLE_NPKIT_EVENT_SEND_RECV_RECV_ENTRY)
      if (isNpKitThread) {
        NpKit::CollectGpuEvent(NPKIT_EVENT_SEND_RECV_RECV_ENTRY, count*sizeof(T), 0, __builtin_amdgcn_s_memrealtime(),
            ncclShmem->comm.npKitEventCollectContexts + npKitCtxIdx);
        prims.npKitDataProcessTotalTime = 0;
      }
#endif

      ssize_t offset = 0;
      do {
        int nelem = min(chunkSize, count-offset);
        prims.directRecv(offset, nelem);
        offset += nelem;
      } while(offset < count);

#if defined(ENABLE_NPKIT) && defined(ENABLE_NPKIT_EVENT_SEND_RECV_RECV_EXIT)
      if (isNpKitThread) {
        NpKit::CollectGpuEvent(NPKIT_EVENT_SEND_RECV_RECV_EXIT, count*sizeof(T), prims.npKitDataProcessTotalTime, __builtin_amdgcn_s_memrealtime(),
            ncclShmem->comm.npKitEventCollectContexts + npKitCtxIdx);
      }
#endif

    }
  }

  __device__ __forceinline__ void run(ncclWork *work) {
    struct ncclWorkElemP2p* args = work->p2pElems;
    int ngroups = args->ngroups;
    int tid = threadIdx.x;
    int wid = tid / WARP_SIZE;
    // This has to work even for groups of 2.5 warps (which is 8 groups, and means 3
    // warps for send, 2 warps for recv).
    // warpStarts were rounded thanks to int division, but for group number we need to round the other way around
    // So we mirror wid then mirror again the group.
    #define NWARPS (NCCL_MAX_NTHREADS/WARP_SIZE)
    int group = ngroups-1- (NWARPS-1-wid) * ngroups / NWARPS;
    args += group;
    if (args->header.type == ncclWorkTypeUnused) return;

    tid -= args->warpStart * WARP_SIZE;
    int nthreads = args->nWarps * WARP_SIZE;
    group |= (args->connIndex<<16); // Used to select connIndex 1
    if (tid >= nthreads || args->peer == -1) return;
    if ((group%2) == 0) {
      runRecv(tid, nthreads, group, args);
    } else {
      runSend(tid, nthreads, group, args);
    }
  }
};
