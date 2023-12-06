/*************************************************************************
 * Copyright (c) 2017-2022, NVIDIA CORPORATION. All rights reserved.
 * Modifications Copyright (c) 2019-2022 Advanced Micro Devices, Inc. All rights reserved.
 * Modifications Copyright (c) Microsoft Corporation. Licensed under the MIT License.
 *
 * See LICENSE.txt for license information
 ************************************************************************/
#ifndef MSSCLKERNELIMPL_H
#define MSSCLKERNELIMPL_H

#include "devcomm.h"
#include "primitives.h"
#include "collectives.h"

#include "msccl/msccl_struct.h"
#include "msccl/msccl_kernel.h"

extern __shared__ struct mscclShmemData mscclShmem;

#define MSCCL_MAX_ITER 65536

// flags are a 3-tuple of (workindex, gridoffset_iter, step) and it follows a lexicographical order. a threadblock is ahead of another iff its flag is ahead
#define COMPUTE_FLAG(__WORKINDEX__,__GRIDOFFSET_ITER__,__STEP__) \
  MSCCL_MAX_ITER*MSCCL_MAX_NUM_STEPS*(uint64_t)__WORKINDEX__ + ((uint64_t)__GRIDOFFSET_ITER__ * MSCCL_MAX_NUM_STEPS + (uint64_t)__STEP__)

#define GET_WORKINDEX_FROM_FLAG(__FLAG__) \
  (__FLAG__) / (MSCCL_MAX_ITER*MSCCL_MAX_NUM_STEPS)

#ifdef ENABLE_COLLTRACE
  #define INC_COLL_TRACE \
    uint32_t pos = atomicAdd(&ncclShmem.collTraceTail->tail, 1)%COLLTRACE_NUM_ITEMS; \
    struct ncclCollTrace* collTrace = ncclShmem.collTrace+pos; \
    collTrace->timeStamp = wall_clock64(); \
    collTrace->bid = blockIdx.x;
    // TODO: switch to atomicInc after llvm crash is fixed
    // uint32_t pos = atomicInc(&ncclShmem.collTraceTail->tail, COLLTRACE_NUM_ITEMS)

  #define traceData(data2, data4, data8_0, data8_1) { \
    INC_COLL_TRACE \
    collTrace->funcIndex = data2; \
    collTrace->data_0 = data4; \
    collTrace->opCount = data8_0; \
    collTrace->data_1 = data8_1; \
    collTrace->type = ncclCollTraceDataType; \
  }
#else
#define traceData(data2, data4, data8_0, data8_1)
#endif


// a copy of the volatile load/store from prims_ll
template<typename U>
__device__ static U load(U *src) {
  union {
    U elt;
    uint8_t u1;
    uint16_t u2;
    uint32_t u4;
    uint64_t u8;
  };
#if defined(__HIP_PLATFORM_HCC__) || defined(__HCC__) || defined(__HIPCC__)
  if(sizeof(U) == 1)
    u1 = __builtin_nontemporal_load((uint8_t*)src);
  else if(sizeof(U) == 2)
    u2 = __builtin_nontemporal_load((uint16_t*)src);
  else if(sizeof(U) == 4)
    u4 = __builtin_nontemporal_load((uint32_t*)src);
  else
    u8 = __builtin_nontemporal_load((uint64_t*)src);
#else
  if(sizeof(U) == 1)
    asm("ld.volatile.global.b8 %0,[%1];" : "=r"(u4) : "l"(src));
  else if(sizeof(U) == 2)
    asm("ld.volatile.global.b16 %0,[%1];" : "=h"(u2) : "l"(src));
  else if(sizeof(U) == 4)
    asm("ld.volatile.global.b32 %0,[%1];" : "=r"(u4) : "l"(src));
  else
    asm("ld.volatile.global.b64 %0,[%1];" : "=l"(u8) : "l"(src));
#endif
  return elt;
}

template<typename U>
__device__ static void store(U *dst, U val) {
  union {
    U elt;
    uint8_t u1;
    uint16_t u2;
    uint32_t u4;
    uint64_t u8;
  };
  elt = val;
#if defined(__HIP_PLATFORM_HCC__) || defined(__HCC__) || defined(__HIPCC__)
  if(sizeof(U) == 1)
    __builtin_nontemporal_store(u1, (uint8_t*)dst);
  else if(sizeof(U) == 2)
    __builtin_nontemporal_store(u2, (uint16_t*)dst);
  else if(sizeof(U) == 4)
    __builtin_nontemporal_store(u4, (uint32_t*)dst);
  else
    __builtin_nontemporal_store(u8, (uint64_t*)dst);
#else
  if(sizeof(U) == 1)
    asm("st.volatile.global.b8 [%0],%1;" :: "l"(dst), "r"(u4));
  else if(sizeof(U) == 2)
    asm("st.volatile.global.b16 [%0],%1;" :: "l"(dst), "h"(u2));
  else if(sizeof(U) == 4)
    asm("st.volatile.global.b32 [%0],%1;" :: "l"(dst), "r"(u4));
  else
    asm("st.volatile.global.b64 [%0],%1;" :: "l"(dst), "l"(u8));
#endif
}

inline __device__ static void barrier(int nthreads) {
#if defined(__HIP_PLATFORM_HCC__) || defined(__HCC__) || defined(__HIPCC__)
  assert(nthreads == NCCL_MAX_NTHREADS);
  __asm__ __volatile__("s_waitcnt vmcnt(0) lgkmcnt(0)\ns_barrier");
#else
  asm volatile ("bar.sync %1, %0;" :: "r"(nthreads), "r"(15));
#endif
}

// Copy 8-byte aligned data. You must call with at least `(bytes+7)/8` threads.
inline __device__ static void copyToShmem8(int tid, void* dst, void const* src, int bytes) {
  int offset = sizeof(uint32_t) * tid;
  if (offset < bytes) {
    uint32_t *src2 = (uint32_t*)((char const*)src + offset);
    uint32_t *dst2 = (uint32_t*)((char*)dst + offset);
    *dst2 = *src2;
    offset += WARP_SIZE*sizeof(uint32_t);
  }
}

__device__ __forceinline__ static void threadBlockCopy(
  uint32_t *dst, uint32_t const *src, uint64_t size, int tid, int nthreads) {
  for (int i = tid; i < size; i += nthreads) {
    dst[i] = src[i];
  }
}

#define MSCCL_REDUCE_UNROLL_LOOP_A(numloops) \
for (int r = 0; r < numloops; r++) { \
  srcOffset = srcBaseOffset + (ssize_t)mscclShmem.mscclTB.reductionSrcOffsets[t->reductionPointer+r] * sizePerMscclChunk; \
  reduceInput = load(srcPointer + srcOffset); \
  o = applyReduce(redFn, reduceInput, o); \
}

#define MSCCL_REDUCE_UNROLL_LOOP_B(numloops) \
for (int r = 0; r < numloops; r++) { \
  srcOffset = srcBaseOffset + (ssize_t)mscclShmem.mscclTB.reductionSrcOffsets[t->reductionPointer+r] * sizePerMscclChunk; \
  srcs[r] = srcPointer + srcOffset; \
}

template<typename T, typename RedOp, typename Proto, bool fullOps>
__device__ __forceinline__ void mscclRunInterpreter(
  struct ncclDevComm* comm, struct mscclAlgo* algo, struct mscclWork* work) {
  const int tid = threadIdx.x;
  const int bid = blockIdx.x;
  const int nthreads = NCCL_MAX_NTHREADS;

#if defined(ENABLE_NPKIT)
  uint64_t timestamp_entry = 0;
  if (tid == 0) {
     timestamp_entry = NPKIT_GET_GPU_TIMESTAMP();
  }
#endif
  // initialize mscclShmem.mscclTB
  threadBlockCopy(
    (uint32_t *)&mscclShmem.mscclTB, (uint32_t *)(algo->mscclTBs + bid),
    sizeof(struct mscclThreadBlock) / sizeof(uint32_t), tid, nthreads);
  __synclds(); // publish mscclShmem.mscclTB.channelId

  // initialize ncclShmem and mscclShmem.work
  int channelId = mscclShmem.mscclTB.channelId;
  {
    void *dst, *src;
    int bytes = 0;
    // Use first 3 warps to load comm, channel, and work into shmem
    switch (tid/WARP_SIZE) {
    case 0:
      dst = &ncclShmem.comm;
      src = comm;
      bytes = sizeof(ncclDevComm);
      break;
    case 1:
      // Get address of channel without incurring indirect load from ncclDevComm::channels
      dst = &ncclShmem.channel;
      src = &((ncclDevCommAndChannels*)comm)->channels[channelId];
      bytes = sizeof(ncclDevChannel);
      break;
    case 2:
      dst = &mscclShmem.work;
      src = work + blockIdx.x;
      bytes = sizeof(mscclWork);
      break;
    case 3:
      /* set abort flag to 0 */
      if (tid%WARP_SIZE == 0) ncclShmem.aborted = 0;
#ifdef ENABLE_COLLTRACE
      else if (tid%WARP_SIZE == 1) ncclShmem.collTrace = comm->collTrace + COLLTRACE_NUM_ITEMS*channelId;
      else if (tid%WARP_SIZE == 2) ncclShmem.collTraceTail = comm->collTraceTail + channelId;
#endif
      break;
    default:
      break;
    }
    copyToShmem8(tid%WARP_SIZE, dst, src, bytes);
  }

#if defined(ENABLE_NPKIT)
  int npKitCtxIdx = bid;
  int xcc_id = 0;
  if (tid == 0) {
    ncclShmem.event_buffer_head = 0;
#if defined(__gfx940__) || defined(__gfx941__) || defined(__gfx942__)
    asm volatile ("s_getreg_b32 %0, hwreg(HW_REG_XCC_ID)" : "=s" (xcc_id));
#endif
  }
#endif
  __synclds(); // publish shmem

  if (fullOps && tid == 0) {
    traceData(__LINE__, mscclShmem.work.fnIndex, (uint64_t)mscclShmem.work.sendBuff, 0);
  }

  if (tid == 0)
    *mscclShmem.work.workFifoDone = mscclShmem.work.workFifoDoneAck;

#if defined(ENABLE_NPKIT) && defined(ENABLE_NPKIT_EVENT_TIME_SYNC_CPU)
  if (tid == 0) {
    uint64_t* cpuTimestamp = ncclShmem.comm.cpuTimestamp;
    NpKit::CollectGpuEventLDS(NPKIT_EVENT_TIME_SYNC_CPU, 0, xcc_id, *cpuTimestamp);
  }
#endif

#if defined(ENABLE_NPKIT) && defined(ENABLE_NPKIT_EVENT_TIME_SYNC_GPU)
  if (tid == 0) {
    NpKit::CollectGpuEventLDS(NPKIT_EVENT_TIME_SYNC_GPU, 0, xcc_id, NPKIT_GET_GPU_TIMESTAMP());
  }
#endif

  // User pointers for primitives
  T* thisInput = (T*)mscclShmem.work.sendBuff;
  T* thisOutput = (T*)mscclShmem.work.recvBuff;
  T* thisScratch = (T*)mscclShmem.work.scratchBuffer;
  int recvPeer = mscclShmem.mscclTB.recvPeer;
  int sendPeer = mscclShmem.mscclTB.sendPeer;

  const ssize_t chunkSize = int(Proto::calcBytePerStep()/sizeof(T) * (Proto::Id == NCCL_PROTO_SIMPLE ? MSCCL_CHUNKSTEPS : 1));
  int minChunkSize;
  if (Proto::Id == NCCL_PROTO_LL)
    minChunkSize = nthreads*(Proto::calcBytePerGrain()/sizeof(T));
  if (Proto::Id == NCCL_PROTO_LL128) {
    // We should not need the final /2 but it makes performance much, much smoother. Might be a bug somewhere.
    minChunkSize = nthreads*(Proto::calcBytePerGrain()/sizeof(T))/2;
  }

  RedOp redFn(mscclShmem.work.redOpArg);
  Primitives<T, RedOp, FanAsymmetric<1,1>, 1, Proto, 0> prims
    (tid, nthreads, &recvPeer, &sendPeer, thisInput, thisOutput, mscclShmem.work.redOpArg);

#if defined(ENABLE_NPKIT)
  if (tid == 0) {
    prims.npKitCtxIdx = npKitCtxIdx;
  }
#endif

  const ssize_t sizePerMscclChunk = mscclShmem.work.sizePerMscclChunk;
  uint32_t maxAllowedCount = mscclShmem.work.maxAllowedCount;

#if defined(ENABLE_NPKIT) && defined(ENABLE_NPKIT_EVENT_MSCCL_RUN_ENTRY)
  if (tid == 0) {
    NpKit::CollectGpuEventLDS(NPKIT_EVENT_MSCCL_RUN_ENTRY, mscclShmem.work.sizePerMscclChunk*mscclShmem.work.nChunksPerLoop, xcc_id, timestamp_entry);
  }
#endif

#if defined(ENABLE_NPKIT) && defined(ENABLE_NPKIT_EVENT_MSCCL_INIT_ENTRY)
  if (tid == 0) {
    NpKit::CollectGpuEventLDS(NPKIT_EVENT_MSCCL_INIT_ENTRY, 0, xcc_id, timestamp_entry);
  }
#endif

#if defined(ENABLE_NPKIT) && defined(ENABLE_NPKIT_EVENT_MSCCL_INIT_EXIT)
  if (tid == 0) {
    NpKit::CollectGpuEventLDS(NPKIT_EVENT_MSCCL_INIT_EXIT, 0, xcc_id, NPKIT_GET_GPU_TIMESTAMP());
  }
#endif

  // msccl flags all start out with 0. this is used as a part of the flag to make sure different work items deal with different synchronization flags
  // this still needs more work. when we make a way around the queue, the flag might have been set to undesired values. will be fixed in subsequent versions.
  const int64_t workIndex = mscclShmem.work.workIndex;
  volatile struct mscclFlag* mscclFlags = mscclShmem.work.syncFlags;
  for (ssize_t gridOffset = 0, iter = 0; gridOffset < sizePerMscclChunk; gridOffset += chunkSize, iter++) {
    ssize_t realChunkSize;
    if (Proto::Id == NCCL_PROTO_SIMPLE) {
      realChunkSize = min(chunkSize, sizePerMscclChunk-gridOffset);
      realChunkSize = roundUp(realChunkSize, nthreads*sizeof(uint64_t)/sizeof(T));
    }
    else
      realChunkSize = min(chunkSize, divUp(sizePerMscclChunk-gridOffset, minChunkSize)*minChunkSize);
    realChunkSize = int(realChunkSize);
    int nelem = min(realChunkSize, sizePerMscclChunk-gridOffset);

    ssize_t srcOffset, dstOffset;
    T *srcPointer, *dstPointer;
    int step = 0;
    for (int i = 0; i < mscclShmem.mscclTB.nSteps; i++){
      struct mscclTransmission* t = &mscclShmem.mscclTB.transmissions[i];
      // first wait if there is a dependence
      int16_t numDependencies = t->numDependencies;
      if (numDependencies > 0){
        if (tid < numDependencies) {
          int16_t dependentPointer = t->dependencePointer;
          int8_t dependentBid = mscclShmem.mscclTB.dependentBid[dependentPointer+tid];
          int16_t dependentStep = mscclShmem.mscclTB.dependentStep[dependentPointer+tid];
          uint64_t goalFlag = COMPUTE_FLAG(workIndex, iter, dependentStep);
          while (true){
            uint64_t curFlag = __atomic_load_n(&(mscclFlags + dependentBid)->flag, __ATOMIC_RELAXED);
            if (curFlag >= goalFlag && GET_WORKINDEX_FROM_FLAG(curFlag) == workIndex) break;
          }
        }
        step += numDependencies-1;
        barrier(nthreads);
      }

      srcPointer = (t->srcBuffer == MSCCL_INPUT_BUFFER) ? thisInput : ((t->srcBuffer == MSCCL_OUTPUT_BUFFER) ? thisOutput : thisScratch);
      dstPointer = (t->dstBuffer == MSCCL_INPUT_BUFFER) ? thisInput : ((t->dstBuffer == MSCCL_OUTPUT_BUFFER) ? thisOutput : thisScratch);
      prims.setDataPtrs(srcPointer, dstPointer);

      int count = t->count;
      for (int c = 0; c < count; c += maxAllowedCount) {
        srcOffset = gridOffset + (ssize_t) (t->srcOffset+c) * sizePerMscclChunk;
        dstOffset = gridOffset + (ssize_t) (t->dstOffset+c) * sizePerMscclChunk;
        int thisCount = min(maxAllowedCount, count - c);
        int thisNelem = nelem * thisCount;
        if (t->type == MSCCL_SEND) {
#if defined(ENABLE_NPKIT) && defined(ENABLE_NPKIT_EVENT_MSCCL_SEND_ENTRY)
            if (tid == 0) {
              NpKit::CollectGpuEventLDS(NPKIT_EVENT_MSCCL_SEND_ENTRY, thisNelem*sizeof(T), 0, NPKIT_GET_GPU_TIMESTAMP());
            }
#endif	
          prims.send(srcOffset, thisNelem); // LL.send is the only situation where there is no barrier at the end.

#if defined(ENABLE_NPKIT) && defined(ENABLE_NPKIT_EVENT_MSCCL_SEND_EXIT)
            if (tid == 0) {
              NpKit::CollectGpuEventLDS(NPKIT_EVENT_MSCCL_SEND_EXIT, thisNelem*sizeof(T), 0, NPKIT_GET_GPU_TIMESTAMP());
            }
#endif						       
        }
        else if (t->type == MSCCL_RECV) {
#if defined(ENABLE_NPKIT) && defined(ENABLE_NPKIT_EVENT_MSCCL_RECV_ENTRY)
            if (tid == 0) {
              NpKit::CollectGpuEventLDS(NPKIT_EVENT_MSCCL_RECV_ENTRY, thisNelem*sizeof(T), 0, NPKIT_GET_GPU_TIMESTAMP());
            }
#endif	
          prims.recv(dstOffset, thisNelem);
#if defined(ENABLE_NPKIT) && defined(ENABLE_NPKIT_EVENT_MSCCL_RECV_EXIT)
            if (tid == 0) {
              NpKit::CollectGpuEventLDS(NPKIT_EVENT_MSCCL_RECV_EXIT, thisNelem*sizeof(T), 0, NPKIT_GET_GPU_TIMESTAMP());
            }
#endif
        }
        else if (t->type == MSCCL_REDUCE) {
          int numReductions = t->numReductions;
          int currIdx = tid;
#if defined(__gfx942__)
          if (Proto::Id == NCCL_PROTO_LL) {
#else
          if (thisNelem < nthreads) {
#endif
#if defined(ENABLE_NPKIT) && defined(ENABLE_NPKIT_EVENT_MSCCL_REDUCE_ENTRY)
            if (tid == 0) {
              NpKit::CollectGpuEventLDS(NPKIT_EVENT_MSCCL_REDUCE_ENTRY, thisNelem*sizeof(T), 0, NPKIT_GET_GPU_TIMESTAMP());
            }
#endif

#if defined(__gfx942__)
            while (currIdx < thisNelem) {
#else
            if (currIdx < thisNelem) {
#endif
              dstOffset = gridOffset + (ssize_t) (t->dstOffset+c) * sizePerMscclChunk;
              T* dstIndex = dstPointer + dstOffset + currIdx;
              T reduceInput;
              T o = load(dstIndex);
              ssize_t srcBaseOffset = gridOffset + (ssize_t)c * sizePerMscclChunk + currIdx;
              switch (numReductions) {
                case 7:
                  #pragma unroll
                  MSCCL_REDUCE_UNROLL_LOOP_A(7);
                  break;
#if defined(__gfx90a__)
                case 15:
                  #pragma unroll
                  MSCCL_REDUCE_UNROLL_LOOP_A(15);
                  break;
#endif
                default:
                  MSCCL_REDUCE_UNROLL_LOOP_A(numReductions);
                  break;
              }
              store(dstIndex, o);
#if defined(__gfx942__)
              currIdx += nthreads;
#endif
            }

#if defined(ENABLE_NPKIT) && defined(ENABLE_NPKIT_EVENT_MSCCL_REDUCE_EXIT)
            if (tid == 0) {
              NpKit::CollectGpuEventLDS(NPKIT_EVENT_MSCCL_REDUCE_EXIT, thisNelem*sizeof(T), 0, NPKIT_GET_GPU_TIMESTAMP());
            }
#endif

            barrier(nthreads);
          } else {
            T* srcs[MSCCL_MAX_REDUCE_FUSION+1]; // +1 is for SIMPLE protocol as dst is added in the list of srcs
            dstOffset = gridOffset + (ssize_t) (t->dstOffset+c) * sizePerMscclChunk;
            T* dst = dstPointer + dstOffset;
            ssize_t srcBaseOffset = gridOffset + (ssize_t)c * sizePerMscclChunk;
            switch (numReductions) {
              case 7:
                #pragma unroll
                MSCCL_REDUCE_UNROLL_LOOP_B(7);
                break;
#if defined(__gfx90a__)
              case 15:
                #pragma unroll
                MSCCL_REDUCE_UNROLL_LOOP_B(15);
                break;
#endif
              default:
                MSCCL_REDUCE_UNROLL_LOOP_B(numReductions);
                break;
            }
            prims.reduce(srcs, numReductions, &dst, 1, thisNelem);
          }
          if (c == 0) step += (numReductions-1); // only advance step once!
        } else if (fullOps && t->type == MSCCL_RECV_COPY_SEND)
          prims.recvCopySend(dstOffset, thisNelem);
        else if (fullOps && t->type == MSCCL_RECV_REDUCE_SEND)
          prims.recvReduceSend(srcOffset, thisNelem);
        else if (fullOps && t->type == MSCCL_RECV_REDUCE_COPY_SEND)
          prims.recvReduceCopySend(srcOffset, dstOffset, thisNelem);
        else if (fullOps && t->type == MSCCL_RECV_REDUCE_COPY) {
#if defined(ENABLE_NPKIT) && defined(ENABLE_NPKIT_EVENT_MSCCL_RECV_REDUCE_COPY_ENTRY)
          if (tid == 0) {
            NpKit::CollectGpuEventLDS(NPKIT_EVENT_MSCCL_RECV_REDUCE_COPY_ENTRY, thisNelem*sizeof(T), 0, NPKIT_GET_GPU_TIMESTAMP());
          }
#endif
          prims.recvReduceCopy(srcOffset, dstOffset, thisNelem);
#if defined(ENABLE_NPKIT) && defined(ENABLE_NPKIT_EVENT_MSCCL_RECV_REDUCE_COPY_EXIT)
          if (tid == 0) {
            NpKit::CollectGpuEventLDS(NPKIT_EVENT_MSCCL_RECV_REDUCE_COPY_EXIT, thisNelem*sizeof(T), 0, NPKIT_GET_GPU_TIMESTAMP());
          }
#endif
        }
        else if (fullOps && t->type == MSCCL_LOCAL_COPY)
          prims.localCopy(srcPointer+srcOffset, dstPointer+dstOffset, thisNelem);
        else
          return;
      }
      if (t->hasDependence && tid == nthreads-1)
        __atomic_store_n(&mscclFlags[bid].flag, (uint64_t) COMPUTE_FLAG(workIndex, iter, step), __ATOMIC_RELAXED);
      step++;
    }
  }
#if defined(ENABLE_NPKIT) && defined(ENABLE_NPKIT_EVENT_MSCCL_RUN_EXIT)
  if (tid == 0) {
    NpKit::CollectGpuEventLDS(NPKIT_EVENT_MSCCL_RUN_EXIT, mscclShmem.work.sizePerMscclChunk*mscclShmem.work.nChunksPerLoop, xcc_id, NPKIT_GET_GPU_TIMESTAMP());
  }
#endif
#if defined(ENABLE_NPKIT)
  __synclds();
  NpKitEventCollectContext* ctx = ncclShmem.comm.npKitEventCollectContexts + npKitCtxIdx;
  copyToShmem16(tid, ctx->event_buffer+ctx->event_buffer_head, ncclShmem.event_buffer, sizeof(NpKitEvent)*ncclShmem.event_buffer_head);
  if (tid == 0) ctx->event_buffer_head += ncclShmem.event_buffer_head;
#endif

  if (fullOps && tid == 0) {
    traceData(__LINE__, mscclShmem.work.fnIndex, (uint64_t)mscclShmem.work.sendBuff, 0);
  }
}

#define MSCCL_IMPL_KERNEL_ENTRY_FUNC_DEVREDOP_TYPE(devredop, type, fullOps) \
__global__ void MSCCL_KERNEL_ENTRY_NAME(devredop, type, LL, fullOps)(struct ncclDevComm* comm, struct mscclAlgo* algo, struct mscclWork* work) { \
  mscclRunInterpreter<type, Func##devredop<type>, ProtoLL, fullOps>(comm, algo, work); \
} \
__global__ void MSCCL_KERNEL_ENTRY_NAME(devredop, type, LL128, fullOps)(struct ncclDevComm* comm, struct mscclAlgo* algo, struct mscclWork* work) { \
  mscclRunInterpreter<type, Func##devredop<type>, ProtoLL128, fullOps>(comm, algo, work); \
} \
__global__ void MSCCL_KERNEL_ENTRY_NAME(devredop, type, Simple, fullOps)(struct ncclDevComm* comm, struct mscclAlgo* algo, struct mscclWork* work) { \
  mscclRunInterpreter<type, Func##devredop<type>, ProtoSimple<MSCCL_CHUNKSTEPS/MSCCL_SLICESTEPS, MSCCL_SLICESTEPS>, fullOps>(comm, algo, work); \
}

#define MSCCL_IMPL_KERNEL_ENTRY_FUNC_DEVREDOP(devredop) \
  MSCCL_IMPL_KERNEL_ENTRY_FUNC_DEVREDOP_TYPE(devredop, int8_t, fullOps) \
  MSCCL_IMPL_KERNEL_ENTRY_FUNC_DEVREDOP_TYPE(devredop, uint8_t, fullOps) \
  MSCCL_IMPL_KERNEL_ENTRY_FUNC_DEVREDOP_TYPE(devredop, int32_t, fullOps) \
  MSCCL_IMPL_KERNEL_ENTRY_FUNC_DEVREDOP_TYPE(devredop, uint32_t, fullOps) \
  MSCCL_IMPL_KERNEL_ENTRY_FUNC_DEVREDOP_TYPE(devredop, int64_t, fullOps) \
  MSCCL_IMPL_KERNEL_ENTRY_FUNC_DEVREDOP_TYPE(devredop, uint64_t, fullOps) \
  MSCCL_IMPL_KERNEL_ENTRY_FUNC_DEVREDOP_TYPE(devredop, half, fullOps) \
  MSCCL_IMPL_KERNEL_ENTRY_FUNC_DEVREDOP_TYPE(devredop, float, fullOps) \
  MSCCL_IMPL_KERNEL_ENTRY_FUNC_DEVREDOP_TYPE(devredop, double, fullOps) \
  MSCCL_IMPL_KERNEL_ENTRY_FUNC_DEVREDOP_TYPE(devredop, rccl_bfloat16, fullOps)

#define MSCCL_IMPL_KERNEL_ENTRY_FUNC_DEVREDOP_NOFLOAT(devredop) \
  MSCCL_IMPL_KERNEL_ENTRY_FUNC_DEVREDOP_TYPE(devredop, int8_t) \
  MSCCL_IMPL_KERNEL_ENTRY_FUNC_DEVREDOP_TYPE(devredop, uint8_t) \
  MSCCL_IMPL_KERNEL_ENTRY_FUNC_DEVREDOP_TYPE(devredop, int32_t) \
  MSCCL_IMPL_KERNEL_ENTRY_FUNC_DEVREDOP_TYPE(devredop, uint32_t) \
  MSCCL_IMPL_KERNEL_ENTRY_FUNC_DEVREDOP_TYPE(devredop, int64_t) \
  MSCCL_IMPL_KERNEL_ENTRY_FUNC_DEVREDOP_TYPE(devredop, uint64_t)

#define MSCCL_IMPL_KERNEL_ENTRY_FUNC() \
  MSCCL_IMPL_KERNEL_ENTRY_FUNC_DEVREDOP(Sum, false) \
  MSCCL_IMPL_KERNEL_ENTRY_FUNC_DEVREDOP(Prod, false) \
  MSCCL_IMPL_KERNEL_ENTRY_FUNC_DEVREDOP(Min, false) \
  MSCCL_IMPL_KERNEL_ENTRY_FUNC_DEVREDOP(Max, false) \
  MSCCL_IMPL_KERNEL_ENTRY_FUNC_DEVREDOP(PreMulSum, false) \
  MSCCL_IMPL_KERNEL_ENTRY_FUNC_DEVREDOP_NOFLOAT(SumPostDiv, false)

#endif
