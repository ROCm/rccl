/*************************************************************************
 * Copyright (c) 2017-2022, NVIDIA CORPORATION. All rights reserved.
 * Modifications Copyright (c) 2019-2023 Advanced Micro Devices, Inc. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

#ifndef NCCL_DEVICE_COMMON_H_
#define NCCL_DEVICE_COMMON_H_

#include "collectives.h"
#include "device.h"
#include "op128.h"
#include "device_table.h"
#include "network/unpack/unpack_defs.h"
#include "comm.h"

#define NCCL_MAX_DEV_ARITY (NCCL_MAX_TREE_ARITY-1)  // Using balanced tree instead of split tree

#define __syncwarp()

#ifdef __GFX12__
#define __synclds() \
  asm volatile("s_waitcnt lgkmcnt(0) \n s_barrier_signal -1 \n s_barrier_wait -1");
#else
#define __synclds() \
  asm volatile("s_waitcnt lgkmcnt(0) \n s_barrier");
#endif

#ifdef __GFX9__
#define STORE(DST, SRC) \
  { __atomic_store_n((DST), (SRC), __ATOMIC_RELAXED); }
#else
#define STORE(DST, SRC) \
  { __atomic_store_n((DST), (SRC), __ATOMIC_SEQ_CST); }
#endif

#if defined(__gfx1100__) || defined(__gfx1101__) || defined(__gfx1102__) || defined(__gfx1200__) || defined(__gfx1201__)
#define __trace_hwreg()
#else
#define __trace_hwreg() \
  asm volatile ("s_getreg_b32 %0, hwreg(HW_REG_HW_ID)" : "=s" (collTrace->data_0));
#endif
#ifdef ENABLE_COLLTRACE
  #define INC_COLL_TRACE \
    uint32_t pos = atomicAdd(&ncclShmem.collTraceTail->tail, 1)%COLLTRACE_NUM_ITEMS; \
    struct ncclCollTrace* collTrace = ncclShmem.collTrace+pos; \
    collTrace->timeStamp = wall_clock64(); \
    collTrace->bid = blockIdx.x;
    // TODO: switch to atomicInc after llvm crash is fixed
    // uint32_t pos = atomicInc(&ncclShmem.collTraceTail->tail, COLLTRACE_NUM_ITEMS)

  #define traceKernelLaunch(launch_type) { \
    INC_COLL_TRACE \
    collTrace->funcIndex = ncclShmem.work.header.funcIndex; \
    __trace_hwreg()\
    if (ncclShmem.work.header.type == ncclWorkTypeP2p) { \
      struct ncclWorkElemP2p *p2pElems = ncclShmem.work.p2pElems; \
      collTrace->p2p[0].connIndex = 0; \
      collTrace->p2pOpCount[0] = p2pElems[0].opCount; \
      collTrace->p2p[0].ngroups = p2pElems[0].ngroups; \
      collTrace->p2p[0].nWarps = p2pElems[0].nWarps; \
      collTrace->p2p[0].warpStart = p2pElems[0].warpStart; \
      collTrace->p2p[0].peer = p2pElems[0].p2pType == ncclWorkP2pTypeRecv ? (uint16_t)(p2pElems[0].peer) : -1; \
      collTrace->p2p[1].connIndex = 0; \
      collTrace->p2pOpCount[1] = p2pElems[1].opCount; \
      collTrace->p2p[1].ngroups = p2pElems[1].ngroups; \
      collTrace->p2p[1].nWarps = p2pElems[1].nWarps; \
      collTrace->p2p[1].warpStart = p2pElems[1].warpStart; \
      collTrace->p2p[1].peer = p2pElems[1].p2pType == ncclWorkP2pTypeSend ? (uint16_t)(p2pElems[1].peer) : -1; \
      collTrace->type = (launch_type) | ncclCollTraceP2pElemType; \
    } else if (ncclShmem.work.header.type == ncclWorkTypeColl) { \
      struct ncclWorkElem *elems = ncclShmem.work.elems; \
      collTrace->opCount = elems[0].opCount; \
      collTrace->coll.nWarps = elems[0].nWarps; \
      collTrace->coll.bid = elems[0].bid; \
      collTrace->coll.nChannels = elems[0].nChannels; \
      collTrace->type = (launch_type) | ncclCollTraceCollElemType; \
    } \
  }
  #define traceKernelEnd(end_type)  { \
    INC_COLL_TRACE \
    if (ncclShmem.work.header.type == ncclWorkTypeP2p) { \
      struct ncclWorkElemP2p *p2pElems = ncclShmem.work.p2pElems; \
      collTrace->p2pOpCount[0] = p2pElems[0].opCount; \
      collTrace->p2pOpCount[1] = p2pElems[1].opCount; \
    } else if (ncclShmem.work.header.type == ncclWorkTypeColl) { \
      struct ncclWorkElem *elems = ncclShmem.work.elems; \
      collTrace->opCount = elems[0].opCount; \
    } \
    collTrace->type = end_type; \
  }
  #define traceData(data2, data4, data8_0, data8_1) { \
    INC_COLL_TRACE \
    collTrace->funcIndex = data2; \
    collTrace->data_0 = data4; \
    collTrace->opCount = data8_0; \
    collTrace->data_1 = data8_1; \
    collTrace->type = ncclCollTraceDataType; \
  }
#else
#define traceKernelLaunch(launch_type)
#define traceKernelEnd(end_type)
#define traceData(data2, data4, data8_0, data8_1)
#endif

struct ncclShmemGroup {
  ncclConnInfo *recvConns[NCCL_MAX_ARITY];
  ncclConnInfo *sendConns[NCCL_MAX_ARITY];
  void* userInput;
  void* userOutput;
  void* srcs[NCCL_MAX_ARITY+1];
  void* dsts[NCCL_MAX_ARITY+1];
  uint64_t barrier;
  uint64_t barrier_next[NCCL_MAX_GROUPS];
  union {
    unpackGroupShmem unpack;
  } devicePlugin;
  int32_t dstSizes[NCCL_MAX_ARITY+1];
};

#define LDS_NUM_EVENTS 64

struct ncclShmemData {
  struct ncclShmemGroup groups[NCCL_MAX_GROUPS];
  uint64_t redOpArgs[NCCL_MAX_ARITY+1];
  int channelId;
  int aborted;
  alignas(16) struct ncclDevComm comm;
  alignas(16) struct ncclDevChannel channel;
  alignas(16) struct ncclWork work;
  alignas(16) union {
    unpackShmem unpack;
  } devicePlugin;
#ifdef ENABLE_COLLTRACE
  struct ncclCollTrace* collTrace;
  union ncclCollTraceTail* collTraceTail;
#endif
#ifdef ENABLE_PROFILING
  struct ncclProf prof;
#endif
#if defined(ENABLE_NPKIT)
  NpKitEvent event_buffer[LDS_NUM_EVENTS];
  uint64_t event_buffer_head;
#endif
};
static_assert(offsetof(struct ncclShmemData, work)%16 == 0, "ncclShmem.work needs to be 16B aligned");

extern __shared__ ncclShmemData ncclShmem;
#if __CUDA_ARCH__ >= 700
  extern __shared__ ulong2 ncclShmemPerWarp[/*ncclShmemDynamicSize()/sizeof(ulong2)*/];
#else
  extern __shared__ ulong2 ncclShmemPerWarp[ncclShmemScratchWarpSize()*(NCCL_MAX_NTHREADS/WARP_SIZE)/sizeof(ulong2)];
#endif

__device__ inline void* ncclScratchForWarp(int warp) {
  return (char*)ncclShmemPerWarp + warp*ncclShmemScratchWarpSize();
}

#ifdef ENABLE_PROFILING
#define __insert_timestamp(line_num) do { \
      if (ncclShmem.prof.count < PROFILE_NUM_ITEMS) { \
        ncclShmem.prof.elem[ncclShmem.prof.count].line = line_num; \
        ncclShmem.prof.elem[ncclShmem.prof.count].timeStamp = wall_clock64(); \
        ncclShmem.prof.count++; \
      } \
    } while(0);
#else
#define __insert_timestamp(line_num)
#endif

// Copy 16-byte aligned data. You must call with at least `(bytes+15)/16` threads.
inline __device__ void copyToShmem16(int tid, void* dst, void const* src, int bytes) {
  int offset = 16*tid;
  if (offset < bytes) {
    ulong2 *src2, *dst2;
    src2 = (ulong2*)((char const*)src + offset);
    dst2 = (ulong2*)((char*)dst + offset);
    dst2->x = src2->x;
    dst2->y = src2->y;
  }
}

template<ncclFunc_t Fn, typename T, typename RedOp, int Algo, int Proto, int COLL_UNROLL>
struct RunWorkElement {
  __device__ void run(ncclWorkElem*) {
    // Put NOT IMPLEMENTED behavior here.
  }
};

template<ncclFunc_t Fn, typename T, typename RedOp, int Algo, int Proto, int COLL_UNROLL>
struct RunWork {
  // This __forceinline__ is necessary. The compiler was inserting a function call
  // here from the LL ncclKernel.
  __device__ __forceinline__ void run(ncclWork *w) {
    int wid = threadIdx.x / WARP_SIZE;
    ncclWorkElem* we = w->header.type == ncclWorkTypeRegColl ? &w->regElems[0].elem : &w->elems[0];
    int stride = w->header.type == ncclWorkTypeRegColl ? sizeof(ncclWorkElemReg) : sizeof(ncclWorkElem);
    #pragma unroll 1
    while ((char*)we + stride <= (char*)(w+1) && we->isUsed) {
      if (wid < we->nWarps) {
        RunWorkElement<Fn, T, RedOp, Algo, Proto, COLL_UNROLL>().run(we);
      }
      we = (ncclWorkElem*)((char*)we + stride);
    }
  }
};

static __forceinline__ __device__ void ncclRedopPtrDeref(struct ncclWorkElem* we) {
  if (we->isUsed && we->redOpArgIsPtr) {
    /* redOpArg is a pointer to the scalar value, so we'll dereference it
     * here so that redOpArg holds the bits of the scalar going forward.
     * The tricky thing is we don't know its type T since that's encoded in
     * the funcIndex. Because it would be difficult to get sizeof(T) from
     * funcIndex, we'll cheat and just dereference the largest possible size
     * given the alignment of the pointer. We might be reading in more bytes
     * than we need but that's harmless.
     */
    if (we->redOpArg%2 != 0)
      we->redOpArg = *reinterpret_cast<uint8_t*>(we->redOpArg);
    else if (we->redOpArg%4 != 0)
      we->redOpArg = *reinterpret_cast<uint16_t*>(we->redOpArg);
    else if (we->redOpArg%8 != 0)
      we->redOpArg = *reinterpret_cast<uint32_t*>(we->redOpArg);
    else
      we->redOpArg = *reinterpret_cast<uint64_t*>(we->redOpArg);
  }
}

template<int SpecializedFnId, typename SpecializedRunWork, bool COLLTRACE, int COLL_UNROLL>
__forceinline__ __device__ void ncclKernelMain(struct ncclDevComm* comm, struct channelMasks channelMask, struct ncclWork* workHead) {
  const int tid = threadIdx.x;
  int x = tid;
  int total = 0, y;
  int num = MAXCHANNELS/64 > 0 ? MAXCHANNELS/64 : 1;

  switch (tid/WARP_SIZE) {
  case 0:
	//ncclShmem.channelId = blockIdx.x;
    for (int i = 0; i < num; i++) {
      if (channelMask.masks[i] & (1ull<<x)) {
        y = __popcll(channelMask.masks[i] & ((1ull<<x)-1));
        y = total + y;
        if (blockIdx.x == y) {
          ncclShmem.channelId = x + total;
	  break;
        }
      }
      if (WARP_SIZE < 64) {
        x = WARP_SIZE + tid;
        if (channelMask.masks[i] & (1ull<<x)) {
	  y = __popcll(channelMask.masks[i] & ((1ull<<x)-1));
	  y = y + total;
          if (blockIdx.x == y) {
	    ncclShmem.channelId = x + total;
	    break;
	  }
        }
      }
      total = total + __popcll(channelMask.masks[i]);
    }
    break;
  case 1:
    if (tid < WARP_SIZE + NCCL_MAX_GROUPS)
      ncclShmem.groups[tid-WARP_SIZE].barrier = 0;
    break;
  case 2:
    if (tid < 2*WARP_SIZE + NCCL_MAX_GROUPS*NCCL_MAX_GROUPS)
      ncclShmem.groups[(tid-2*WARP_SIZE)/NCCL_MAX_GROUPS].barrier_next[(tid-2*WARP_SIZE)%NCCL_MAX_GROUPS] = 0;
    break;
  case 3:
    /* set abort flag to 0 */
    if (tid == 3*WARP_SIZE) ncclShmem.aborted = 0;
    break;
  default:
    break;
  }
  __synclds(); // publish ncclShmem.channelId
  // To map blockId to channelId, we need the n'th set bit of channelMask which
  // is the inverse of counting the number of set bits among the the first n.
  int channelId = ncclShmem.channelId;

  if (true) {
    void *dst, *src;
    int bytes;
    // Use first 3 warps to load comm, channel, and work into shmem
    switch (tid/WARP_SIZE) {
    case 0:
      dst = &ncclShmem.comm;
      src = comm;
      bytes = sizeof(ncclDevComm);
      static_assert(sizeof(ncclDevComm) <= 16*WARP_SIZE, "ncclDevComm cannot be loaded by a single warp in one insn.");
      break;
    case 1:
      // Get address of channel without incurring indirect load from ncclDevComm::channels
      dst = &ncclShmem.channel;
      src = &((ncclDevCommAndChannels*)comm)->channels[channelId];
      bytes = sizeof(ncclDevChannel);
      static_assert(sizeof(ncclDevChannel) <= 16*WARP_SIZE, "ncclDevChannel cannot be loaded by a single warp in one insn.");
      break;
    case 2:
      dst = &ncclShmem.work;
      src = workHead + blockIdx.x;
      bytes = sizeof(ncclWork);
      static_assert(sizeof(ncclWork) <= 16*WARP_SIZE, "ncclWork cannot be loaded by a single warp in one insn.");
      break;
    default:
      bytes = 0;
      break;
    }
    if (bytes) copyToShmem16(tid%WARP_SIZE, dst, src, bytes);
  }
#ifdef ENABLE_COLLTRACE
  if (tid == 0) {
    ncclShmem.collTrace = comm->collTrace + COLLTRACE_NUM_ITEMS*ncclShmem.channelId;
    ncclShmem.collTraceTail = comm->collTraceTail + ncclShmem.channelId;
  }
#endif
  __synclds(); // publish shmem
#ifdef ENABLE_PROFILING
  if (tid == 0) {
    ncclShmem.prof.count = 0;
    ncclShmem.prof.seq = ncclShmem.comm.devProf[blockIdx.x].seq;
  }
#endif
  if (tid == 0) __insert_timestamp(__LINE__);
  if (COLLTRACE && tid == 0) traceKernelLaunch(ncclCollTraceKernelLaunchType);

  while (true) {
    // Notify host that all fifo reads are complete.
    if (tid == 0 && ncclShmem.work.header.isLast && ncclShmem.work.header.inFifo) {
      *ncclShmem.channel.workFifoDone = ncclShmem.work.header.doneAcks;
    }

    __syncwarp();
    if (ncclShmem.work.header.type == ncclWorkTypeColl) {
      if (tid < NCCL_MAX_WORK_ELEMENTS) ncclRedopPtrDeref(&ncclShmem.work.elems[tid]);
    } else if (ncclShmem.work.header.type == ncclWorkTypeRegColl) {
      if (tid < NCCL_MAX_WORK_ELEMENTS_REG) ncclRedopPtrDeref(&ncclShmem.work.regElems[tid].elem);
    }
    __synclds();

    if (tid == 0) __insert_timestamp(__LINE__);

    if (0 <= SpecializedFnId && ncclShmem.work.header.funcIndex == (unsigned)SpecializedFnId) {
      SpecializedRunWork().run(&ncclShmem.work);
    } else {
#ifdef USE_INDIRECT_FUNCTION_CALL
      if (COLL_UNROLL == 4)
        ncclDevFuncTable_4[ncclShmem.work.header.funcIndex]();
      else
        ncclDevFuncTable[ncclShmem.work.header.funcIndex]();
#else
      if (COLL_UNROLL == 4)
        NCCL_CALL_FUNCTIONS_4(ncclShmem.work.header.funcIndex);
      else
        NCCL_CALL_FUNCTIONS(ncclShmem.work.header.funcIndex);
#endif
    }

    int workIxNext = ncclShmem.work.header.workNext;
    __synclds();
    if (ncclShmem.work.header.isLast) break;

    copyToShmem16(tid, &ncclShmem.work, workHead + workIxNext, sizeof(ncclWork));

    { // Check whether the last operation was aborted and make sure all threads exit
      int aborted = tid == 0 ? *comm->abortFlag : 0;
      if (__any(aborted)) { // publish ncclShmem.work
        traceKernelEnd(ncclCollTraceAbortType);
        break;
      }
    }
    if (COLLTRACE && tid == 0) traceKernelLaunch(ncclCollTraceCollLaunchType);
  }
  if (COLLTRACE && tid == 0) traceKernelEnd(ncclCollTraceKernelEndType);

#ifdef ENABLE_PROFILING
  if (ncclShmem.comm.devProf->seq < PROFILE_NUM_LAUNCHES) {
    __synclds();
    copyToShmem16(tid, ncclShmem.comm.devProf+MAXCHANNELS*ncclShmem.prof.seq+blockIdx.x, &ncclShmem.prof, sizeof(struct ncclProf));
    if (tid == 0) ncclShmem.comm.devProf[blockIdx.x].seq++;
  }
#endif
}

__global__ void ncclDevKernel_Generic(struct ncclDevComm* comm, struct channelMasks channelMask, struct ncclWork* workHead);
__global__ void ncclDevKernel_Generic_4(struct ncclDevComm* comm, struct channelMasks channelMask, struct ncclWork* workHead);
#ifdef ENABLE_COLLTRACE
__global__ void ncclDevKernelDebug_Generic(struct ncclDevComm* comm, struct channelMasks channelMask, struct ncclWork* workHead);
__global__ void ncclDevKernelDebug_Generic_4(struct ncclDevComm* comm, struct channelMasks channelMask, struct ncclWork* workHead);
#endif

#ifdef USE_INDIRECT_FUNCTION_CALL
#define DEFINE_ncclDevFunc(suffix, coll, redop, ty, algo, proto) \
  __device__ void ncclDevFunc_##suffix() { \
    RunWork<coll, ty, redop<ty>, algo, proto, 2>().run(&ncclShmem.work); \
  } \
  __device__ void ncclDevFunc_##suffix##_4() { \
    RunWork<coll, ty, redop<ty>, algo, proto, 4>().run(&ncclShmem.work); \
  }
#else
#define DEFINE_ncclDevFunc(suffix, coll, redop, ty, algo, proto) \
  __device__ __attribute__((noinline)) void ncclDevFunc_##suffix() { \
    RunWork<coll, ty, redop<ty>, algo, proto, 2>().run(&ncclShmem.work); \
  } \
  __device__ __attribute__((noinline)) void ncclDevFunc_##suffix##_4() { \
    RunWork<coll, ty, redop<ty>, algo, proto, 4>().run(&ncclShmem.work); \
  }
#endif

#endif
