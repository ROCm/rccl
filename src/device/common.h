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
#include "reduce_kernel.h"
#include "device_table.h"
#include "network/unpack/unpack_defs.h"

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
    collTrace->bid = blockIdx.x; \
    collTrace->channelId = ncclShmem.channelId;
    // TODO: switch to atomicInc after llvm crash is fixed
    // uint32_t pos = atomicInc(&ncclShmem.collTraceTail->tail, COLLTRACE_NUM_ITEMS)

  #define traceKernelLaunch(launch_type) { \
    INC_COLL_TRACE \
    collTrace->funcIndex = ncclShmem.funcId; \
    __trace_hwreg()\
    if (ncclShmem.workType == ncclDevWorkTypeP2p) { \
      struct ncclDevWorkP2p *p2pWork = (struct ncclDevWorkP2p*)ncclShmem.workStorage; \
      collTrace->p2p.sendRank = p2pWork->sendRank; \
      collTrace->p2p.recvRank = p2pWork->recvRank; \
      collTrace->p2p.nP2pChannels = p2pWork->nP2pChannels; \
      collTrace->p2p.nSendChannels = p2pWork->nSendChannels; \
      collTrace->p2p.nRecvChannels = p2pWork->nRecvChannels; \
      collTrace->p2p.channelBase = p2pWork->channelBase; \
      collTrace->p2p.sendConnIndex = p2pWork->sendConnIndex; \
      collTrace->p2p.recvConnIndex = p2pWork->recvConnIndex; \
      collTrace->p2p.sendProtoLL = p2pWork->sendProtoLL; \
      collTrace->p2p.recvProtoLL = p2pWork->recvProtoLL; \
      collTrace->p2pOpCount[0] = p2pWork->sendOpCount; \
      collTrace->p2pOpCount[1] = p2pWork->recvOpCount; \
      collTrace->type = (launch_type) | ncclCollTraceP2pElemType; \
    } else if (ncclShmem.workType == ncclDevWorkTypeColl) { \
      struct ncclDevWorkColl *collWork = (struct ncclDevWorkColl*)ncclShmem.workStorage; \
      collTrace->coll.nWarps = collWork->nWarps; \
      collTrace->coll.nChannels = collWork->channelHi-collWork->channelLo+1; \
      collTrace->coll.bid = ncclShmem.channelId - collWork->channelLo; \
      collTrace->coll.root = collWork->root; \
      collTrace->opCount = collWork->opCount; \
      collTrace->type = (launch_type) | ncclCollTraceCollElemType; \
    } \
  }
  #define traceKernelEnd(end_type)  { \
    INC_COLL_TRACE \
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

#if __CUDA_ARCH__ >= 700
// __grid_constant__ appears to break cuda-gdb
//#define NCCL_GRID_CONSTANT __grid_constant__
#define NCCL_GRID_CONSTANT
#else
#define NCCL_GRID_CONSTANT
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
  struct ncclDevKernelArgs args;
  int channelId;
  int aborted;
  alignas(16) struct ncclDevComm comm;
  alignas(16) struct ncclDevChannel channel;

  int batchIx, nextBatchIx;
  enum ncclDevWorkType workType;
  uint8_t directMode;
  uint16_t funcId;
  int nWorks;
  int workSize;
  uint32_t workConsumed;
  struct ncclShmemGroup groups[NCCL_MAX_GROUPS];
  uint64_t redOpArgs[NCCL_MAX_NVLS_ARITY+1];

  alignas(16) char workStorage[1024];

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

extern __shared__ ncclShmemData ncclShmem;
#if __CUDA_ARCH__ >= 700
  extern __shared__ ulong2 ncclShmemPerWarp[/*ncclShmemDynamicSize()/sizeof(ulong2)*/];
#else
  extern __shared__ ulong2 ncclShmemPerWarp[ncclShmemScratchWarpSize()*(NCCL_MAX_NTHREADS/WARP_SIZE)/sizeof(ulong2)];
#endif

__device__ inline void* ncclScratchForWarp(int warp) {
  return (char*)ncclShmemPerWarp + warp*ncclShmemScratchWarpSize();
}

__device__ inline void barrier_sync(int name) {
  #if 0
  asm volatile("barrier.sync %0;" :: "r"(name) : "memory");
  #else
  asm volatile("barrier.sync.aligned %0;" :: "r"(name) : "memory");
  #endif
}
__device__ inline void barrier_sync(int name, int nThreads) {
  #if 0
  asm volatile("barrier.sync %0, %1;" :: "r"(name), "r"(nThreads) : "memory");
  #else
  asm volatile("barrier.sync.aligned %0, %1;" :: "r"(name), "r"(nThreads) : "memory");
  #endif
}
__device__ inline void barrier_sync_aligned(int name) {
  asm volatile("barrier.sync.aligned %0;" :: "r"(name) : "memory");
}
__device__ inline void barrier_sync_aligned(int name, int nThreads) {
  asm volatile("barrier.sync.aligned %0, %1;" :: "r"(name), "r"(nThreads) : "memory");
}

__device__ inline bool barrier_red_or(bool vote, int name) {
  int ans;
  asm("{ .reg .pred p;"
      "  setp.ne.s32 p, %1, 0;"
      "  barrier.red.or.pred p, %2, p; "
      "  selp.s32 %0, 1, 0, p; }"
      : "=r"(ans) : "r"((int)vote), "r"(name) : "memory");
  return bool(ans);
}
__device__ inline bool barrier_red_or(bool vote, int name, int nThreads) {
  int ans;
  asm("{ .reg .pred p;"
      "  setp.ne.s32 p, %1, 0;"
      "  barrier.red.or.pred p, %2, %3, p; "
      "  selp.s32 %0, 1, 0, p; }"
      : "=r"(ans) : "r"((int)vote), "r"(name), "r"(nThreads) : "memory");
  return bool(ans);
}
__device__ inline bool barrier_red_or_aligned(bool vote, int name) {
  int ans;
  asm("{ .reg .pred p;"
      "  setp.ne.s32 p, %1, 0;"
      "  barrier.red.or.pred.aligned p, %2, p; "
      "  selp.s32 %0, 1, 0, p; }"
      : "=r"(ans) : "r"((int)vote), "r"(name) : "memory");
  return bool(ans);
}
__device__ inline bool barrier_red_or_aligned(bool vote, int name, int nThreads) {
  int ans;
  asm("{ .reg .pred p;"
      "  setp.ne.s32 p, %1, 0;"
      "  barrier.red.or.pred.aligned p, %2, %3, p; "
      "  selp.s32 %0, 1, 0, p; }"
      : "=r"(ans) : "r"((int)vote), "r"(name), "r"(nThreads) : "memory");
  return bool(ans);
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

// Must run with at least 64 threads
__device__ __forceinline__ void loadWorkBatchToShmem(
    int tid, int tn, struct ncclDevKernelArgs const* args, int batchIx
  ) {
  int lane = tid%WARP_SIZE;
  int workCursor = 0; // num works written in previous loop iterations.
  while (true) {
    struct ncclDevWorkBatch batch = ((struct ncclDevWorkBatch*)(args+1))[batchIx];

    // fnsOfBitset[n] = index of n'th set bit in batch.offsetBitset.
    // PTX has instruction "fns" (find n-th set) but it expands to a lot of SASS,
    // since we know all lanes will be querying the same bitmask we can compute
    // much faster using shared memory.
    uint8_t* fnsOfBitset = (uint8_t*)ncclScratchForWarp(threadIdx.x/WARP_SIZE);
    int nWorks = 0;
    __syncwarp();

    if (WARP_SIZE == 64) {
      if (uint64_t(batch.offsetBitset) & (1ull<<lane)) {
        int nWorksBelow = __popc(uint64_t(batch.offsetBitset) & ((1ull<<lane)-1));
        fnsOfBitset[nWorksBelow] = lane;
      }
      nWorks = __popc(uint64_t(batch.offsetBitset));
    } else {
      // WARP_SIZE == 32
      if (uint32_t(batch.offsetBitset) & (1u<<lane)) {
        int nWorksBelow = __popc(uint32_t(batch.offsetBitset) & ((1u<<lane)-1));
        fnsOfBitset[nWorksBelow] = lane;
      }
      int nWorksLow32 = __popc(uint32_t(batch.offsetBitset)); // just of low 32 bits
      if (uint32_t(batch.offsetBitset>>32) & (1u<<lane)) {
        int nWorksBelow = nWorksLow32;
        nWorksBelow += __popc(uint32_t(batch.offsetBitset>>32) & ((1u<<lane)-1));
        fnsOfBitset[nWorksBelow] = 32 + lane;
      }
      nWorks = nWorksLow32 + __popc(uint32_t(batch.offsetBitset>>32)); // add high 32 bits
    }

    int workSize;
    int nPacks; // total number of packs loaded, each pack is 16 bytes
    int packInWork; // my pack index within work struct
    int dstWork; // my work index in contiguous destination shmem
    switch (batch.workType) {
    case (int)ncclDevWorkTypeP2p:
      workSize = sizeof(struct ncclDevWorkP2p);
      nPacks = nWorks*(workSize/16);
      packInWork = tid%(workSize/16);
      dstWork = tid/(workSize/16);
      break;
    case (int)ncclDevWorkTypeColl:
      workSize = sizeof(struct ncclDevWorkColl);
      nPacks = nWorks*(workSize/16);
      packInWork = tid%(workSize/16);
      dstWork = tid/(workSize/16);
      break;
    case (int)ncclDevWorkTypeCollReg:
    default:
      workSize = sizeof(struct ncclDevWorkCollReg);
      nPacks = nWorks*(workSize/16);
      packInWork = tid%(workSize/16);
      dstWork = tid/(workSize/16);
      break;
    }
    if (tid == 0) {
      ncclShmem.workSize = workSize;
      ncclShmem.workConsumed = batch.offsetBase + (64-__clzll(batch.offsetBitset))*workSize;
    }
    // We deliberately replicate these div and mod calculations into the case
    // blocks above so that they get constant divisor optimizations by the compiler.
    //   packInWork = tid%(workSize/16);
    //   dstWork = tid/(workSize/16);

    // We can only assume we have 64 threads, which means we can read at most 1024 bytes
    // here which is the per batch maximum.
    if (tid < nPacks) {
      int srcWork = fnsOfBitset[dstWork]; // find n'th set bit in batch.offsetBitset
      ulong2 tmp;
      // The loads done in these two cases must be kept separate since we are
      // relying on the compiler to use "ld.param" in the first one. The parameter
      // space is not generically addressable, so any attempt to load through
      // a pointer that *might* be parameter space backed will cause the
      // compiler to spill the parameter struct (4K!) to each thread's local space
      // before creating a pointer (to the spill) and decimate perf.
      //
      // An example of what not to do would be the following:
      //
      // if (condition) {
      //   // The compiler could spill parameter_variable to local space and take
      //   // the address of that, since when src is loaded below it could also
      //   // be global space.
      //   src = &parameter_variable;
      // } else {
      //   src = &global_variable;
      // }
      // memcpy(dst, src, n);
      if (ncclShmem.args.workStorageType == ncclDevWorkStorageTypeArgs) {
        char* src = (char*)args + (batch.offsetBase + srcWork*workSize + packInWork*16);
        tmp = *(ulong2*)src; // becomes ld.param.v2.u64
      } 
      if (ncclShmem.args.workStorageType != ncclDevWorkStorageTypeArgs) {
        char* src = (char*)ncclShmem.args.workBuf + ((batch.offsetBase + srcWork*workSize + packInWork*16) & ncclShmem.args.workMask);
        tmp = *(ulong2*)src; // becomes ld.v2.u64
      }
      char* dst = ncclShmem.workStorage;
      dst += (workCursor + dstWork)*workSize + packInWork*16;
      *(ulong2*)dst = tmp;
    }
    workCursor += nWorks;

    if (batch.nextExtends) {
      batchIx += batch.nextJump;
      tid -= 64; // Rotate threads so we use the next two warps for next batch struct.
      if (tid < 0) tid += tn;
    } else {
      if (tid == 0) {
        ncclShmem.batchIx = batchIx;
        ncclShmem.nextBatchIx = (batch.nextJump == 0) ? -1 : batchIx + batch.nextJump;
        ncclShmem.workType = (enum ncclDevWorkType)batch.workType;
        ncclShmem.nWorks = workCursor;
        ncclShmem.funcId = batch.funcId;
      }
      break;
    }
  }
}

template<ncclFunc_t Fn, typename T, typename RedOp, int Algo, int Proto, int COLL_UNROLL>
struct RunWorkColl {
  __device__ void run(int tid, int tn, struct ncclDevWorkColl* work) {
    // Put NOT IMPLEMENTED behavior here.
  }
};

template<ncclFunc_t Fn, typename T, typename RedOp, int Algo, int Proto, int COLL_UNROLL>
struct RunWorkBatch;

// Specialized for P2p in sendrecv.h
template<typename T, typename RedOp>
struct RunWorkBatch<ncclFuncSendRecv, T, RedOp, NCCL_ALGO_RING, NCCL_PROTO_SIMPLE>;

// Specialized here for non-P2p (Coll and CollReg)
template<ncclFunc_t Fn, typename T, typename RedOp, int Algo, int Proto, int COLL_UNROLL>
struct RunWorkBatch {
  // This __forceinline__ is necessary. The compiler was inserting a function call
  // here from the LL ncclKernel.
  __device__ __forceinline__ void run() {
    int tid = threadIdx.x;
    int tn = blockDim.x;

    if (RedOpArg<RedOp>::ArgUsed) {
      int nWorks = ncclShmem.nWorks;
      for (int w=tid; w < nWorks; w += tn) {
        struct ncclDevWorkColl* work = (ncclDevWorkColl*)(ncclShmem.workStorage + w*ncclShmem.workSize);
        if (work->redOpArgIsPtr) {
          work->redOpArg = RedOpArg<RedOp>::loadArg(reinterpret_cast<void*>(work->redOpArg));
        }
      }
      __synclds();
    }

    #pragma unroll 1
    for (int w=0; w < ncclShmem.nWorks; w++) {
      struct ncclDevWorkColl* work = (struct ncclDevWorkColl*)(ncclShmem.workStorage + w*ncclShmem.workSize);
      if (w != 0) {
        struct ncclDevWorkColl* workPrev = (struct ncclDevWorkColl*)(ncclShmem.workStorage + (w-1)*ncclShmem.workSize);
        if (work->nWarps != workPrev->nWarps) __synclds();
      }
      int subtn = work->nWarps*WARP_SIZE;
      if (tid < subtn) RunWorkColl<Fn, T, RedOp, Algo, Proto, COLL_UNROLL>().run(tid, subtn, work);
    }
  }
};

template<int SpecializedFnId, typename SpecializedRunWorkBatch, bool COLLTRACE, int COLL_UNROLL>
__device__ __forceinline__ void ncclKernelMain(struct ncclDevKernelArgs const* args) {
  const int tid = threadIdx.x;
  int tn = blockDim.x;
  int x = tid;
  int total = 0, y;
  int num = MAXCHANNELS/64 > 0 ? MAXCHANNELS/64 : 1;

  // Copy kernel args to shmem and then only read those. Otherwise the compiler
  // will end up putting the args into thread local stack which is very wasteful.
  if (tid < sizeof(ncclDevKernelArgs)/sizeof(uint32_t)) {
    ((uint32_t*)&ncclShmem.args)[tid] = ((uint32_t*)args)[tid];
  }

  // To map blockId to channelId, we need the n'th set bit of channelMask which
  // is the inverse of counting the number of set bits among the the first n.
  // PTX has the fns instruction which does this but is extremely slow. We can
  // do better when we know all threads are querying the same bitmask.
  switch (tid/WARP_SIZE) {
  case 0:
	//ncclShmem.channelId = blockIdx.x;
    for (int i = 0; i < num; i++) {
      if (args->channelMask.masks[i] & (1ull<<x)) {
        y = __popcll(args->channelMask.masks[i] & ((1ull<<x)-1));
        y = total + y;
        if (blockIdx.x == y) {
          ncclShmem.channelId = x + total;
	        break;
        }
      }
      if (WARP_SIZE < 64) {
        x = WARP_SIZE + tid;
        if (args->channelMask.masks[i] & (1ull<<x)) {
	        y = __popcll(args->channelMask.masks[i] & ((1ull<<x)-1));
	        y = y + total;
          if (blockIdx.x == y) {
	          ncclShmem.channelId = x + total;
	          break;
	        }
        }
      }
      total = total + __popcll(args->channelMask.masks[i]);
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
  __synclds(); // publish ncclShmem.{args, channelId}

  // Use first 2 warps to load comm and channel, and reamaining load work batch.
  switch (tid/WARP_SIZE) {
  case 0:
    { void* dst = &ncclShmem.comm;
      void* src = ncclShmem.args.comm;
      int bytes = sizeof(ncclDevComm);
      static_assert(sizeof(ncclDevComm) <= 16*WARP_SIZE, "ncclDevComm cannot be loaded by a single warp in one insn.");
      copyToShmem16(tid, dst, src, bytes);
    } break;
  case 1:
    { // Get address of channel without incurring indirect load from ncclDevComm::channels
      void* dst = &ncclShmem.channel;
      void* src = &((ncclDevCommAndChannels*)ncclShmem.args.comm)->channels[ncclShmem.channelId];
      int bytes = sizeof(ncclDevChannel);
      static_assert(sizeof(ncclDevChannel) <= 16*WARP_SIZE, "ncclDevChannel cannot be loaded by a single warp in one insn.");
      copyToShmem16(tid-WARP_SIZE, dst, src, bytes);
    } break;
  default:
    { int subtid = tid - 2*WARP_SIZE;
      int subtn = tn - 2*WARP_SIZE;
      loadWorkBatchToShmem(subtid, subtn, args, /*batchIx=*/blockIdx.x);
    } break;
  }
#ifdef ENABLE_COLLTRACE
  if (tid == 0) {
    ncclShmem.collTrace = args->comm->collTrace + COLLTRACE_NUM_ITEMS*ncclShmem.channelId;
    ncclShmem.collTraceTail = args->comm->collTraceTail + ncclShmem.channelId;
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

  if (tid == 0 && ncclShmem.args.workStorageType == ncclDevWorkStorageTypeFifo) {
    // ncclShmem.workConsumed written by loadWorkBatchToShmem before __syncthreads()
    ncclShmem.comm.workConsumed[ncclShmem.channelId] = ncclShmem.workConsumed;
  }

  while (true) {
    if (tid == 0) __insert_timestamp(__LINE__);
    
    if (0 <= SpecializedFnId && ncclShmem.funcId == (unsigned)SpecializedFnId) {
      SpecializedRunWorkBatch().run();
    } else {
#ifdef USE_INDIRECT_FUNCTION_CALL
      if (COLL_UNROLL == 4)
        ncclDevFuncTable_4[ncclShmem.funcId]();
      else
        ncclDevFuncTable[ncclShmem.funcId]();
#else
      if (COLL_UNROLL == 4)
        NCCL_CALL_FUNCTIONS_4(ncclShmem.funcId);
      else
        NCCL_CALL_FUNCTIONS(ncclShmem.funcId);
#endif
    }

    if (ncclShmem.nextBatchIx == -1) break;
    int batchIx = ncclShmem.nextBatchIx;
    __synclds();
    loadWorkBatchToShmem(tid%WARP_SIZE, tn, args, batchIx);

    // Check whether the last operation was aborted and make sure all threads exit
    bool aborted = false;
    if (tid == 0) aborted = *ncclShmem.comm.abortFlag;
    aborted = __any(aborted); // publish ncclShmem.work
    if (tid == 0 && ncclShmem.args.workStorageType == ncclDevWorkStorageTypeFifo) {
      // ncclShmem.workConsumed written by loadWorkBatchToShmem before barrier_red_or()
      ncclShmem.comm.workConsumed[ncclShmem.channelId] = ncclShmem.workConsumed;
    }
    if (aborted) break;

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

__global__ void ncclDevKernel_Generic(ncclDevKernelArgs4K NCCL_GRID_CONSTANT const args4K);
__global__ void ncclDevKernel_Generic_4(ncclDevKernelArgs4K NCCL_GRID_CONSTANT const args4K);
#ifdef ENABLE_COLLTRACE
__global__ void ncclDevKernelDebug_Generic(ncclDevKernelArgs4K NCCL_GRID_CONSTANT const args4K);
__global__ void ncclDevKernelDebug_Generic_4(ncclDevKernelArgs4K NCCL_GRID_CONSTANT const args4K);
#endif

#ifdef USE_INDIRECT_FUNCTION_CALL
#define DEFINE_ncclDevFunc(suffix, coll, redop, ty, algo, proto, unroll) \
  __device__ void ncclDevFunc_##suffix() { \
    RunWorkBatch<coll, ty, redop<ty>, algo, proto, unroll>().run(); \
  }
#else
#define DEFINE_ncclDevFunc(suffix, coll, redop, ty, algo, proto, unroll) \
  __device__ __attribute__((noinline)) void ncclDevFunc_##suffix() { \
    RunWorkBatch<coll, ty, redop<ty>, algo, proto, unroll>().run(); \
  }
#endif

#endif
