/*************************************************************************
 * Copyright (c) 2017-2021, NVIDIA CORPORATION. All rights reserved.
 * Modifications Copyright (c) 2019-2021 Advanced Micro Devices, Inc. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

#ifndef NCCL_DEVICE_COMMON_H_
#define NCCL_DEVICE_COMMON_H_

#include "collectives.h"
#include "devcomm.h"

#define COLL_UNROLL 2
#define NCCL_MAX_DEV_ARITY (NCCL_MAX_TREE_ARITY-1)  // Using balanced tree instead of split tree

#define __syncwarp()

#define NCCL_FUNC5(func, algo, devredop, type, nullify) \
  MACRO_IF(nullify, nullptr, NCCL_FUNC_NAME(func, algo, LL,     devredop, type)), \
  MACRO_IF(nullify, nullptr, NCCL_FUNC_NAME(func, algo, LL,  devredop, type)), \
  MACRO_IF(nullify, nullptr, NCCL_FUNC_NAME(func, algo, SIMPLE, devredop, type))

#define NCCL_FUNC4(func, devredop, type, nullify) \
  NCCL_FUNC5(func, TREE,    devredop, type, nullify), \
  NCCL_FUNC5(func, RING,    devredop, type, nullify), \
  NCCL_FUNC5(func, COLLNET, devredop, type, nullify)

// Must be consistent with ncclDataType_t
#define NCCL_FUNCS3A(func, devredop, nullForFloat) \
  NCCL_FUNC4(func, devredop, int8_t, 0), \
  NCCL_FUNC4(func, devredop, uint8_t, 0), \
  NCCL_FUNC4(func, devredop, int32_t, 0), \
  NCCL_FUNC4(func, devredop, uint32_t, 0), \
  NCCL_FUNC4(func, devredop, int64_t, 0), \
  NCCL_FUNC4(func, devredop, uint64_t, 0), \
  NCCL_FUNC4(func, devredop, half, nullForFloat), \
  NCCL_FUNC4(func, devredop, float, nullForFloat), \
  NCCL_FUNC4(func, devredop, double, nullForFloat), \
  NCCL_FUNC4(func, devredop, rccl_bfloat16, nullForFloat)
#define NCCL_FUNCS3B(func, devredop) \
  NCCL_FUNC4(func, devredop, int8_t, 0), \
  NCCL_FUNC4(func, devredop, int8_t, 0), \
  NCCL_FUNC4(func, devredop, int8_t, 0), \
  NCCL_FUNC4(func, devredop, int8_t, 0), \
  NCCL_FUNC4(func, devredop, int8_t, 0), \
  NCCL_FUNC4(func, devredop, int8_t, 0), \
  NCCL_FUNC4(func, devredop, int8_t, 0), \
  NCCL_FUNC4(func, devredop, int8_t, 0), \
  NCCL_FUNC4(func, devredop, int8_t, 0), \
  NCCL_FUNC4(func, devredop, int8_t, 0)

// Must be consistent with ncclRedOp_t
#define NCCL_FUNCS2A(func) \
  NCCL_FUNCS3A(func, Sum,        /*nullForFloat=*/0), \
  NCCL_FUNCS3A(func, Prod,       /*nullForFloat=*/0), \
  NCCL_FUNCS3A(func, Max,        /*nullForFloat=*/0), \
  NCCL_FUNCS3A(func, Min,        /*nullForFloat=*/0), \
  NCCL_FUNCS3A(func, PreMulSum,  /*nullForFloat=*/0), \
  NCCL_FUNCS3A(func, SumPostDiv, /*nullForFloat=*/1)

#define NCCL_FUNCS2B(func) \
  NCCL_FUNCS3B(func, Sum), \
  NCCL_FUNCS3B(func, Sum), \
  NCCL_FUNCS3B(func, Sum), \
  NCCL_FUNCS3B(func, Sum), \
  NCCL_FUNCS3B(func, Sum), \
  NCCL_FUNCS3B(func, Sum)

// [RCCL] Adding clique-based kernels for AllReduce, in-place of unused RingLL28 kernels
#define NCCL_FUNC5B(func, algo, devredop, type, nullify) \
  MACRO_IF(nullify, nullptr, NCCL_FUNC_NAME(func, algo, LL,     devredop, type)), \
  MACRO_IF(nullify, nullptr, NCCL_FUNC_NAME(func, algo, LL128,  devredop, type)), \
  MACRO_IF(nullify, nullptr, NCCL_FUNC_NAME(func, algo, SIMPLE, devredop, type))

#define NCCL_FUNC4B(func, devredop, type, nullify) \
  NCCL_FUNC5B(func, TREE,    devredop, type, nullify), \
  NCCL_FUNC5B(func, RING,    devredop, type, nullify), \
  NCCL_FUNC5B(func, COLLNET, devredop, type, nullify)

#define NCCL_FUNCS3C(func, devredop, nullForFloat) \
  NCCL_FUNC4B(func, devredop, int8_t, 0), \
  NCCL_FUNC4B(func, devredop, uint8_t, 0), \
  NCCL_FUNC4B(func, devredop, int32_t, 0), \
  NCCL_FUNC4B(func, devredop, uint32_t, 0), \
  NCCL_FUNC4B(func, devredop, int64_t, 0), \
  NCCL_FUNC4B(func, devredop, uint64_t, 0), \
  NCCL_FUNC4B(func, devredop, half, nullForFloat), \
  NCCL_FUNC4B(func, devredop, float, nullForFloat), \
  NCCL_FUNC4B(func, devredop, double, nullForFloat), \
  NCCL_FUNC4B(func, devredop, rccl_bfloat16, nullForFloat)

#define NCCL_FUNCS2C(func) \
  NCCL_FUNCS3C(func, Sum,        /*nullForFloat=*/0), \
  NCCL_FUNCS3C(func, Prod,       /*nullForFloat=*/0), \
  NCCL_FUNCS3C(func, Max,        /*nullForFloat=*/0), \
  NCCL_FUNCS3C(func, Min,        /*nullForFloat=*/0), \
  NCCL_FUNCS3C(func, PreMulSum,  /*nullForFloat=*/0), \
  NCCL_FUNCS3C(func, SumPostDiv, /*nullForFloat=*/1)


// Must be consistent with the ncclFuncSet enum
using ncclKernelFunc_t = void (*)(struct ncclWorkElem* args);

static const __device__ constexpr ncclKernelFunc_t ncclFuncs[]{
// Don't try to initialize the host shadow copy of this device-side global
// variable. There is no host pointer to a device-side function, which
// confuses clang. This will be fixed in the next clang release.
#if defined(__HIP_DEVICE_COMPILE__)
#if defined(BUILD_ALLREDUCE_ONLY)
  NCCL_FUNC4B(AllReduce, Sum, float, 0),
#else
  NCCL_FUNCS2B(Broadcast),
  NCCL_FUNCS2A(Reduce),
  NCCL_FUNCS2B(AllGather),
  NCCL_FUNCS2A(ReduceScatter),
  NCCL_FUNCS2C(AllReduce),
  NCCL_ONERANK_REDUCE_NAME(PreMulSum, int8_t),
  NCCL_ONERANK_REDUCE_NAME(PreMulSum, uint8_t),
  NCCL_ONERANK_REDUCE_NAME(PreMulSum, int32_t),
  NCCL_ONERANK_REDUCE_NAME(PreMulSum, uint32_t),
  NCCL_ONERANK_REDUCE_NAME(PreMulSum, int64_t),
  NCCL_ONERANK_REDUCE_NAME(PreMulSum, uint64_t),
  NCCL_ONERANK_REDUCE_NAME(PreMulSum, half),
  NCCL_ONERANK_REDUCE_NAME(PreMulSum, float),
  NCCL_ONERANK_REDUCE_NAME(PreMulSum, double),
#if defined(RCCL_BFLOAT16)
  NCCL_ONERANK_REDUCE_NAME(PreMulSum, rccl_bfloat16),
#endif
  NCCL_FUNC_NAME(SendRecv, RING, SIMPLE, Sum, int8_t),
#endif
#endif
};

template<unsigned short f, unsigned short l>
struct Caller {
  static __device__ __host__
  void call(struct ncclWorkElem* const c) noexcept
  {
    constexpr unsigned short m = f + (l - f) / 2;

     return (c->funcIndex < m) ? Caller<f, m>::call(c) : Caller<m, l>::call(c);
  }
};

template<unsigned short f>
struct Caller<f, f + 1>{
  static __device__ __host__
  void call(struct ncclWorkElem* const c) noexcept { ncclFuncs[f](c); }
};

static_assert(FUNC_INDEX_P2P == 2710, "Wrong P2P function index");

inline
__device__
void NCCL_CALL_FUNCTIONS(struct ncclWorkElem* const c) noexcept {
#if defined(BUILD_ALLREDUCE_ONLY)
  if (c->funcIndex == FUNC_INDEX(ncclFuncAllReduce, ncclSum, ncclFloat32, NCCL_ALGO_RING, NCCL_PROTO_SIMPLE))
    ncclFunction_AllReduce_RING_SIMPLE_Sum_float(c);
  else if (c->funcIndex == FUNC_INDEX(ncclFuncAllReduce, ncclSum, ncclFloat32, NCCL_ALGO_RING, NCCL_PROTO_LL))
    ncclFunction_AllReduce_RING_LL_Sum_float(c);
  else if (c->funcIndex == FUNC_INDEX(ncclFuncAllReduce, ncclSum, ncclFloat32, NCCL_ALGO_RING, NCCL_PROTO_LL128))
    ncclFunction_AllReduce_RING_LL128_Sum_float(c);
  else if (c->funcIndex == FUNC_INDEX(ncclFuncAllReduce, ncclSum, ncclFloat32, NCCL_ALGO_TREE, NCCL_PROTO_SIMPLE))
    ncclFunction_AllReduce_TREE_SIMPLE_Sum_float(c);
  else if (c->funcIndex == FUNC_INDEX(ncclFuncAllReduce, ncclSum, ncclFloat32, NCCL_ALGO_TREE, NCCL_PROTO_LL))
    ncclFunction_AllReduce_TREE_LL_Sum_float(c);
  else if (c->funcIndex == FUNC_INDEX(ncclFuncAllReduce, ncclSum, ncclFloat32, NCCL_ALGO_COLLNET, NCCL_PROTO_SIMPLE))
    ncclFunction_AllReduce_COLLNET_SIMPLE_Sum_float(c);
  else if (c->funcIndex == FUNC_INDEX(ncclFuncAllReduce, ncclSum, ncclFloat32, NCCL_ALGO_COLLNET, NCCL_PROTO_LL))
    ncclFunction_AllReduce_COLLNET_LL_Sum_float(c);
  else
    assert("Unsupported function index");
#else
  if (c->funcIndex < 540) {
    if (c->funcIndex % 9 == 0) ncclFunction_Broadcast_TREE_LL_Sum_int8_t(c);
    else if (c->funcIndex % 9 == 1) ncclFunction_Broadcast_TREE_LL_Sum_int8_t(c);
    else if (c->funcIndex % 9 == 2) ncclFunction_Broadcast_TREE_SIMPLE_Sum_int8_t(c);
    else if (c->funcIndex % 9 == 3) ncclFunction_Broadcast_RING_LL_Sum_int8_t(c);
    else if (c->funcIndex % 9 == 4) ncclFunction_Broadcast_RING_LL_Sum_int8_t(c);
    else if (c->funcIndex % 9 == 5) ncclFunction_Broadcast_RING_SIMPLE_Sum_int8_t(c);
    else if (c->funcIndex % 9 == 6) ncclFunction_Broadcast_COLLNET_LL_Sum_int8_t(c);
    else if (c->funcIndex % 9 == 7) ncclFunction_Broadcast_COLLNET_LL_Sum_int8_t(c);
    else ncclFunction_Broadcast_COLLNET_SIMPLE_Sum_int8_t(c);
  }
  else if (c->funcIndex < 1080) Caller<540, 1080>::call(c);
  else if (c->funcIndex < 1620) {
    if (c->funcIndex % 9 == 0) ncclFunction_AllGather_TREE_LL_Sum_int8_t(c);
    else if (c->funcIndex % 9 == 1) ncclFunction_AllGather_TREE_LL_Sum_int8_t(c);
    else if (c->funcIndex % 9 == 2) ncclFunction_AllGather_TREE_SIMPLE_Sum_int8_t(c);
    else if (c->funcIndex % 9 == 3) ncclFunction_AllGather_RING_LL_Sum_int8_t(c);
    else if (c->funcIndex % 9 == 4) ncclFunction_AllGather_RING_LL_Sum_int8_t(c);
    else if (c->funcIndex % 9 == 5) ncclFunction_AllGather_RING_SIMPLE_Sum_int8_t(c);
    else if (c->funcIndex % 9 == 6) ncclFunction_AllGather_COLLNET_LL_Sum_int8_t(c);
    else if (c->funcIndex % 9 == 7) ncclFunction_AllGather_COLLNET_LL_Sum_int8_t(c);
    else ncclFunction_AllGather_COLLNET_SIMPLE_Sum_int8_t(c);
  }
  else if (c->funcIndex < 2700) Caller<1620, 2700>::call(c);
  else {
    switch (c->funcIndex - 2700) {
      case 0:
        ncclFunction_OneRankReduce_PreMulSum_int8_t(c);
        break;
      case 1:
        ncclFunction_OneRankReduce_PreMulSum_uint8_t(c);
        break;
      case 2:
        ncclFunction_OneRankReduce_PreMulSum_int32_t(c);
        break;
      case 3:
        ncclFunction_OneRankReduce_PreMulSum_uint32_t(c);
        break;
      case 4:
        ncclFunction_OneRankReduce_PreMulSum_int64_t(c);
        break;
      case 5:
        ncclFunction_OneRankReduce_PreMulSum_uint64_t(c);
        break;
      case 6:
        ncclFunction_OneRankReduce_PreMulSum_half(c);
        break;
      case 7:
        ncclFunction_OneRankReduce_PreMulSum_float(c);
        break;
      case 8:
        ncclFunction_OneRankReduce_PreMulSum_double(c);
        break;
      case 9:
        ncclFunction_OneRankReduce_PreMulSum_rccl_bfloat16(c);
        break;
      case 10:
        ncclFunction_SendRecv_RING_SIMPLE_Sum_int8_t(c);
        break;
      default:
        break;
    }
  }
#endif
}

template <ncclFunc_t FUNCTION, int ALGO, int PROTO, class REDOP, typename T, int UNROLL>
class ncclFunction {
  public:
  __device__ __attribute__((noinline)) void run(struct ncclWorkElem* args) {}
};

#ifdef ENABLE_COLLTRACE
#define traceColl(fIdx)  \
    uint32_t pos = __atomic_fetch_add(shmem.comm.collTraceTail, 1, __ATOMIC_SEQ_CST)%COLLTRACE_NUM_ITEMS; \
    shmem.comm.collTrace[pos].timeStamp = __builtin_amdgcn_s_memrealtime(); \
    shmem.comm.collTrace[pos].bid = bid; \
    shmem.comm.collTrace[pos].funcIndex = fIdx; \
    if (fIdx == FUNC_INDEX_P2P) { \
      shmem.comm.collTrace[pos].opCount = elems[0].p2p.opCount; \
      shmem.comm.collTrace[pos].p2p.nThreads = elems[0].p2p.nThreads; \
      shmem.comm.collTrace[pos].p2p.delta = (uint16_t)(elems[0].p2p.delta); \
    } else { \
      shmem.comm.collTrace[pos].opCount = elems[0].coll.opCount; \
      shmem.comm.collTrace[pos].coll.nThreads = elems[0].nThreads; \
      shmem.comm.collTrace[pos].coll.bid = elems[0].coll.bid; \
      shmem.comm.collTrace[pos].coll.nChannels = elems[0].coll.nChannels; \
    }
#define traceKernelLaunch(fIdx)  { \
    if (!(fIdx == FUNC_INDEX_P2P && elems[0].p2p.nThreads == 0)) { \
      traceColl(fIdx); \
      asm volatile ("s_getreg_b32 %0, hwreg(HW_REG_HW_ID)" : "=s" (shmem.comm.collTrace[pos].data_0)); \
      shmem.comm.collTrace[pos].type = ncclCollTraceKernelLaunchType; \
    } \
  }
#define traceCollEnd(fIdx)  { \
    if (!(fIdx == FUNC_INDEX_P2P && elems[0].p2p.nThreads == 0)) { \
      traceColl(fIdx); \
      shmem.comm.collTrace[pos].type = ncclCollTraceCollEndType; \
    } \
  }
#define traceKernelEnd(fIdx)  { \
    if (!(fIdx == FUNC_INDEX_P2P && elems[0].p2p.nThreads == 0)) { \
      traceColl(fIdx); \
      shmem.comm.collTrace[pos].type = ncclCollTraceKernelEndType; \
    } \
  }
#define traceAbort(fIdx)  { \
    if (!(fIdx == FUNC_INDEX_P2P && elems[0].p2p.nThreads == 0)) { \
      traceColl(fIdx); \
      shmem.comm.collTrace[pos].type = ncclCollTraceAbortType; \
    } \
  }
//  traceData(int16_t data2, uint32_t data4, uint64_t data8_0, uint64_t data8_1)
#define traceData(data2, data4, data8_0, data8_1) { \
    uint32_t pos = __atomic_fetch_add(ncclShmem->comm.collTraceTail, 1, __ATOMIC_SEQ_CST)%COLLTRACE_NUM_ITEMS; \
    ncclShmem->comm.collTrace[pos].bid = blockIdx.x; \
    ncclShmem->comm.collTrace[pos].timeStamp = __builtin_amdgcn_s_memrealtime(); \
    ncclShmem->comm.collTrace[pos].funcIndex = data2; \
    ncclShmem->comm.collTrace[pos].data_0 = data4; \
    ncclShmem->comm.collTrace[pos].opCount = data8_0; \
    ncclShmem->comm.collTrace[pos].data_1 = data8_1; \
    ncclShmem->comm.collTrace[pos].type = ncclCollTraceDataType; \
  }
#else
#define traceKernelLaunch(fIdx)
#define traceCollEnd(fIdx)
#define traceAbort(fIdx)
#define traceData(data2, data4, data8_0, data8_1)
#endif

__device__ inline bool barrierReduceAny(int bit, uint32_t* abortCount) {
  if (bit) atomicAdd(abortCount, 1); \
  __syncthreads(); \
  return atomicAdd(abortCount, 0) != 0;
}

template<typename T>
__device__ int copyToShmem(T *dst, T const *src, int turn=0) {
  static_assert(sizeof(uint64_t) <= alignof(T), "Uhoh");
  uint64_t *d = reinterpret_cast<uint64_t*>(dst);
  uint64_t const *s = reinterpret_cast<uint64_t const*>(src);
  int t = threadIdx.x - turn;
  if (t < 0) t += blockDim.x;
  int n = sizeof(T)/sizeof(uint64_t);

  int delta = (n + WARP_SIZE-1) & -WARP_SIZE; // round up to warp lane 0
  if (delta < blockDim.x) {
    turn += delta;
    if (turn >= blockDim.x) turn -= blockDim.x;
  }
  else
    turn = 0;

  n -= t;
  d += t;
  s += t;
  #pragma unroll
  for (int i=0; i < divUp(sizeof(T), WARP_SIZE*sizeof(uint64_t)); i++) {
    if (n > 0) {
      *d = *s;
      d += blockDim.x;
      s += blockDim.x;
      n -= blockDim.x;
    }
  }
  return turn;
}

template<ncclFunc_t Fn, typename T, typename RedOp, int Algo, int Proto>
struct RunWorkElement {
  __device__ void run(ncclWorkElem*) {
    // Put NOT IMPLEMENTED behavior here.
  }
};

#if CUDART_VERSION >= 11030
__device__ constexpr int ncclWorkElemFactors[NCCL_NUM_ALGORITHMS] =
#else
static __device__ __constant__ int ncclWorkElemFactors[NCCL_NUM_ALGORITHMS] =
#endif
{/*Tree*/1, /*Ring and P2P*/1, /*CollNet*/NCCL_REG_ELEM_FACTOR};

template<ncclFunc_t Fn, typename T, typename RedOp, int Algo, int Proto>
struct RunWork {
  // This __forceinline__ is necessary. The compiler was inserting a function call
  // here from the LL ncclKernel.
  __device__ __forceinline__ void run(ncclWork *w) {
    int tid = threadIdx.x;
    /* Some invariants that must hold:
     * 1. All elems[] have same funcIndex.
     * 2. All elems[] have same nThreads.
     * 3. The thread-to-group relation (as in prims group numbers) is the same
     *    for all elems[].
     *
     * If (1) isn't true then we might be in the wrong function since dispatch
     * on ncclFuncs[w->funcIndex] is how we got here.
     *
     * If (2) or (3) aren't true, then threads from different work elements
     * could race for barrier resources (barrier numbers 0...15) which is fatal.
     *
     * IMPORTANT!!! To ensure (3), implementations of
     * `RunWorkElement<Fn,T,RedOp,Algo,Proto>::run()` may only use the following
     * when deciding how to map threads to groups:
     *    Fn, T, RedOp, Algo, Proto, nThreads
     *
     * This last one is difficult to enforce so I hope everyone reads this.
     */
    if (tid < w->elems[0].nThreads) {
      #pragma unroll 1
      for(int e=0; e < NCCL_MAX_WORK_ELEMENTS && w->elems[e].active != 0; e+=ncclWorkElemFactors[Algo])
        RunWorkElement<Fn, T, RedOp, Algo, Proto>().run(&w->elems[e]);
    }
  }
};

#define MAXWARPS (NCCL_MAX_NTHREADS/WARP_SIZE)
struct ncclShmemGroup {
  ncclConnInfo *recvConns[NCCL_MAX_DIRECT_ARITY];
  ncclConnInfo *sendConns[NCCL_MAX_DIRECT_ARITY];
  void* srcs[NCCL_MAX_DIRECT_ARITY+1];
  void* dsts[NCCL_MAX_DIRECT_ARITY+1];
  int totalSendSize[NCCL_MAX_SLICE_PER_CHUNK];
  uint64_t barrier;
  uint64_t barrier_next[MAXWARPS];
};

struct ncclShmemData {
  union {
    uint64_t ll128warp[NCCL_MAX_GROUPS][NCCL_MAX_GROUPS];
    struct ncclShmemGroup groups[NCCL_MAX_GROUPS];
  };
  uint32_t sync[MAXWARPS];
  uint64_t redOpArgs[NCCL_MAX_DIRECT_ARITY+1];
  ncclDevComm comm;
  ncclChannel channel;
  ncclWork work;
};

extern __device__ struct ncclShmemData *ncclShmem;

template<ncclFunc_t Fn, typename T, typename RedOp, int Algo, int Proto, int FnIndex, bool COLLTRACE>
__device__ void ncclKernel(ncclWorkElem first)  {
  int tid = threadIdx.x;
  int bid = blockIdx.x;
  __shared__ struct ncclShmemData shmem;
  ncclShmem = &shmem;
  __shared__ uint32_t abortCount;
  if (tid == 0) {
    abortCount = 0;
    for (auto i = 0; i < NCCL_MAX_GROUPS; i++) {
      shmem.groups[i].barrier = 0;
      for (auto j = 0; j < MAXWARPS; j++) shmem.groups[i].barrier_next[j] = 0;
    }
  }
  __syncthreads();

  int turn = copyToShmem(&shmem.comm, first.comm);
  // get address of channel without incurring indirect load from ncclDevCom::channels
  ncclChannel *channel = &((ncclDevCommAndChannels*)first.comm)->channels[bid];
  turn = copyToShmem(&shmem.channel, channel, turn);

  // To optimize for latency, (only) the first operation is passed as argument.
  if (bid == 0 && first.active != 0) {
    turn = copyToShmem(&shmem.work.elems[0], &first, turn);
    if (1 <= tid && tid < NCCL_MAX_WORK_ELEMENTS && tid % ncclWorkElemFactors[Algo] == 0) {
      shmem.work.elems[tid].active = 0;
      shmem.work.elems[tid].redOpArgIsPtr = 0;
    }
  }
  struct ncclWorkElem* elems = shmem.work.elems;
  __syncthreads(); // publish shmem

  ncclWork *workFifoHost = shmem.channel.workFifo;
  ncclWork *workFifoDev = shmem.channel.workFifoDev;
  int workFifoIx = shmem.channel.index;

  bool skipLoadWork = false, firstLaunch = true;
  if (bid == 0 && first.active != 0)
    skipLoadWork = true;

  while (true) {
    if (!skipLoadWork) {
      copyToShmem(&shmem.work, &workFifoDev[workFifoIx]); // turn no longer helps
      // Check whether the last operation was aborted and make sure all threads exit
      int aborted = tid == 0 ? *shmem.comm.abortFlag : 0;
      if (barrierReduceAny(aborted, &abortCount)) { // publish shmem.work
        if (COLLTRACE && tid == 0) traceAbort(elems->funcIndex);
        break;
      }
      if (tid == 0)
        workFifoHost[workFifoIx].elems[0].active = 0;
      if (COLLTRACE && tid == 0) {
        if (firstLaunch) traceKernelLaunch(elems->funcIndex);
        if (!firstLaunch) traceCollEnd(elems->funcIndex);
        firstLaunch = false;
      }
    } else if (COLLTRACE && tid == 0) {
        traceKernelLaunch(elems->funcIndex);
        firstLaunch = false;
    }

    workFifoIx = (workFifoIx + 1)%NCCL_MAX_OPS;
    if (tid == 0)
      channel->index = workFifoIx; // write back to real channel, not shmem shadow

    if (tid < NCCL_MAX_WORK_ELEMENTS && tid % ncclWorkElemFactors[Algo] == 0) {
      ncclWorkElem *we = &shmem.work.elems[tid];
      if (we->redOpArgIsPtr && we->active != 0) {
        /* redOpArg is a pointer to the scalar value, so we'll dereference it
         * here so that redOpArg holds the bits of the scalar going forward.
         * The tricky thing is we don't know its type T since that's encoded in
         * the funcIndex. Because it would be difficult to get sizeof(T) from
         * funcIndex, we'll cheat and just dereference the largest possible size
         * given the alignment of the pointer. We might be reading in more bytes
         * than we need but that's harmless.
         */
        if (we->coll.redOpArg%2 != 0)
          we->coll.redOpArg = *reinterpret_cast<uint8_t*>(we->coll.redOpArg);
        else if (we->coll.redOpArg%4 != 0)
          we->coll.redOpArg = *reinterpret_cast<uint16_t*>(we->coll.redOpArg);
        else if (we->coll.redOpArg%8 != 0)
          we->coll.redOpArg = *reinterpret_cast<uint32_t*>(we->coll.redOpArg);
        else
          we->coll.redOpArg = *reinterpret_cast<uint64_t*>(we->coll.redOpArg);
      }
    }
    __syncthreads();

    if (shmem.work.elems[0].funcIndex == FnIndex)
      RunWork<Fn, T, RedOp, Algo, Proto>().run(&shmem.work);
    else
      NCCL_CALL_FUNCTIONS(&elems[0]);

    if (shmem.work.elems[0].active == 2) {
      if (COLLTRACE && tid == 0) traceKernelEnd(elems->funcIndex)
      break;
    }
    __syncthreads();
    skipLoadWork = false;
  }
}

#define IMPL_COLL_KERN(func, algo, proto, devredop, type, fIndex) \
__launch_bounds__(NCCL_MAX_NTHREADS, 1) \
__global__ void NCCL_KERN_NAME(func, algo, proto, devredop, type)(ncclWorkElem first) { \
  if (first.comm->collTraceThread) \
    ncclKernel<ncclFunc##func, type, Func##devredop<type>, NCCL_ALGO_##algo, NCCL_PROTO_##proto, fIndex, true>(first); \
  else \
    ncclKernel<ncclFunc##func, type, Func##devredop<type>, NCCL_ALGO_##algo, NCCL_PROTO_##proto, fIndex, false>(first); \
}

// Examples :     AllReduce, RING, LL,    Sum,   uint8
/* Functions for aggregation case */
#define IMPL_COLL_FUNC(func, algo, proto, devredop, type) \
__device__  __attribute__((noinline)) void NCCL_FUNC_NAME(func, algo, proto, devredop, type)(struct ncclWorkElem* args) { \
  RunWork<ncclFunc##func, type, Func##devredop<type>, NCCL_ALGO_##algo, NCCL_PROTO_##proto>().run(&ncclShmem->work); \
}

// Only generate inline kernels for LL
#define IMPL_COLL4(func, algo, devredop, type, ncclType) \
  IMPL_COLL_FUNC(func, algo, LL,     devredop, type) \
  IMPL_COLL_FUNC(func, algo, SIMPLE, devredop, type) \

#define IMPL_COLL3(func, devredop, type, ncclType) \
  IMPL_COLL4(func, TREE,    devredop, type, ncclType) \
  IMPL_COLL4(func, RING,    devredop, type, ncclType) \
  IMPL_COLL4(func, COLLNET, devredop, type, ncclType)

#define IMPL_COLL2(func, devredop) \
  IMPL_COLL3(func, devredop, int8_t,   ncclInt8) \
  IMPL_COLL3(func, devredop, uint8_t,  ncclUint8) \
  IMPL_COLL3(func, devredop, int32_t,  ncclInt32) \
  IMPL_COLL3(func, devredop, uint32_t, ncclUint32) \
  IMPL_COLL3(func, devredop, int64_t,  ncclInt64) \
  IMPL_COLL3(func, devredop, uint64_t, ncclUint64) \
  IMPL_COLL3(func, devredop, half,     ncclFloat16) \
  IMPL_COLL3(func, devredop, float,    ncclFloat32) \
  IMPL_COLL3(func, devredop, double,   ncclFloat64) \
  IMPL_COLL3(func, devredop, rccl_bfloat16, ncclBfloat16)

#define IMPL_COLL2A(func, devredop) \
  IMPL_COLL3(func, devredop, int8_t,   ncclInt8) \
  IMPL_COLL3(func, devredop, uint8_t,  ncclUint8) \
  IMPL_COLL3(func, devredop, int32_t,  ncclInt32) \
  IMPL_COLL3(func, devredop, uint32_t, ncclUint32) \
  IMPL_COLL3(func, devredop, int64_t,  ncclInt64) \
  IMPL_COLL3(func, devredop, uint64_t, ncclUint64)

// Reduction define all functions
#define IMPL_COLL_R(func) \
  IMPL_COLL2(func, Sum) \
  IMPL_COLL2(func, Prod) \
  IMPL_COLL2(func, Min) \
  IMPL_COLL2(func, Max) \
  IMPL_COLL2(func, PreMulSum) \
  IMPL_COLL2A(func, SumPostDiv)

// [RCCL] Define clique-based implementations (repurposed LL128)
#define IMPL_COLL4_CLIQUE(func, algo, devredop, type, ncclType) \
  IMPL_COLL_FUNC(func, algo, LL,     devredop, type) \
  IMPL_COLL_FUNC(func, algo, LL128,  devredop, type) \
  IMPL_COLL_FUNC(func, algo, SIMPLE, devredop, type) \

#define IMPL_COLL3_CLIQUE(func, devredop, type, ncclType) \
  IMPL_COLL4_CLIQUE(func, TREE,    devredop, type, ncclType) \
  IMPL_COLL4_CLIQUE(func, RING,    devredop, type, ncclType) \
  IMPL_COLL4_CLIQUE(func, COLLNET, devredop, type, ncclType)

#define IMPL_COLL2_CLIQUE(func, devredop) \
  IMPL_COLL3_CLIQUE(func, devredop, int8_t,   ncclInt8) \
  IMPL_COLL3_CLIQUE(func, devredop, uint8_t,  ncclUint8) \
  IMPL_COLL3_CLIQUE(func, devredop, int32_t,  ncclInt32) \
  IMPL_COLL3_CLIQUE(func, devredop, uint32_t, ncclUint32) \
  IMPL_COLL3_CLIQUE(func, devredop, int64_t,  ncclInt64) \
  IMPL_COLL3_CLIQUE(func, devredop, uint64_t, ncclUint64) \
  IMPL_COLL3_CLIQUE(func, devredop, half,     ncclFloat16) \
  IMPL_COLL3_CLIQUE(func, devredop, float,    ncclFloat32) \
  IMPL_COLL3_CLIQUE(func, devredop, double,   ncclFloat64) \
  IMPL_COLL3_CLIQUE(func, devredop, rccl_bfloat16, ncclBfloat16)

#define IMPL_COLL2A_CLIQUE(func, devredop) \
  IMPL_COLL3_CLIQUE(func, devredop, int8_t,   ncclInt8) \
  IMPL_COLL3_CLIQUE(func, devredop, uint8_t,  ncclUint8) \
  IMPL_COLL3_CLIQUE(func, devredop, int32_t,  ncclInt32) \
  IMPL_COLL3_CLIQUE(func, devredop, uint32_t, ncclUint32) \
  IMPL_COLL3_CLIQUE(func, devredop, int64_t,  ncclInt64) \
  IMPL_COLL3_CLIQUE(func, devredop, uint64_t, ncclUint64)

#define IMPL_COLL_CLIQUE(func) \
  IMPL_COLL2_CLIQUE(func, Sum) \
  IMPL_COLL2_CLIQUE(func, Prod) \
  IMPL_COLL2_CLIQUE(func, Min) \
  IMPL_COLL2_CLIQUE(func, Max) \
  IMPL_COLL2_CLIQUE(func, PreMulSum) \
  IMPL_COLL2A_CLIQUE(func, SumPostDiv)
// [/RCCL]

// Copy primitives only define one function for copy
#define IMPL_COLL_C(func) IMPL_COLL3(func, Sum, int8_t, ncclInt8);

// Point-to-point primitives only have one function/kernel.
#define IMPL_COLL_P(func) \
  IMPL_COLL_FUNC(func, RING, SIMPLE, Sum, int8_t); \
  IMPL_COLL_KERN(func, RING, SIMPLE, Sum, int8_t, FUNC_INDEX_P2P);

#endif
