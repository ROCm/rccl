/*************************************************************************
 * Copyright (c) 2017-2020, NVIDIA CORPORATION. All rights reserved.
 * Modifications Copyright (c) 2019-2020 Advanced Micro Devices, Inc. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

#ifndef NCCL_DEVICE_COMMON_H_
#define NCCL_DEVICE_COMMON_H_

#include "collectives.h"
#include "devcomm.h"

#define COLL_UNROLL 2
#define NCCL_MAX_DEV_ARITY NCCL_MAX_TREE_ARITY

// Exit If Abort Barrier across CTA: make sure all threads exit consistently
// Each thread sets a predicate to true if abort == 1
// all CTA's threads enter the barrier and do a popc on their predicates being True
// If any of the thread's predicate was True, all the threads call exit()
#define exitIfAbortBarrier(abort, abortCount) \
  if (abort) __atomic_fetch_add(abortCount, 1, __ATOMIC_SEQ_CST); \
  __syncthreads(); \
  if (LOAD(abortCount)) { /*asm volatile ("s_endpgm");*/ return false; }
#define __syncwarp()

#define NCCL_FUNC5(func, algo, redop, type) \
  NCCL_FUNC_NAME(func, algo, LL,     redop, type), \
  NCCL_FUNC_NAME(func, algo, LL,     redop, type), \
  NCCL_FUNC_NAME(func, algo, SIMPLE, redop, type)

#define NCCL_FUNC4(func, redop, type) \
  NCCL_FUNC5(func, TREE,    redop, type), \
  NCCL_FUNC5(func, RING,    redop, type), \
  NCCL_FUNC5(func, COLLNET, redop, type)

// Must be consistent with ncclDataType_t
#define NCCL_FUNCS3A(func, redop) \
  NCCL_FUNC4(func, redop, int8_t), \
  NCCL_FUNC4(func, redop, uint8_t), \
  NCCL_FUNC4(func, redop, int32_t), \
  NCCL_FUNC4(func, redop, uint32_t), \
  NCCL_FUNC4(func, redop, int64_t), \
  NCCL_FUNC4(func, redop, uint64_t), \
  NCCL_FUNC4(func, redop, half), \
  NCCL_FUNC4(func, redop, float), \
  NCCL_FUNC4(func, redop, double), \
  NCCL_FUNC4(func, redop, rccl_bfloat16)
#define NCCL_FUNCS3B(func, redop) \
  NCCL_FUNC4(func, redop, int8_t), \
  NCCL_FUNC4(func, redop, int8_t), \
  NCCL_FUNC4(func, redop, int8_t), \
  NCCL_FUNC4(func, redop, int8_t), \
  NCCL_FUNC4(func, redop, int8_t), \
  NCCL_FUNC4(func, redop, int8_t), \
  NCCL_FUNC4(func, redop, int8_t), \
  NCCL_FUNC4(func, redop, int8_t), \
  NCCL_FUNC4(func, redop, int8_t), \
  NCCL_FUNC4(func, redop, int8_t)

// Must be consistent with ncclRedOp_t
#define NCCL_FUNCS2A(func) \
  NCCL_FUNCS3A(func, Sum ), \
  NCCL_FUNCS3A(func, Prod), \
  NCCL_FUNCS3A(func, Max ), \
  NCCL_FUNCS3A(func, Min )
#define NCCL_FUNCS2B(func) \
  NCCL_FUNCS3B(func, Sum), \
  NCCL_FUNCS3B(func, Sum), \
  NCCL_FUNCS3B(func, Sum), \
  NCCL_FUNCS3B(func, Sum)

// Must be consistent with ncclFunc_t
#define NCCL_FUNCS() { \
  NCCL_FUNCS2B(Broadcast), \
  NCCL_FUNCS2A(Reduce), \
  NCCL_FUNCS2B(AllGather), \
  NCCL_FUNCS2A(ReduceScatter), \
  NCCL_FUNCS2A(AllReduce), \
  NCCL_FUNC_NAME(SendRecv, RING, SIMPLE, Sum, int8_t), \
  NCCL_FUNC_NAME(AllToAll, RING, SIMPLE, Sum, int8_t), \
  NCCL_FUNC_NAME(AllToAllv, RING, SIMPLE, Sum, int8_t) }

// Must be consistent with the ncclFuncSet enum
using ncclKernelFunc_t = void (*)(struct ncclWorkElem* args);

static const __device__ constexpr ncclKernelFunc_t ncclFuncs[]{
// Don't try to initialize the host shadow copy of this device-side global
// variable. There is no host pointer to a device-side function, which
// confuses clang. This will be fixed in the next clang release.
#if defined(__HIP_DEVICE_COMPILE__)
  NCCL_FUNCS2B(Broadcast),
  NCCL_FUNCS2A(Reduce),
  NCCL_FUNCS2B(AllGather),
  NCCL_FUNCS2A(ReduceScatter),
  NCCL_FUNCS2A(AllReduce),
  NCCL_FUNC_NAME(SendRecv, RING, SIMPLE, Sum, int8_t),
  NCCL_FUNC_NAME(AllToAll, RING, SIMPLE, Sum, int8_t),
  NCCL_FUNC_NAME(AllToAllv, RING, SIMPLE, Sum, int8_t),
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

inline
__device__
void NCCL_CALL_FUNCTIONS(struct ncclWorkElem* const c) noexcept {
  if (c->funcIndex < 360) {
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
  else if (c->funcIndex < 720) Caller<360, 720>::call(c);
  else if (c->funcIndex < 1080) {
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
  else if (c->funcIndex < 1800) Caller<1080, 1800>::call(c);
  else if (c->funcIndex == 1800) ncclFunction_SendRecv_RING_SIMPLE_Sum_int8_t(c);
  else if (c->funcIndex == 1801) ncclFunction_AllToAll_RING_SIMPLE_Sum_int8_t(c);
  else if (c->funcIndex == 1802) ncclFunction_AllToAllv_RING_SIMPLE_Sum_int8_t(c);
}

static __device__ void load_parallel(void* dst, void* src, size_t size, int tid) {
  int* d = (int*)dst;
  int* s = (int*)src;
  for (int o = tid; o < (size/sizeof(int)); o += blockDim.x) d[o] = s[o];
}

static __device__ bool load_coll(struct ncclWork* localWork, struct ncclWork* hostWork, int tid, struct ncclDevComm* comm, uint32_t* abortCount) {
  __syncthreads();
  load_parallel(localWork, hostWork, sizeof(struct ncclWork), tid);
  // Check whether the last operation was aborted and make sure all threads exit
  int abort = tid == 0 ? LOAD(comm->abortFlag) : 0;
  exitIfAbortBarrier(abort, abortCount);
  if (tid == 0) hostWork->elems[0].active = 0;
  return true;
}

template <ncclFunc_t FUNCTION, int ALGO, int PROTO, class REDOP, typename T, int UNROLL>
class ncclFunction {
  public:
  __device__ void run(struct ncclWorkElem* args) {}
};

#ifdef ENABLE_COLLTRACE
#define traceColl(fIdx)  \
    uint32_t pos = __atomic_fetch_add(comm->collTraceTail, 1, __ATOMIC_SEQ_CST)%COLLTRACE_NUM_ITEMS; \
    comm->collTrace[pos].timeStamp = __builtin_amdgcn_s_memrealtime(); \
    comm->collTrace[pos].opCount = w->opCount; \
    comm->collTrace[pos].bid = bid; \
    comm->collTrace[pos].funcIndex = fIdx;
#define traceKernelLaunch(fIdx)  { \
    traceColl(fIdx); \
    asm volatile ("s_getreg_b32 %0, hwreg(HW_REG_HW_ID)" : "=s" (comm->collTrace[pos].data_0)); \
    comm->collTrace[pos].type = ncclCollTraceKernelLaunchType; \
  }
#define traceCollEnd(fIdx)  { \
    traceColl(fIdx); \
    comm->collTrace[pos].type = ncclCollTraceCollEndType; \
  }
#define traceAbort(fIdx)  { \
    traceColl(fIdx); \
    comm->collTrace[pos].type = ncclCollTraceAbortType; \
  }
//  traceData(int16_t data2, uint32_t data4, uint64_t data8_0, uint64_t data8_1)
#define traceData(data2, data4, data8_0, data8_1) { \
    uint32_t pos = __atomic_fetch_add(comm->collTraceTail, 1, __ATOMIC_SEQ_CST)%COLLTRACE_NUM_ITEMS; \
    comm->collTrace[pos].bid = blockIdx.x; \
    comm->collTrace[pos].timeStamp = __builtin_amdgcn_s_memrealtime(); \
    comm->collTrace[pos].funcIndex = data2; \
    comm->collTrace[pos].data_0 = data4; \
    comm->collTrace[pos].opCount = data8_0; \
    comm->collTrace[pos].data_1 = data8_1; \
    comm->collTrace[pos].type = ncclCollTraceDataType; \
  }
#else
#define traceKernelLaunch(fIdx)
#define traceCollEnd(fIdx)
#define traceAbort(fIdx)
#define traceData(data2, data4, data8_0, data8_1)
#endif

#define MAXWARPS (NCCL_MAX_NTHREADS/WARP_SIZE)

struct ncclShmemPtrs {
  void* srcs[NCCL_MAX_DEV_ARITY+1];
  void* dsts[NCCL_MAX_DEV_ARITY+1];
  uint64_t barrier;
  uint64_t barrier_next[MAXWARPS];
};

struct ncclShmemData {
  union {
#ifdef ENABLE_LL128
    volatile uint64_t data[NCCL_LL128_SHMEM_SIZE];
#else
    volatile uint64_t* data;
#endif
    struct ncclShmemPtrs ptrs[NCCL_MAX_GROUPS];
  };
  uint32_t sync[MAXWARPS];
  struct ncclWork localWork;
};

extern __device__ struct ncclShmemData *ncclShmem;
template <ncclFunc_t FUNCTION, int ALGO, int PROTO, class REDOP, typename T, int UNROLL, int FINDEX, bool COLLTRACE>
__device__ void ncclKernel(struct ncclWorkElem first)  {
  int tid = threadIdx.x;
  int bid = blockIdx.x;
  __shared__ struct ncclShmemData shmem;
  ncclShmem = &shmem;
  __shared__ uint32_t abortCount;
  if (tid == 0) {
    abortCount = 0;
    for (auto i = 0; i < NCCL_MAX_GROUPS; i++) {
      shmem.ptrs[i].barrier = 0;
      for (auto j = 0; j < MAXWARPS; j++) shmem.ptrs[i].barrier_next[j] = 0;
    }
  }
  __syncthreads();

  auto f = ncclFunction<FUNCTION, ALGO, PROTO, REDOP, T, UNROLL>();

  struct ncclDevComm* comm = first.comm;
  struct ncclChannel* channel = comm->channels+bid;
  struct ncclWorkElem* w = NULL;
  uint16_t index = first.index;
  bool firstLaunch = true;

  if (bid == 0 && first.funcIndex != FUNC_INDEX_P2P) w = &first;

  while (1) {
    if (w == NULL) {
      w = shmem.localWork.elems;
      if (!load_coll(&shmem.localWork, channel->workFifo+index, tid, comm, &abortCount)) {
        if (COLLTRACE && tid == 0) traceAbort(-1);
        return;
      }
      if (COLLTRACE && tid == 0) {
        if (firstLaunch) traceKernelLaunch(w->funcIndex);
        if (!firstLaunch) traceCollEnd(w->funcIndex);
        firstLaunch = false;
      }
    } else {
      if (COLLTRACE && tid == 0) {
        traceKernelLaunch(w->funcIndex);
        firstLaunch = false;
      }
    }
    if (tid < w->nThreads) {
      if (w->funcIndex == FINDEX) {
        f.run(w);
      } else {
        NCCL_CALL_FUNCTIONS(w);
      }
    }
    index = (index+1) % NCCL_MAX_OPS;
    if (w->active == 2) {
      if (COLLTRACE && tid == 0) traceCollEnd(-1);
      return;
    }
    w = NULL;
  }
}

#define IMPL_COLL_KERN(func, algo, proto, redop, type, fIndex) \
__launch_bounds__(NCCL_MAX_NTHREADS, 1) \
__global__ void NCCL_KERN_NAME(func, algo, proto, redop, type)(struct ncclWorkElem first) { \
  if (first.comm->collTraceThread) \
    ncclKernel<ncclFunc##func, NCCL_ALGO_##algo, NCCL_PROTO_##proto, Func##redop<type>, type, COLL_UNROLL, fIndex, true>(first); \
  else \
    ncclKernel<ncclFunc##func, NCCL_ALGO_##algo, NCCL_PROTO_##proto, Func##redop<type>, type, COLL_UNROLL, fIndex, false>(first); \
}

// Examples :     AllReduce, RING, LL,    Sum,   uint8
/* Functions for aggregation case */
#define IMPL_COLL_FUNC(func, algo, proto, redop, type) \
__device__  __attribute__((noinline)) void NCCL_FUNC_NAME(func, algo, proto, redop, type)(struct ncclWorkElem* args) { \
  auto f = ncclFunction<ncclFunc##func, NCCL_ALGO_##algo, NCCL_PROTO_##proto, Func##redop<type>, type, COLL_UNROLL>(); \
  f.run(args); \
}

// Only generate inline kernels for LL
#define IMPL_COLL4(func, algo, redop, type, ncclType) \
  IMPL_COLL_FUNC(func, algo, LL,     redop, type) \
  IMPL_COLL_FUNC(func, algo, SIMPLE, redop, type) \

#define IMPL_COLL3(func, redop, type, ncclType) \
  IMPL_COLL4(func, TREE,    redop, type, ncclType) \
  IMPL_COLL4(func, RING,    redop, type, ncclType) \
  IMPL_COLL4(func, COLLNET, redop, type, ncclType)

#define IMPL_COLL2(func, redop) \
  IMPL_COLL3(func, redop, int8_t,   ncclInt8) \
  IMPL_COLL3(func, redop, uint8_t,  ncclUint8) \
  IMPL_COLL3(func, redop, int32_t,  ncclInt32) \
  IMPL_COLL3(func, redop, uint32_t, ncclUint32) \
  IMPL_COLL3(func, redop, int64_t,  ncclInt64) \
  IMPL_COLL3(func, redop, uint64_t, ncclUint64) \
  IMPL_COLL3(func, redop, half,     ncclFloat16) \
  IMPL_COLL3(func, redop, float,    ncclFloat32) \
  IMPL_COLL3(func, redop, double,   ncclFloat64) \
  IMPL_COLL3(func, redop, rccl_bfloat16,   ncclBfloat16)

// Reduction define all functions
#define IMPL_COLL_R(func) \
  IMPL_COLL2(func, Sum) \
  IMPL_COLL2(func, Prod) \
  IMPL_COLL2(func, Min) \
  IMPL_COLL2(func, Max)

// Copy primitives only define one function for copy
#define IMPL_COLL_C(func) IMPL_COLL3(func, Sum, int8_t, ncclInt8);

// Point-to-point primitives only have one function/kernel.
#define IMPL_COLL_P(func) \
  IMPL_COLL_FUNC(func, RING, SIMPLE, Sum, int8_t); \
  IMPL_COLL_KERN(func, RING, SIMPLE, Sum, int8_t, FUNC_INDEX_P2P);

#endif
