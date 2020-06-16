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

__device__
inline  __attribute((always_inline))
long long int __rtc64() {
#if __HIP__
  return (long long int) __builtin_amdgcn_s_memrealtime();
#else
  return (long long int) __clock_u64();
#endif
}

// Exit If Abort Barrier across CTA: make sure all threads exit consistently
// Each thread sets a predicate to true if abort == 1
// all CTA's threads enter the barrier and do a popc on their predicates being True
// If any of the thread's predicate was True, all the threads call exit()
#if defined(__HIP_PLATFORM_HCC__) || defined(__HCC__) || defined(__HIPCC__)
#define exitIfAbortBarrier(abort, abortCount) \
  if (abort) __atomic_fetch_add(abortCount, 1, __ATOMIC_SEQ_CST); \
  __syncthreads(); \
  if (LOAD(abortCount)) { /*asm volatile ("s_endpgm");*/ return false; }
#define __syncwarp()
#else
static inline __device__ void exitIfAbortBarrier(int abort) {
  uint32_t popc;
  asm ("{");
  asm volatile ("   .reg .pred barr_pred;");
  asm volatile ("   setp.eq.u32 barr_pred,%0,1;" :: "r"(abort));
  asm volatile ("   bar.red.popc.u32 %0, 13, barr_pred;" : "=r"(popc));
  asm ("}");
  if (popc) { asm volatile ("exit;"); }
}
#endif

#define NCCL_FUNC5(coll, op, dtype) \
  NCCL_COLL_NAME(coll##LL, op, dtype), \
  NCCL_COLL_NAME(coll##LL, op, dtype), \
  NCCL_COLL_NAME(coll, op, dtype)

#define NCCL_FUNC4(coll, op, dtype) \
  NCCL_FUNC5(coll##Tree, op, dtype), \
  NCCL_FUNC5(coll##Ring, op, dtype), \
  NCCL_FUNC5(coll##CollNet, op, dtype)

// Must be consistent with ncclDataType_t
#define NCCL_FUNCS3A(coll, op) \
  NCCL_FUNC4(coll, op,  i8), \
  NCCL_FUNC4(coll, op,  u8), \
  NCCL_FUNC4(coll, op, i32), \
  NCCL_FUNC4(coll, op, u32), \
  NCCL_FUNC4(coll, op, i64), \
  NCCL_FUNC4(coll, op, u64), \
  NCCL_FUNC4(coll, op, f16), \
  NCCL_FUNC4(coll, op, f32), \
  NCCL_FUNC4(coll, op, f64), \
  NCCL_FUNC4(coll, op, b16)
#define NCCL_FUNCS3B(coll, op) \
  NCCL_FUNC4(coll, op,  i8), \
  NCCL_FUNC4(coll, op,  i8), \
  NCCL_FUNC4(coll, op,  i8), \
  NCCL_FUNC4(coll, op,  i8), \
  NCCL_FUNC4(coll, op,  i8), \
  NCCL_FUNC4(coll, op,  i8), \
  NCCL_FUNC4(coll, op,  i8), \
  NCCL_FUNC4(coll, op,  i8), \
  NCCL_FUNC4(coll, op,  i8), \
  NCCL_FUNC4(coll, op,  i8)

// Must be consistent with ncclRedOp_t
#define NCCL_FUNCS2A(coll) \
  NCCL_FUNCS3A(coll, sum ), \
  NCCL_FUNCS3A(coll, prod), \
  NCCL_FUNCS3A(coll, max ), \
  NCCL_FUNCS3A(coll, min )
#define NCCL_FUNCS2B(coll) \
  NCCL_FUNCS3B(coll, copy), \
  NCCL_FUNCS3B(coll, copy), \
  NCCL_FUNCS3B(coll, copy), \
  NCCL_FUNCS3B(coll, copy)

// Must be consistent with ncclFunc_t
#define NCCL_FUNCS() { \
  NCCL_FUNCS2B(ncclBroadcast), \
  NCCL_FUNCS2A(ncclReduce), \
  NCCL_FUNCS2B(ncclAllGather), \
  NCCL_FUNCS2A(ncclReduceScatter), \
  NCCL_FUNCS2A(ncclAllReduce), \
  NCCL_COLL_NAME(ncclGather, copy, i8), \
  NCCL_COLL_NAME(ncclScatter, copy, i8), \
  NCCL_COLL_NAME(ncclAllToAll, copy, i8), \
  NCCL_COLL_NAME(ncclSendRecv, copy, i8) }

// Must be consistent with the ncclFuncSet enum
using ncclKernelFunc_t = void (*)(struct CollectiveArgs*);

static const __device__ constexpr ncclKernelFunc_t ncclFuncs[]{
// Don't try to initialize the host shadow copy of this device-side global
// variable. There is no host pointer to a device-side function, which
// confuses clang. This will be fixed in the next clang release.
#if defined(__HIP_DEVICE_COMPILE__)
  NCCL_FUNCS2B(ncclBroadcast),
  NCCL_FUNCS2A(ncclReduce),
  NCCL_FUNCS2B(ncclAllGather),
  NCCL_FUNCS2A(ncclReduceScatter),
  NCCL_FUNCS2A(ncclAllReduce),
  NCCL_COLL_NAME(ncclGather, copy, i8),
  NCCL_COLL_NAME(ncclScatter, copy, i8),
  NCCL_COLL_NAME(ncclAllToAll, copy, i8),
  NCCL_COLL_NAME(ncclSendRecv, copy, i8)
#endif
};

template<unsigned short f, unsigned short l>
struct Caller {
  static __device__ __host__
  void call(ncclColl* const c) noexcept
  {
    constexpr unsigned short m = f + (l - f) / 2;

     return (c->funcIndex < m) ? Caller<f, m>::call(c) : Caller<m, l>::call(c);
  }
};

template<unsigned short f>
struct Caller<f, f + 1>{
  static __device__ __host__
  void call(struct ncclColl* const c) noexcept { ncclFuncs[f](&c->args); }
};

inline
__device__
void NCCL_CALL_FUNCTIONS(struct ncclColl* const c) noexcept {
  if (c->funcIndex < 360) {
    if (c->funcIndex % 9 == 0) ncclBroadcastTreeLL_copy_i8(&c->args);
    else if (c->funcIndex % 9 == 1) ncclBroadcastTreeLL128_copy_i8(&c->args);
    else if (c->funcIndex % 9 == 2) ncclBroadcastTree_copy_i8(&c->args);
    else if (c->funcIndex % 9 == 3) ncclBroadcastRingLL_copy_i8(&c->args);
    else if (c->funcIndex % 9 == 4) ncclBroadcastRingLL128_copy_i8(&c->args);
    else if (c->funcIndex % 9 == 5) ncclBroadcastRing_copy_i8(&c->args);
    else if (c->funcIndex % 9 == 6) ncclBroadcastCollNetLL_copy_i8(&c->args);
    else if (c->funcIndex % 9 == 7) ncclBroadcastCollNetLL128_copy_i8(&c->args);
    else ncclBroadcastCollNet_copy_i8(&c->args);
  }
  else if (c->funcIndex < 720) Caller<360, 720>::call(c);
  else if (c->funcIndex < 1080) {
    if (c->funcIndex % 9 == 0) ncclAllGatherTreeLL_copy_i8(&c->args);
    else if (c->funcIndex % 9 == 1) ncclAllGatherTreeLL128_copy_i8(&c->args);
    else if (c->funcIndex % 9 == 2) ncclAllGatherTree_copy_i8(&c->args);
    else if (c->funcIndex % 9 == 3) ncclAllGatherRingLL_copy_i8(&c->args);
    else if (c->funcIndex % 9 == 4) ncclAllGatherRingLL128_copy_i8(&c->args);
    else if (c->funcIndex % 9 == 5) ncclAllGatherRing_copy_i8(&c->args);
    else if (c->funcIndex % 9 == 6) ncclAllGatherCollNetLL_copy_i8(&c->args);
    else if (c->funcIndex % 9 == 7) ncclAllGatherCollNetLL128_copy_i8(&c->args);
    else ncclAllGatherCollNet_copy_i8(&c->args);
  }
  else if (c->funcIndex < 1800) Caller<1080, 1800>::call(c);
  else if (c->funcIndex == 1800) {
    ncclGather_copy_i8(&c->args);
  }
  else if (c->funcIndex == 1801) {
    ncclScatter_copy_i8(&c->args);
  }
  else if (c->funcIndex == 1802) {
    ncclAllToAll_copy_i8(&c->args);
  }
  else ncclSendRecv_copy_i8(&c->args);
}

static __device__ void load_parallel(void* dst, void* src, size_t size, int tid, uint32_t* abortCount) {
  int* d = (int*)dst;
  int* s = (int*)src;
  for (int o = tid; o < (size/sizeof(int)); o += blockDim.x) d[o] = s[o];
}

static __device__ bool load_coll(struct ncclColl* localColl, struct ncclColl* hostColl, int tid, struct ncclDevComm* comm, uint32_t* abortCount) {
  // Check whether the last operation was aborted and make sure all threads exit
  int abort = tid == 0 ? *(comm->abortFlag) : 0;
  exitIfAbortBarrier(abort, abortCount);
  load_parallel(localColl, hostColl, sizeof(struct ncclColl), tid, abortCount);
  __syncthreads();
  if (tid == 0) hostColl->active = 0;
  return true;
}

#ifdef ENABLE_COLLTRACE
#define traceColl(fIdx)  \
    uint32_t pos = __atomic_fetch_add(comm->collTraceTail, 1, __ATOMIC_SEQ_CST)%COLLTRACE_NUM_ITEMS; \
    comm->collTrace[pos].timeStamp = __rtc64(); \
    comm->collTrace[pos].opCount = localColl.args.opCount; \
    comm->collTrace[pos].bid = bid; \
    comm->collTrace[pos].funcIndex = fIdx;
#define traceKernelLaunch(fIdx)  { \
    traceColl(fIdx); \
    comm->collTrace[pos].type = ncclCollTraceKernelLaunchType; \
    asm volatile ("s_getreg_b32 %0, hwreg(HW_REG_HW_ID)" : "=s" (comm->collTrace[pos].data_0)); \
  }
#define traceCollEnd(fIdx)  { \
    traceColl(fIdx); \
    comm->collTrace[pos].type = ncclCollTraceCollEndType; \
  }
#define traceAbort(fIdx)  { \
    traceColl(fIdx); \
    comm->collTrace[pos].type = ncclCollTraceAbortType; \
  }
#else
#define traceKernelLaunch()
#define traceCollEnd()
#define traceAbort()
#endif

extern __device__ volatile uint64_t* ncclShmem;

#ifdef ENABLE_LL128
#define ALLOCATE_SHMEM \
  __shared__ volatile uint64_t shmem[NCCL_LL128_SHMEM_SIZE]; \
  ncclShmem = shmem; \
  __shared__ uint32_t sync[NCCL_LL128_MAX_NTHREADS/WARP_SIZE];
#else
#define ALLOCATE_SHMEM \
  uint32_t* sync = 0;
#endif

/* Functions for aggregation case */
#define IMPL_COLL_FUNC(coll, op, ncclFunc, dtype, ctype) \
__device__ void NCCL_COLL_NAME(coll, op, dtype)(struct CollectiveArgs* args) { \
  coll##Kernel<COLL_UNROLL, ncclFunc<ctype>, ctype>(args); \
}

/* Kernels with the first operation inlined */
#define IMPL_COLL_KERN(coll, op, ncclFunc, dtype, ctype, fIndex) \
__launch_bounds__(NCCL_MAX_NTHREADS, 1) \
__global__ void NCCL_KERN_NAME(coll, op, dtype)(struct ncclDevComm* comm) { \
  int tid = threadIdx.x; \
  int bid = blockIdx.x; \
  ALLOCATE_SHMEM; \
  __shared__ struct ncclColl localColl; \
  __shared__ uint32_t abortCount; \
  if (tid == 0) abortCount = 0; \
  __syncthreads(); \
 \
  struct ncclChannel* channel = comm->channels+bid; \
  channel->sync = sync; \
  if (!load_coll(&localColl, channel->collectives+channel->collFifoHead, tid, comm, &abortCount)) { \
    if (tid == 0) traceAbort(-1); \
    return; \
  } \
  if (tid == 0) traceKernelLaunch(localColl.funcIndex); \
  while (1) { \
    if (tid < localColl.args.common.nThreads) { \
      if (localColl.funcIndex == fIndex) { \
        coll##Kernel<COLL_UNROLL, ncclFunc<ctype>, ctype>(&localColl.args); \
      } else { \
        NCCL_CALL_FUNCTIONS(&localColl); \
      } \
    } \
    int nextIndex = localColl.nextIndex; \
    if (tid == 0) channel->collFifoHead = nextIndex; \
 \
    if (localColl.active == 2) { \
      if (tid == 0) traceCollEnd(-1); \
      return; \
    } \
 \
    /* Load next collective operation*/ \
    if (!load_coll(&localColl, channel->collectives+nextIndex, tid, comm, &abortCount)) { \
      if (tid == 0) traceAbort(-1); \
      break; \
    } \
    if (tid == 0) traceCollEnd(localColl.funcIndex); \
  } \
}

#define IMPL_COLL_KERN_sum(coll, op, ncclFunc, dtype, ctype, fIndex) \
  IMPL_COLL_KERN(coll, op, ncclFunc, dtype, ctype, fIndex)
#define IMPL_COLL_KERN_copy(coll, op, ncclFunc, dtype, ctype, fIndex) \
  IMPL_COLL_KERN(coll, op, ncclFunc, dtype, ctype, fIndex)
#define IMPL_COLL_KERN_prod(coll, op, ncclFunc, dtype, ctype, fIndex)
#define IMPL_COLL_KERN_min(coll, op, ncclFunc, dtype, ctype, fIndex)
#define IMPL_COLL_KERN_max(coll, op, ncclFunc, dtype, ctype, fIndex)

// Only generate inline kernels for LL
#define IMPL_COLL4(coll, op, ncclFunc, dtype, ctype, ncclColl, ncclOp, ncclType, al) \
  IMPL_COLL_FUNC(coll##LL, op, ncclFunc, dtype, ctype) \
  IMPL_COLL_FUNC(coll##LL128, op, ncclFunc, dtype, ctype) \
  IMPL_COLL_FUNC(coll, op, ncclFunc, dtype, ctype) \

#define IMPL_COLL3(coll, op, ncclFunc, dtype, ctype, ncclColl, ncclOp, ncclType) \
  IMPL_COLL4(coll##Tree, op, ncclFunc, dtype, ctype, ncclColl, ncclOp, ncclType, NCCL_ALGO_TREE) \
  IMPL_COLL4(coll##Ring, op, ncclFunc, dtype, ctype, ncclColl, ncclOp, ncclType, NCCL_ALGO_RING) \
  IMPL_COLL4(coll##CollNet, op, ncclFunc, dtype, ctype, ncclColl, ncclOp, ncclType, NCCL_ALGO_COLLNET)

#define IMPL_COLL2(coll, op, ncclFunc, ncclColl, ncclOp) \
  IMPL_COLL3(coll, op, ncclFunc, i8,  int8_t,   ncclColl, ncclOp, ncclInt8) \
  IMPL_COLL3(coll, op, ncclFunc, u8,  uint8_t,  ncclColl, ncclOp, ncclUint8) \
  IMPL_COLL3(coll, op, ncclFunc, i32, int32_t,  ncclColl, ncclOp, ncclInt32) \
  IMPL_COLL3(coll, op, ncclFunc, u32, uint32_t, ncclColl, ncclOp, ncclUint32) \
  IMPL_COLL3(coll, op, ncclFunc, i64, int64_t,  ncclColl, ncclOp, ncclInt64) \
  IMPL_COLL3(coll, op, ncclFunc, u64, uint64_t, ncclColl, ncclOp, ncclUint64) \
  IMPL_COLL3(coll, op, ncclFunc, f16, half,     ncclColl, ncclOp, ncclFloat16) \
  IMPL_COLL3(coll, op, ncclFunc, f32, float,    ncclColl, ncclOp, ncclFloat32) \
  IMPL_COLL3(coll, op, ncclFunc, f64, double,   ncclColl, ncclOp, ncclFloat64) \
  IMPL_COLL3(coll, op, ncclFunc, b16, rccl_bfloat16, ncclColl, ncclOp, ncclBfloat16)

// Reduction define all functions
#define IMPL_COLL_R(collf, colln) \
  IMPL_COLL2(collf, sum,  FuncSum,  colln, ncclSum); \
  IMPL_COLL2(collf, prod, FuncProd, colln, ncclProd); \
  IMPL_COLL2(collf, min,  FuncMin,  colln, ncclMin); \
  IMPL_COLL2(collf, max,  FuncMax,  colln, ncclMax);

// Copy primitives only define one
#define IMPL_COLL_C(collf, colln) \
  IMPL_COLL3(collf, copy, FuncSum, i8, int8_t, colln, ncclSum, ncclInt8);

#define COLL_UNROLL 2

#endif
