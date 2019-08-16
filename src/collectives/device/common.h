#include "hip/hip_runtime.h"
/*************************************************************************
 * Copyright (c) 2017-2019, NVIDIA CORPORATION. All rights reserved.
 * Modifications Copyright (c) 2019 Advanced Micro Devices, Inc. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

#ifndef NCCL_DEVICE_COMMON_H_
#define NCCL_DEVICE_COMMON_H_

#include "../collectives.h"
#include "devcomm.h"
#include "nccl.h"
#include <type_traits>

// Exit If Abort Barrier across CTA: make sure all threads exit consistently
// Each thread sets a predicate to true if abort == 1
// all CTA's threads enter the barrier and do a popc on their predicates being True
// If any of the thread's predicate was True, all the threads call exit()
#if defined(__HIP_PLATFORM_HCC__) || defined(__HCC__) || defined(__HIPCC__)
#define exitIfAbortBarrier(abort, abortCount) \
  if (abort) __atomic_fetch_add(abortCount, 1, __ATOMIC_SEQ_CST); \
  __syncthreads(); \
  if (LOAD(abortCount)) { asm volatile ("s_endpgm"); return; }
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
  NCCL_COLL_NAME(coll, op, dtype), \
  NCCL_COLL_NAME(coll##LL, op, dtype)

#define NCCL_FUNC4(coll, op, dtype) \
  NCCL_FUNC5(coll##Ring, op, dtype)

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
  NCCL_FUNC4(coll, op, f64)
#define NCCL_FUNCS3B(coll, op) \
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

// Must be consistent with ncclColl_t
#define NCCL_FUNCS() { \
  NCCL_FUNCS2B(ncclBroadcast), \
  NCCL_FUNCS2A(ncclReduce), \
  NCCL_FUNCS2B(ncclAllGather), \
  NCCL_FUNCS2A(ncclReduceScatter), \
  NCCL_FUNCS2A(ncclAllReduce) }

// Must be consistent with the ncclFuncSet enum
using ncclFunc_t = void (*)(struct CollectiveArgs*);

static const __device__ constexpr ncclFunc_t ncclFuncs[]{
// Don't try to initialize the host shadow copy of this device-side global
// variable. There is no host pointer to a device-side function, which
// confuses clang. This will be fixed in the next clang release.
#if defined(__HIP_DEVICE_COMPILE__)
  NCCL_FUNCS2B(ncclBroadcast),
  NCCL_FUNCS2A(ncclReduce),
  NCCL_FUNCS2B(ncclAllGather),
  NCCL_FUNCS2A(ncclReduceScatter),
  NCCL_FUNCS2A(ncclAllReduce)
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
  if (c->funcIndex < 72) {
    if (c->funcIndex % 2) ncclBroadcastRingLL_copy_i8(&c->args);
    else ncclBroadcastRing_copy_i8(&c->args);
  }
  else if (c->funcIndex < 144) Caller<72, 144>::call(c);
  else if (c->funcIndex < 216) {
    if (c->funcIndex % 2) ncclAllGatherRingLL_copy_i8(&c->args);
    else ncclAllGatherRing_copy_i8(&c->args);
  }
  else Caller<216, 360>::call(c);
}

static __device__ void load_parallel(void* dst, void* src, size_t size, int tid, uint32_t* abortCount) {
  int* d = (int*)dst;
  int* s = (int*)src;
  // When aggregation is effective, if some threads have aborted inside the LL kernel,
  // make sure the rest of the threads abort as well
  exitIfAbortBarrier(0, abortCount);
  for (int o = tid; o < (size/sizeof(int)); o += blockDim.x) d[o] = s[o];
  __syncthreads();
}
static __device__ void load_coll(struct ncclColl* localColl, struct ncclColl* hostColl, int tid, uint32_t* abortCount) {
  load_parallel(localColl, hostColl, sizeof(struct ncclColl), tid, abortCount);
  if (tid == 0) hostColl->active = 0;
}

/* Functions for aggregation case */
#define IMPL_COLL_FUNC(coll, op, ncclFunc, dtype, ctype) \
__device__ void NCCL_COLL_NAME(coll, op, dtype)(struct CollectiveArgs* args) { \
  coll##Kernel<COLL_UNROLL, ncclFunc<ctype>, ctype>(args); \
}

#if NCCL_OP == 0
/* Kernels with the first operation inlined */
#define IMPL_COLL_KERN(coll, op, ncclFunc, dtype, ctype, fIndex) \
__launch_bounds__(MAXTHREADS+WARP_SIZE, 1) \
__global__ void NCCL_KERN_NAME(coll, op, dtype)(struct ncclColl firstColl) { \
  int tid = threadIdx.x; \
  int bid = blockIdx.x; \
  __shared__ struct ncclColl localColl; \
  __shared__ uint32_t abortCount; \
  if (tid == 0) abortCount = 0; \
  __syncthreads(); \
 \
  struct ncclDevComm* comm = firstColl.args.comm; \
  struct ncclChannel* channel = comm->channels+bid; \
  struct ncclColl* c; \
  channel->abortCount = &abortCount; \
  if (bid == 0) { \
    /* To optimize for latency, (only) the first operation is passed as argument.*/ \
    c = &firstColl; \
  } else { \
    c = &localColl; \
    load_coll(c, channel->devCollectives+channel->collFifoHead, tid, &abortCount); \
  } \
  while (1) { \
    if (tid < c->args.nThreads) { \
      if (c->funcIndex == fIndex) { \
        coll##Kernel<COLL_UNROLL, ncclFunc<ctype>, ctype>(&c->args); \
      } else { \
        NCCL_CALL_FUNCTIONS(c); \
      } \
    } \
    int nextIndex = c->nextIndex; \
    if (tid == 0) channel->collFifoHead = nextIndex; \
 \
    if (c->active == 2) { \
      return; \
    } \
 \
    /* Load next collective operation*/ \
    c = &localColl; /* for bid 0 */ \
    load_coll(c, channel->devCollectives+nextIndex, tid, &abortCount); \
  } \
}
#else
#define IMPL_COLL_KERN(coll, op, ncclFunc, dtype, ctype, fIndex)
#endif

// Only generate inline kernels for LL
#define IMPL_COLL4(coll, op, ncclFunc, dtype, ctype, ncclColl, ncclOp, ncclType, al) \
  IMPL_COLL_FUNC(coll, op, ncclFunc, dtype, ctype) \
  IMPL_COLL_FUNC(coll##LL, op, ncclFunc, dtype, ctype) \
  IMPL_COLL_KERN(coll##LL, op, ncclFunc, dtype, ctype, FUNC_INDEX(ncclColl, ncclOp, ncclType, 1, al)) \

#define IMPL_COLL3(coll, op, ncclFunc, dtype, ctype, ncclColl, ncclOp, ncclType) \
  IMPL_COLL4(coll##Ring, op, ncclFunc, dtype, ctype, ncclColl, ncclOp, ncclType, 0)

#define IMPL_COLL2(coll, op, ncclFunc, ncclColl, ncclOp) \
  IMPL_COLL3(coll, op, ncclFunc, i8,  int8_t,   ncclColl, ncclOp, ncclInt8) \
  IMPL_COLL3(coll, op, ncclFunc, u8,  uint8_t,  ncclColl, ncclOp, ncclUint8) \
  IMPL_COLL3(coll, op, ncclFunc, i32, int32_t,  ncclColl, ncclOp, ncclInt32) \
  IMPL_COLL3(coll, op, ncclFunc, u32, uint32_t, ncclColl, ncclOp, ncclUint32) \
  IMPL_COLL3(coll, op, ncclFunc, i64, int64_t,  ncclColl, ncclOp, ncclInt64) \
  IMPL_COLL3(coll, op, ncclFunc, u64, uint64_t, ncclColl, ncclOp, ncclUint64) \
  IMPL_COLL3(coll, op, ncclFunc, f16, half,     ncclColl, ncclOp, ncclFloat16) \
  IMPL_COLL3(coll, op, ncclFunc, f32, float,    ncclColl, ncclOp, ncclFloat32) \
  IMPL_COLL3(coll, op, ncclFunc, f64, double,   ncclColl, ncclOp, ncclFloat64)

#define COLL_UNROLL 2

#endif
