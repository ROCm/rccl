#include "hip/hip_runtime.h"
/*************************************************************************
 * Copyright (c) 2017-2019, NVIDIA CORPORATION. All rights reserved.
 * Modifications Copyright (c) 2019 Advanced Micro Devices, Inc. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

#ifndef NCCL_DEVICE_COMMON_H_
#define NCCL_DEVICE_COMMON_H_

#include "collectives.h"
#include "devcomm.h"

// Exit If Abort Barrier across CTA: make sure all threads exit consistently
// Each thread sets a predicate to true if abort == 1
// all CTA's threads enter the barrier and do a popc on their predicates being True
// If any of the thread's predicate was True, all the threads call exit()
#if defined(__HIP_PLATFORM_HCC__) || defined(__HCC__) || defined(__HIPCC__)
#define exitIfAbortBarrier(abort, abortCount) \
  if (abort) __atomic_fetch_add(abortCount, 1, __ATOMIC_SEQ_CST); \
  __syncthreads(); \
  if (LOAD(abortCount)) { asm volatile ("s_endpgm"); return; }
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
  NCCL_COLL_NAME(coll##LL128, op, dtype), \
  NCCL_COLL_NAME(coll, op, dtype)

#define NCCL_FUNC4(coll, op, dtype) \
  NCCL_FUNC5(coll##Tree, op, dtype), \
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
  NCCL_FUNCS2A(ncclAllReduce) }

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
  if (c->funcIndex < 240) {
    if (c->funcIndex % 6 == 0) ncclBroadcastTreeLL_copy_i8(&c->args);
    else if (c->funcIndex % 6 == 1) ncclBroadcastTreeLL128_copy_i8(&c->args);
    else if (c->funcIndex % 6 == 2) ncclBroadcastTree_copy_i8(&c->args);
    else if (c->funcIndex % 6 == 3) ncclBroadcastRingLL_copy_i8(&c->args);
    else if (c->funcIndex % 6 == 4) ncclBroadcastRingLL128_copy_i8(&c->args);
    else ncclBroadcastRing_copy_i8(&c->args);
  }
  else if (c->funcIndex < 480) Caller<240, 480>::call(c);
  else if (c->funcIndex < 720) {
    if (c->funcIndex % 6 == 0) ncclAllGatherTreeLL_copy_i8(&c->args);
    else if (c->funcIndex % 6 == 1) ncclAllGatherTreeLL128_copy_i8(&c->args);
    else if (c->funcIndex % 6 == 2) ncclAllGatherTree_copy_i8(&c->args);
    else if (c->funcIndex % 6 == 3) ncclAllGatherRingLL_copy_i8(&c->args);
    else if (c->funcIndex % 6 == 4) ncclAllGatherRingLL128_copy_i8(&c->args);
    else ncclAllGatherRing_copy_i8(&c->args);
  }
  else Caller<720, 1200>::call(c);
}

static __device__ void load_parallel(void* dst, void* src, size_t size, int tid, uint32_t* abortCount) {
  int* d = (int*)dst;
  int* s = (int*)src;
  for (int o = tid; o < (size/sizeof(int)); o += blockDim.x) d[o] = s[o];
}

static __device__ void load_coll(struct ncclColl* localColl, struct ncclColl* hostColl, int tid, struct ncclDevComm* comm, uint32_t* abortCount) {
  // Check whether the last operation was aborted and make sure all threads exit
  int abort = tid == 0 ? *(comm->abortFlag) : 0;
  exitIfAbortBarrier(abort, abortCount);
  load_parallel(localColl, hostColl, sizeof(struct ncclColl), tid, abortCount);
  __syncthreads();
  if (tid == 0) hostColl->active = 0;
}

extern __device__ volatile uint64_t* ncclShmem;

/* Functions for aggregation case */
#define IMPL_COLL_FUNC(coll, op, ncclFunc, dtype, ctype) \
__device__ void NCCL_COLL_NAME(coll, op, dtype)(struct CollectiveArgs* args) { \
  coll##Kernel<COLL_UNROLL, ncclFunc<ctype>, ctype>(args); \
}

/* Kernels with the first operation inlined */
#define IMPL_COLL_KERN(coll, op, ncclFunc, dtype, ctype, fIndex) \
__global__ void NCCL_KERN_NAME(coll, op, dtype)(struct ncclColl firstColl) { \
  int tid = threadIdx.x; \
  int bid = blockIdx.x; \
  __shared__ volatile uint64_t shmem[NCCL_LL128_SHMEM_SIZE]; \
  ncclShmem = shmem; \
  __shared__ struct ncclColl localColl; \
  __shared__ uint32_t abortCount; \
  __shared__ uint32_t sync[NCCL_LL128_MAX_NTHREADS/WARP_SIZE]; \
  if (tid == 0) abortCount = 0; \
  __syncthreads(); \
 \
  struct ncclDevComm* comm = firstColl.args.comm; \
  struct ncclChannel* channel = comm->channels+bid; \
  struct ncclColl* c; \
  channel->abortCount = &abortCount; \
  channel->sync = sync; \
  if (bid == 0) { \
    /* To optimize for latency, (only) the first operation is passed as argument.*/ \
    c = &firstColl; \
  } else { \
    c = &localColl; \
    load_coll(c, channel->devCollectives+channel->collFifoHead, tid, comm, &abortCount); \
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
    load_coll(c, channel->devCollectives+nextIndex, tid, comm, &abortCount); \
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
  IMPL_COLL_KERN_##op(coll##LL, op, ncclFunc, dtype, ctype, FUNC_INDEX(ncclColl, ncclOp, ncclType, al, NCCL_PROTO_LL)) \

#define IMPL_COLL3(coll, op, ncclFunc, dtype, ctype, ncclColl, ncclOp, ncclType) \
  IMPL_COLL4(coll##Tree, op, ncclFunc, dtype, ctype, ncclColl, ncclOp, ncclType, NCCL_ALGO_TREE) \
  IMPL_COLL4(coll##Ring, op, ncclFunc, dtype, ctype, ncclColl, ncclOp, ncclType, NCCL_ALGO_RING)

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
