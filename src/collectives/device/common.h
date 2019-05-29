/*************************************************************************
 * Copyright (c) 2017-2018, NVIDIA CORPORATION. All rights reserved.
 * Modifications Copyright (c) 2019 Advanced Micro Devices, Inc. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

#ifndef NCCL_DEVICE_COMMON_H_
#define NCCL_DEVICE_COMMON_H_

#include <hip/hip_runtime.h>

#include "../collectives.h"
#include "core.h"
#include "nccl.h"

#include <type_traits>

typedef void(*ncclKern_t)(struct CollectiveArgs* args);
#define NCCL_FUNC4(coll, op, dtype) \
  NCCL_COLL_NAME(coll, op, dtype), \
  NCCL_COLL_NAME(coll##LL, op, dtype)  \

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
using ncclKern_t = void (*)(struct CollectiveArgs*);

static const __device__ constexpr ncclKern_t ncclFuncs[]{
#if defined(__HIP_DEVICE_COMPILE__)
  NCCL_FUNCS2B(ncclBroadcast),
  NCCL_FUNCS2A(ncclReduce),
  NCCL_FUNCS2B(ncclAllGather),
  NCCL_FUNCS2A(ncclReduceScatter),
  NCCL_FUNCS2A(ncclAllReduce)
#endif
// Don't try to initialize the host shadow copy of this device-side global
// variable. There is no host pointer to a device-side function, which
// confuses clang. This will be fixed in the next clang release.
#if __CUDA_ARCH__
  NCCL_FUNCS2B(ncclBroadcast),
  NCCL_FUNCS2A(ncclReduce),
  NCCL_FUNCS2B(ncclAllGather),
  NCCL_FUNCS2A(ncclReduceScatter),
  NCCL_FUNCS2A(ncclAllReduce)
#endif
};

template<unsigned short f, unsigned short l>
struct Caller {
  static
  __device__ void call(ncclColl* const c) noexcept
  {
    constexpr unsigned short m = f + (l - f) / 2;

    return (c->funcIndex < m) ? Caller<f, m>::call(c) : Caller<m, l>::call(c);
  }
};

template<unsigned short f>
struct Caller<f, f + 1>{
  static
  __device__ void call(struct ncclColl* const c) noexcept { ncclFuncs[f](&c->args); }
};

inline
__device__
void NCCL_CALL_FUNCTIONS(struct ncclColl* const c) noexcept
{
  if (c->funcIndex < 72) {
    if (c->funcIndex % 2) ncclBroadcastLL_copy_i8(&c->args);
    else ncclBroadcast_copy_i8(&c->args);
  }
  else if (c->funcIndex < 144) Caller<72, 144>::call(c);
  else if (c->funcIndex < 216) {
    if (c->funcIndex % 2) ncclAllGatherLL_copy_i8(&c->args);
    else ncclAllGather_copy_i8(&c->args);
  }
  else Caller<216, 360>::call(c);
}

static __device__ void load_parallel(void* dst, void* src, size_t size, int tid) {
  int* d = (int*)dst;
  int* s = (int*)src;
  __syncthreads();
  for (int o = tid; o < (size/sizeof(int)); o += blockDim.x) d[o] = s[o];
  __syncthreads();
}
static __device__ void load_coll(struct ncclColl* localColl, struct ncclColl* hostColl, int tid) {
  load_parallel(localColl, hostColl, sizeof(struct ncclColl), tid);
  if (tid == 0) hostColl->active = 0;
}

/* Functions for aggregation case */
#define IMPL_COLL4(coll, op, ncclFunc, dtype, ctype) \
__device__ void NCCL_COLL_NAME(coll, op, dtype)(struct CollectiveArgs* args) { \
  coll##Kernel<UNROLL, ncclFunc<ctype>, ctype>(args); \
}
/* Kernels with the first operation inlined */
#define IMPL_COLL4K(coll, op, ncclFunc, dtype, ctype, fIndex) \
__launch_bounds__(MAXTHREADS+WARP_SIZE, 1) \
__global__ void NCCL_KERN_NAME(coll, op, dtype)(struct ncclColl firstColl) { \
  int tid = threadIdx.x; \
  int bid = blockIdx.x; \
  __shared__ struct ncclColl localColl; \
 \
  struct ncclComm* comm = firstColl.args.comm; \
  struct ncclRing* ring = comm->rings+bid; \
  struct ncclColl* c; \
  if (bid == 0) { \
    /* To optimize for latency, (only) the first operation is passed as argument.*/ \
    c = &firstColl; \
  } else { \
    c = &localColl; \
    load_coll(c, ring->devCollectives+ring->collFifoHead, tid); \
  } \
  while (1) { \
    if (tid < c->nThreads) { \
      if (c->funcIndex == fIndex) { \
        coll##Kernel<UNROLL, ncclFunc<ctype>, ctype>(&c->args); \
      } else { \
        NCCL_CALL_FUNCTIONS(c); \
      } \
    } \
    int nextIndex = c->nextIndex; \
    if (tid == 0) ring->collFifoHead = nextIndex; \
 \
    if (c->active == 2) { \
      return; \
    } \
 \
    /* Load next collective operation*/ \
    c = &localColl; /* for bid 0 */ \
    load_coll(c, ring->devCollectives+nextIndex, tid); \
  } \
}

#define IMPL_COLL3(coll, op, ncclFunc, dtype, ctype, ncclColl, ncclOp, ncclType) \
  IMPL_COLL4(coll##LL, op, ncclFunc, dtype, ctype) \
  IMPL_COLL4K(coll##LL, op, ncclFunc, dtype, ctype, FUNC_INDEX(ncclColl, ncclOp, ncclType, 1)) \
  IMPL_COLL4(coll, op, ncclFunc, dtype, ctype) \
  IMPL_COLL4K(coll, op, ncclFunc, dtype, ctype, FUNC_INDEX(ncclColl, ncclOp, ncclType, 0)) \

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

#endif
