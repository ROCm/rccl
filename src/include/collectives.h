/*************************************************************************
 * Copyright (c) 2017-2021, NVIDIA CORPORATION. All rights reserved.
 * Modifications Copyright (c) 2019-2021 Advanced Micro Devices, Inc. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

#ifndef NCCL_COLLECTIVES_H_
#define NCCL_COLLECTIVES_H_

enum ncclDevRedOp_t {
  ncclDevSum, ncclDevProd, ncclDevMax, ncclDevMin,
  ncclDevPreMulSum, ncclDevSumPostDiv,
  ncclNumDevRedOps
};
struct ncclDevRedOpFull {
  ncclDevRedOp_t op;
  bool scalarArgIsPtr;
  uint64_t scalarArg;
};

#define FUNC_INDEX_P2P (ncclNumTypes+NCCL_NUM_FUNCTIONS*NCCL_NUM_ALGORITHMS*NCCL_NUM_PROTOCOLS*ncclNumTypes*ncclNumDevRedOps)
#define FUNC_INDEX(func, devredop, ncclType, al, pr) ((((((func)*ncclNumDevRedOps + (devredop))*ncclNumTypes) + (ncclType))*NCCL_NUM_ALGORITHMS+(al))*NCCL_NUM_PROTOCOLS+(pr))

#define NCCL_FUNC_NAME(func, algo, proto, devredop, type) \
  ncclFunction_##func##_##algo##_##proto##_##devredop##_##type

#define NCCL_ONERANK_REDUCE_NAME(devredop, type) \
  ncclFunction_OneRankReduce_##devredop##_##type

#define NCCL_KERN_NAME(func, algo, proto, devredop, type) \
  ncclKernel_##func##_##algo##_##proto##_##devredop##_##type

#define NCCL_IMPL_NAME(func, algo, proto) \
  nccl##func##algo##proto

/* Declare all collective operations */
#define DECL5(func, algo, proto, devredop, type) \
  extern __device__ __attribute__((noinline)) void NCCL_FUNC_NAME(func, algo, proto, devredop, type)(struct ncclWorkElem* args); \
  extern __global__ void NCCL_KERN_NAME(func, algo, proto, devredop, type)(ncclWorkElem c); \

#define CONCAT(a,b) a##b
#define MACRO_IF(cond, t, f) CONCAT(MACRO_IF_, cond)(t, f)
#define MACRO_IF_0(t, f) f
#define MACRO_IF_1(t, f) t

#define DECL4(func, algo, devredop, type, undef) \
  MACRO_IF(undef, /*undefined*/, DECL5(func, algo, SIMPLE, devredop, type)) \
  MACRO_IF(undef, /*undefined*/, DECL5(func, algo, LL,     devredop, type)) \
  MACRO_IF(undef, /*undefined*/, DECL5(func, algo, LL128,  devredop, type))

#define DECL3(func, devredop, type, undef) \
  DECL4(func, RING,    devredop, type, undef) \
  DECL4(func, TREE,    devredop, type, undef) \
  DECL4(func, COLLNET, devredop, type, undef)

#if defined(RCCL_BFLOAT16)
#define DECL2(func, devredop, undefForFloat) \
  DECL3(func, devredop, int8_t, /*undef=*/0) \
  DECL3(func, devredop, uint8_t, /*undef=*/0) \
  DECL3(func, devredop, int32_t, /*undef=*/0) \
  DECL3(func, devredop, uint32_t, /*undef=*/0) \
  DECL3(func, devredop, int64_t, /*undef=*/0) \
  DECL3(func, devredop, uint64_t, /*undef=*/0) \
  DECL3(func, devredop, half, /*undef=*/undefForFloat) \
  DECL3(func, devredop, float, /*undef=*/undefForFloat) \
  DECL3(func, devredop, double, /*undef=*/undefForFloat) \
  DECL3(func, devredop, rccl_bfloat16, /*undef=*/undefForFloat)
#else
#define DECL2(func, devredop, undefForFloat) \
  DECL3(func, devredop, int8_t, /*undef=*/0) \
  DECL3(func, devredop, uint8_t, /*undef=*/0) \
  DECL3(func, devredop, int32_t, /*undef=*/0) \
  DECL3(func, devredop, uint32_t, /*undef=*/0) \
  DECL3(func, devredop, int64_t, /*undef=*/0) \
  DECL3(func, devredop, uint64_t, /*undef=*/0) \
  DECL3(func, devredop, half, /*undef=*/undefForFloat) \
  DECL3(func, devredop, float, /*undef=*/undefForFloat) \
  DECL3(func, devredop, double, /*undef=*/undefForFloat)
#endif

#define DECL(func) \
  DECL2(func, Sum, /*undefForFloat=*/0) \
  DECL2(func, Prod, /*undefForFloat=*/0) \
  DECL2(func, Min, /*undefForFloat=*/0) \
  DECL2(func, Max, /*undefForFloat=*/0) \
  DECL2(func, PreMulSum, /*undefForFloat=*/0) \
  DECL2(func, SumPostDiv, /*undefForFloat=*/1)

DECL2(Broadcast, Sum, /*undefForFloat=*/0)
DECL(Reduce)
DECL2(AllGather, Sum, /*undefForFloat=*/0)
DECL(ReduceScatter)
DECL(AllReduce)
DECL5(SendRecv, RING, SIMPLE, Sum, int8_t)

extern __device__ void NCCL_ONERANK_REDUCE_NAME(PreMulSum, int8_t)(struct ncclWorkElem* args);
extern __device__ void NCCL_ONERANK_REDUCE_NAME(PreMulSum, uint8_t)(struct ncclWorkElem* args);
extern __device__ void NCCL_ONERANK_REDUCE_NAME(PreMulSum, int32_t)(struct ncclWorkElem* args);
extern __device__ void NCCL_ONERANK_REDUCE_NAME(PreMulSum, uint32_t)(struct ncclWorkElem* args);
extern __device__ void NCCL_ONERANK_REDUCE_NAME(PreMulSum, int64_t)(struct ncclWorkElem* args);
extern __device__ void NCCL_ONERANK_REDUCE_NAME(PreMulSum, uint64_t)(struct ncclWorkElem* args);
extern __device__ void NCCL_ONERANK_REDUCE_NAME(PreMulSum, half)(struct ncclWorkElem* args);
#if defined(RCCL_BFLOAT16)
extern __device__ void NCCL_ONERANK_REDUCE_NAME(PreMulSum, rccl_bfloat16)(struct ncclWorkElem* args);
#endif
extern __device__ void NCCL_ONERANK_REDUCE_NAME(PreMulSum, float)(struct ncclWorkElem* args);
extern __device__ void NCCL_ONERANK_REDUCE_NAME(PreMulSum, double)(struct ncclWorkElem* args);

// CHUNKSIZE must be a multiple of SLICESIZE
//#define ALLREDUCE_SLICESTEPS (NCCL_STEPS/4)
//#define ALLREDUCE_CHUNKSTEPS (NCCL_STEPS/2)
//#define ALLGATHER_SLICESTEPS (NCCL_STEPS/4)
//#define ALLGATHER_CHUNKSTEPS (NCCL_STEPS/2)
//#define REDUCESCATTER_SLICESTEPS (NCCL_STEPS/4)
//#define REDUCESCATTER_CHUNKSTEPS (NCCL_STEPS/2)
#define ALLREDUCE_SLICESTEPS 1
#define ALLREDUCE_CHUNKSTEPS 1
#define ALLGATHER_SLICESTEPS 1
#define ALLGATHER_CHUNKSTEPS 1
#define REDUCESCATTER_SLICESTEPS 2
#define REDUCESCATTER_CHUNKSTEPS 4
#define BROADCAST_SLICESTEPS 1
#define BROADCAST_CHUNKSTEPS 2
#define REDUCE_SLICESTEPS 1
#define REDUCE_CHUNKSTEPS 2
#define SENDRECV_SLICEFACTOR 1
#define NCCL_MAX_SLICE_PER_CHUNK 2  // max value for CHUNKSTEPS/SLICESTEPS, must accord with above

#endif
