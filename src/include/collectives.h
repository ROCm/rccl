/*************************************************************************
 * Copyright (c) 2017-2021, NVIDIA CORPORATION. All rights reserved.
 * Modifications Copyright (c) 2019-2021 Advanced Micro Devices, Inc. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

#ifndef NCCL_COLLECTIVES_H_
#define NCCL_COLLECTIVES_H_

#define FUNC_INDEX_P2P (NCCL_NUM_FUNCTIONS*NCCL_NUM_ALGORITHMS*NCCL_NUM_PROTOCOLS*ncclNumTypes*ncclNumOps)
#if defined(BUILD_ALLREDUCE_ONLY)
#define FUNC_INDEX(func, redop, ncclType, al, pr) (0)
#else
#define FUNC_INDEX(func, redop, ncclType, al, pr) ((((((func)*ncclNumOps + (redop))*ncclNumTypes) + (ncclType))*NCCL_NUM_ALGORITHMS+(al))*NCCL_NUM_PROTOCOLS+(pr))
#endif
#define NCCL_FUNC_NAME(func, algo, proto, redop, type) \
  ncclFunction_##func##_##algo##_##proto##_##redop##_##type

#define NCCL_KERN_NAME(func, algo, proto, redop, type) \
  ncclKernel_##func##_##algo##_##proto##_##redop##_##type

#define NCCL_IMPL_NAME(func, algo, proto) \
  nccl##func##algo##proto

/* Declare all collective operations */
#define DECL5(func, algo, proto, redop, type) \
  extern __device__ __attribute__((noinline)) void NCCL_FUNC_NAME(func, algo, proto, redop, type)(struct ncclWorkElem* args); \
  extern __global__ void NCCL_KERN_NAME(func, algo, proto, redop, type)(struct ncclWorkElem first); \

  //extern __device__ void NCCL_FUNC_NAME(func, algo, proto, redop, type)(); \
  //extern __global__ void NCCL_KERN_NAME(func, algo, proto, redop, type)(ncclWorkElem c); \

#define DECL4(func, algo, redop, type) \
  DECL5(func, algo, SIMPLE, redop, type) \
  DECL5(func, algo, LL,     redop, type) \
  DECL5(func, algo, LL128,  redop, type)

#define DECL3(func, redop, type) \
  DECL4(func, RING,    redop, type) \
  DECL4(func, TREE,    redop, type) \
  DECL4(func, COLLNET, redop, type)

#if defined(__CUDA_BF16_TYPES_EXIST__)
#define DECL2(func, redop) \
  DECL3(func, redop, int8_t) \
  DECL3(func, redop, uint8_t) \
  DECL3(func, redop, int32_t) \
  DECL3(func, redop, uint32_t) \
  DECL3(func, redop, int64_t) \
  DECL3(func, redop, uint64_t) \
  DECL3(func, redop, half) \
  DECL3(func, redop, float) \
  DECL3(func, redop, double) \
  DECL3(func, redop, __nv_bfloat16)
#else
#define DECL2(func, redop) \
  DECL3(func, redop, int8_t) \
  DECL3(func, redop, uint8_t) \
  DECL3(func, redop, int32_t) \
  DECL3(func, redop, uint32_t) \
  DECL3(func, redop, int64_t) \
  DECL3(func, redop, uint64_t) \
  DECL3(func, redop, half) \
  DECL3(func, redop, float) \
  DECL3(func, redop, double) \
  DECL3(func, redop, rccl_bfloat16)
#endif

#define DECL(func) \
  DECL2(func, Sum) \
  DECL2(func, Prod) \
  DECL2(func, Min) \
  DECL2(func, Max) \
  DECL2(func, Avg) \

#define DECL_ALL \
  DECL2(Broadcast, Sum) \
  DECL(Reduce) \
  DECL2(AllGather, Sum) \
  DECL(ReduceScatter) \
  DECL(AllReduce) \
  DECL5(SendRecv, RING, SIMPLE, Sum, int8_t) \

DECL_ALL

// CHUNKSIZE must be a multiple of SLICESIZE
//#define ALLREDUCE_SLICESTEPS (NCCL_STEPS/4)
//#define ALLREDUCE_CHUNKSTEPS (NCCL_STEPS/2)
//#define ALLGATHER_SLICESTEPS (NCCL_STEPS/4)
//#define ALLGATHER_CHUNKSTEPS (NCCL_STEPS/2)
//#define REDUCESCATTER_SLICESTEPS (NCCL_STEPS/4)
//#define REDUCESCATTER_CHUNKSTEPS (NCCL_STEPS/2)
#define ALLREDUCE_SLICESTEPS 4
#define ALLREDUCE_CHUNKSTEPS 4
#define ALLGATHER_SLICESTEPS 4
#define ALLGATHER_CHUNKSTEPS 4
#define REDUCESCATTER_SLICESTEPS 4
#define REDUCESCATTER_CHUNKSTEPS 4
#define BROADCAST_SLICESTEPS 1
#define BROADCAST_CHUNKSTEPS 1
#define REDUCE_SLICESTEPS 1
#define REDUCE_CHUNKSTEPS 1
#define SENDRECV_SLICEFACTOR 1

#endif
