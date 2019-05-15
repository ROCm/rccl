/*************************************************************************
 * Copyright (c) 2017-2018, NVIDIA CORPORATION. All rights reserved.
 * Modifications Copyright (c) 2019 Advanced Micro Devices, Inc. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

#ifndef NCCL_COLLECTIVES_H_
#define NCCL_COLLECTIVES_H_

typedef enum { ncclCollBroadcast, ncclCollReduce, ncclCollAllGather, ncclCollReduceScatter, ncclCollAllReduce, ncclCollCount } ncclColl_t;

#define FUNC_INDEX(coll, redop, dtype, ll) ((((coll*ncclNumOps + redop)*ncclNumTypes) + dtype)*2+ll)

#define NCCL_COLL_NAME(coll, op, dtype) \
  coll##_##op##_##dtype

#define NCCL_KERN_NAME(coll, op, dtype) \
  coll##Kernel_##op##_##dtype

/* Declare all collective operations */
#define DECL_COLL4(coll, op, dtype) \
  extern __device__ __attribute__((noinline)) void NCCL_COLL_NAME(coll, op, dtype)(struct CollectiveArgs* args); \
  extern __global__ void NCCL_KERN_NAME(coll, op, dtype)(struct ncclColl coll); \

#define DECL_COLL3(coll, op, dtype) \
  DECL_COLL4(coll##LL, op, dtype) \
  DECL_COLL4(coll, op, dtype)

#define DECL_COLL2(coll, op) \
  DECL_COLL3(coll, op, i8) \
  DECL_COLL3(coll, op, u8) \
  DECL_COLL3(coll, op, i32) \
  DECL_COLL3(coll, op, u32) \
  DECL_COLL3(coll, op, i64) \
  DECL_COLL3(coll, op, u64) \
  DECL_COLL3(coll, op, f16) \
  DECL_COLL3(coll, op, f32) \
  DECL_COLL3(coll, op, f64)

#define DECL_COLL(coll) \
  DECL_COLL2(coll, sum) \
  DECL_COLL2(coll, prod) \
  DECL_COLL2(coll, min) \
  DECL_COLL2(coll, max)

#define DECL_ALL_COLLS \
  DECL_COLL2(ncclBroadcast, copy) \
  DECL_COLL(ncclReduce) \
  DECL_COLL2(ncclAllGather, copy) \
  DECL_COLL(ncclReduceScatter) \
  DECL_COLL(ncclAllReduce) \

DECL_ALL_COLLS

#define ALLREDUCE_SUBSTEPS 2
#define ALLREDUCE_BUFCHUNKS 2
#define ALLGATHER_SUBSTEPS 2
#define ALLGATHER_BUFCHUNKS 2
#define REDUCESCATTER_SUBSTEPS 2
#define REDUCESCATTER_BUFCHUNKS 2
#define BROADCAST_SUBSTEPS 8
#define BROADCAST_BUFCHUNKS 2
#define REDUCE_SUBSTEPS 8
#define REDUCE_BUFCHUNKS 2

#endif
