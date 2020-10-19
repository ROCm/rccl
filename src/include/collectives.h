/*************************************************************************
 * Copyright (c) 2017-2020, NVIDIA CORPORATION. All rights reserved.
 * Modifications Copyright (c) 2019-2020 Advanced Micro Devices, Inc. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

#ifndef NCCL_COLLECTIVES_H_
#define NCCL_COLLECTIVES_H_

#define FUNC_INDEX_P2P (4+NCCL_NUM_FUNCTIONS*NCCL_NUM_ALGORITHMS*NCCL_NUM_PROTOCOLS*ncclNumTypes*ncclNumOps)
#define FUNC_INDEX(coll, redop, dtype, al, pr) ((coll >= NCCL_NUM_FUNCTIONS) \
  ? (coll-NCCL_NUM_FUNCTIONS+NCCL_NUM_FUNCTIONS*NCCL_NUM_ALGORITHMS*NCCL_NUM_PROTOCOLS*ncclNumTypes*ncclNumOps) \
  : ((((((coll)*ncclNumOps + (redop))*ncclNumTypes) + (dtype))*NCCL_NUM_ALGORITHMS+(al))*NCCL_NUM_PROTOCOLS+(pr)))

#define NCCL_COLL_NAME(coll, op, dtype) \
  coll##_##op##_##dtype

#define NCCL_KERN_NAME(coll, op, dtype) \
  coll##Kernel_##op##_##dtype

/* Declare all collective operations */
#define DECL_COLL5(coll, op, dtype) \
  extern __device__ __attribute__((noinline)) void NCCL_COLL_NAME(coll, op, dtype)(struct CollectiveArgs* args); \
  extern __global__ void NCCL_KERN_NAME(coll, op, dtype)(struct ncclDevComm* comm); \

#define DECL_COLL4(coll, op, dtype) \
  DECL_COLL5(coll, op, dtype) \
  DECL_COLL5(coll##LL, op, dtype) \
  DECL_COLL5(coll##LL128, op, dtype)

#define DECL_COLL3(coll, op, dtype) \
  DECL_COLL4(coll##Ring, op, dtype) \
  DECL_COLL4(coll##Tree, op, dtype) \
  DECL_COLL4(coll##CollNet, op, dtype)

#define DECL_COLL2(coll, op) \
  DECL_COLL3(coll, op, i8) \
  DECL_COLL3(coll, op, u8) \
  DECL_COLL3(coll, op, i32) \
  DECL_COLL3(coll, op, u32) \
  DECL_COLL3(coll, op, i64) \
  DECL_COLL3(coll, op, u64) \
  DECL_COLL3(coll, op, f16) \
  DECL_COLL3(coll, op, f32) \
  DECL_COLL3(coll, op, f64) \
  DECL_COLL3(coll, op, b16)

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
  DECL_COLL5(ncclGather, copy, i8) \
  DECL_COLL5(ncclScatter, copy, i8) \
  DECL_COLL5(ncclAllToAll, copy, i8) \
  DECL_COLL5(ncclAllToAllv, copy, i8) \
  DECL_COLL5(ncclSendRecv, copy, i8) \

DECL_ALL_COLLS

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
#define GATHER_SLICESTEPS 4
#define GATHER_CHUNKSTEPS 4
#define SCATTER_SLICESTEPS 4
#define SCATTER_CHUNKSTEPS 4
#define ALLTOALL_SLICESTEPS 4
#define ALLTOALL_CHUNKSTEPS 4
#define ALLTOALLV_SLICESTEPS 4
#define ALLTOALLV_CHUNKSTEPS 4

#endif
