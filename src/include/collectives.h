/*************************************************************************
 * Copyright (c) 2017-2022, NVIDIA CORPORATION. All rights reserved.
 * Modifications Copyright (c) 2019-2022 Advanced Micro Devices, Inc. All rights reserved.
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

#define FUNC_INDEX_P2P 1015
#define FUNC_INDEX_ALLTOALL_PIVOT 675

#define NCCL_FUNC_NAME(func, algo, proto, devredop, type) \
  ncclFunction_##func##_##algo##_##proto##_##devredop##_##type

#define NCCL_ONERANK_REDUCE_NAME(devredop, type) \
  ncclFunction_OneRankReduce_##devredop##_##type

#define NCCL_KERN_NAME(func, algo, proto, devredop, type) \
  ncclKernel_##func##_##algo##_##proto##_##devredop##_##type

#define NCCL_KERN_NAME_DEBUG(func, algo, proto, devredop, type) \
  ncclKernelDebug_##func##_##algo##_##proto##_##devredop##_##type

#define NCCL_IMPL_NAME(func, algo, proto) \
  nccl##func##algo##proto

// Declare rccl main/general kernel
extern __global__ void rccl_main_kernel(struct ncclDevComm* comm, uint64_t channelMask, struct ncclWork* workHead);
#ifdef ENABLE_COLLTRACE
extern __global__ void rccl_main_kernel_debug(struct ncclDevComm* comm, uint64_t channelMask, struct ncclWork* workHead);
#endif

// Declare OneRankReduce
extern __device__ void NCCL_ONERANK_REDUCE_NAME(PreMulSum, int8_t)();
extern __device__ void NCCL_ONERANK_REDUCE_NAME(PreMulSum, uint8_t)();
extern __device__ void NCCL_ONERANK_REDUCE_NAME(PreMulSum, int32_t)();
extern __device__ void NCCL_ONERANK_REDUCE_NAME(PreMulSum, uint32_t)();
extern __device__ void NCCL_ONERANK_REDUCE_NAME(PreMulSum, int64_t)();
extern __device__ void NCCL_ONERANK_REDUCE_NAME(PreMulSum, uint64_t)();
extern __device__ void NCCL_ONERANK_REDUCE_NAME(PreMulSum, half)();
extern __device__ void NCCL_ONERANK_REDUCE_NAME(PreMulSum, rccl_bfloat16)();
extern __device__ void NCCL_ONERANK_REDUCE_NAME(PreMulSum, float)();
extern __device__ void NCCL_ONERANK_REDUCE_NAME(PreMulSum, double)();

// CHUNKSIZE must be a multiple of SLICESIZE
#define ALLREDUCE_SLICESTEPS (NCCL_STEPS/4)
#define ALLREDUCE_CHUNKSTEPS (NCCL_STEPS/2)
#define ALLGATHER_SLICESTEPS (NCCL_STEPS/4)
#define ALLGATHER_CHUNKSTEPS (NCCL_STEPS/2)
#define REDUCESCATTER_SLICESTEPS (NCCL_STEPS/4)
#define REDUCESCATTER_CHUNKSTEPS (NCCL_STEPS/2)
#define BROADCAST_SLICESTEPS 1
#define BROADCAST_CHUNKSTEPS 1
#define REDUCE_SLICESTEPS 1
#define REDUCE_CHUNKSTEPS 1
#define NCCL_MAX_SLICE_PER_CHUNK 2  // max value for CHUNKSTEPS/SLICESTEPS, must accord with above
#define ALLTOALL_PIVOT_SLICESTEPS 2
#define ALLTOALL_PIVOT_CHUNKSTEPS 4

// We can't use the enum identifiers like ncclSum, ncclFloat, etc since this
// macro will be used in preprocessor conditionals where enums have no meaning.
#define NCCL_NVLS_SUPPORTS(/*ncclDataType_t*/ type, /*ncclDevRedOp_t*/ red) \
  (((type==2 || type==3) && (red==0 || red==2 || red==3)) || \
   ((type==4 || type==5) && (red==0 || red==2 || red==3)) || \
   ((type==6 || type==9) && (red==0 || red==2 || red==3)) || \
   (type==7 && red==0) || \
   (type==8 && red==0))

#endif
