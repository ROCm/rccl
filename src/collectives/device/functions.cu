/*************************************************************************
 * Copyright (c) 2015-2020, NVIDIA CORPORATION. All rights reserved.
 * Modifications Copyright (c) 2019-2021 Advanced Micro Devices, Inc. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

#include "devcomm.h"
#include "all_gather.h"
#include "all_reduce.h"
#include "broadcast.h"
#include "reduce.h"
#include "reduce_scatter.h"
#include "sendrecv.h"
#include "common.h"
#include "collectives.h"

__device__ struct ncclShmemData* ncclShmem;

IMPL_COLL_P(SendRecv);
IMPL_COLL_C(AllGather);
IMPL_COLL_CLIQUE(AllReduce);
IMPL_COLL_C(Broadcast);
IMPL_COLL_R(Reduce);
IMPL_COLL_R(ReduceScatter);

#if defined(__HIP_PLATFORM_HCC__) || defined(__HCC__) || defined(__HIPCC__)
#else
#define NCCL_FUNC5(func, algo, redop, type) \
  NCCL_FUNC_NAME(func, algo, LL,     redop, type), \
  NCCL_FUNC_NAME(func, algo, LL128,  redop, type), \
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
  NCCL_FUNC4(func, redop, double)
#define NCCL_FUNCS3B(func, redop) \
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
  NCCL_FUNC_NAME(SendRecv, RING, SIMPLE, Sum, int8_t),\
  NCCL_FUNCS2B(Broadcast), \
  NCCL_FUNCS2A(Reduce), \
  NCCL_FUNCS2B(AllGather), \
  NCCL_FUNCS2A(ReduceScatter), \
  NCCL_FUNCS2A(AllReduce) }

// Must be consistent with the ncclFuncSet enum
__device__ ncclKern_t ncclFuncs[1+NCCL_NUM_FUNCTIONS*ncclNumOps*ncclNumTypes*NCCL_NUM_ALGORITHMS*NCCL_NUM_PROTOCOLS] = {
// Don't try to initialize the host shadow copy of this device-side global
// variable. There is no host pointer to a device-side function, which
// confuses clang. This will be fixed in the next clang release.
#if __CUDA_ARCH__
  NCCL_FUNC_NAME(SendRecv, RING, SIMPLE, Sum, int8_t),
  NCCL_FUNCS2B(Broadcast),
  NCCL_FUNCS2A(Reduce),
  NCCL_FUNCS2B(AllGather),
  NCCL_FUNCS2A(ReduceScatter),
  NCCL_FUNCS2A(AllReduce)
#endif
};
#endif

// Workaround for https://reviews.llvm.org/D55580
__device__ void ncclWorkaroundClangD55580() {}
