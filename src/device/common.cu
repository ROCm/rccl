/*************************************************************************
 * Copyright (c) 2015-2021, NVIDIA CORPORATION. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

#include "collectives.h"
#include "common.h"
#include "device.h"
#include "device_table.h"

__shared__ ncclShmemData ncclShmem;
#if __CUDA_ARCH__ < 700
  __shared__ ulong2 ncclShmemPerWarp[ncclShmemScratchWarpSize()*(NCCL_MAX_NTHREADS/WARP_SIZE)/sizeof(ulong2)];
#endif

struct RunWorkNop {
  __device__ void run() {}
};

#ifdef BUILD_GENERIC_KERNELS
// Generic kernel that dispatches to type-specific device functions
// Uses type-specific tables for efficient dispatch

// Helper macro to generate type dispatch switch cases
#define NCCL_TYPE_SWITCH_CASES(unroll) \
    case 0: NCCL_CALL_FUNCTIONS_i8_##unroll(funcId); break; \
    case 1: NCCL_CALL_FUNCTIONS_u8_##unroll(funcId); break; \
    case 2: NCCL_CALL_FUNCTIONS_i32_##unroll(funcId); break; \
    case 3: NCCL_CALL_FUNCTIONS_u32_##unroll(funcId); break; \
    case 4: NCCL_CALL_FUNCTIONS_i64_##unroll(funcId); break; \
    case 5: NCCL_CALL_FUNCTIONS_u64_##unroll(funcId); break; \
    case 6: NCCL_CALL_FUNCTIONS_f16_##unroll(funcId); break; \
    case 7: NCCL_CALL_FUNCTIONS_f32_##unroll(funcId); break; \
    case 8: NCCL_CALL_FUNCTIONS_f64_##unroll(funcId); break; \
    case 9: NCCL_CALL_FUNCTIONS_bf16_##unroll(funcId); break; \
    case 10: NCCL_CALL_FUNCTIONS_f8e4m3_##unroll(funcId); break; \
    case 11: NCCL_CALL_FUNCTIONS_f8e5m2_##unroll(funcId); break; \
    default: NCCL_CALL_FUNCTIONS_i8_##unroll(funcId); break;

// Macro to define callTypeSpecificFunction for each unroll factor
#define DEFINE_CALL_TYPE_SPECIFIC_FUNCTION(unroll) \
__device__ __forceinline__ void callTypeSpecificFunction_##unroll(int funcId) \
{ \
    extern __device__ __constant__ unsigned char ncclDevFuncIdToType[]; \
    int typeIdx = ncclDevFuncIdToType[funcId]; \
    switch(typeIdx) { NCCL_TYPE_SWITCH_CASES(unroll) } \
}

DEFINE_CALL_TYPE_SPECIFIC_FUNCTION(1)
DEFINE_CALL_TYPE_SPECIFIC_FUNCTION(2)
DEFINE_CALL_TYPE_SPECIFIC_FUNCTION(4)

#undef DEFINE_CALL_TYPE_SPECIFIC_FUNCTION
#undef NCCL_TYPE_SWITCH_CASES

// Macro to define GenericDispatcher structs
#define DEFINE_GENERIC_DISPATCHER(unroll) \
struct GenericDispatcher_##unroll \
{ \
    static __device__ __forceinline__ void dispatch(int funcId, int /*unroll*/) \
    { \
        callTypeSpecificFunction_##unroll(funcId); \
    } \
};

DEFINE_GENERIC_DISPATCHER(1)
DEFINE_GENERIC_DISPATCHER(2)
DEFINE_GENERIC_DISPATCHER(4)

#undef DEFINE_GENERIC_DISPATCHER

// Generic kernels - dispatch to type-specific tables at runtime
__launch_bounds__(NCCL_MAX_NTHREADS, 1) __global__ void ncclDevKernel_Generic_1(ncclDevKernelArgsDefaultStorage NCCL_GRID_CONSTANT const argsStorage) {
  ncclKernelMain<-1, RunWorkNop, GenericDispatcher_1, /*COLLTRACE*/ false, /*Unroll*/ 1>(&argsStorage.args);
}
__launch_bounds__(NCCL_MAX_NTHREADS, 1) __global__ void ncclDevKernel_Generic_2(ncclDevKernelArgsDefaultStorage NCCL_GRID_CONSTANT const argsStorage) {
  ncclKernelMain<-1, RunWorkNop, GenericDispatcher_2, /*COLLTRACE*/ false, /*Unroll*/ 2>(&argsStorage.args);
}
__launch_bounds__(NCCL_MAX_NTHREADS, 1) __global__ void ncclDevKernel_Generic_4(ncclDevKernelArgsDefaultStorage NCCL_GRID_CONSTANT const argsStorage) {
  ncclKernelMain<-1, RunWorkNop, GenericDispatcher_4, /*COLLTRACE*/ false, /*Unroll*/ 4>(&argsStorage.args);
}

#ifdef ENABLE_COLLTRACE
__launch_bounds__(NCCL_MAX_NTHREADS, 1) __global__ void ncclDevKernelDebug_Generic_1(ncclDevKernelArgsDefaultStorage NCCL_GRID_CONSTANT const argsStorage) {
  ncclKernelMain<-1, RunWorkNop, GenericDispatcher_1, /*COLLTRACE*/ true, /*Unroll*/ 1>(&argsStorage.args);
}
__launch_bounds__(NCCL_MAX_NTHREADS, 1) __global__ void ncclDevKernelDebug_Generic_2(ncclDevKernelArgsDefaultStorage NCCL_GRID_CONSTANT const argsStorage) {
  ncclKernelMain<-1, RunWorkNop, GenericDispatcher_2, /*COLLTRACE*/ true, /*Unroll*/ 2>(&argsStorage.args);
}
__launch_bounds__(NCCL_MAX_NTHREADS, 1) __global__ void ncclDevKernelDebug_Generic_4(ncclDevKernelArgsDefaultStorage NCCL_GRID_CONSTANT const argsStorage) {
  ncclKernelMain<-1, RunWorkNop, GenericDispatcher_4, /*COLLTRACE*/ true, /*Unroll*/ 4>(&argsStorage.args);
}
#endif // ENABLE_COLLTRACE
#endif // BUILD_GENERIC_KERNELS

__device__ void ncclDevFunc_Nop();
