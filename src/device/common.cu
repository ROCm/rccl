/*************************************************************************
 * Copyright (c) 2015-2021, NVIDIA CORPORATION. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

#include "device.h"
#include "collectives.h"
#include "common.h"

__shared__ ncclShmemData ncclShmem;
#if __CUDA_ARCH__ < 700
  __shared__ ulong2 ncclShmemPerWarp[ncclShmemScratchWarpSize()*(NCCL_MAX_NTHREADS/WARP_SIZE)/sizeof(ulong2)];
#endif

struct RunWorkNop {
  __device__ void run(ncclWork *w) {}
};

__launch_bounds__(NCCL_MAX_NTHREADS, 1) __global__ void ncclDevKernel_Generic(struct ncclDevComm* comm, struct channelMasks channelMask, struct ncclWork* workHead) {
  ncclKernelMain<-1, RunWorkNop, false, 2>(comm, channelMask, workHead);
}
__launch_bounds__(NCCL_MAX_NTHREADS, 1) __global__ void ncclDevKernel_Generic_4(struct ncclDevComm* comm, struct channelMasks channelMask, struct ncclWork* workHead) {
  ncclKernelMain<-1, RunWorkNop, false, 4>(comm, channelMask, workHead);
}
#ifdef ENABLE_COLLTRACE
__launch_bounds__(NCCL_MAX_NTHREADS, 1) __global__ void ncclDevKernelDebug_Generic(struct ncclDevComm* comm, struct channelMasks channelMask, struct ncclWork* workHead) {
  ncclKernelMain<-1, RunWorkNop, true, 2>(comm, channelMask, workHead);
}
__launch_bounds__(NCCL_MAX_NTHREADS, 1) __global__ void ncclDevKernelDebug_Generic_4(struct ncclDevComm* comm, struct channelMasks channelMask, struct ncclWork* workHead) {
  ncclKernelMain<-1, RunWorkNop, true, 4>(comm, channelMask, workHead);
}
#endif

#ifdef USE_INDIRECT_FUNCTION_CALL
__device__ void ncclDevFunc_Nop();
#else
__device__ __attribute__((noinline)) void ncclDevFunc_Nop();
#endif
