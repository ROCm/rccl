/*************************************************************************
 * Copyright (c) 2022, NVIDIA CORPORATION. All rights reserved.
 * Modifications Copyright (c) 2019-2022 Advanced Micro Devices, Inc. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

#include "devcomm.h"
#include "collectives.h"
#include "common_kernel.h"
#include "common.h"

namespace {
  template<typename T, typename RedOp>
#if defined(USE_INDIRECT_FUNCTION_CALL) && !defined(__gfx940__) && !defined(__gfx941__) && !defined(__gfx942__)
  __device__ void oneRankReduce() {
#else
  __device__ __attribute__((noinline)) void oneRankReduce() {
#endif
    ncclWork *w = &ncclShmem.work;
    int tid = threadIdx.x;
    int tn = blockDim.x;
    #pragma unroll 1
    for(int e=0; e < NCCL_MAX_WORK_ELEMENTS && w->elems[e].isUsed; e++) {
      ncclWorkElem *we = &w->elems[e];
      intptr_t eltN = we->count;
      int bid = we->bid;
      int bn = we->nChannels;
      T const *src = (T const*)we->sendbuff;
      T *dst = (T*)we->recvbuff;

      // each block/channel gets a roughly equal segment of 16 byte packs
      constexpr int EltPerPack = 16/sizeof(T);
      intptr_t packN = (eltN + EltPerPack-1) - (eltN + EltPerPack-1)%EltPerPack;
      intptr_t i0 = (bid+0)*(packN/bn) + (bid+0 < packN%bn ? bid+0 : packN%bn);
      intptr_t i1 = (bid+1)*(packN/bn) + (bid+1 < packN%bn ? bid+1 : packN%bn);
      i0 *= EltPerPack;
      i0 = i0 < eltN ? i0 : eltN;
      i1 *= EltPerPack;
      i1 = i1 < eltN ? i1 : eltN;
      src += i0;
      dst += i0;
      void *vsrc = (void*)src;
      void *vdst = (void*)dst;
      reduceCopy<COLL_UNROLL, RedOp, T, 0,1,1, 0,1,1, /*PreOpSrcs=*/1>
        (tid, tn, we->redOpArg, &(we->redOpArg), true, 1, &vsrc, 1, &vdst, i1-i0);
    }
  }
}

#if defined(USE_INDIRECT_FUNCTION_CALL) && !defined(__gfx940__) && !defined(__gfx941__) && !defined(__gfx942__)
#define INSTANTIATE(devredop, type) \
  __device__ void NCCL_ONERANK_REDUCE_NAME(devredop, type)() { \
    oneRankReduce<type, Func##devredop<type>>(); \
  }
#else
#define INSTANTIATE(devredop, type) \
  __device__ __attribute__((noinline)) void NCCL_ONERANK_REDUCE_NAME(devredop, type)() { \
    oneRankReduce<type, Func##devredop<type>>(); \
  }
#endif

INSTANTIATE(PreMulSum, int8_t)
INSTANTIATE(PreMulSum, uint8_t)
INSTANTIATE(PreMulSum, int32_t)
INSTANTIATE(PreMulSum, uint32_t)
INSTANTIATE(PreMulSum, int64_t)
INSTANTIATE(PreMulSum, uint64_t)
INSTANTIATE(PreMulSum, half)
#if defined(RCCL_BFLOAT16)
INSTANTIATE(PreMulSum, rccl_bfloat16)
#endif
INSTANTIATE(PreMulSum, float)
INSTANTIATE(PreMulSum, double)
