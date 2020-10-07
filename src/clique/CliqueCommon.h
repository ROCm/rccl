/*
Copyright (c) 2020 Advanced Micro Devices, Inc. All rights reserved.

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in
all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
THE SOFTWARE.
*/

#ifndef CLIQUE_COMMON_H
#define CLIQUE_COMMON_H

#include "nccl.h"
#include <cstdint>
#include "rccl_bfloat16.h"
#include "reduce_kernel.h"

#define MIN_CLIQUE_SIZE 2
#define MAX_CLIQUE_SIZE 8

typedef struct
{
  void const*   inputs[MAX_CLIQUE_SIZE];
  void*         outputs[MAX_CLIQUE_SIZE];
  int*          barrierCounter;
} cliqueDevicePtrs_t;

// Helper macro to generate a table of templated kernel functions
// This is expected to run between MIN_CLIQUE_SIZE to MAX_CLIQUE_SIZE
#define KERNEL_LIST_RANK(kernelname, datatype, func) \
  {                                              \
    kernelname<datatype, func<datatype>, 2>,     \
    kernelname<datatype, func<datatype>, 3>,     \
    kernelname<datatype, func<datatype>, 4>,     \
    kernelname<datatype, func<datatype>, 5>,     \
    kernelname<datatype, func<datatype>, 6>,     \
    kernelname<datatype, func<datatype>, 7>,     \
    kernelname<datatype, func<datatype>, 8>      \
  }

// Helper macro to generate a table of templated kernel functions
// This is expected to match the number of supported reduction operations (ncclNumOps)
#define KERNEL_LIST_OP(kernelname, datatype)          \
  {                                                   \
    KERNEL_LIST_RANK(kernelname, datatype, FuncSum),  \
    KERNEL_LIST_RANK(kernelname, datatype, FuncProd), \
    KERNEL_LIST_RANK(kernelname, datatype, FuncMax),  \
    KERNEL_LIST_RANK(kernelname, datatype, FuncMin)   \
  }

// Helper Macro to generate table of templated kernel functions
// This is expected to match the number of supported datatypes (ncclNumTypes)
#define KERNEL_LIST_MACRO(kernelname)         \
  {                                           \
    KERNEL_LIST_OP(kernelname, int8_t),       \
    KERNEL_LIST_OP(kernelname, uint8_t),      \
    KERNEL_LIST_OP(kernelname, int32_t),      \
    KERNEL_LIST_OP(kernelname, uint32_t),     \
    KERNEL_LIST_OP(kernelname, int64_t),      \
    KERNEL_LIST_OP(kernelname, uint64_t),     \
    KERNEL_LIST_OP(kernelname, half),         \
    KERNEL_LIST_OP(kernelname, float),        \
    KERNEL_LIST_OP(kernelname, double),       \
    KERNEL_LIST_OP(kernelname, rccl_bfloat16) \
  }

template <int NUM_RANKS>
__forceinline__ __device__ void WaitForBarrier(int* counter)
{
  if (threadIdx.x == 0 & blockIdx.x == 0)
  {
    // Assumes counter starts at 0 prior to any rank access
    __atomic_add_fetch(counter, 1, __ATOMIC_SEQ_CST);

    // Wait for all ranks to reach barrier
    while (__atomic_load_n(counter, __ATOMIC_SEQ_CST) < NUM_RANKS);

    // Each rank increments again, last one resets barrier
    if (__atomic_add_fetch(counter, 1, __ATOMIC_SEQ_CST) == (2*NUM_RANKS))
      __atomic_store_n(counter, 0, __ATOMIC_SEQ_CST);

    // Wait for counter to be zeroed
    while (__atomic_load_n(counter, __ATOMIC_SEQ_CST) != 0);
  }
}

__forceinline__ __host__ __device__ int RoundUp(int X, int Y)
{
  return (X+Y-1)/Y * Y;
}

#endif
