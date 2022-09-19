/*
Copyright (c) 2020-2021 Advanced Micro Devices, Inc. All rights reserved.

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

#define MIN_CLIQUE_SIZE 2
#define MAX_CLIQUE_SIZE 8

typedef struct
{
  int* globalCount;   // Shared across GPUs
  int* globalSense;   // Shared across GPUs
  int* localSense;    // Local to this GPU
} gpuBarrier_t;

typedef struct
{
  // Input/output pointers from participating ranks
  void const* inputs[MAX_CLIQUE_SIZE];
  void*       outputs[MAX_CLIQUE_SIZE];

  // Barrier variable
  gpuBarrier_t barrier;
} cliqueDevicePtrs_t;

// Helper macro to launch an appropriate kernel by converting rank to a template argument
#define LAUNCH_CLIQUE_KERNEL(kernelname, FUNC, T, args)  \
  {                                                      \
    switch (args->comm->nRanks){                         \
    case 2: kernelname<FUNC, T, 2>(args); break;         \
    case 3: kernelname<FUNC, T, 3>(args); break;         \
    case 4: kernelname<FUNC, T, 4>(args); break;         \
    case 5: kernelname<FUNC, T, 5>(args); break;         \
    case 6: kernelname<FUNC, T, 6>(args); break;         \
    case 7: kernelname<FUNC, T, 7>(args); break;         \
    case 8: kernelname<FUNC, T, 8>(args); break;         \
    }                                                    \
  }

// Multi-GPU (on same node) barrier.  One thread per grid per GPU updates barrier / waits
template <int NUM_RANKS>
__forceinline__ __device__ void WaitForBarrier(gpuBarrier_t const& barrier)
{
  if (threadIdx.x == 0)
  {
    // Sense inversion barrier
    *barrier.localSense = 1 - *barrier.localSense;
    int localSense = *barrier.localSense;

    int val = __atomic_add_fetch(barrier.globalCount, 1, __ATOMIC_ACQ_REL);
    if (val == NUM_RANKS)
    {
      // Last arrival resets barrier
      __atomic_store_n(barrier.globalCount, 0, __ATOMIC_RELEASE);
      __atomic_store_n(barrier.globalSense, localSense, __ATOMIC_RELEASE);
    }
    else
    {
      // Wait for all ranks to reach barrier
      while (__atomic_load_n(barrier.globalSense, __ATOMIC_ACQUIRE) != localSense);
    }
  }
}

__forceinline__ __host__ __device__ size_t RoundUp(size_t X, size_t Y)
{
  return (X+Y-1)/Y * Y;
}

#endif
