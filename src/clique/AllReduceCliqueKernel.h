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

#ifndef ALLREDUCECLIQUEKERNEL_H
#define ALLREDUCECLIQUEKERNEL_H

#include "CliqueCommon.h"
#include "common_kernel.h"
#include <hip/hip_runtime.h>
#include "hip/hip_ext.h"

#define ALL_REDUCE_SPLIT_BLOCKSIZE 256

template <typename T, class FUNC, int NUM_RANKS>
__global__ __launch_bounds__(ALL_REDUCE_SPLIT_BLOCKSIZE)
void AllReduceCliqueSplitKernel(int N,
                                size_t startIdx,
                                cliqueDevicePtrs_t cliquePtrs)
{
  if (N > 0)
  {
    // Each workgroup operates on a contiguous portion of memory
    // Divide the # of elements evenly across workgroups, then round up to multiple of blocksize
    int baseSize         = (N + gridDim.x - 1) / gridDim.x;
    int chunkSize        = RoundUp(baseSize, ALL_REDUCE_SPLIT_BLOCKSIZE);
    int blockOffsetStart = min(blockIdx.x * chunkSize, N);
    int blockOffsetStop  = min(blockOffsetStart + chunkSize, N);
    int blockN           = blockOffsetStop - blockOffsetStart;

    if (blockN > 0)
    {
      T const** inputs  = (T const**)cliquePtrs.inputs;
      T**       outputs = (T      **)cliquePtrs.outputs;

      T const* srcs[NUM_RANKS];
      T*       dsts[NUM_RANKS];

      #pragma unroll
      for (int r = 0; r < NUM_RANKS; r++)
      {
        srcs[r] = inputs[r]  + startIdx + blockOffsetStart;
        dsts[r] = outputs[r] + startIdx + blockOffsetStart;
      }

      #define ALL_REDUCE_CLIQUE_UNROLL 4
      ReduceOrCopyMulti<ALL_REDUCE_CLIQUE_UNROLL, FUNC, T, NUM_RANKS, NUM_RANKS, NUM_RANKS, NUM_RANKS>(
        threadIdx.x, ALL_REDUCE_SPLIT_BLOCKSIZE, NUM_RANKS, srcs, NUM_RANKS, dsts, blockN);
    }
  }

  // Each GPU works on a separate subsection, however we cannot finish the kernel
  // until all GPUs have finished otherwise part of the result may not be correct yet
  WaitForBarrier<NUM_RANKS>(cliquePtrs.barrierCounter);
}

class AllReduceCliqueKernel
{
public:
  static ncclResult_t Launch(int                const rank,
                             int                const numRanks,
                             int                const maxGridSize,
                             size_t             const count,
                             ncclDataType_t     const datatype,
                             ncclRedOp_t        const op,
                             hipStream_t        const stream,
                             cliqueDevicePtrs_t const& cliquePtrs,
                             bool               const doTiming = false)
  {
    if (numRanks < MIN_CLIQUE_SIZE || numRanks > MAX_CLIQUE_SIZE)
    {
      WARN("Number of ranks exceeds supported.  Expected %d <= %d < %d for numRanks",
           MIN_CLIQUE_SIZE, numRanks, MAX_CLIQUE_SIZE);
      return ncclInvalidUsage;
    }

    // Divide the # of elements done per GPU evenly across ranks, then round up to blocksize
    int baseSize  = (count + numRanks - 1) / numRanks;
    int chunkSize = RoundUp(baseSize, ALL_REDUCE_SPLIT_BLOCKSIZE);
    int startIdx  = min(chunkSize * rank, count);
    int stopIdx   = min(startIdx + chunkSize, count);
    int rankN     = max(stopIdx - startIdx, 0);

    // Adjust gridsize if there isn't enough work to prevent empty workgroups
    int realGridSize = std::max(std::min(maxGridSize,
                                         (rankN + ALL_REDUCE_SPLIT_BLOCKSIZE - 1) / ALL_REDUCE_SPLIT_BLOCKSIZE),
                                1);

    hipEvent_t startEvent = 0;
    hipEvent_t stopEvent = 0;
    float kernelTimeMs;
    if (doTiming)
    {
      hipEventCreate(&startEvent);
      hipEventCreate(&stopEvent);
      hipEventRecord(startEvent, stream);
    }

    // Launch even if empty for this rank, because all ranks must hit sync barrier
    hipLaunchKernelGGL(m_allReduceCliqueKernels[datatype][op][numRanks - MIN_CLIQUE_SIZE],
                       dim3(realGridSize, 1, 1),
                       dim3(ALL_REDUCE_SPLIT_BLOCKSIZE, 1, 1),
                       0, stream,
                       rankN, startIdx, cliquePtrs);

    if (doTiming)
    {
      hipEventRecord(stopEvent, stream);
      hipEventSynchronize(stopEvent);
      hipEventElapsedTime(&kernelTimeMs, startEvent, stopEvent);
      printf("[%d/%d:%d] %lu %13.6f ms\n", rank, numRanks, maxGridSize, count, kernelTimeMs);
      hipEventDestroy(startEvent);
      hipEventDestroy(stopEvent);
    }

    return ncclSuccess;
  }

protected:
  // List of all templated device kernels function pointers
  typedef void(*allReduceCliqueFunc_t)(int, size_t, cliqueDevicePtrs_t);
  static constexpr allReduceCliqueFunc_t
    m_allReduceCliqueKernels[ncclNumTypes][ncclNumOps][MAX_CLIQUE_SIZE - MIN_CLIQUE_SIZE + 1] =
    KERNEL_LIST_MACRO(AllReduceCliqueSplitKernel);
};

#endif
