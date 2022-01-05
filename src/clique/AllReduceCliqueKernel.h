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

#ifndef ALLREDUCECLIQUEKERNEL_H
#define ALLREDUCECLIQUEKERNEL_H

#include "CliqueCommon.h"
#include "devcomm.h"
#include "reduce_kernel.h"
#include "common_kernel.h"

template <class FUNC, typename T, int NUM_RANKS>
__device__ void AllReduceCliqueSplitKernel(struct ncclWorkElem* args)
{
  // Clique-specific kernel arguments
  cliqueDevicePtrs_t* cliquePtrs = args->clique.ptrs;      // Collection of all input/output pointers across ranks in clique
  size_t const N                 = args->clique.count;     // Total number of elements to reduce
  int    const nBlocks           = args->clique.nChannels; // Total number of blocks assigned to this kernel (may be different than gridDim.x)
  int    const blockId           = args->clique.bid;       // 0-indexed blockIdx for this threadblock (may be different than blockIdx.x)
  int    const rank              = args->comm->rank;       // Current rank

  // Each threadblock works independently of others on a subsection of the input
  // First split evently across ranks, while maintaining multiples of blocksize
  size_t const perRankN       = RoundUp((N + NUM_RANKS - 1) / NUM_RANKS, blockDim.x);
  size_t const perBlockN      = RoundUp((perRankN + nBlocks - 1) / nBlocks, blockDim.x);
  size_t const currBlockStart = min((rank * nBlocks + blockId) * perBlockN, N);
  size_t const currBlockStop  = min(currBlockStart + perBlockN, N);
  size_t const blockN         = currBlockStop - currBlockStart;

  if (blockN > 0)
  {
    // Prepare input / output subarrays
    T const** inputs  = (T const**)cliquePtrs->inputs;
    T**       outputs = (T      **)cliquePtrs->outputs;
    T const* srcs[NUM_RANKS];
    T*       dsts[NUM_RANKS];

    #pragma unroll
    for (int r = 0; r < NUM_RANKS; r++)
    {
      srcs[r] = inputs[r]  + currBlockStart;
      dsts[r] = outputs[r] + currBlockStart;
    }

    // Perform the reduction
    #define ALL_REDUCE_CLIQUE_UNROLL 1
    ReduceOrCopyMulti<ALL_REDUCE_CLIQUE_UNROLL, FUNC, T, NUM_RANKS, NUM_RANKS, NUM_RANKS, NUM_RANKS, 0>(
      threadIdx.x, blockDim.x, nullptr, false, NUM_RANKS, srcs, NUM_RANKS, dsts, blockN);
  }

  // Even if there was nothing for this GPU to do, it must participate in a barrier
  // because other GPUs may be modifying this GPUs output buffer still
  if (blockId == 0) WaitForBarrier<NUM_RANKS>(cliquePtrs->barrier);
}

#endif
