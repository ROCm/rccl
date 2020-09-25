/*
Copyright (c) 2019-2020 Advanced Micro Devices, Inc. All rights reserved.

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

#include <vector>
#include <sstream>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <set>
#include <unistd.h>
#include <map>
#include <iostream>
#include <sstream>
#include <hip/hip_runtime.h>
#include <hip/hip_ext.h>
#include <hsa/hsa_ext_amd.h>
#include "copy_kernel.h"

// Helper macro for catching HIP errors
#define HIP_CALL(cmd)                                                   \
    do {                                                                \
        hipError_t error = (cmd);                                       \
        if (error != hipSuccess)                                        \
        {                                                               \
            std::cerr << "Encountered HIP error (" << hipGetErrorString(error) << ") at line " \
                      << __LINE__ << " in file " << __FILE__ << "\n";   \
            exit(-1);                                                   \
        }                                                               \
    } while (0)

// Different src/dst memory types supported
typedef enum
{
  MEM_CPU = 0,             // Pinned CPU memory
  MEM_GPU = 1              // Global GPU memory
} MemType;

char const MemTypeStr[3] = "CG";

typedef enum
{
  MODE_FILL  = 0,         // Fill data with pattern
  MODE_CHECK = 1          // Check data against pattern
} ModeType;

// Each Link is a uni-direction operation from a src memory to dst memory executed by a specific GPU
struct Link
{
  int     exeIndex;        // GPU to execute on
  MemType srcMemType;      // Source memory type
  int     srcIndex;        // Source device index
  MemType dstMemType;      // Destination memory type
  int     dstIndex;        // Destination device index
  int     numBlocksToUse;  // Number of threadblocks to use for this Link
};

// Each threadblock copies N floats from src to dst
struct BlockParam
{
    int N;
    float* src;
    float* dst;
};

void DisplayUsage(char const* cmdName);                // Display usage instructions
void DisplayTopology();                                // Display GPU topology
void ParseLinks(char* line, std::vector<Link>& links); // Parse Link information
void AllocateMemory(MemType memType, int devIndex, size_t numBytes, bool useFineGrainMem, float** memPtr);
void DeallocateMemory(MemType memType, int devIndex, float* memPtr);
void CheckOrFill(ModeType mode, int N, bool isMemset, bool isHipCall, float* ptr);

#define MAX_NAME_LEN 64
#define BLOCKSIZE 256
#define COPY_UNROLL 4
#define MEMSET_UNROLL 4

// GPU copy kernel
__global__ void __launch_bounds__(BLOCKSIZE)
CopyKernel(BlockParam* blockParams)
{
    // Collect the arguments for this block
    int N = blockParams[blockIdx.x].N;
    const float* __restrict__ src = (float* )blockParams[blockIdx.x].src;
    float* __restrict__ dst = (float* )blockParams[blockIdx.x].dst;

    Copy<COPY_UNROLL, BLOCKSIZE>(dst, src, N);
}

// GPU set kernel
__global__ void __launch_bounds__(BLOCKSIZE)
MemsetKernel(BlockParam* blockParams)
{
    // Collect the arguments for this block
    int N = blockParams[blockIdx.x].N;
    float* __restrict__ dst = (float*)blockParams[blockIdx.x].dst;

    // Use non-zero value
    #pragma unroll MEMSET_UNROLL
    for (int tid = threadIdx.x; tid < N; tid += BLOCKSIZE)
    {
      dst[tid] = 1234.0;
    }
}
