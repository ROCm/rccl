/*
Copyright (c) 2022 Advanced Micro Devices, Inc. All rights reserved.

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

#pragma once

#define WARP_SIZE 64
#define BLOCKSIZE 256

// GPU copy kernel
__global__ void __launch_bounds__(BLOCKSIZE)
GpuCopyKernel(BlockParam* blockParams)
{
  #define PackedFloat_t float4
  #define FLOATS_PER_PACK (sizeof(PackedFloat_t) / sizeof(float))

  // Collect the arguments for this threadblock
  int          Nrem = blockParams[blockIdx.x].N;
  float const* src  = blockParams[blockIdx.x].src;
  float*       dst  = blockParams[blockIdx.x].dst;
  if (threadIdx.x == 0) blockParams[blockIdx.x].startCycle = __builtin_amdgcn_s_memrealtime();

  // Operate on wavefront granularity
  int numWaves = BLOCKSIZE   / WARP_SIZE; // Number of wavefronts per threadblock
  int waveId   = threadIdx.x / WARP_SIZE; // Wavefront number
  int threadId = threadIdx.x % WARP_SIZE; // Thread index within wavefront

  #define LOOP1_UNROLL 8
  // 1st loop - each wavefront operates on LOOP1_UNROLL x FLOATS_PER_PACK per thread per iteration
  // Determine the number of packed floats processed by the first loop
  int const loop1Npack  = (Nrem / (FLOATS_PER_PACK * LOOP1_UNROLL * WARP_SIZE)) * (LOOP1_UNROLL * WARP_SIZE);
  int const loop1Nelem  = loop1Npack * FLOATS_PER_PACK;
  int const loop1Inc    = BLOCKSIZE * LOOP1_UNROLL;
  int       loop1Offset = waveId * LOOP1_UNROLL * WARP_SIZE + threadId;

  PackedFloat_t const* packedSrc = (PackedFloat_t const*)(src) + loop1Offset;
  PackedFloat_t*       packedDst = (PackedFloat_t      *)(dst) + loop1Offset;
  while (loop1Offset < loop1Npack)
  {
    PackedFloat_t vals[LOOP1_UNROLL];
    #pragma unroll
    for (int u = 0; u < LOOP1_UNROLL; ++u)
      vals[u] = *(packedSrc + u * WARP_SIZE);

    #pragma unroll
    for (int u = 0; u < LOOP1_UNROLL; ++u)
      *(packedDst + u * WARP_SIZE) = vals[u];

    packedSrc   += loop1Inc;
    packedDst   += loop1Inc;
    loop1Offset += loop1Inc;
  }
  Nrem -= loop1Nelem;
  if (Nrem > 0)
  {
    // 2nd loop - Each thread operates on FLOATS_PER_PACK per iteration
    int const loop2Npack  = Nrem / FLOATS_PER_PACK;
    int const loop2Nelem  = loop2Npack * FLOATS_PER_PACK;
    int const loop2Inc    = BLOCKSIZE;
    int       loop2Offset = threadIdx.x;

    packedSrc = (PackedFloat_t const*)(src + loop1Nelem);
    packedDst = (PackedFloat_t      *)(dst + loop1Nelem);
    while (loop2Offset < loop2Npack)
    {
      packedDst[loop2Offset] = packedSrc[loop2Offset];
      loop2Offset += loop2Inc;
    }
    Nrem -= loop2Nelem;

    // Deal with leftovers less than FLOATS_PER_PACK)
    if (threadIdx.x < Nrem)
    {
      int offset = loop1Nelem + loop2Nelem + threadIdx.x;
      dst[offset] = src[offset];
    }
  }

  __threadfence_system();
  if (threadIdx.x == 0)
    blockParams[blockIdx.x].stopCycle = __builtin_amdgcn_s_memrealtime();
}

#define MEMSET_UNROLL 8
__global__ void __launch_bounds__(BLOCKSIZE)
GpuMemsetKernel(BlockParam* blockParams)
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

// CPU copy kernel
void CpuCopyKernel(BlockParam const& blockParams)
{
  memcpy(blockParams.dst, blockParams.src, blockParams.N * sizeof(float));
}

// CPU memset kernel
void CpuMemsetKernel(BlockParam const& blockParams)
{
  for (int i = 0; i < blockParams.N; i++)
    blockParams.dst[i] = 1234.0;
}
