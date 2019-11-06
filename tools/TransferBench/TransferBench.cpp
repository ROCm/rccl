/*
Copyright (c) 2019 - present Advanced Micro Devices, Inc. All rights reserved.

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

// This program measures simultaneous copy performance across multiple GPUs
// on the same node

#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <set>

#include <hip/hip_runtime.h>
#include "copy_kernel.h"
#include "TransferBench.hpp"

int main(int argc, char **argv)
{
  // Display usage
  if (argc <= 1)
  {
    printf("Usage: %s configFile <N>\n", argv[0]);
    printf("- configFile: file describing topologies to test\n");
    printf("  Each line should contain a single topology\n");
    printf("    L - number of links followed by L white-space separated triples (src, dst, # blocks)\n");
    printf("    For example:\n");
    printf("      2 0 1 3  1 0 3\n");
    printf("      would define 2 links each using 3 threadblocks from GPU0 -> GPU1, and GPU1->GPU0\n");
    printf("- N: (Optional) Number of bytes to transfer per link.\n");
    printf("     If not specified, defaults to 2^28=256MB. Must be a multiple of 128 bytes\n");
    printf("\n");
    printf("Environment variables:\n");
    printf("======================\n");
    printf(" USE_HIP_CALL    - Use hip calls (hipMemcpyAsync/hipMemset) instead of kernel\n");
    printf(" USE_MEMSET      - Write constant value (instead of doing a copy)\n");
    printf(" USE_COARSE_MEM  - Use coarse-grained dst GPU memory (instead of fine-grained)\n");
    printf(" USE_SINGLE_SYNC - Only synchronize once at end of iterations (disables GPU times)\n");
    printf(" USE_INTERACTIVE - Waits for user-input prior to start and after transfer loop (for profiling)\n");
    printf(" USE_ITERATIONS=N - Sets number of iterations to run (default is 10)\n");
    exit(0);
  }

  // Parse number of bytes to use (or use default if not specified)
  size_t const numBytesPerLink = argc > 2 ? atoll(argv[2]) : (1<<28);
  size_t N = numBytesPerLink / sizeof(float);
  if (numBytesPerLink % 128)
  {
    printf("[ERROR] numBytesPerLink (%lu) must be a multiple of 128\n", numBytesPerLink);
    exit(1);
  }
  printf("Operating on %zu bytes per link (%zu floats)\n", numBytesPerLink, N);

  // Collect environment variables / display current run configuration
  bool useHipCall = getenv("USE_HIP_CALL");
  bool useMemset = getenv("USE_MEMSET");
  bool useCoarseMem = getenv("USE_COARSE_MEM");
  bool useSingleSync = getenv("USE_SINGLE_SYNC");
  bool useInteractive = getenv("USE_INTERACTIVE");

  int numWarmups = 3;
  int numIterations = getenv("USE_ITERATIONS") ? atoi(getenv("USE_ITERATIONS")) : 10;

  printf("Running %s%s tests (control using USE_HIP_CALL/USE_MEMSET)\n",
         useHipCall ? "hipMem" : "mem",
         useMemset ? "set" : "cpy");
  printf("Destination memory: %s-grained (control using USE_COARSE_MEM)\n",
         useCoarseMem ? "coarse" : "fine");
  if (useHipCall && !useMemset)
  {
    if (getenv("HSA_ENABLE_SDMA") && !strcmp(getenv("HSA_ENABLE_SDMA"), "0"))
      printf("Using blit kernels for hipMemcpy. (HSA_ENABLE_SDMA=0)\n");
    else
      printf("Using DMA copy engines (disable by setting HSA_ENABLE_SDMA=0)\n");
  }
  if (useSingleSync)
    printf("Synchronizing only once, after all iterations (disables GPU timers)\n");
  else
    printf("Synchronizing per iteration  (disable via USE_SINGLE_SYNC)\n");

  if (useInteractive)
    printf("Running in interactive mode (USE_INTERACTIVE)\n");
  else
    printf("Running in non-interactive mode (enable interactive mode via USE_INTERACTIVE)\n");

  printf("Executing %d warmup iteration(s), and %d timed iteration(s) (Set via USE_ITERATION=#)\n",
         numWarmups, numIterations);

  // Currently an environment variable is required in order to enable fine-grained VRAM allocations
  if (!useCoarseMem && !getenv("HSA_FORCE_FINE_GRAIN_PCIE"))
  {
    printf("[ERROR] Currently you must set HSA_FORCE_FINE_GRAIN_PCIE=1 prior to execution\n");
    exit(1);
  }

  // Collect the number of available GPUs on this machine
  int numDevices;
  HIP_CALL(hipGetDeviceCount(&numDevices));
  if (numDevices < 1)
  {
    printf("[ERROR] No GPU devices found\n");
    exit(1);
  }

  // Print header
  printf("%*s", MAX_NAME_LEN, "");
  printf("%*s | ", 8*(numDevices+1), "Bandwidth (GB/s)");
  printf("%*s", 8*(numDevices+1), "Duration (msec)");
  printf(" | Overhead\n");
  printf("%-*s", MAX_NAME_LEN, "Configuration");
  for (int i = 0; i < numDevices; i++)
    printf("  GPU %02d", i);
  printf("   Total");
  printf(" | ");
  for (int i = 0; i < numDevices; i++)
    printf("  GPU %02d", i);
  printf(" CpuTime");
  printf(" |   (msec)\n");

  for (int i = 0; i < MAX_NAME_LEN + (8 * (numDevices + 1)); i++) printf("=");
  printf("=|=");
  for (int i = 0; i < (8 * (numDevices + 1)); i++) printf("=");
  printf("=|=========\n");

  // Read configuration file
  FILE* fp = fopen(argv[1], "r");
  if (!fp)
  {
    printf("[ERROR] Unable to open link configuration file: [%s]\n", argv[1]);
    exit(1);
  }

  // Track links that get used
  std::map<std::pair<int, int>, int> linkMap;

  char line[2048];
  while(fgets(line, 2048, fp))
  {
    // Parse links from configuration file
    std::vector<Link> links;
    ParseLinks(line, links);

    int const numLinks = links.size();
    if (numLinks == 0) continue;

    // Clear counters
    int linkCount[numDevices];
    for (int i = 0; i < numDevices; i++)
      linkCount[i] = 0;

    float* linkSrcMem[numLinks];
    float* linkDstMem[numLinks];
    hipStream_t streams[numLinks];
    hipEvent_t startEvents[numLinks];
    hipEvent_t stopEvents[numLinks];
    std::vector<BlockParam> cpuBlockParams[numLinks];
    BlockParam* gpuBlockParams[numLinks];

    char name[MAX_NAME_LEN+1] = {};

    for (int i = 0; i < numLinks; i++)
    {
      int const src = links[i].srcGpu;
      int const dst = links[i].dstGpu;
      if (src < 0 || src >= numDevices ||
          dst < 0 || dst >= numDevices)
      {
        printf("[ERROR] Invalid link (%d to %d). Total devices: %d\n", src, dst, numDevices);
        exit(1);
      }
      snprintf(name + strlen(name), MAX_NAME_LEN, "%d->%d:%d ", src, dst, links[i].numBlocksToUse);

      // Enable peer-to-peer access if this is the first time seeing this pair
      auto linkPair = std::make_pair(src, dst);
      linkMap[linkPair]++;
      if (linkMap[linkPair] == 1)
      {
        int canAccess;
        HIP_CALL(hipDeviceCanAccessPeer(&canAccess, src, dst));
        if (!canAccess)
        {
          printf("[ERROR] Unable to enable peer access between device %d and %d\n", src, dst);
          exit(1);
        }
        HIP_CALL(hipSetDevice(src));
        HIP_CALL(hipDeviceEnablePeerAccess(dst, 0));
      }

      // Count # of links / total blocks each GPU will be working on
      linkCount[src]++;

      // Allocate GPU memory on source GPU / streams / events
      HIP_CALL(hipSetDevice(links[i].srcGpu));
      HIP_CALL(hipStreamCreate(&streams[i]));
      HIP_CALL(hipEventCreate(&startEvents[i]));
      HIP_CALL(hipEventCreate(&stopEvents[i]));
      HIP_CALL(hipMalloc((void **)&linkSrcMem[i], numBytesPerLink));
      HIP_CALL(hipMalloc((void**)&gpuBlockParams[i], sizeof(BlockParam) * numLinks));
      CheckOrFill(N, linkSrcMem[i], false, useMemset, useHipCall);

      // Allocate GPU memory on destination GPU
      HIP_CALL(hipSetDevice(links[i].dstGpu));
      if (useCoarseMem)
        HIP_CALL(hipMalloc((void**)&linkDstMem[i], numBytesPerLink));
      else
        HIP_CALL(hipExtMallocWithFlags((void**)&linkDstMem[i], numBytesPerLink, hipDeviceMallocFinegrained));

      // Each block needs to know src/dst pointers and how many elements to transfer
      // Figure out the sub-array each block does for this link
      // NOTE: Have each sub-array to work on multiple of 32-floats (128-bytes),
      //       but divide as evenly as possible
      // NOTE: N is always a multiple of 32
      int blocksWithExtra = (N / 32) % links[i].numBlocksToUse;
      int perBlockBaseN   = (N / 32) / links[i].numBlocksToUse * 32;
      for (int j = 0; j < links[i].numBlocksToUse; j++)
      {
        BlockParam param;
        param.N   = perBlockBaseN + ((j < blocksWithExtra) ? 32 : 0);
        param.src = linkSrcMem[i] + ((j * perBlockBaseN) + ((j < blocksWithExtra) ?
                                                            j : blocksWithExtra) * 32);
        param.dst = linkDstMem[i] + ((j * perBlockBaseN) + ((j < blocksWithExtra) ?
                                                            j : blocksWithExtra) * 32);
        cpuBlockParams[i].push_back(param);
      }

      HIP_CALL(hipMemcpy(gpuBlockParams[i], cpuBlockParams[i].data(),
                         sizeof(BlockParam) * links[i].numBlocksToUse, hipMemcpyHostToDevice));
    }

    // Launch kernels (warmup iterations are not counted)
    double totalCpuTime = 0;
    double totalGpuTime[numDevices];
    for (int i = 0; i < numDevices; i++) totalGpuTime[i] = 0.0;

    for (int iteration = -numWarmups; iteration < numIterations; iteration++)
    {
      if (useInteractive && iteration == 0)
      {
        printf("Hit <Enter> to continue: ");
        scanf("%*c");
        printf("\n");
      }

      auto cpuStart = std::chrono::high_resolution_clock::now();

      #pragma omp parallel for num_threads(numLinks)
      for (int i = 0; i < numLinks; i++)
      {
        HIP_CALL(hipSetDevice(links[i].srcGpu));

        HIP_CALL(hipEventRecord(startEvents[i], streams[i]));

        if (useHipCall)
        {
          if (useMemset)
          {
            HIP_CALL(hipMemsetAsync(linkDstMem[i], 42, numBytesPerLink, streams[i]));
          }
          else
          {
            HIP_CALL(hipMemcpyAsync(linkDstMem[i], linkSrcMem[i],
                                    numBytesPerLink, hipMemcpyDeviceToDevice,
                                    streams[i]));
          }
        }
        else
        {
          if (useMemset)
          {
            hipLaunchKernelGGL(MemsetKernel,
                               dim3(links[i].numBlocksToUse, 1, 1),
                               dim3(BLOCKSIZE, 1, 1),
                               0,
                               streams[i],
                               gpuBlockParams[i]);
          }
          else
          {
            hipLaunchKernelGGL(CopyKernel,
                               dim3(links[i].numBlocksToUse, 1, 1),
                               dim3(BLOCKSIZE, 1, 1),
                               0,
                               streams[i],
                               gpuBlockParams[i]);
          }
        }
        HIP_CALL(hipEventRecord(stopEvents[i], streams[i]));
      }

      // Synchronize per iteration, unless in single sync mode, in which case
      // synchronize during last warmup / last actual iteration
      if (!useSingleSync || iteration == -1  || iteration == numIterations - 1)
      {
        for (int i = 0; i < numLinks; i++)
          hipStreamSynchronize(streams[i]);
      }

      auto cpuDelta = std::chrono::high_resolution_clock::now() - cpuStart;
      double deltaSec = std::chrono::duration_cast<std::chrono::duration<double>>(cpuDelta).count();

      if (iteration >= 0)
      {
        totalCpuTime += deltaSec;

        for (int i = 0; i < numDevices; i++)
        {
          // Collect GPU information only if this is the last iteration for single sync mode
          if (useSingleSync && iteration != numIterations - 1)
          {
            totalGpuTime[i] = 0.00;
          }
          else
          {
            // Multiple links running on the same device may be running simultaneously
            // so try to figure out the first/last event across all links
            float maxTime = 0.0f;
            for (int j = 0; j < numLinks; j++)
            {
              if (links[j].srcGpu != i) continue;
              for (int k = 0; k < numLinks; k++)
              {
                if (links[k].srcGpu != i) continue;

                float gpuDeltaMsec;
                HIP_CALL(hipEventElapsedTime(&gpuDeltaMsec, startEvents[j], stopEvents[k]));
                maxTime = std::max(maxTime, gpuDeltaMsec);
              }
            }
            totalGpuTime[i] += maxTime / 1000.0;
          }
        }
      }
    }

    if (useInteractive)
    {
      printf("Transfers complete. Hit <Enter> to continue: ");
      scanf("%*c");
      printf("\n");
    }

    // Validate that each link has transferred correctly
    for (int i = 0; i < numLinks; i++)
      CheckOrFill(N, linkDstMem[i], true, useMemset, useHipCall);

    // Report timings
    printf("%-*s", MAX_NAME_LEN, name);
    for (int i = 0; i < numDevices; i++)
    {
      if (linkCount[i] == 0)
      {
        printf("%8.3f", 0.0f);
      }
      else
      {
        if (!useSingleSync)
          totalGpuTime[i] /= (1.0 * numIterations);
        printf("%8.3f", (linkCount[i] * numBytesPerLink / 1.0E9) / totalGpuTime[i]);
      }
    }
    // Print off bandwidth (based on CPU wall-time timer)
    totalCpuTime /= numIterations;
    printf("%8.3f", (numLinks * numBytesPerLink / 1.0E9) / totalCpuTime);
    printf(" | ");

    double maxGpuTime = 0;
    for (int i = 0; i < numDevices; i++)
    {
      if (linkCount[i] == 0)
      {
        printf("%8.3f", 0.0f);
      }
      else
      {
        printf("%8.3f", totalGpuTime[i] * 1000.0f);
        maxGpuTime = std::max(maxGpuTime, totalGpuTime[i]);
      }
    }
    printf("%8.3f | %8.3f\n", totalCpuTime * 1000.0f, (totalCpuTime - maxGpuTime) * 1000.0f);

    // Release GPU memory
    for (int i = 0; i < numLinks; i++)
    {
      HIP_CALL(hipFree(linkSrcMem[i]));
      HIP_CALL(hipFree(linkDstMem[i]));
      HIP_CALL(hipFree(gpuBlockParams[i]));
      HIP_CALL(hipStreamDestroy(streams[i]));
      HIP_CALL(hipEventDestroy(startEvents[i]));
      HIP_CALL(hipEventDestroy(stopEvents[i]));

    }
  }
  fclose(fp);

  // Print link information
  for (int i = 0; i < MAX_NAME_LEN + (8 * (numDevices + 1)); i++) printf("=");
  printf("=|=");
  for (int i = 0; i < (8 * (numDevices + 1)); i++) printf("=");
  printf("=|=========\n");
  printf("Link topology:\n");
  uint32_t linkType;
  uint32_t hopCount;
  for (auto mapPair : linkMap)
  {
    int src = mapPair.first.first;
    int dst = mapPair.first.second;
    HIP_CALL(hipExtGetLinkTypeAndHopCount(src, dst, &linkType, &hopCount));
    printf("%d -> %d: %s [%d hop(s)]\n", src, dst,
           linkType == HSA_AMD_LINK_INFO_TYPE_HYPERTRANSPORT ? "HYPERTRANSPORT" :
           linkType == HSA_AMD_LINK_INFO_TYPE_QPI            ? "QPI" :
           linkType == HSA_AMD_LINK_INFO_TYPE_PCIE           ? "PCIE" :
           linkType == HSA_AMD_LINK_INFO_TYPE_INFINBAND      ? "INFINIBAND" :
           linkType == HSA_AMD_LINK_INFO_TYPE_XGMI           ? "XGMI" : "UNKNOWN",
           hopCount);
  }
  return 0;
}
