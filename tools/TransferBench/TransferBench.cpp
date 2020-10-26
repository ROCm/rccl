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

// This program measures simultaneous copy performance across multiple GPUs
// on the same node

#include "TransferBench.hpp"

// Simple configuration parameters
size_t const DEFAULT_BYTES_PER_LINK = (1<<26);
int    const DEFAULT_NUM_WARMUPS    = 3;
int    const DEFAULT_NUM_ITERATIONS = 10;

int main(int argc, char **argv)
{
  // Display usage
  if (argc <= 1)
  {
    DisplayUsage(argv[0]);
    DisplayTopology();
    exit(0);
  }

  // If a negative value is listed for N, generate a comprehensive config file for this node
  if (argc > 2 && atoi(argv[2]) < 0)
  {
    GenerateConfigFile(argv[1], -1*atoi(argv[2]));
    exit(0);
  }

  // Collect environment variables / display current run configuration
  bool useHipCall      = getenv("USE_HIP_CALL");       // Use hipMemcpy/hipMemset instead of custom shader kernels
  bool useMemset       = getenv("USE_MEMSET");         // Perform a memset instead of a copy (ignores source memory)
  bool useFineGrainMem = getenv("USE_FINEGRAIN_MEM");  // Allocate fine-grained GPU memory instead of coarse-grained GPU memory
  bool useSingleSync   = getenv("USE_SINGLE_SYNC");    // Perform synchronization only once after all iterations instead of per iteration
  bool useInteractive  = getenv("USE_INTERACTIVE");    // Pause for user-input before starting transfer loop
  bool useSleep        = getenv("USE_SLEEP");          // Adds a 100ms sleep after each synchronization
  bool reuseStreams    = getenv("REUSE_STREAMS");      // Re-use streams instead of creating / destroying per test
  bool showAddr        = getenv("SHOW_ADDR");          // Print out memory addresses for each Link
  bool outputToCsv     = getenv("OUTPUT_TO_CSV");      // Output in CSV format
  int  byteOffset      = getenv("BYTE_OFFSET") ? atoi(getenv("BYTE_OFFSET")) : 0; // Byte-offset for memory allocations
  int  numWarmups      = getenv("NUM_WARMUPS") ? atoi(getenv("NUM_WARMUPS")) : DEFAULT_NUM_WARMUPS;
  int  numIterations   = getenv("NUM_ITERATIONS") ? atoi(getenv("NUM_ITERATIONS")) : DEFAULT_NUM_ITERATIONS;

  // Determine number of bytes to run per link
  // If a non-zero number of bytes is specified, use it
  // Otherwise generate array of bytes values to execute over
  std::vector<size_t> valuesOfN;
  size_t const numBytesPerLink = argc > 2 ? atoll(argv[2]) : DEFAULT_BYTES_PER_LINK;

  if (numBytesPerLink % 128)
  {
    printf("[ERROR] numBytesPerLink (%lu) must be a multiple of 128\n", numBytesPerLink);
    exit(1);
  }

  if (numBytesPerLink != 0)
  {
    size_t N = numBytesPerLink / sizeof(float);
    valuesOfN.push_back(N);
  }
  else
  {
    for (int N = 256; N <= (1<<27); N *= 2)
    {
      int decimationFactor = 1;  // This can be modified to increase number of samples between powers of two
      int delta = std::max(32, N / decimationFactor);
      int curr = N;
      while (curr < N * 2)
      {
        valuesOfN.push_back(curr);
        curr += delta;
      }
    }
  }

  if (byteOffset % 4)
  {
    printf("[ERROR] byteOffset must be a multiple of 4\n");
    exit(1);
  }
  int initOffset = byteOffset / sizeof(float);

  char *env;
  if (!outputToCsv)
  {
    printf("Run configuration\n");
    printf("=====================================================\n");
    printf("%-20s %8s: Using %s\n",
           "USE_HIP_CALL", useHipCall ? "(set)" : "(unset)",
           useHipCall ? "HIP functions" : "custom kernels");
    printf("%-20s %8s: Performing %s\n",
           "USE_MEMSET", useMemset ? "(set)" : "(unset)",
           useMemset ? "memset" : "memcopy");
    if (useHipCall && !useMemset)
    {
      env = getenv("HSA_ENABLE_SDMA");
      printf("%-20s %8s: %s\n",
             "HSA_ENABLE_SDMA", env ? env : "(unset)",
             (env && !strcmp(env, "0")) ? "Using blit kernels for hipMemcpy" : "Using DMA copy engines");
    }
    printf("%-20s %8s: GPU destination memory type: %s-grained\n",
           "USE_FINEGRAIN_MEM", useFineGrainMem ? "(set)" : "(unset)",
           useFineGrainMem ? "fine" : "coarse");
    printf("%-20s %8s: %s\n",
           "USE_SINGLE_SYNC", useSingleSync ? "(set)" : "(unset)",
           useSingleSync ? "Synchronizing only once, after all iterations" : "Synchronizing per iteration");
    printf("%-20s %8s: Running in %s mode\n",
           "USE_INTERACTIVE", useInteractive ? "(set)" : "(unset)",
           useInteractive ? "interactive" : "non-interactive");
    printf("%-20s %8s: %s\n",
           "USE_SLEEP", useSleep ? "(set)" : "(unset)",
           useSleep ? "Add sleep after each sync" : "No sleep per sync");
    printf("%-20s %8s: %s\n",
           "REUSE_STREAMS", reuseStreams ? "(set)" : "(unset)",
           reuseStreams ? "Re-using streams per topology" : "Creating/destroying streams per topology");
    printf("%-20s %8s: %s\n",
           "SHOW_ADDR", showAddr ? "(set)" : "(unset)",
           showAddr ? "Displaying src/dst mem addresses" : "Not displaying src/dst mem addresses");
    env = getenv("OUTPUT_TO_CSV");
    printf("%-20s %8s: Output to csv\n",
           "OUTPUT_TO_CSV", env ? env : "(unset)");
    env = getenv("BYTE_OFFSET");
    printf("%-20s %8s: Using byte offset of %d\n",
           "BYTE_OFFSET", env ? env : "(unset)", byteOffset);
    env = getenv("NUM_WARMUPS");
    printf("%-20s %8s: Running %d warmup iteration(s) per topology\n",
           "NUM_WARMUPS", env ? env : "(unset)", numWarmups);
    env = getenv("NUM_ITERATIONS");
    printf("%-20s %8s: Running %d timed iteration(s) per topology\n",
           "NUM_ITERATIONS", env ? env : "(unset)", numIterations);
    printf("\n");
  }

  // Collect the number of available CPUs/GPUs on this machine
  int numGpuDevices;
  HIP_CALL(hipGetDeviceCount(&numGpuDevices));
  if (numGpuDevices < 1)
  {
    printf("[ERROR] No GPU devices found\n");
    exit(1);
  }

  // Read configuration file
  FILE* fp = fopen(argv[1], "r");
  if (!fp)
  {
    printf("[ERROR] Unable to open link configuration file: [%s]\n", argv[1]);
    exit(1);
  }

  // Track links that get used
  std::map<std::pair<int, int>, int> linkMap;
  std::vector<std::vector<hipStream_t>> streamCache(numGpuDevices);

  // Print CSV header
  if (outputToCsv)
  {
    printf("Test,NumBytes,ExeGpu,SrcMem,DstMem,BW(GB/s),Time(ms),LinkDesc,SrcAddr,DstAddr,numWarmups,numIters,useHipCall,useMemSet,useFineGrain,useSingleSync,resuseStreams\n");
  }

  // Loop over each line in the configuration file
  int testNum = 0;
  char line[2048];
  while(fgets(line, 2048, fp))
  {
    // Parse links from configuration file
    std::vector<Link> links;
    ParseLinks(line, links);

    int const numLinks = links.size();
    if (numLinks == 0) continue;
    testNum++;

    // Loop over all the different number of bytes to use per Link
    for (auto N : valuesOfN)
    {
      if (!outputToCsv) printf("Test %d: [%lu bytes]\n", testNum, N * sizeof(float));
      float*                  linkSrcMem[numLinks];        // Source memory per Link
      float*                  linkDstMem[numLinks];        // Destination memory per Link
      hipStream_t             streams[numLinks];           // hipStream to use per Link
      hipEvent_t              startEvents[numLinks];       // Start event per Link
      hipEvent_t              stopEvents[numLinks];        // Stop event per Link
      hipEvent_t              dummyEvents[numLinks];       // Dummy event per Link
      std::vector<BlockParam> cpuBlockParams[numLinks];    // CPU copy of block parameters
      BlockParam*             gpuBlockParams[numLinks];    // GPU copy of block parameters

      // Clear counters
      int linkCount[numGpuDevices];
      for (int i = 0; i < numGpuDevices; i++)
        linkCount[i] = 0;

      for (int i = 0; i < numLinks; i++)
      {
        MemType srcMemType  = links[i].srcMemType;
        MemType dstMemType  = links[i].dstMemType;
        int     exeIndex    = links[i].exeIndex;
        int     srcIndex    = links[i].srcIndex;
        int     dstIndex    = links[i].dstIndex;
        int     blocksToUse = links[i].numBlocksToUse;

        // Check for valid src/dst indices
        if ((srcIndex < 0 || srcIndex >= numGpuDevices) ||
            (dstIndex < 0 || dstIndex >= numGpuDevices) ||
            (exeIndex < 0 || exeIndex >= numGpuDevices))
        {
          printf("[ERROR] Invalid link %d:(%c%d->%c%d) GPU index must be between 0 and %d inclusively\n",
                 exeIndex, MemTypeStr[srcMemType], srcIndex, MemTypeStr[dstMemType], dstIndex, numGpuDevices-1);
          exit(1);
        }

        // Enable peer-to-peer access if this is the first time seeing this pair
        if (srcMemType == MEM_GPU && dstMemType == MEM_GPU)
        {
          auto linkPair = std::make_pair(srcIndex, dstIndex);
          linkMap[linkPair]++;
          if (linkMap[linkPair] == 1 && srcIndex != dstIndex)
          {
            int canAccess;
            HIP_CALL(hipDeviceCanAccessPeer(&canAccess, srcIndex, dstIndex));
            if (!canAccess)
            {
              printf("[ERROR] Unable to enable peer access between GPU devices %d and %d\n", srcIndex, dstIndex);
              exit(1);
            }
            HIP_CALL(hipSetDevice(srcIndex));
            HIP_CALL(hipDeviceEnablePeerAccess(dstIndex, 0));
          }
        }

        // Allocate hipEvents / hipStreams on executing GPU
        HIP_CALL(hipSetDevice(exeIndex));
        HIP_CALL(hipEventCreate(&startEvents[i]));
        HIP_CALL(hipEventCreate(&stopEvents[i]));
        HIP_CALL(hipEventCreate(&dummyEvents[i]));
        HIP_CALL(hipMalloc((void**)&gpuBlockParams[i], sizeof(BlockParam) * numLinks));
        if (reuseStreams)
        {
          // If re-using streams, create new stream, otherwise point to existing stream
          if (streamCache[exeIndex].size() <= linkCount[exeIndex])
          {
            streamCache[exeIndex].resize(linkCount[exeIndex] + 1);
            HIP_CALL(hipStreamCreate(&streamCache[exeIndex][linkCount[exeIndex]]));
          }
          streams[i] = streamCache[exeIndex][linkCount[exeIndex]];
        }
        else
        {
          HIP_CALL(hipStreamCreate(&streams[i]));
        }

        // Allocate source / destination memory based on type / device index
        AllocateMemory(srcMemType, srcIndex, N * sizeof(float) + byteOffset, useFineGrainMem, &linkSrcMem[i]);
        AllocateMemory(dstMemType, dstIndex, N * sizeof(float) + byteOffset, useFineGrainMem, &linkDstMem[i]);

        // Initialize source memory with patterned data
        CheckOrFill(MODE_FILL, N, useMemset, useHipCall, linkSrcMem[i] + initOffset);

        // Count # of links / total blocks each GPU will be working on
        linkCount[exeIndex]++;

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
                                                              j : blocksWithExtra) * 32) + initOffset;
          param.dst = linkDstMem[i] + ((j * perBlockBaseN) + ((j < blocksWithExtra) ?
                                                              j : blocksWithExtra) * 32) + initOffset;
          cpuBlockParams[i].push_back(param);
        }

        HIP_CALL(hipMemcpy(gpuBlockParams[i], cpuBlockParams[i].data(),
                           sizeof(BlockParam) * links[i].numBlocksToUse, hipMemcpyHostToDevice));
      }

      // Launch kernels (warmup iterations are not counted)
      double totalCpuTime = 0;
      double totalGpuTime[numLinks];

      for (int i = 0; i < numLinks; i++) totalGpuTime[i] = 0.0;

      for (int iteration = -numWarmups; iteration < numIterations; iteration++)
      {
        // Pause before starting first timed iteration in interactive mode
        if (useInteractive && iteration == 0)
        {
          printf("Hit <Enter> to continue: ");
          scanf("%*c");
          printf("\n");
        }

        // Start CPU timing for this iteration
        auto cpuStart = std::chrono::high_resolution_clock::now();

        // Enqueue all links
        for (int i = 0; i < numLinks; i++)
        {
          HIP_CALL(hipSetDevice(links[i].exeIndex));

          bool recordStart = (!useSingleSync || iteration == 0);
          bool recordStop  = (!useSingleSync || iteration == numIterations - 1);

          if (useHipCall)
          {
            // Record start event
            if (recordStart) HIP_CALL(hipEventRecord(startEvents[i], streams[i]));

            // Execute hipMemset / hipMemcpy
            if (useMemset)
              HIP_CALL(hipMemsetAsync(linkDstMem[i] + initOffset, 42, N * sizeof(float), streams[i]));
            else
              HIP_CALL(hipMemcpyAsync(linkDstMem[i] + initOffset,
                                      linkSrcMem[i] + initOffset,
                                      N * sizeof(float), hipMemcpyDeviceToDevice,
                                      streams[i]));
            // Record stop event
            if (recordStop) HIP_CALL(hipEventRecord(stopEvents[i], streams[i]));
          }
          else
          {
            hipExtLaunchKernelGGL(useMemset ? MemsetKernel : CopyKernel,
                                  dim3(links[i].numBlocksToUse, 1, 1),
                                  dim3(BLOCKSIZE, 1, 1),
                                  0, streams[i],
                                  recordStart ? startEvents[i] : NULL,
                                  recordStop  ?  stopEvents[i] : NULL,
                                  0, gpuBlockParams[i]);
          }
        }

        // Synchronize per iteration, unless in single sync mode, in which case
        // synchronize during last warmup / last actual iteration
        if (!useSingleSync || iteration == -1 || iteration == numIterations - 1)
        {
          for (int i = 0; i < numLinks; i++)
          {
            HIP_CALL(hipSetDevice(links[i].exeIndex));
            hipStreamSynchronize(streams[i]);
          }
        }

        // Stop CPU timing for this iteration
        auto cpuDelta = std::chrono::high_resolution_clock::now() - cpuStart;
        double deltaSec = std::chrono::duration_cast<std::chrono::duration<double>>(cpuDelta).count();
        if (useSleep) usleep(100000);

        if (iteration >= 0)
        {
          totalCpuTime += deltaSec;

          // Record GPU timing
          if (!useSingleSync || iteration == numIterations - 1)
          {
            for (int i = 0; i < numLinks; i++)
            {
              HIP_CALL(hipSetDevice(links[i].exeIndex));
              HIP_CALL(hipEventSynchronize(startEvents[i]));
              HIP_CALL(hipEventSynchronize(stopEvents[i]));
              float gpuDeltaMsec;
              HIP_CALL(hipEventElapsedTime(&gpuDeltaMsec, startEvents[i], stopEvents[i]));
              totalGpuTime[i] += gpuDeltaMsec;
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
        CheckOrFill(MODE_CHECK, N, useMemset, useHipCall, linkDstMem[i] + initOffset);

      // Report timings
      totalCpuTime = totalCpuTime / (1.0 * numIterations) * 1000;
      double totalBandwidthGbs = 0.0;
      for (int i = 0; i < numLinks; i++)
      {
        double linkDurationMsec = totalGpuTime[i] / (1.0 * numIterations);
        double linkBandwidthGbs = (N * sizeof(float) / 1.0E9) / linkDurationMsec * 1000.0f;
        totalBandwidthGbs += linkBandwidthGbs;
        if (!outputToCsv)
        {
          printf(" Link %02d: %c%02d -> [GPU %02d:%02d] -> %c%02d | %9.3f GB/s | %8.3f ms | %9s |",
                 i + 1,
                 MemTypeStr[links[i].srcMemType], links[i].srcIndex,
                 links[i].exeIndex, links[i].numBlocksToUse,
                 MemTypeStr[links[i].dstMemType], links[i].dstIndex,
                 linkBandwidthGbs, linkDurationMsec,
                 GetLinkDesc(links[i]).c_str());
          if (showAddr) printf(" %16p | %16p |", linkSrcMem[i] + initOffset, linkDstMem[i] + initOffset);
          printf("\n");
        }
        else
        {
          printf("%d,%lu,%02d,%c%02d,%c%02d,%9.3f,%8.3f,%s,%p,%p,%d,%d,%s,%s,%s,%s,%s\n",
                 testNum, N * sizeof(float), links[i].exeIndex,
                 MemTypeStr[links[i].srcMemType], links[i].srcIndex,
                 MemTypeStr[links[i].dstMemType], links[i].dstIndex,
                 linkBandwidthGbs, linkDurationMsec,
                 GetLinkDesc(links[i]).c_str(),
                 linkSrcMem[i] + initOffset, linkDstMem[i] + initOffset,
                 numWarmups, numIterations,
                 useHipCall ? "true" : "false",
                 useMemset ? "true" : "false",
                 useFineGrainMem ? "true" : "false",
                 useSingleSync ? "true" : "false",
                 reuseStreams ? "true" : "false");
        }
      }

      // Display aggregate statistics
      if (!outputToCsv)
      {
        printf(" Aggregate Bandwidth                | %9.3f GB/s | %8.3f ms |\n", totalBandwidthGbs, totalCpuTime);
      }
      else
      {
        printf("%d,%lu,ALL,ALL,ALL,%9.3f,%8.3f,ALL,ALL,ALL,%d,%d,%s,%s,%s,%s,%s\n",
               testNum, N * sizeof(float), totalBandwidthGbs, totalCpuTime, numWarmups, numIterations,
               useHipCall ? "true" : "false",
               useMemset ? "true" : "false",
               useFineGrainMem ? "true" : "false",
               useSingleSync ? "true" : "false",
               reuseStreams ? "true" : "false");
      }

      // Release GPU memory
      for (int i = 0; i < numLinks; i++)
      {
        DeallocateMemory(links[i].srcMemType, links[i].srcIndex, linkSrcMem[i]);
        DeallocateMemory(links[i].dstMemType, links[i].dstIndex, linkDstMem[i]);
        HIP_CALL(hipFree(gpuBlockParams[i]));
        if (!reuseStreams)
          HIP_CALL(hipStreamDestroy(streams[i]));
        HIP_CALL(hipEventDestroy(startEvents[i]));
        HIP_CALL(hipEventDestroy(stopEvents[i]));
      }
    }
  }
  fclose(fp);

  // Clean up stream cache if re-using streams
  if (reuseStreams)
  {
    for (auto streamVector : streamCache)
      for (auto stream : streamVector)
        HIP_CALL(hipStreamDestroy(stream));
  }

  return 0;
}

void DisplayUsage(char const* cmdName)
{
  printf("Usage: %s configFile <N>\n", cmdName);

  printf("  configFile: File containing Links to execute (see below for format)\n");
  printf("  N         : (Optional) Number of bytes to transfer per link.\n");
  printf("              If not specified, defaults to %lu bytes. Must be a multiple of 128 bytes\n", DEFAULT_BYTES_PER_LINK);
  printf("              If 0 is specified, a range of Ns will be benchmarked\n");
  printf("              If a negative number is specified, a configFile gets generated with this number as default number of CUs per link\n");
  printf("\n");
  printf("Configfile Format:\n");
  printf("==================\n");
  printf("A Link is defined as a uni-directional transfer from src memory location to dst memory location\n");
  printf("Each single line in the configuration file defines a set of Links to run in parallel\n");
  printf("\n");
  printf("There are two ways to specify the configuration file:\n");
  printf("\n");
  printf("1) Basic\n");
  printf("   The basic specification assumes the same number of threadblocks/CUs used per link\n");
  printf("   A positive number of Links is specified followed by that number of triplets describing each Link\n");
  printf("\n");
  printf("   #Links #CUs (GPUIndex1 srcMem1 dstMem1) ... (GPUIndexL srcMemL dstMemL)\n");
  printf("\n");
  printf("2) Advanced\n");
  printf("   The advanced specification allows different number of threadblocks/CUs used per Link\n");
  printf("   A negative number of links is specified, followed by quadruples describing each Link\n");
  printf("   -#Links (GPUIndex1 #CUs1 srcMem1 dstMem1) ... (GPUIndexL #CUsL srcMemL dstMemL)\n");
  printf("\n");
  printf("Argument Details:\n");
  printf("  #Links  :   Number of Links to be run in parallel\n");
  printf("  #CUs    :   Number of threadblocks/CUs to use for a Link\n");
  printf("  GpuIndex:   0-indexed GPU id executing the Link\n");
  printf("  srcMemL :   Source memory location (Where the data is to be read from). Ignored in memset mode\n");
  printf("  dstMemL :   Destination memory location (Where the data is to be written to)\n");
  printf("              Memory locations are specified by a character indicating memory type, followed by GPU device index (0-indexed)\n");
  printf("              Supported memory locations are:\n");
  printf("              - P:    Pinned host memory   (on CPU, on NUMA node closest to provided GPU index)\n");
  printf("              - G:    Global device memory (on GPU)\n");
  printf("Round brackets may be included for human clarity, but will be ignored\n");
  printf("\n");
  printf("Examples:\n");
  printf("1 4 (0 G0 G1)              Single Link that uses 4 CUs on GPU 0 that reads memory from GPU 0 and copies it to memory on GPU 1\n");
  printf("1 4 (0 G1 G0)              Single Link that uses 4 CUs on GPU 0 that reads memory from GPU 1 and copies it to memory on GPU 0\n");
  printf("1 4 (2 P0 G2)              Single Link that uses 4 CUs on GPU 2 that reads memory from CPU 0 and copies it to memory on GPU 2\n");
  printf("2 4 (0 G0 G1) (1 G1 G0)    Runs 2 Links in parallel.  GPU 0 - > GPU1, and GP1 -> GPU 0, each with 4 CUs\n");
  printf("-2 (0 G0 G1 4) (1 G1 G0 2) Runs 2 Links in parallel.  GPU 0 - > GPU 1 using four CUs, and GPU1 -> GPU 0 using two CUs\n");
  printf("\n");
  printf("\n");
  printf("Environment variables:\n");
  printf("======================\n");
  printf(" USE_HIP_CALL       - Use hipMemcpy/hipMemset instead of custom shader kernels\n");
  printf(" USE_MEMSET         - Perform a memset instead of a copy (ignores source memory)\n");
  printf(" USE_FINEGRAIN_MEM  - Allocate fine-grained GPU memory instead of coarse-grained GPU memory\n");
  printf(" USE_SINGLE_SYNC    - Perform synchronization only once after all iterations instead of per iteration\n");
  printf(" USE_INTERACTIVE    - Pause for user-input before starting transfer loop\n");
  printf(" USE_SLEEP          - Adds a 100ms sleep after each synchronization\n");
  printf(" REUSE_STREAMS      - Re-use streams instead of creating / destroying per test\n");
  printf(" SHOW_ADDR          - Print out memory addresses for each Link\n");
  printf(" OUTPUT_TO_CSV      - Outputs to CSV format if set\n");
  printf(" BYTE_OFFSET        - Initial byte-offset for memory allocations.  Must be multiple of 4. Defaults to 0\n");
  printf(" NUM_WARMUPS=W      - Perform W untimed warmup iteration(s) per test\n");
  printf(" NUM_ITERATIONS=I   - Perform I timed iteration(s) per test\n");
}

void GenerateConfigFile(char const* cfgFile, int numBlocks)
{
  // Detect number of available GPUs and skip if less than 2
  int numGpuDevices;
  HIP_CALL(hipGetDeviceCount(&numGpuDevices));
  printf("Generated configFile %s for %d device(s) / %d CUs per link\n", cfgFile, numGpuDevices, numBlocks);
  if (numGpuDevices < 2)
  {
    printf("Skipping. (Less than 2 GPUs detected)\n");
    exit(0);
  }

  // Open config file for writing
  FILE* fp = fopen(cfgFile, "w");
  if (!fp)
  {
    printf("Unable to open [%s] for writing\n", cfgFile);
    exit(1);
  }

  // CU testing
  fprintf(fp, "# CU scaling tests\n");
  for (int i = 1; i < 16; i++)
    fprintf(fp, "1 %d (0 G0 G1)\n", i);
  fprintf(fp, "\n");

  // Pinned memory testing
  fprintf(fp, "# Pinned CPU memory read tests\n");
  for (int i = 0; i < numGpuDevices; i++)
    fprintf(fp, "1 %d (%d C%d G%d)\n", numBlocks, i, i, i);
  fprintf(fp, "\n");

  fprintf(fp, "# Pinned CPU memory write tests\n");
  for (int i = 0; i < numGpuDevices; i++)
    fprintf(fp, "1 %d (%d G%d C%d)\n", numBlocks, i, i, i);
  fprintf(fp, "\n");

  // Single link testing GPU testing
  fprintf(fp, "# Unidirectional link GPU tests\n");
  for (int i = 0; i < numGpuDevices; i++)
    for (int j = 0; j < numGpuDevices; j++)
    {
      if (i == j) continue;
      fprintf(fp, "1 %d (%d G%d G%d)\n", numBlocks, i, i, j);
    }
  fprintf(fp, "\n");

  // Bi-directional link testing
  fprintf(fp, "# Bi-directional link tests\n");
  for (int i = 0; i < numGpuDevices; i++)
    for (int j = 0; j < numGpuDevices; j++)
    {
      if (i == j) continue;
      fprintf(fp, "2 %d (%d G%d G%d) (%d G%d G%d)\n", numBlocks, i, i, j, j, j, i);
    }
  fprintf(fp, "\n");

  // Simple uni-directional ring
  fprintf(fp, "# Simple unidirectional ring\n");
  fprintf(fp, "%d %d", numGpuDevices, numBlocks);
  for (int i = 0; i < numGpuDevices; i++)
  {
    fprintf(fp, " (%d G%d G%d)", i, i, (i+1)%numGpuDevices);
  }
  fprintf(fp, "\n\n");

  // Simple bi-directional ring
  fprintf(fp, "# Simple bi-directional ring\n");
  fprintf(fp, "%d %d", numGpuDevices * 2, numBlocks);
  for (int i = 0; i < numGpuDevices; i++)
    fprintf(fp, " (%d G%d G%d)", i, i, (i+1)%numGpuDevices);
  for (int i = 0; i < numGpuDevices; i++)
    fprintf(fp, " (%d G%d G%d)", i, i, (i+numGpuDevices-1)%numGpuDevices);
  fprintf(fp, "\n\n");

  // Broadcast from GPU 0
  fprintf(fp, "# GPU 0 Broadcast\n");
  fprintf(fp, "%d %d", numGpuDevices-1, numBlocks);
  for (int i = 1; i < numGpuDevices; i++)
    fprintf(fp, " (%d G%d G%d)", 0, 0, i);
  fprintf(fp, "\n\n");

  // Gather to GPU 0
  fprintf(fp, "# GPU 0 Gather\n");
  fprintf(fp, "%d %d", numGpuDevices-1, numBlocks);
  for (int i = 1; i < numGpuDevices; i++)
    fprintf(fp, " (%d G%d G%d)", 0, i, 0);
  fprintf(fp, "\n\n");

  // Full stress test
  fprintf(fp, "# Full stress test\n");
  fprintf(fp, "%d %d", numGpuDevices * (numGpuDevices-1), numBlocks);
  for (int i = 0; i < numGpuDevices; i++)
    for (int j = 0; j < numGpuDevices; j++)
    {
      if (i == j) continue;
      fprintf(fp, " (%d G%d G%d)", i, i, j);
    }
  fprintf(fp, "\n\n");

  fclose(fp);
}

void DisplayTopology()
{
  printf("\nDetected topology:\n");
  int numGpuDevices;
  HIP_CALL(hipGetDeviceCount(&numGpuDevices));

  printf("        |");
  for (int j = 0; j < numGpuDevices; j++)
    printf(" GPU %02d |", j);
  printf("\n");
  for (int j = 0; j <= numGpuDevices; j++)
    printf("--------+");
  printf("\n");

  for (int i = 0; i < numGpuDevices; i++)
  {
    printf(" GPU %02d |", i);
    for (int j = 0; j < numGpuDevices; j++)
    {
      if (i == j)
        printf("    -   |");
      else
      {
        uint32_t linkType, hopCount;
        HIP_CALL(hipExtGetLinkTypeAndHopCount(i, j, &linkType, &hopCount));
        printf(" %s-%d |",
               linkType == HSA_AMD_LINK_INFO_TYPE_HYPERTRANSPORT ? "  HT" :
               linkType == HSA_AMD_LINK_INFO_TYPE_QPI            ? " QPI" :
               linkType == HSA_AMD_LINK_INFO_TYPE_PCIE           ? "PCIE" :
               linkType == HSA_AMD_LINK_INFO_TYPE_INFINBAND      ? "INFB" :
               linkType == HSA_AMD_LINK_INFO_TYPE_XGMI           ? "XGMI" : "????",
               hopCount);
      }
    }
    printf("\n");
  }
}

void ParseMemType(std::string const& token, MemType* memType, int* memIndex)
{
  char typeChar;
  if (sscanf(token.c_str(), " %c %d", &typeChar, memIndex) != 2)
  {
    printf("Error parsing memory type token %s\n", token.c_str());
    exit(1);
  }

  switch (typeChar)
  {
  case 'C': case 'c': *memType = MEM_CPU; break;
  case 'G': case 'g': *memType = MEM_GPU; break;
  default: printf("Unrecognized memory type %s\n", token.c_str()); exit(1);
  }
}

// Helper function to parse a link of link definitions
void ParseLinks(char* line, std::vector<Link>& links)
{
  // Replace any round brackets with spaces
  for (int i = 0; line[i]; i++)
    if (line[i] == '(' || line[i] == ')') line[i] = ' ';

  links.clear();
  int numLinks = 0;

  std::istringstream iss;
  iss.clear();
  iss.str(line);
  iss >> numLinks;
  if (iss.fail()) return;

  std::string srcMem;
  std::string dstMem;
  if (numLinks > 0)
  {
    // Method 1: Take in triples (exeGpu, srcMem, dstMem)
    int numBlocksToUse;
    iss >> numBlocksToUse;
    if (numBlocksToUse <= 0)
    {
      printf("Parsing error: Number of blocks to use (%d) must be greater than 0\n", numBlocksToUse);
      exit(1);
    }
    links.resize(numLinks);
    for (int i = 0; i < numLinks; i++)
    {
      iss >> links[i].exeIndex >> srcMem >> dstMem;
      ParseMemType(srcMem, &links[i].srcMemType, &links[i].srcIndex);
      ParseMemType(dstMem, &links[i].dstMemType, &links[i].dstIndex);
      links[i].numBlocksToUse = numBlocksToUse;
    }
  }
  else
  {
    // Method 2: Read in quads (exeGpu, srcMem, dstMem,  Read common # blocks to use, then read (src, dst) doubles
    numLinks *= -1;
    links.resize(numLinks);

    for (int i = 0; i < numLinks; i++)
    {
      iss >> links[i].exeIndex >> srcMem >> dstMem >> links[i].numBlocksToUse;
      ParseMemType(srcMem, &links[i].srcMemType, &links[i].srcIndex);
      ParseMemType(dstMem, &links[i].dstMemType, &links[i].dstIndex);
    }
  }
}

void AllocateMemory(MemType memType, int devIndex, size_t numBytes, bool useFineGrainMem, float** memPtr)
{
  HIP_CALL(hipSetDevice(devIndex));

  if (memType == MEM_CPU)
  {
    // // Allocate pinned-memory on NUMA node closest to the selected GPU
    HIP_CALL(hipHostMalloc((void **)memPtr, numBytes, hipHostMallocPortable));
  }
  else if (memType == MEM_GPU)
  {
    // Allocate GPU memory
    if (useFineGrainMem)
      HIP_CALL(hipExtMallocWithFlags((void**)memPtr, numBytes, hipDeviceMallocFinegrained));
    else
      HIP_CALL(hipMalloc((void**)memPtr, numBytes));
  }
  else
  {
    printf("Error: Unsupported memory type %d\n", memType);
    exit(1);
  }
}

void DeallocateMemory(MemType memType, int devIndex, float* memPtr)
{
  if (memType == MEM_CPU)
  {
    HIP_CALL(hipHostFree(memPtr));
  }
  else if (memType == MEM_GPU)
  {
    HIP_CALL(hipFree(memPtr));
  }
}

// Helper function to either fill a device pointer with pseudo-random data, or to check to see if it matches
void CheckOrFill(ModeType mode, int N, bool isMemset, bool isHipCall, float* ptr)
{
  // Prepare reference resultx
  float* refBuffer = (float*)malloc(N * sizeof(float));
  if (isMemset)
  {
    if (isHipCall)
    {
      memset(refBuffer, 42, N * sizeof(float));
    }
    else
    {
      for (int i = 0; i < N; i++)
        refBuffer[i] = 1234.0f;
    }
  }
  else
  {
    for (int i = 0; i < N; i++)
        refBuffer[i] = (i % 383 + 31);
  }

  // Either fill the memory with the reference buffer, or compare against it
  if (mode == MODE_FILL)
  {
    HIP_CALL(hipMemcpy(ptr, refBuffer, N * sizeof(float), hipMemcpyDefault));
  }
  else if (mode == MODE_CHECK)
  {
    float* hostBuffer = (float*) malloc(N * sizeof(float));
    HIP_CALL(hipMemcpy(hostBuffer, ptr, N * sizeof(float), hipMemcpyDefault));
    for (int i = 0; i < N; i++)
    {
      if (refBuffer[i] != hostBuffer[i])
      {
        printf("[ERROR] Mismatch at element %d Ref: %f Actual: %f\n", i, refBuffer[i], hostBuffer[i]);
        exit(1);
      }
    }
  }

  free(refBuffer);
}

std::string GetLinkTypeDesc(uint32_t linkType, uint32_t hopCount)
{
  char result[10];

  switch (linkType)
  {
  case HSA_AMD_LINK_INFO_TYPE_HYPERTRANSPORT: sprintf(result, "  HT-%d", hopCount); break;
  case HSA_AMD_LINK_INFO_TYPE_QPI           : sprintf(result, " QPI-%d", hopCount); break;
  case HSA_AMD_LINK_INFO_TYPE_PCIE          : sprintf(result, "PCIE-%d", hopCount); break;
  case HSA_AMD_LINK_INFO_TYPE_INFINBAND     : sprintf(result, "INFB-%d", hopCount); break;
  case HSA_AMD_LINK_INFO_TYPE_XGMI          : sprintf(result, "XGMI-%d", hopCount); break;
  default: sprintf(result, "??????");
  }
  return result;
}

std::string GetLinkDesc(Link const& link)
{
  std::string result = "";

  // Currently only describe links between src/dst on GPU
  if (link.srcMemType == MEM_GPU && link.dstMemType == MEM_GPU)
  {
    if (link.exeIndex != link.srcIndex)
    {
      uint32_t linkType, hopCount;
      HIP_CALL(hipExtGetLinkTypeAndHopCount(link.srcIndex, link.exeIndex, &linkType, &hopCount));
      result += GetLinkTypeDesc(linkType, hopCount);
    }

    if (link.exeIndex != link.dstIndex)
    {
      uint32_t linkType, hopCount;
      HIP_CALL(hipExtGetLinkTypeAndHopCount(link.exeIndex, link.dstIndex, &linkType, &hopCount));
      if (result != "") result += "+";
      result += GetLinkTypeDesc(linkType, hopCount);
    }
  }
  else
  {
    result = "???";
  }
  return result;
}
