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
#include <numa.h>
#include <numaif.h>
#include <stack>
#include <thread>

#include "TransferBench.hpp"
#include "GetClosestNumaNode.hpp"
#include "Kernels.hpp"

// Simple configuration parameters
size_t const DEFAULT_BYTES_PER_LINK = (1<<26);  // Amount of data transferred per Link

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
  if (argc > 2 && atoll(argv[2]) < 0)
  {
    GenerateConfigFile(argv[1], -1*atoi(argv[2]));
    exit(0);
  }

  // Collect environment variables / display current run configuration
  EnvVars ev;

  // Determine number of bytes to run per Link
  // If a non-zero number of bytes is specified, use it
  // Otherwise generate array of bytes values to execute over
  std::vector<size_t> valuesOfN;
  size_t numBytesPerLink = argc > 2 ? atoll(argv[2]) : DEFAULT_BYTES_PER_LINK;
  if (argc > 2)
  {
    // Adjust bytes if unit specified
    char units = argv[2][strlen(argv[2])-1];
    switch (units)
    {
    case 'K': case 'k': numBytesPerLink *= 1024; break;
    case 'M': case 'm': numBytesPerLink *= 1024*1024; break;
    case 'G': case 'g': numBytesPerLink *= 1024*1024*1024; break;
    }
  }
  PopulateTestSizes(numBytesPerLink, ev.samplingFactor, valuesOfN);

  // Find the largest N to be used - memory will only be allocated once per link config
  size_t maxN = valuesOfN[0];
  for (auto N : valuesOfN)
    maxN = std::max(maxN, N);

  // Execute only peer to peer benchmark mode, similar to rocm-bandwidth-test
  if (!strcmp(argv[1], "p2p") || !strcmp(argv[1], "p2p_rr") ||
      !strcmp(argv[1], "g2g") || !strcmp(argv[1], "g2g_rr"))
  {
    int numBlocksToUse = 0;
    if (argc > 3)
      numBlocksToUse = atoi(argv[3]);
    else
      HIP_CALL(hipDeviceGetAttribute(&numBlocksToUse, hipDeviceAttributeMultiprocessorCount, 0));

    // Perform either local read (+remote write) [EXE = SRC] or
    // remote read (+local write)                [EXE = DST]
    int readMode = (!strcmp(argv[1], "p2p_rr") || !strcmp(argv[1], "g2g_rr") ? 1 : 0);
    int skipCpu = (!strcmp(argv[1], "g2g") || !strcmp(argv[1], "g2g_rr") ? 1 : 0);

    // Execute peer to peer benchmark mode
    RunPeerToPeerBenchmarks(ev, numBytesPerLink / sizeof(float), numBlocksToUse, readMode, skipCpu);
    exit(0);
  }

  // Check that Link configuration file can be opened
  FILE* fp = fopen(argv[1], "r");
  if (!fp)
  {
    printf("[ERROR] Unable to open link configuration file: [%s]\n", argv[1]);
    exit(1);
  }

  // Check for NUMA library support
  if (numa_available() == -1)
  {
    printf("[ERROR] NUMA library not supported. Check to see if libnuma has been installed on this system\n");
    exit(1);
  }
  ev.DisplayEnvVars();

  int const initOffset = ev.byteOffset / sizeof(float);
  std::stack<std::thread> threads;

  // Collect the number of available CPUs/GPUs on this machine
  int numGpuDevices;
  HIP_CALL(hipGetDeviceCount(&numGpuDevices));
  int const numCpuDevices = numa_num_configured_nodes();

  // Track links that get used
  std::set<std::pair<int, int>> peerAccessTracker;

  // Print CSV header
  if (ev.outputToCsv)
  {
    printf("Test,NumBytes,SrcMem,Executor,DstMem,CUs,BW(GB/s),Time(ms),LinkDesc,SrcAddr,DstAddr,ByteOffset,numWarmups,numIters,useHipCall,useMemSet,useSingleSync,combinedTiming\n");
  }

  // Loop over each line in the Link configuration file
  int testNum = 0;
  char line[2048];
  while(fgets(line, 2048, fp))
  {
    // Check if line is a comment
    if (!ev.outputToCsv && line[0] == '#' && line[1] == '#')
      printf("%s", line);

    // Parse links from configuration file
    std::vector<Link> links;
    ParseLinks(line, numCpuDevices, numGpuDevices, links);

    int const numLinks = links.size();
    if (numLinks == 0) continue;
    testNum++;

    // Prepare link
    for (int i = 0; i < numLinks; i++)
    {
      // Get some aliases to link variables
      MemType const& exeMemType  = links[i].exeMemType;
      MemType const& srcMemType  = links[i].srcMemType;
      MemType const& dstMemType  = links[i].dstMemType;
      int     const& blocksToUse = links[i].numBlocksToUse;

      // Get potentially remapped device indices
      int const srcIndex = RemappedIndex(links[i].srcIndex, srcMemType);
      int const exeIndex = RemappedIndex(links[i].exeIndex, exeMemType);
      int const dstIndex = RemappedIndex(links[i].dstIndex, dstMemType);

      // Enable peer-to-peer access if necessary (can only be called once per unique pair)
      if (exeMemType == MEM_GPU)
      {
        // Ensure executing GPU can access source memory
        if ((srcMemType == MEM_GPU || srcMemType == MEM_GPU_FINE) && srcIndex != exeIndex)
        {
          auto exeSrcPair = std::make_pair(exeIndex, srcIndex);
          if (!peerAccessTracker.count(exeSrcPair))
          {
            EnablePeerAccess(exeIndex, srcIndex);
            peerAccessTracker.insert(exeSrcPair);
          }
        }

        // Ensure executing GPU can access destination memory
        if ((dstMemType == MEM_GPU || dstMemType == MEM_GPU_FINE) && dstIndex != exeIndex)
        {
          auto exeDstPair = std::make_pair(exeIndex, dstIndex);
          if (!peerAccessTracker.count(exeDstPair))
          {
            EnablePeerAccess(exeIndex, dstIndex);
            peerAccessTracker.insert(exeDstPair);
          }
        }
      }

      // Allocate (maximum) source / destination memory based on type / device index
      AllocateMemory(srcMemType, srcIndex, maxN * sizeof(float) + ev.byteOffset, &links[i].srcMem);
      AllocateMemory(dstMemType, dstIndex, maxN * sizeof(float) + ev.byteOffset, &links[i].dstMem);

      // Prepare execution agent
      if (exeMemType == MEM_GPU)
      {
        HIP_CALL(hipSetDevice(exeIndex));
        HIP_CALL(hipEventCreate(&links[i].startEvent));
        HIP_CALL(hipEventCreate(&links[i].stopEvent));
        HIP_CALL(hipMalloc((void**)&links[i].blockParam, sizeof(BlockParam) * blocksToUse));
        HIP_CALL(hipStreamCreate(&links[i].stream));
      }
      else if (exeMemType == MEM_CPU)
      {
        links[i].blockParam = (BlockParam*)malloc(ev.numCpuPerLink * sizeof(BlockParam));
      }
    }

    // Loop over all the different number of bytes to use per Link
    for (auto N : valuesOfN)
    {
      if (!ev.outputToCsv) printf("Test %d: [%lu bytes]\n", testNum, N * sizeof(float));

      // Prepare links based on current N
      for (int i = 0; i < numLinks; i++)
      {
        // Initialize source memory with patterned data
        CheckOrFill(MODE_FILL, N, ev.useMemset, ev.useHipCall, ev.fillPattern, links[i].srcMem + initOffset);


        // Each block needs to know src/dst pointers and how many elements to transfer
        // Figure out the sub-array each block does for this Link
        // - Partition N as evenly as posible, but try to keep blocks as multiples of BLOCK_BYTES bytes,
        //   except the very last one, for alignment reasons
        int targetMultiple = ev.blockBytes / sizeof(float);
        if (links[i].exeMemType == MEM_GPU)
        {
          size_t assigned = 0;
          int maxNumBlocksToUse = std::min((N + targetMultiple - 1) / targetMultiple, (size_t)links[i].numBlocksToUse);
          for (int j = 0; j < links[i].numBlocksToUse; j++)
          {
            BlockParam param;
            int blocksLeft = std::max(0, maxNumBlocksToUse - j);
            size_t leftover = N - assigned;
            size_t roundedN = (leftover + targetMultiple - 1) / targetMultiple;
            param.N = blocksLeft ? std::min(leftover, ((roundedN / blocksLeft) * targetMultiple)) : 0;
            param.src = links[i].srcMem + assigned + initOffset;
            param.dst = links[i].dstMem + assigned + initOffset;
            assigned += param.N;

            HIP_CALL(hipMemcpy(&links[i].blockParam[j], &param, sizeof(BlockParam), hipMemcpyHostToDevice));
          }
        }
        else if (links[i].exeMemType == MEM_CPU)
        {
          // For CPU-based copy, divded based on the number of child threads
          size_t assigned = 0;
          int maxNumBlocksToUse = std::min((N + targetMultiple - 1) / targetMultiple, (size_t)ev.numCpuPerLink);
          for (int j = 0; j < ev.numCpuPerLink; j++)
          {
            int blocksLeft = std::max(0, maxNumBlocksToUse - j);
            size_t leftover = N - assigned;
            size_t roundedN = (leftover + targetMultiple - 1) / targetMultiple;
            links[i].blockParam[j].N = blocksLeft ? std::min(leftover, ((roundedN / blocksLeft) * targetMultiple)) : 0;
            links[i].blockParam[j].src = links[i].srcMem + assigned + initOffset;
            links[i].blockParam[j].dst = links[i].dstMem + assigned + initOffset;
            assigned += links[i].blockParam[j].N;
          }
        }

        // Initialize timing
        links[i].totalTime = 0.0;
      }

      double totalCpuTime = 0;

      // Launch kernels (warmup iterations are not counted)
      for (int iteration = -ev.numWarmups; iteration < ev.numIterations; iteration++)
      {
        // Pause before starting first timed iteration in interactive mode
        if (ev.useInteractive && iteration == 0)
        {
          printf("Hit <Enter> to continue: ");
          scanf("%*c");
          printf("\n");
        }

        // Start CPU timing for this iteration
        auto cpuStart = std::chrono::high_resolution_clock::now();

        // Execute all links in parallel
        for (int i = 0; i < numLinks; i++)
          threads.push(std::thread(RunLink, std::ref(ev), N, iteration, std::ref(links[i])));

        // Wait for all threads to finish
        for (int i = 0; i < numLinks; i++)
        {
          threads.top().join();
          threads.pop();
        }

        // Stop CPU timing for this iteration
        auto cpuDelta = std::chrono::high_resolution_clock::now() - cpuStart;
        double deltaSec = std::chrono::duration_cast<std::chrono::duration<double>>(cpuDelta).count();

        if (iteration >= 0) totalCpuTime += deltaSec;
      }

      // Pause for interactive mode
      if (ev.useInteractive)
      {
        printf("Transfers complete. Hit <Enter> to continue: ");
        scanf("%*c");
        printf("\n");
      }

      // Validate that each link has transferred correctly
      for (int i = 0; i < numLinks; i++)
        CheckOrFill(MODE_CHECK, N, ev.useMemset, ev.useHipCall, ev.fillPattern, links[i].dstMem + initOffset);

      // Report timings
      totalCpuTime = totalCpuTime / (1.0 * ev.numIterations) * 1000;
      double totalBandwidthGbs = (numLinks * N * sizeof(float) / 1.0E6) / totalCpuTime;
      double maxGpuTime = 0;
      for (int i = 0; i < numLinks; i++)
      {
        double linkDurationMsec = links[i].totalTime / (1.0 * ev.numIterations);
        double linkBandwidthGbs = (N * sizeof(float) / 1.0E9) / linkDurationMsec * 1000.0f;
        maxGpuTime = std::max(maxGpuTime, linkDurationMsec);
        if (!ev.outputToCsv)
        {
          printf(" Link %02d: %c%02d -> [%cPU %02d:%02d] -> %c%02d | %9.3f GB/s | %8.3f ms | %-16s",
                 i + 1,
                 MemTypeStr[links[i].srcMemType], links[i].srcIndex,
                 MemTypeStr[links[i].exeMemType], links[i].exeIndex,
                 links[i].exeMemType == MEM_CPU ? ev.numCpuPerLink : links[i].numBlocksToUse,
                 MemTypeStr[links[i].dstMemType], links[i].dstIndex,
                 linkBandwidthGbs, linkDurationMsec,
                 GetLinkDesc(links[i]).c_str());
          if (ev.showAddr) printf(" %16p | %16p |", links[i].srcMem + initOffset, links[i].dstMem + initOffset);
          printf("\n");
        }
        else
        {
          printf("%d,%lu,%c%02d,%c%02d,%c%02d,%d,%9.3f,%8.3f,%s,%p,%p,%d,%d,%d,%s,%s,%s,%s\n",
                 testNum, N * sizeof(float),
                 MemTypeStr[links[i].srcMemType], links[i].srcIndex,
                 MemTypeStr[links[i].exeMemType], links[i].exeIndex,
                 MemTypeStr[links[i].dstMemType], links[i].dstIndex,
                 links[i].exeMemType == MEM_CPU ? ev.numCpuPerLink : links[i].numBlocksToUse,
                 linkBandwidthGbs, linkDurationMsec,
                 GetLinkDesc(links[i]).c_str(),
                 links[i].srcMem + initOffset, links[i].dstMem + initOffset,
                 ev.byteOffset,
                 ev.numWarmups, ev.numIterations,
                 ev.useHipCall ? "true" : "false",
                 ev.useMemset ? "true" : "false",
                 ev.useSingleSync ? "true" : "false",
                 ev.combineTiming ? "true" : "false");
        }
      }

      // Display aggregate statistics
      if (!ev.outputToCsv)
      {
        printf(" Aggregate Bandwidth (CPU timed)    | %9.3f GB/s | %8.3f ms | Overhead: %.3f ms\n", totalBandwidthGbs, totalCpuTime,
               totalCpuTime - maxGpuTime);
      }
      else
      {
        printf("%d,%lu,ALL,ALL,ALL,ALL,%9.3f,%8.3f,ALL,ALL,ALL,%d,%d,%d,%s,%s,%s,%s\n",
               testNum, N * sizeof(float), totalBandwidthGbs, totalCpuTime, ev.byteOffset,
               ev.numWarmups, ev.numIterations,
               ev.useHipCall ? "true" : "false",
               ev.useMemset ? "true" : "false",
               ev.useSingleSync ? "true" : "false",
               ev.combineTiming ? "true" : "false");
      }
    }

    // Release GPU memory
    for (int i = 0; i < numLinks; i++)
    {
      DeallocateMemory(links[i].srcMemType, links[i].srcMem);
      DeallocateMemory(links[i].dstMemType, links[i].dstMem);

      if (links[i].exeMemType == MEM_GPU)
      {
        HIP_CALL(hipEventDestroy(links[i].startEvent));
        HIP_CALL(hipEventDestroy(links[i].stopEvent));
        HIP_CALL(hipStreamDestroy(links[i].stream));
        HIP_CALL(hipFree(links[i].blockParam));
      }
      else if (links[i].exeMemType == MEM_CPU)
      {
        free(links[i].blockParam);
      }
    }
  }
  fclose(fp);

  return 0;
}

void DisplayUsage(char const* cmdName)
{
  if (numa_available() == -1)
  {
    printf("[ERROR] NUMA library not supported. Check to see if libnuma has been installed on this system\n");
    exit(1);
  }
  int numGpuDevices;
  HIP_CALL(hipGetDeviceCount(&numGpuDevices));
  int const numCpuDevices = numa_num_configured_nodes();

  printf("Usage: %s config <N>\n", cmdName);
  printf("  config: Either:\n");
  printf("          - Filename of configFile containing Links to execute (see below for format)\n");
  printf("          - Name of preset benchmark:\n");
  printf("              p2p    - All CPU/GPU pairs benchmark\n");
  printf("              p2p_rr - All CPU/GPU pairs benchmark with remote reads\n");
  printf("              g2g    - All GPU/GPU pairs benchmark\n");
  printf("              g2g_rr - All GPU/GPU pairs benchmark with remote reads\n");
  printf("            - 3rd optional argument will be used as # of CUs to use (uses all by default)\n");
  printf("  N     : (Optional) Number of bytes to transfer per link.\n");
  printf("          If not specified, defaults to %lu bytes. Must be a multiple of 4 bytes\n", DEFAULT_BYTES_PER_LINK);
  printf("          If 0 is specified, a range of Ns will be benchmarked\n");
  printf("          If a negative number is specified, a configFile gets generated with this number as default number of CUs per link\n");
  printf("          May append a suffix ('K', 'M', 'G') for kilobytes / megabytes / gigabytes\n");
  printf("\n");
  printf("Configfile Format:\n");
  printf("==================\n");
  printf("A Link is defined as a uni-directional transfer from src memory location to dst memory location executed by either CPU or GPU\n");
  printf("Each single line in the configuration file defines a set of Links to run in parallel\n");
  printf("\n");
  printf("There are two ways to specify the configuration file:\n");
  printf("\n");
  printf("1) Basic\n");
  printf("   The basic specification assumes the same number of threadblocks/CUs used per GPU-executed Link\n");
  printf("   A positive number of Links is specified followed by that number of triplets describing each Link\n");
  printf("\n");
  printf("   #Links #CUs (srcMem1->Executor1->dstMem1) ... (srcMemL->ExecutorL->dstMemL)\n");
  printf("\n");
  printf("2) Advanced\n");
  printf("   The advanced specification allows different number of threadblocks/CUs used per GPU-executed Link\n");
  printf("   A negative number of links is specified, followed by quadruples describing each Link\n");
  printf("   -#Links (srcMem1->Executor1->dstMem1 #CUs1) ... (srcMemL->ExecutorL->dstMemL #CUsL)\n");
  printf("\n");
  printf("Argument Details:\n");
  printf("  #Links  :   Number of Links to be run in parallel\n");
  printf("  #CUs    :   Number of threadblocks/CUs to use for a GPU-executed Link\n");
  printf("  srcMemL :   Source memory location (Where the data is to be read from). Ignored in memset mode\n");
  printf("  Executor:   Executor are specified by a character indicating executor type, followed by device index (0-indexed)\n");
  printf("              - C: CPU-executed  (Indexed from 0 to %d)\n", numCpuDevices-1);
  printf("              - G: GPU-executed  (Indexed from 0 to %d)\n", numGpuDevices-1);
  printf("  dstMemL :   Destination memory location (Where the data is to be written to)\n");
  printf("\n");
  printf("              Memory locations are specified by a character indicating memory type, followed by device index (0-indexed)\n");
  printf("              Supported memory locations are:\n");
  printf("              - C:    Pinned host memory       (on NUMA node, indexed from 0 to %d)\n", numCpuDevices-1);
  printf("              - G:    Global device memory     (on GPU device indexed from 0 to %d)\n", numGpuDevices-1);
  printf("              - F:    Fine-grain device memory (on GPU device indexed from 0 to %d)\n", numGpuDevices-1);
  printf("\n");
  printf("Examples:\n");
  printf("1 4 (G0->G0->G1)             Single Link that uses 4 CUs on GPU 0 that reads memory from GPU 0 and copies it to memory on GPU 1\n");
  printf("1 4 (G1->C0->G0)             Single Link that uses CPU 0 to read memory from GPU 1 and then copies it to memory on GPU 0\n");
  printf("1 4 (C0->G2->G2)             Single Link that uses 4 CUs on GPU 2 that reads memory from CPU 0 and copies it to memory on GPU 2\n");
  printf("2 4 G0->G0->G1 G1->G1->G0    Runs 2 Links in parallel.  GPU 0 - > GPU1, and GP1 -> GPU 0, each with 4 CUs\n");
  printf("-2 (G0 G0 G1 4) (G1 G1 G0 2) Runs 2 Links in parallel.  GPU 0 - > GPU 1 using four CUs, and GPU1 -> GPU 0 using two CUs\n");
  printf("\n");
  printf("Round brackets and arrows' ->' may be included for human clarity, but will be ignored and are unnecessary\n");
  printf("Lines starting with # will be ignored. Lines starting with ## will be echoed to output\n");
  printf("\n");

  EnvVars::DisplayUsage();
}

void GenerateConfigFile(char const* cfgFile, int numBlocks)
{
  // Detect number of available GPUs and skip if less than 2
  int numGpuDevices;
  HIP_CALL(hipGetDeviceCount(&numGpuDevices));
  printf("Generating configFile %s for %d device(s) / %d CUs per link\n", cfgFile, numGpuDevices, numBlocks);
  if (numGpuDevices < 2)
  {
    printf("Skipping. (Less than 2 GPUs detected)\n");
    exit(0);
  }

  // Check first to see if file exists, and issue warning
  FILE* exists = fopen(cfgFile, "r");
  if (exists)
  {
    fclose(exists);
    printf("[WARN] File %s alreadys exists.  Enter 'Y' to confirm overwrite\n", cfgFile);
    char ch;
    scanf(" %c", &ch);
    if (ch != 'Y' && ch != 'y')
    {
      printf("Aborting\n");
      exit(0);
    }
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
    fprintf(fp, "1 %d (G0->G0->G1)\n", i);
  fprintf(fp, "\n");

  // Pinned memory testing
  fprintf(fp, "# Pinned CPU memory read tests\n");
  for (int i = 0; i < numGpuDevices; i++)
    fprintf(fp, "1 %d (C0->G%d->G%d)\n", numBlocks, i, i);
  fprintf(fp, "\n");

  fprintf(fp, "# Pinned CPU memory write tests\n");
  for (int i = 0; i < numGpuDevices; i++)
    fprintf(fp, "1 %d (G%d->G%d->C0)\n", numBlocks, i, i);
  fprintf(fp, "\n");

  // Single link testing GPU testing
  fprintf(fp, "# Unidirectional link GPU tests\n");
  for (int i = 0; i < numGpuDevices; i++)
    for (int j = 0; j < numGpuDevices; j++)
    {
      if (i == j) continue;
      fprintf(fp, "1 %d (G%d->G%d->G%d)\n", numBlocks, i, i, j);
    }
  fprintf(fp, "\n");

  // Bi-directional link testing
  fprintf(fp, "# Bi-directional link tests\n");
  for (int i = 0; i < numGpuDevices; i++)
    for (int j = 0; j < numGpuDevices; j++)
    {
      if (i == j) continue;
      fprintf(fp, "2 %d (G%d->G%d->G%d) (G%d->G%d->G%d)\n", numBlocks, i, i, j, j, j, i);
    }
  fprintf(fp, "\n");

  // Simple uni-directional ring
  fprintf(fp, "# Simple unidirectional ring\n");
  fprintf(fp, "%d %d", numGpuDevices, numBlocks);
  for (int i = 0; i < numGpuDevices; i++)
  {
    fprintf(fp, " (G%d->G%d->G%d)", i, i, (i+1)%numGpuDevices);
  }
  fprintf(fp, "\n\n");

  // Simple bi-directional ring
  fprintf(fp, "# Simple bi-directional ring\n");
  fprintf(fp, "%d %d", numGpuDevices * 2, numBlocks);
  for (int i = 0; i < numGpuDevices; i++)
    fprintf(fp, " (G%d->G%d->G%d)", i, i, (i+1)%numGpuDevices);
  for (int i = 0; i < numGpuDevices; i++)
    fprintf(fp, " (G%d->G%d->G%d)", i, i, (i+numGpuDevices-1)%numGpuDevices);
  fprintf(fp, "\n\n");

  // Broadcast from GPU 0
  fprintf(fp, "# GPU 0 Broadcast\n");
  fprintf(fp, "%d %d", numGpuDevices-1, numBlocks);
  for (int i = 1; i < numGpuDevices; i++)
    fprintf(fp, " (G%d->G%d->G%d)", 0, 0, i);
  fprintf(fp, "\n\n");

  // Gather to GPU 0
  fprintf(fp, "# GPU 0 Gather\n");
  fprintf(fp, "%d %d", numGpuDevices-1, numBlocks);
  for (int i = 1; i < numGpuDevices; i++)
    fprintf(fp, " (G%d->G%d->G%d)", i, 0, 0);
  fprintf(fp, "\n\n");

  // Full stress test
  fprintf(fp, "# Full stress test\n");
  fprintf(fp, "%d %d", numGpuDevices * (numGpuDevices-1), numBlocks);
  for (int i = 0; i < numGpuDevices; i++)
    for (int j = 0; j < numGpuDevices; j++)
    {
      if (i == j) continue;
      fprintf(fp, " (G%d->G%d->G%d)", i, i, j);
    }
  fprintf(fp, "\n\n");

  // All single-hop XGMI links
  int numSingleHopXgmiLinks = 0;
  for (int i = 0; i < numGpuDevices; i++)
    for (int j = 0; j < numGpuDevices; j++)
    {
      if (i == j) continue;
      uint32_t linkType, hopCount;
      HIP_CALL(hipExtGetLinkTypeAndHopCount(i, j, &linkType, &hopCount));
      if (linkType == HSA_AMD_LINK_INFO_TYPE_XGMI && hopCount == 1) numSingleHopXgmiLinks++;
    }
  if (numSingleHopXgmiLinks > 0)
  {
    fprintf(fp, "# All single-hop links\n");
    fprintf(fp, "%d %d", numSingleHopXgmiLinks, numBlocks);
    for (int i = 0; i < numGpuDevices; i++)
      for (int j = 0; j < numGpuDevices; j++)
      {
        if (i == j) continue;
        uint32_t linkType, hopCount;
        HIP_CALL(hipExtGetLinkTypeAndHopCount(i, j, &linkType, &hopCount));
        if (linkType == HSA_AMD_LINK_INFO_TYPE_XGMI && hopCount == 1)
        {
          fprintf(fp, " (G%d G%d F%d)", i, i, j);
        }
      }
    fprintf(fp, "\n\n");
  }
  fclose(fp);
}

int RemappedIndex(int const origIdx, MemType const memType)
{
  static std::vector<int> remapping;

  // No need to re-map CPU devices
  if (memType == MEM_CPU) return origIdx;

  // Build remapping on first use
  if (remapping.empty())
  {
    int numGpuDevices;
    HIP_CALL(hipGetDeviceCount(&numGpuDevices));
    remapping.resize(numGpuDevices);

    int const usePcieIndexing = getenv("USE_PCIE_INDEX") ? atoi(getenv("USE_PCIE_INDEX")) : 0;
    if (!usePcieIndexing)
    {
      // For HIP-based indexing no remapping is necessary
      for (int i = 0; i < numGpuDevices; ++i)
        remapping[i] = i;
    }
    else
    {
      // Collect PCIe address for each GPU
      std::vector<std::pair<std::string, int>> mapping;
      char pciBusId[20];
      for (int i = 0; i < numGpuDevices; ++i)
      {
        HIP_CALL(hipDeviceGetPCIBusId(pciBusId, 20, i));
        mapping.push_back(std::make_pair(pciBusId, i));
      }
      // Sort GPUs by PCIe address then use that as mapping
      std::sort(mapping.begin(), mapping.end());
      for (int i = 0; i < numGpuDevices; ++i)
        remapping[i] = mapping[i].second;
    }
  }
  return remapping[origIdx];
}

void DisplayTopology()
{
  int numGpuDevices;
  HIP_CALL(hipGetDeviceCount(&numGpuDevices));
  printf("\nDetected topology: %d CPU NUMA node(s)   %d GPU device(s)\n", numa_num_configured_nodes(), numGpuDevices);
  printf("        |");
  for (int j = 0; j < numGpuDevices; j++)
    printf(" GPU %02d |", j);
  printf(" PCIe Bus ID  | Closest NUMA\n");
  for (int j = 0; j <= numGpuDevices; j++)
    printf("--------+");
  printf("--------------+-------------\n");

  char pciBusId[20];
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
        HIP_CALL(hipExtGetLinkTypeAndHopCount(RemappedIndex(i, MEM_GPU),
                                              RemappedIndex(j, MEM_GPU),
                                              &linkType, &hopCount));
        printf(" %s-%d |",
               linkType == HSA_AMD_LINK_INFO_TYPE_HYPERTRANSPORT ? "  HT" :
               linkType == HSA_AMD_LINK_INFO_TYPE_QPI            ? " QPI" :
               linkType == HSA_AMD_LINK_INFO_TYPE_PCIE           ? "PCIE" :
               linkType == HSA_AMD_LINK_INFO_TYPE_INFINBAND      ? "INFB" :
               linkType == HSA_AMD_LINK_INFO_TYPE_XGMI           ? "XGMI" : "????",
               hopCount);
      }
    }
    HIP_CALL(hipDeviceGetPCIBusId(pciBusId, 20, RemappedIndex(i, MEM_GPU)));
    printf(" %11s |  %d  \n", pciBusId, GetClosestNumaNode(RemappedIndex(i, MEM_GPU)));
  }
}

void PopulateTestSizes(size_t const numBytesPerLink,
                       int const samplingFactor,
                       std::vector<size_t>& valuesOfN)
{
  valuesOfN.clear();

  // If the number of bytes is specified, use it
  if (numBytesPerLink != 0)
  {
    if (numBytesPerLink % 4)
    {
      printf("[ERROR] numBytesPerLink (%lu) must be a multiple of 4\n", numBytesPerLink);
      exit(1);
    }
    size_t N = numBytesPerLink / sizeof(float);
    valuesOfN.push_back(N);
  }
  else
  {
    // Otherwise generate a range of values
    // (Powers of 2, with samplingFactor samples between successive powers of 2)
    for (int N = 256; N <= (1<<27); N *= 2)
    {
      int delta = std::max(32, N / samplingFactor);
      int curr = N;
      while (curr < N * 2)
      {
        valuesOfN.push_back(curr);
        curr += delta;
      }
    }
  }
}

void ParseMemType(std::string const& token, int const numCpus, int const numGpus, MemType* memType, int* memIndex)
{
  char typeChar;
  if (sscanf(token.c_str(), " %c %d", &typeChar, memIndex) != 2)
  {
    printf("[ERROR] Unable to parse memory type token %s - expecting either 'C' or 'G' or 'F' followed by an index\n", token.c_str());
    exit(1);
  }

  switch (typeChar)
  {
  case 'C': case 'c':
    *memType = MEM_CPU;
    if (*memIndex < 0 || *memIndex >= numCpus)
    {
      printf("[ERROR] CPU index must be between 0 and %d (instead of %d)\n", numCpus-1, *memIndex);
      exit(1);
    }
    break;
  case 'G': case 'g':
    *memType = MEM_GPU;
    if (*memIndex < 0 || *memIndex >= numGpus)
    {
      printf("[ERROR] GPU index must be between 0 and %d (instead of %d)\n", numGpus-1, *memIndex);
      exit(1);
    }
    break;
  case 'F': case 'f':
    *memType = MEM_GPU_FINE;
    if (*memIndex < 0 || *memIndex >= numGpus)
    {
      printf("[ERROR] GPU index must be between 0 and %d (instead of %d)\n", numGpus-1, *memIndex);
      exit(1);
    }
    break;
  default:
    printf("[ERROR] Unrecognized memory type %s.  Expecting either 'C' or 'G' or 'F'\n", token.c_str());
    exit(1);
  }
}

// Helper function to parse a list of link definitions
void ParseLinks(char* line, int numCpus, int numGpus, std::vector<Link>& links)
{
  // Replace any round brackets or '->' with spaces,
  for (int i = 1; line[i]; i++)
    if (line[i] == '(' || line[i] == ')' || line[i] == '-' || line[i] == '>' ) line[i] = ' ';

  links.clear();
  int numLinks = 0;

  std::istringstream iss;
  iss.clear();
  iss.str(line);
  iss >> numLinks;
  if (iss.fail()) return;

  std::string exeMem;
  std::string srcMem;
  std::string dstMem;
  if (numLinks > 0)
  {
    // Method 1: Take in triples (srcMem, exeMem, dstMem)
    int numBlocksToUse;
    iss >> numBlocksToUse;
    if (numBlocksToUse <= 0 || iss.fail())
    {
      printf("Parsing error: Number of blocks to use (%d) must be greater than 0\n", numBlocksToUse);
      exit(1);
    }
    links.resize(numLinks);
    for (int i = 0; i < numLinks; i++)
    {
      iss >> srcMem >> exeMem >> dstMem;
      if (iss.fail())
      {
        printf("Parsing error: Unable to read valid Link triplet (possibly missing a SRC or EXE or DST)\n");
        exit(1);
      }
      ParseMemType(srcMem, numCpus, numGpus, &links[i].srcMemType, &links[i].srcIndex);
      ParseMemType(exeMem, numCpus, numGpus, &links[i].exeMemType, &links[i].exeIndex);
      ParseMemType(dstMem, numCpus, numGpus, &links[i].dstMemType, &links[i].dstIndex);
      links[i].numBlocksToUse = numBlocksToUse;
      if (links[i].exeMemType != MEM_CPU && links[i].exeMemType != MEM_GPU)
      {
        printf("[ERROR] Executor must either be CPU ('C') or GPU ('G'), (from (%s->%s->%s %d))\n",
               srcMem.c_str(), exeMem.c_str(), dstMem.c_str(), links[i].numBlocksToUse);
        exit(1);
      }
    }
  }
  else
  {
    // Method 2: Read in quads (srcMem, exeMem, dstMem,  Read common # blocks to use, then read (src, dst) doubles
    numLinks *= -1;
    links.resize(numLinks);

    for (int i = 0; i < numLinks; i++)
    {
      iss >> srcMem >> exeMem >> dstMem >> links[i].numBlocksToUse;
      if (iss.fail())
      {
        printf("Parsing error: Unable to read valid Link quadruple (possibly missing a SRC or EXE or DST or #CU)\n");
        exit(1);
      }
      ParseMemType(srcMem, numCpus, numGpus, &links[i].srcMemType, &links[i].srcIndex);
      ParseMemType(exeMem, numCpus, numGpus, &links[i].exeMemType, &links[i].exeIndex);
      ParseMemType(dstMem, numCpus, numGpus, &links[i].dstMemType, &links[i].dstIndex);
      if (links[i].exeMemType != MEM_CPU && links[i].exeMemType != MEM_GPU)
      {
        printf("[ERROR] Executor must either be CPU ('C') or GPU ('G'), (from (%s->%s->%s %d))\n"
,               srcMem.c_str(), exeMem.c_str(), dstMem.c_str(), links[i].numBlocksToUse);
        exit(1);
      }
    }
  }
}

void EnablePeerAccess(int const deviceId, int const peerDeviceId)
{
  int canAccess;
  HIP_CALL(hipDeviceCanAccessPeer(&canAccess, deviceId, peerDeviceId));
  if (!canAccess)
  {
    printf("[ERROR] Unable to enable peer access from GPU devices %d to %d\n", peerDeviceId, deviceId);
    exit(1);
  }
  HIP_CALL(hipSetDevice(deviceId));
  HIP_CALL(hipDeviceEnablePeerAccess(peerDeviceId, 0));
}

void AllocateMemory(MemType memType, int devIndex, size_t numBytes, float** memPtr)
{
  if (numBytes == 0)
  {
    printf("[ERROR] Unable to allocate 0 bytes\n");
    exit(1);
  }

  if (memType == MEM_CPU)
  {
    // Set numa policy prior to call to hipHostMalloc
    // NOTE: It may be possible that the actual configured numa nodes do not start at 0
    //       so remapping may be necessary
    // Find the 'deviceId'-th available NUMA node
    int numaIdx = 0;
    for (int i = 0; i <= devIndex; i++)
      while (!numa_bitmask_isbitset(numa_get_mems_allowed(), numaIdx))
        ++numaIdx;

    unsigned long nodemask = (1ULL << numaIdx);
    long retCode = set_mempolicy(MPOL_BIND, &nodemask, sizeof(nodemask)*8);
    if (retCode)
    {
      printf("[ERROR] Unable to set NUMA memory policy to bind to NUMA node %d\n", numaIdx);
      exit(1);
    }

    // Allocate host-pinned memory (should respect NUMA mem policy)
    HIP_CALL(hipHostMalloc((void **)memPtr, numBytes, hipHostMallocNumaUser | hipHostMallocNonCoherent));

    // Check that the allocated pages are actually on the correct NUMA node
    CheckPages((char*)*memPtr, numBytes, numaIdx);

    // Reset to default numa mem policy
    retCode = set_mempolicy(MPOL_DEFAULT, NULL, 8);
    if (retCode)
    {
      printf("[ERROR] Unable reset to default NUMA memory policy\n");
      exit(1);
    }
  }
  else if (memType == MEM_GPU)
  {
    // Allocate GPU memory on appropriate device
    HIP_CALL(hipSetDevice(devIndex));
    HIP_CALL(hipMalloc((void**)memPtr, numBytes));
  }
  else if (memType == MEM_GPU_FINE)
  {
    HIP_CALL(hipSetDevice(devIndex));
    HIP_CALL(hipExtMallocWithFlags((void**)memPtr, numBytes, hipDeviceMallocFinegrained));
  }
  else
  {
    printf("[ERROR] Unsupported memory type %d\n", memType);
    exit(1);
  }
}

void DeallocateMemory(MemType memType, float* memPtr)
{
  if (memType == MEM_CPU)
  {
    HIP_CALL(hipHostFree(memPtr));
  }
  else if (memType == MEM_GPU || memType == MEM_GPU_FINE)
  {
    HIP_CALL(hipFree(memPtr));
  }
}

void CheckPages(char* array, size_t numBytes, int targetId)
{
  unsigned long const pageSize = getpagesize();
  unsigned long const numPages = (numBytes + pageSize - 1) / pageSize;

  std::vector<void *> pages(numPages);
  std::vector<int> status(numPages);

  pages[0] = array;
  for (int i = 1; i < numPages; i++)
  {
    pages[i] = (char*)pages[i-1] + pageSize;
  }

  long const retCode = move_pages(0, numPages, pages.data(), NULL, status.data(), 0);
  if (retCode)
  {
    printf("[ERROR] Unable to collect page info\n");
    exit(1);
  }

  size_t mistakeCount = 0;
  for (int i = 0; i < numPages; i++)
  {
    if (status[i] < 0)
    {
      printf("[ERROR] Unexpected page status %d for page %d\n", status[i], i);
      exit(1);
    }
    if (status[i] != targetId) mistakeCount++;
  }
  if (mistakeCount > 0)
  {
    printf("[ERROR] %lu out of %lu pages for memory allocation were not on NUMA node %d\n", mistakeCount, numPages, targetId);
    printf("[ERROR] Ensure up-to-date ROCm is installed\n");
    exit(1);
  }
}

// Helper function to either fill a device pointer with pseudo-random data, or to check to see if it matches
void CheckOrFill(ModeType mode, int N, bool isMemset, bool isHipCall, std::vector<float>const& fillPattern, float* ptr)
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
    // Fill with repeated pattern if specified
    size_t patternLen = fillPattern.size();
    if (patternLen > 0)
    {
      for (int i = 0; i < N; i++)
        refBuffer[i] = fillPattern[i % patternLen];
    }
    else // Otherwise fill with pseudo-random values
    {
      for (int i = 0; i < N; i++)
        refBuffer[i] = (i % 383 + 31);
    }
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
    free(hostBuffer);
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

std::string GetDesc(MemType srcMemType, int srcIndex,
                    MemType dstMemType, int dstIndex)
{
  if (srcMemType == MEM_CPU)
  {
    if (dstMemType == MEM_CPU)
      return (srcIndex == dstIndex) ? "LOCAL" : "NUMA";
    else if (dstMemType == MEM_GPU || dstMemType == MEM_GPU_FINE)
      return "PCIE";
    else
      goto error;
  }
  else if (srcMemType == MEM_GPU || srcMemType == MEM_GPU_FINE)
  {
    if (dstMemType == MEM_CPU)
      return "PCIE";
    else if (dstMemType == MEM_GPU || dstMemType == MEM_GPU_FINE)
    {
      if (srcIndex == dstIndex) return "LOCAL";
      else
      {
        uint32_t linkType, hopCount;
        HIP_CALL(hipExtGetLinkTypeAndHopCount(RemappedIndex(srcIndex, MEM_GPU),
                                              RemappedIndex(dstIndex, MEM_GPU),
                                              &linkType, &hopCount));
        return GetLinkTypeDesc(linkType, hopCount);
      }
    }
    else
      goto error;
  }
error:
  printf("[ERROR] Unrecognized memory type\n");
  exit(1);
}

std::string GetLinkDesc(Link const& link)
{
  return GetDesc(link.srcMemType, link.srcIndex, link.exeMemType, link.exeIndex) + "-"
    + GetDesc(link.exeMemType, link.exeIndex, link.dstMemType, link.dstIndex);
}

void RunLink(EnvVars const& ev, size_t const N, int const iteration, Link& link)
{
  // GPU execution agent
  if (link.exeMemType == MEM_GPU)
  {
    // Switch to executing GPU
    HIP_CALL(hipSetDevice(RemappedIndex(link.exeIndex, MEM_GPU)));

    bool recordStart = (!ev.useSingleSync || iteration == 0);
    bool recordStop  = (!ev.useSingleSync || iteration == ev.numIterations - 1);

    int const initOffset = ev.byteOffset / sizeof(float);

    if (ev.useHipCall)
    {
      // Record start event
      if (recordStart) HIP_CALL(hipEventRecord(link.startEvent, link.stream));

      // Execute hipMemset / hipMemcpy
      if (ev.useMemset)
        HIP_CALL(hipMemsetAsync(link.dstMem + initOffset, 42, N * sizeof(float), link.stream));
      else
        HIP_CALL(hipMemcpyAsync(link.dstMem + initOffset,
                                link.srcMem + initOffset,
                                N * sizeof(float), hipMemcpyDefault,
                                link.stream));
      // Record stop event
      if (recordStop) HIP_CALL(hipEventRecord(link.stopEvent, link.stream));
    }
    else
    {
      if (!ev.combineTiming && recordStart) HIP_CALL(hipEventRecord(link.startEvent, link.stream));
      hipExtLaunchKernelGGL(ev.useMemset ? GpuMemsetKernel : GpuCopyKernel,
                            dim3(link.numBlocksToUse, 1, 1),
                            dim3(BLOCKSIZE, 1, 1),
                            ev.sharedMemBytes, link.stream,
                            (ev.combineTiming && recordStart) ? link.startEvent : NULL,
                            (ev.combineTiming && recordStop)  ? link.stopEvent : NULL,
                            0, link.blockParam);
      if (!ev.combineTiming & recordStop) HIP_CALL(hipEventRecord(link.stopEvent, link.stream));
    }

    // Synchronize per iteration, unless in single sync mode, in which case
    // synchronize during last warmup / last actual iteration
    if (!ev.useSingleSync || iteration == -1 || iteration == ev.numIterations - 1)
    {
      HIP_CALL(hipStreamSynchronize(link.stream));
    }

    if (iteration >= 0)
    {
      // Record GPU timing
      if (!ev.useSingleSync || iteration == ev.numIterations - 1)
      {
        HIP_CALL(hipEventSynchronize(link.stopEvent));
        float gpuDeltaMsec;
        HIP_CALL(hipEventElapsedTime(&gpuDeltaMsec, link.startEvent, link.stopEvent));
        link.totalTime += gpuDeltaMsec;
      }
    }
  }
  else if (link.exeMemType == MEM_CPU) // CPU execution agent
  {
    // Force this thread and all child threads onto correct NUMA node
    if (numa_run_on_node(link.exeIndex))
    {
      printf("[ERROR] Unable to set CPU to NUMA node %d\n", link.exeIndex);
      exit(1);
    }

    std::vector<std::thread> childThreads;

    auto cpuStart = std::chrono::high_resolution_clock::now();

    // Launch child-threads to perform memcopies
    for (int i = 0; i < ev.numCpuPerLink; i++)
      childThreads.push_back(std::thread(ev.useMemset ? CpuMemsetKernel : CpuCopyKernel, std::ref(link.blockParam[i])));

    // Wait for child-threads to finish
    for (int i = 0; i < ev.numCpuPerLink; i++)
      childThreads[i].join();

    auto cpuDelta = std::chrono::high_resolution_clock::now() - cpuStart;

    // Record time if not a warmup iteration
    if (iteration >= 0)
      link.totalTime += (std::chrono::duration_cast<std::chrono::duration<double>>(cpuDelta).count() * 1000.0);
  }
}

void RunPeerToPeerBenchmarks(EnvVars const& ev, size_t N, int numBlocksToUse, int readMode, int skipCpu)
{
  // Collect the number of available CPUs/GPUs on this machine
  int numGpus;
  HIP_CALL(hipGetDeviceCount(&numGpus));
  int const numCpus = numa_num_configured_nodes();
  int const numDevices = numCpus + numGpus;

  // Enable peer to peer for each GPU
  for (int i = 0; i < numGpus; i++)
    for (int j = 0; j < numGpus; j++)
      if (i != j) EnablePeerAccess(i, j);

  printf("Performing copies in each direction of %lu bytes\n", N * sizeof(float));
  printf("Using %d threads per NUMA node for CPU copies\n", ev.numCpuPerLink);
  printf("Using %d CUs per transfer\n", numBlocksToUse);

  // Perform unidirectional / bidirectional
  for (int isBidirectional = 0; isBidirectional <= 1; isBidirectional++)
  {
    // Print header
    printf("%sdirectional copy peak bandwidth GB/s [%s read / %s write]\n", isBidirectional ? "Bi" : "Uni",
           readMode == 0 ? "Local" : "Remote",
           readMode == 0 ? "Remote" : "Local");
    printf("%10s", "D/D");
    if (!skipCpu)
    {
      for (int i = 0; i < numCpus; i++)
        printf("%7s %02d", "CPU", i);
    }
    for (int i = 0; i < numGpus; i++)
      printf("%7s %02d", "GPU", i);
    printf("\n");

    // Loop over all possible src/dst pairs
    for (int src = 0; src < numDevices; src++)
    {
      MemType const& srcMemType = (src < numCpus ? MEM_CPU : MEM_GPU);
      if (skipCpu && srcMemType == MEM_CPU) continue;
      int srcIndex = (srcMemType == MEM_CPU ? src : src - numCpus);
      printf("%7s %02d", (srcMemType == MEM_CPU) ? "CPU" : "GPU", srcIndex);
      for (int dst = 0; dst < numDevices; dst++)
      {
        MemType const& dstMemType = (dst < numCpus ? MEM_CPU : MEM_GPU);
        if (skipCpu && dstMemType == MEM_CPU) continue;
        int dstIndex = (dstMemType == MEM_CPU ? dst : dst - numCpus);
        double bandwidth = GetPeakBandwidth(ev, N, isBidirectional, srcMemType, srcIndex, dstMemType, dstIndex, readMode);
        if (bandwidth == 0)
          printf("%10s", "N/A");
        else
          printf("%10.2f", bandwidth);
        fflush(stdout);
      }
      printf("\n");
    }
    printf("\n");
  }
}

double GetPeakBandwidth(EnvVars const& ev, size_t N, int isBidirectional,
                        MemType srcMemType, int srcIndex,
                        MemType dstMemType, int dstIndex,
                        int readMode)
{
  Link links[2];
  int const initOffset = ev.byteOffset / sizeof(float);

  // Skip bidirectional on same device
  if (isBidirectional && srcMemType == dstMemType && srcIndex == dstIndex) return 0.0f;

  // Prepare Links
  links[0].srcMemType = links[1].dstMemType = srcMemType;
  links[0].srcIndex   = links[1].dstIndex   = RemappedIndex(srcIndex, srcMemType);
  links[0].dstMemType = links[1].srcMemType = dstMemType;
  links[0].dstIndex   = links[1].srcIndex   = RemappedIndex(dstIndex, dstMemType);
  // Either perform local read / remote write, or remote read / local write
  links[0].exeMemType = (readMode == 0 ? srcMemType : dstMemType);
  links[0].exeIndex   = RemappedIndex((readMode == 0 ? srcIndex   : dstIndex), links[0].exeMemType);
  links[1].exeMemType = (readMode == 0 ? dstMemType : srcMemType);
  links[1].exeIndex   = RemappedIndex((readMode == 0 ? dstIndex   : srcIndex), links[1].exeMemType);

  for (int i = 0; i <= isBidirectional; i++)
  {
    AllocateMemory(links[i].srcMemType, links[i].srcIndex, N * sizeof(float) + ev.byteOffset, &links[i].srcMem);
    AllocateMemory(links[i].dstMemType, links[i].dstIndex, N * sizeof(float) + ev.byteOffset, &links[i].dstMem);
    links[i].totalTime = 0.0;

    CheckOrFill(MODE_FILL, N, ev.useMemset, ev.useHipCall, ev.fillPattern, links[i].srcMem + initOffset);
    if (links[i].exeMemType == MEM_GPU)
    {
      HIP_CALL(hipDeviceGetAttribute(&links[i].numBlocksToUse, hipDeviceAttributeMultiprocessorCount, links[i].exeIndex));
      HIP_CALL(hipSetDevice(links[i].exeIndex));
      HIP_CALL(hipEventCreate(&links[i].startEvent));
      HIP_CALL(hipEventCreate(&links[i].stopEvent));
      HIP_CALL(hipMalloc((void**)&links[i].blockParam, sizeof(BlockParam) * links[i].numBlocksToUse));
      HIP_CALL(hipStreamCreate(&links[i].stream));

      size_t assigned = 0;
      int maxNumBlocksToUse = std::min((N + 31) / 32, (size_t)links[i].numBlocksToUse);
      for (int j = 0; j < links[i].numBlocksToUse; j++)
      {
        BlockParam param;
        int blocksLeft = std::max(0, maxNumBlocksToUse - j);
        size_t leftover = N - assigned;
        size_t roundedN = (leftover + 31) / 32;
        param.N = blocksLeft ? std::min(leftover, ((roundedN / blocksLeft) * 32)) : 0;
        param.src = links[i].srcMem + assigned + initOffset;
        param.dst = links[i].dstMem + assigned + initOffset;
        assigned += param.N;

        HIP_CALL(hipMemcpy(&links[i].blockParam[j], &param, sizeof(BlockParam), hipMemcpyHostToDevice));
      }
    }
    else
    {
      links[i].blockParam = (BlockParam*)malloc(ev.numCpuPerLink * sizeof(BlockParam));
      // For CPU-based copy, divded based on the number of child threads
      size_t assigned = 0;
      int maxNumBlocksToUse = std::min((N + 31) / 32, (size_t)ev.numCpuPerLink);
      for (int j = 0; j < ev.numCpuPerLink; j++)
      {
        int blocksLeft = std::max(0, maxNumBlocksToUse - j);
        size_t leftover = N - assigned;
        size_t roundedN = (leftover + 31) / 32;
        links[i].blockParam[j].N = blocksLeft ? std::min(leftover, ((roundedN / blocksLeft) * 32)) : 0;
        links[i].blockParam[j].src = links[i].srcMem + assigned + initOffset;
        links[i].blockParam[j].dst = links[i].dstMem + assigned + initOffset;
        assigned += links[i].blockParam[j].N;
      }
    }
  }

  std::stack<std::thread> threads;

  // Perform iteration
  for (int iteration = -ev.numWarmups; iteration < ev.numIterations; iteration++)
  {
    // Perform timed iterations
    for (int i = 0; i <= isBidirectional; i++)
      threads.push(std::thread(RunLink, std::ref(ev), N, iteration, std::ref(links[i])));

    // Wait for all threads to finish
    for (int i = 0; i <= isBidirectional; i++)
    {
      threads.top().join();
      threads.pop();
    }
  }

  // Validate that each link has transferred correctly
  for (int i = 0; i <= isBidirectional; i++)
    CheckOrFill(MODE_CHECK, N, ev.useMemset, ev.useHipCall, ev.fillPattern, links[i].dstMem + initOffset);

  // Collect aggregate bandwidth
  double totalBandwidth = 0;
  for (int i = 0; i <= isBidirectional; i++)
  {
    double linkDurationMsec = links[i].totalTime / (1.0 * ev.numIterations);
    double linkBandwidthGbs = (N * sizeof(float) / 1.0E9) / linkDurationMsec * 1000.0f;
    totalBandwidth += linkBandwidthGbs;
  }

  // Release GPU memory
  for (int i = 0; i <= isBidirectional; i++)
  {
    DeallocateMemory(links[i].srcMemType, links[i].srcMem);
    DeallocateMemory(links[i].dstMemType, links[i].dstMem);

    if (links[i].exeMemType == MEM_GPU)
      {
        HIP_CALL(hipEventDestroy(links[i].startEvent));
        HIP_CALL(hipEventDestroy(links[i].stopEvent));
        HIP_CALL(hipStreamDestroy(links[i].stream));
        HIP_CALL(hipFree(links[i].blockParam));
      }
      else if (links[i].exeMemType == MEM_CPU)
      {
        free(links[i].blockParam);
      }
  }
  return totalBandwidth;
}
