/*
Copyright (c) 2019-2022 Advanced Micro Devices, Inc. All rights reserved.

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

int main(int argc, char **argv)
{
  // Display usage instructions and detected topology
  if (argc <= 1)
  {
    int const outputToCsv = EnvVars::GetEnvVar("OUTPUT_TO_CSV", 0);
    if (!outputToCsv) DisplayUsage(argv[0]);
    DisplayTopology(outputToCsv);
    exit(0);
  }

  // Collect environment variables / display current run configuration
  EnvVars ev;

  // Determine number of bytes to run per Transfer
  // If a non-zero number of bytes is specified, use it
  // Otherwise generate array of bytes values to execute over
  std::vector<size_t> valuesOfN;
  size_t numBytesPerTransfer = argc > 2 ? atoll(argv[2]) : DEFAULT_BYTES_PER_TRANSFER;
  if (argc > 2)
  {
    // Adjust bytes if unit specified
    char units = argv[2][strlen(argv[2])-1];
    switch (units)
    {
    case 'K': case 'k': numBytesPerTransfer *= 1024; break;
    case 'M': case 'm': numBytesPerTransfer *= 1024*1024; break;
    case 'G': case 'g': numBytesPerTransfer *= 1024*1024*1024; break;
    }
  }
  PopulateTestSizes(numBytesPerTransfer, ev.samplingFactor, valuesOfN);

  // Find the largest N to be used - memory will only be allocated once per set of simulatenous Transfers
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
    int skipCpu  = (!strcmp(argv[1], "g2g"   ) || !strcmp(argv[1], "g2g_rr") ? 1 : 0);

    // Execute peer to peer benchmark mode
    RunPeerToPeerBenchmarks(ev, numBytesPerTransfer / sizeof(float), numBlocksToUse, readMode, skipCpu);
    exit(0);
  }

  // Check that Transfer configuration file can be opened
  FILE* fp = fopen(argv[1], "r");
  if (!fp)
  {
    printf("[ERROR] Unable to open transfer configuration file: [%s]\n", argv[1]);
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

  // Track unique pair of transfers that get used
  std::set<std::pair<int, int>> peerAccessTracker;

  // Print CSV header
  if (ev.outputToCsv)
  {
    printf("Test,NumBytes,SrcMem,Executor,DstMem,CUs,BW(GB/s),Time(ms),"
           "TransferDesc,SrcAddr,DstAddr,ByteOffset,numWarmups,numIters\n");
  }

  // Loop over each line in the Transfer configuration file
  int testNum = 0;
  char line[2048];
  while(fgets(line, 2048, fp))
  {
    // Check if line is a comment to be echoed to output (starts with ##)
    if (!ev.outputToCsv && line[0] == '#' && line[1] == '#') printf("%s", line);

    // Parse transfers from configuration file
    TransferMap transferMap;
    ParseTransfers(line, numCpuDevices, numGpuDevices, transferMap);
    if (transferMap.size() == 0) continue;

    testNum++;

    // Prepare (maximum) memory for each transfer
    std::vector<Transfer*> transferList;
    for (auto& exeInfoPair : transferMap)
    {
      ExecutorInfo& exeInfo = exeInfoPair.second;
      exeInfo.totalTime = 0.0;
      exeInfo.totalBlocks = 0;

      for (Transfer& transfer : exeInfo.transfers)
      {
        // Get some aliases to transfer variables
        MemType const& exeMemType  = transfer.exeMemType;
        MemType const& srcMemType  = transfer.srcMemType;
        MemType const& dstMemType  = transfer.dstMemType;
        int     const& blocksToUse = transfer.numBlocksToUse;

        // Get potentially remapped device indices
        int const srcIndex = RemappedIndex(transfer.srcIndex, srcMemType);
        int const exeIndex = RemappedIndex(transfer.exeIndex, exeMemType);
        int const dstIndex = RemappedIndex(transfer.dstIndex, dstMemType);

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
        AllocateMemory(srcMemType, srcIndex, maxN * sizeof(float) + ev.byteOffset, (void**)&transfer.srcMem);
        AllocateMemory(dstMemType, dstIndex, maxN * sizeof(float) + ev.byteOffset, (void**)&transfer.dstMem);
        transfer.blockParam.resize(exeMemType == MEM_CPU ? ev.numCpuPerTransfer : blocksToUse);
        exeInfo.totalBlocks += transfer.blockParam.size();
        transferList.push_back(&transfer);
      }

      // Prepare GPU resources for GPU executors
      MemType const exeMemType = exeInfoPair.first.first;
      int     const exeIndex   = RemappedIndex(exeInfoPair.first.second, exeMemType);
      if (exeMemType == MEM_GPU)
      {
        AllocateMemory(exeMemType, exeIndex, exeInfo.totalBlocks * sizeof(BlockParam),
                       (void**)&exeInfo.blockParamGpu);

        int const numTransfersToRun = ev.useSingleStream ? 1 : exeInfo.transfers.size();
        exeInfo.streams.resize(numTransfersToRun);
        exeInfo.startEvents.resize(numTransfersToRun);
        exeInfo.stopEvents.resize(numTransfersToRun);
        for (int i = 0; i < numTransfersToRun; ++i)
        {
          HIP_CALL(hipSetDevice(exeIndex));
          HIP_CALL(hipStreamCreate(&exeInfo.streams[i]));
          HIP_CALL(hipEventCreate(&exeInfo.startEvents[i]));
          HIP_CALL(hipEventCreate(&exeInfo.stopEvents[i]));
        }

        int transferOffset = 0;
        for (int i = 0; i < exeInfo.transfers.size(); i++)
        {
          exeInfo.transfers[i].blockParamGpuPtr = exeInfo.blockParamGpu + transferOffset;
          transferOffset += exeInfo.transfers[i].blockParam.size();
        }
      }
    }

    // Loop over all the different number of bytes to use per Transfer
    for (auto N : valuesOfN)
    {
      if (!ev.outputToCsv) printf("Test %d: [%lu bytes]\n", testNum, N * sizeof(float));

      // Prepare input memory and block parameters for current N
      for (auto& exeInfoPair : transferMap)
      {
        ExecutorInfo& exeInfo = exeInfoPair.second;

        int transferOffset = 0;

        for (int i = 0; i < exeInfo.transfers.size(); ++i)
        {
          Transfer& transfer = exeInfo.transfers[i];
          transfer.PrepareBlockParams(ev, N);

          // Copy block parameters to GPU for GPU executors
          if (transfer.exeMemType == MEM_GPU)
          {
            HIP_CALL(hipMemcpy(&exeInfo.blockParamGpu[transferOffset],
                               transfer.blockParam.data(),
                               transfer.blockParam.size() * sizeof(BlockParam),
                               hipMemcpyHostToDevice));
            transferOffset += transfer.blockParam.size();
          }
        }
      }

      // Launch kernels (warmup iterations are not counted)
      double totalCpuTime = 0;
      size_t numTimedIterations = 0;
      for (int iteration = -ev.numWarmups; ; iteration++)
      {
        if (ev.numIterations > 0 && iteration >= ev.numIterations) break;
        if (ev.numIterations < 0 && totalCpuTime > -ev.numIterations) break;

        // Pause before starting first timed iteration in interactive mode
        if (ev.useInteractive && iteration == 0)
        {
          printf("Hit <Enter> to continue: ");
          scanf("%*c");
          printf("\n");
        }

        // Start CPU timing for this iteration
        auto cpuStart = std::chrono::high_resolution_clock::now();

        // Execute all Transfers in parallel
        for (auto& exeInfoPair : transferMap)
        {
          ExecutorInfo& exeInfo = exeInfoPair.second;
          int const numTransfersToRun = ev.useSingleStream ? 1 : exeInfo.transfers.size();
          for (int i = 0; i < numTransfersToRun; ++i)
            threads.push(std::thread(RunTransfer, std::ref(ev), N, iteration, std::ref(exeInfo), i));
        }

        // Wait for all threads to finish
        int const numTransfers = threads.size();
        for (int i = 0; i < numTransfers; i++)
        {
          threads.top().join();
          threads.pop();
        }

        // Stop CPU timing for this iteration
        auto cpuDelta = std::chrono::high_resolution_clock::now() - cpuStart;
        double deltaSec = std::chrono::duration_cast<std::chrono::duration<double>>(cpuDelta).count();

        if (iteration >= 0)
        {
          ++numTimedIterations;
          totalCpuTime += deltaSec;
        }
      }

      // Pause for interactive mode
      if (ev.useInteractive)
      {
        printf("Transfers complete. Hit <Enter> to continue: ");
        scanf("%*c");
        printf("\n");
      }

      // Validate that each transfer has transferred correctly
      int const numTransfers = transferList.size();
      for (auto transfer : transferList)
        CheckOrFill(MODE_CHECK, N, ev.useMemset, ev.useHipCall, ev.fillPattern, transfer->dstMem + initOffset);

      // Report timings
      totalCpuTime = totalCpuTime / (1.0 * numTimedIterations) * 1000;
      double totalBandwidthGbs = (numTransfers * N * sizeof(float) / 1.0E6) / totalCpuTime;
      double maxGpuTime = 0;

      if (ev.useSingleStream)
      {
        for (auto& exeInfoPair : transferMap)
        {
          ExecutorInfo const& exeInfo = exeInfoPair.second;
          MemType const exeMemType    = exeInfoPair.first.first;
          int     const exeIndex      = exeInfoPair.first.second;

          double exeDurationMsec = exeInfo.totalTime / (1.0 * numTimedIterations);
          double exeBandwidthGbs = (exeInfo.transfers.size() * N * sizeof(float) / 1.0E9) / exeDurationMsec * 1000.0f;
          maxGpuTime = std::max(maxGpuTime, exeDurationMsec);

          if (!ev.outputToCsv)
          {
            printf(" Executor: %cPU %02d        (# Transfers %02lu)| %9.3f GB/s | %8.3f ms |\n",
                   MemTypeStr[exeMemType], exeIndex, exeInfo.transfers.size(), exeBandwidthGbs, exeDurationMsec);
            for (auto transfer : exeInfo.transfers)
            {
              double transferDurationMsec = transfer.transferTime / (1.0 * numTimedIterations);
              double transferBandwidthGbs = (N * sizeof(float) / 1.0E9) / transferDurationMsec * 1000.0f;

              printf("                            Transfer  %02d | %9.3f GB/s | %8.3f ms | %c%02d -> %c%02d:(%03d) -> %c%02d\n",
                     transfer.transferIndex,
                     transferBandwidthGbs,
                     transferDurationMsec,
                     MemTypeStr[transfer.srcMemType], transfer.srcIndex,
                     MemTypeStr[transfer.exeMemType], transfer.exeIndex,
                     transfer.exeMemType == MEM_CPU ? ev.numCpuPerTransfer : transfer.numBlocksToUse,
                     MemTypeStr[transfer.dstMemType], transfer.dstIndex);
            }
          }
          else
          {
            printf("%d,%lu,ALL,%c%02d,ALL,ALL,%.3f,%.3f,ALL,ALL,ALL,%d,%d,%lu\n",
                   testNum, N * sizeof(float),
                   MemTypeStr[exeMemType], exeIndex,
                   exeBandwidthGbs, exeDurationMsec,
                   ev.byteOffset,
                   ev.numWarmups, numTimedIterations);
          }
        }
      }
      else
      {
        for (auto transfer : transferList)
        {
          double transferDurationMsec = transfer->transferTime / (1.0 * numTimedIterations);
          double transferBandwidthGbs = (N * sizeof(float) / 1.0E9) / transferDurationMsec * 1000.0f;
          maxGpuTime = std::max(maxGpuTime, transferDurationMsec);
          if (!ev.outputToCsv)
          {
            printf(" Transfer %02d: %c%02d -> [%cPU %02d:%03d] -> %c%02d | %9.3f GB/s | %8.3f ms | %-16s\n",
                   transfer->transferIndex,
                   MemTypeStr[transfer->srcMemType], transfer->srcIndex,
                   MemTypeStr[transfer->exeMemType], transfer->exeIndex,
                   transfer->exeMemType == MEM_CPU ? ev.numCpuPerTransfer : transfer->numBlocksToUse,
                   MemTypeStr[transfer->dstMemType], transfer->dstIndex,
                   transferBandwidthGbs, transferDurationMsec,
                   GetTransferDesc(*transfer).c_str());
          }
          else
          {
            printf("%d,%lu,%c%02d,%c%02d,%c%02d,%d,%.3f,%.3f,%s,%p,%p,%d,%d,%lu\n",
                   testNum, N * sizeof(float),
                   MemTypeStr[transfer->srcMemType], transfer->srcIndex,
                   MemTypeStr[transfer->exeMemType], transfer->exeIndex,
                   MemTypeStr[transfer->dstMemType], transfer->dstIndex,
                   transfer->exeMemType == MEM_CPU ? ev.numCpuPerTransfer : transfer->numBlocksToUse,
                   transferBandwidthGbs, transferDurationMsec,
                   GetTransferDesc(*transfer).c_str(),
                   transfer->srcMem + initOffset, transfer->dstMem + initOffset,
                   ev.byteOffset,
                   ev.numWarmups, numTimedIterations);
          }
        }
      }

      // Display aggregate statistics
      if (!ev.outputToCsv)
      {
        printf(" Aggregate Bandwidth (CPU timed)         | %9.3f GB/s | %8.3f ms | Overhead: %.3f ms\n", totalBandwidthGbs, totalCpuTime,
               totalCpuTime - maxGpuTime);
      }
      else
      {
        printf("%d,%lu,ALL,ALL,ALL,ALL,%.3f,%.3f,ALL,ALL,ALL,%d,%d,%lu\n",
               testNum, N * sizeof(float), totalBandwidthGbs, totalCpuTime, ev.byteOffset,
               ev.numWarmups, numTimedIterations);
      }
    }

    // Release GPU memory
    for (auto exeInfoPair : transferMap)
    {
      ExecutorInfo& exeInfo = exeInfoPair.second;
      for (auto& transfer : exeInfo.transfers)
      {
        // Get some aliases to Transfer variables
        MemType const& exeMemType = transfer.exeMemType;
        MemType const& srcMemType = transfer.srcMemType;
        MemType const& dstMemType = transfer.dstMemType;

        // Allocate (maximum) source / destination memory based on type / device index
        DeallocateMemory(srcMemType, transfer.srcMem);
        DeallocateMemory(dstMemType, transfer.dstMem);
        transfer.blockParam.clear();
      }

      MemType const exeMemType = exeInfoPair.first.first;
      int     const exeIndex   = RemappedIndex(exeInfoPair.first.second, exeMemType);
      if (exeMemType == MEM_GPU)
      {
        DeallocateMemory(exeMemType, exeInfo.blockParamGpu);
        int const numTransfersToRun = ev.useSingleStream ? 1 : exeInfo.transfers.size();
        for (int i = 0; i < numTransfersToRun; ++i)
        {
          HIP_CALL(hipEventDestroy(exeInfo.startEvents[i]));
          HIP_CALL(hipEventDestroy(exeInfo.stopEvents[i]));
          HIP_CALL(hipStreamDestroy(exeInfo.streams[i]));
        }
      }
    }
  }
  fclose(fp);

  return 0;
}

void DisplayUsage(char const* cmdName)
{
  printf("TransferBench v%s\n", TB_VERSION);
  printf("========================================\n");

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
  printf("          - Filename of configFile containing Transfers to execute (see example.cfg for format)\n");
  printf("          - Name of preset benchmark:\n");
  printf("              p2p    - All CPU/GPU pairs benchmark\n");
  printf("              p2p_rr - All CPU/GPU pairs benchmark with remote reads\n");
  printf("              g2g    - All GPU/GPU pairs benchmark\n");
  printf("              g2g_rr - All GPU/GPU pairs benchmark with remote reads\n");
  printf("            - 3rd optional argument will be used as # of CUs to use (uses all by default)\n");
  printf("  N     : (Optional) Number of bytes to copy per Transfer.\n");
  printf("          If not specified, defaults to %lu bytes. Must be a multiple of 4 bytes\n",
         DEFAULT_BYTES_PER_TRANSFER);
  printf("          If 0 is specified, a range of Ns will be benchmarked\n");
  printf("          May append a suffix ('K', 'M', 'G') for kilobytes / megabytes / gigabytes\n");
  printf("\n");

  EnvVars::DisplayUsage();
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

void DisplayTopology(bool const outputToCsv)
{
  int numGpuDevices;
  HIP_CALL(hipGetDeviceCount(&numGpuDevices));

  if (outputToCsv)
  {
    printf("NumCpus,%d\n", numa_num_configured_nodes());
    printf("NumGpus,%d\n", numGpuDevices);
    printf("GPU");
    for (int j = 0; j < numGpuDevices; j++)
      printf(",GPU %02d", j);
    printf(",PCIe Bus ID,ClosestNUMA\n");
  }
  else
  {
    printf("\nDetected topology: %d CPU NUMA node(s)   %d GPU device(s)\n", numa_num_configured_nodes(), numGpuDevices);
    printf("        |");
    for (int j = 0; j < numGpuDevices; j++)
      printf(" GPU %02d |", j);
    printf(" PCIe Bus ID  | Closest NUMA\n");
    for (int j = 0; j <= numGpuDevices; j++)
      printf("--------+");
    printf("--------------+-------------\n");
  }

  char pciBusId[20];

  for (int i = 0; i < numGpuDevices; i++)
  {
    printf("%sGPU %02d%s", outputToCsv ? "" : " ", i, outputToCsv ? "," : " |");
    for (int j = 0; j < numGpuDevices; j++)
    {
      if (i == j)
      {
        if (outputToCsv)
          printf("-,");
        else
          printf("    -   |");
      }
      else
      {
        uint32_t linkType, hopCount;
        HIP_CALL(hipExtGetLinkTypeAndHopCount(RemappedIndex(i, MEM_GPU),
                                              RemappedIndex(j, MEM_GPU),
                                              &linkType, &hopCount));
        printf("%s%s-%d%s",
               outputToCsv ? "" : " ",
               linkType == HSA_AMD_LINK_INFO_TYPE_HYPERTRANSPORT ? "  HT" :
               linkType == HSA_AMD_LINK_INFO_TYPE_QPI            ? " QPI" :
               linkType == HSA_AMD_LINK_INFO_TYPE_PCIE           ? "PCIE" :
               linkType == HSA_AMD_LINK_INFO_TYPE_INFINBAND      ? "INFB" :
               linkType == HSA_AMD_LINK_INFO_TYPE_XGMI           ? "XGMI" : "????",
               hopCount, outputToCsv ? "," : " |");
      }
    }
    HIP_CALL(hipDeviceGetPCIBusId(pciBusId, 20, RemappedIndex(i, MEM_GPU)));
    if (outputToCsv)
      printf("%s,%d\n", pciBusId, GetClosestNumaNode(RemappedIndex(i, MEM_GPU)));
    else
      printf(" %11s |  %d  \n", pciBusId, GetClosestNumaNode(RemappedIndex(i, MEM_GPU)));
  }
}

void PopulateTestSizes(size_t const numBytesPerTransfer,
                       int const samplingFactor,
                       std::vector<size_t>& valuesOfN)
{
  valuesOfN.clear();

  // If the number of bytes is specified, use it
  if (numBytesPerTransfer != 0)
  {
    if (numBytesPerTransfer % 4)
    {
      printf("[ERROR] numBytesPerTransfer (%lu) must be a multiple of 4\n", numBytesPerTransfer);
      exit(1);
    }
    size_t N = numBytesPerTransfer / sizeof(float);
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
    printf("[ERROR] Unable to parse memory type token %s - expecting either 'B,C,G or F' followed by an index\n",
           token.c_str());
    exit(1);
  }

  switch (typeChar)
  {
  case 'C': case 'c': case 'B': case 'b':
    *memType = (typeChar == 'C' || typeChar == 'c') ? MEM_CPU : MEM_CPU_FINE;
    if (*memIndex < 0 || *memIndex >= numCpus)
    {
      printf("[ERROR] CPU index must be between 0 and %d (instead of %d)\n", numCpus-1, *memIndex);
      exit(1);
    }
    break;
  case 'G': case 'g': case 'F': case 'f':
    *memType = (typeChar == 'G' || typeChar == 'g') ? MEM_GPU : MEM_GPU_FINE;
    if (*memIndex < 0 || *memIndex >= numGpus)
    {
      printf("[ERROR] GPU index must be between 0 and %d (instead of %d)\n", numGpus-1, *memIndex);
      exit(1);
    }
    break;
  default:
    printf("[ERROR] Unrecognized memory type %s.  Expecting either 'B', 'C' or 'G' or 'F'\n", token.c_str());
    exit(1);
  }
}

// Helper function to parse a list of Transfer definitions
void ParseTransfers(char* line, int numCpus, int numGpus, TransferMap& transferMap)
{
  // Replace any round brackets or '->' with spaces,
  for (int i = 1; line[i]; i++)
    if (line[i] == '(' || line[i] == ')' || line[i] == '-' || line[i] == '>' ) line[i] = ' ';

  transferMap.clear();
  int numTransfers = 0;

  std::istringstream iss(line);
  iss >> numTransfers;
  if (iss.fail()) return;

  std::string exeMem;
  std::string srcMem;
  std::string dstMem;
  if (numTransfers > 0)
  {
    // Method 1: Take in triples (srcMem, exeMem, dstMem)
    int numBlocksToUse;
    iss >> numBlocksToUse;
    if (numBlocksToUse <= 0 || iss.fail())
    {
      printf("Parsing error: Number of blocks to use (%d) must be greater than 0\n", numBlocksToUse);
      exit(1);
    }
    for (int i = 0; i < numTransfers; i++)
    {
      Transfer transfer;
      transfer.transferIndex = i;
      iss >> srcMem >> exeMem >> dstMem;
      if (iss.fail())
      {
        printf("Parsing error: Unable to read valid Transfer triplet (possibly missing a SRC or EXE or DST)\n");
        exit(1);
      }
      ParseMemType(srcMem, numCpus, numGpus, &transfer.srcMemType, &transfer.srcIndex);
      ParseMemType(exeMem, numCpus, numGpus, &transfer.exeMemType, &transfer.exeIndex);
      ParseMemType(dstMem, numCpus, numGpus, &transfer.dstMemType, &transfer.dstIndex);
      transfer.numBlocksToUse = numBlocksToUse;

      // Ensure executor is either CPU or GPU
      if (transfer.exeMemType != MEM_CPU && transfer.exeMemType != MEM_GPU)
      {
        printf("[ERROR] Executor must either be CPU ('C') or GPU ('G'), (from (%s->%s->%s %d))\n",
               srcMem.c_str(), exeMem.c_str(), dstMem.c_str(), transfer.numBlocksToUse);
        exit(1);
      }

      Executor executor(transfer.exeMemType, transfer.exeIndex);
      ExecutorInfo& executorInfo = transferMap[executor];
      executorInfo.totalBlocks += transfer.numBlocksToUse;
      executorInfo.transfers.push_back(transfer);
    }
  }
  else
  {
    // Method 2: Read in quads (srcMem, exeMem, dstMem,  Read common # blocks to use, then read (src, dst) doubles
    numTransfers *= -1;

    for (int i = 0; i < numTransfers; i++)
    {
      Transfer transfer;
      transfer.transferIndex = i;
      iss >> srcMem >> exeMem >> dstMem >> transfer.numBlocksToUse;
      if (iss.fail())
      {
        printf("Parsing error: Unable to read valid Transfer quadruple (possibly missing a SRC or EXE or DST or #CU)\n");
        exit(1);
      }
      ParseMemType(srcMem, numCpus, numGpus, &transfer.srcMemType, &transfer.srcIndex);
      ParseMemType(exeMem, numCpus, numGpus, &transfer.exeMemType, &transfer.exeIndex);
      ParseMemType(dstMem, numCpus, numGpus, &transfer.dstMemType, &transfer.dstIndex);
      if (transfer.exeMemType != MEM_CPU && transfer.exeMemType != MEM_GPU)
      {
        printf("[ERROR] Executor must either be CPU ('C') or GPU ('G'), (from (%s->%s->%s %d))\n"
,               srcMem.c_str(), exeMem.c_str(), dstMem.c_str(), transfer.numBlocksToUse);
        exit(1);
      }

      Executor executor(transfer.exeMemType, transfer.exeIndex);
      ExecutorInfo& executorInfo = transferMap[executor];
      executorInfo.totalBlocks += transfer.numBlocksToUse;
      executorInfo.transfers.push_back(transfer);
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

void AllocateMemory(MemType memType, int devIndex, size_t numBytes, void** memPtr)
{
  if (numBytes == 0)
  {
    printf("[ERROR] Unable to allocate 0 bytes\n");
    exit(1);
  }

  if (memType == MEM_CPU || memType == MEM_CPU_FINE)
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

    if (memType == MEM_CPU_FINE)
    {
      HIP_CALL(hipHostMalloc((void **)memPtr, numBytes, hipHostMallocNumaUser));
    }
    else
    {
      HIP_CALL(hipHostMalloc((void **)memPtr, numBytes, hipHostMallocNumaUser | hipHostMallocNonCoherent));
    }

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

void DeallocateMemory(MemType memType, void* memPtr)
{
  if (memType == MEM_CPU || memType == MEM_CPU_FINE)
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
  if (srcMemType == MEM_CPU || srcMemType == MEM_CPU_FINE)
  {
    if (dstMemType == MEM_CPU || dstMemType == MEM_CPU_FINE)
      return (srcIndex == dstIndex) ? "LOCAL" : "NUMA";
    else if (dstMemType == MEM_GPU || dstMemType == MEM_GPU_FINE)
      return "PCIE";
    else
      goto error;
  }
  else if (srcMemType == MEM_GPU || srcMemType == MEM_GPU_FINE)
  {
    if (dstMemType == MEM_CPU || dstMemType == MEM_CPU_FINE)
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

std::string GetTransferDesc(Transfer const& transfer)
{
  return GetDesc(transfer.srcMemType, transfer.srcIndex, transfer.exeMemType, transfer.exeIndex) + "-"
    + GetDesc(transfer.exeMemType, transfer.exeIndex, transfer.dstMemType, transfer.dstIndex);
}

void RunTransfer(EnvVars const& ev, size_t const N, int const iteration, ExecutorInfo& exeInfo, int const transferIdx)
{
  Transfer& transfer = exeInfo.transfers[transferIdx];

  // GPU execution agent
  if (transfer.exeMemType == MEM_GPU)
  {
    // Switch to executing GPU
    int const exeIndex = RemappedIndex(transfer.exeIndex, MEM_GPU);
    HIP_CALL(hipSetDevice(exeIndex));

    hipStream_t& stream     = exeInfo.streams[transferIdx];
    hipEvent_t&  startEvent = exeInfo.startEvents[transferIdx];
    hipEvent_t&  stopEvent  = exeInfo.stopEvents[transferIdx];

    int const initOffset = ev.byteOffset / sizeof(float);

    if (ev.useHipCall)
    {
      // Record start event
      HIP_CALL(hipEventRecord(startEvent, stream));

      // Execute hipMemset / hipMemcpy
      if (ev.useMemset)
        HIP_CALL(hipMemsetAsync(transfer.dstMem + initOffset, 42, N * sizeof(float), stream));
      else
        HIP_CALL(hipMemcpyAsync(transfer.dstMem + initOffset,
                                transfer.srcMem + initOffset,
                                N * sizeof(float), hipMemcpyDefault,
                                stream));
      // Record stop event
      HIP_CALL(hipEventRecord(stopEvent, stream));
    }
    else
    {
      int const numBlocksToRun = ev.useSingleStream ? exeInfo.totalBlocks : transfer.numBlocksToUse;
      hipExtLaunchKernelGGL(ev.useMemset ? GpuMemsetKernel : GpuCopyKernel,
                            dim3(numBlocksToRun, 1, 1),
                            dim3(BLOCKSIZE, 1, 1),
                            ev.sharedMemBytes, stream,
                            startEvent, stopEvent,
                            0, transfer.blockParamGpuPtr);
    }

    // Synchronize per iteration, unless in single sync mode, in which case
    // synchronize during last warmup / last actual iteration
    HIP_CALL(hipStreamSynchronize(stream));

    if (iteration >= 0)
    {
      // Record GPU timing
      float gpuDeltaMsec;
      HIP_CALL(hipEventElapsedTime(&gpuDeltaMsec, startEvent, stopEvent));

      if (ev.useSingleStream)
      {
        for (Transfer& currTransfer : exeInfo.transfers)
        {
          long long minStartCycle = currTransfer.blockParamGpuPtr[0].startCycle;
          long long maxStopCycle  = currTransfer.blockParamGpuPtr[0].stopCycle;
          for (int i = 1; i < currTransfer.numBlocksToUse; i++)
          {
            minStartCycle = std::min(minStartCycle, currTransfer.blockParamGpuPtr[i].startCycle);
            maxStopCycle  = std::max(maxStopCycle,  currTransfer.blockParamGpuPtr[i].stopCycle);
          }
          int const wallClockRate = GetWallClockRate(exeIndex);
          double iterationTimeMs = (maxStopCycle - minStartCycle) / (double)(wallClockRate);
          currTransfer.transferTime += iterationTimeMs;
        }
        exeInfo.totalTime += gpuDeltaMsec;
      }
      else
      {
        transfer.transferTime += gpuDeltaMsec;
      }
    }
  }
  else if (transfer.exeMemType == MEM_CPU) // CPU execution agent
  {
    // Force this thread and all child threads onto correct NUMA node
    if (numa_run_on_node(transfer.exeIndex))
    {
      printf("[ERROR] Unable to set CPU to NUMA node %d\n", transfer.exeIndex);
      exit(1);
    }

    std::vector<std::thread> childThreads;

    auto cpuStart = std::chrono::high_resolution_clock::now();

    // Launch child-threads to perform memcopies
    for (int i = 0; i < ev.numCpuPerTransfer; i++)
      childThreads.push_back(std::thread(ev.useMemset ? CpuMemsetKernel : CpuCopyKernel, std::ref(transfer.blockParam[i])));

    // Wait for child-threads to finish
    for (int i = 0; i < ev.numCpuPerTransfer; i++)
      childThreads[i].join();

    auto cpuDelta = std::chrono::high_resolution_clock::now() - cpuStart;

    // Record time if not a warmup iteration
    if (iteration >= 0)
      transfer.transferTime += (std::chrono::duration_cast<std::chrono::duration<double>>(cpuDelta).count() * 1000.0);
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

  if (!ev.outputToCsv)
  {
    printf("Performing copies in each direction of %lu bytes\n", N * sizeof(float));
    printf("Using %d threads per NUMA node for CPU copies\n", ev.numCpuPerTransfer);
    printf("Using %d CUs per transfer\n", numBlocksToUse);
  }
  else
  {
    printf("SRC,DST,Direction,ReadMode,BW(GB/s),Bytes\n");
  }

  // Perform unidirectional / bidirectional
  for (int isBidirectional = 0; isBidirectional <= 1; isBidirectional++)
  {
    // Print header
    if (!ev.outputToCsv)
    {
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
    }

    // Loop over all possible src/dst pairs
    for (int src = 0; src < numDevices; src++)
    {
      MemType const& srcMemType = (src < numCpus ? MEM_CPU : MEM_GPU);
      if (skipCpu && srcMemType == MEM_CPU) continue;
      int srcIndex = (srcMemType == MEM_CPU ? src : src - numCpus);
      if (!ev.outputToCsv)
        printf("%7s %02d", (srcMemType == MEM_CPU) ? "CPU" : "GPU", srcIndex);
      for (int dst = 0; dst < numDevices; dst++)
      {
        MemType const& dstMemType = (dst < numCpus ? MEM_CPU : MEM_GPU);
        if (skipCpu && dstMemType == MEM_CPU) continue;
        int dstIndex = (dstMemType == MEM_CPU ? dst : dst - numCpus);
        double bandwidth = GetPeakBandwidth(ev, N, isBidirectional, readMode, numBlocksToUse,
                                            srcMemType, srcIndex, dstMemType, dstIndex);
        if (!ev.outputToCsv)
        {
          if (bandwidth == 0)
            printf("%10s", "N/A");
          else
            printf("%10.2f", bandwidth);
        }
        else
        {
          printf("%s %02d,%s %02d,%s,%s,%.2f,%lu\n",
                 srcMemType == MEM_CPU ? "CPU" : "GPU",
                 srcIndex,
                 dstMemType == MEM_CPU ? "CPU" : "GPU",
                 dstIndex,
                 isBidirectional ? "bidirectional" : "unidirectional",
                 readMode == 0 ? "Local" : "Remote",
                 bandwidth,
                 N * sizeof(float));
        }
        fflush(stdout);
      }
      if (!ev.outputToCsv) printf("\n");
    }
    if (!ev.outputToCsv) printf("\n");
  }
}

double GetPeakBandwidth(EnvVars const& ev,
                        size_t  const  N,
                        int     const  isBidirectional,
                        int     const  readMode,
                        int     const  numBlocksToUse,
                        MemType const  srcMemType,
                        int     const  srcIndex,
                        MemType const  dstMemType,
                        int     const  dstIndex)
{
  // Skip bidirectional on same device
  if (isBidirectional && srcMemType == dstMemType && srcIndex == dstIndex) return 0.0f;

  int const initOffset = ev.byteOffset / sizeof(float);

  // Prepare Transfers
  std::vector<Transfer*> transfers;
  ExecutorInfo exeInfo[2];
  for (int i = 0; i < 2; i++)
  {
    exeInfo[i].transfers.resize(1);
    exeInfo[i].streams.resize(1);
    exeInfo[i].startEvents.resize(1);
    exeInfo[i].stopEvents.resize(1);
    transfers.push_back(&exeInfo[i].transfers[0]);
  }

  transfers[0]->srcMemType = transfers[1]->dstMemType = srcMemType;
  transfers[0]->dstMemType = transfers[1]->srcMemType = dstMemType;
  transfers[0]->srcIndex   = transfers[1]->dstIndex   = RemappedIndex(srcIndex, srcMemType);
  transfers[0]->dstIndex   = transfers[1]->srcIndex   = RemappedIndex(dstIndex, dstMemType);

  // Either perform (local read + remote write), or (remote read + local write)
  transfers[0]->exeMemType = (readMode == 0 ? srcMemType : dstMemType);
  transfers[1]->exeMemType = (readMode == 0 ? dstMemType : srcMemType);
  transfers[0]->exeIndex   = RemappedIndex((readMode == 0 ? srcIndex : dstIndex), transfers[0]->exeMemType);
  transfers[1]->exeIndex   = RemappedIndex((readMode == 0 ? dstIndex : srcIndex), transfers[1]->exeMemType);

  for (int i = 0; i <= isBidirectional; i++)
  {
    AllocateMemory(transfers[i]->srcMemType, transfers[i]->srcIndex,
                   N * sizeof(float) + ev.byteOffset, (void**)&transfers[i]->srcMem);
    AllocateMemory(transfers[i]->dstMemType, transfers[i]->dstIndex,
                   N * sizeof(float) + ev.byteOffset, (void**)&transfers[i]->dstMem);

    // Prepare block parameters on CPU
    transfers[i]->numBlocksToUse = (transfers[i]->exeMemType == MEM_GPU) ? numBlocksToUse : ev.numCpuPerTransfer;
    transfers[i]->blockParam.resize(transfers[i]->numBlocksToUse);
    transfers[i]->PrepareBlockParams(ev, N);

    if (transfers[i]->exeMemType == MEM_GPU)
    {
      // Copy block parameters onto GPU
      AllocateMemory(MEM_GPU, transfers[i]->exeIndex, numBlocksToUse * sizeof(BlockParam),
                     (void **)&transfers[i]->blockParamGpuPtr);
      HIP_CALL(hipMemcpy(transfers[i]->blockParamGpuPtr,
                         transfers[i]->blockParam.data(),
                         numBlocksToUse * sizeof(BlockParam),
                         hipMemcpyHostToDevice));

      // Prepare GPU resources
      HIP_CALL(hipSetDevice(transfers[i]->exeIndex));
      HIP_CALL(hipStreamCreate(&exeInfo[i].streams[0]));
      HIP_CALL(hipEventCreate(&exeInfo[i].startEvents[0]));
      HIP_CALL(hipEventCreate(&exeInfo[i].stopEvents[0]));
    }
  }

  std::stack<std::thread> threads;

  // Perform iteration
  for (int iteration = -ev.numWarmups; iteration < ev.numIterations; iteration++)
  {
    // Perform timed iterations
    for (int i = 0; i <= isBidirectional; i++)
      threads.push(std::thread(RunTransfer, std::ref(ev), N, iteration, std::ref(exeInfo[i]), 0));

    // Wait for all threads to finish
    for (int i = 0; i <= isBidirectional; i++)
    {
      threads.top().join();
      threads.pop();
    }
  }

  // Validate that each Transfer has transferred correctly
  for (int i = 0; i <= isBidirectional; i++)
    CheckOrFill(MODE_CHECK, N, ev.useMemset, ev.useHipCall, ev.fillPattern, transfers[i]->dstMem + initOffset);

  // Collect aggregate bandwidth
  double totalBandwidth = 0;
  for (int i = 0; i <= isBidirectional; i++)
  {
    double transferDurationMsec = transfers[i]->transferTime / (1.0 * ev.numIterations);
    double transferBandwidthGbs = (N * sizeof(float) / 1.0E9) / transferDurationMsec * 1000.0f;
    totalBandwidth += transferBandwidthGbs;
  }

  // Release GPU memory
  for (int i = 0; i <= isBidirectional; i++)
  {
    DeallocateMemory(transfers[i]->srcMemType, transfers[i]->srcMem);
    DeallocateMemory(transfers[i]->dstMemType, transfers[i]->dstMem);

    if (transfers[i]->exeMemType == MEM_GPU)
    {
      DeallocateMemory(MEM_GPU, transfers[i]->blockParamGpuPtr);
      HIP_CALL(hipStreamDestroy(exeInfo[i].streams[0]));
      HIP_CALL(hipEventDestroy(exeInfo[i].startEvents[0]));
      HIP_CALL(hipEventDestroy(exeInfo[i].stopEvents[0]));
    }
  }
  return totalBandwidth;
}

void Transfer::PrepareBlockParams(EnvVars const& ev, size_t const N)
{
  int const initOffset = ev.byteOffset / sizeof(float);

  // Initialize source memory with patterned data
  CheckOrFill(MODE_FILL, N, ev.useMemset, ev.useHipCall, ev.fillPattern, this->srcMem + initOffset);

  // Each block needs to know src/dst pointers and how many elements to transfer
  // Figure out the sub-array each block does for this Transfer
  // - Partition N as evenly as possible, but try to keep blocks as multiples of BLOCK_BYTES bytes,
  //   except the very last one, for alignment reasons
  int const targetMultiple = ev.blockBytes / sizeof(float);
  int const maxNumBlocksToUse = std::min((N + targetMultiple - 1) / targetMultiple, this->blockParam.size());
  size_t assigned = 0;
  for (int j = 0; j < this->blockParam.size(); j++)
  {
    int    const blocksLeft = std::max(0, maxNumBlocksToUse - j);
    size_t const leftover   = N - assigned;
    size_t const roundedN   = (leftover + targetMultiple - 1) / targetMultiple;

    BlockParam& param = this->blockParam[j];
    param.N          = blocksLeft ? std::min(leftover, ((roundedN / blocksLeft) * targetMultiple)) : 0;
    param.src        = this->srcMem + assigned + initOffset;
    param.dst        = this->dstMem + assigned + initOffset;
    param.startCycle = 0;
    param.stopCycle  = 0;
    assigned += param.N;
  }

  this->transferTime = 0.0;
}

// NOTE: This is a stop-gap solution until HIP provides wallclock values
int GetWallClockRate(int deviceId)
{
  static std::vector<int> wallClockPerDeviceMhz;

  if (wallClockPerDeviceMhz.size() == 0)
  {
    int numGpuDevices;
    HIP_CALL(hipGetDeviceCount(&numGpuDevices));
    wallClockPerDeviceMhz.resize(numGpuDevices);

    hipDeviceProp_t prop;
    for (int i = 0; i < numGpuDevices; i++)
    {
      HIP_CALL(hipGetDeviceProperties(&prop, i));
      int value = 25000;
      switch (prop.gcnArch)
      {
      case 906: case 910: value = 25000; break;
      default:
        printf("Unrecognized GCN arch %d\n", prop.gcnArch);
      }
      wallClockPerDeviceMhz[i] = value;
    }
  }
  return wallClockPerDeviceMhz[deviceId];
}
