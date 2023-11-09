/*
Copyright (c) 2023 Advanced Micro Devices, Inc. All rights reserved.

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

#include <chrono>
#include <cstdio>
#include <iostream>
#include <thread>
#include <vector>
#include <numa.h>
#include <omp.h>
#include <unistd.h>

#include "Common.hpp"
#include "Compatibility.hpp"
#include "GetClosestNumaNode.hpp"
#include "Timeline.hpp"

struct SyncData
{
  uint64_t cpuStart;
  uint64_t cpuStop;
  int32_t  xccId;
  uint64_t gpuStart;
  uint64_t gpuStop;
};

enum
{
  HOST_START_CPU      = 0,
  HOST_RETURN_CPU     = 1,
  HOST_ABORT_CPU      = 2,
  HOST_STOP_CPU       = 3,
  NUM_HOST_TIMESTAMPS = 4
};

enum
{
  DEV_START_CPU       = 0,
  DEV_STOP_CPU        = 1,
  NUM_DEV_TIMESTAMPS  = 2
};

#define LOAD(VAR)       __atomic_load_n((VAR),         __ATOMIC_ACQUIRE)
#define STORE(DST, SRC) __atomic_store_n((DST), (SRC), __ATOMIC_RELEASE)

__global__ void SyncKernel(uint64_t* cpuTime, uint32_t* abortFlag, SyncData* syncData)
{
  SyncData sd;
  // Only first thread in threadblock participates
  if (threadIdx.x != 0) return;

  // Collect timestamp upon kernel entry
  sd.cpuStart = LOAD(cpuTime);
  sd.gpuStart = wall_clock64();

  // Wait for abort flag to be modified
  while (!LOAD(abortFlag));

  // Collect timestamps after abort flag
  sd.cpuStop = LOAD(cpuTime);
  sd.gpuStop = wall_clock64();

  // Save timestamps
  GetXccId(sd.xccId);
  syncData[blockIdx.x] = sd;
}

void SetNumaNode(int numaId)
{
  // Move CPU thread to targeted NUMA node
  if (numa_run_on_node(numaId))
  {
    printf("[ERROR] Unable to migrate to NUMA node %d\n", numaId);
    exit(1);
  }

  // Set memory to allocate on targeted NUMA node
  numa_set_preferred(numaId);
}

void UpdateCpuTime(int const useNuma, int const numaId, uint64_t* cpuTimestamp, bool* abortThread)
{
  if (useNuma) SetNumaNode(numaId);
  while (!LOAD(abortThread))
  {
    // Unroll to increase update vs abort check ratio
    #pragma unroll
    for (int i = 0; i < 64; i++)
      STORE(cpuTimestamp, std::chrono::steady_clock::now().time_since_epoch().count());
  }
}

void HostMalloc(void** pinnedHostPtr, size_t size)
{
#if !defined(__NVCC__)
  HIP_CALL(hipHostMalloc(pinnedHostPtr, size, hipHostMallocNumaUser));
#else
  HIP_CALL(hipHostMalloc(pinnedHostPtr, size));
#endif
  memset(*pinnedHostPtr, 0, size);
}

int main(int argc, char **argv)
{
  // Check for NUMA library support
  if (numa_available() == -1)
  {
    printf("[ERROR] NUMA library not supported. Check to see if libnuma has been installed on this system\n");
    exit(1);
  }

  int numGpus;
  HIP_CALL(hipGetDeviceCount(&numGpus));

  #define GETARG(IDX, STR, DEFAULT) \
    (argc > IDX ? atoi(argv[IDX]) : (getenv(STR) ? atoi(getenv(STR)) : DEFAULT))

  int  numBlocks        = GETARG(1, "NUM_BLOCKS",        4);
  int  blockSize        = GETARG(2, "BLOCKSIZE",        64);
  int  numUpdateThreads = GETARG(3, "NUM_UPDATERS",      1);
  int  useNuma          = GETARG(4, "USE_NUMA",          0);
  int  numIterations    = GETARG(5, "NUM_ITERATIONS",   10);
  int  numWarmups       = GETARG(6, "NUM_WARMUPS",    1000);
  int  numSleepUsec     = GETARG(7, "SLEEP_USEC",      100);
  int  totalIterations  = numWarmups + numIterations;

  int  verbose          = (getenv("VERBOSE"    ) ? atoi(getenv("VERBOSE"))     : 1);
  int  launchMode       = (getenv("LAUNCH_MODE") ? atoi(getenv("LAUNCH_MODE")) : 1);

  if (numUpdateThreads == 0) numUpdateThreads = numGpus;

  // Print off configuration and machine information
  printf("NUM_BLOCKS     = %8d\n", numBlocks);
  printf("BLOCKSIZE      = %8d\n", blockSize);
  printf("NUM_UPDATERS   = %8d\n", numUpdateThreads);
  printf("USE_NUMA       = %8d\n", useNuma);
  printf("NUM_ITERATIONS = %8d\n", numIterations);
  printf("NUM_WARMUPS    = %8d\n", numWarmups);
  printf("SLEEP_USEC     = %8d\n", numSleepUsec);

  char archName[100];
  std::vector<double> uSecPerCycle(numGpus);
  for (int i = 0; i < numGpus; i++)
  {
    hipDeviceProp_t prop;
    HIP_CALL(hipGetDeviceProperties(&prop, i));
    sscanf(prop.gcnArchName, "%[^:]", archName);
    int wallClockMhz;
    HIP_CALL(hipDeviceGetAttribute(&wallClockMhz, hipDeviceAttributeWallClockRate, i));
    uSecPerCycle[i] = 1000.0 / wallClockMhz;
    printf("GPU %02d: %s: Closest NUMA: %d usecPerWallClockCycle %g\n", i, archName, GetClosestNumaNode(i), uSecPerCycle[i]);
  }

  typedef typename std::ratio_multiply<std::chrono::steady_clock::period,std::mega>::type MicroSec;
  printf("std::chrono::steady_clock precision: %8.3f usec\n",
         static_cast<double>(MicroSec::num)/MicroSec::den);

  // Allocate per-update-thread resources and start update threads
  bool abortUpdateThreads = false;
  std::vector<uint64_t*> cpuTimestamps(numUpdateThreads);
  std::vector<std::thread> updateThreads;
  for (int i = 0; i < numUpdateThreads; i++)
  {
    int numaId = GetClosestNumaNode(i);
    HIP_CALL(hipSetDevice(i));
    if (useNuma) SetNumaNode(numaId);

    HostMalloc((void**)&cpuTimestamps[i], 256);  // Allocate larger buffer to avoid multiple timestamps on same cacheline

    // Launch update thread
    updateThreads.push_back(std::thread(UpdateCpuTime, useNuma, numaId, cpuTimestamps[i], &abortUpdateThreads));
  }

  // Allocate per-GPU resources
  std::vector<SyncData*>   syncDataGpu(numGpus);
  std::vector<SyncData*>   syncDataCpu(numGpus);
  std::vector<uint32_t*>   abortFlags(numGpus);
  std::vector<hipStream_t> streams(numGpus);
  for (int i = 0; i < numGpus; i++)
  {
    HIP_CALL(hipSetDevice(i));
    if (useNuma) SetNumaNode(GetClosestNumaNode(i));

    HIP_CALL(hipMalloc((void**)&syncDataGpu[i], totalIterations * numBlocks * sizeof(SyncData)));
    HostMalloc((void**)&syncDataCpu[i], totalIterations * numBlocks * sizeof(SyncData));
    HostMalloc((void**)&abortFlags[i], 256); // Allocate larger buffer to avoid multiple abort flags on same cacheline

    HIP_CALL(hipStreamCreate(&streams[i]));
  }

  // Allocate per-iteration resources
  std::vector<std::vector<std::vector<uint64_t>>> hostTimes(numGpus, std::vector<std::vector<uint64_t>>(totalIterations, std::vector<uint64_t>(NUM_HOST_TIMESTAMPS, 0)));

  // Launch one thread per GPU
  #pragma omp parallel num_threads(numGpus)
  {
    int deviceId = omp_get_thread_num();
    HIP_CALL(hipSetDevice(deviceId));
    if (useNuma) SetNumaNode(GetClosestNumaNode(deviceId));

    uint64_t* cpuTimestamp = cpuTimestamps[deviceId % numUpdateThreads];
    uint32_t* abortFlag    = abortFlags[deviceId];

    for (int iteration = 0; iteration < totalIterations; iteration++)
    {
      // Prepare for this iteration
      // Clear abort flag
      STORE(abortFlag, 0);
      SyncData* syncData = syncDataGpu[deviceId] + (iteration * numBlocks);

      // Wait for all threads to arrive before launching all kernels
      #pragma omp barrier

      // Launch kernel
      uint64_t cpuStart = std::chrono::steady_clock::now().time_since_epoch().count();
      if (launchMode == 0)
      {
        SyncKernel<<<numBlocks, blockSize, 0, streams[deviceId]>>>(cpuTimestamp, abortFlag, syncData);
      }
      else
      {
        hipLaunchKernelGGL(SyncKernel, numBlocks, blockSize, 0, streams[deviceId], cpuTimestamp, abortFlag, syncData);
      }
      uint64_t cpuReturn = std::chrono::steady_clock::now().time_since_epoch().count();

      // Busy wait performs more accurately than usleep / sleep_for
      while (std::chrono::steady_clock::now().time_since_epoch().count() - cpuStart < numSleepUsec * 1000);
      STORE(abortFlag, 1);
      uint64_t cpuAbort = std::chrono::steady_clock::now().time_since_epoch().count();

      // Wait for kernel to finish
      HIP_CALL(hipStreamSynchronize(streams[deviceId]));
      uint64_t cpuStop = std::chrono::steady_clock::now().time_since_epoch().count();

      // Store values (after all timings to avoid false sharing)
      hostTimes[deviceId][iteration][HOST_START_CPU]  = cpuStart;
      hostTimes[deviceId][iteration][HOST_RETURN_CPU] = cpuReturn;
      hostTimes[deviceId][iteration][HOST_ABORT_CPU]  = cpuAbort;
      hostTimes[deviceId][iteration][HOST_STOP_CPU]   = cpuStop;

      #pragma omp barrier
    }
  }

  // Stop all the update threads
  STORE(&abortUpdateThreads, true);
  for (auto& t : updateThreads)
    t.join();

  for (int i = 0; i < numGpus; i++)
    HIP_CALL(hipMemcpy(syncDataCpu[i], syncDataGpu[i],totalIterations * numBlocks * sizeof(SyncData), hipMemcpyDeviceToHost));

  std::vector<double> minDiffHost(NUM_HOST_TIMESTAMPS, 0);
  std::vector<double> sumDiffHost(NUM_HOST_TIMESTAMPS, 0);
  std::vector<double> maxDiffHost(NUM_HOST_TIMESTAMPS, 0);
  std::vector<std::vector<double>> minDiffDev(numGpus, std::vector<double>(NUM_DEV_TIMESTAMPS, 0.0));
  std::vector<std::vector<double>> sumDiffDev(numGpus, std::vector<double>(NUM_DEV_TIMESTAMPS, 0.0));
  std::vector<std::vector<double>> maxDiffDev(numGpus, std::vector<double>(NUM_DEV_TIMESTAMPS, 0.0));

  std::vector<TimelineData> timelineData;
  char buff[1000];
  for (int iteration = 1; iteration <= numIterations; iteration++)
  {
    // Ignore warmup iterations
    int iter = iteration + numWarmups - 1;
    if (verbose)
    {
      printf("---------------------------------------------------------------------------------------------------\n");
      printf("Iteration %d: (All times in usec)\n", iteration);
    }

    // Figure out which timestamp is "earliest" to use as origin for this iteration
    uint64_t origin = hostTimes[0][iter][HOST_START_CPU];
    for (int gpu = 1; gpu < numGpus; gpu++)
      origin = std::min(origin, hostTimes[gpu][iter][HOST_START_CPU]);

    if (verbose) printf("| GPU | BLOCK | XCC | START(CPU) | RETURN(CPU)| START(GPU) | ABORT(CPU) | STOP (GPU) | STOP (CPU) | Kernel(CPU)| Kernel(GPU)|\n");

    std::vector<double> minHostTimes(NUM_HOST_TIMESTAMPS, 0);
    std::vector<double> maxHostTimes(NUM_HOST_TIMESTAMPS, 0);
    std::vector<double> sumHostTimes(NUM_HOST_TIMESTAMPS, 0);
    for (int gpu = 0; gpu < numGpus; gpu++)
    {
      std::vector<double> hTimes(NUM_HOST_TIMESTAMPS);
      for (int i = 0; i < NUM_HOST_TIMESTAMPS; i++)
      {
        hTimes[i] = (hostTimes[gpu][iter][i] - origin) / 1000.0;
        minHostTimes[i] = (gpu == 0 || minHostTimes[i] > hTimes[i]) ? hTimes[i] : minHostTimes[i];
        maxHostTimes[i] = (gpu == 0 || maxHostTimes[i] < hTimes[i]) ? hTimes[i] : maxHostTimes[i];
        sumHostTimes[i] += hTimes[i];
      }

      TimelineData td;
      sprintf(buff, "Iteration %d GPU %02d (CPU)", iteration, gpu); td.rowLabel = buff;
      td.barLabel  = "Launch (";
      sprintf(buff, "%.3f to %.3f", hTimes[HOST_START_CPU], hTimes[HOST_RETURN_CPU]); td.toolTip = buff;
      td.startTime = hTimes[HOST_START_CPU];
      td.stopTime  = hTimes[HOST_RETURN_CPU];
      timelineData.push_back(td);

      td.barLabel  = "Pause";
      sprintf(buff, "%.3f to %.3f", hTimes[HOST_RETURN_CPU], hTimes[HOST_ABORT_CPU]); td.toolTip = buff;
      td.startTime = hTimes[HOST_RETURN_CPU];
      td.stopTime  = hTimes[HOST_ABORT_CPU];
      timelineData.push_back(td);

      td.barLabel  = "Sync";
      sprintf(buff, "%.3f to %.3f", hTimes[HOST_ABORT_CPU], hTimes[HOST_STOP_CPU]); td.toolTip = buff;
      td.startTime = hTimes[HOST_ABORT_CPU];
      td.stopTime  = hTimes[HOST_STOP_CPU];
      timelineData.push_back(td);

      std::vector<double> minDevTimes(NUM_DEV_TIMESTAMPS);
      std::vector<double> maxDevTimes(NUM_DEV_TIMESTAMPS);
      std::vector<double> sumDevTimes(NUM_DEV_TIMESTAMPS);

      for (int block = 0; block < numBlocks; block++)
      {
        std::vector<double> dTimes(NUM_DEV_TIMESTAMPS);

        int    blockIdx  = iter * numBlocks + block;
        int    xccId     = syncDataCpu[gpu][blockIdx].xccId;
        double gpuStart  = dTimes[DEV_START_CPU] = (syncDataCpu[gpu][blockIdx].cpuStart - origin) / 1000.0;
        double gpuStop   = dTimes[DEV_STOP_CPU]  = (syncDataCpu[gpu][blockIdx].cpuStop  - origin) / 1000.0;

        double kernelTimeGpu = (syncDataCpu[gpu][blockIdx].gpuStop - syncDataCpu[gpu][blockIdx].gpuStart) * uSecPerCycle[gpu];

        for (int i = 0; i < NUM_DEV_TIMESTAMPS; i++)
        {
          minDevTimes[i] = (block == 0 || minDevTimes[i] > dTimes[i]) ? dTimes[i] : minDevTimes[i];
          maxDevTimes[i] = (block == 0 || maxDevTimes[i] < dTimes[i]) ? dTimes[i] : maxDevTimes[i];
          sumDevTimes[i] += dTimes[i];
        }

        if (verbose)
        {
          printf("| %3d |  %3d  | %3d | %10.3f | %10.3f | %10.3f | %10.3f | %10.3f | %10.3f | %10.3f | %10.3f |\n",
                 gpu, block, xccId, hTimes[HOST_START_CPU], hTimes[HOST_RETURN_CPU], dTimes[DEV_START_CPU],
                 hTimes[HOST_ABORT_CPU], dTimes[DEV_STOP_CPU], hTimes[HOST_STOP_CPU],
                 dTimes[DEV_STOP_CPU] - dTimes[DEV_START_CPU],
                 kernelTimeGpu);
        }

        sprintf(buff, "Iteration %d GPU %02d (GPU)", iteration, gpu); td.rowLabel = buff;
        sprintf(buff, "Block %02d", block); td.barLabel = buff;
        sprintf(buff, "%.3f to %.3f", gpuStart, gpuStop); td.toolTip = buff;
        td.startTime = gpuStart;
        td.stopTime  = gpuStop;
        timelineData.push_back(td);
      }
      if (verbose)
      {
        printf("\n");
      }
    }
/*
    if (verbose)
    {
      printf("---------------------------------------------------------------------------------------------------\n");
      printf("|               MIN | %10.3f | %10.3f | %10.3f | %10.3f | %10.3f | %10.3f |\n",
             minCpuStart, minCpuReturn, minGpuStart, minCpuAbort, minGpuStop, minCpuStop);
      printf("|               MAX | %10.3f | %10.3f | %10.3f | %10.3f | %10.3f | %10.3f |\n",
             maxCpuStart, maxCpuReturn, maxGpuStart, maxCpuAbort, maxGpuStop, maxCpuStop);
      printf("|              DIFF | %10.3f | %10.3f | %10.3f | %10.3f | %10.3f | %10.3f |\n",
             diffCpuStart, diffCpuReturn, diffGpuStart, diffCpuAbort, diffGpuStop, diffCpuStop);
    }

    #define CHECK_MINMAXDIFF(VAL, MINVAL, AVGVAL, MAXVAL)             \
        MINVAL = ((iteration == 1) || (MINVAL > VAL)) ? VAL : MINVAL; \
        AVGVAL += VAL;                                                \
        MAXVAL = ((iteration == 1) || (MAXVAL < VAL)) ? VAL : MAXVAL

    CHECK_MINMAXDIFF(diffCpuStart,  minDiffCpuStart,  avgDiffCpuStart,  maxDiffCpuStart);
    CHECK_MINMAXDIFF(diffCpuReturn, minDiffCpuReturn, avgDiffCpuReturn, maxDiffCpuReturn);
    CHECK_MINMAXDIFF(diffGpuStart,  minDiffGpuStart,  avgDiffGpuStart,  maxDiffGpuStart);
    CHECK_MINMAXDIFF(diffCpuAbort,  minDiffCpuAbort,  avgDiffCpuAbort,  maxDiffCpuAbort);
    CHECK_MINMAXDIFF(diffGpuStop,   minDiffGpuStop,   avgDiffGpuStop,   maxDiffGpuStop);
    CHECK_MINMAXDIFF(diffCpuStop,   minDiffCpuStop,   avgDiffCpuStop,   maxDiffCpuStop);
*/
  }
  /*
  printf("===================================================================================================\n");
  printf("|           SUMMARY | START(CPU) | RETURN(CPU)| START(GPU) | ABORT(CPU) | STOP (GPU) | STOP (CPU) |\n");
  printf("===================================================================================================\n");
  printf("|          DIFF MIN | %10.3f | %10.3f | %10.3f | %10.3f | %10.3f | %10.3f |\n",
         minDiffCpuStart, minDiffCpuReturn, minDiffGpuStart, minDiffCpuAbort, minDiffGpuStop, minDiffCpuStop);
  printf("|          DIFF AVG | %10.3f | %10.3f | %10.3f | %10.3f | %10.3f | %10.3f |\n",
         avgDiffCpuStart / numIterations, avgDiffCpuReturn / numIterations, avgDiffGpuStart / numIterations,
         avgDiffCpuAbort / numIterations, avgDiffGpuStop   / numIterations, avgDiffCpuStop  / numIterations);
  printf("|          DIFF MAX | %10.3f | %10.3f | %10.3f | %10.3f | %10.3f | %10.3f |\n",
         maxDiffCpuStart, maxDiffCpuReturn, maxDiffGpuStart, maxDiffCpuAbort, maxDiffGpuStop, maxDiffCpuStop);

  sprintf(buff, "timeline_%dx%s_%dx%dblockSize_%dCUTs_Numa%d_Sleep%d.html", numGpus, archName, numBlocks, blockSize, numUpdateThreads, useNuma, numSleepUsec);
  printf("Timeline exported to %s\n", buff);
  ExportToTimeLine(buff, "Device", "Call", timelineData);
  */
  return 0;
}
