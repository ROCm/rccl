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

#if defined(ENABLE_GPU_WALLCLOCK)
  uint64_t gpuStart;
  uint64_t gpuStop;
#endif
};

__global__ void SyncKernel(volatile uint64_t* cpuTime,
                           volatile uint32_t* abortFlag,
                           SyncData* syncData)
{
  // Only first thread in threadblock participates
  if (threadIdx.x != 0) return;

  // Collect timestamp upon kernel entry
  uint64_t cpuStart = *cpuTime;
#if defined(ENABLE_GPU_WALLCLOCK)
  uint64_t gpuStart = wall_clock64();
#endif

  // Wait for abort flag to be modified
  while (*abortFlag == 0);

  // Collect timestamps after abort flag
  uint64_t cpuStop = *cpuTime;
#if defined(ENABLE_GPU_WALLCLOCK)
  uint64_t gpuStop = wall_clock64();
#endif

  // Save timestamps
  SyncData sd;
  GetXccId(sd.xccId);
  sd.cpuStart = cpuStart;
  sd.cpuStop  = cpuStop;
#if defined(ENABLE_GPU_WALLCLOCK)
  sd.gpuStart = gpuStart;
  sd.gpuStop  = gpuStop;
#endif

  syncData[blockIdx.x] = sd;
}

void SetNumaNode(int numaId)
{
  if (getenv("IGNORE_NUMA")) return;

  // Move CPU thread to targeted NUMA node
  if (numa_run_on_node(numaId))
  {
    printf("[ERROR] Unable to migrate to NUMA node %d\n", numaId);
    exit(1);
  }

  // Set memory to allocate on targeted NUMA node
  numa_set_preferred(numaId);
}


void UpdateCpuTime(int const numaId, volatile uint64_t* cpuTimestamp, volatile bool& abortThread)
{
  SetNumaNode(numaId);

  while (!abortThread)
  {
    *cpuTimestamp = std::chrono::steady_clock::now().time_since_epoch().count();
  }
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

  int  numBlocks        = (argc > 1 ? atoi(argv[1]) :       4);
  int  blockSize        = (argc > 2 ? atoi(argv[2]) :      32);
  int  numUpdateThreads = (argc > 3 ? atoi(argv[3]) :       1);
  int  numIterations    = (argc > 4 ? atoi(argv[4]) :      10);
  int  numWarmups       = (argc > 5 ? atoi(argv[5]) :    1000);
  int  numSleepUsec     = (argc > 6 ? atoi(argv[6]) :      20);
  int  totalIterations  = numWarmups + numIterations;

  // Print off configuration and machine information
  printf("Running %d GPUs with %d block(s) each of size %d, %d update threads, %d timed iterations, %d warmup iterations, sleeping for %d usec\n",
         numGpus, numBlocks, blockSize, numUpdateThreads, numIterations, numWarmups, numSleepUsec);
  char archName[100];
  for (int i = 0; i < numGpus; i++)
  {
    hipDeviceProp_t prop;
    HIP_CALL(hipGetDeviceProperties(&prop, i));
    sscanf(prop.gcnArchName, "%[^:]", archName);
    printf("GPU %02d: %s: Closest NUMA: %d\n", i, archName, GetClosestNumaNode(i));
  }

  typedef typename std::ratio_multiply<std::chrono::steady_clock::period,std::mega>::type MicroSec;
  printf("std::chrono::steady_clock precision: %8.3f usec\n",
         static_cast<double>(MicroSec::num)/MicroSec::den);

  // Allocate per-update-thread resources and start update threads
  volatile bool abortUpdateThreads = false;
  std::vector<volatile uint64_t*> cpuTimestamps(numUpdateThreads);
  std::vector<std::thread> updateThreads;
  for (int i = 0; i < numUpdateThreads; i++)
  {
    int numaId = GetClosestNumaNode(i);
    HIP_CALL(hipSetDevice(i));
    SetNumaNode(numaId);
#if !defined(__NVCC__)
    HIP_CALL(hipHostMalloc((void**)&cpuTimestamps[i], sizeof(uint64_t), hipHostMallocNumaUser));
#else
    HIP_CALL(hipHostMalloc((void**)&cpuTimestamps[i], sizeof(uint64_t)));
#endif

    updateThreads.push_back(std::thread(UpdateCpuTime, numaId, cpuTimestamps[i], std::ref(abortUpdateThreads)));
  }

  // Allocate per-GPU resources
  std::vector<SyncData*>          syncDataGpu(numGpus);
  std::vector<SyncData*>          syncDataCpu(numGpus);
  std::vector<volatile uint32_t*> abortFlags(numGpus);
  std::vector<hipStream_t>        streams(numGpus);
  for (int i = 0; i < numGpus; i++)
  {
    HIP_CALL(hipSetDevice(i));
    SetNumaNode(GetClosestNumaNode(i));

    HIP_CALL(    hipMalloc((void**)&syncDataGpu[i], totalIterations * numBlocks * sizeof(SyncData)));
#if !defined(__NVCC__)
    HIP_CALL(hipHostMalloc((void**)&syncDataCpu[i], totalIterations * numBlocks * sizeof(SyncData), hipHostMallocNumaUser));
    HIP_CALL(hipHostMalloc((void**)&abortFlags[i],  sizeof(uint32_t),                               hipHostMallocNumaUser));
#else
    HIP_CALL(hipHostMalloc((void**)&syncDataCpu[i], totalIterations * numBlocks * sizeof(SyncData)));
    HIP_CALL(hipHostMalloc((void**)&abortFlags[i],  sizeof(uint32_t)));
#endif
    HIP_CALL(hipStreamCreate(&streams[i]));
  }

  // Allocate per-iteration resources
  std::vector<std::vector<uint64_t>>  cpuAbortTime(numGpus, std::vector<uint64_t>(totalIterations, 0));
  std::vector<std::vector<uint64_t>>  cpuStartList(numGpus, std::vector<uint64_t>(totalIterations, 0));
  std::vector<std::vector<uint64_t>> cpuReturnList(numGpus, std::vector<uint64_t>(totalIterations, 0));
  std::vector<std::vector<uint64_t>>   cpuStopList(numGpus, std::vector<uint64_t>(totalIterations, 0));

  // Launch one thread per GPU
  #pragma omp parallel num_threads(numGpus)
  {
    int deviceId = omp_get_thread_num();
    HIP_CALL(hipSetDevice(deviceId));
    SetNumaNode(GetClosestNumaNode(deviceId));

    volatile uint64_t* cpuTimestamp = cpuTimestamps[deviceId % numUpdateThreads];
    volatile uint32_t* abortFlag    = abortFlags[deviceId];

    for (int iteration = 0; iteration < totalIterations; iteration++)
    {
      // Prepare for this iteration
      // Clear abort flag
      *abortFlag = 0;
      SyncData* syncData = syncDataGpu[deviceId] + (iteration * numBlocks);

      // Wait for all threads to arrive before launching all kernels
      #pragma omp barrier

      // Launch kernel
      uint64_t cpuStart = std::chrono::steady_clock::now().time_since_epoch().count();
      SyncKernel<<<numBlocks, blockSize, 0, streams[deviceId]>>>(cpuTimestamp, abortFlag, syncData);
      uint64_t cpuReturn = std::chrono::steady_clock::now().time_since_epoch().count();

      // Busy wait performs more accurately than usleep / sleep_for
      while (std::chrono::steady_clock::now().time_since_epoch().count() - cpuStart < numSleepUsec * 1000);
      *abortFlag = 1;
      uint64_t cpuAbort = std::chrono::steady_clock::now().time_since_epoch().count();

      // Wait for kernel to finish
      HIP_CALL(hipStreamSynchronize(streams[deviceId]));
      uint64_t cpuStop = std::chrono::steady_clock::now().time_since_epoch().count();

      // Store values (after all timings to avoid false sharing)
      cpuStartList [deviceId][iteration] = cpuStart;
      cpuReturnList[deviceId][iteration] = cpuReturn;
      cpuAbortTime [deviceId][iteration] = cpuAbort;
      cpuStopList  [deviceId][iteration] = cpuStop;

      #pragma omp barrier
    }
  }

  // Stop all the update threads
  abortUpdateThreads = true;
  for (auto& t : updateThreads)
    t.join();

  for (int i = 0; i < numGpus; i++)
    HIP_CALL(hipMemcpy(syncDataCpu[i], syncDataGpu[i],totalIterations * numBlocks * sizeof(SyncData), hipMemcpyDeviceToHost));

  double minDiffCpuStart, minDiffGpuStart, minDiffCpuReturn, minDiffCpuAbort, minDiffGpuStop, minDiffCpuStop;
  double avgDiffCpuStart, avgDiffGpuStart, avgDiffCpuReturn, avgDiffCpuAbort, avgDiffGpuStop, avgDiffCpuStop;
  double maxDiffCpuStart, maxDiffGpuStart, maxDiffCpuReturn, maxDiffCpuAbort, maxDiffGpuStop, maxDiffCpuStop;

  std::vector<TimelineData> timelineData;
  char buff[1000];
  for (int iteration = 1; iteration <= numIterations; iteration++)
  {
    // Ignore warmup iterations
    int iter = iteration + numWarmups - 1;
    printf("---------------------------------------------------------------------------------------------------\n");
    printf("Iteration %d: (All times in usec)\n", iteration);

    uint64_t origin = cpuStartList[0][iter];
    for (int gpu = 0; gpu < numGpus; gpu++)
    {
      for (int block = 0; block < numBlocks; block++)
      {
        origin = std::min(origin, cpuStartList[gpu][iter]);
        origin = std::min(origin, syncDataCpu[gpu][iter * numBlocks + block].cpuStart);
        origin = std::min(origin, cpuAbortTime[gpu][iter]);
        origin = std::min(origin, syncDataCpu[gpu][iter * numBlocks + block].cpuStop);
        origin = std::min(origin, cpuStopList[gpu][iter]);
      }
    }

    printf("| GPU | BLOCK | XCC | START(CPU) | RETURN(CPU)| START(GPU) | ABORT(CPU) | STOP (GPU) | STOP (CPU) |\n");

    double minCpuStart, minGpuStart, minCpuReturn, minCpuAbort, minGpuStop, minCpuStop;
    double maxCpuStart, maxGpuStart, maxCpuReturn, maxCpuAbort, maxGpuStop, maxCpuStop;

    for (int gpu = 0; gpu < numGpus; gpu++)
    {
      for (int block = 0; block < numBlocks; block++)
      {
        double cpuStart  = ( cpuStartList[gpu][iter] - origin) / 1000.0;
        double cpuReturn = (cpuReturnList[gpu][iter] - origin) / 1000.0;
        double cpuAbort  = ( cpuAbortTime[gpu][iter] - origin) / 1000.0;
        double cpuStop   = (  cpuStopList[gpu][iter] - origin) / 1000.0;

        int    blockIdx  = iter * numBlocks + block;
        int    xccId     = syncDataCpu[gpu][blockIdx].xccId;
        double gpuStart  = (syncDataCpu[gpu][blockIdx].cpuStart - origin) / 1000.0;
        double gpuStop   = (syncDataCpu[gpu][blockIdx].cpuStop  - origin) / 1000.0;

        #define CHECK_MINMAX(VAL, MINVAL, MAXVAL) \
        MINVAL = ((gpu == 0 && block == 0) || (MINVAL > VAL)) ? VAL : MINVAL; \
        MAXVAL = ((gpu == 0 && block == 0) || (MAXVAL < VAL)) ? VAL : MAXVAL

        CHECK_MINMAX(cpuStart,  minCpuStart,  maxCpuStart);
        CHECK_MINMAX(cpuReturn, minCpuReturn, maxCpuReturn);
        CHECK_MINMAX(cpuAbort,  minCpuAbort,  maxCpuAbort);
        CHECK_MINMAX(cpuStop,   minCpuStop,   maxCpuStop);
        CHECK_MINMAX(gpuStart,  minGpuStart,  maxGpuStart);
        CHECK_MINMAX(gpuStop,   minGpuStop,   maxGpuStop);

        printf("| %3d |  %3d  | %3d | %10.3f | %10.3f | %10.3f | %10.3f | %10.3f | %10.3f |\n",
               gpu, block, xccId, cpuStart, cpuReturn, gpuStart, cpuAbort, gpuStop, cpuStop);

        TimelineData td;
        sprintf(buff, "Iteration %d GPU %02d Block %02d (CPU)", iteration, gpu, block); td.rowLabel = buff;
        td.barLabel  = "Launch (";
        sprintf(buff, "%.3f to %.3f", cpuStart, cpuReturn); td.toolTip = buff;
        td.startTime = cpuStart;
        td.stopTime  = cpuReturn;
        timelineData.push_back(td);

        td.barLabel  = "Pause";
        sprintf(buff, "%.3f to %.3f", cpuReturn, cpuAbort); td.toolTip = buff;
        td.startTime = cpuReturn;
        td.stopTime  = cpuAbort;
        timelineData.push_back(td);

        td.barLabel  = "Sync";
        sprintf(buff, "%.3f to %.3f", cpuAbort, cpuStop); td.toolTip = buff;
        td.startTime = cpuAbort;
        td.stopTime  = cpuStop;
        timelineData.push_back(td);

        sprintf(buff, "Iteration %d GPU %02d Block %02d (GPU)", iteration, gpu, block); td.rowLabel = buff;
        td.barLabel  = "Kernel";
        sprintf(buff, "%.3f to %.3f", gpuStart, gpuStop); td.toolTip = buff;
        td.startTime = gpuStart;
        td.stopTime  = gpuStop;
        timelineData.push_back(td);
      }
    }
    printf("---------------------------------------------------------------------------------------------------\n");
    printf("|               MIN | %10.3f | %10.3f | %10.3f | %10.3f | %10.3f | %10.3f |\n",
           minCpuStart, minCpuReturn, minGpuStart, minCpuAbort, minGpuStop, minCpuStop);
    printf("|               MAX | %10.3f | %10.3f | %10.3f | %10.3f | %10.3f | %10.3f |\n",
           maxCpuStart, maxCpuReturn, maxGpuStart, maxCpuAbort, maxGpuStop, maxCpuStop);

    double diffCpuStart  = maxCpuStart  - minCpuStart;
    double diffCpuReturn = maxCpuReturn - minCpuReturn;
    double diffGpuStart  = maxGpuStart  - minGpuStart;
    double diffCpuAbort  = maxCpuAbort  - minCpuAbort;
    double diffGpuStop   = maxGpuStop   - minGpuStop;
    double diffCpuStop   = maxCpuStop   - minCpuStop;

    printf("|              DIFF | %10.3f | %10.3f | %10.3f | %10.3f | %10.3f | %10.3f |\n",
           diffCpuStart, diffCpuReturn, diffGpuStart, diffCpuAbort, diffGpuStop, diffCpuStop);

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
  }
  printf("===================================================================================================\n");
  printf("|          DIFF MIN | %10.3f | %10.3f | %10.3f | %10.3f | %10.3f | %10.3f |\n",
         minDiffCpuStart, minDiffCpuReturn, minDiffGpuStart, minDiffCpuAbort, minDiffGpuStop, minDiffCpuStop);
  printf("|          DIFF AVG | %10.3f | %10.3f | %10.3f | %10.3f | %10.3f | %10.3f |\n",
         avgDiffCpuStart / numIterations, avgDiffCpuReturn / numIterations, avgDiffGpuStart / numIterations,
         avgDiffCpuAbort / numIterations, avgDiffGpuStop   / numIterations, avgDiffCpuStop  / numIterations);
  printf("|          DIFF MAX | %10.3f | %10.3f | %10.3f | %10.3f | %10.3f | %10.3f |\n",
         maxDiffCpuStart, maxDiffCpuReturn, maxDiffGpuStart, maxDiffCpuAbort, maxDiffGpuStop, maxDiffCpuStop);

  sprintf(buff, "timeline_%dx%s_%dx%dblockSize_%dCUTs_Sleep%d.html", numGpus, archName, numBlocks, blockSize, numUpdateThreads, numSleepUsec);
  printf("Timeline exported to %s\n", buff);
  ExportToTimeLine(buff, "Device", "Call", timelineData);

  return 0;
}
