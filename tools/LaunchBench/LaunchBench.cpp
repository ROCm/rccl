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

#include <cstdio>
#include <iostream>
#include <vector>
#include "Common.hpp"
#include <omp.h>
#include <unistd.h>

struct SyncData
{
  uint64_t cpuStart;
  uint64_t cpuStop;
  int32_t  xccId;
};

__global__ void SyncKernel(volatile uint64_t* cpuTime,
                           volatile uint32_t* abortFlag,
                           SyncData* syncData)
{
  // Collect timestamp upon kernel entry
  uint64_t cpuStart = *cpuTime;

  // Wait for abort flag to be modified
  while (*abortFlag == 0);

  // Collect timestamps after abort flag
  uint64_t cpuStop = *cpuTime;

  // Save timestamps
  syncData[blockIdx.x].cpuStart = cpuStart;
  syncData[blockIdx.x].cpuStop  = cpuStop;
  GetXccId(syncData[blockIdx.x].xccId);
}

void UpdateCpuTime(volatile uint64_t* cpuTimestamp, volatile bool& abortThread)
{
  while (!abortThread)
  {
    *cpuTimestamp = std::chrono::steady_clock::now().time_since_epoch().count();
  }
}


int main(int argc, char **argv)
{
  int numBlocks       = (argc > 1 ? atoi(argv[1]) :   4);
  int numIterations   = (argc > 2 ? atoi(argv[2]) :   1);
  int numWarmups      = (argc > 3 ? atoi(argv[3]) : 100);
  int numSleepUsec    = (argc > 4 ? atoi(argv[4]) :  20);

  int totalIterations =  numWarmups + numIterations;

  int numGpus;
  HIP_CALL(hipGetDeviceCount(&numGpus));
  printf("Running %d GPUs with %d block(s) each, %d timed iterations, %d untimed warmup iterations, sleeping for %d usec\n",
         numGpus, numBlocks, numIterations, numWarmups, numSleepUsec);
  for (int i = 0; i < numGpus; i++)
  {
    hipDeviceProp_t prop;
    HIP_CALL(hipGetDeviceProperties(&prop, i));
    printf("GPU %02d: %s\n", i, prop.gcnArchName);
  }

  typedef typename std::ratio_multiply<std::chrono::steady_clock::period,std::mega>::type MicroSec;
  printf("std::chrono::steady_clock precision: %8.3f usec\n",
         static_cast<double>(MicroSec::num)/MicroSec::den);


  // Allocate pinned host memory for CPU timestamp / abort flag
  volatile uint64_t* cpuTimestamp;
  volatile uint32_t* abortFlag;
  HIP_CALL(hipHostMalloc((void**)&cpuTimestamp, sizeof(uint64_t)));
  HIP_CALL(hipHostMalloc((void**)&abortFlag,    sizeof(uint32_t)));

  // Allocate device memory for collecting timestamps
  std::vector<SyncData*> syncDataList(numGpus);
  std::vector<hipStream_t>streams(numGpus);
  for (int i = 0; i < numGpus; i++)
  {
    HIP_CALL(hipSetDevice(i));
    HIP_CALL(hipMalloc((void**)&syncDataList[i], numIterations * numBlocks * sizeof(SyncData)));
    HIP_CALL(hipStreamCreate(&streams[i]));
  }

  // Start update thread
  // NOTE: NPKit usually runs 1 GPU per process which means 1 update thread per GPU
  //       However in this case, only a single CPU update thread is used
  volatile bool abortThread = false;
  std::thread updateThread(UpdateCpuTime, cpuTimestamp, std::ref(abortThread));

  // Launch one thread per GPU
  std::vector<uint64_t> cpuAbortTime(totalIterations);
  std::vector<std::vector<uint64_t>>  cpuStartList(numGpus, std::vector<uint64_t>(totalIterations, 0));
  std::vector<std::vector<uint64_t>> cpuReturnList(numGpus, std::vector<uint64_t>(totalIterations, 0));
  std::vector<std::vector<uint64_t>>   cpuStopList(numGpus, std::vector<uint64_t>(totalIterations, 0));

  uint64_t cpuAbort;
  #pragma omp parallel num_threads(numGpus)
  {
    int deviceId = omp_get_thread_num();
    HIP_CALL(hipSetDevice(deviceId));

    for (int iteration = 0; iteration < totalIterations; iteration++)
    {
      // Single thread resets abort flag
      #pragma omp single
      *abortFlag = 0;

      // Prepare for this iteration
      SyncData* syncData = syncDataList[deviceId] + (iteration * numBlocks);

      // Wait for all threads to arrive before launching all kernels
      #pragma omp barrier
      uint64_t cpuStart = std::chrono::steady_clock::now().time_since_epoch().count();
      SyncKernel<<<numBlocks, 1, 0, streams[deviceId]>>>(cpuTimestamp, abortFlag, syncData);
      uint64_t cpuReturn = std::chrono::steady_clock::now().time_since_epoch().count();

      // Busy wait performs more accurately than usleep / sleep_for
      if (deviceId == 0)
      {
        while (std::chrono::steady_clock::now().time_since_epoch().count() - cpuStart < numSleepUsec * 1000);
        *abortFlag = 1;
        cpuAbort = std::chrono::steady_clock::now().time_since_epoch().count();
      }

      // Wait for kernels to finish
      HIP_CALL(hipStreamSynchronize(streams[deviceId]));
      uint64_t cpuStop = std::chrono::steady_clock::now().time_since_epoch().count();

      // Store values (after all timings to avoid false sharing)
      cpuStartList[deviceId][iteration] = cpuStart;
      cpuReturnList[deviceId][iteration] = cpuReturn;
      #pragma omp single
      cpuAbortTime[iteration] = cpuAbort;
      cpuStopList[deviceId][iteration] = cpuStop;

      #pragma omp barrier
    }
  }

  abortThread = true;
  updateThread.join();

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
        origin = std::min(origin, syncDataList[gpu][iter * numBlocks + block].cpuStart);
        origin = std::min(origin, cpuAbortTime[iter]);
        origin = std::min(origin, syncDataList[gpu][iter * numBlocks + block].cpuStop);
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
        int    xccId     = syncDataList[gpu][iter * numBlocks + block].xccId;
        double cpuStart  = (cpuStartList[gpu][iter] - origin) / 1000.0;
        double gpuStart  = (syncDataList[gpu][iter * numBlocks + block].cpuStart - origin) / 1000.0;
        double cpuReturn = (cpuReturnList[gpu][iter] - origin) / 1000.0;
        double cpuAbort  = (cpuAbortTime[iter] - origin) / 1000.0;
        double gpuStop   = (syncDataList[gpu][iter * numBlocks + block].cpuStop - origin) / 1000.0;
        double cpuStop   = (cpuStopList[gpu][iter] - origin) / 1000.0;

        minCpuStart  = ((gpu == 0 && block == 0) || (minCpuStart  > cpuStart))  ? cpuStart  : minCpuStart;
        maxCpuStart  = ((gpu == 0 && block == 0) || (maxCpuStart  < cpuStart))  ? cpuStart  : maxCpuStart;
        minGpuStart  = ((gpu == 0 && block == 0) || (minGpuStart  > gpuStart))  ? gpuStart  : minGpuStart;
        maxGpuStart  = ((gpu == 0 && block == 0) || (maxGpuStart  < gpuStart))  ? gpuStart  : maxGpuStart;
        minCpuReturn = ((gpu == 0 && block == 0) || (minCpuReturn > cpuReturn)) ? cpuReturn : minCpuReturn;
        maxCpuReturn = ((gpu == 0 && block == 0) || (maxCpuReturn < cpuReturn)) ? cpuReturn : maxCpuReturn;
        minCpuAbort  = ((gpu == 0 && block == 0) || (minCpuAbort  > cpuAbort))  ? cpuAbort  : minCpuAbort;
        maxCpuAbort  = ((gpu == 0 && block == 0) || (maxCpuAbort  < cpuAbort))  ? cpuAbort  : maxCpuAbort;
        minGpuStop   = ((gpu == 0 && block == 0) || (minGpuStop   > gpuStop))   ? gpuStop   : minGpuStop;
        maxGpuStop   = ((gpu == 0 && block == 0) || (maxGpuStop   < gpuStop))   ? gpuStop   : maxGpuStop;
        minCpuStop   = ((gpu == 0 && block == 0) || (minCpuStop   > gpuStop))   ? gpuStop   : minCpuStop;
        maxCpuStop   = ((gpu == 0 && block == 0) || (maxCpuStop   < gpuStop))   ? gpuStop   : maxCpuStop;

        printf("| %3d |  %3d  | %3d | %10.3f | %10.3f | %10.3f | %10.3f | %10.3f | %10.3f |\n",
               gpu, block, xccId, cpuStart, cpuReturn, gpuStart, cpuAbort, gpuStop, cpuStop);
      }
    }
    printf("---------------------------------------------------------------------------------------------------\n");
    printf("|               MIN | %10.3f | %10.3f | %10.3f | %10.3f | %10.3f | %10.3f |\n",
           minCpuStart, minCpuReturn, minGpuStart, minCpuAbort, minGpuStop, minCpuStop);
    printf("|               MAX | %10.3f | %10.3f | %10.3f | %10.3f | %10.3f | %10.3f |\n",
           maxCpuStart, maxCpuReturn, maxGpuStart, maxCpuAbort, maxGpuStop, maxCpuStop);
    printf("|              DIFF | %10.3f | %10.3f | %10.3f | %10.3f | %10.3f | %10.3f |\n",
           maxCpuStart - minCpuStart,  maxCpuReturn - minCpuReturn, maxGpuStart - minGpuStart,
           maxCpuAbort - minCpuAbort,  maxGpuStop   - minGpuStop,    maxCpuStop - minCpuStop);
  }

  return 0;
}
