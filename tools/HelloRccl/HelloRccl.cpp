/*
Copyright (c) 2020 Advanced Micro Devices, Inc. All rights reserved.

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

#include <sys/socket.h>
#include <ifaddrs.h>
#include <netdb.h>
#include <unistd.h>
#include <cstdio>
#include <string>
#include <chrono>
#include <hip/hip_runtime.h>
#include <rccl.h>
#include "HelloRccl.hpp"


void Usage(char *argv0);

int main(int argc, char **argv)
{
  // This example program uses NCCL_COMM_ID to perform bootstrapping
  // to sidestep the need to communicate the ncclUniqueId (e.g. via MPI)
  if (argc < 3 || getenv("NCCL_COMM_ID") == NULL)
  {
    Usage(argv[0]);
    return 1;
  }
  // Collect command-line arguments
  int nranks   = atoi(argv[1]);
  int rank     = atoi(argv[2]);
  int deviceId = atoi(argv[3]);

  // Allocate GPU resources
  hipStream_t stream;
  hipEvent_t startEvent, stopEvent;
  HIP_CALL(hipSetDevice(deviceId));
  HIP_CALL(hipStreamCreate(&stream));
  HIP_CALL(hipEventCreate(&startEvent));
  HIP_CALL(hipEventCreate(&stopEvent));

  // Create communicator
  ncclUniqueId commId;
  NCCL_CALL(ncclGetUniqueId(&commId));

  // Initialize communicator
  ncclComm_t comm;
  NCCL_CALL(ncclCommInitRank(&comm, nranks, commId, rank));

  // Loop over powers of 2
  int minPow = 10;
  int maxPow = 28;

  if (rank == 0)
  {
    printf("AllReduce Performance (sum of floats):\n");
    printf("%10s %10s %10s\n", "Bytes", "CpuTime(ms)", "GpuTime(ms)");
  }

  for (int power = minPow; power <= maxPow; power++)
  {
    int N = 1 << power;

    // Allocate GPU memory
    float *iputGpu, *oputGpu;
    HIP_CALL(hipMalloc((void **)&iputGpu, N * sizeof(float)));
    HIP_CALL(hipMalloc((void **)&oputGpu, N * sizeof(float)));

    // Allocate CPU memory
    float *iputCpu = (float *)malloc(N * sizeof(float));
    float *oputCpu = (float *)malloc(N * sizeof(float));

    // Fill CPU with a simple pattern
    for (int i = 0; i < N; i++)
    {
      iputCpu[i] = 1.0f;
      oputCpu[i] = 0.0f;
    }

    // Copy the input from CPU memory to GPU memory
    HIP_CALL(hipMemcpy(iputGpu, iputCpu, N * sizeof(float), hipMemcpyHostToDevice));

    // Perform some untimed initial warmup iterations
    int numWarmups = 3;
    for (int iteration = 0; iteration < numWarmups; iteration++)
    {
      NCCL_CALL(ncclAllReduce(iputGpu, oputGpu, N, ncclFloat, ncclSum, comm, stream));
    }
    HIP_CALL(hipStreamSynchronize(stream));

    // Perform timed iterations
    int numIterations = 10;
    auto cpuStart = std::chrono::high_resolution_clock::now();
    HIP_CALL(hipEventRecord(startEvent, stream));
    for (int iteration = 0; iteration < numIterations; iteration++)
    {
      NCCL_CALL(ncclAllReduce(iputGpu, oputGpu, N, ncclFloat, ncclSum, comm, stream));
    }
    HIP_CALL(hipEventRecord(stopEvent, stream));
    HIP_CALL(hipStreamSynchronize(stream));
    auto cpuDelta = std::chrono::high_resolution_clock::now() - cpuStart;
    double totalCpuTime = std::chrono::duration_cast<std::chrono::duration<double, std::milli>>(cpuDelta).count();
    float totalGpuTime;
    HIP_CALL(hipEventElapsedTime(&totalGpuTime, startEvent, stopEvent));

    if (rank == 0) printf("%10lu %10.3f %10.3f\n", N * sizeof(float), (totalCpuTime / numIterations), (totalGpuTime / numIterations));


    // Validate results
    HIP_CALL(hipMemcpy(oputCpu, oputGpu, N * sizeof(float), hipMemcpyDeviceToHost));
    bool isOK = true;
    int expected = nranks;
    for (int i = 0; i < N; i++)
    {
      isOK &= (oputCpu[i] == expected);
    }
    if (!isOK)
    {
      printf("[ERROR] Rank %d Incorrect results for N = %d\n", rank, N);
      exit(1);
    }

    // Release GPU resources
    HIP_CALL(hipFree(oputGpu));
    HIP_CALL(hipFree(iputGpu));

    free(iputCpu);
    free(oputCpu);
  }

  HIP_CALL(hipStreamDestroy(stream));
  HIP_CALL(hipEventDestroy(startEvent));
  HIP_CALL(hipEventDestroy(stopEvent));
  NCCL_CALL(ncclCommDestroy(comm));
  return 0;
}

void Usage(char *argv0)
{
  printf("Usage: %s numRanks rank deviceId [N=8] [verbose=0]\n", argv0);
  printf(" - NCCL_COMM_ID must be set in order to use this\n\n");
  printf(" - To use this process as the root process you may use any of the following:\n");

  char hostname[256];
  gethostname(hostname, 256);
  printf("    export NCCL_COMM_ID=%s:12345\n", hostname);

  // Loop over linked list of interfaces
  struct ifaddrs *ifaddr;
  getifaddrs(&ifaddr);
  for (struct ifaddrs* ifa = ifaddr; ifa != NULL; ifa = ifa->ifa_next)
  {
    // Skip anything not based on IPv4 / IPv6
    int family = ifa->ifa_addr->sa_family;
    if (family != AF_INET && family != AF_INET6) continue;

    // Skip iPv6 loopback interface
    if (family == AF_INET6)
    {
      struct sockaddr_in6* sa = (struct sockaddr_in6*)(ifa->ifa_addr);
      if (IN6_IS_ADDR_LOOPBACK(&sa->sin6_addr)) continue;
    }

    socklen_t saLen = (family == AF_INET ? sizeof(struct sockaddr_in) : sizeof(struct sockaddr_in6));
    char host[NI_MAXHOST];
    char service[NI_MAXSERV];

    getnameinfo(ifa->ifa_addr, saLen, host, NI_MAXHOST, service, NI_MAXSERV,
                NI_NUMERICHOST|NI_NUMERICSERV);

    std::string result = std::string(host) + ":12345";
    printf("    export NCCL_COMM_ID=%s\n", result.c_str());
  }
  freeifaddrs(ifaddr);
}
