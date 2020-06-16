/*************************************************************************
 * Copyright (c) 2016-2020, NVIDIA CORPORATION. All rights reserved.
 * Modifications Copyright (c) 2019-2020 Advanced Micro Devices, Inc. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

#include <sched.h>
#include <fcntl.h>
#include <unistd.h>
#include <hip/hip_runtime.h>
#include <string.h>
#include <errno.h>
#include <assert.h>
#include <dlfcn.h>
#include <sys/types.h>
#include <sys/stat.h>
#include <unistd.h>
#include <cstdio>
#include <iostream>
#include <cstring>
#include "comm.h"
#include "net.h"
#include "graph.h"
#include <sys/time.h>
#include <numaif.h>
#include <sys/socket.h>
#include <netinet/in.h>
#include <arpa/inet.h>
#include <numa.h>

ncclResult_t initNet();

char* getCmdOption(char ** begin, char ** end, const std::string & option) {
    char ** itr = std::find(begin, end, option);
    if (itr != end && ++itr != end)
    {
        return *itr;
    }
    return 0;
}

bool cmdOptionExists(char** begin, char** end, const std::string& option) {
    return std::find(begin, end, option) != end;
}

#define DEFAULT_BUFFSIZE (1LL << 22) /* 4MiB */
#define SLICE_STEPS 4
#define ITERATIONS 2000
#define VEGA_GPU_RTC_FREQUENCY 2.5E7
#define ENABLE_VALIDATION
#define USE_MEMALIGN
#define USE_UNROLL 8

typedef ulong2 Pack128;

struct sockaddr_in netConnectAddr;
void* netSendComm;
int netSendDev;
char *sendDevBuffer;
char *sendHostBuffer, *d_sendHostBuffer;
void *sendDevHandle;
void *sendHostHandle;
int sendBuffSize;
uint64_t *sendHead, *sendTail, *sourceCycle, *sourceBytes;
struct timeval send_tvs;
uint64_t send_sizes;
int send_active_req;
float send_bw_cumulative;
int send_bw_count;

struct sockaddr_in netListenAddr;
void* netListenComm;
void* netRecvComm;
int netRecvDev;
char *recvDevBuffer;
char *recvHostBuffer, *d_recvHostBuffer;
void *recvDevHandle;
void *recvHostHandle;
int recvBuffSize;
uint64_t *recvHead, *recvTail, *recvErrorCount, *sinkCycle, *sinkBytes;
struct timeval recv_tvs;
uint64_t recv_sizes;
int recv_active_req;
float recv_bw_cumulative;
int recv_bw_count;

bool use_gdr_read = false, use_gdr_write = true;
bool runSend = false, runRecv = false;

uint64_t send_byte;
uint64_t recv_byte;

__device__
inline  __attribute((always_inline))
long long int __rtc64() {
#if __HIP__
  return (long long int) __builtin_amdgcn_s_memrealtime();
#else
  return (long long int) __clock_u64();
#endif
}

inline __device__ void Fetch128(Pack128& v, const Pack128* p) {
  v.x = p->x;
  v.y = p->y;
}
inline __device__ void Store128(Pack128* p, Pack128& v) {
  p->x = v.x;
  p->y = v.y;
}

template<int UNROLL, bool SINK>
inline __device__ void DataSourceOrSink(const int w, const int nw, const int t,
    Pack128* buff, const int Npack, uint64_t seq, uint64_t *error) {
  const int inc = nw * UNROLL * WARP_SIZE;
  int offset = w * UNROLL * WARP_SIZE + t;

  Pack128* src = buff + offset;

  uint64_t x = (uint64_t)(offset) + (seq<<32);
  uint64_t y = seq + (((uint64_t)(offset))<<32);
  while (offset < Npack) {
    Pack128 vals[UNROLL];
    if (SINK) {
      for (int u = 0; u < UNROLL; ++u) Fetch128(vals[u], src + u*WARP_SIZE);
      for (int u = 0; u < UNROLL; ++u) {
          if (vals[u].x != x++ || vals[u].y != y++ ) {
          __atomic_fetch_add(error, 1, __ATOMIC_SEQ_CST);
        }
      }
    } else {
      for (int u = 0; u < UNROLL; ++u) {
        vals[u].x = x++;
        vals[u].y = y++;
      }
      for (int u = 0; u < UNROLL; ++u) Store128(src + u*WARP_SIZE, vals[u]);
    }
    src += inc;
    offset += inc;
  }
}

__global__ void DataSinkKernel(const uint64_t end, Pack128* data, uint64_t* recv_head, uint64_t* recv_tail, uint64_t* mismatch, uint64_t *sink_cycle, uint64_t *sink_bytes) {
  const int N = DEFAULT_BUFFSIZE*SLICE_STEPS/NCCL_STEPS/sizeof(Pack128);
  Pack128* recvBuff[NCCL_STEPS];
  const int tid = threadIdx.x;
  uint64_t tail = LOAD(recv_tail);
  __shared__ uint64_t error;
  const int w = tid / WARP_SIZE;
  const int nw = blockDim.x / WARP_SIZE;
  const int t = tid % WARP_SIZE;
  uint64_t t0;
  if (tid == 0) error = 0;
  __syncthreads();
  for (int i = 0; i < NCCL_STEPS; i++)
    recvBuff[i] = data + (i/SLICE_STEPS)*N;
  do {
    if (tid == 0) while (LOAD(recv_head) < tail + SLICE_STEPS);
    __syncthreads();
    if (tid == 0) t0 = __rtc64();
#ifdef ENABLE_VALIDATION
    Pack128* d = recvBuff[tail%NCCL_STEPS];
    DataSourceOrSink<USE_UNROLL, 1>(w, nw, t, recvBuff[tail%NCCL_STEPS], N, tail, &error);
    __syncthreads();
#endif
    tail += SLICE_STEPS;
    if (tid == 0) {
      STORE(recv_tail, tail);
      *sink_cycle += (__rtc64() - t0);
      *sink_bytes += N;
    }
  } while (tail < end);
  if (tid == 0) STORE(mismatch, error);
}

ncclResult_t netRecvProxy(struct ncclProxyArgs* args) {
  char* localBuff = use_gdr_write ? recvDevBuffer : recvHostBuffer;
  void* mhandle = use_gdr_write ? recvDevHandle : recvHostHandle;
  int stepSize = recvBuffSize / NCCL_STEPS;
  if (args->head < args->end) {
    if ((args->tail < args->head + NCCL_STEPS) && (args->tail < LOAD(recvTail) + NCCL_STEPS) && (args->tail < args->end)) {
      int buffSlot = args->tail%NCCL_STEPS;
      int sliceSize = stepSize * args->sliceSteps;
      NCCLCHECK(ncclNetIrecv(netRecvComm, localBuff+buffSlot*stepSize, sliceSize, mhandle, args->requests+buffSlot));
      if (args->requests[buffSlot] != NULL) {
        if (recv_active_req == 0) {
          gettimeofday(&recv_tvs, NULL);
          recv_sizes = 0;
        }
        recv_active_req ++;
        args->tail += args->sliceSteps;
        args->idle = 0;
      }
    }
    if (args->tail > args->head) {
      int buffSlot = args->head%NCCL_STEPS;
      int done, size;
      NCCLCHECK(ncclNetTest(args->requests[buffSlot], &done, &size));
      if (done) {
        recv_active_req --;
        recv_sizes += size;
        if (recv_active_req == 0) {
          struct timeval tv;
          gettimeofday(&tv, NULL);
          recv_bw_cumulative += (float)recv_sizes/((tv.tv_sec - recv_tvs.tv_sec)*1000*1000 + tv.tv_usec - recv_tvs.tv_usec)/1000.0;
          recv_bw_count ++;
        }
        args->head += args->sliceSteps;
        recv_byte += size;
        NCCLCHECK(ncclNetFlush(netRecvComm, localBuff+buffSlot*stepSize, size, mhandle));
        STORE(recvHead, args->head);
        args->idle = 0;
      }
    }
  } else {
    runRecv = false;
  }
  return ncclSuccess;
}

__global__ void DataSourceKernel(const uint64_t end, Pack128* data, uint64_t* send_head, uint64_t* send_tail, uint64_t *source_cycle, uint64_t *source_bytes) {
  const int N = DEFAULT_BUFFSIZE*SLICE_STEPS/NCCL_STEPS/sizeof(Pack128);
  Pack128* sendBuff[NCCL_STEPS];
  const int tid = threadIdx.x;
  uint64_t head = LOAD(send_head);
  const int w = tid / WARP_SIZE;
  const int nw = blockDim.x / WARP_SIZE;
  const int t = tid % WARP_SIZE;
  uint64_t t0;
  for (int i = 0; i < NCCL_STEPS; i++)
    sendBuff[i] = data + (i/SLICE_STEPS)*N;
  do {
    if (tid == 0) while (LOAD(send_tail) + NCCL_STEPS < head + SLICE_STEPS);
    __syncthreads();
    if (tid == 0) t0 = __rtc64();
    DataSourceOrSink<USE_UNROLL, 0>(w, nw, t, sendBuff[head%NCCL_STEPS], N, head, 0);
    __syncthreads();
    head += SLICE_STEPS;
    if (tid == 0) {
      STORE(send_head, head);
      *source_cycle += (__rtc64() - t0);
      *source_bytes += N;
    }
  } while (head < end);
}

ncclResult_t netSendProxy(struct ncclProxyArgs* args) {
  char* localBuff = use_gdr_read ? sendDevBuffer : sendHostBuffer;
  void* mhandle = use_gdr_read ? sendDevHandle : sendHostHandle;
  int stepSize = sendBuffSize / NCCL_STEPS;
  int sliceSize = stepSize * args->sliceSteps;
  if (args->head < args->end) {
    if (args->tail < args->end && args->tail < args->head + NCCL_STEPS) {
      if (args->tail < LOAD(sendHead)) {
        int buffSlot = args->tail%NCCL_STEPS;
        NCCLCHECK(ncclNetIsend(netSendComm, localBuff+buffSlot*stepSize, sliceSize, mhandle, args->requests+buffSlot));
        if (args->requests[buffSlot] != NULL) {
          if (send_active_req == 0) {
            gettimeofday(&send_tvs, NULL);
            send_sizes = 0;
          }
          send_active_req ++;
          send_sizes += sliceSize;
          send_byte += sliceSize;
          __sync_synchronize();
          args->tail += args->sliceSteps;
          args->idle = 0;
        }
      }
    }
    if (args->head < args->tail) {
      int done;
      int buffSlot = args->head%NCCL_STEPS;
      NCCLCHECK(ncclNetTest(args->requests[buffSlot], &done, NULL));
      if (done) {
        send_active_req --;
        if (send_active_req == 0) {
          struct timeval tv;
          gettimeofday(&tv, NULL);
          send_bw_cumulative += (float)send_sizes/((tv.tv_sec - send_tvs.tv_sec)*1000*1000 + tv.tv_usec - send_tvs.tv_usec)/1000.0;
          send_bw_count ++;
        }
        args->head += args->sliceSteps;
        STORE(sendTail, args->head);
        args->idle = 0;
      }
    }
  }
  else
    runSend = false;
  return ncclSuccess;
}

int main(int argc,char* argv[])
{
  struct ncclComm *comm;
  int sliceSteps = SLICE_STEPS;
  bool isSource;

  NCCLCHECK(initNet());
  int ndev;
  NCCLCHECK(ncclNetDevices(&ndev));
  if (ndev == 0) {
    printf("No IB devices found.\n");
    return 0;
  }
  else
    printf("Found %d IB devices\n", ndev);

  sendBuffSize = recvBuffSize = DEFAULT_BUFFSIZE;

  char *gpu = getCmdOption(argv, argv + argc, "-g");
  if (gpu) {
    printf("Select GPU %s\n", gpu);
    CUDACHECK(hipSetDevice(atol(gpu)));
  }

  char *gdr_read = getCmdOption(argv, argv + argc, "-r");
  if (gdr_read) {
    use_gdr_read = atol(gdr_read);
  }

  char *gdr_write = getCmdOption(argv, argv + argc, "-w");
  if (gdr_write) {
    use_gdr_write = atol(gdr_write);
  }

  char *node = getCmdOption(argv, argv + argc, "-n");
  if (node) {
#if 0
    unsigned long nodemask = 1;
    nodemask <<= atol(node);
    set_mempolicy(MPOL_PREFERRED, (const unsigned long*)&nodemask, 16);
    printf("Select node %s for preferred memory allocation\n", node);
#else
    int ret = numa_run_on_node(atol(node));
    if (ret != 0)
      printf("Failed to run on numa node %s\n", node);
    else
      printf("thread is set to run on numa node %ld\n", atol(node));
#endif
  }

  if (cmdOptionExists(argv, argv + argc, "-d")) {
    char *ip = getCmdOption(argv, argv + argc, "-d");
    if (ip)
      inet_pton(AF_INET, ip, &netConnectAddr.sin_addr);
    char *port = getCmdOption(argv, argv + argc, "-p");
    if (port)
      netConnectAddr.sin_port = htons(atoi(port));
    else
      netConnectAddr.sin_port = htons(23456);

    netConnectAddr.sin_family = AF_INET;
    printf("Connecting to %s:%d\n", ip, ntohs(netConnectAddr.sin_port));

    printf("GDR Read %s\n", use_gdr_read ? "enabled" : "disabled");

    if (use_gdr_read) {
      NCCLCHECK(ncclCudaCalloc(&sendDevBuffer, sendBuffSize, 1));
      printf("Allocated sendDevBuffer %p of %d bytes, sliceSteps %d\n",
                sendDevBuffer, sendBuffSize, sliceSteps);
    }
    else {
#ifdef USE_MEMALIGN
      int page_size = getpagesize();
      posix_memalign((void **)&sendHostBuffer, page_size, sendBuffSize);
      hipHostRegister(sendHostBuffer, sendBuffSize, hipHostRegisterMapped);
      hipHostGetDevicePointer((void **)&d_sendHostBuffer, sendHostBuffer, 0);
#else
      NCCLCHECK(ncclCudaHostCalloc(&sendHostBuffer, sendBuffSize));
      d_sendHostBuffer = sendHostBuffer;
#endif
      int status[1] = {-1};
      if (!move_pages(0, 1, (void **)&sendHostBuffer, NULL, status, 0))
        printf("Allocated sendHostBuffer %p of %d bytes on node %d, sliceSteps %d\n",
          sendHostBuffer, sendBuffSize, status[0], sliceSteps);
    }

    NCCLCHECK(ncclCudaHostCalloc(&sendHead, 1));
    NCCLCHECK(ncclCudaHostCalloc(&sendTail, 1));
    NCCLCHECK(ncclCudaHostCalloc(&sourceCycle, 1));
    NCCLCHECK(ncclCudaHostCalloc(&sourceBytes, 1));
    netSendDev = 0;
    NCCLCHECK(ncclNetConnect(netSendDev, &netConnectAddr, &netSendComm));

    if (use_gdr_read) {
      NCCLCHECK(ncclNetRegMr(netSendComm, sendDevBuffer, sendBuffSize, NCCL_PTR_CUDA, &sendDevHandle));
    } else {
      NCCLCHECK(ncclNetRegMr(netSendComm, sendHostBuffer, sendBuffSize, NCCL_PTR_HOST, &sendHostHandle));
    }

    hipLaunchKernelGGL(DataSourceKernel, dim3(1, 1, 1), dim3(256, 1, 1), 0, 0,
      NCCL_STEPS, (Pack128 *)(use_gdr_read ? sendDevBuffer : d_sendHostBuffer), sendHead, sendTail, sourceCycle, sourceBytes);

    runSend = true;
    isSource = true;
  } else {
    printf("GDR Write %s\n", use_gdr_write ? "enabled" : "disabled");

    if (use_gdr_write) {
      NCCLCHECK(ncclCudaCalloc(&recvDevBuffer, recvBuffSize, 1));
      printf("Allocated recvDevBuffer %p of %d bytes, sliceSteps %d\n",
                recvDevBuffer, recvBuffSize, sliceSteps);
    }
    else {
#ifdef USE_MEMALIGN
      int page_size = getpagesize();
      posix_memalign((void **)&recvHostBuffer, page_size, recvBuffSize);
      hipHostRegister(recvHostBuffer, recvBuffSize, hipHostRegisterMapped);
      hipHostGetDevicePointer((void **)&d_recvHostBuffer, recvHostBuffer, 0);
#else
      NCCLCHECK(ncclCudaHostCalloc(&recvHostBuffer, recvBuffSize));
      d_recvHostBuffer = recvHostBuffer;
#endif
      int status[1] = {-1};
      if (!move_pages(0, 1, (void **)&recvHostBuffer, NULL, status, 0))
        printf("Allocated recvHostBuffer %p of %d bytes on node %d, sliceSteps %d\n",
          recvHostBuffer, recvBuffSize, status[0], sliceSteps);
    }

    NCCLCHECK(ncclCudaHostCalloc(&recvHead, 1));
    NCCLCHECK(ncclCudaHostCalloc(&recvTail, 1));
    NCCLCHECK(ncclCudaHostCalloc(&recvErrorCount, 1));
    NCCLCHECK(ncclCudaHostCalloc(&sinkCycle, 1));
    NCCLCHECK(ncclCudaHostCalloc(&sinkBytes, 1));
    netRecvDev = 0;
    NCCLCHECK(ncclNetListen(netRecvDev, (void *)&netListenAddr, &netListenComm));
    char ip[INET_ADDRSTRLEN];
    uint16_t port;
    inet_ntop(AF_INET, &netListenAddr.sin_addr, ip, sizeof(ip));
    port = htons(netListenAddr.sin_port);
    printf("Listening on socket %s:%d\n", ip, port);

    NCCLCHECK(ncclNetAccept(netListenComm, &netRecvComm));
    NCCLCHECK(ncclNetCloseListen(netListenComm));

    if (use_gdr_write) {
      NCCLCHECK(ncclNetRegMr(netRecvComm, recvDevBuffer, recvBuffSize, NCCL_PTR_CUDA, &recvDevHandle));
    } else {
      NCCLCHECK(ncclNetRegMr(netRecvComm, recvHostBuffer, recvBuffSize, NCCL_PTR_HOST, &recvHostHandle));
    }

    hipLaunchKernelGGL(DataSinkKernel, dim3(1, 1, 1), dim3(256, 1, 1), 0, 0,
      NCCL_STEPS, (Pack128 *)(use_gdr_write ? recvDevBuffer : d_recvHostBuffer), recvHead, recvTail, recvErrorCount, sinkCycle, sinkBytes);

    runRecv = true;
    isSource = false;
  }

  struct ncclProxyArgs sendArgs, recvArgs;

  memset(&sendArgs, 0, sizeof(struct ncclProxyArgs));
  sendArgs.head = 0;
  sendArgs.tail = 0;
  sendArgs.end = NCCL_STEPS;
  sendArgs.sliceSteps = sliceSteps;

  memset(&recvArgs, 0, sizeof(struct ncclProxyArgs));
  recvArgs.head = 0;
  recvArgs.tail = 0;
  recvArgs.end = NCCL_STEPS;
  recvArgs.sliceSteps = sliceSteps;

  printf("Running warm up...");
  do {
    if (runRecv)
      NCCLCHECK(netRecvProxy(&recvArgs));
    if (runSend)
      NCCLCHECK(netSendProxy(&sendArgs));
  } while (runSend || runRecv);

  CUDACHECK(hipDeviceSynchronize());
  printf("completed\n");

  // reset all counters after warm up cycle
  if (isSource) {
    *sendHead = 0; *sendTail = 0; *sourceCycle = 0; *sourceBytes = 0;
    send_sizes = 0; send_bw_cumulative = 0; send_bw_count =0; send_byte = 0;
    hipLaunchKernelGGL(DataSourceKernel, dim3(1, 1, 1), dim3(256, 1, 1), 0, 0,
      NCCL_STEPS*ITERATIONS, (Pack128 *)(use_gdr_read ? sendDevBuffer : d_sendHostBuffer), sendHead, sendTail, sourceCycle, sourceBytes);
    runSend = true;
  } else {
    *recvHead = 0; *recvTail = 0; *recvErrorCount = 0; *sinkCycle = 0, *sinkBytes = 0;
    recv_sizes = 0; recv_bw_cumulative = 0; recv_bw_count =0; recv_byte = 0;
    hipLaunchKernelGGL(DataSinkKernel, dim3(1, 1, 1), dim3(256, 1, 1), 0, 0,
      NCCL_STEPS*ITERATIONS, (Pack128 *)(use_gdr_write ? recvDevBuffer : d_recvHostBuffer), recvHead, recvTail, recvErrorCount, sinkCycle, sinkBytes);
    runRecv = true;
  }

  struct timeval tv_start, tv_end;
  gettimeofday(&tv_start, NULL);

  memset(&sendArgs, 0, sizeof(struct ncclProxyArgs));
  sendArgs.head = 0;
  sendArgs.tail = 0;
  sendArgs.end = NCCL_STEPS*ITERATIONS;
  sendArgs.sliceSteps = sliceSteps;

  memset(&recvArgs, 0, sizeof(struct ncclProxyArgs));
  recvArgs.head = 0;
  recvArgs.tail = 0;
  recvArgs.end = NCCL_STEPS*ITERATIONS;
  recvArgs.sliceSteps = sliceSteps;

  do {
    if (runRecv)
      NCCLCHECK(netRecvProxy(&recvArgs));
    if (runSend)
      NCCLCHECK(netSendProxy(&sendArgs));
  } while (runSend || runRecv);

  CUDACHECK(hipDeviceSynchronize());

  gettimeofday(&tv_end, NULL);
  uint64_t total_time = ((uint64_t)(tv_end.tv_sec - tv_start.tv_sec)*1000*1000 + tv_end.tv_usec - tv_start.tv_usec);

  if (send_byte) printf("# Send %6.2f GB/s (%ld bytes %ld us) Proxy %6.2f GB/s (%d mmts) Kernel %6.2f GB/s (%ld bytes)\n",
    (total_time) ? (double)send_byte/total_time/1000.0 : 0,
    send_byte, total_time, send_bw_count ? (float)send_bw_cumulative/send_bw_count : 0, send_bw_count,
    *sourceCycle ? (double)(*sourceBytes)*sizeof(Pack128)/((double)(*sourceCycle)/VEGA_GPU_RTC_FREQUENCY*1.0E9) : 0, *sourceBytes*sizeof(Pack128));
  if (recv_byte) printf("# Recv %6.2f GB/s (%ld bytes %ld us) Proxy %6.2f GB/s (%d mmts) Kernel %6.2f GB/s (%ld bytes) Data Error Counts %ld\n",
    (total_time) ? (double)recv_byte/total_time/1000.0 : 0,
    recv_byte, total_time, recv_bw_count ? (float)recv_bw_cumulative/recv_bw_count : 0, recv_bw_count,
    *sinkCycle ? (double)(*sinkBytes)*sizeof(Pack128)/((double)(*sinkCycle)/VEGA_GPU_RTC_FREQUENCY*1.0E9) : 0, *sinkBytes*sizeof(Pack128),
    *recvErrorCount);

  if (isSource) {
    NCCLCHECK(ncclCudaHostFree(sourceCycle));
    NCCLCHECK(ncclCudaHostFree(sourceBytes));
    NCCLCHECK(ncclCudaHostFree(sendHead));
    NCCLCHECK(ncclCudaHostFree(sendTail));
    if (use_gdr_read) {
      NCCLCHECK(ncclNetDeregMr(netSendComm, sendDevHandle));
      CUDACHECK(hipFree(sendDevBuffer));
    } else {
      NCCLCHECK(ncclNetDeregMr(netSendComm, sendHostHandle));
#ifdef USE_MEMALIGN
      hipHostUnregister(sendHostBuffer);
      free(sendHostBuffer);
#else
      NCCLCHECK(ncclCudaHostFree(sendHostBuffer));
#endif
    }
    NCCLCHECK(ncclNetCloseSend(netSendComm));
  } else {
    NCCLCHECK(ncclCudaHostFree(sinkCycle));
    NCCLCHECK(ncclCudaHostFree(sinkBytes));
    NCCLCHECK(ncclCudaHostFree(recvErrorCount));
    NCCLCHECK(ncclCudaHostFree(recvHead));
    NCCLCHECK(ncclCudaHostFree(recvTail));
    if (use_gdr_write) {
      NCCLCHECK(ncclNetDeregMr(netRecvComm, recvDevHandle));
      CUDACHECK(hipFree(recvDevBuffer));
    } else {
      NCCLCHECK(ncclNetDeregMr(netRecvComm, recvHostHandle));
#ifdef USE_MEMALIGN
      hipHostUnregister(recvHostBuffer);
      free(recvHostBuffer);
#else
      NCCLCHECK(ncclCudaHostFree(recvHostBuffer));
#endif
    }
    NCCLCHECK(ncclNetCloseRecv(netRecvComm));
  }

  return 0;
}
