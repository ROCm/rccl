/*************************************************************************
 * Copyright (c) Microsoft Corporation.
 * Modifications Copyright (c) 2023 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 ************************************************************************/

#include <cstdio>
#include <cstdlib>
#include <chrono>
#include <fstream>
#include <string>
#include <unistd.h>
#include <hip/hip_runtime.h>
#include <iostream> //cerr
#include <cstring>

#define HIP_IPC_MEM_MIN_SIZE 2097152UL

#define NUM_LOOPS_WARMUP 2000
#define NUM_LOOPS_RUN 10000

#define PING_MODE 0
#define PONG_MODE 1

#define LL_MAX_THREAD 256

union LLFifoLine {
  /* Flags have to be *after* data, because otherwise, an incomplete receive
     from the network may receive the flag but not the data.
     Note this is assuming that either we receive contiguous chunks of data
     (sockets) or data is written with an atomicity of 8 bytes (IB/RDMA). */
  struct {
    uint32_t data1;
    uint32_t flag1;
    uint32_t data2;
    uint32_t flag2;
  };
  uint64_t v[2];
  int4 i4;
};

__device__ void storeLL(union LLFifoLine* dst, uint64_t val, uint32_t flag) {
  union LLFifoLine i4;
  i4.data1 = val & 0xffffffff;
  i4.flag1 = flag;
  i4.data2 = (val >> 32);
  i4.flag2 = flag;
  __builtin_nontemporal_store(i4.v[0], dst->v);
  __builtin_nontemporal_store(i4.v[1], dst->v+1);
}

#define LL_SPINS_BEFORE_CHECK_ABORT 1000000

inline __device__ int checkAbort(int &spins, uint32_t* abortFlag) {
  uint32_t abort = 0;
  spins++;
  if (spins == LL_SPINS_BEFORE_CHECK_ABORT) {
    abort = __atomic_load_n(abortFlag, __ATOMIC_SEQ_CST);
    spins = 0;
  }
  return abort;
}

__device__ uint64_t readLL(union LLFifoLine* src, uint32_t flag, uint32_t* abortFlag) {
  uint32_t data1, flag1, data2, flag2;
  int spins = 0;

  union LLFifoLine i4;
  do {
    i4.v[0] = __builtin_nontemporal_load(src->v);
    i4.v[1] = __builtin_nontemporal_load(src->v+1);
    if (checkAbort(spins, abortFlag)) break;
  } while ((i4.flag1 != flag) || (i4.flag2 != flag));
  uint64_t val64 = (uint64_t)(i4.data1) + (((uint64_t)i4.data2) << 32);

  return val64;
}


__global__ void PingKernel(LLFifoLine* local_flag, LLFifoLine* remote_flag, uint64_t* time_delta, uint32_t* abortFlag) {
  int tid = threadIdx.x;
  #pragma unroll
  for (uint32_t i = 1; i < NUM_LOOPS_WARMUP; i++) {
    storeLL(remote_flag+tid, i, i);
    while (readLL(local_flag+tid, i, abortFlag) != i);
  }
  uint64_t start_time, end_time;
  if (tid == 0) start_time = wall_clock64();
  #pragma unroll
  for (uint32_t i = NUM_LOOPS_WARMUP; i <= NUM_LOOPS_WARMUP + NUM_LOOPS_RUN; i++) {
    storeLL(remote_flag+tid, i, i);
    while (readLL(local_flag+tid, i, abortFlag) != i);
  }
  __syncthreads();
  if (tid == 0) end_time = wall_clock64();
  if (tid == 0) *time_delta = end_time - start_time;
}

__global__ void PongKernel(LLFifoLine* local_flag, LLFifoLine* remote_flag, uint64_t* time_delta, uint32_t* abortFlag) {
  int tid = threadIdx.x;
  #pragma unroll
  for (uint32_t i = 1; i < NUM_LOOPS_WARMUP; i++) {
    while (readLL(local_flag+tid, i, abortFlag) != i);
    storeLL(remote_flag+tid, i, i);
  }
  uint64_t start_time, end_time;
  if (tid == 0) start_time = wall_clock64();
  #pragma unroll
  for (uint32_t i = NUM_LOOPS_WARMUP; i <= NUM_LOOPS_WARMUP + NUM_LOOPS_RUN; i++) {
    while (readLL(local_flag+tid, i, abortFlag) != i);
    storeLL(remote_flag+tid, i, i);
  }
  __syncthreads();
  if (tid == 0) end_time = wall_clock64();
  if (tid == 0) *time_delta = end_time - start_time;
}

#define HIPCHECK(cmd)                                                          \
do {                                                                           \
  hipError_t error = (cmd);                                                    \
  if (error != hipSuccess)                                                     \
  {                                                                            \
    std::cerr << "Encountered HIP error (" << error << ") at line "            \
              << __LINE__ << " in file " << __FILE__ << "\n";                  \
    exit(-1);                                                                  \
  }                                                                            \
} while (0)

int main(int argc, char** argv) {
  hipStream_t stream[2];
  hipError_t err = hipSuccess;
  int device_id[2];
  hipDeviceProp_t prop[2];

  if (argc != 3) {
    fprintf(stderr, "Usage: ./ll_latency_test ping_dev_id pong_dev_id\n");
    return -1;
  }
  device_id[0] = atoi(argv[1]);
  device_id[1] = atoi(argv[2]);

  fprintf(stdout, "Using devices %d %d\n", device_id[0], device_id[1]);

  LLFifoLine *flag[2];
  uint64_t *time_delta[2];
  uint32_t *abortFlag[2];

  HIPCHECK(hipSetDevice(device_id[0]));
  HIPCHECK(hipStreamCreateWithFlags(&stream[0], hipStreamNonBlocking));
  HIPCHECK(hipDeviceEnablePeerAccess(device_id[1], 0));
  HIPCHECK(hipGetDeviceProperties(&prop[0], device_id[0]));
  HIPCHECK(hipExtMallocWithFlags((void**)&flag[0], HIP_IPC_MEM_MIN_SIZE, prop[0].gcnArch / 10 == 94 ? hipDeviceMallocUncached : hipDeviceMallocFinegrained));
  HIPCHECK(hipMalloc((void**)&time_delta[0], HIP_IPC_MEM_MIN_SIZE));
  HIPCHECK(hipMalloc((void**)&abortFlag[0], sizeof(uint32_t)));
  HIPCHECK(hipMemsetAsync(flag[0], 0, HIP_IPC_MEM_MIN_SIZE, stream[0]));
  HIPCHECK(hipMemsetAsync(abortFlag[0], 0, sizeof(uint32_t), stream[0]));
  HIPCHECK(hipStreamSynchronize(stream[0]));

  HIPCHECK(hipSetDevice(device_id[1]));
  HIPCHECK(hipStreamCreateWithFlags(&stream[1], hipStreamNonBlocking));
  HIPCHECK(hipDeviceEnablePeerAccess(device_id[0], 0));
  HIPCHECK(hipGetDeviceProperties(&prop[1], device_id[1]));
  HIPCHECK(hipExtMallocWithFlags((void**)&flag[1], HIP_IPC_MEM_MIN_SIZE, prop[1].gcnArch / 10 == 94 ? hipDeviceMallocUncached : hipDeviceMallocFinegrained));
  HIPCHECK(hipMalloc((void**)&time_delta[1], HIP_IPC_MEM_MIN_SIZE));
  HIPCHECK(hipMalloc((void**)&abortFlag[1], sizeof(uint32_t)));
  HIPCHECK(hipMemsetAsync(flag[1], 0, HIP_IPC_MEM_MIN_SIZE, stream[1]));
  HIPCHECK(hipMemsetAsync(abortFlag[1], 0, sizeof(uint32_t), stream[0]));
  HIPCHECK(hipStreamSynchronize(stream[1]));

  HIPCHECK(hipSetDevice(device_id[0]));
  PingKernel<<<1, LL_MAX_THREAD, 0, stream[0]>>>(flag[0], flag[1], time_delta[0], abortFlag[0]);

  HIPCHECK(hipSetDevice(device_id[1]));
  PongKernel<<<1, LL_MAX_THREAD, 0, stream[1]>>>(flag[1], flag[0], time_delta[1], abortFlag[1]);

  double vega_gpu_rtc_freq;

  HIPCHECK(hipStreamSynchronize(stream[0]));
  vega_gpu_rtc_freq = (prop[0].gcnArch / 10 == 94) ? 1.0E8 : 2.5E7;
  fprintf(stdout, "One-way latency in us: %g\n", double(*time_delta[0]) * 1e6 / NUM_LOOPS_RUN / vega_gpu_rtc_freq / 2);

  HIPCHECK(hipStreamSynchronize(stream[1]));
  vega_gpu_rtc_freq = (prop[1].gcnArch / 10 == 94) ? 1.0E8 : 2.5E7;
  fprintf(stdout, "One-way latency in us: %g\n", double(*time_delta[1]) * 1e6 / NUM_LOOPS_RUN / vega_gpu_rtc_freq / 2);

  HIPCHECK(hipFree(flag[0]));
  HIPCHECK(hipFree(time_delta[0]));
  HIPCHECK(hipFree(abortFlag[0]));
  HIPCHECK(hipFree(flag[1]));
  HIPCHECK(hipFree(time_delta[1]));
  HIPCHECK(hipFree(abortFlag[1]));
  return 0;
}