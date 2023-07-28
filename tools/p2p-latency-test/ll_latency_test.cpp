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

__device__ uint32_t *abortFlag;

inline __device__ int checkAbort(int &spins) {
  uint32_t abort = 0;
  spins++;
  if (spins == LL_SPINS_BEFORE_CHECK_ABORT) {
    abort = __atomic_load_n(abortFlag, __ATOMIC_SEQ_CST);
    spins = 0;
  }
  return abort;
}

__device__ uint64_t readLL(union LLFifoLine* src, uint32_t flag) {
  uint32_t data1, flag1, data2, flag2;
  int spins = 0;

  union LLFifoLine i4;
  do {
    i4.v[0] = __builtin_nontemporal_load(src->v);
    i4.v[1] = __builtin_nontemporal_load(src->v+1);
    if (checkAbort(spins)) break;
  } while ((i4.flag1 != flag) || (i4.flag2 != flag));
  uint64_t val64 = (uint64_t)(i4.data1) + (((uint64_t)i4.data2) << 32);

  return val64;
}


__global__ void PingKernel(LLFifoLine* local_flag, LLFifoLine* remote_flag, uint64_t* time_delta) {
  int tid = threadIdx.x;
  #pragma unroll
  for (uint32_t i = 1; i < NUM_LOOPS_WARMUP; i++) {
    storeLL(remote_flag+tid, i, i);
    while (readLL(local_flag+tid, i) != i);
  }
  uint64_t start_time, end_time;
  if (tid == 0) start_time = wall_clock64();
  #pragma unroll
  for (uint32_t i = NUM_LOOPS_WARMUP; i <= NUM_LOOPS_WARMUP + NUM_LOOPS_RUN; i++) {
    storeLL(remote_flag+tid, i, i);
    while (readLL(local_flag+tid, i) != i);
  }
  __syncthreads();
  if (tid == 0) end_time = wall_clock64();
  if (tid == 0) *time_delta = end_time - start_time;
}

__global__ void PongKernel(LLFifoLine* local_flag, LLFifoLine* remote_flag, uint64_t* time_delta) {
  int tid = threadIdx.x;
  #pragma unroll
  for (uint32_t i = 1; i < NUM_LOOPS_WARMUP; i++) {
    while (readLL(local_flag+tid, i) != i);
    storeLL(remote_flag+tid, i, i);
  }
  uint64_t start_time, end_time;
  if (tid == 0) start_time = wall_clock64();
  #pragma unroll
  for (uint32_t i = NUM_LOOPS_WARMUP; i <= NUM_LOOPS_WARMUP + NUM_LOOPS_RUN; i++) {
    while (readLL(local_flag+tid, i) != i);
    storeLL(remote_flag+tid, i, i);
  }
  __syncthreads();
  if (tid == 0) end_time = wall_clock64();
  if (tid == 0) *time_delta = end_time - start_time;
}

int main(int argc, char** argv) {
  hipStream_t stream;
  hipError_t err = hipSuccess;

  if (argc != 2) {
    fprintf(stderr, "Usage: ./p2p_latency_test <flag>; flag=%d for ping mode, flag=%d for pong mode\n", PING_MODE, PONG_MODE);
    return -1;
  }

  hipDeviceProp_t prop;
  int device_id;
  hipGetDevice(&device_id);
  hipGetDeviceProperties(&prop, device_id);

  LLFifoLine *local_flag = nullptr;
  LLFifoLine *remote_flag = nullptr;
  uint64_t *time_delta = nullptr;
  hipStreamCreateWithFlags(&stream, hipStreamNonBlocking);
  hipExtMallocWithFlags((void**)&local_flag, HIP_IPC_MEM_MIN_SIZE, prop.gcnArch / 10 == 94 ? hipDeviceMallocUncached : hipDeviceMallocFinegrained);
  hipExtMallocWithFlags((void**)&time_delta, HIP_IPC_MEM_MIN_SIZE, prop.gcnArch / 10 == 94 ? hipDeviceMallocUncached : hipDeviceMallocFinegrained);
  hipMemsetAsync(local_flag, 0, HIP_IPC_MEM_MIN_SIZE, stream);
  hipMemsetAsync(time_delta, 0, HIP_IPC_MEM_MIN_SIZE, stream);
  hipStreamSynchronize(stream);
  hipIpcMemHandle_t local_handle;
  hipIpcMemHandle_t remote_handle;
  hipIpcGetMemHandle(&local_handle, local_flag);

  const char* ping_file_path = "/tmp/ping_ipc_handle";
  const char* pong_file_path = "/tmp/pong_ipc_handle";
  const char* file_paths[2] = {ping_file_path, pong_file_path};
  int self_mode = atoi(argv[1]);
  if (self_mode == PING_MODE || self_mode == PONG_MODE) {
    int peer_mode = 1 - self_mode;
    auto self_file = std::fstream(file_paths[self_mode], std::ios::out | std::ios::binary);
    self_file.write((char*)&local_handle, sizeof(local_handle));
    self_file.close();
    sleep(5);
    auto peer_file = std::fstream(file_paths[peer_mode], std::ios::in | std::ios::binary);
    peer_file.read((char*)&remote_handle, sizeof(remote_handle));
    peer_file.close();
    err = hipIpcOpenMemHandle((void**)(&remote_flag), remote_handle, hipIpcMemLazyEnablePeerAccess);
    if (err != hipSuccess) {
      fprintf(stderr, "hipIpcOpenMemHandle error %d\n", (int)err);
      return -1;
    }
    if (self_mode == PING_MODE) {
      PingKernel<<<1, LL_MAX_THREAD, 0, stream>>>(local_flag, remote_flag, time_delta);
    } else {
      PongKernel<<<1, LL_MAX_THREAD, 0, stream>>>(local_flag, remote_flag, time_delta);
    }
    err = hipStreamSynchronize(stream);
    if (err != hipSuccess) {
      fprintf(stderr, "hipStreamSynchronize error %d\n", (int)err);
      return -1;
    }
    if (self_mode == PING_MODE) {
      double vega_gpu_rtc_freq = (prop.gcnArch / 10 == 94) ? 1.0E8 : 2.5E7;
      fprintf(stdout, "One-way latency in us: %g\n", double(*time_delta) * 1e6 / NUM_LOOPS_RUN / vega_gpu_rtc_freq / 2);
    }
    std::remove(file_paths[self_mode]);
  } else {
    fprintf(stderr, "Invalid mode %d\n", self_mode);
    return -1;
  }

  return 0;
}
