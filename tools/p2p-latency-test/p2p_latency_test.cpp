/*************************************************************************
 * Copyright (c) Microsoft Corporation.
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

__global__ void PingKernel(uint64_t* local_flag, uint64_t* remote_flag, uint64_t* time_delta) {
  #pragma unroll
  for (uint32_t i = 1; i < NUM_LOOPS_WARMUP; i++) {
    __atomic_store_n(remote_flag, i, __ATOMIC_RELAXED);
    while (__atomic_load_n(local_flag, __ATOMIC_RELAXED) != i);
  }
  uint64_t start_time = wall_clock64();
  #pragma unroll
  for (uint32_t i = NUM_LOOPS_WARMUP; i <= NUM_LOOPS_WARMUP + NUM_LOOPS_RUN; i++) {
    __atomic_store_n(remote_flag, i, __ATOMIC_RELAXED);
    while (__atomic_load_n(local_flag, __ATOMIC_RELAXED) != i);
  }
  uint64_t end_time = wall_clock64();
  *time_delta = end_time - start_time;
}

__global__ void PongKernel(uint64_t* local_flag, uint64_t* remote_flag, uint64_t* time_delta) {
  #pragma unroll
  for (uint32_t i = 1; i < NUM_LOOPS_WARMUP; i++) {
    while (__atomic_load_n(local_flag, __ATOMIC_RELAXED) != i);
    __atomic_store_n(remote_flag, i, __ATOMIC_RELAXED);
  }
  uint64_t start_time = wall_clock64();
  #pragma unroll
  for (uint32_t i = NUM_LOOPS_WARMUP; i <= NUM_LOOPS_WARMUP + NUM_LOOPS_RUN; i++) {
    while (__atomic_load_n(local_flag, __ATOMIC_RELAXED) != i);
    __atomic_store_n(remote_flag, i, __ATOMIC_RELAXED);
  }
  uint64_t end_time = wall_clock64();
  *time_delta = end_time - start_time;
}

int main(int argc, char** argv) {
  hipStream_t stream;
  hipError_t err = hipSuccess;

  if (argc != 2) {
    fprintf(stderr, "Usage: ./ll_latency_test <flag>; flag=%d for ping mode, flag=%d for pong mode\n", PING_MODE, PONG_MODE);
    return -1;
  }

  hipDeviceProp_t prop;
  int device_id;
  hipGetDevice(&device_id);
  hipGetDeviceProperties(&prop, device_id);

  uint64_t *local_flag = nullptr;
  uint64_t *remote_flag = nullptr;
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
      PingKernel<<<1, 1, 0, stream>>>(local_flag, remote_flag, time_delta);
    } else {
      PongKernel<<<1, 1, 0, stream>>>(local_flag, remote_flag, time_delta);
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
