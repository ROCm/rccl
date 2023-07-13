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

#define NUM_LOOPS_WARMUP 2000
#define NUM_LOOPS_RUN 10000
#define VEGA_GPU_RTC_FREQUENCY 2.5E7

#define PING_MODE 0
#define PONG_MODE 1

__global__ void PingKernel(uint64_t* local_flag, uint64_t* remote_flag, uint64_t* time_delta) {
  volatile uint16_t *volatile_local_flag = (volatile uint16_t *)local_flag;
  volatile uint16_t *volatile_remote_flag = (volatile uint16_t *)remote_flag;
  #pragma unroll
  for (uint16_t i = 1; i < NUM_LOOPS_WARMUP; i++) {
    *volatile_remote_flag = i;
    while (*volatile_local_flag != i);
  }
  uint64_t start_time = wall_clock64();
  #pragma unroll
  for (uint16_t i = NUM_LOOPS_WARMUP; i <= NUM_LOOPS_WARMUP + NUM_LOOPS_RUN; i++) {
    *volatile_remote_flag = i;
    while (*volatile_local_flag != i);
  }
  uint64_t end_time = wall_clock64();
  *time_delta = end_time - start_time;
}

__global__ void PongKernel(uint64_t* local_flag, uint64_t* remote_flag, uint64_t* time_delta) {
  volatile uint16_t *volatile_local_flag = (volatile uint16_t *)local_flag;
  volatile uint16_t *volatile_remote_flag = (volatile uint16_t *)remote_flag;
  #pragma unroll
  for (uint16_t i = 1; i < NUM_LOOPS_WARMUP; i++) {
    while (*volatile_local_flag != i);
    *volatile_remote_flag = i;
  }
  uint64_t start_time = wall_clock64();
  #pragma unroll
  for (uint16_t i = NUM_LOOPS_WARMUP; i <= NUM_LOOPS_WARMUP + NUM_LOOPS_RUN; i++) {
    while (*volatile_local_flag != i);
    *volatile_remote_flag = i;
  }
  uint64_t end_time = wall_clock64();
  *time_delta = end_time - start_time;
}

int main(int argc, char** argv) {
  hipStream_t stream;
  hipError_t err = hipSuccess;

  if (argc != 2) {
    fprintf(stderr, "Usage: ./p2p_latency_test <flag>; flag=%d for ping mode, flag=%d for pong mode\n", PING_MODE, PONG_MODE);
    return -1;
  }

  uint64_t *local_flag = nullptr;
  uint64_t *remote_flag = nullptr;
  uint64_t *time_delta = nullptr;
  hipStreamCreateWithFlags(&stream, hipStreamNonBlocking);
  hipExtMallocWithFlags((void**)&local_flag, sizeof(uint64_t), hipDeviceMallocFinegrained);
  hipExtMallocWithFlags((void**)&time_delta, sizeof(uint64_t), hipDeviceMallocFinegrained);
  hipMemsetAsync(local_flag, 0, sizeof(uint64_t), stream);
  hipMemsetAsync(time_delta, 0, sizeof(uint64_t), stream);
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
      fprintf(stderr, "HIP error %d\n", (int)err);
      return -1;
    }
    if (self_mode == PING_MODE) {
      PingKernel<<<1, 1, 0, stream>>>(local_flag, remote_flag, time_delta);
    } else {
      PongKernel<<<1, 1, 0, stream>>>(local_flag, remote_flag, time_delta);
    }
    hipStreamSynchronize(stream);
    if (self_mode == PING_MODE) {
      printf("Ping-pong latency in us: %g\n", double(*time_delta) / NUM_LOOPS_RUN * 1e6 / VEGA_GPU_RTC_FREQUENCY);
    }
  } else {
    fprintf(stderr, "Invalid mode %d\n", self_mode);
    return -1;
  }

  return 0;
}
