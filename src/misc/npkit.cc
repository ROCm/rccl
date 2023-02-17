/*************************************************************************
 * Copyright (c) Microsoft Corporation.
 * Licensed under the MIT License.
 ************************************************************************/

#include <chrono>
#include <fstream>
#include <unistd.h>

#include "alloc.h"
#include "npkit/npkit.h"

uint64_t NpKit::rank_ = 0;

NpKitEvent** NpKit::gpu_event_buffers_ = nullptr;
NpKitEvent** NpKit::cpu_event_buffers_ = nullptr;

NpKitEventCollectContext* NpKit::gpu_collect_contexts_ = nullptr;
NpKitEventCollectContext* NpKit::cpu_collect_contexts_ = nullptr;

uint64_t NpKit::base_cpu_timestamp_global_ = 0;
uint64_t NpKit::base_cpu_timestamp_local_ = 0;
uint64_t NpKit::base_gpu_timestamp_cpu_ = 0;
uint64_t NpKit::base_gpu_timestamp_gpu_ = 0;

__global__ void TimeCalibrationKernel(uint64_t *gpu_timestamp, uint64_t *cpu_timestamp_in, uint64_t *cpu_timestamp_out) {
  uint64_t gpu_timestamp_1 = NPKIT_GET_GPU_TIMESTAMP();
  uint64_t cpu_timestamp = *cpu_timestamp_in;
  uint64_t gpu_timestamp_2 = NPKIT_GET_GPU_TIMESTAMP();
  *gpu_timestamp = gpu_timestamp_1 + (gpu_timestamp_2 - gpu_timestamp_1) / 2;
  *cpu_timestamp_out = cpu_timestamp;
}

static volatile bool cpu_timestamp_update_thread_should_stop = false;
static volatile uint64_t* volatile_cpu_timestamp = nullptr;

void NpKit::CpuTimestampUpdateThread() {
  uint64_t init_system_clock = std::chrono::system_clock::now().time_since_epoch().count();
  uint64_t init_steady_clock = std::chrono::steady_clock::now().time_since_epoch().count();
  uint64_t curr_steady_clock = 0;
  while (!cpu_timestamp_update_thread_should_stop) {
    curr_steady_clock = std::chrono::steady_clock::now().time_since_epoch().count();
    *volatile_cpu_timestamp = init_system_clock + (curr_steady_clock - init_steady_clock);
  }
}

ncclResult_t NpKit::Init(int rank) {
  uint64_t i = 0;
  NpKitEventCollectContext ctx;
  ctx.event_buffer_head = 0;
  rank_ = rank;

  // Init event data structures
  NCCLCHECK(ncclCalloc(&gpu_event_buffers_, kNumGpuEventBuffers));
  NCCLCHECK(ncclCudaCalloc(&gpu_collect_contexts_, kNumGpuEventBuffers));
  for (i = 0; i < kNumGpuEventBuffers; i++) {
    NCCLCHECK(ncclCudaCalloc(gpu_event_buffers_ + i, kMaxNumGpuEventsPerBuffer));
    ctx.event_buffer = gpu_event_buffers_[i];
    NCCLCHECK(ncclCudaMemcpy(gpu_collect_contexts_ + i, &ctx, 1));
  }

  NCCLCHECK(ncclCalloc(&cpu_event_buffers_, kNumCpuEventBuffers));
  NCCLCHECK(ncclCalloc(&cpu_collect_contexts_, kNumCpuEventBuffers));
  for (i = 0; i < kNumCpuEventBuffers; i++) {
    NCCLCHECK(ncclCalloc(cpu_event_buffers_ + i, kMaxNumCpuEventsPerBuffer));
    ctx.event_buffer = cpu_event_buffers_[i];
    cpu_collect_contexts_[i] = ctx;
  }

  // Calibrate CPU timestamp
  uint64_t cpu_timestamp_1 = std::chrono::steady_clock::now().time_since_epoch().count();
  base_cpu_timestamp_global_ = std::chrono::system_clock::now().time_since_epoch().count();
  uint64_t cpu_timestamp_2 = std::chrono::steady_clock::now().time_since_epoch().count();
  base_cpu_timestamp_local_ = cpu_timestamp_1 + (cpu_timestamp_2 - cpu_timestamp_1) / 2;

  // Calibrate GPU timestamp
  uint64_t *cpu_timestamp_in = nullptr;
  uint64_t *cpu_timestamp_out = nullptr;
  uint64_t *gpu_timestamp = nullptr;
  NCCLCHECK(ncclCudaHostCalloc(&cpu_timestamp_in, 1));
  NCCLCHECK(ncclCudaCalloc(&cpu_timestamp_out, 1));
  NCCLCHECK(ncclCudaCalloc(&gpu_timestamp, 1));

  cpu_timestamp_update_thread_should_stop = false;
  volatile_cpu_timestamp = cpu_timestamp_in;
  *volatile_cpu_timestamp = 0;

  hipStream_t timingStream;
  CUDACHECK(hipStreamCreateWithFlags(&timingStream, hipStreamNonBlocking));

  std::thread cpu_timestamp_update_thread(CpuTimestampUpdateThread);
  while (!*volatile_cpu_timestamp);
  TimeCalibrationKernel<<<1, 1, 0, timingStream>>>(gpu_timestamp, cpu_timestamp_in, cpu_timestamp_out);
  CUDACHECK(hipStreamSynchronize(timingStream));
  cpu_timestamp_update_thread_should_stop = true;
  cpu_timestamp_update_thread.join();

  NCCLCHECK(ncclCudaMemcpy(&base_gpu_timestamp_cpu_, cpu_timestamp_out, 1));
  NCCLCHECK(ncclCudaMemcpy(&base_gpu_timestamp_gpu_, gpu_timestamp, 1));

  NCCLCHECK(ncclCudaHostFree(cpu_timestamp_in));
  CUDACHECK(hipFree(cpu_timestamp_out));
  CUDACHECK(hipFree(gpu_timestamp));

  return ncclSuccess;
}

ncclResult_t NpKit::Dump(const std::string& dump_dir) {
  uint64_t i = 0;
  std::string dump_file_path;

  // Dump CPU events
  for (i = 0; i < kNumCpuEventBuffers; i++) {
    dump_file_path = dump_dir;
    dump_file_path += "/cpu_events_rank_";
    dump_file_path += std::to_string(rank_);
    dump_file_path += "_channel_";
    dump_file_path += std::to_string(i);
    auto cpu_trace_file = std::fstream(dump_file_path, std::ios::out | std::ios::binary);
    cpu_trace_file.write(reinterpret_cast<char*>(cpu_event_buffers_[i]),
        cpu_collect_contexts_[i].event_buffer_head * sizeof(NpKitEvent));
    cpu_trace_file.close();
  }

  // Dump CPU clock info
  dump_file_path = dump_dir;
  dump_file_path += "/cpu_clock_period_num_rank_";
  dump_file_path += std::to_string(rank_);
  std::string clock_period_num_str = std::to_string(std::chrono::steady_clock::duration::period::num);
  auto clock_period_num_file = std::fstream(dump_file_path, std::ios::out);
  clock_period_num_file.write(clock_period_num_str.c_str(), clock_period_num_str.length());
  clock_period_num_file.close();

  dump_file_path = dump_dir;
  dump_file_path += "/cpu_clock_period_den_rank_";
  dump_file_path += std::to_string(rank_);
  std::string clock_period_den_str = std::to_string(std::chrono::steady_clock::duration::period::den);
  auto clock_period_den_file = std::fstream(dump_file_path, std::ios::out);
  clock_period_den_file.write(clock_period_den_str.c_str(), clock_period_den_str.length());
  clock_period_den_file.close();

  // Dump GPU events, reuse CPU struct
  for (i = 0; i < kNumGpuEventBuffers; i++) {
    dump_file_path = dump_dir;
    dump_file_path += "/gpu_events_rank_";
    dump_file_path += std::to_string(rank_);
    dump_file_path += "_buf_";
    dump_file_path += std::to_string(i);
    NCCLCHECK(ncclCudaMemcpy(cpu_event_buffers_[0], gpu_event_buffers_[i], kMaxNumGpuEventsPerBuffer));
    NCCLCHECK(ncclCudaMemcpy(cpu_collect_contexts_, gpu_collect_contexts_ + i, 1));
    auto gpu_trace_file = std::fstream(dump_file_path, std::ios::out | std::ios::binary);
    gpu_trace_file.write(reinterpret_cast<char*>(cpu_event_buffers_[0]),
        cpu_collect_contexts_[0].event_buffer_head * sizeof(NpKitEvent));
    gpu_trace_file.close();
  }

  // Dump GPU clockRate
  dump_file_path = dump_dir;
  dump_file_path += "/gpu_clock_rate_rank_";
  dump_file_path += std::to_string(rank_);
  constexpr int vega_gpu_rtc_freq_in_khz = 25000;
  std::string clock_rate_str = std::to_string(vega_gpu_rtc_freq_in_khz);
  auto gpu_clock_rate_file = std::fstream(dump_file_path, std::ios::out);
  gpu_clock_rate_file.write(clock_rate_str.c_str(), clock_rate_str.length());
  gpu_clock_rate_file.close();

  // Dump clock calibration info
  dump_file_path = dump_dir;
  dump_file_path += "/clock_calibration_cpu_global_rank_";
  dump_file_path += std::to_string(rank_);
  std::string base_cpu_timestamp_global_str = std::to_string(base_cpu_timestamp_global_);
  auto base_cpu_timestamp_global_file = std::fstream(dump_file_path, std::ios::out);
  base_cpu_timestamp_global_file.write(base_cpu_timestamp_global_str.c_str(), base_cpu_timestamp_global_str.length());
  base_cpu_timestamp_global_file.close();

  dump_file_path = dump_dir;
  dump_file_path += "/clock_calibration_cpu_local_rank_";
  dump_file_path += std::to_string(rank_);
  std::string base_cpu_timestamp_local_str = std::to_string(base_cpu_timestamp_local_);
  auto base_cpu_timestamp_local_file = std::fstream(dump_file_path, std::ios::out);
  base_cpu_timestamp_local_file.write(base_cpu_timestamp_local_str.c_str(), base_cpu_timestamp_local_str.length());
  base_cpu_timestamp_local_file.close();

  dump_file_path = dump_dir;
  dump_file_path += "/clock_calibration_gpu_cpu_rank_";
  dump_file_path += std::to_string(rank_);
  std::string base_gpu_timestamp_cpu_str = std::to_string(base_gpu_timestamp_cpu_);
  auto base_gpu_timestamp_cpu_file = std::fstream(dump_file_path, std::ios::out);
  base_gpu_timestamp_cpu_file.write(base_gpu_timestamp_cpu_str.c_str(), base_gpu_timestamp_cpu_str.length());
  base_gpu_timestamp_cpu_file.close();

  dump_file_path = dump_dir;
  dump_file_path += "/clock_calibration_gpu_gpu_rank_";
  dump_file_path += std::to_string(rank_);
  std::string base_gpu_timestamp_gpu_str = std::to_string(base_gpu_timestamp_gpu_);
  auto base_gpu_timestamp_gpu_file = std::fstream(dump_file_path, std::ios::out);
  base_gpu_timestamp_gpu_file.write(base_gpu_timestamp_gpu_str.c_str(), base_gpu_timestamp_gpu_str.length());
  base_gpu_timestamp_gpu_file.close();

  return ncclSuccess;
}

ncclResult_t NpKit::Shutdown() {
  uint64_t i = 0;

  // Free CPU event data structures
  for (i = 0; i < kNumCpuEventBuffers; i++) {
    free(cpu_event_buffers_[i]);
  }
  free(cpu_event_buffers_);
  free(cpu_collect_contexts_);

  // Free GPU event data structures
  for (i = 0; i < kNumGpuEventBuffers; i++) {
    CUDACHECK(hipFree(gpu_event_buffers_[i]));
  }
  free(gpu_event_buffers_);
  CUDACHECK(hipFree(gpu_collect_contexts_));

  return ncclSuccess;
}

NpKitEventCollectContext* NpKit::GetGpuEventCollectContexts() {
  return gpu_collect_contexts_;
}

void NpKit::CollectCpuEvent(uint8_t type, uint32_t size, uint32_t rsvd, uint64_t timestamp, int channel_id) {
  uint64_t event_buffer_head = cpu_collect_contexts_[channel_id].event_buffer_head;
  if (event_buffer_head < kMaxNumCpuEventsPerBuffer) {
    NpKitEvent& event = cpu_collect_contexts_[channel_id].event_buffer[event_buffer_head];
    event.fields.type = type;
    event.fields.size = size;
    event.fields.rsvd = rsvd;
    event.fields.timestamp = timestamp;
    cpu_collect_contexts_[channel_id].event_buffer_head++;
  }
}

uint64_t NpKit::GetCpuTimestamp() {
  return std::chrono::steady_clock::now().time_since_epoch().count();
}
