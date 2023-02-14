/*************************************************************************
 * Copyright (c) Microsoft Corporation.
 * Licensed under the MIT License.
 ************************************************************************/

#ifndef NPKIT_H_
#define NPKIT_H_

#include <string>
#include <thread>

#include <hip/hip_runtime.h>

#include "npkit/npkit_event.h"
#include "npkit/npkit_struct.h"

#if defined(__GFX9__) || defined(__GFX10__)
#define NPKIT_GET_GPU_TIMESTAMP __builtin_amdgcn_s_memrealtime
#else
#define NPKIT_GET_GPU_TIMESTAMP clock64
#endif

class NpKit {
 public:
  static const uint64_t kNumGpuEventBuffers = 512;

  static const uint64_t kNumCpuEventBuffers = 32;

  static ncclResult_t Init(int rank);

  static ncclResult_t Dump(const std::string& dump_dir);

  static ncclResult_t Shutdown();

  static NpKitEventCollectContext* GetGpuEventCollectContexts();

  static inline __device__ void CollectGpuEvent(uint8_t type, uint32_t size, uint32_t rsvd, uint64_t timestamp,
                                                NpKitEventCollectContext* ctx) {
    uint64_t event_buffer_head = ctx->event_buffer_head;
    if (event_buffer_head < kMaxNumGpuEventsPerBuffer) {
      NpKitEvent& event = ctx->event_buffer[event_buffer_head];
      event.fields.type = type;
      event.fields.size = size;
      event.fields.rsvd = rsvd;
      event.fields.timestamp = timestamp;
      ctx->event_buffer_head++;
    }
  }

  static void CollectCpuEvent(uint8_t type, uint32_t size, uint32_t rsvd, uint64_t timestamp, int channel_id);

  static uint64_t GetCpuTimestamp();

 private:
  static void CpuTimestampUpdateThread();

  // 64K * 512 * 16B = 512MB per GPU
  static const uint64_t kMaxNumGpuEventsPerBuffer = 1ULL << 16;

  // 64K * 2 (send/recv) * (512/32) = 2M, 2M * 32 * 16B = 1GB per CPU
  static const uint64_t kMaxNumCpuEventsPerBuffer = 1ULL << 21;

  static NpKitEvent** gpu_event_buffers_;
  static NpKitEvent** cpu_event_buffers_;

  static NpKitEventCollectContext* gpu_collect_contexts_;
  static NpKitEventCollectContext* cpu_collect_contexts_;

  static uint64_t base_cpu_timestamp_global_;
  static uint64_t base_cpu_timestamp_local_;
  static uint64_t base_gpu_timestamp_cpu_;
  static uint64_t base_gpu_timestamp_gpu_;

  static uint64_t rank_;
};

#endif
