/*
Copyright (c) 2025 Advanced Micro Devices, Inc. All rights reserved.

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

#pragma once

namespace rccl::cooperative_threadfence {
using signal_type = uint64_t;

constexpr __device__ __host__ ssize_t getMax(ssize_t val1, ssize_t val2){
  return (val1 > val2 ? val1 : val2);
}
constexpr __device__ __host__ ssize_t getExponent(ssize_t channels){
  return (CHAR_BIT * sizeof(signal_type) - __builtin_clzll(channels) - 2 + 
    (__builtin_popcountll(channels) != 1 ? 1 : 0));
}
constexpr __device__ __host__ signal_type getCohortSize(ssize_t channels = gridDim.x) {
  return 1llu << getMax(0, getExponent(channels)); 
}

// Ye olde compile time testing to ensure the cohort size is correct
static_assert(getCohortSize(1) == 1, "Cohort size for 1 channel should be 1");
static_assert(getCohortSize(2) == 1, "Cohort size for 2 channels should be 1");
static_assert(getCohortSize(3) == 2, "Cohort size for 3 channels should be 2");
static_assert(getCohortSize(4) == 2, "Cohort size for 4 channels should be 2");
static_assert(getCohortSize(8) == 4, "Cohort size for 8 channels should be 4");
static_assert(getCohortSize(56) == 32, "Cohort size for 56 channels should be 32");
static_assert(getCohortSize(64) == 32, "Cohort size for 64 channels should be 32");
static_assert(getCohortSize(112) == 64, "Cohort size for 128 channels should be 64");

struct RCCLCooperativeThreadFence {
    signal_type* __restrict__ want_to_signal_;
    signal_type* __restrict__ signaled_;
    signal_type cohort_size_ = getCohortSize();
    

    explicit __device__ RCCLCooperativeThreadFence(): want_to_signal_(nullptr), signaled_(nullptr) {}
    
    explicit __device__ RCCLCooperativeThreadFence(signal_type* want_to_signal, signal_type* signaled)
        : want_to_signal_(want_to_signal), signaled_(signaled) {}

    __device__ __forceinline__ void operator()(){
        constexpr signal_type max_spins = 500;

        signal_type prev_count = __hip_atomic_fetch_add(want_to_signal_, 1, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_AGENT);
        if (prev_count % (cohort_size_) == cohort_size_ - 1){
          __threadfence();
          __hip_atomic_fetch_add(signaled_, prev_count + 1, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_AGENT);
        }
        else{
          // Wait for flushing thread to signal that the L2 cache has completed flush and invalidation
          signal_type read_signal = 0;
          // Calculate what value the counter was at immediately after the previous flush
          signal_type signal_condition = prev_count + cohort_size_ - (prev_count & (cohort_size_ - 1));
          signal_type spins = 0;
          while (read_signal < signal_condition && spins < max_spins) {
            read_signal = __hip_atomic_load(signaled_, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_AGENT);
          }
          // Handle very degenerate cases where one thread is waiting for a signal that never comes
          if (spins >= max_spins){
              __threadfence();
              __hip_atomic_fetch_add(signaled_, prev_count + 1, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_AGENT);
              #ifdef RCCL_TRACK_THREADFENCE_FALLBACK
                printf("RCCL Warning: Fallback hit, indicating a potential issue with the signaling mechanism.\n");
              }
              #endif
          }
          else{
            __threadfence_block();
          }
        }
    }
};

} // namespace rccl::cooperative_threadfence