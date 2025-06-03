// (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.
#pragma once

#include <cuda_runtime.h>
#include <functional>
#include <memory>
#include <optional>
#include <queue>
#include "checks.h"
#include "comms/rcclx/develop/meta/lib/CollTraceUtils.h"

namespace meta::colltrace {

// CUDA event pointer w/ deleter
struct CudaEventDeleter {
  void operator()(cudaEvent_t e) {
    // Ignore error at destroy
    cudaEventDestroy(e);
  }
};
using CudaEventPtr = std::unique_ptr<
    std::pointer_traits<cudaEvent_t>::element_type,
    CudaEventDeleter>;

// a wrapper class for cuda event
class CudaWaitEvent {
 public:
  CudaWaitEvent(CudaEventPtr e) : event_(std::move(e)) {}
  ~CudaWaitEvent() {}

  cudaEvent_t getCudaEvent() {
    return event_.get();
  }

  std::optional<float> getElapsedTimeSinceEvent(CudaWaitEvent* start);
  ncclResult_t waitEventFinish();

 private:
  CudaEventPtr event_;
};

// Event data structure
struct CollTraceEvent {
  enum class EventType { COMM, TERMINATE };

  CollTraceInfo coll;
  std::unique_ptr<CudaWaitEvent> start{nullptr};
  std::unique_ptr<CudaWaitEvent> stop{nullptr};
  EventType eventType = EventType::COMM;

  CollTraceEvent(EventType type) : eventType(type) {}
  CollTraceEvent() = default;

  ~CollTraceEvent() {}

  // CollTraceEvent is not copyable
  CollTraceEvent(const CollTraceEvent&) = delete;
  CollTraceEvent& operator=(const CollTraceEvent&) = delete;

  // CollTraceEvent is movable
  CollTraceEvent(CollTraceEvent&&) = default;
  CollTraceEvent& operator=(CollTraceEvent&&) = default;
};
} // namespace meta::colltrace
