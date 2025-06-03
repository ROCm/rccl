// (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

#include "CollTraceEvent.h"
#include "param.h"

NCCL_PARAM(ColltraceCheckIntervalMs, "COLLTRACE_CHECK_INTERVAL_MS", 10);

namespace meta::colltrace {

ncclResult_t CudaWaitEvent::waitEventFinish() {
  // async polling case, query cuda whether event is ready every
  // NCCL_COLLTRACE_CHECK_INTERVAL_MS ms
  auto res = cudaEventQuery(event_.get());
  while (res != cudaSuccess) {
    if (res != cudaErrorNotReady) {
      CUDACHECK(res);
    }
    std::this_thread::sleep_for(
        std::chrono::milliseconds(ncclParamColltraceCheckIntervalMs()));
    res = cudaEventQuery(event_.get());
  }
  return ncclSuccess;
}

std::optional<float> CudaWaitEvent::getElapsedTimeSinceEvent(
    CudaWaitEvent* start) {
  float elapsedTime;
  auto res =
      cudaEventElapsedTime(&elapsedTime, start->event_.get(), event_.get());
  if (res != cudaSuccess) {
    WARN("get elapsed time failed error: %s, tag: %s", cudaGetErrorString(res));
    return std::nullopt;
  }
  return elapsedTime;
}

} // namespace meta::colltrace
