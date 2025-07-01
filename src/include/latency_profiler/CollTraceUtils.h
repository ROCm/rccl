// (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

#pragma once

#include <deque>
#include <memory>
#include <string>
#include <vector>

namespace meta {
namespace colltrace {

struct CollStats {
  const int collId;
  const int percent;
  const float minLatencyUs;
  const float maxLatencyUs;
  const std::string opName;
  const std::string dataType;
  int64_t count;
  CollStats(const int collId, const int percent, const float minLatencyUs, const float maxLatencyUs, const std::string& opName, const std::string& dataType, const int64_t count) : collId(collId), percent(percent), minLatencyUs(minLatencyUs), maxLatencyUs(maxLatencyUs), opName(opName), dataType(dataType), count(count) {
    
  }
};

struct CollTraceInfo {
  int64_t collId;
  std::string opName;
  std::string dataType;
  int64_t count;
  float latencyMs{-1};
};

void reportToFile(
    const std::deque<std::vector<CollStats>>& stats,
    const std::string& commHash);

std::vector<CollStats> aggregateResults(
    const std::deque<std::unique_ptr<CollTraceInfo>>& info,
    const std::vector<float>& latencyAllGather,
    int RANKS_PER_HOST,
    int NCCL_COLLTRACE_RECORD_MAX);

float getSizeMb(const std::string& dataType, int count);

}} // namespace meta::colltrace
