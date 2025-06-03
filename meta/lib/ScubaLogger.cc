// (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

#include "ScubaLogger.h"

#include <fmt/format.h>
#include <folly/logging/xlog.h>
#include <memory>

#include "dsi/logger/configs/RcclCollTraceLoggerConfig/Logger.h"

namespace facebook::rcclx {

ScubaLogger::ScubaLogger() noexcept
    : hostnameAndPid_(initHostnamePid()), jobIdStr_(initJobIdStr()) {}

std::string ScubaLogger::tryGetEnvStr(
    const char* envName,
    const std::string& defVal) const {
  const char* env = std::getenv(envName);
  if (env == nullptr) {
    return defVal;
  }
  return std::string(env);
}

uint64_t ScubaLogger::getCurrentTimestamp() const {
  auto now = std::chrono::system_clock::now();
  return std::chrono::duration_cast<std::chrono::microseconds>(
             now.time_since_epoch())
      .count();
}

std::string ScubaLogger::initHostnamePid() const {
  char hostname[HOST_NAME_MAX] = "unknown";
  int process_id = getpid();
  if (gethostname(hostname, HOST_NAME_MAX) != 0) {
    XLOG(ERR) << "Failed to get hostname: " << strerror(errno);
  }
  std::ostringstream oss;
  oss << hostname << ":" << process_id;
  return oss.str();
}

std::string ScubaLogger::initJobIdStr() const {
  auto jobName = tryGetEnvStr(kJobNameEnv, "");
  if (jobName == "") {
    XLOG(WARN) << "detected non-Mast job, use empty job id";
    return "";
  }
  auto jobVersion = tryGetEnvStr(kJobVersionEnv);
  auto jobAttempt = tryGetEnvStr(kJobAttemptEnv);
  return fmt::format("{}:{}:{}", jobName, jobVersion, jobAttempt);
}

static folly::Singleton<facebook::rcclx::ScubaLogger> scubaLoggerSingleton;

std::shared_ptr<facebook::rcclx::ScubaLogger>
facebook::rcclx::ScubaLogger::getInstance() {
  return scubaLoggerSingleton.try_get();
}

void ScubaLogger::logCollEnqueueTrace(RcclCollTraceLogPayload&& payload) const {
  facebook::logger::RcclCollTraceLogger logRecord;
  logRecord.setTimeUS(getCurrentTimestamp())
      .setHostProcessInfo(hostnameAndPid_)
      .setGlobalRank(payload.global_rank)
      .setJobInfo(jobIdStr_)
      .setCollName(payload.coll_name)
      .setOpCount(payload.op_count)
      .setRoot(payload.root)
      .setCudaDev(payload.cuda_dev)
      .setElems(payload.count)
      .setNRanks(payload.n_ranks)
      .setTaskID(payload.task_id)
      .setHipStream(fmt::format("{:p}", payload.hip_stream))
      .setCommhash(payload.comm_hash)
      .setSendbuff(fmt::format("{:p}", payload.sendbuff))
      .setRecvbuff(fmt::format("{:p}", payload.recvbuff))
      .setRedOp(static_cast<facebook::logger::RcclCollTraceLoggerEnum::red_op>(
          payload.red_op))
      .setDataType(
          static_cast<facebook::logger::RcclCollTraceLoggerEnum::data_type>(
              payload.data_type));
  LOG_VIA_LOGGER_ASYNC(logRecord);
}

void ScubaLogger::logCollEnqueueTrace(
    const void* sendBuff,
    const size_t sendCounts[],
    const size_t sDisPls[],
    const void* recvBuff,
    const size_t recvCounts[],
    const size_t rDisPls[],
    size_t count,
    uint8_t dataType,
    uint8_t op,
    int root,
    int peer,
    int32_t cudaDev,
    int32_t globalRank,
    int32_t nRanks,
    int64_t opCount,
    int32_t taskId,
    const char* funcName,
    uint64_t commHash,
    const void* stream) const {
  facebook::logger::RcclCollTraceLogger logRecord;
  logRecord.setTimeUS(getCurrentTimestamp())
      .setHostProcessInfo(hostnameAndPid_)
      .setGlobalRank(globalRank)
      .setJobInfo(jobIdStr_)
      .setCollName(funcName)
      .setOpCount(opCount)
      .setRoot(root)
      .setCudaDev(cudaDev)
      .setElems(count)
      .setNRanks(nRanks)
      .setTaskID(taskId)
      .setHipStream(fmt::format("{:p}", stream))
      .setCommhash(commHash)
      .setSendbuff(fmt::format("{:p}", sendBuff))
      .setRecvbuff(fmt::format("{:p}", recvBuff))
      .setRedOp(
          static_cast<facebook::logger::RcclCollTraceLoggerEnum::red_op>(op))
      .setDataType(
          static_cast<facebook::logger::RcclCollTraceLoggerEnum::data_type>(
              dataType));
  LOG_VIA_LOGGER_ASYNC(logRecord);
}
} // namespace facebook::rcclx
