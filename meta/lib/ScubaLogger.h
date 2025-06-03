// (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

#pragma once

#include <unistd.h>
#include <cstdlib>
#include <cstring>

#include "folly/Singleton.h"

namespace facebook::rcclx {
constexpr auto kScubaTableName = "rccl_coll_trace";
// check Mast job env variables names here:
// https://www.internalfb.com/code/fbsource/fbcode/aiplatform/hpc_scheduler/if/Constants.thrift?lines=183
constexpr auto kJobNameEnv = "MAST_HPC_JOB_NAME";
constexpr auto kJobVersionEnv = "MAST_HPC_JOB_VERSION";
constexpr auto kJobAttemptEnv = "MAST_HPC_JOB_ATTEMPT_INDEX";

struct RcclCollTraceLogPayload {
  // Single-byte fields
  uint8_t ccl_type = 0;
  uint8_t red_op = 0;
  uint8_t data_type = 0;
  // 32-bit fields
  int32_t global_rank = -1;
  int32_t cuda_dev = -1;
  int32_t root = -1;
  int32_t n_ranks = -1;
  int32_t task_id = -1;
  // 64-bit fields
  uint64_t op_count = 0;
  uint64_t count = 0;
  // Pointers
  const void* sendbuff{nullptr};
  const void* recvbuff{nullptr};
  uint64_t comm_hash{0};
  const void* hip_stream{nullptr};
  // Strings
  std::string coll_name;
  std::string extra;
};

class ScubaLogger {
 public:
  /*! @brief    Log collective call trace to Scuba
    @details    The parameter list merges all possible parameters required by
    different operations as this is a general-purposed API. This API is expected
    to be called by RCCL/MSCCL/MSCCLPP
    @param[in]  sendBuff         Data array to send
    @param[in]  sendCounts       Array containing number of elements to send to
    each participating rank
    @param[in]  sDisPls          Array of offsets into *sendbuff* for each
    participating rank
    @param[in]  recvBuff         Data array to receive
    @param[in]  recvCounts       Array containing number of elements to receive
    from each participating rank
    @param[in]  rDisPls          Array of offsets into *recvbuff* for each
    participating rank
    @param[in]  count            Number of elements
    @param[in]  dataType         Data buffer element datatype
    @param[in]  op               Reduction operator
    @param[in]  root             Root rank index
    @param[in]  peer             Peer rank index, peer only makes sense for
    send/recv, and root is the peer for send/recv
    @param[in]  cudaDev          Local device index,
    @param[in]  globalRank       Global rank index
    @param[in]  nRanjobIdStr_   Number of ranks in the communicator
    @param[in]  opCount          Collective invocation count
    @param[in]  taskId           Task ID within the collective
    @param[in]  funcName         Function name of the collective
    @param[in]  commHash             Communicator group object to execute on
    @param[in]  stream           HIP stream to execute collective on */
  void logCollEnqueueTrace(
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
      uint64_t comm,
      const void* stream) const;

  // A overloaded API taking a single RcclCollTraceLogPayload object
  void logCollEnqueueTrace(RcclCollTraceLogPayload&& payload) const;

  uint64_t getCurrentTimestamp() const;

  std::string tryGetEnvStr(const char* envName, const std::string& defVal = {})
      const;

  static std::shared_ptr<facebook::rcclx::ScubaLogger> getInstance();

  inline std::string getHostnamePid() const {
    return hostnameAndPid_;
  }

  inline std::string getJobIdStr() const {
    return jobIdStr_;
  }

  ScubaLogger(const ScubaLogger&) = delete;
  ScubaLogger& operator=(const ScubaLogger&) = delete;

 private:
  ScubaLogger() noexcept;
  std::string initHostnamePid() const;
  std::string initJobIdStr() const;
  const std::string hostnameAndPid_;
  const std::string jobIdStr_;
  friend class folly::Singleton<ScubaLogger>;
};

} // namespace facebook::rcclx

#ifdef ENABLE_META_COLLTRACE
#define META_COLLTRACE_LOG_TO_SCUBA(comm, info)                        \
  do {                                                                 \
    auto loggerInstance = facebook::rcclx::ScubaLogger::getInstance(); \
    if (loggerInstance) {                                              \
      NEQCHECK(                                                        \
          (info->comm->localRank >= 0 &&                               \
           info->comm->localRank < comm->localRanks),                  \
          true);                                                       \
      loggerInstance->logCollEnqueueTrace(                             \
          info->sendbuff,                                              \
          {}, /*sendCounts*/                                           \
          {}, /*sendDispls*/                                           \
          info->recvbuff, /*recvbuff*/                                 \
          {}, /*recvcounts*/                                           \
          {}, /*recvDispls*/                                           \
          info->count,                                                 \
          info->datatype,                                              \
          info->op,                                                    \
          info->root,                                                  \
          info->root, /*peer only makes sense for send/recv*/          \
          comm->cudaDev,                                               \
          comm->localRankToRank[info->comm->localRank],                \
          comm->nRanks,                                                \
          comm->opCount,                                               \
          comm->planner.nTasksP2p + comm->planner.nTasksColl,          \
          info->opName,                                                \
          comm->commHash,                                              \
          info->stream);                                               \
    }                                                                  \
  } while (0)
#else
#define META_COLLTRACE_LOG_TO_SCUBA(comm, info) \
  do {                                          \
  } while (0)
#endif
