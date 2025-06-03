#include "CollTraceFunc.h"

namespace meta::colltrace {

namespace {
bool enableCollTrace() {
  const char* colltraceEnable = ncclGetEnv("RCCL_LATENCY_PROFILER");
  if (colltraceEnable != NULL) {
    INFO(
        NCCL_INIT,
        "RCCL_LATENCY_PROFILER set by environment to %s.",
        colltraceEnable);
    if (strcmp(colltraceEnable, "1") == 0) {
      return true;
    }
  }
  return false;
}
} // namespace

ncclResult_t collTraceInit(ncclComm* comm) {
  if (!enableCollTrace()) {
    return ncclSuccess;
  }
  comm->ctrace = std::make_unique<CollTrace>(comm);
  return ncclSuccess;
}

ncclResult_t collTraceDestroy(ncclComm* comm) {
  if (comm->ctrace == nullptr) {
    return ncclSuccess;
  }
  comm->ctrace.reset();
  return ncclSuccess;
}

ncclResult_t collTraceRecordStartEvent(
    ncclComm* comm,
    cudaStream_t launchStream,
    CollTraceEvent* event) {
  if (comm->ctrace && event) {
    CUDACHECK(
        cudaEventRecord(event->start.get()->getCudaEvent(), launchStream));
  }
  return ncclSuccess;
}

ncclResult_t collTraceRecordEndEvent(
    ncclComm* comm,
    ncclKernelPlan* plan,
    cudaStream_t launchStream,
    std::unique_ptr<CollTraceEvent> event) {
  if (comm->ctrace && event) {
    CUDACHECK(cudaEventRecord(event->stop.get()->getCudaEvent(), launchStream));
    comm->ctrace->enqueueEvent(std::move(event));
  }
  return ncclSuccess;
}

CollTraceInfo parseCollInfoFromCollTask(const ncclTaskColl& collTask) {
  return CollTraceInfo{
      .opName = std::string{ncclFuncToString(collTask.func)},
      .dataType = std::string{ncclDatatypeToString(collTask.datatype)},
      .count = (int64_t)collTask.count,
  };
}

std::optional<CollTraceInfo> parseCollInfoFromNcclKernelPlan(
    ncclKernelPlan& plan,
    cudaStream_t stream) {
  if (plan.comm == nullptr || plan.comm->ctrace == nullptr) {
    return std::nullopt;
  }
  auto collTaskHead = ncclIntruQueueHead(&plan.collTaskQueue);
  if (collTaskHead == nullptr) {
    WARN("CollTrace: no coll task in this plan, this plan is empty");
    return std::nullopt;
  }

  if (collTaskHead->next != nullptr) {
    WARN(
        "CollTrace: more than one coll task in this plan, this is currently not supported");
    return std::nullopt;
  }

  CollTraceInfo collInfo = parseCollInfoFromCollTask(*collTaskHead);
  collInfo.collId = plan.comm->opCount;

  return collInfo;
}

std::unique_ptr<CollTraceEvent> collTraceAquireEventCommon(
    ncclComm* comm,
    CollTraceEvent::EventType type,
    cudaStream_t stream) {
  if (!comm->ctrace) {
    return nullptr;
  }
  struct ncclCudaGraph graph;
  auto res = ncclCudaGetCapturingGraph(&graph, stream);
  if (res != ncclSuccess) {
    WARN("Internal error: ncclCudaGetCapturingGraph failed by %d", res);
    return nullptr;
  }
  if (graph.graph != nullptr) {
    // We are in a cuda graph, this is currently unsupported
    WARN(
        "COLLTRACE: does not support cuda graph. Collectives from comm %lx will be skipped",
        comm->commHash);
    return nullptr;
  }
  auto event = comm->ctrace->createEvent(type);
  if (!event) {
    throw CollTraceError("Event init failed");
    return nullptr; /*Event init failed*/
  }
  return event;
}

std::unique_ptr<CollTraceEvent> collTraceAquireEventBaseline(
    ncclKernelPlan* plan,
    cudaStream_t stream) {
  auto collOpt = parseCollInfoFromNcclKernelPlan(*plan, stream);
  if (!collOpt.has_value()) {
    return nullptr;
  }
  auto comm = plan->comm;
  if (!comm->ctrace) {
    // WARN("COLLTRACE: comm %lx does not have ctrace", comm->commHash);
    return nullptr;
  }

  auto event =
      collTraceAquireEventCommon(comm, CollTraceEvent::EventType::COMM, stream);
  if (event == nullptr) {
    WARN("COLLTRACE: failed to aquire event");
    return nullptr;
  }
  event->coll = collOpt.value();
  return event;
}

} // namespace meta::colltrace
