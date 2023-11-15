/*************************************************************************
 * Copyright (c) Microsoft Corporation.
 * Licensed under the MIT License.
 ************************************************************************/

#include "enqueue.h"
#include "msccl/msccl_parser.h"
#include "msccl/msccl_setup.h"
#include "msccl/msccl_status.h"
#include <cstdio>
#include <cstdlib>

NCCL_API(ncclResult_t, mscclLoadAlgo, const char *mscclAlgoFilePath, mscclAlgoHandle_t *mscclAlgoHandle, int rank);
ncclResult_t mscclLoadAlgo(const char *mscclAlgoFilePath, mscclAlgoHandle_t *mscclAlgoHandle, int rank) {
  mscclStatus& status = mscclGetStatus();

  if (status.freeAlgoHandles.size() == 0) {
    WARN("MSCCL: MSCCL_MAX_NUM_ALGOS (%d) limit reached", MSCCL_MAX_NUM_ALGOS);
    return ncclInvalidUsage;
  }
  *mscclAlgoHandle = *status.freeAlgoHandles.rbegin();
  status.freeAlgoHandles.pop_back();

  struct mscclAlgo* hostAlgo;
  NCCLCHECK(ncclCalloc(&hostAlgo, 1));
  NCCLCHECK(mscclGetAlgoFromXmlFile(mscclAlgoFilePath, hostAlgo, rank));
  status.hostAlgos[*mscclAlgoHandle] = hostAlgo;

  struct mscclAlgo* devAlgo;
  NCCLCHECK(ncclCudaMalloc(&devAlgo, 1));
  CUDACHECK(hipMemcpy(devAlgo, hostAlgo, sizeof(struct mscclAlgo), hipMemcpyHostToDevice));
  status.devAlgos[*mscclAlgoHandle] = devAlgo;

  return ncclSuccess;
}

NCCL_API(ncclResult_t, mscclRunAlgo,
    const void* sendBuff, const size_t sendCounts[], const size_t sDisPls[],
    void* recvBuff, const size_t recvCounts[], const size_t rDisPls[],
    size_t count, ncclDataType_t dataType, int root, int peer, ncclRedOp_t op,
    mscclAlgoHandle_t mscclAlgoHandle, ncclComm_t comm, hipStream_t stream);
ncclResult_t mscclRunAlgo(
    const void* sendBuff, const size_t sendCounts[], const size_t sDisPls[],
    void* recvBuff, const size_t recvCounts[], const size_t rDisPls[],
    size_t count, ncclDataType_t dataType, int root, int peer, ncclRedOp_t op,
    mscclAlgoHandle_t mscclAlgoHandle, ncclComm_t comm, hipStream_t stream) {
  mscclStatus& status = mscclGetStatus();
  struct mscclAlgo* hostAlgo = status.hostAlgos[mscclAlgoHandle];
  struct mscclAlgo* devAlgo = status.devAlgos[mscclAlgoHandle];

  NCCLCHECK(mscclGetCaptureStatus(stream));

  NCCLCHECK(mscclSetupCount(hostAlgo, comm, count, dataType));

  NCCLCHECK(mscclSetupScratch(hostAlgo, stream));

  NCCLCHECK(mscclSetupSyncFlags(stream));

  if (status.connectedAlgos[comm].find(mscclAlgoHandle) == status.connectedAlgos[comm].end()) {
    hipStreamCaptureMode mode = hipStreamCaptureModeRelaxed;
    CUDACHECK(hipThreadExchangeStreamCaptureMode(&mode));
    NCCLCHECK(mscclSetupConnections(hostAlgo, comm));
    CUDACHECK(hipThreadExchangeStreamCaptureMode(&mode));
    status.connectedAlgos[comm].insert(mscclAlgoHandle);
  }

  NCCLCHECK(mscclSetupProxy(hostAlgo, comm, stream));

  NCCLCHECK(mscclSetupKernel(sendBuff, recvBuff, count, dataType, op, hostAlgo, devAlgo, comm, stream));

  return ncclSuccess;
}

NCCL_API(ncclResult_t, mscclUnloadAlgo, mscclAlgoHandle_t mscclAlgoHandle);
ncclResult_t mscclUnloadAlgo(mscclAlgoHandle_t mscclAlgoHandle) {
  mscclStatus& status = mscclGetStatus();

  free(status.hostAlgos[mscclAlgoHandle]);
  status.hostAlgos.erase(mscclAlgoHandle);

  NCCLCHECK(ncclCudaFree(status.devAlgos[mscclAlgoHandle]));
  status.devAlgos.erase(mscclAlgoHandle);

  status.freeAlgoHandles.push_back(mscclAlgoHandle);

  for (auto &s : status.connectedAlgos) {
    s.second.erase(mscclAlgoHandle);
  }

  return ncclSuccess;
}
