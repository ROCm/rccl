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

NCCL_API(ncclResult_t, mscclLoadAlgo, const char *mscclAlgoFilePath, mscclAlgoHandle_t *mscclAlgoHandle);
ncclResult_t mscclLoadAlgo(const char *mscclAlgoFilePath, mscclAlgoHandle_t *mscclAlgoHandle) {
  mscclStatus& status = mscclGetStatus();

  if (status.freeAlgoHandles.size() == 0) {
    WARN("MSCCL: MSCCL_MAX_NUM_ALGOS (%d) limit reached", MSCCL_MAX_NUM_ALGOS);
    return ncclInvalidUsage;
  }
  mscclAlgoHandle_t handle = *status.freeAlgoHandles.rbegin();
  status.freeAlgoHandles.pop_back();

  struct mscclAlgo* hostAlgo;
  NCCLCHECK(ncclCalloc(&hostAlgo, 1));
  NCCLCHECK(mscclGetAlgoFromXmlFile(mscclAlgoFilePath, hostAlgo, status.rank));
  status.hostAlgos[handle] = hostAlgo;

  struct mscclAlgo* devAlgo;
  NCCLCHECK(ncclCudaCalloc(&devAlgo, 1));
  CUDACHECK(hipMemcpy(devAlgo, hostAlgo, sizeof(struct mscclAlgo), hipMemcpyHostToDevice));
  status.devAlgos[handle] = devAlgo;

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

  NCCLCHECK(mscclSetupCount(hostAlgo, comm, count, dataType));

  NCCLCHECK(mscclSetupScratch(hostAlgo, stream));

  NCCLCHECK(mscclSetupSyncFlags(stream));

  NCCLCHECK(mscclSetupConnections(hostAlgo, comm));

  NCCLCHECK(mscclSetupProxy(hostAlgo, comm));

  NCCLCHECK(mscclSetupKernel(sendBuff, recvBuff, count, dataType, op, hostAlgo, devAlgo, comm, stream));

  return ncclSuccess;
}

NCCL_API(ncclResult_t, mscclUnloadAlgo, mscclAlgoHandle_t mscclAlgoHandle);
ncclResult_t mscclUnloadAlgo(mscclAlgoHandle_t mscclAlgoHandle) {
  mscclStatus& status = mscclGetStatus();

  free(status.hostAlgos[mscclAlgoHandle]);
  status.hostAlgos.erase(mscclAlgoHandle);

  CUDACHECK(hipFree(status.devAlgos[mscclAlgoHandle]));
  status.devAlgos.erase(mscclAlgoHandle);

  status.freeAlgoHandles.push_back(mscclAlgoHandle);

  return ncclSuccess;
}
