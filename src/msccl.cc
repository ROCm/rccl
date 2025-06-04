/*************************************************************************
 * Copyright (c) Microsoft Corporation.
 * Licensed under the MIT License.
 ************************************************************************/

#include "enqueue.h"
#include "msccl/msccl_parser.h"
#include "msccl/msccl_setup.h"
#include "msccl/msccl_status.h"
#include "api_trace.h"
#include "nvtx_payload_schemas.h"
#include <cstdio>
#include <cstdlib>

using namespace rccl;

NCCL_API(ncclResult_t, mscclLoadAlgo, const char *mscclAlgoFilePath, mscclAlgoHandle_t *mscclAlgoHandle, const ncclComm_t comm);
ncclResult_t mscclLoadAlgo_impl(const char *mscclAlgoFilePath, mscclAlgoHandle_t *mscclAlgoHandle, const ncclComm_t comm) {
  Recorder::instance().record("mscclLoadAlgo");
  mscclStatus& status = mscclGetStatus(comm);

  if (status.freeAlgoHandles.size() == 0) {
    WARN("MSCCL: MSCCL_MAX_NUM_ALGOS (%d) limit reached", MSCCL_MAX_NUM_ALGOS);
    return ncclInvalidUsage;
  }
  *mscclAlgoHandle = *status.freeAlgoHandles.rbegin();
  status.freeAlgoHandles.pop_back();

  struct mscclAlgo* hostAlgo;
  NCCLCHECK(ncclCalloc(&hostAlgo, 1));
  NCCLCHECK(mscclGetAlgoFromXmlFile(mscclAlgoFilePath, hostAlgo, comm->rank));
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
ncclResult_t mscclRunAlgo_impl(
    const void* sendBuff, const size_t sendCounts[], const size_t sDisPls[],
    void* recvBuff, const size_t recvCounts[], const size_t rDisPls[],
    size_t count, ncclDataType_t dataType, int root, int peer, ncclRedOp_t op,
    mscclAlgoHandle_t mscclAlgoHandle, ncclComm_t comm, hipStream_t stream) {
  Recorder::instance().record("mscclRunAlgo");
  NVTX3_FUNC_WITH_PARAMS(MSCCL, NcclNvtxParamsMSCCL,
    NVTX3_PAYLOAD(comm ? comm->commHash : 0, count * ncclTypeSize(dataType), op, dataType));
  
  mscclStatus& status = mscclGetStatus(comm);
  struct mscclAlgo* hostAlgo = status.hostAlgos[mscclAlgoHandle];
  struct mscclAlgo* devAlgo = status.devAlgos[mscclAlgoHandle];

  // NCCL adds a lot of guarantees that target device is getting used
  // in its group management code, which we entirely skip when MSCCL is used
  // Therefore, in single thread multiGPU mode
  // setting the device is critical to be sure 
  // communication is done on the intended device

  CUDACHECK(hipSetDevice(comm->cudaDev)); 

  NCCLCHECK(mscclGetCaptureStatus(comm, stream));

  NCCLCHECK(mscclSetupCount(hostAlgo, comm, count, dataType));

  NCCLCHECK(mscclSetupScratch(hostAlgo, stream));

  NCCLCHECK(mscclSetupSyncFlags(comm, stream));

  NCCLCHECK(mscclSetupProxy(hostAlgo, comm, stream));

  NCCLCHECK(mscclSetupKernel(sendBuff, recvBuff, count, dataType, op, hostAlgo, devAlgo, comm, stream));

  return ncclSuccess;
}

NCCL_API(ncclResult_t, mscclUnloadAlgo, mscclAlgoHandle_t mscclAlgoHandle);
ncclResult_t mscclUnloadAlgo_impl(mscclAlgoHandle_t mscclAlgoHandle) {
  // deprecated
  Recorder::instance().record("mscclUnloadAlgo");
  return ncclSuccess;
}
