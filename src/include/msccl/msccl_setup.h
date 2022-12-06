/*************************************************************************
 * Copyright (c) Microsoft Corporation.
 * Licensed under the MIT License.
 ************************************************************************/

#ifndef MSCCL_SETUP_H_
#define MSCCL_SETUP_H_

#include <hip/hip_runtime.h>

#include "comm.h"
#include "msccl/msccl_struct.h"

ncclResult_t mscclSetupScratch(struct mscclAlgo* hostAlgo, hipStream_t stream);

ncclResult_t mscclSetupSyncFlags(hipStream_t stream);

ncclResult_t mscclSetupConnections(struct mscclAlgo* hostAlgo, ncclComm_t comm);

ncclResult_t mscclSetupCount(struct mscclAlgo* hostAlgo, ncclComm_t comm, size_t count, ncclDataType_t dataType);

ncclResult_t mscclSetupProxy(struct mscclAlgo* hostAlgo, ncclComm_t comm);

ncclResult_t mscclSetupKernel(const void* sendBuff, void* recvBuff, size_t count,
    ncclDataType_t dataType, ncclRedOp_t op, struct mscclAlgo* hostAlgo, struct mscclAlgo* devAlgo,
    ncclComm_t comm, hipStream_t stream);

#endif
