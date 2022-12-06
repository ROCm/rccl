/*************************************************************************
 * Copyright (c) Microsoft Corporation.
 * Licensed under the MIT License.
 ************************************************************************/

#ifndef MSCCL_LIFECYCLE_H_
#define MSCCL_LIFECYCLE_H_

#include "enqueue.h"

#include "msccl/msccl_struct.h"

bool mscclEnabled();

ncclResult_t mscclInit(int rank);

ncclResult_t mscclScheduler(
    const void* sendbuff, const size_t sendcounts[], const size_t sdispls[],
    void* recvbuff, const size_t recvcounts[], const size_t rdispls[],
    size_t count, ncclDataType_t datatype, int root, int peer, ncclRedOp_t op,
    mscclFunc_t mscclFunc, bool* mscclScheduled,
    ncclComm_t comm, hipStream_t stream);

ncclResult_t mscclTeardown();

#endif
