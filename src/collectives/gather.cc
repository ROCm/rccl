/*************************************************************************
 * Copyright (c) 2015-2019, NVIDIA CORPORATION. All rights reserved.
 * Modifications Copyright (c) 2019-2021 Advanced Micro Devices, Inc. All rights reserved.
 * Modifications Copyright (c) Microsoft Corporation. Licensed under the MIT License.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

#include "enqueue.h"
#include "collectives.h"

#include "msccl/msccl_lifecycle.h"

NCCL_API(ncclResult_t, ncclGather, const void* sendbuff, void* recvbuff, size_t sendcount,
    ncclDataType_t datatype, int root, ncclComm_t comm, hipStream_t stream);
ncclResult_t ncclGather(const void* sendbuff, void* recvbuff, size_t sendcount,
    ncclDataType_t datatype, int root, ncclComm_t comm, hipStream_t stream) {
    if (mscclAvailable() && !mscclIsCaller()) {
      return mscclEnqueueCheck(
        sendbuff, nullptr, nullptr, recvbuff, nullptr, nullptr,
        sendcount, datatype, root, 0, ncclSum, mscclFuncGather, comm, stream);
    }

    int nRanks;
    NCCLCHECK(ncclCommCount(comm, &nRanks));
    size_t rankOffset = sendcount * ncclTypeSize(datatype);
    if (sendcount == 0) return ncclSuccess;
    int rank;
    NCCLCHECK(ncclCommUserRank(comm, &rank));
    NCCLCHECK(ncclGroupStart());
    if (rank == root) {
      for (int r=0; r<nRanks; r++)
        NCCLCHECK(ncclRecv(((char*)recvbuff)+r*rankOffset, sendcount, datatype, r, comm, stream));
    }
    NCCLCHECK(ncclSend(sendbuff, sendcount, datatype, root, comm, stream));
    NCCLCHECK(ncclGroupEnd());
    return ncclSuccess;
}
