/*************************************************************************
 * Copyright (c) 2015-2019, NVIDIA CORPORATION. All rights reserved.
 * Modifications Copyright (c) 2019-2021 Advanced Micro Devices, Inc. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

#include "enqueue.h"
#include "collectives.h"

NCCL_API(ncclResult_t, ncclScatter, const void* sendbuff, void* recvbuff, size_t recvcount, ncclDataType_t datatype, int root,
    ncclComm_t comm, hipStream_t stream);
ncclResult_t ncclScatter(const void* sendbuff, void* recvbuff, size_t recvcount, ncclDataType_t datatype, int root,
    ncclComm_t comm, hipStream_t stream) {
    int nRanks;
    NCCLCHECK(ncclCommCount(comm, &nRanks));
    size_t rankOffset = recvcount * ncclTypeSize(datatype);
    if (recvcount == 0) return ncclSuccess;
    int rank;
    NCCLCHECK(ncclCommUserRank(comm, &rank));
    NCCLCHECK(ncclGroupStart());
    if (rank == root) {
      for (int r=0; r<nRanks; r++)
        NCCLCHECK(ncclSend(((char*)sendbuff)+r*rankOffset, recvcount, datatype, r, comm, stream));
    }
    NCCLCHECK(ncclRecv(recvbuff, recvcount, datatype, root, comm, stream));
    NCCLCHECK(ncclGroupEnd());
    return ncclSuccess;
}
