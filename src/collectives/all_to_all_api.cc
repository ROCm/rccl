/*************************************************************************
 * Copyright (c) 2015-2019, NVIDIA CORPORATION. All rights reserved.
 * Modifications Copyright (c) 2019-2020 Advanced Micro Devices, Inc. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

#include "enqueue.h"
#include "collectives.h"

NCCL_API(ncclResult_t, ncclAllToAll, const void* sendbuff, void* recvbuff, size_t count, ncclDataType_t datatype,
  ncclComm_t comm, hipStream_t stream);
ncclResult_t ncclAllToAll(const void* sendbuff, void* recvbuff, size_t count, ncclDataType_t datatype,
  ncclComm_t comm, hipStream_t stream) {
  int nRanks;
  NCCLCHECK(ncclCommCount(comm, &nRanks));
  size_t rankOffset = count * ncclTypeSize(datatype);
  if (count == 0) return ncclSuccess;
  NCCLCHECK(ncclGroupStart());
  for (int r=0; r<nRanks; r++) {
    NCCLCHECK(ncclSend(((char*)sendbuff)+r*rankOffset, count, datatype, r, comm, stream));
    NCCLCHECK(ncclRecv(((char*)recvbuff)+r*rankOffset, count, datatype, r, comm, stream));
  }
  NCCLCHECK(ncclGroupEnd());
  return ncclSuccess;
}
