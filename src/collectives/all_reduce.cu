/*************************************************************************
 * Copyright (c) 2015-2018, NVIDIA CORPORATION. All rights reserved.
 * Modifications Copyright (c) 2019 Advanced Micro Devices, Inc. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

#include "core.h"
#include "common_coll.h"
#include "enqueue.h"
#include "collectives.h"

ncclResult_t ncclAllReduceFunc(const void* sendbuff, void* recvbuff, size_t count,
    ncclDataType_t datatype, ncclRedOp_t op, int root, ncclComm_t comm, hipStream_t stream) {
  size_t nbytes = count*ncclTypeSize(datatype);
  INFO(NCCL_COLL,"AllReduce: opCount %lx sendbuff %p recvbuff %p count %zi datatype %d op %d root %d comm %p [nranks=%d] stream %p", comm->opCount, sendbuff, recvbuff, count, datatype, op, root, comm, comm->nRanks, stream);
  if (comm->nRanks == 1) {
    if (sendbuff != recvbuff)
      CUDACHECK(hipMemcpyAsync(recvbuff, sendbuff, nbytes, hipMemcpyDeviceToDevice, stream));
  } else {
    NCCLCHECK(transportSaveProxies(ALLREDUCE_SUBSTEPS, ALLREDUCE_BUFCHUNKS, (comm->nRanks)*2-2, comm->nRanks, nbytes, proxyPatternRing, comm));
    NCCLCHECK(saveKernel(ncclCollAllReduce, sendbuff, recvbuff, count, datatype, op, root, comm, stream, nbytes, comm->nRanks));
  }
  return ncclSuccess;
}

NCCL_API(ncclResult_t, ncclAllReduce, const void* sendbuff, void* recvbuff, size_t count,
    ncclDataType_t datatype, ncclRedOp_t op, ncclComm_t comm, hipStream_t stream);
ncclResult_t ncclAllReduce(const void* sendbuff, void* recvbuff, size_t count,
    ncclDataType_t datatype, ncclRedOp_t op, ncclComm_t comm, hipStream_t stream) {
  return ncclEnqueueCheck(ncclAllReduceFunc, "AllReduce", sendbuff, recvbuff, count, datatype,
          op, 0, comm, stream);
}
