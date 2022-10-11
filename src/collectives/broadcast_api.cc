/*************************************************************************
 * Copyright (c) 2015-2020, NVIDIA CORPORATION. All rights reserved.
 * Modifications Copyright (c) 2019-2021 Advanced Micro Devices, Inc. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

#include "enqueue.h"
#include "collectives.h"

RCCL_PARAM(BroadcastUseAllReduce, "BROADCAST_USE_ALLREDUCE", 0);

NCCL_API(ncclResult_t, ncclBroadcast, const void* sendbuff, void* recvbuff, size_t count, ncclDataType_t datatype, int root,
    ncclComm_t comm, hipStream_t stream);
ncclResult_t ncclBroadcast(const void* sendbuff, void* recvbuff, size_t count, ncclDataType_t datatype, int root,
    ncclComm_t comm, hipStream_t stream) {
  NVTX3_FUNC_RANGE_IN(nccl_domain);
  if (rcclParamBroadcastUseAllReduce()) {
      if (comm->rank != root)
        hipMemsetAsync(const_cast<void*>(recvbuff), 0, count * ncclTypeSize(datatype), stream);
      struct ncclInfo info = { ncclFuncAllReduce, "AllReduce",
        comm->rank == root ? sendbuff : recvbuff, recvbuff, count, datatype, ncclSum, 0, comm, stream, /* Args */
        ALLREDUCE_CHUNKSTEPS, ALLREDUCE_SLICESTEPS };
      return ncclEnqueueCheck(&info);
  } else {
      struct ncclInfo info = { ncclFuncBroadcast, "Broadcast",
        sendbuff, recvbuff, count, datatype, ncclSum, root, comm, stream, /* Args */
        BROADCAST_CHUNKSTEPS, BROADCAST_SLICESTEPS };
      return ncclEnqueueCheck(&info);
  }
}
/* Deprecated original "in place" function, similar to MPI */
NCCL_API(ncclResult_t, ncclBcast, void* buff, size_t count, ncclDataType_t datatype, int root,
    ncclComm_t comm, hipStream_t stream);
ncclResult_t ncclBcast(void* buff, size_t count, ncclDataType_t datatype, int root,
    ncclComm_t comm, hipStream_t stream) {
  return ncclBroadcast(buff, buff, count, datatype, root, comm, stream);
}

