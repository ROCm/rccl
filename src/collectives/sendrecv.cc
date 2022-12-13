/*************************************************************************
 * Copyright (c) 2015-2022, NVIDIA CORPORATION. All rights reserved.
 * Modifications Copyright (c) Microsoft Corporation. Licensed under the MIT License.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

#include "enqueue.h"
#include "collectives.h"
#include "argcheck.h" // Need some checks here since we access comm

#include "msccl/msccl_lifecycle.h"

NCCL_API(ncclResult_t, ncclSend, const void* sendbuff, size_t count, ncclDataType_t datatype, int peer,
    ncclComm_t comm, cudaStream_t stream);
ncclResult_t ncclSend(const void* sendbuff, size_t count, ncclDataType_t datatype, int peer,
    ncclComm_t comm, cudaStream_t stream) {
  NVTX3_FUNC_RANGE_IN(nccl_domain);

  if (mscclAvailable() && !mscclIsCaller()) {
    return mscclEnqueueCheck(
      sendbuff, nullptr, nullptr, nullptr, nullptr, nullptr,
      count, datatype, 0, peer, ncclSum, mscclFuncSend, comm, stream);
  }

  struct ncclInfo info = { ncclFuncSend, "Send",
    NULL, (void*)sendbuff, count, datatype, ncclSum, peer, comm, stream, /* Args */
    1, 1 };
  ncclResult_t ret;
  NCCLCHECK(ncclGroupStart());
  ret = ncclEnqueueCheck(&info);
  NCCLCHECK(ncclGroupEnd());
  return ret;
}

NCCL_API(ncclResult_t, ncclRecv, void* recvbuff, size_t count, ncclDataType_t datatype, int peer,
    ncclComm_t comm, cudaStream_t stream);
ncclResult_t ncclRecv(void* recvbuff, size_t count, ncclDataType_t datatype, int peer,
    ncclComm_t comm, cudaStream_t stream) {
  NVTX3_FUNC_RANGE_IN(nccl_domain);

  if (mscclAvailable() && !mscclIsCaller()) {
    return mscclEnqueueCheck(
      nullptr, nullptr, nullptr, recvbuff, nullptr, nullptr,
      count, datatype, 0, peer, ncclSum, mscclFuncRecv, comm, stream);
  }

  struct ncclInfo info = { ncclFuncRecv, "Recv",
    NULL, recvbuff, count, datatype, ncclSum, peer, comm, stream, /* Args */
    1, 1 };
  ncclResult_t ret;
  NCCLCHECK(ncclGroupStart());
  ret = ncclEnqueueCheck(&info);
  NCCLCHECK(ncclGroupEnd());
  return ret;
}
