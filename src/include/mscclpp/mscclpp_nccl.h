/*************************************************************************
 * Copyright (c) 2024 Advanced Micro Devices, Inc. All rights reserved.
 *
 * See LICENSE.txt and NOTICES.txt for license information
 ************************************************************************/

#ifndef MSCCLPP_NCCL_H_
#define MSCCLPP_NCCL_H_

#include "nccl.h"
#include <unordered_map>

typedef struct mscclppComm* mscclppComm_t;

typedef ncclUniqueId mscclppUniqueId;

bool mscclpp_init();

/* A ncclUniqueId and a mscclppUniqueId will always be created together and used alternatively. This maps between them. */
extern std::unordered_map<ncclUniqueId, mscclppUniqueId> mscclpp_uniqueIdMap;

/* See ncclGetUniqueId. */
extern ncclResult_t  (*mscclpp_ncclGetUniqueId)(mscclppUniqueId* uniqueId);

/* See ncclCommInitRank. */
extern ncclResult_t  (*mscclpp_ncclCommInitRank)(mscclppComm_t* comm, int nranks, mscclppUniqueId commId, int rank);

/* See ncclCommDestroy. */
extern ncclResult_t  (*mscclpp_ncclCommDestroy)(mscclppComm_t comm);

/* See ncclAllReduce. */
extern ncclResult_t  (*mscclpp_ncclAllReduce)(const void* sendbuff, void* recvbuff, size_t count,
    ncclDataType_t datatype, ncclRedOp_t op, mscclppComm_t comm, hipStream_t stream);

/* See ncclAllGather. */
extern ncclResult_t  (*mscclpp_ncclAllGather)(const void* sendbuff, void* recvbuff, size_t sendcount,
    ncclDataType_t datatype, mscclppComm_t comm, hipStream_t stream);

namespace std {
  template <>
  struct hash<ncclUniqueId> {
    size_t operator ()(const ncclUniqueId& uniqueId) const noexcept;
  };
}

bool operator ==(const ncclUniqueId& a, const ncclUniqueId& b);

#endif
