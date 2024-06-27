/*************************************************************************
 * Copyright (c) 2024 Advanced Micro Devices, Inc. All rights reserved.
 *
 * See LICENSE.txt and NOTICES.txt for license information
 ************************************************************************/

#ifndef MSCCLPP_NCCL_H_
#define MSCCLPP_NCCL_H_

#include "nccl.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef struct mscclpp_ncclComm* mscclpp_ncclComm_t;

typedef struct { char internal[NCCL_UNIQUE_ID_BYTES]; } mscclpp_ncclUniqueId;

/* Generates an Id to be used in mscclpp_ncclCommInitRank. mscclpp_ncclGetUniqueId should be
 * called once and the Id should be distributed to all ranks in the
 * communicator before calling mscclpp_ncclCommInitRank. */
extern ncclResult_t  (*mscclpp_ncclGetUniqueId)(mscclpp_ncclUniqueId* uniqueId);

/* Creates a new communicator (multi thread/process version).
 * rank must be between 0 and nranks-1 and unique within a communicator clique.
 * Each rank is associated to a CUDA device, which has to be set before calling
 * mscclpp_ncclCommInitRank.
 * mscclpp_ncclCommInitRank implicitly syncronizes with other ranks, so it must be
 * called by different threads/processes or use mscclpp_ncclGroupStart/mscclpp_ncclGroupEnd. */
extern ncclResult_t  (*mscclpp_ncclCommInitRank)(mscclpp_ncclComm_t* comm, int nranks, mscclpp_ncclUniqueId commId, int rank);

/* Frees local resources associated with communicator object. */
extern ncclResult_t  (*mscclpp_ncclCommDestroy)(mscclpp_ncclComm_t comm);

/*
 * All-Reduce
 *
 * Reduces data arrays of length count in sendbuff using op operation, and
 * leaves identical copies of result on each recvbuff.
 *
 * In-place operation will happen if sendbuff == recvbuff.
 */
extern ncclResult_t  (*mscclpp_ncclAllReduce)(const void* sendbuff, void* recvbuff, size_t count,
    ncclDataType_t datatype, ncclRedOp_t op, mscclpp_ncclComm_t comm, hipStream_t stream);

/*
 * All-Gather
 *
 * Each device gathers sendcount values from other GPUs into recvbuff,
 * receiving data from rank i at offset i*sendcount.
 * Assumes recvcount is equal to nranks*sendcount, which means that recvbuff
 * should have a size of at least nranks*sendcount elements.
 *
 * In-place operations will happen if sendbuff == recvbuff + rank * sendcount.
 */
extern ncclResult_t  (*mscclpp_ncclAllGather)(const void* sendbuff, void* recvbuff, size_t sendcount,
    ncclDataType_t datatype, mscclpp_ncclComm_t comm, hipStream_t stream);

#ifdef __cplusplus
}
#endif

#endif
