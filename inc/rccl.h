/*
Copyright (c) 2017-Present Advanced Micro Devices, Inc. 
All rights reserved.
*/

#pragma once

#include <hip/hip_runtime_api.h>

typedef enum {
    rcclChar    = 0, rcclInt8           = 0,
    rcclUchar   = 1, rcclUint8          = 1,
    rcclShort   = 2, rcclInt16          = 2,
    rcclUshort  = 3, rcclUint16         = 3,
    rcclInt     = 4, rcclInt32          = 4,
    rcclUint    = 5, rcclUint32         = 5,
    rcclLong    = 6, rcclInt64          = 6,
    rcclUlong   = 7, rcclUint64         = 7,
    rcclHalf    = 8, rcclFloat16        = 8,
    rcclFloat   = 9, rcclFloat32        = 9,
    rcclDouble  = 10, rcclFloat64        = 10,
    rcclNumTypes= 11,
} rcclDataType_t ;

typedef enum {
    rcclSuccess = 0,
    rcclUnhandledHipError = 1,
    rcclSystemError,
    rcclInternalError,
    rcclInvalidDevicePointer,
    rcclInvalidRank,
    rcclUnsupportedDeviceCount,
    rcclDeviceNotFound,
    rcclInvalidDeviceIndex,
    rcclLibWrapperNotSet,
    rcclHipMallocFailed,
    rcclRankMismatch,
    rcclInvalidArguments,
    rcclInvalidType,
    rcclInvalidOperation,
    rccl_NUM_RESULTS
} rcclResult_t;

typedef enum {
    rcclSum = 0,
    rcclProd,
    rcclMax,
    rcclMin,
    rcclNumOps
} rcclRedOp_t; 

typedef struct RcclComm_t* rcclComm_t;

typedef struct RcclUniqueId* rcclUniqueId;

const char* rcclGetErrorString(rcclResult_t result);

rcclResult_t rcclGetUniqueId(rcclUniqueId *uniqueId);

rcclResult_t rcclCommInitRank(rcclComm_t *comm, int ndev, rcclUniqueId commId, int rank);

rcclResult_t rcclCommInitAll(rcclComm_t *comm, int ndev, int *devlist);

rcclResult_t rcclCommCuDevice(rcclComm_t comm, int *dev);

rcclResult_t rcclCommUserRank(rcclComm_t comm, int *rank);

rcclResult_t rcclCommCount(rcclComm_t comm, int *count);

rcclResult_t rcclCommDestroy(rcclComm_t comm);

rcclResult_t rcclBcast(void *buff, int count, rcclDataType_t datatype, int root, rcclComm_t comm, hipStream_t stream);

rcclResult_t rcclAllReduce(const void *sendbuff, void *recvbuff, size_t count, rcclDataType_t datatype, rcclRedOp_t op, rcclComm_t comm, hipStream_t stream);

