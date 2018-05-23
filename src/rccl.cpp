/*
Copyright (c) 2017 - Present Advanced Micro Devices, Inc.
All rights reserved.
*/


#include <vector>
#include "rccl.h"

struct RcclUniqueId {
    unsigned long uid;
};

const char* rcclGetErrorString(rcclResult_t result) {
    switch(result) {
        case rcclSuccess : return "rcclSuccess";
        case rcclUnhandledHipError : return "rcclUnhandledHipError";
        case rcclSystemError: return "rcclSystemError";
        case rcclInternalError: return "rcclInternalError";
        case rcclInvalidDevicePointer: return "rcclInvalidDevicePointer";
        case rcclInvalidRank: return "rcclInvalidRank";
        case rcclUnsupportedDeviceCount: return "rcclUnsupportedDeviceCount";
        case rcclDeviceNotFound: return "rcclDeviceNotFound";
        case rcclInvalidDeviceIndex: return "rcclInvalidDeviceIndex";
        case rcclLibWrapperNotSet: return "rcclLibWrapperNotSet";
        case rcclHipMallocFailed: return "rcclHipMallocFailed";
        case rcclRankMismatch: return "rcclRankMismatch";
        case rcclInvalidArguments: return "rcclInvalidArguments";
        case rcclInvalidType: return "rcclInvalidType";
        case rcclInvalidOperation: return "rcclInvalidOperation";
        default: return "rcclErrorNotFound";
    }
}

rcclResult_t rcclGetUniqueId(rcclUniqueId *uniqueId) {
    return rcclSuccess;
}

rcclResult_t rcclCommInitRank(rcclComm_t *comm, int ndev, rcclUniqueId commId, int rank) {
    return rcclSuccess;
}

rcclResult_t rcclCommInitAll(rcclComm_t *comm, int ndev, int *devlist) {
    return rcclSuccess;
}

rcclResult_t rcclCommCuDevice(rcclComm_t comm, int *dev) {
    return rcclSuccess;
}

rcclResult_t rcclCommUserRank(rcclComm_t comm, int *rank) {
    return rcclSuccess;
}

rcclResult_t rcclCommCount(rcclComm_t comm, int *count) {
    return rcclSuccess;
}

rcclResult_t rcclCommDestroy(rcclComm_t comm) {
    return rcclSuccess;
}

rcclResult_t rcclBcast(void *buff, int count, rcclDataType_t datatype, int root, rcclComm_t comm, hipStream_t stream) {
    return rcclSuccess;
}

rcclResult_t rcclAllReduce(const void *sendbuff, void *recvbuff, size_t count, rcclDataType_t datatype, rcclRedOp_t op, rcclComm_t comm, hipStream_t stream) {
    return rcclSuccess;
}

