/*
Copyright (c) 2017 - Present Advanced Micro Devices, Inc.
All rights reserved.
*/

#include "rcclTracker.h"
#include "rcclSetKernels.h"
#include "rcclReduceRuntime.h"
#include "rcclAllReduceRuntime.h"
#include "rcclReduceScatterRuntime.h"
#include "rcclBroadcastRuntime.h"
#include "rcclAllGatherRuntime.h"

#include <vector>

//
// All rccl apis are implemented here.
// Ops are implemented in a different header file
// one header file for each op
//

std::vector<DevTrackerPool_t*> pools;

// used as rcclUniqueId
struct RcclUniqueId {
    DevTrackerPool_t *pool;
    RcclUniqueId() {
        pool = new DevTrackerPool_t;
    }
    ~RcclUniqueId() {
        delete pool;
    }
};

// implementation of rcclGetErrorString api
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
        case rcclInvalidArgument: return "rcclInvalidArgument";
        case rcclInvalidType: return "rcclInvalidType";
        case rcclInvalidOperation: return "rcclInvalidOperation";
        default: return "rcclErrorNotFound";
    }
}

rcclResult_t rcclGetUniqueId(rcclUniqueId *uniqueId) {
    if(uniqueId == nullptr) {
        return rcclInvalidArgument;
    }
    *uniqueId = new RcclUniqueId;
    return rcclSuccess;
}

rcclResult_t rcclCommInitRank(rcclComm_t *comm, int ndev, rcclUniqueId commId, int rank) {
    if(comm == nullptr) {
        return rcclInvalidArgument;
    }
    if(rank >= ndev) {
        return rcclInvalidRank;
    }
    if(commId == nullptr) {
        return rcclInvalidArgument;
    }

    auto pool = commId->pool;
    int dev;
    HIPCHECK(hipGetDevice(&dev));
    RcclComm_t *prcomm = pool->AddDevice(dev, rank, ndev);
    prcomm->pool_ = pool;
    *comm = prcomm;
    return rcclSuccess;
}

rcclResult_t rcclCommInitAll(rcclComm_t *comm, int ndev, int *devlist) {
    if(comm == nullptr || devlist == nullptr || ndev < 1) {
        return rcclInvalidArgument;
    }

    // save current device set by user
    int user_device;
    HIPCHECK(hipGetDevice(&user_device));

    int device_count;
    HIPCHECK(hipGetDeviceCount(&device_count));
    if(ndev > device_count) {
        return rcclUnsupportedDeviceCount;
    }

    // if gpus are not peer enabled, enable them    
    for(int i = 0; i < ndev; i++) {
        HIPCHECK(hipSetDevice(devlist[i]));
        for(int j = 0; j < ndev; j++) {
            if(devlist[i] != devlist[j]) {
                if(hipDeviceEnablePeerAccess(devlist[j], 0) != hipErrorPeerAccessAlreadyEnabled) {
                    HIPCHECK(hipSetDevice(user_device));
                    return rcclDeviceNotFound;
                }
            }
        }
    }

    RcclComm_t *prcomm;
    // a pool of device trackers are created
    DevTrackerPool_t *ppool = new DevTrackerPool_t(devlist, ndev);

    DeviceControl_t *ptrack;

    // populate rcclComm_t using DevTrackerPool
    for(int i=0;i<ndev;i++) {
        prcomm = new RcclComm_t;
        ptrack = ppool->GetPoolByDeviceIndex(devlist[i]);
        prcomm->pool_ = ppool;
        prcomm->track_ = ptrack;
        prcomm->device_ = devlist[i];
        prcomm->rank_ = i;
        prcomm->num_devices_ = ndev;
        comm[i] = prcomm;
        HIPCHECK(hipSetDevice(devlist[i]));
        HIPCHECK(hipEventCreateWithFlags(&prcomm->event_, hipEventReleaseToSystem));
    }

    // restore saved device user
    HIPCHECK(hipSetDevice(user_device));

    return rcclSuccess;
}

rcclResult_t rcclCommCuDevice(rcclComm_t comm, int *dev) {
    RcclComm_t *prcomm = comm;
    *dev = prcomm->device_;
    return rcclSuccess;
}

rcclResult_t rcclCommUserRank(rcclComm_t comm, int *rank) {
    RcclComm_t *prcomm = comm;
    *rank = prcomm->rank_;
    return rcclSuccess;
}

rcclResult_t rcclCommCount(rcclComm_t comm, int *count) {
    RcclComm_t *prcomm = comm;
    *count = prcomm->num_devices_;
    return rcclSuccess;
}

rcclResult_t rcclCommDestroy(rcclComm_t comm) {
    RcclComm_t *prcomm = comm;
    prcomm->pool_->active_devices_--;
    if(prcomm->pool_->active_devices_ == 0) {
        delete prcomm->pool_;
    }
    delete prcomm;
    return rcclSuccess;
}

//
// Instead of setting buffers (in trackers) on host, do it on gpu (inside kernel).
// This way, there will be no race conditions on src_buffer and dst_buffer
//
rcclResult_t rcclReduce(const void* sendbuff, void* recvbuff, int count, rcclDataType_t datatype, rcclRedOp_t op, int root, rcclComm_t comm, hipStream_t stream) {
    if(sendbuff == nullptr || recvbuff == nullptr) {
        return rcclInvalidDevicePointer;
    }

    if(datatype >= rccl_NUM_TYPES) {
        return rcclInvalidType;
    }

    if(op >= rccl_NUM_OPS) {
        return rcclInvalidOperation;
    }

    RcclComm_t *pcomm = comm;

    if(pcomm == nullptr || count <= 0 || root < 0) {
        return rcclInvalidArgument;
    }

    DeviceControl_t *pcurr_track = pcomm->track_;

    // dispatch kernel only on root
    bool is_root = pcomm->track_->hip_current_device_index == root;

    if(is_root) {
        if(op == rcclSum) {
        switch(datatype) {
            case rcclChar: {
                RcclInternalReduce<signed char, rccl_char16_t, rcclSum>(pcurr_track, count, stream, sendbuff, recvbuff);
                break;
            }
            case rcclUchar: {
                RcclInternalReduce<unsigned char, rccl_uchar16_t, rcclSum>(pcurr_track, count, stream, sendbuff, recvbuff);
                break;
            }
            case rcclShort: {
                RcclInternalReduce<signed short, rccl_short8_t, rcclSum>(pcurr_track, count, stream, sendbuff, recvbuff);
                break;
            }
            case rcclUshort: {
                RcclInternalReduce<unsigned short, rccl_ushort8_t, rcclSum>(pcurr_track, count, stream, sendbuff, recvbuff);
                break;
            }
            case rcclHalf: {
                RcclInternalReduce<__fp16, rccl_half8_t, rcclSum>(pcurr_track, count, stream, sendbuff, recvbuff);
                break;
            }
            case rcclInt: {
                RcclInternalReduce<signed int, rccl_int4_t, rcclSum>(pcurr_track, count, stream, sendbuff, recvbuff);
                break;
            }
            case rcclUint: {
                RcclInternalReduce<unsigned int, rccl_uint4_t, rcclSum>(pcurr_track, count, stream, sendbuff, recvbuff);
                break;
            }
            case rcclFloat: {
                RcclInternalReduce<float, rccl_float4_t, rcclSum>(pcurr_track, count, stream, sendbuff, recvbuff);
                break;
            }
            case rcclLong: {
                RcclInternalReduce<signed long, rccl_long2_t, rcclSum>(pcurr_track, count, stream, sendbuff, recvbuff);
                break;
            }
            case rcclUlong: {
                RcclInternalReduce<unsigned long, rccl_ulong2_t, rcclSum>(pcurr_track, count, stream, sendbuff, recvbuff);
                break;
            }
            case rcclDouble: {
                RcclInternalReduce<double, rccl_double2_t, rcclSum>(pcurr_track, count, stream, sendbuff, recvbuff);
                break;
            }
            default: {
                return rcclInvalidType;
            }
        }
        }
        if(op == rcclProd) {
        switch(datatype) {
            case rcclChar: {
                RcclInternalReduce<signed char, rccl_char16_t, rcclProd>(pcurr_track, count, stream, sendbuff, recvbuff);
                break;
            }
            case rcclUchar: {
                RcclInternalReduce<unsigned char, rccl_uchar16_t, rcclProd>(pcurr_track, count, stream, sendbuff, recvbuff);
                break;
            }
            case rcclShort: {
                RcclInternalReduce<signed short, rccl_short8_t, rcclProd>(pcurr_track, count, stream, sendbuff, recvbuff);
                break;
            }
            case rcclUshort: {
                RcclInternalReduce<unsigned short, rccl_ushort8_t, rcclProd>(pcurr_track, count, stream, sendbuff, recvbuff);
                break;
            }
            case rcclHalf: {
                RcclInternalReduce<__fp16, rccl_half8_t, rcclProd>(pcurr_track, count, stream, sendbuff, recvbuff);
                break;
            }
            case rcclInt: {
                RcclInternalReduce<signed int, rccl_int4_t, rcclProd>(pcurr_track, count, stream, sendbuff, recvbuff);
                break;
            }
            case rcclUint: {
                RcclInternalReduce<unsigned int, rccl_uint4_t, rcclProd>(pcurr_track, count, stream, sendbuff, recvbuff);
                break;
            }
            case rcclFloat: {
                RcclInternalReduce<float, rccl_float4_t, rcclProd>(pcurr_track, count, stream, sendbuff, recvbuff);
                break;
            }
            case rcclLong: {
                RcclInternalReduce<signed long, rccl_long2_t, rcclProd>(pcurr_track, count, stream, sendbuff, recvbuff);
                break;
            }
            case rcclUlong: {
                RcclInternalReduce<unsigned long, rccl_ulong2_t, rcclProd>(pcurr_track, count, stream, sendbuff, recvbuff);
                break;
            }
            case rcclDouble: {
                RcclInternalReduce<double, rccl_double2_t, rcclProd>(pcurr_track, count, stream, sendbuff, recvbuff);
                break;
            }
            default: {
                return rcclInvalidType;
            }
        }
        }

        if(op == rcclMax) {
        switch(datatype) {
            case rcclChar: {
                RcclInternalReduce<signed char, rccl_char16_t, rcclMax>(pcurr_track, count, stream, sendbuff, recvbuff);
                break;
            }
            case rcclUchar: {
                RcclInternalReduce<unsigned char, rccl_uchar16_t, rcclMax>(pcurr_track, count, stream, sendbuff, recvbuff);
                break;
            }
            case rcclShort: {
                RcclInternalReduce<signed short, rccl_short8_t, rcclMax>(pcurr_track, count, stream, sendbuff, recvbuff);
                break;
            }
            case rcclUshort: {
                RcclInternalReduce<unsigned short, rccl_ushort8_t, rcclMax>(pcurr_track, count, stream, sendbuff, recvbuff);
                break;
            }
            case rcclHalf: {
                RcclInternalReduce<__fp16, rccl_half8_t, rcclMax>(pcurr_track, count, stream, sendbuff, recvbuff);
                break;
            }
            case rcclInt: {
                RcclInternalReduce<signed int, rccl_int4_t, rcclMax>(pcurr_track, count, stream, sendbuff, recvbuff);
                break;
            }
            case rcclUint: {
                RcclInternalReduce<unsigned int, rccl_uint4_t, rcclMax>(pcurr_track, count, stream, sendbuff, recvbuff);
                break;
            }
            case rcclFloat: {
                RcclInternalReduce<float, rccl_float4_t, rcclMax>(pcurr_track, count, stream, sendbuff, recvbuff);
                break;
            }
            case rcclLong: {
                RcclInternalReduce<signed long, rccl_long2_t, rcclMax>(pcurr_track, count, stream, sendbuff, recvbuff);
                break;
            }
            case rcclUlong: {
                RcclInternalReduce<unsigned long, rccl_ulong2_t, rcclMax>(pcurr_track, count, stream, sendbuff, recvbuff);
                break;
            }
            case rcclDouble: {
                RcclInternalReduce<double, rccl_double2_t, rcclMax>(pcurr_track, count, stream, sendbuff, recvbuff);
                break;
            }
            default: {
                return rcclInvalidType;
            }
        }
        }

        if(op == rcclMin) {
        switch(datatype) {
            case rcclChar: {
                RcclInternalReduce<signed char, rccl_char16_t, rcclMin>(pcurr_track, count, stream, sendbuff, recvbuff);
                break;
            }
            case rcclUchar: {
                RcclInternalReduce<unsigned char, rccl_uchar16_t, rcclMin>(pcurr_track, count, stream, sendbuff, recvbuff);
                break;
            }
            case rcclShort: {
                RcclInternalReduce<signed short, rccl_short8_t, rcclMin>(pcurr_track, count, stream, sendbuff, recvbuff);
                break;
            }
            case rcclUshort: {
                RcclInternalReduce<unsigned short, rccl_ushort8_t, rcclMin>(pcurr_track, count, stream, sendbuff, recvbuff);
                break;
            }
            case rcclHalf: {
                RcclInternalReduce<__fp16, rccl_half8_t, rcclMin>(pcurr_track, count, stream, sendbuff, recvbuff);
                break;
            }
            case rcclInt: {
                RcclInternalReduce<signed int, rccl_int4_t, rcclMin>(pcurr_track, count, stream, sendbuff, recvbuff);
                break;
            }
            case rcclUint: {
                RcclInternalReduce<unsigned int, rccl_uint4_t, rcclMin>(pcurr_track, count, stream, sendbuff, recvbuff);
                break;
            }
            case rcclFloat: {
                RcclInternalReduce<float, rccl_float4_t, rcclMin>(pcurr_track, count, stream, sendbuff, recvbuff);
                break;
            }
            case rcclLong: {
                RcclInternalReduce<signed long, rccl_long2_t, rcclMin>(pcurr_track, count, stream, sendbuff, recvbuff);
                break;
            }
            case rcclUlong: {
                RcclInternalReduce<unsigned long, rccl_ulong2_t, rcclMin>(pcurr_track, count, stream, sendbuff, recvbuff);
                break;
            }
            case rcclDouble: {
                RcclInternalReduce<double, rccl_double2_t, rcclMin>(pcurr_track, count, stream, sendbuff, recvbuff);
                break;
            }
            default: {
                return rcclInvalidType;
            }
        }
        }


    } else {
        hipLaunchKernelGGL(RcclKernelSetSrcPtr, dim3(1, 1, 1), dim3(1, 1, 1), 0, 0, pcurr_track, sendbuff);
    }
    return rcclSuccess;
}


rcclResult_t rcclAllReduce(const void* sendbuff, void* recvbuff, int count, rcclDataType_t datatype, rcclRedOp_t op, rcclComm_t comm, hipStream_t stream) {
    if(sendbuff == nullptr || recvbuff == nullptr) {
        return rcclInvalidDevicePointer;
    }
    if(datatype >= rccl_NUM_TYPES) {
        return rcclInvalidType;
    }
    if(op >= rccl_NUM_OPS) {
        return rcclInvalidOperation;
    }

    RcclComm_t* pcomm = comm;

    if(pcomm == nullptr || count <= 0) {
        return rcclInvalidArgument;
    }

    DeviceControl_t* pcurr_track = pcomm->track_;


    if(op == rcclSum) {
        switch(datatype) {
            case rcclChar: {
                RcclInternalAllReduce<signed char, rccl_char16_t, rcclSum>(pcurr_track, count, stream, sendbuff, recvbuff);
                break;
            }
            case rcclUchar: {
                RcclInternalAllReduce<unsigned char, rccl_uchar16_t, rcclSum>(pcurr_track, count, stream, sendbuff, recvbuff);
                break;
            }
            case rcclShort: {
                RcclInternalAllReduce<signed short, rccl_short8_t, rcclSum>(pcurr_track, count, stream, sendbuff, recvbuff);
                break;
            }
            case rcclUshort: {
                RcclInternalAllReduce<unsigned short, rccl_ushort8_t, rcclSum>(pcurr_track, count, stream, sendbuff, recvbuff);
                break;
            }
            case rcclHalf: {
                RcclInternalAllReduce<__fp16, rccl_half8_t, rcclSum>(pcurr_track, count, stream, sendbuff, recvbuff);
                break;
            }
            case rcclInt: {
                RcclInternalAllReduce<signed int, rccl_int4_t, rcclSum>(pcurr_track, count, stream, sendbuff, recvbuff);
                break;
            }
            case rcclUint: {
                RcclInternalAllReduce<unsigned int, rccl_uint4_t, rcclSum>(pcurr_track, count, stream, sendbuff, recvbuff);
                break;
            }
            case rcclFloat: {
                RcclInternalAllReduce<float, rccl_float4_t, rcclSum>(pcurr_track, count, stream, sendbuff, recvbuff);
                break;
            }
            case rcclLong: {
                RcclInternalAllReduce<signed long, rccl_long2_t, rcclSum>(pcurr_track, count, stream, sendbuff, recvbuff);
                break;
            }
            case rcclUlong: {
                RcclInternalAllReduce<unsigned long, rccl_ulong2_t, rcclSum>(pcurr_track, count, stream, sendbuff, recvbuff);
                break;
            }
            case rcclDouble: {
                RcclInternalAllReduce<double, rccl_double2_t, rcclSum>(pcurr_track, count, stream, sendbuff, recvbuff);
                break;
            }
            default: {
                return rcclInvalidType;
            }
        }
    }
    if(op == rcclProd) {
        switch(datatype) {
            case rcclChar: {
                RcclInternalAllReduce<signed char, rccl_char16_t, rcclProd>(pcurr_track, count, stream, sendbuff, recvbuff);
                break;
            }
            case rcclUchar: {
                RcclInternalAllReduce<unsigned char, rccl_uchar16_t, rcclProd>(pcurr_track, count, stream, sendbuff, recvbuff);
                break;
            }
            case rcclShort: {
                RcclInternalAllReduce<signed short, rccl_short8_t, rcclProd>(pcurr_track, count, stream, sendbuff, recvbuff);
                break;
            }
            case rcclUshort: {
                RcclInternalAllReduce<unsigned short, rccl_ushort8_t, rcclProd>(pcurr_track, count, stream, sendbuff, recvbuff);
                break;
            }
            case rcclHalf: {
                RcclInternalAllReduce<__fp16, rccl_half8_t, rcclProd>(pcurr_track, count, stream, sendbuff, recvbuff);
                break;
            }
            case rcclInt: {
                RcclInternalAllReduce<signed int, rccl_int4_t, rcclProd>(pcurr_track, count, stream, sendbuff, recvbuff);
                break;
            }
            case rcclUint: {
                RcclInternalAllReduce<unsigned int, rccl_uint4_t, rcclProd>(pcurr_track, count, stream, sendbuff, recvbuff);
                break;
            }
            case rcclFloat: {
                RcclInternalAllReduce<float, rccl_float4_t, rcclProd>(pcurr_track, count, stream, sendbuff, recvbuff);
                break;
            }
            case rcclLong: {
                RcclInternalAllReduce<signed long, rccl_long2_t, rcclProd>(pcurr_track, count, stream, sendbuff, recvbuff);
                break;
            }
            case rcclUlong: {
                RcclInternalAllReduce<unsigned long, rccl_ulong2_t, rcclProd>(pcurr_track, count, stream, sendbuff, recvbuff);
                break;
            }
            case rcclDouble: {
                RcclInternalAllReduce<double, rccl_double2_t, rcclProd>(pcurr_track, count, stream, sendbuff, recvbuff);
                break;
            }
            default: {
                return rcclInvalidType;
            }
        }
    }

    if(op == rcclMax) {
        switch(datatype) {
            case rcclChar: {
                RcclInternalAllReduce<signed char, rccl_char16_t, rcclMax>(pcurr_track, count, stream, sendbuff, recvbuff);
                break;
            }
            case rcclUchar: {
                RcclInternalAllReduce<unsigned char, rccl_uchar16_t, rcclMax>(pcurr_track, count, stream, sendbuff, recvbuff);
                break;
            }
            case rcclShort: {
                RcclInternalAllReduce<signed short, rccl_short8_t, rcclMax>(pcurr_track, count, stream, sendbuff, recvbuff);
                break;
            }
            case rcclUshort: {
                RcclInternalAllReduce<unsigned short, rccl_ushort8_t, rcclMax>(pcurr_track, count, stream, sendbuff, recvbuff);
                break;
            }
            case rcclHalf: {
                RcclInternalAllReduce<__fp16, rccl_half8_t, rcclMax>(pcurr_track, count, stream, sendbuff, recvbuff);
                break;
            }
            case rcclInt: {
                RcclInternalAllReduce<signed int, rccl_int4_t, rcclMax>(pcurr_track, count, stream, sendbuff, recvbuff);
                break;
            }
            case rcclUint: {
                RcclInternalAllReduce<unsigned int, rccl_uint4_t, rcclMax>(pcurr_track, count, stream, sendbuff, recvbuff);
                break;
            }
            case rcclFloat: {
                RcclInternalAllReduce<float, rccl_float4_t, rcclMax>(pcurr_track, count, stream, sendbuff, recvbuff);
                break;
            }
            case rcclLong: {
                RcclInternalAllReduce<signed long, rccl_long2_t, rcclMax>(pcurr_track, count, stream, sendbuff, recvbuff);
                break;
            }
            case rcclUlong: {
                RcclInternalAllReduce<unsigned long, rccl_ulong2_t, rcclMax>(pcurr_track, count, stream, sendbuff, recvbuff);
                break;
            }
            case rcclDouble: {
                RcclInternalAllReduce<double, rccl_double2_t, rcclMax>(pcurr_track, count, stream, sendbuff, recvbuff);
                break;
            }
            default: {
                return rcclInvalidType;
            }
        }
    }

    if(op == rcclMin) {
        switch(datatype) {
            case rcclChar: {
                RcclInternalAllReduce<signed char, rccl_char16_t, rcclMin>(pcurr_track, count, stream, sendbuff, recvbuff);
                break;
            }
            case rcclUchar: {
                RcclInternalAllReduce<unsigned char, rccl_uchar16_t, rcclMin>(pcurr_track, count, stream, sendbuff, recvbuff);
                break;
            }
            case rcclShort: {
                RcclInternalAllReduce<signed short, rccl_short8_t, rcclMin>(pcurr_track, count, stream, sendbuff, recvbuff);
                break;
            }
            case rcclUshort: {
                RcclInternalAllReduce<unsigned short, rccl_ushort8_t, rcclMin>(pcurr_track, count, stream, sendbuff, recvbuff);
                break;
            }
            case rcclHalf: {
                RcclInternalAllReduce<__fp16, rccl_half8_t, rcclMin>(pcurr_track, count, stream, sendbuff, recvbuff);
                break;
            }
            case rcclInt: {
                RcclInternalAllReduce<signed int, rccl_int4_t, rcclMin>(pcurr_track, count, stream, sendbuff, recvbuff);
                break;
            }
            case rcclUint: {
                RcclInternalAllReduce<unsigned int, rccl_uint4_t, rcclMin>(pcurr_track, count, stream, sendbuff, recvbuff);
                break;
            }
            case rcclFloat: {
                RcclInternalAllReduce<float, rccl_float4_t, rcclMin>(pcurr_track, count, stream, sendbuff, recvbuff);
                break;
            }
            case rcclLong: {
                RcclInternalAllReduce<signed long, rccl_long2_t, rcclMin>(pcurr_track, count, stream, sendbuff, recvbuff);
                break;
            }
            case rcclUlong: {
                RcclInternalAllReduce<unsigned long, rccl_ulong2_t, rcclMin>(pcurr_track, count, stream, sendbuff, recvbuff);
                break;
            }
            case rcclDouble: {
                RcclInternalAllReduce<double, rccl_double2_t, rcclMin>(pcurr_track, count, stream, sendbuff, recvbuff);
                break;
            }
            default: {
                return rcclInvalidType;
            }
        }
    }

    return rcclSuccess;
}

rcclResult_t rcclBcast(void* buff, int count, rcclDataType_t datatype, int root, rcclComm_t comm, hipStream_t stream) {
    if(buff == nullptr) {
        return rcclInvalidDevicePointer;
    }
    if(datatype >= rccl_NUM_TYPES) {
        return rcclInvalidType;
    }

    RcclComm_t* pcomm = comm;

    if(pcomm == nullptr || root < 0 || count <= 0) {
        return rcclInvalidArgument;
    }

    DeviceControl_t* pcurr_track = pcomm->track_;
    bool is_root = pcomm->track_->hip_current_device_index == root;

    if(is_root) {
        switch(datatype) {
            case rcclChar: {
                RcclInternalBroadcast<signed char, rccl_char16_t>(pcurr_track, count, stream, buff);
                break;
            }
            case rcclUchar: {
                RcclInternalBroadcast<unsigned char, rccl_uchar16_t>(pcurr_track, count, stream, buff);
                break;
            }
            case rcclShort: {
                RcclInternalBroadcast<signed short, rccl_short8_t>(pcurr_track, count, stream, buff);
                break;
            }
            case rcclUshort: {
                RcclInternalBroadcast<unsigned short, rccl_ushort8_t>(pcurr_track, count, stream, buff);
                break;
            }
            case rcclHalf: {
                RcclInternalBroadcast<__fp16, rccl_half8_t>(pcurr_track, count, stream, buff);
                break;
            }
            case rcclInt: {
                RcclInternalBroadcast<signed int, rccl_int4_t>(pcurr_track, count, stream, buff);
                break;
            }
            case rcclUint: {
                RcclInternalBroadcast<unsigned int, rccl_uint4_t>(pcurr_track, count, stream, buff);
                break;
            }
            case rcclFloat: {
                RcclInternalBroadcast<float, rccl_float4_t>(pcurr_track, count, stream, buff);
                break;
            }
            case rcclLong: {
                RcclInternalBroadcast<signed long, rccl_long2_t>(pcurr_track, count, stream, buff);
                break;
            }
            case rcclUlong: {
                RcclInternalBroadcast<unsigned long, rccl_ulong2_t>(pcurr_track, count, stream, buff);
                break;
            }
            case rcclDouble: {
                RcclInternalBroadcast<double, rccl_double2_t>(pcurr_track, count, stream, buff);
                break;
            }
            default: {
                return rcclInvalidType;
            }
        }
    } else {
        hipLaunchKernelGGL(RcclKernelSetDstPtr, dim3(1, 1, 1), dim3(1, 1, 1), 0, stream, pcurr_track, buff);
    }
    return rcclSuccess;
}


rcclResult_t rcclAllGather(const void* sendbuff, int count, rcclDataType_t datatype, void* recvbuff, rcclComm_t comm, hipStream_t stream) {
    if(sendbuff == nullptr || recvbuff == nullptr) {
        return rcclInvalidDevicePointer;
    }

    if(datatype >= rccl_NUM_TYPES || datatype < rcclInt8) {
        return rcclInvalidType;
    }

    RcclComm_t *pcomm = comm;

    if(pcomm == nullptr || count <= 0) {
        return rcclInvalidArgument;
    }

    DeviceControl_t *pcurr_track = pcomm->track_;
    int rank = pcomm->rank_;

    switch(datatype) {
        case rcclChar: {
            RcclInternalAllGather<signed char, rccl_char16_t>(pcurr_track, count, rank, stream, sendbuff, recvbuff);
            break;
        }
        case rcclUchar: {
            RcclInternalAllGather<unsigned char, rccl_uchar16_t>(pcurr_track, count, rank, stream, sendbuff, recvbuff);
            break;
        }
        case rcclShort: {
            RcclInternalAllGather<signed short, rccl_short8_t>(pcurr_track, count, rank, stream, sendbuff, recvbuff);
            break;
        }
        case rcclUshort: {
            RcclInternalAllGather<unsigned short, rccl_ushort8_t>(pcurr_track, count, rank, stream, sendbuff, recvbuff);
            break;
        }
        case rcclHalf: {
            RcclInternalAllGather<__fp16, rccl_half8_t>(pcurr_track, count, rank, stream, sendbuff, recvbuff);
            break;
        }
        case rcclInt: {
            RcclInternalAllGather<signed int, rccl_int4_t>(pcurr_track, count, rank, stream, sendbuff, recvbuff);
            break;
        }
        case rcclUint: {
            RcclInternalAllGather<unsigned int, rccl_uint4_t>(pcurr_track, count, rank, stream, sendbuff, recvbuff);
            break;
        }
        case rcclFloat: {
            RcclInternalAllGather<float, rccl_float4_t>(pcurr_track, count, rank, stream, sendbuff, recvbuff);
            break;
        }
        case rcclLong: {
            RcclInternalAllGather<signed long, rccl_long2_t>(pcurr_track, count, rank, stream, sendbuff, recvbuff);
            break;
        }
        case rcclUlong: {
            RcclInternalAllGather<unsigned long, rccl_ulong2_t>(pcurr_track, count, rank, stream, sendbuff, recvbuff);
            break;
        }
        case rcclDouble: {
            RcclInternalAllGather<double, rccl_double2_t>(pcurr_track, count, rank, stream, sendbuff, recvbuff);
            break;
        }
        default: {
            return rcclInvalidType;
        }
    }
    return rcclSuccess;
}

rcclResult_t rcclReduceScatter(const void* sendbuff, void* recvbuff, int count, rcclDataType_t datatype, rcclRedOp_t op, rcclComm_t comm, hipStream_t stream) {
    if(sendbuff == nullptr || recvbuff == nullptr) {
        return rcclInvalidDevicePointer;
    }
    if(datatype >= rccl_NUM_TYPES) {
        return rcclInvalidType;
    }
    if(op >= rccl_NUM_OPS) {
        return rcclInvalidOperation;
    }

    RcclComm_t* pcomm = comm;

    if(pcomm == nullptr || count <= 0) {
        return rcclInvalidArgument;
    }

    DeviceControl_t* pcurr_track = pcomm->track_;
    int rank = pcomm->rank_;


    if(op == rcclSum) {
        switch(datatype) {
            case rcclChar: {
                RcclInternalReduceScatter<signed char, rccl_char16_t, rcclSum>(pcurr_track, rank, count, stream, sendbuff, recvbuff);
                break;
            }
            case rcclUchar: {
                RcclInternalReduceScatter<unsigned char, rccl_uchar16_t, rcclSum>(pcurr_track, rank, count, stream, sendbuff, recvbuff);
                break;
            }
            case rcclShort: {
                RcclInternalReduceScatter<signed short, rccl_short8_t, rcclSum>(pcurr_track, rank, count, stream, sendbuff, recvbuff);
                break;
            }
            case rcclUshort: {
                RcclInternalReduceScatter<unsigned short, rccl_ushort8_t, rcclSum>(pcurr_track, rank, count, stream, sendbuff, recvbuff);
                break;
            }
            case rcclHalf: {
                RcclInternalReduceScatter<__fp16, rccl_half8_t, rcclSum>(pcurr_track, rank, count, stream, sendbuff, recvbuff);
                break;
            }
            case rcclInt: {
                RcclInternalReduceScatter<signed int, rccl_int4_t, rcclSum>(pcurr_track, rank, count, stream, sendbuff, recvbuff);
                break;
            }
            case rcclUint: {
                RcclInternalReduceScatter<unsigned int, rccl_uint4_t, rcclSum>(pcurr_track, rank, count, stream, sendbuff, recvbuff);
                break;
            }
            case rcclFloat: {
                RcclInternalReduceScatter<float, rccl_float4_t, rcclSum>(pcurr_track, rank, count, stream, sendbuff, recvbuff);
                break;
            }
            case rcclLong: {
                RcclInternalReduceScatter<signed long, rccl_long2_t, rcclSum>(pcurr_track, rank, count, stream, sendbuff, recvbuff);
                break;
            }
            case rcclUlong: {
                RcclInternalReduceScatter<unsigned long, rccl_ulong2_t, rcclSum>(pcurr_track, rank, count, stream, sendbuff, recvbuff);
                break;
            }
            case rcclDouble: {
                RcclInternalReduceScatter<double, rccl_double2_t, rcclSum>(pcurr_track, rank, count, stream, sendbuff, recvbuff);
                break;
            }
            default: {
                return rcclInvalidType;
            }
        }
    }
    if(op == rcclProd) {
        switch(datatype) {
            case rcclChar: {
                RcclInternalReduceScatter<signed char, rccl_char16_t, rcclProd>(pcurr_track, rank, count, stream, sendbuff, recvbuff);
                break;
            }
            case rcclUchar: {
                RcclInternalReduceScatter<unsigned char, rccl_uchar16_t, rcclProd>(pcurr_track, rank, count, stream, sendbuff, recvbuff);
                break;
            }
            case rcclShort: {
                RcclInternalReduceScatter<signed short, rccl_short8_t, rcclProd>(pcurr_track, rank, count, stream, sendbuff, recvbuff);
                break;
            }
            case rcclUshort: {
                RcclInternalReduceScatter<unsigned short, rccl_ushort8_t, rcclProd>(pcurr_track, rank, count, stream, sendbuff, recvbuff);
                break;
            }
            case rcclHalf: {
                RcclInternalReduceScatter<__fp16, rccl_half8_t, rcclProd>(pcurr_track, rank, count, stream, sendbuff, recvbuff);
                break;
            }
            case rcclInt: {
                RcclInternalReduceScatter<signed int, rccl_int4_t, rcclProd>(pcurr_track, rank, count, stream, sendbuff, recvbuff);
                break;
            }
            case rcclUint: {
                RcclInternalReduceScatter<unsigned int, rccl_uint4_t, rcclProd>(pcurr_track, rank, count, stream, sendbuff, recvbuff);
                break;
            }
            case rcclFloat: {
                RcclInternalReduceScatter<float, rccl_float4_t, rcclProd>(pcurr_track, rank, count, stream, sendbuff, recvbuff);
                break;
            }
            case rcclLong: {
                RcclInternalReduceScatter<signed long, rccl_long2_t, rcclProd>(pcurr_track, rank, count, stream, sendbuff, recvbuff);
                break;
            }
            case rcclUlong: {
                RcclInternalReduceScatter<unsigned long, rccl_ulong2_t, rcclProd>(pcurr_track, rank, count, stream, sendbuff, recvbuff);
                break;
            }
            case rcclDouble: {
                RcclInternalReduceScatter<double, rccl_double2_t, rcclProd>(pcurr_track, rank, count, stream, sendbuff, recvbuff);
                break;
            }
            default: {
                return rcclInvalidType;
            }
        }
    }

    if(op == rcclMax) {
        switch(datatype) {
            case rcclChar: {
                RcclInternalReduceScatter<signed char, rccl_char16_t, rcclMax>(pcurr_track, rank, count, stream, sendbuff, recvbuff);
                break;
            }
            case rcclUchar: {
                RcclInternalReduceScatter<unsigned char, rccl_uchar16_t, rcclMax>(pcurr_track, rank, count, stream, sendbuff, recvbuff);
                break;
            }
            case rcclShort: {
                RcclInternalReduceScatter<signed short, rccl_short8_t, rcclMax>(pcurr_track, rank, count, stream, sendbuff, recvbuff);
                break;
            }
            case rcclUshort: {
                RcclInternalReduceScatter<unsigned short, rccl_ushort8_t, rcclMax>(pcurr_track, rank, count, stream, sendbuff, recvbuff);
                break;
            }
            case rcclHalf: {
                RcclInternalReduceScatter<__fp16, rccl_half8_t, rcclMax>(pcurr_track, rank, count, stream, sendbuff, recvbuff);
                break;
            }
            case rcclInt: {
                RcclInternalReduceScatter<signed int, rccl_int4_t, rcclMax>(pcurr_track, rank, count, stream, sendbuff, recvbuff);
                break;
            }
            case rcclUint: {
                RcclInternalReduceScatter<unsigned int, rccl_uint4_t, rcclMax>(pcurr_track, rank, count, stream, sendbuff, recvbuff);
                break;
            }
            case rcclFloat: {
                RcclInternalReduceScatter<float, rccl_float4_t, rcclMax>(pcurr_track, rank, count, stream, sendbuff, recvbuff);
                break;
            }
            case rcclLong: {
                RcclInternalReduceScatter<signed long, rccl_long2_t, rcclMax>(pcurr_track, rank, count, stream, sendbuff, recvbuff);
                break;
            }
            case rcclUlong: {
                RcclInternalReduceScatter<unsigned long, rccl_ulong2_t, rcclMax>(pcurr_track, rank, count, stream, sendbuff, recvbuff);
                break;
            }
            case rcclDouble: {
                RcclInternalReduceScatter<double, rccl_double2_t, rcclMax>(pcurr_track, rank, count, stream, sendbuff, recvbuff);
                break;
            }
            default: {
                return rcclInvalidType;
            }
        }
    }

    if(op == rcclMin) {
        switch(datatype) {
            case rcclChar: {
                RcclInternalReduceScatter<signed char, rccl_char16_t, rcclMin>(pcurr_track, rank, count, stream, sendbuff, recvbuff);
                break;
            }
            case rcclUchar: {
                RcclInternalReduceScatter<unsigned char, rccl_uchar16_t, rcclMin>(pcurr_track, rank, count, stream, sendbuff, recvbuff);
                break;
            }
            case rcclShort: {
                RcclInternalReduceScatter<signed short, rccl_short8_t, rcclMin>(pcurr_track, rank, count, stream, sendbuff, recvbuff);
                break;
            }
            case rcclUshort: {
                RcclInternalReduceScatter<unsigned short, rccl_ushort8_t, rcclMin>(pcurr_track, rank, count, stream, sendbuff, recvbuff);
                break;
            }
            case rcclHalf: {
                RcclInternalReduceScatter<__fp16, rccl_half8_t, rcclMin>(pcurr_track, rank, count, stream, sendbuff, recvbuff);
                break;
            }
            case rcclInt: {
                RcclInternalReduceScatter<signed int, rccl_int4_t, rcclMin>(pcurr_track, rank, count, stream, sendbuff, recvbuff);
                break;
            }
            case rcclUint: {
                RcclInternalReduceScatter<unsigned int, rccl_uint4_t, rcclMin>(pcurr_track, rank, count, stream, sendbuff, recvbuff);
                break;
            }
            case rcclFloat: {
                RcclInternalReduceScatter<float, rccl_float4_t, rcclMin>(pcurr_track, rank, count, stream, sendbuff, recvbuff);
                break;
            }
            case rcclLong: {
                RcclInternalReduceScatter<signed long, rccl_long2_t, rcclMin>(pcurr_track, rank, count, stream, sendbuff, recvbuff);
                break;
            }
            case rcclUlong: {
                RcclInternalReduceScatter<unsigned long, rccl_ulong2_t, rcclMin>(pcurr_track, rank, count, stream, sendbuff, recvbuff);
                break;
            }
            case rcclDouble: {
                RcclInternalReduceScatter<double, rccl_double2_t, rcclMin>(pcurr_track, rank, count, stream, sendbuff, recvbuff);
                break;
            }
            default: {
                return rcclInvalidType;
            }
        }
    }
    return rcclSuccess;
}

