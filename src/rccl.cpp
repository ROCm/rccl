/*
Copyright (c) 2017 - Present Advanced Micro Devices, Inc.
All rights reserved.
*/

#include "rcclTracker.h"
#include "rcclDataTypes.h"
#include "rcclSetKernels.h"
#include "rcclLog.h"

#include "rcclScalarAllReduceRuntime.h"
#include "rcclScalarBroadcastRuntime.h"

#include <vector>

//
// All rccl apis are implemented here.
// Ops are implemented in a different header file
// one header file for each op
//

std::vector<RingNodePool_t*> pools;

// used as rcclUniqueId
struct RcclUniqueId {
    RingNodePool_t *pool;
    RcclUniqueId() {
        pool = new RingNodePool_t;
    }
    ~RcclUniqueId() {
        delete pool;
    }
};

const char* get_env_val = getenv("RCCL_TRACE_RT");
int RCCL_TRACE_RT = get_env_val != nullptr ? atoi(get_env_val) : 0;

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
    RcclComm_t *pcomm = pool->AddDevice(dev, rank, ndev);
    pcomm->pool_ = pool;
    *comm = pcomm;
    return rcclSuccess;
}

rcclResult_t rcclCommInitAll(rcclComm_t *comm, int ndev, int *devlist) {
    if((RCCL_TRACE_RT & krccl_print_api) == krccl_print_api) {
        fprintf(stderr, "%s<<rccl-api: %s comm:%p ndev:%d devlist:%p%s\n", API_COLOR, __func__, comm, ndev, devlist, API_COLOR_END);
    }
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
                hipError_t err = hipDeviceEnablePeerAccess(devlist[j], 0);
                if(err != hipErrorPeerAccessAlreadyEnabled && err != hipSuccess) {
                    HIPCHECK(hipSetDevice(user_device));
                    return rcclDeviceNotFound;
                }
            }
        }
    }

    RcclComm_t *pcomm;
    // a pool of device trackers are created
    RingNodePool_t *ppool = new RingNodePool_t(devlist, ndev);

    RingNode_t *ptrack;

    // populate rcclComm_t using DevTrackerPool
    for(int i=0;i<ndev;i++) {
        pcomm = new RcclComm_t;
        ptrack = ppool->GetPoolByDeviceIndex(devlist[i]);
        pcomm->pool_ = ppool;
        pcomm->track_ = ptrack;
        pcomm->device_ = devlist[i];
        pcomm->rank_ = i;
        pcomm->num_devices_ = ndev;
        pcomm->this_time_ = 0;
        pcomm->stream_ = NULL;
        comm[i] = pcomm;
        HIPCHECK(hipSetDevice(devlist[i]));
        HIPCHECK(hipEventCreateWithFlags(&pcomm->event_, hipEventReleaseToSystem));
    }

    // restore saved device user
    HIPCHECK(hipSetDevice(user_device));

    return rcclSuccess;
}

rcclResult_t rcclCommCuDevice(rcclComm_t comm, int *dev) {
    if((RCCL_TRACE_RT & krccl_print_api) == krccl_print_api) {
        fprintf(stderr, "%s<<rccl-api: %s comm:%p *dev:%d dev:%p%s\n", API_COLOR, __func__, comm, *dev, dev, API_COLOR_END);
    }
    RcclComm_t *pcomm = comm;
    *dev = pcomm->device_;
    return rcclSuccess;
}

rcclResult_t rcclCommUserRank(rcclComm_t comm, int *rank) {
    if((RCCL_TRACE_RT & krccl_print_api) == krccl_print_api) {
        fprintf(stderr, "%s<<rccl-api: %s comm:%p *rank:%d rank:%p%s\n", API_COLOR, __func__, comm, *rank, rank, API_COLOR_END);
    }
    RcclComm_t *pcomm = comm;
    *rank = pcomm->rank_;
    return rcclSuccess;
}

rcclResult_t rcclCommCount(rcclComm_t comm, int *count) {
    if((RCCL_TRACE_RT & krccl_print_api) == krccl_print_api) {
        fprintf(stderr, "%s<<rccl-api: %s comm:%p *count:%d count:%p%s\n", API_COLOR, __func__, comm, *count, count, API_COLOR_END);
    }
    RcclComm_t *pcomm = comm;
    *count = pcomm->num_devices_;
    return rcclSuccess;
}

rcclResult_t rcclCommDestroy(rcclComm_t comm) {
    if((RCCL_TRACE_RT & krccl_print_api) == krccl_print_api) {
        fprintf(stderr, "%s<<rccl-api: %s comm:%p%s\n", API_COLOR, __func__, comm, API_COLOR_END);
    }
    RcclComm_t *pcomm = comm;
    pcomm->pool_->active_devices_--;
    if(pcomm->pool_->active_devices_ == 0) {
        delete pcomm->pool_;
    }
    delete pcomm;
    return rcclSuccess;
}

void PostEnqueueEventRecord(RcclComm_t* pcomm, hipStream_t stream) {
    hipEventRecord(pcomm->event_, stream);
}

void PreEnqueueEventRecord(RcclComm_t* pcomm, hipStream_t stream) {
    if(stream != pcomm->stream_) {
        hipStreamWaitEvent(stream, pcomm->event_, 0);
        pcomm->stream_ = stream;
    }
}

rcclResult_t rcclAllReduce(const void* sendbuff, void* recvbuff, int count, rcclDataType_t datatype, rcclRedOp_t op, rcclComm_t comm, hipStream_t stream) {
    if((RCCL_TRACE_RT & krccl_print_api) == krccl_print_api) {
        int dev;
        hipGetDevice(&dev);
        fprintf(stderr, "%s<<rccl-api:%s rccl-device:%d sendbuff:%p recvbuff:%p count:%d datatype:%s op:%s comm:%p stream:%p%s\n", API_COLOR, __func__, dev, sendbuff, recvbuff, count, umap_datatype[datatype].c_str(), umap_red_op[op].c_str(), comm, stream, API_COLOR_END);
    }
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

    hipEvent_t event = pcomm->event_;

    if(pcomm == nullptr || count <= 0) {
        return rcclInvalidArgument;
    }

    PreEnqueueEventRecord(pcomm, stream);

    RingNode_t* pcurr_track = pcomm->track_;
    int rank = pcomm->rank_;
    int num_gpus = pcomm->num_devices_;

    int* this_time = &(pcomm->this_time_);
    if(num_gpus == 1) {
        switch(datatype) {
            case rcclChar:
            case rcclUchar: {
                hipMemcpyAsync(recvbuff, sendbuff, count * sizeof(char), hipMemcpyDeviceToDevice, stream);
                break;
            }
            case rcclShort:
            case rcclUshort:
            case rcclHalf: {
                hipMemcpyAsync(recvbuff, sendbuff, count * sizeof(short), hipMemcpyDeviceToDevice, stream);
                break;
            }
            case rcclInt:
            case rcclUint:
            case rcclFloat: {
                hipMemcpyAsync(recvbuff, sendbuff, count * sizeof(int), hipMemcpyDeviceToDevice, stream);
                break;
            }
            case rcclLong:
            case rcclUlong:
            case rcclDouble: {
                hipMemcpyAsync(recvbuff, sendbuff, count * sizeof(double), hipMemcpyDeviceToDevice, stream);
                break;
            }
            default: {
                PostEnqueueEventRecord(pcomm, stream);
                return rcclInvalidType;
            }
            }
            PostEnqueueEventRecord(pcomm, stream);
            return rcclSuccess;
    }

    if(op == rcclSum) {
        switch(datatype) {
            case rcclChar: {
                RcclInternalAllReduce<signed char, rccl_char16_t, rcclSum>(pcurr_track, sendbuff, recvbuff, stream, count, num_gpus, rank, event, this_time);
                break;
            }
            case rcclUchar: {
                RcclInternalAllReduce<unsigned char, rccl_uchar16_t, rcclSum>(pcurr_track, sendbuff, recvbuff, stream, count, num_gpus, rank, event, this_time);
                break;
            }
            case rcclShort: {
                RcclInternalAllReduce<signed short, rccl_short8_t, rcclSum>(pcurr_track, sendbuff, recvbuff, stream, count, num_gpus, rank, event, this_time);
                break;
            }
            case rcclUshort: {
                RcclInternalAllReduce<unsigned short, rccl_ushort8_t, rcclSum>(pcurr_track, sendbuff, recvbuff, stream, count, num_gpus, rank, event, this_time);
                break;
            }
            case rcclHalf: {
                RcclInternalAllReduce<__fp16, rccl_half8_t, rcclSum>(pcurr_track, sendbuff, recvbuff, stream, count, num_gpus, rank, event, this_time);
                break;
            }
            case rcclInt: {
                RcclInternalAllReduce<signed int, rccl_int4_t, rcclSum>(pcurr_track, sendbuff, recvbuff, stream, count, num_gpus, rank, event, this_time);
                break;
            }
            case rcclUint: {
                RcclInternalAllReduce<unsigned int, rccl_uint4_t, rcclSum>(pcurr_track, sendbuff, recvbuff, stream, count, num_gpus, rank, event, this_time);
                break;
            }
            case rcclFloat: {
                RcclInternalAllReduce<float, rccl_float4_t, rcclSum>(pcurr_track, sendbuff, recvbuff, stream, count, num_gpus, rank, event, this_time);
                break;
            }
            case rcclLong: {
                RcclInternalAllReduce<signed long, rccl_long2_t, rcclSum>(pcurr_track, sendbuff, recvbuff, stream, count, num_gpus, rank, event, this_time);
                break;
            }
            case rcclUlong: {
                RcclInternalAllReduce<unsigned long, rccl_ulong2_t, rcclSum>(pcurr_track, sendbuff, recvbuff, stream, count, num_gpus, rank, event, this_time);
                break;
            }
            case rcclDouble: {
                RcclInternalAllReduce<double, rccl_double2_t, rcclSum>(pcurr_track, sendbuff, recvbuff, stream, count, num_gpus, rank, event, this_time);
                break;
            }
            default: {
                PostEnqueueEventRecord(pcomm, stream);
                return rcclInvalidType;
            }
        }
    }
    if(op == rcclProd) {
        switch(datatype) {
            case rcclChar: {
                RcclInternalAllReduce<signed char, rccl_char16_t, rcclProd>(pcurr_track, sendbuff, recvbuff, stream, count, num_gpus, rank, event, this_time);
                break;
            }
            case rcclUchar: {
                RcclInternalAllReduce<unsigned char, rccl_uchar16_t, rcclProd>(pcurr_track, sendbuff, recvbuff, stream, count, num_gpus, rank, event, this_time);
                break;
            }
            case rcclShort: {
                RcclInternalAllReduce<signed short, rccl_short8_t, rcclProd>(pcurr_track, sendbuff, recvbuff, stream, count, num_gpus, rank, event, this_time);
                break;
            }
            case rcclUshort: {
                RcclInternalAllReduce<unsigned short, rccl_ushort8_t, rcclProd>(pcurr_track, sendbuff, recvbuff, stream, count, num_gpus, rank, event, this_time);
                break;
            }
            case rcclHalf: {
                RcclInternalAllReduce<__fp16, rccl_half8_t, rcclProd>(pcurr_track, sendbuff, recvbuff, stream, count, num_gpus, rank, event, this_time);
                break;
            }
            case rcclInt: {
                RcclInternalAllReduce<signed int, rccl_int4_t, rcclProd>(pcurr_track, sendbuff, recvbuff, stream, count, num_gpus, rank, event, this_time);
                break;
            }
            case rcclUint: {
                RcclInternalAllReduce<unsigned int, rccl_uint4_t, rcclProd>(pcurr_track, sendbuff, recvbuff, stream, count, num_gpus, rank, event, this_time);
                break;
            }
            case rcclFloat: {
                RcclInternalAllReduce<float, rccl_float4_t, rcclProd>(pcurr_track, sendbuff, recvbuff, stream, count, num_gpus, rank, event, this_time);
                break;
            }
            case rcclLong: {
                RcclInternalAllReduce<signed long, rccl_long2_t, rcclProd>(pcurr_track, sendbuff, recvbuff, stream, count, num_gpus, rank, event, this_time);
                break;
            }
            case rcclUlong: {
                RcclInternalAllReduce<unsigned long, rccl_ulong2_t, rcclProd>(pcurr_track, sendbuff, recvbuff, stream, count, num_gpus, rank, event, this_time);
                break;
            }
            case rcclDouble: {
                RcclInternalAllReduce<double, rccl_double2_t, rcclProd>(pcurr_track, sendbuff, recvbuff, stream, count, num_gpus, rank, event, this_time);
                break;
            }
            default: {
                PostEnqueueEventRecord(pcomm, stream);
                return rcclInvalidType;
            }
        }
    }

    if(op == rcclMax) {
        switch(datatype) {
            case rcclChar: {
                RcclInternalAllReduce<signed char, rccl_char16_t, rcclMax>(pcurr_track, sendbuff, recvbuff, stream, count, num_gpus, rank, event, this_time);
                break;
            }
            case rcclUchar: {
                RcclInternalAllReduce<unsigned char, rccl_uchar16_t, rcclMax>(pcurr_track, sendbuff, recvbuff, stream, count, num_gpus, rank, event, this_time);
                break;
            }
            case rcclShort: {
                RcclInternalAllReduce<signed short, rccl_short8_t, rcclMax>(pcurr_track, sendbuff, recvbuff, stream, count, num_gpus, rank, event, this_time);
                break;
            }
            case rcclUshort: {
                RcclInternalAllReduce<unsigned short, rccl_ushort8_t, rcclMax>(pcurr_track, sendbuff, recvbuff, stream, count, num_gpus, rank, event, this_time);
                break;
            }
            case rcclHalf: {
                RcclInternalAllReduce<__fp16, rccl_half8_t, rcclMax>(pcurr_track, sendbuff, recvbuff, stream, count, num_gpus, rank, event, this_time);
                break;
            }
            case rcclInt: {
                RcclInternalAllReduce<signed int, rccl_int4_t, rcclMax>(pcurr_track, sendbuff, recvbuff, stream, count, num_gpus, rank, event, this_time);
                break;
            }
            case rcclUint: {
                RcclInternalAllReduce<unsigned int, rccl_uint4_t, rcclMax>(pcurr_track, sendbuff, recvbuff, stream, count, num_gpus, rank, event, this_time);
                break;
            }
            case rcclFloat: {
                RcclInternalAllReduce<float, rccl_float4_t, rcclMax>(pcurr_track, sendbuff, recvbuff, stream, count, num_gpus, rank, event, this_time);
                break;
            }
            case rcclLong: {
                RcclInternalAllReduce<signed long, rccl_long2_t, rcclMax>(pcurr_track, sendbuff, recvbuff, stream, count, num_gpus, rank, event, this_time);
                break;
            }
            case rcclUlong: {
                RcclInternalAllReduce<unsigned long, rccl_ulong2_t, rcclMax>(pcurr_track, sendbuff, recvbuff, stream, count, num_gpus, rank, event, this_time);
                break;
            }
            case rcclDouble: {
                RcclInternalAllReduce<double, rccl_double2_t, rcclMax>(pcurr_track, sendbuff, recvbuff, stream, count, num_gpus, rank, event, this_time);
                break;
            }
            default: {
                PostEnqueueEventRecord(pcomm, stream);
                return rcclInvalidType;
            }
        }
    }

    if(op == rcclMin) {
        switch(datatype) {
            case rcclChar: {
                RcclInternalAllReduce<signed char, rccl_char16_t, rcclMin>(pcurr_track, sendbuff, recvbuff, stream, count, num_gpus, rank, event, this_time);
                break;
            }
            case rcclUchar: {
                RcclInternalAllReduce<unsigned char, rccl_uchar16_t, rcclMin>(pcurr_track, sendbuff, recvbuff, stream, count, num_gpus, rank, event, this_time);
                break;
            }
            case rcclShort: {
                RcclInternalAllReduce<signed short, rccl_short8_t, rcclMin>(pcurr_track, sendbuff, recvbuff, stream, count, num_gpus, rank, event, this_time);
                break;
            }
            case rcclUshort: {
                RcclInternalAllReduce<unsigned short, rccl_ushort8_t, rcclMin>(pcurr_track, sendbuff, recvbuff, stream, count, num_gpus, rank, event, this_time);
                break;
            }
            case rcclHalf: {
                RcclInternalAllReduce<__fp16, rccl_half8_t, rcclMin>(pcurr_track, sendbuff, recvbuff, stream, count, num_gpus, rank, event, this_time);
                break;
            }
            case rcclInt: {
                RcclInternalAllReduce<signed int, rccl_int4_t, rcclMin>(pcurr_track, sendbuff, recvbuff, stream, count, num_gpus, rank, event, this_time);
                break;
            }
            case rcclUint: {
                RcclInternalAllReduce<unsigned int, rccl_uint4_t, rcclMin>(pcurr_track, sendbuff, recvbuff, stream, count, num_gpus, rank, event, this_time);
                break;
            }
            case rcclFloat: {
                RcclInternalAllReduce<float, rccl_float4_t, rcclMin>(pcurr_track, sendbuff, recvbuff, stream, count, num_gpus, rank, event, this_time);
                break;
            }
            case rcclLong: {
                RcclInternalAllReduce<signed long, rccl_long2_t, rcclMin>(pcurr_track, sendbuff, recvbuff, stream, count, num_gpus, rank, event, this_time);
                break;
            }
            case rcclUlong: {
                RcclInternalAllReduce<unsigned long, rccl_ulong2_t, rcclMin>(pcurr_track, sendbuff, recvbuff, stream, count, num_gpus, rank, event, this_time);
                break;
            }
            case rcclDouble: {
                RcclInternalAllReduce<double, rccl_double2_t, rcclMin>(pcurr_track, sendbuff, recvbuff, stream, count, num_gpus, rank, event, this_time);
                break;
            }
            default: {
                PostEnqueueEventRecord(pcomm, stream);
                return rcclInvalidType;
            }
        }
    }

    PostEnqueueEventRecord(pcomm, stream);

    return rcclSuccess;
}

rcclResult_t rcclBcast(void* buff, int count, rcclDataType_t datatype, int root, rcclComm_t comm, hipStream_t stream) {
    if((RCCL_TRACE_RT & krccl_print_api) == krccl_print_api) {
        int dev;
        hipGetDevice(&dev);
        fprintf(stderr, "%s<<rccl-api:%s rccl-device:%d buff:%p count:%d datatype:%s root:%d comm:%p stream:%p%s\n", API_COLOR, __func__, dev, buff, count, umap_datatype[datatype].c_str(), root, comm, stream, API_COLOR_END);
    }
    
    if(datatype >= rccl_NUM_TYPES) {
        return rcclInvalidType;
    }

    RcclComm_t* pcomm = comm;

    if(pcomm == nullptr || root < 0 || count <= 0) {
        return rcclInvalidArgument;
    }

    int num_gpus = pcomm->num_devices_;

    RingNode_t* pcurr_track = pcomm->track_;
    bool is_root = pcomm->track_->rank == root;

    int* this_time = &(pcomm->this_time_);

    PreEnqueueEventRecord(pcomm, stream);

    if(is_root) {
        RcclInternalBroadcastRoot(pcurr_track, stream, buff, this_time, num_gpus);
    } else {
        if(buff == nullptr) return rcclInvalidDevicePointer;
        RingNode_t* proot_track = pcurr_track->next_gpu;
        while(proot_track->rank != root) {
            proot_track = proot_track->next_gpu;
        }
        switch(datatype) {
            case rcclChar: {
                RcclInternalBroadcast<signed char>(pcurr_track, proot_track, count, stream, buff, this_time, num_gpus);
                break;
            }
            case rcclUchar: {
                RcclInternalBroadcast<unsigned char>(pcurr_track, proot_track, count, stream, buff, this_time, num_gpus);
                break;
            }
            case rcclShort: {
                RcclInternalBroadcast<signed short>(pcurr_track, proot_track, count, stream, buff, this_time, num_gpus);
                break;
            }
            case rcclUshort: {
                RcclInternalBroadcast<unsigned short>(pcurr_track, proot_track, count, stream, buff, this_time, num_gpus);
                break;
            }
            case rcclHalf: {
                RcclInternalBroadcast<__fp16>(pcurr_track, proot_track, count, stream, buff, this_time, num_gpus);
                break;
            }
            case rcclInt: {
                RcclInternalBroadcast<signed int>(pcurr_track, proot_track, count, stream, buff, this_time, num_gpus);
                break;
            }
            case rcclUint: {
                RcclInternalBroadcast<unsigned int>(pcurr_track, proot_track, count, stream, buff, this_time, num_gpus);
                break;
            }
            case rcclFloat: {
                RcclInternalBroadcast<float>(pcurr_track, proot_track, count, stream, buff, this_time, num_gpus);
                break;
            }
            case rcclLong: {
                RcclInternalBroadcast<signed long>(pcurr_track, proot_track, count, stream, buff, this_time, num_gpus);
                break;
            }
            case rcclUlong: {
                RcclInternalBroadcast<unsigned long>(pcurr_track, proot_track, count, stream, buff, this_time, num_gpus);
                break;
            }
            case rcclDouble: {
                RcclInternalBroadcast<double>(pcurr_track, proot_track, count, stream, buff, this_time, num_gpus);
                break;
            }
            default: {
                PostEnqueueEventRecord(pcomm, stream);
                return rcclInvalidType;
            }
        }
    }

    PostEnqueueEventRecord(pcomm, stream);
    return rcclSuccess;
}