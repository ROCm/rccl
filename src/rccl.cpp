/*
Copyright (c) 2017 - Present Advanced Micro Devices, Inc.
All rights reserved.
*/

#include "rcclDataTypes.h"
#include "rcclHelper.h"
#include "rcclSetKernels.h"
#include "rcclTracker.h"

#include <string>
#include <unordered_map>
#include <vector>

//
// All rccl apis are implemented here.
// Ops are implemented in a different header file
// one header file for each op
//

#define MAKE_STR_PAIR(val) \
    { int(val), #val }

std::unordered_map<int, std::string> umap_red_op = {
    MAKE_STR_PAIR(rcclSum), MAKE_STR_PAIR(rcclProd), MAKE_STR_PAIR(rcclMax),
    MAKE_STR_PAIR(rcclMin)};

std::unordered_map<int, std::string> umap_datatype = {
    MAKE_STR_PAIR(rcclUchar),  MAKE_STR_PAIR(rcclChar),
    MAKE_STR_PAIR(rcclUshort), MAKE_STR_PAIR(rcclShort),
    MAKE_STR_PAIR(rcclUint),   MAKE_STR_PAIR(rcclInt),
    MAKE_STR_PAIR(rcclUlong),  MAKE_STR_PAIR(rcclLong),
    MAKE_STR_PAIR(rcclFloat),  MAKE_STR_PAIR(rcclHalf),
    MAKE_STR_PAIR(rcclDouble)};

std::vector<RingNodePool_t *> pools;

// used as rcclUniqueId
struct RcclUniqueId {
    RingNodePool_t *pool;
    RcclUniqueId() { pool = new RingNodePool_t; }
    ~RcclUniqueId() { delete pool; }
};

const char *get_env_val = getenv("RCCL_TRACE_RT");
int RCCL_TRACE_RT = get_env_val != nullptr ? atoi(get_env_val) : 0;

// implementation of rcclGetErrorString api
const char *rcclGetErrorString(rcclResult_t result) {
    switch (result) {
    case rcclSuccess:
        return "rcclSuccess";
    case rcclUnhandledHipError:
        return "rcclUnhandledHipError";
    case rcclSystemError:
        return "rcclSystemError";
    case rcclInternalError:
        return "rcclInternalError";
    case rcclInvalidDevicePointer:
        return "rcclInvalidDevicePointer";
    case rcclInvalidRank:
        return "rcclInvalidRank";
    case rcclUnsupportedDeviceCount:
        return "rcclUnsupportedDeviceCount";
    case rcclDeviceNotFound:
        return "rcclDeviceNotFound";
    case rcclInvalidDeviceIndex:
        return "rcclInvalidDeviceIndex";
    case rcclLibWrapperNotSet:
        return "rcclLibWrapperNotSet";
    case rcclHipMallocFailed:
        return "rcclHipMallocFailed";
    case rcclRankMismatch:
        return "rcclRankMismatch";
    case rcclInvalidArgument:
        return "rcclInvalidArgument";
    case rcclInvalidType:
        return "rcclInvalidType";
    case rcclInvalidOperation:
        return "rcclInvalidOperation";
    default:
        return "rcclErrorNotFound";
    }
}

rcclResult_t rcclGetUniqueId(rcclUniqueId *uniqueId) {
    if (uniqueId == nullptr) {
        return rcclInvalidArgument;
    }
    *uniqueId = new RcclUniqueId;
    return rcclSuccess;
}

rcclResult_t rcclCommInitRank(rcclComm_t *comm, int ndev, rcclUniqueId commId,
                              int rank) {
    if (comm == nullptr) {
        return rcclInvalidArgument;
    }
    if (rank >= ndev) {
        return rcclInvalidRank;
    }
    if (commId == nullptr) {
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
    if ((RCCL_TRACE_RT & krccl_print_api) == krccl_print_api) {
        fprintf(stderr, "%s<<rccl-api: %s comm:%p ndev:%d devlist:%p%s\n",
                API_COLOR, __func__, comm, ndev, devlist, API_COLOR_END);
    }
    if (comm == nullptr || devlist == nullptr || ndev < 1) {
        return rcclInvalidArgument;
    }

    // save current device set by user
    int user_device;
    HIPCHECK(hipGetDevice(&user_device));

    int device_count;
    HIPCHECK(hipGetDeviceCount(&device_count));
    if (ndev > device_count) {
        return rcclUnsupportedDeviceCount;
    }

    // if gpus are not peer enabled, enable them
    for (int i = 0; i < ndev; i++) {
        HIPCHECK(hipSetDevice(devlist[i]));
        for (int j = 0; j < ndev; j++) {
            if (devlist[i] != devlist[j]) {
                hipError_t err = hipDeviceEnablePeerAccess(devlist[j], 0);
                if (err != hipErrorPeerAccessAlreadyEnabled &&
                    err != hipSuccess) {
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
    for (int i = 0; i < ndev; i++) {
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
        HIPCHECK(
            hipEventCreateWithFlags(&pcomm->event_, hipEventReleaseToSystem));
    }

    // restore saved device user
    HIPCHECK(hipSetDevice(user_device));

    return rcclSuccess;
}

rcclResult_t rcclCommCuDevice(rcclComm_t comm, int *dev) {
    if ((RCCL_TRACE_RT & krccl_print_api) == krccl_print_api) {
        fprintf(stderr, "%s<<rccl-api: %s comm:%p *dev:%d dev:%p%s\n",
                API_COLOR, __func__, comm, *dev, dev, API_COLOR_END);
    }
    RcclComm_t *pcomm = comm;
    *dev = pcomm->device_;
    return rcclSuccess;
}

rcclResult_t rcclCommUserRank(rcclComm_t comm, int *rank) {
    if ((RCCL_TRACE_RT & krccl_print_api) == krccl_print_api) {
        fprintf(stderr, "%s<<rccl-api: %s comm:%p *rank:%d rank:%p%s\n",
                API_COLOR, __func__, comm, *rank, rank, API_COLOR_END);
    }
    RcclComm_t *pcomm = comm;
    *rank = pcomm->rank_;
    return rcclSuccess;
}

rcclResult_t rcclCommCount(rcclComm_t comm, int *count) {
    if ((RCCL_TRACE_RT & krccl_print_api) == krccl_print_api) {
        fprintf(stderr, "%s<<rccl-api: %s comm:%p *count:%d count:%p%s\n",
                API_COLOR, __func__, comm, *count, count, API_COLOR_END);
    }
    RcclComm_t *pcomm = comm;
    *count = pcomm->num_devices_;
    return rcclSuccess;
}

rcclResult_t rcclCommDestroy(rcclComm_t comm) {
    if ((RCCL_TRACE_RT & krccl_print_api) == krccl_print_api) {
        fprintf(stderr, "%s<<rccl-api: %s comm:%p%s\n", API_COLOR, __func__,
                comm, API_COLOR_END);
    }
    RcclComm_t *pcomm = comm;
    pcomm->pool_->active_devices_--;
    if (pcomm->pool_->active_devices_ == 0) {
        delete pcomm->pool_;
    }
    HIPCHECK(hipEventDestroy(pcomm->event_));
    delete pcomm;
    return rcclSuccess;
}

void PostEnqueueEventRecord(RcclComm_t *pcomm, hipStream_t stream) {
    if (stream != pcomm->stream_) {
        hipEventRecord(pcomm->event_, stream);
    }
}

void PreEnqueueEventRecord(RcclComm_t *pcomm, hipStream_t stream) {
    if (stream != pcomm->stream_) {
        hipStreamWaitEvent(stream, pcomm->event_, 0);
        pcomm->stream_ = stream;
    }
}
