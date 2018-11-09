/*
Copyright (c) 2017 - Present Advanced Micro Devices, Inc.
All rights reserved.
*/

/**
 * @file rccl.cpp
 * @brief rccl library implementation of communicator APIs
 *
 * This file contains implementation of just the communicator APIs. The Ops are
 * implemented in a different file.
 *
 * @author Aditya Atluri
 */

#include "rcclHelper.h"
#include "rcclTracker.h"
#include "rccl-version.h"

#include <string>
#include <unordered_map>
#include <vector>

#define MAKE_STR_PAIR(val) \
    { int(val), #val }

//! @brief Holds redOp_t to string hash table
std::unordered_map<int, std::string> umap_red_op = {
    MAKE_STR_PAIR(rcclSum), MAKE_STR_PAIR(rcclProd), MAKE_STR_PAIR(rcclMax),
    MAKE_STR_PAIR(rcclMin)};

//! @brief Holds rcclDataType_t to string hash table
std::unordered_map<int, std::string> umap_datatype = {
    MAKE_STR_PAIR(rcclChar),   MAKE_STR_PAIR(rcclInt),
    MAKE_STR_PAIR(rcclHalf),   MAKE_STR_PAIR(rcclFloat),
    MAKE_STR_PAIR(rcclDouble), MAKE_STR_PAIR(rcclInt64),
    MAKE_STR_PAIR(rcclUint64)};

// TODO: @adityaatluri, delete this variable
std::vector<RingNodePool_t *> pools;

//! @brief Internal representation of rcclUniqueId
struct RcclUniqueId {
    RingNodePool_t *pool;
    RcclUniqueId() { pool = new RingNodePool_t; }
    ~RcclUniqueId() { delete pool; }
};

//! @brief Get value of environment variable RCCL_TRACE_RT
const char *get_env_val = getenv("RCCL_TRACE_RT");
//! @brief Get debug trace level from environment variable
int RCCL_TRACE_RT = get_env_val != nullptr ? atoi(get_env_val) : 0;

//! @brief Implementation of rcclGetErrorString
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

//! @brief Definition of rcclGetUniqueId
rcclResult_t rcclGetUniqueId(rcclUniqueId *uniqueId) {
    if ((RCCL_TRACE_RT & krccl_print_api) == krccl_print_api) {
        fprintf(stderr, "%s<<rccl-api: %s uniqueId:%p%s\n", API_COLOR, __func__,
                uniqueId, API_COLOR_END);
    }

    //! Check if pointer to rcclUniqueId is valid or not
    if (uniqueId == nullptr) {
        return rcclInvalidArgument;
    }

    //! Allocate RcclUniqueId and return success
    *uniqueId = new RcclUniqueId;
    return rcclSuccess;
}

//! @brief Definition of rcclCommInitRank
rcclResult_t rcclCommInitRank(rcclComm_t *comm, int ndev, rcclUniqueId commId,
                              int rank) {
    if ((RCCL_TRACE_RT & krccl_print_api) == krccl_print_api) {
        fprintf(stderr,
                "%s<<rccl-api: %s RCCL version %d.%d.%d comm:%p ndev:%d, commId:%p rank:%d%s\n",
                API_COLOR, __func__, RCCL_VERSION_MAJOR, RCCL_VERSION_MINOR, RCCL_VERSION_PATCH,
                comm, ndev, commId, rank, API_COLOR_END);
    }

    //! Check if pointer to communicator is valid or not
    if (comm == nullptr) {
        return rcclInvalidArgument;
    }

    //! Check if rank of gpu is less than number of gpus in clique
    if (rank >= ndev) {
        return rcclInvalidRank;
    }

    if (ndev < 1) {
        return rcclUnsupportedDeviceCount;
    }

    //! Check if rcclUniqueId is valid or not
    if (commId == nullptr) {
        return rcclInvalidArgument;
    }

    auto pool = commId->pool;
    int dev;

    //! Check if the number of devices unique id is created is same as ndev
    if (pool->GetNumDevices() != 0) {
        if (ndev != pool->GetNumDevices()) {
            return rcclUnsupportedDeviceCount;
        }
    }

    //! Get current hip device index
    HIPCHECK(hipGetDevice(&dev));

    //! Add new GPU to the pool
    RcclComm_t *pcomm = pool->AddDevice(dev, rank, ndev);
    pcomm->pool_ = pool;

    //! Give communicator to application
    *comm = pcomm;
    return rcclSuccess;
}

//! @brief Definition of rcclCommInitAll
rcclResult_t rcclCommInitAll(rcclComm_t *comm, int ndev, int *devlist) {
    if ((RCCL_TRACE_RT & krccl_print_api) == krccl_print_api) {
        fprintf(stderr, "%s<<rccl-api: %s RCCL version %d.%d.%d comm:%p ndev:%d devlist:%p%s\n",
                API_COLOR, __func__, RCCL_VERSION_MAJOR, RCCL_VERSION_MINOR, RCCL_VERSION_PATCH,
                comm, ndev, devlist, API_COLOR_END);
    }

    //! Check if pointers and number of devices are valid
    if (comm == nullptr || devlist == nullptr || ndev < 1) {
        return rcclInvalidArgument;
    }

    //! Save current device set by user
    int user_device;
    HIPCHECK(hipGetDevice(&user_device));

    //! Check if the system contains number of gpus requested
    int device_count;
    HIPCHECK(hipGetDeviceCount(&device_count));
    if (ndev > device_count) {
        return rcclUnsupportedDeviceCount;
    }

    //! Check if the device indices are less the the number of devices present
    //! in the system
    for (int i = 0; i < ndev; i++) {
        if (devlist[i] >= ndev) return rcclDeviceNotFound;
    }

    //! If gpus are not peer enabled, enable them
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

    //! Create pool of RingNode_ts
    RingNodePool_t *ppool = new RingNodePool_t(devlist, ndev);

    RingNode_t *ptrack;

    //! Populate rcclComm_t using RingNodePool_t
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

    //! Restore saved device user
    HIPCHECK(hipSetDevice(user_device));

    return rcclSuccess;
}

//! @brief Declaration of rcclCommCuDevice
rcclResult_t rcclCommCuDevice(rcclComm_t comm, int *dev) {
    if ((RCCL_TRACE_RT & krccl_print_api) == krccl_print_api) {
        fprintf(stderr, "%s<<rccl-api: %s comm:%p *dev:%d dev:%p%s\n",
                API_COLOR, __func__, comm, *dev, dev, API_COLOR_END);
    }
    RcclComm_t *pcomm = comm;

    //! Get HIP device index from communicator
    *dev = pcomm->device_;
    return rcclSuccess;
}

//! @brief Declaration of rcclCommUserRank
rcclResult_t rcclCommUserRank(rcclComm_t comm, int *rank) {
    if ((RCCL_TRACE_RT & krccl_print_api) == krccl_print_api) {
        fprintf(stderr, "%s<<rccl-api: %s comm:%p *rank:%d rank:%p%s\n",
                API_COLOR, __func__, comm, *rank, rank, API_COLOR_END);
    }
    RcclComm_t *pcomm = comm;

    //! Get rank of gpu in clique from communicator
    *rank = pcomm->rank_;
    return rcclSuccess;
}

//! @brief Declaration of rcclCommCount
rcclResult_t rcclCommCount(rcclComm_t comm, int *count) {
    if ((RCCL_TRACE_RT & krccl_print_api) == krccl_print_api) {
        fprintf(stderr, "%s<<rccl-api: %s comm:%p *count:%d count:%p%s\n",
                API_COLOR, __func__, comm, *count, count, API_COLOR_END);
    }
    RcclComm_t *pcomm = comm;

    //! Get number of devices in clique from communicator
    *count = pcomm->num_devices_;
    return rcclSuccess;
}

//! @brief Declaration of rcclCommDestroy
rcclResult_t rcclCommDestroy(rcclComm_t comm) {
    if ((RCCL_TRACE_RT & krccl_print_api) == krccl_print_api) {
        fprintf(stderr, "%s<<rccl-api: %s comm:%p%s\n", API_COLOR, __func__,
                comm, API_COLOR_END);
    }
    RcclComm_t *pcomm = comm;

    //! Remove communicator from clique
    pcomm->pool_->RemoveDevice(pcomm);
    //! Free the pointer
    delete pcomm;
    return rcclSuccess;
}

//! @brief Declaration of PostEnqueueEventRecord
void PostEnqueueEventRecord(RcclComm_t *pcomm, hipStream_t stream) {
    if (stream != pcomm->stream_) {
        hipEventRecord(pcomm->event_, stream);
    }
}

//! @brief Declaration of PreEnqueueEventRecord
void PreEnqueueEventRecord(RcclComm_t *pcomm, hipStream_t stream) {
    if (stream != pcomm->stream_) {
        hipStreamWaitEvent(stream, pcomm->event_, 0);
        pcomm->stream_ = stream;
    }
}
