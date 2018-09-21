/*
Copyright (c) 2017 - Present Advanced Micro Devices, Inc.
All rights reserved.
*/

//
// This file contains implementation of rcclBcast.
//

#include "rcclDataTypes.h"
#include "rcclHelper.h"
#include "rcclSetKernels.h"
#include "rcclTracker.h"

#include "rcclScalarBroadcastRuntime.h"

#include <string>
#include <unordered_map>

extern std::unordered_map<int, std::string> umap_red_op;
extern std::unordered_map<int, std::string> umap_datatype;

extern int RCCL_TRACE_RT;

rcclResult_t rcclBcast(void *buff, int count, rcclDataType_t datatype, int root,
                       rcclComm_t comm, hipStream_t stream) {
    if ((RCCL_TRACE_RT & krccl_print_api) == krccl_print_api) {
        int dev;
        hipGetDevice(&dev);
        fprintf(stderr,
                "%s<<rccl-api:%s rccl-device:%d buff:%p count:%d datatype:%s "
                "root:%d comm:%p stream:%p%s\n",
                API_COLOR, __func__, dev, buff, count,
                umap_datatype[datatype].c_str(), root, comm, stream,
                API_COLOR_END);
    }

    //
    // Check if arguments are correct or not.
    //
    if (buff == nullptr) {
        return rcclInvalidDevicePointer;
    }

    if (datatype >= rccl_NUM_TYPES) {
        return rcclInvalidType;
    }

    RcclComm_t *pcomm = comm;

    if (pcomm == nullptr || root < 0 || count <= 0) {
        return rcclInvalidArgument;
    }

    int num_gpus = pcomm->num_devices_;

    if (root >= num_gpus) {
        return rcclInvalidArgument;
    }

    //
    // Get current value of barrier
    //
    int *this_time = &(pcomm->this_time_);

    //
    // If same comm is used on a different stream,
    // synchronize it with current stream before launching op.
    //
    PreEnqueueEventRecord(pcomm, stream);

    RingNode_t *pcurr_track = pcomm->track_;

    //
    // Check if current gpu is root or not
    //
    bool is_root = pcomm->track_->rank == root;

    //
    // If current gpu is root, call internal implementation for root
    //
    if (is_root) {
        RcclInternalBroadcastRoot(pcurr_track, stream, buff, this_time,
                                  num_gpus);
    }
    //
    // If current gpu is not root, call internal implementation
    //
    else {
        //
        // Get RingNode for current gpu
        //
        RingNode_t *proot_track = pcurr_track->next_gpu;
        while (proot_track->rank != root) {
            proot_track = proot_track->next_gpu;
        }

        switch (datatype) {
        case rcclChar: {
            RcclInternalBroadcast<signed char>(pcurr_track, proot_track, count,
                                               stream, buff, this_time,
                                               num_gpus);
            break;
        }
        case rcclUchar: {
            RcclInternalBroadcast<unsigned char>(pcurr_track, proot_track,
                                                 count, stream, buff, this_time,
                                                 num_gpus);
            break;
        }
        case rcclShort: {
            RcclInternalBroadcast<signed short>(pcurr_track, proot_track, count,
                                                stream, buff, this_time,
                                                num_gpus);
            break;
        }
        case rcclUshort: {
            RcclInternalBroadcast<unsigned short>(pcurr_track, proot_track,
                                                  count, stream, buff,
                                                  this_time, num_gpus);
            break;
        }
        case rcclHalf: {
            RcclInternalBroadcast<__fp16>(pcurr_track, proot_track, count,
                                          stream, buff, this_time, num_gpus);
            break;
        }
        case rcclInt: {
            RcclInternalBroadcast<signed int>(pcurr_track, proot_track, count,
                                              stream, buff, this_time,
                                              num_gpus);
            break;
        }
        case rcclUint: {
            RcclInternalBroadcast<unsigned int>(pcurr_track, proot_track, count,
                                                stream, buff, this_time,
                                                num_gpus);
            break;
        }
        case rcclFloat: {
            RcclInternalBroadcast<float>(pcurr_track, proot_track, count,
                                         stream, buff, this_time, num_gpus);
            break;
        }
        case rcclLong: {
            RcclInternalBroadcast<signed long>(pcurr_track, proot_track, count,
                                               stream, buff, this_time,
                                               num_gpus);
            break;
        }
        case rcclUlong: {
            RcclInternalBroadcast<unsigned long>(pcurr_track, proot_track,
                                                 count, stream, buff, this_time,
                                                 num_gpus);
            break;
        }
        case rcclDouble: {
            RcclInternalBroadcast<double>(pcurr_track, proot_track, count,
                                          stream, buff, this_time, num_gpus);
            break;
        }
        default: {
            return rcclInvalidType;
        }
        }
    }

    //
    // Track current stream so that op launched on different stream can be
    // synchronized with current stream
    //
    PostEnqueueEventRecord(pcomm, stream);
    return rcclSuccess;
}
