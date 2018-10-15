/*
Copyright (c) 2017 - Present Advanced Micro Devices, Inc.
All rights reserved.
*/

/**
 * @file rcclAllReduce.cpp
 * @brief rccl library implementation of rcclAllGather API
 *
 * This file contains implementation of rcclAllGather API.
 *
 */

#include "rcclDataTypes.h"
#include "rcclHelper.h"
#include "rcclSetKernels.h"
#include "rcclTracker.h"

#include "rcclScalarAllGatherRuntime.h"

#include <string>
#include <unordered_map>

extern std::unordered_map<int, std::string> umap_datatype;

extern int RCCL_TRACE_RT;

//! @brief Definition of rcclAllGather
rcclResult_t rcclAllGather(const void *sendbuff, int count,
                           rcclDataType_t datatype, void *recvbuff,
                           rcclComm_t comm, hipStream_t stream) {
    if ((RCCL_TRACE_RT & krccl_print_api) == krccl_print_api) {
        int dev;
        hipGetDevice(&dev);
        fprintf(stderr,
                "%s<<rccl-api:%s rccl-device:%d sendbuff:%p recvbuff:%p "
                "count:%d datatype:%s comm:%p stream:%p%s\n",
                API_COLOR, __func__, dev, sendbuff, recvbuff, count,
                umap_datatype[datatype].c_str(), comm, stream, API_COLOR_END);
    }

    //! Check if buffer pointers are not null
    if (sendbuff == nullptr || recvbuff == nullptr) {
        return rcclInvalidDevicePointer;
    }

    //! Check if data type of buffers is valid or not
    if (datatype >= rccl_NUM_TYPES) {
        return rcclInvalidType;
    }

    //! Get internal communicator from rcclComm_t
    RcclComm_t *pcomm = comm;

    //! Check if communicator is valid or number of elements is > 0
    if (pcomm == nullptr || count <= 0) {
        return rcclInvalidArgument;
    }

    int rank = pcomm->rank_;
    int num_gpus = pcomm->num_devices_;
    hipEvent_t event = pcomm->event_;

    //! Get pointer to current barrier
    int *this_time = &(pcomm->this_time_);

    //! If same comm is used on a different stream, synchronize it with current
    //! stream before launching op.
    PreEnqueueEventRecord(pcomm, stream);

    //! Get tracker to current gpu
    RingNode_t *pcurr_track = pcomm->track_;

    //! If the number of gpus equal to 1, do a simple memory copy
    if (num_gpus == 1) {
        switch (datatype) {
        case rcclChar:
        case rcclUchar: {
            hipMemcpyAsync(recvbuff, sendbuff, count * sizeof(char),
                           hipMemcpyDeviceToDevice, stream);
            break;
        }
        case rcclShort:
        case rcclUshort:
        case rcclHalf: {
            hipMemcpyAsync(recvbuff, sendbuff, count * sizeof(short),
                           hipMemcpyDeviceToDevice, stream);
            break;
        }
        case rcclInt:
        case rcclUint:
        case rcclFloat: {
            hipMemcpyAsync(recvbuff, sendbuff, count * sizeof(int),
                           hipMemcpyDeviceToDevice, stream);
            break;
        }
        case rcclLong:
        case rcclUlong:
        case rcclDouble: {
            hipMemcpyAsync(recvbuff, sendbuff, count * sizeof(double),
                           hipMemcpyDeviceToDevice, stream);
            break;
        }
        default: { return rcclInvalidType; }
        }

        //! Track current stream so that op launched on different stream can be
        //! synchronized with current stream
        PostEnqueueEventRecord(pcomm, stream);
        return rcclSuccess;
    }

    switch (datatype) {
    case rcclChar: {
        RcclInternalAllGather<signed char, rccl_char16_t>(
            pcurr_track, sendbuff, recvbuff, stream, count, num_gpus, rank,
            event, this_time);
        break;
    }
    case rcclUchar: {
        RcclInternalAllGather<unsigned char, rccl_uchar16_t>(
            pcurr_track, sendbuff, recvbuff, stream, count, num_gpus, rank,
            event, this_time);
        break;
    }
    case rcclShort: {
        RcclInternalAllGather<signed short, rccl_short8_t>(
            pcurr_track, sendbuff, recvbuff, stream, count, num_gpus, rank,
            event, this_time);
        break;
    }
    case rcclUshort: {
        RcclInternalAllGather<unsigned short, rccl_ushort8_t>(
            pcurr_track, sendbuff, recvbuff, stream, count, num_gpus, rank,
            event, this_time);
        break;
    }
    case rcclHalf: {
        RcclInternalAllGather<__fp16, rccl_half8_t>(
            pcurr_track, sendbuff, recvbuff, stream, count, num_gpus, rank,
            event, this_time);
        break;
    }
    case rcclInt: {
        RcclInternalAllGather<signed int, rccl_int4_t>(
            pcurr_track, sendbuff, recvbuff, stream, count, num_gpus, rank,
            event, this_time);
        break;
    }
    case rcclUint: {
        RcclInternalAllGather<unsigned int, rccl_uint4_t>(
            pcurr_track, sendbuff, recvbuff, stream, count, num_gpus, rank,
            event, this_time);
        break;
    }
    case rcclFloat: {
        RcclInternalAllGather<float, rccl_float4_t>(
            pcurr_track, sendbuff, recvbuff, stream, count, num_gpus, rank,
            event, this_time);
        break;
    }
    case rcclLong: {
        RcclInternalAllGather<signed long, rccl_long2_t>(
            pcurr_track, sendbuff, recvbuff, stream, count, num_gpus, rank,
            event, this_time);
        break;
    }
    case rcclUlong: {
        RcclInternalAllGather<unsigned long, rccl_ulong2_t>(
            pcurr_track, sendbuff, recvbuff, stream, count, num_gpus, rank,
            event, this_time);
        break;
    }
    case rcclDouble: {
        RcclInternalAllGather<double, rccl_double2_t>(
            pcurr_track, sendbuff, recvbuff, stream, count, num_gpus, rank,
            event, this_time);
        break;
    }
    default: { return rcclInvalidType; }
    }

    //! Track current stream so that op launched on different stream can be
    //! synchronized with current stream
    PostEnqueueEventRecord(pcomm, stream);
    return rcclSuccess;
}
