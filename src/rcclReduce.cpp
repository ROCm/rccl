/*
Copyright (c) 2017 - Present Advanced Micro Devices, Inc.
All rights reserved.
*/

//
// This file contains implementation of rcclReduce.
//

#include "rcclDataTypes.h"
#include "rcclHelper.h"
#include "rcclSetKernels.h"
#include "rcclTracker.h"

#include "rcclScalarReduceRuntime.h"

#include <string>
#include <unordered_map>
#include <vector>

extern std::unordered_map<int, std::string> umap_red_op;
extern std::unordered_map<int, std::string> umap_datatype;

extern int RCCL_TRACE_RT;

rcclResult_t rcclReduce(const void *sendbuff, void *recvbuff, int count,
                        rcclDataType_t datatype, rcclRedOp_t op, int root,
                        rcclComm_t comm, hipStream_t stream) {
    if ((RCCL_TRACE_RT & krccl_print_api) == krccl_print_api) {
        int dev;
        hipGetDevice(&dev);
        fprintf(stderr,
                "%s<<rccl-api:%s rccl-device:%d sendbuff:%p recvbuff:%p "
                "count:%d datatype:%s op:%s root:%d comm:%p stream:%p%s\n",
                API_COLOR, __func__, dev, sendbuff, recvbuff, count,
                umap_datatype[datatype].c_str(), umap_red_op[op].c_str(), root,
                comm, stream, API_COLOR_END);
    }

    //
    // Check if arguments are correct or not.
    //
    if (sendbuff == nullptr) {
        return rcclInvalidDevicePointer;
    }

    if (datatype >= rccl_NUM_TYPES) {
        return rcclInvalidType;
    }

    if (op >= rccl_NUM_OPS) {
        return rcclInvalidOperation;
    }

    RcclComm_t *pcomm = comm;

    if (pcomm == nullptr || count <= 0 || root < 0) {
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

    if (is_root) {
        if (recvbuff == nullptr) return rcclInvalidDevicePointer;
        if (op == rcclSum) {
            switch (datatype) {
            case rcclChar: {
                RcclInternalReduce<signed char, rccl_char16_t, rcclSum>(
                    pcurr_track, count, stream, sendbuff, recvbuff, this_time,
                    num_gpus);
                break;
            }
            case rcclUchar: {
                RcclInternalReduce<unsigned char, rccl_uchar16_t, rcclSum>(
                    pcurr_track, count, stream, sendbuff, recvbuff, this_time,
                    num_gpus);
                break;
            }
            case rcclShort: {
                RcclInternalReduce<signed short, rccl_short8_t, rcclSum>(
                    pcurr_track, count, stream, sendbuff, recvbuff, this_time,
                    num_gpus);
                break;
            }
            case rcclUshort: {
                RcclInternalReduce<unsigned short, rccl_ushort8_t, rcclSum>(
                    pcurr_track, count, stream, sendbuff, recvbuff, this_time,
                    num_gpus);
                break;
            }
            case rcclHalf: {
                RcclInternalReduce<__fp16, rccl_half8_t, rcclSum>(
                    pcurr_track, count, stream, sendbuff, recvbuff, this_time,
                    num_gpus);
                break;
            }
            case rcclInt: {
                RcclInternalReduce<signed int, rccl_int4_t, rcclSum>(
                    pcurr_track, count, stream, sendbuff, recvbuff, this_time,
                    num_gpus);
                break;
            }
            case rcclUint: {
                RcclInternalReduce<unsigned int, rccl_uint4_t, rcclSum>(
                    pcurr_track, count, stream, sendbuff, recvbuff, this_time,
                    num_gpus);
                break;
            }
            case rcclFloat: {
                RcclInternalReduce<float, rccl_float4_t, rcclSum>(
                    pcurr_track, count, stream, sendbuff, recvbuff, this_time,
                    num_gpus);
                break;
            }
            case rcclLong: {
                RcclInternalReduce<signed long, rccl_long2_t, rcclSum>(
                    pcurr_track, count, stream, sendbuff, recvbuff, this_time,
                    num_gpus);
                break;
            }
            case rcclUlong: {
                RcclInternalReduce<unsigned long, rccl_ulong2_t, rcclSum>(
                    pcurr_track, count, stream, sendbuff, recvbuff, this_time,
                    num_gpus);
                break;
            }
            case rcclDouble: {
                RcclInternalReduce<double, rccl_double2_t, rcclSum>(
                    pcurr_track, count, stream, sendbuff, recvbuff, this_time,
                    num_gpus);
                break;
            }
            default: { return rcclInvalidType; }
            }
        }
        if (op == rcclProd) {
            switch (datatype) {
            case rcclChar: {
                RcclInternalReduce<signed char, rccl_char16_t, rcclProd>(
                    pcurr_track, count, stream, sendbuff, recvbuff, this_time,
                    num_gpus);
                break;
            }
            case rcclUchar: {
                RcclInternalReduce<unsigned char, rccl_uchar16_t, rcclProd>(
                    pcurr_track, count, stream, sendbuff, recvbuff, this_time,
                    num_gpus);
                break;
            }
            case rcclShort: {
                RcclInternalReduce<signed short, rccl_short8_t, rcclProd>(
                    pcurr_track, count, stream, sendbuff, recvbuff, this_time,
                    num_gpus);
                break;
            }
            case rcclUshort: {
                RcclInternalReduce<unsigned short, rccl_ushort8_t, rcclProd>(
                    pcurr_track, count, stream, sendbuff, recvbuff, this_time,
                    num_gpus);
                break;
            }
            case rcclHalf: {
                RcclInternalReduce<__fp16, rccl_half8_t, rcclProd>(
                    pcurr_track, count, stream, sendbuff, recvbuff, this_time,
                    num_gpus);
                break;
            }
            case rcclInt: {
                RcclInternalReduce<signed int, rccl_int4_t, rcclProd>(
                    pcurr_track, count, stream, sendbuff, recvbuff, this_time,
                    num_gpus);
                break;
            }
            case rcclUint: {
                RcclInternalReduce<unsigned int, rccl_uint4_t, rcclProd>(
                    pcurr_track, count, stream, sendbuff, recvbuff, this_time,
                    num_gpus);
                break;
            }
            case rcclFloat: {
                RcclInternalReduce<float, rccl_float4_t, rcclProd>(
                    pcurr_track, count, stream, sendbuff, recvbuff, this_time,
                    num_gpus);
                break;
            }
            case rcclLong: {
                RcclInternalReduce<signed long, rccl_long2_t, rcclProd>(
                    pcurr_track, count, stream, sendbuff, recvbuff, this_time,
                    num_gpus);
                break;
            }
            case rcclUlong: {
                RcclInternalReduce<unsigned long, rccl_ulong2_t, rcclProd>(
                    pcurr_track, count, stream, sendbuff, recvbuff, this_time,
                    num_gpus);
                break;
            }
            case rcclDouble: {
                RcclInternalReduce<double, rccl_double2_t, rcclProd>(
                    pcurr_track, count, stream, sendbuff, recvbuff, this_time,
                    num_gpus);
                break;
            }
            default: { return rcclInvalidType; }
            }
        }

        if (op == rcclMax) {
            switch (datatype) {
            case rcclChar: {
                RcclInternalReduce<signed char, rccl_char16_t, rcclMax>(
                    pcurr_track, count, stream, sendbuff, recvbuff, this_time,
                    num_gpus);
                break;
            }
            case rcclUchar: {
                RcclInternalReduce<unsigned char, rccl_uchar16_t, rcclMax>(
                    pcurr_track, count, stream, sendbuff, recvbuff, this_time,
                    num_gpus);
                break;
            }
            case rcclShort: {
                RcclInternalReduce<signed short, rccl_short8_t, rcclMax>(
                    pcurr_track, count, stream, sendbuff, recvbuff, this_time,
                    num_gpus);
                break;
            }
            case rcclUshort: {
                RcclInternalReduce<unsigned short, rccl_ushort8_t, rcclMax>(
                    pcurr_track, count, stream, sendbuff, recvbuff, this_time,
                    num_gpus);
                break;
            }
            case rcclHalf: {
                RcclInternalReduce<__fp16, rccl_half8_t, rcclMax>(
                    pcurr_track, count, stream, sendbuff, recvbuff, this_time,
                    num_gpus);
                break;
            }
            case rcclInt: {
                RcclInternalReduce<signed int, rccl_int4_t, rcclMax>(
                    pcurr_track, count, stream, sendbuff, recvbuff, this_time,
                    num_gpus);
                break;
            }
            case rcclUint: {
                RcclInternalReduce<unsigned int, rccl_uint4_t, rcclMax>(
                    pcurr_track, count, stream, sendbuff, recvbuff, this_time,
                    num_gpus);
                break;
            }
            case rcclFloat: {
                RcclInternalReduce<float, rccl_float4_t, rcclMax>(
                    pcurr_track, count, stream, sendbuff, recvbuff, this_time,
                    num_gpus);
                break;
            }
            case rcclLong: {
                RcclInternalReduce<signed long, rccl_long2_t, rcclMax>(
                    pcurr_track, count, stream, sendbuff, recvbuff, this_time,
                    num_gpus);
                break;
            }
            case rcclUlong: {
                RcclInternalReduce<unsigned long, rccl_ulong2_t, rcclMax>(
                    pcurr_track, count, stream, sendbuff, recvbuff, this_time,
                    num_gpus);
                break;
            }
            case rcclDouble: {
                RcclInternalReduce<double, rccl_double2_t, rcclMax>(
                    pcurr_track, count, stream, sendbuff, recvbuff, this_time,
                    num_gpus);
                break;
            }
            default: { return rcclInvalidType; }
            }
        }

        if (op == rcclMin) {
            switch (datatype) {
            case rcclChar: {
                RcclInternalReduce<signed char, rccl_char16_t, rcclMin>(
                    pcurr_track, count, stream, sendbuff, recvbuff, this_time,
                    num_gpus);
                break;
            }
            case rcclUchar: {
                RcclInternalReduce<unsigned char, rccl_uchar16_t, rcclMin>(
                    pcurr_track, count, stream, sendbuff, recvbuff, this_time,
                    num_gpus);
                break;
            }
            case rcclShort: {
                RcclInternalReduce<signed short, rccl_short8_t, rcclMin>(
                    pcurr_track, count, stream, sendbuff, recvbuff, this_time,
                    num_gpus);
                break;
            }
            case rcclUshort: {
                RcclInternalReduce<unsigned short, rccl_ushort8_t, rcclMin>(
                    pcurr_track, count, stream, sendbuff, recvbuff, this_time,
                    num_gpus);
                break;
            }
            case rcclHalf: {
                RcclInternalReduce<__fp16, rccl_half8_t, rcclMin>(
                    pcurr_track, count, stream, sendbuff, recvbuff, this_time,
                    num_gpus);
                break;
            }
            case rcclInt: {
                RcclInternalReduce<signed int, rccl_int4_t, rcclMin>(
                    pcurr_track, count, stream, sendbuff, recvbuff, this_time,
                    num_gpus);
                break;
            }
            case rcclUint: {
                RcclInternalReduce<unsigned int, rccl_uint4_t, rcclMin>(
                    pcurr_track, count, stream, sendbuff, recvbuff, this_time,
                    num_gpus);
                break;
            }
            case rcclFloat: {
                RcclInternalReduce<float, rccl_float4_t, rcclMin>(
                    pcurr_track, count, stream, sendbuff, recvbuff, this_time,
                    num_gpus);
                break;
            }
            case rcclLong: {
                RcclInternalReduce<signed long, rccl_long2_t, rcclMin>(
                    pcurr_track, count, stream, sendbuff, recvbuff, this_time,
                    num_gpus);
                break;
            }
            case rcclUlong: {
                RcclInternalReduce<unsigned long, rccl_ulong2_t, rcclMin>(
                    pcurr_track, count, stream, sendbuff, recvbuff, this_time,
                    num_gpus);
                break;
            }
            case rcclDouble: {
                RcclInternalReduce<double, rccl_double2_t, rcclMin>(
                    pcurr_track, count, stream, sendbuff, recvbuff, this_time,
                    num_gpus);
                break;
            }
            default: { return rcclInvalidType; }
            }
        }

    } else {
        RcclInternalReduceNotRoot(pcurr_track, stream, sendbuff, this_time,
                                  num_gpus);
    }

    //
    // Track current stream so that op launched on different stream can be
    // synchronized with current stream
    //
    PostEnqueueEventRecord(pcomm, stream);
    return rcclSuccess;
}
