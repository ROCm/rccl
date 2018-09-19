/*
Copyright (c) 2017 - Present Advanced Micro Devices, Inc.
All rights reserved.
*/

#pragma once

#include "rcclBarrierKernels.h"
#include "rcclScalarReduceKernels.h"

//
// The code here figures out the launch parameters for reduce op
//

extern int RCCL_TRACE_RT;

template <typename DataType_t, typename VectorType_t, rcclRedOp_t Op>
void RcclInternalReduce(RingNode_t* pcurr_track, int count, hipStream_t stream,
                        const void* send_buff, void* recv_buff, int* this_time,
                        int num_gpus) {
    int num_workitems = 0, num_workgroups = 0;

    if (count > knum_workitems) {
        num_workitems = knum_workitems;
        num_workgroups = count / knum_workitems + 1;
    } else {
        num_workitems = count;
        num_workgroups = 1;
    }

    int barrier_value = *this_time;

    hipLaunchKernelGGL(RcclKernelBarrierWait, dim3(1, 1, 1), dim3(1, 1, 1), 0,
                       stream, pcurr_track, barrier_value++, num_gpus);

    hipLaunchKernelGGL((RcclKernelScalarReduce<DataType_t, Op>),
                       dim3(num_workgroups, 1, 1), dim3(num_workitems, 1, 1), 0,
                       stream, pcurr_track, send_buff, recv_buff, count);

    hipLaunchKernelGGL(RcclKernelBarrierWait, dim3(1, 1, 1), dim3(1, 1, 1), 0,
                       stream, pcurr_track, barrier_value++, num_gpus);

    *this_time = barrier_value;
}

void RcclInternalReduceNotRoot(RingNode_t* pcurr_track, hipStream_t stream,
                               const void* send_buff, int* this_time,
                               int num_gpus) {
    int barrier_value = *this_time;

    hipLaunchKernelGGL(RcclKernelSetSrcPtr, dim3(1, 1, 1), dim3(1, 1, 1), 0,
                       stream, pcurr_track, (void*)send_buff);

    hipLaunchKernelGGL(RcclKernelBarrierWait, dim3(1, 1, 1), dim3(1, 1, 1), 0,
                       stream, pcurr_track, barrier_value++, num_gpus);

    hipLaunchKernelGGL(RcclKernelBarrierWait, dim3(1, 1, 1), dim3(1, 1, 1), 0,
                       stream, pcurr_track, barrier_value++, num_gpus);

    *this_time = barrier_value;
}
