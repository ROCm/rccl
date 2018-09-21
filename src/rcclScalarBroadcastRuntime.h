/*
Copyright (c) 2017-Present Advanced Micro Devices, Inc.
All rights reserved.
*/

//
// This file contains code which launches kernels for rcclBcast
//

#pragma once

#include "rcclBarrierKernels.h"
#include "rcclScalarBroadcastKernels.h"

//
// The root does not do the copy
//
void RcclInternalBroadcastRoot(RingNode_t* pcurr_track, hipStream_t stream,
                               void* send_buff, int* this_time, int num_gpus) {
    //
    // Set source pointer on root gpu
    //
    hipLaunchKernelGGL((RcclKernelSetSrcPtr), dim3(1, 1, 1), dim3(1, 1, 1), 0,
                       stream, pcurr_track, send_buff);

    int barrier_value = *this_time;

    //
    // Wait until root gpu sets its source pointer
    //
    hipLaunchKernelGGL((RcclKernelBarrierWait), dim3(1, 1, 1), dim3(1, 1, 1), 0,
                       stream, pcurr_track, barrier_value++, num_gpus);

    //
    // wait until everyone finished reading
    //
    hipLaunchKernelGGL((RcclKernelBarrierWait), dim3(1, 1, 1), dim3(1, 1, 1), 0,
                       stream, pcurr_track, barrier_value++, num_gpus);

    //
    // Update how many times barrier is used
    //
    *this_time = barrier_value;
}

template <typename DataType_t>
void RcclInternalBroadcast(RingNode_t* pcurr_track, RingNode_t* proot_track,
                           int count, hipStream_t stream, void* recv_buff,
                           int* this_time, int num_gpus) {
    bool check_count = count > knum_workitems;
    int num_workitems = check_count ? knum_workitems : count;
    int num_workgroups = check_count ? count / knum_workitems + 1 : 1;

    int barrier_value = *this_time;

    //
    // Wait until root gpu sets its source pointer
    //
    hipLaunchKernelGGL((RcclKernelBarrierWait), dim3(1, 1, 1), dim3(1, 1, 1), 0,
                       stream, pcurr_track, barrier_value++, num_gpus);

    //
    // Read data from root gpu
    //
    hipLaunchKernelGGL((RcclKernelScalarCopyFromRoot<DataType_t>),
                       dim3(num_workgroups, 1, 1), dim3(num_workitems, 1, 1), 0,
                       stream, proot_track, recv_buff, count);

    //
    // Wait until everyone finishes reading
    //
    hipLaunchKernelGGL((RcclKernelBarrierWait), dim3(1, 1, 1), dim3(1, 1, 1), 0,
                       stream, pcurr_track, barrier_value++, num_gpus);

    //
    // Update how many times barrier is used
    //
    *this_time = barrier_value;
}
