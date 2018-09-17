/*
Copyright (c) 2017-Present Advanced Micro Devices, Inc. 
All rights reserved.
*/

#pragma once

#include "rcclScalarBroadcastKernels.h"
#include "rcclBarrierKernels.h"

//
// The code here figures out the launch parameters for broadcast op
//

//
// The root does not do the copy
//
void RcclInternalBroadcastRoot(RingNode_t* pcurr_track, hipStream_t stream, void* send_buff, int* this_time, int num_gpus) {
    hipLaunchKernelGGL((RcclKernelSetSrcPtr), dim3(1, 1, 1), dim3(1,1,1), 0, stream, pcurr_track, send_buff);
    int barrier_value = *this_time;
    // wait until root gpu sets its source pointer
    hipLaunchKernelGGL((RcclKernelBarrierWait), dim3(1,1,1), dim3(1,1,1), 0, stream, pcurr_track, barrier_value++, num_gpus);
    // wait until everyone is finish reading
    hipLaunchKernelGGL((RcclKernelBarrierWait), dim3(1,1,1), dim3(1,1,1), 0, stream, pcurr_track, barrier_value++, num_gpus);
    *this_time = barrier_value;
}

template<typename DataType_t>
void RcclInternalBroadcast(RingNode_t* pcurr_track, RingNode_t* proot_track, int count, hipStream_t stream, void* recv_buff, int* this_time, int num_gpus) {
    int num_workitems = 0, num_workgroups = 0;

    if(count > knum_workitems) { // knum_workitems = 1024
        num_workitems = knum_workitems;
        num_workgroups = count / knum_workitems + 1;
    } else {
        num_workitems = count;
        num_workgroups = 1;
    }

    int barrier_value = *this_time;
    hipLaunchKernelGGL((RcclKernelBarrierWait), dim3(1,1,1), dim3(1,1,1), 0, stream, pcurr_track, barrier_value++, num_gpus);
    hipLaunchKernelGGL((RcclKernelScalarCopyFromRoot<DataType_t>), dim3(num_workgroups, 1, 1), dim3(num_workitems, 1, 1), 0, stream, proot_track, recv_buff, count);
    hipLaunchKernelGGL((RcclKernelBarrierWait), dim3(1,1,1), dim3(1,1,1), 0, stream, pcurr_track, barrier_value++, num_gpus);
    *this_time = barrier_value;
}
