/*
Copyright (c) 2017-Present Advanced Micro Devices, Inc. 
All rights reserved.
*/

#pragma once

#include "rcclScalarBroadcastKernels.h"

//
// The code here figures out the launch parameters for broadcast op
//

template<typename DataType_t, typename VectorType_t>
void RcclInternalBroadcast(DeviceControl_t* pcurr_track, int count, hipStream_t stream, void* send_buff) {

    int num_workitems = 0, num_workgroups = 0;

    if(count > knum_workitems) { // knum_workitems = 1024
        num_workitems = knum_workitems;
        num_workgroups = count / knum_workitems + 1;
    } else {
        num_workitems = count;
        num_workgroups = 1;
    }

    if((RCCL_TRACE_RT & krccl_print_kernel) == krccl_print_kernel) {
        int dev;
        hipGetDevice(&dev);
        fprintf(stderr, "%s<<<rccl-kernel: RcclKernelScalarBroadcast rccl-device:%d num_workgroups:%d num_workitems:%d stream:%p pcurr_track:%p send_buff:%p count:%d%s\n", KBLU, dev, num_workgroups, num_workitems, stream, pcurr_track, send_buff, count, KNRM);
    }

    hipLaunchKernelGGL((RcclKernelScalarBroadcast<DataType_t, VectorType_t>), dim3(num_workgroups, 1, 1), dim3(num_workitems, 1, 1), 0, stream, pcurr_track, send_buff, count);

    if((RCCL_TRACE_RT * krccl_print_kernel) == krccl_print_kernel) {
        fprintf(stderr, "%s<<<rccl-kernel: RcclKernelScalarBroadcast %s\n", KBLU, KNRM);
    }

    hipLaunchKernelGGL((RcclKernelSet), dim3(1,1,1), dim3(1,1,1), 0, stream, pcurr_track);
}


//
// The root does not do the copy
//
void RcclInternalBroadcastRoot(DeviceControl_t* pcurr_track, hipStream_t stream, void* send_buff) {

// Reset all the fields in pcurr_track before using it (all other gpus should set it to zeros too)
    hipLaunchKernelGGL((RcclKernelResetAll), dim3(1, 1, 1), dim3(1, 1, 1), 0, stream, pcurr_track);

// Set source buffer pointer on root gpu
    hipLaunchKernelGGL((RcclKernelSetSrcPtr), dim3(1,1,1), dim3(1,1,1), 0, stream, pcurr_track, send_buff);

// Set wait signal on current gpu
    hipLaunchKernelGGL((RcclKernelSetWaitSignal), dim3(1,1,1), dim3(1,1,1), 0, stream, pcurr_track, int(1));

// Wait for all wait_signals on other gpus
    hipLaunchKernelGGL((RcclKernelWaitForAllSignals), dim3(1,1,1), dim3(1,1,1), 0, stream, pcurr_track, int(1));
}

template<typename DataType_t>
void RcclInternalBroadcast(DeviceControl_t* pcurr_track, DeviceControl_t* proot_track, int count, hipStream_t stream, void* recv_buff) {

    int num_workitems = 0, num_workgroups = 0;

    if(count > knum_workitems) { // knum_workitems = 1024
        num_workitems = knum_workitems;
        num_workgroups = count / knum_workitems + 1;
    } else {
        num_workitems = count;
        num_workgroups = 1;
    }


    hipLaunchKernelGGL((RcclKernelResetAll), dim3(1, 1, 1), dim3(1, 1, 1), 0, stream, pcurr_track);

    hipLaunchKernelGGL((RcclKernelWaitSignal), dim3(1,1,1), dim3(1,1,1), 0, stream, proot_track, int(1));

    hipLaunchKernelGGL((RcclKernelScalarCopyFromRoot<DataType_t>), dim3(num_workgroups, 1, 1), dim3(num_workitems, 1, 1), 0, stream, proot_track, recv_buff, count);

    hipLaunchKernelGGL((RcclKernelSetWaitSignal), dim3(1,1,1), dim3(1,1,1), 0, stream, pcurr_track, int(1));
}
