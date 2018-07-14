/*
Copyright (c) 2017 - Present Advanced Micro Devices, Inc.
All rights reserved.
*/

#pragma once

#include "rcclScalarAllReduceKernels.h"

//
// The code here figures out the launch parameters for allreduce ops
//

extern int RCCL_TRACE_RT;

template<typename DataType_t, typename VectorType_t, rcclRedOp_t Op>
void RcclInternalAllReduce(DeviceControl_t *pcurr_track, int count, hipStream_t stream, const void* send_buff, void* recv_buff) {

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
        fprintf(stderr, "%s<<<rccl-kernel: RcclKernelScalarAllReduce rccl-device:%d num_workgroups:%d num_workitems:%d stream:%p pcurr_track:%p send_buff:%p recv_buff:%p count:%d op:%u%s\n", KBLU, dev, num_workgroups, num_workitems, stream, pcurr_track, send_buff, recv_buff, count, int(Op), KNRM);
    }

    hipLaunchKernelGGL((RcclKernelScalarAllReduce<DataType_t, Op>), dim3(num_workgroups, 1, 1), dim3(num_workitems, 1, 1), 0, stream, pcurr_track, send_buff, recv_buff, count);


    if((RCCL_TRACE_RT & krccl_print_kernel) == krccl_print_kernel) {
        fprintf(stderr, "%s<<<rccl-kernel-launched: RcclKernelScalarAllReduce %s\n", KBLU, KNRM);
    }
}
