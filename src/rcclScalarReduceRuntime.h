/*
Copyright (c) 2017 - Present Advanced Micro Devices, Inc.
All rights reserved.
*/

#pragma once

#include "rcclScalarReduceKernels.h"

//
// The code here figures out the launch parameters for reduce op
//

extern int RCCL_TRACE_RT;

template<typename DataType_t, typename VectorType_t, rcclRedOp_t Op>
void RcclInternalReduce(RingNode_t *pcurr_track, int count, hipStream_t stream, const void* send_buff, void* recv_buff) {
    if((RCCL_TRACE_RT & krccl_print_kernel) == krccl_print_kernel) {
        int dev;
        hipGetDevice(&dev);
        fprintf(stderr, "%s<<rccl-kernel: RcclKernelScalarReduce rccl-device:%d knum_workitems:%u stream:%p pcurr_track:%p send_buff:%p recv_buff:%p count:%u op:%u %s\n", KBLU, dev, knum_workitems, stream, pcurr_track, send_buff, recv_buff, count, unsigned(Op), KNRM);
    }

    int num_workitems = 0, num_workgroups = 0;
    if(count > 1024) {
        num_workitems = 1024;
        num_workgroups = count / 1024 + 1;
    } else {
        num_workitems = count;
        num_workgroups = 1;
    }

    hipLaunchKernelGGL((RcclKernelScalarReduce<DataType_t, Op>), dim3(num_workgroups, 1, 1), dim3(num_workitems, 1, 1), 0, stream, pcurr_track, send_buff, recv_buff, count);


    if((RCCL_TRACE_RT & krccl_print_kernel) == krccl_print_kernel) {
        fprintf(stderr, "%s<<rccl-kernel-launched: RcclKernelReduce %s\n", KBLU, KNRM);
    }
}
