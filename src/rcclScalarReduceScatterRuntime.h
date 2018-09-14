/*
Copyright (c) 2017-Present Advanced Micro Devices, Inc. 
All rights reserved.
*/

#pragma once

#include "rcclScalarReduceScatterKernels.h"

extern int RCCL_TRACE_RT;

//
// The code here figures out the launch parameters for reduce-scatter op
//

template<typename DataType_t, typename VectorType_t, rcclRedOp_t Op>
void RcclInternalReduceScatter(RingNode_t* pcurr_track, int rank, int count, hipStream_t stream, const void* send_buff, void* recv_buff) {

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
        fprintf(stderr, "%s<<<rccl-kernel: RcclKernelScalarReduceScatter rccl-device:%d num_workgroups:%d num_workitems:%d stream:%p pcurr_track:%p send_buff:%p recv_buff:%p rank:%d count:%d%s\n", KBLU, dev, num_workgroups, num_workitems, stream, pcurr_track, send_buff, recv_buff, rank, count, KNRM);
    }


    hipLaunchKernelGGL((RcclKernelScalarReduceScatter<DataType_t, VectorType_t, Op>), dim3(num_workgroups, 1, 1), dim3(num_workitems, 1, 1), 0, stream, pcurr_track, send_buff, recv_buff, rank, count);

    if((RCCL_TRACE_RT & krccl_print_kernel) == krccl_print_kernel) {
        fprintf(stderr, "%s<<<rccl-kernel-launched: RcclKernelScalarReduceScatter %s\n", KBLU, KNRM);
    }
}
