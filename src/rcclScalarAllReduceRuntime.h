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
void RcclInternalAllReduce(DeviceControl_t *pcurr_track, const void* send_buff, void* recv_buff, hipStream_t stream, int count, int num_gpus, int rank) {

    int num_workitems = 0, num_workgroups = 0;

    int offset = (count / num_gpus) * rank;
    int count_per_gpu = (rank == num_gpus - 1) ? ((count / num_gpus) + (count % num_gpus)) : (count / num_gpus);

    if(count_per_gpu > knum_workitems) { // knum_workitems = 1024
        num_workitems = knum_workitems; 
        num_workgroups = count_per_gpu / knum_workitems + 1;
    } else {
        num_workitems = count_per_gpu;
        num_workgroups = 1;
    }

    hipEvent_t event;
    hipEventCreateWithFlags(&event, hipEventReleaseToSystem);

    if((RCCL_TRACE_RT & krccl_print_kernel) == krccl_print_kernel) {
        int dev;
        hipGetDevice(&dev);
        fprintf(stderr, "%s<<<rccl-kernel: RcclKernelScalarAllReduce rccl-device:%d num_workgroups:%d num_workitems:%d stream:%p pcurr_track:%p send_buff:%p recv_buff:%p count:%d offset:%d op:%u%s\n", KBLU, dev, num_workgroups, num_workitems, stream, pcurr_track, send_buff, recv_buff, count_per_gpu, offset, int(Op), KNRM);
    }

    hipLaunchKernelGGL((RcclKernelScalarAllReduce<DataType_t, Op>), dim3(num_workgroups, 1, 1), dim3(num_workitems, 1, 1), 0, stream, pcurr_track, (void*)send_buff, recv_buff, count_per_gpu, offset);

    if((RCCL_TRACE_RT & krccl_print_kernel) == krccl_print_kernel) {
        fprintf(stderr, "%s<<<rccl-kernel-launched: RcclKernelScalarAllReduce %s\n", KBLU, KNRM);
    }

    hipEventRecord(event, stream);
    // do something to wait on all rccl calls

    if((RCCL_TRACE_RT & krccl_print_kernel) == krccl_print_kernel) {
        int dev;
        hipGetDevice(&dev);
        fprintf(stderr, "%s<<<rccl-kernel: RcclKernelSetWaitReset rccl-device:%d num_workgroups:%d num_workitems:%d stream:%p pcurr_track:%p num_gpus:%d%s\n", KBLU, dev, 1, 1, stream, pcurr_track, num_gpus, KNRM);
    }

    hipLaunchKernelGGL((RcclKernelSetWaitReset), dim3(1,1,1), dim3(1,1,1), 0, stream, pcurr_track, num_gpus);

    if((RCCL_TRACE_RT & krccl_print_kernel) == krccl_print_kernel) {
        fprintf(stderr, "%s<<<rccl-kernel-launched: RcclKernelSetWaitReset %s\n", KBLU, KNRM);
    }

    hipEventRecord(event, stream);
}
