/*
Copyright (c) 2017 - Present Advanced Micro Devices, Inc.
All rights reserved.
*/

#pragma once

#include "rcclReduceKernels.h"

//
// The code here figures out the launch parameters for reduce op
//

extern int RCCL_TRACE_RT;

template<typename DataType_t, typename VectorType_t, rcclRedOp_t Op>
void RcclInternalReduce(DeviceControl_t *pcurr_track, int count, hipStream_t stream, const void* send_buff, void* recv_buff) {
    const int knum_elements_per_vector = sizeof(VectorType_t) / sizeof(DataType_t);
    const int knum_elements_per_workgroup = knum_elements_per_vector * knum_workitems;

    unsigned num_vector_workgroups = (count / knum_elements_per_workgroup);
    unsigned num_scalars = (count % knum_elements_per_workgroup);

    unsigned total_workgroups = num_vector_workgroups + (num_scalars > 0 ? 1 : 0);

    if((RCCL_TRACE_RT & krccl_print_kernel) == krccl_print_kernel) {
        int dev;
        hipGetDevice(&dev);
        fprintf(stderr, "%s<<rccl-kernel: RcclKernelReduce rccl-device:%d total_workgroups:%u knum_workitems:%u stream:%p pcurr_track:%p send_buff:%p recv_buff:%p num_vector_workgroups:%u num_scalars:%u%s\n", KBLU, dev, total_workgroups, knum_workitems, stream, pcurr_track, send_buff, recv_buff, num_vector_workgroups, num_scalars, KNRM);
    }

//    hipLaunchKernelGGL((RcclKernelReduce<DataType_t, VectorType_t, Op>), dim3(total_workgroups, 1, 1), dim3(knum_workitems, 1, 1), 0, stream, pcurr_track, send_buff, recv_buff, num_vector_workgroups, num_scalars);

    hipLaunchKernelGGL((RcclKernelReduce<DataType_t, VectorType_t, Op>), dim3(total_workgroups, 1, 1), dim3(knum_workitems, 1, 1), 0, stream, pcurr_track, send_buff, recv_buff, num_vector_workgroups);

    if((RCCL_TRACE_RT & krccl_print_kernel) == krccl_print_kernel) {
        fprintf(stderr, "%s<<rccl-kernel-launched: RcclKernelReduce %s\n", KBLU, KNRM);
    }

}
