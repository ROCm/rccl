/*
Copyright (c) 2017-Present Advanced Micro Devices, Inc. 
All rights reserved.
*/

#pragma once

#include "rcclBroadcastKernels.h"

//
// The code here figures out the launch parameters for broadcast op
//

template<typename DataType_t, typename VectorType_t>
void RcclInternalBroadcast(DeviceControl_t* pcurr_track, int count, hipStream_t stream, void* send_buff) {
    const int knum_elements_per_vector = sizeof(VectorType_t) / sizeof(DataType_t);
    const int knum_elements_per_workgroup = knum_elements_per_vector * knum_workitems;

    unsigned num_vector_workgroups = (count / knum_elements_per_workgroup);
    unsigned num_scalars = (count % knum_elements_per_workgroup);

    unsigned total_workgroups = num_vector_workgroups + (num_scalars > 0 ? 1 : 0);

    if((RCCL_TRACE_RT & krccl_print_kernel) == krccl_print_kernel) {
        int dev;
        hipGetDevice(&dev);
        fprintf(stderr, "%s<<rccl-kernel: RcclKernelBroadcast rccl-device:%d total_workgroups:%u knum_workitems:%u stream:%p pcurr_track:%p send_buff:%p num_vector_workgroups:%u num_scalars:%u%s\n", KBLU, dev, total_workgroups, knum_workitems, stream, pcurr_track, send_buff, num_vector_workgroups, num_scalars, KNRM);
    }


    // Number of workitems = 1024, number of workgroups = total_workgroups
    hipLaunchKernelGGL((RcclKernelBroadcast<DataType_t, VectorType_t>), dim3(total_workgroups, 1, 1), dim3(knum_workitems, 1, 1), 0, stream, pcurr_track, send_buff, num_vector_workgroups, num_scalars);
}
