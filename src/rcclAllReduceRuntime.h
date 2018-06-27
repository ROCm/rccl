/*
Copyright (c) 2017 - Present Advanced Micro Devices, Inc.
All rights reserved.
*/

#pragma once

#include "rcclAllReduceKernels.h"

//
// The code here figures out the launch parameters for allreduce ops
//

template<typename DataType_t, typename VectorType_t, rcclRedOp_t Op>
void RcclInternalAllReduce(DeviceControl_t *pcurr_track, int count, hipStream_t stream, const void* send_buff, void* recv_buff) {
    // get number of elements present in a vector
    const int knum_elements_per_vector = sizeof(VectorType_t) / sizeof(DataType_t);
    // find number of elements operated in a workgroup
    const int knum_elements_per_workgroup = knum_elements_per_vector * knum_workitems;

    // number of workgroups which work on vector elements
    unsigned num_vector_workgroups = (count / knum_elements_per_workgroup);
    // number of scalars should be operated on
    unsigned num_scalars = (count % knum_elements_per_workgroup);

    // total workgroups needed to launch
    unsigned total_workgroups = num_vector_workgroups + (num_scalars > 0 ? 1 : 0);

    // Number of workitems = 1024, number of workgroups = total_workgroups
    hipLaunchKernelGGL((RcclKernelAllReduce<DataType_t, VectorType_t, Op>), dim3(total_workgroups, 1, 1), dim3(knum_workitems, 1, 1), 0, stream, pcurr_track, send_buff, recv_buff, num_vector_workgroups, num_scalars);
}
