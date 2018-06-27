/*
Copyright (c) 2017-Present Advanced Micro Devices, Inc.
All rights reserved.
*/

#pragma once

#include "rcclAllGatherKernels.h"

//
// The code here figures out the launch parameters for AllGather
//

template<typename DataType_t, typename VectorType_t>
void RcclInternalAllGather(DeviceControl_t *pcurr_track, int count, int rank, hipStream_t stream, const void *send_buff, void *recv_buff) {
    const int knum_elements_per_vector = sizeof(VectorType_t) / sizeof(DataType_t);
    const int knum_elements_per_workgroup = knum_elements_per_vector * knum_workitems;

    unsigned num_vector_workgroups = (count / knum_elements_per_workgroup);
    unsigned num_scalars = (count % knum_elements_per_workgroup);

    unsigned total_workgroups = num_vector_workgroups + (num_scalars > 0 ? 1 : 0);

    hipLaunchKernelGGL((RcclKernelAllGather<DataType_t, VectorType_t>), dim3(total_workgroups, 1, 1), dim3(knum_workitems, 1, 1), 0, stream, pcurr_track, send_buff, recv_buff, rank, count, num_vector_workgroups, num_scalars);
}
