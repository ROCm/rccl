/*
Copyright (c) 2017-Present Advanced Micro Devices, Inc. 
All rights reserved.
*/

#pragma once

#include "rcclReduceScatterKernels.h"

//
// The code here figures out the launch parameters for reduce-scatter op
//

template<typename DataType_t, typename VectorType_t, rcclRedOp_t Op>
void RcclInternalReduceScatter(DeviceControl_t* pcurr_track, int rank, int count, hipStream_t stream, const void* send_buff, void* recv_buff) {
    const int knum_elements_per_vector = sizeof(VectorType_t) / sizeof(DataType_t);

    unsigned num_vectors = (count / knum_elements_per_vector);
    unsigned num_scalars = (count % knum_elements_per_vector);
    unsigned num_chunks = num_vectors / knum_workitems;
    unsigned num_elements_in_last_chunk = num_vectors % knum_workitems;
    unsigned total_workgroups = num_chunks + (num_elements_in_last_chunk > 0 ? 1 : 0);

    hipLaunchKernelGGL((RcclKernelReduceScatter<DataType_t, VectorType_t, Op>), dim3(total_workgroups, 1, 1), dim3(knum_workitems, 1, 1), 0, stream, pcurr_track, send_buff, recv_buff, rank, count);    
}
