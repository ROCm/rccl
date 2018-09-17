/*
Copyright (c) 2017 - Present Advanced Micro Devices, Inc.
All rights reserved.
*/

#pragma once

//
// This file declares kernels for broadcast op
//

template <typename DataType_t>
__global__ void RcclKernelScalarCopyFromRoot(RingNode_t* proot_track,
                                             void* recv_buff, int count) {
    int tx = threadIdx.x;
    int bx = blockIdx.x;
    int tid = tx + bx * knum_vectors_per_workgroup;

    if (tid < count) {
        reinterpret_cast<DataType_t*>(recv_buff)[tid] =
            reinterpret_cast<DataType_t*>(std::atomic_load_explicit(
                &(proot_track->src_buffer), std::memory_order_seq_cst))[tid];
    }
    __syncthreads();
}
