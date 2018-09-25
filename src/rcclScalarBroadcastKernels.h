/*
Copyright (c) 2017 - Present Advanced Micro Devices, Inc.
All rights reserved.
*/

/**
 * @file rcclScalarBcastKernels.h
 * @brief Implementation of root copy kernel
 *
 * This file contains a kernel which reads data from root gpu
 *
 * @author Aditya Atluri
 */
#pragma once

//! Definition of RcclKernelScalarCopyFromRoot
template <typename DataType_t>
__global__ void RcclKernelScalarCopyFromRoot(RingNode_t* proot_track,
                                             void* recv_buff, int count) {
    int tx = threadIdx.x;
    int bx = blockIdx.x;
    int tid = tx + bx * knum_vectors_per_workgroup;

    if (tid < count) {
        //! Copy data from root gpu source buffer to current gpu destination
        //! buffer
        reinterpret_cast<DataType_t*>(recv_buff)[tid] =
            reinterpret_cast<DataType_t*>(proot_track->src_buffer)[tid];
    }
    __syncthreads();
}
