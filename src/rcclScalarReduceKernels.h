/*
Copyright (c) 2017 - Present Advanced Micro Devices, Inc.
All rights reserved.
*/

#pragma once

/**
 * @file rcclScalarReduceKernels.h
 * @brief Kernels to implement reduce operation
 *
 * This file contains implementation of kernels used by rcclReduce
 *
 * @author Aditya Atluri
 */

//! @brief Definition of RcclKernelScalarReduce
//! Gather data from non-root gpus and do reduction op on it
template <typename DataType_t, rcclRedOp_t Op>
__global__ void RcclKernelScalarReduce(RingNode_t* pcurr_track,
                                       const void* send_buff, void* recv_buff,
                                       int count) {
    int tx = threadIdx.x;
    int bx = blockIdx.x;
    int tid = tx + bx * knum_vectors_per_workgroup;

    //! Get pointers to current gpu source and destination buffers
    DataType_t* curr_dst_buff = reinterpret_cast<DataType_t*>(recv_buff);
    DataType_t* curr_src_buff = reinterpret_cast<DataType_t*>(send_buff);

    //! Use only count number of workitems to do the reduction operation
    if (tid < count) {
        int index = tid;

        RingNode_t* pnext_track = pcurr_track->next_gpu;

        DataType_t result = curr_src_buff[index];

        //! Iterate over all the gpus, gather data from them and do reduction
        //! operation on them
        while (pnext_track != pcurr_track) {
            DataType_t* next_src_buff =
                reinterpret_cast<DataType_t*>(pnext_track->src_buffer);

            if (Op == rcclSum) result = result + next_src_buff[index];
            if (Op == rcclProd) result = result + next_src_buff[index];
            if (Op == rcclMax)
                result = result > next_src_buff[index] ? result
                                                       : next_src_buff[index];
            if (Op == rcclMin)
                result = result < next_src_buff[index] ? result
                                                       : next_src_buff[index];

            //! Get next gpu tracker
            pnext_track = pnext_track->next_gpu;
        }

        curr_dst_buff[index] = result;
    }

    __syncthreads();
}
