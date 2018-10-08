/*
Copyright (c) 2017 - Present Advanced Micro Devices, Inc.
All rights reserved.
*/

#pragma once

/**
 * @file rcclScalarAllGatherKernels.h
 * @brief Kernels to implement allgather operation
 *
 * This file contains implementation of kernels used by rcclAllGather
 *
 */

//! @brief Definition of RcclKernelScalarAllGather
//! Gather data from all gpus and store to current gpu destination buffer
template <typename DataType_t>
__global__ void RcclKernelScalarAllGather(RingNode_t* pcurr_track, int rank,
                                          int count) {
    int tx = threadIdx.x;
    int bx = blockIdx.x;
    int tid = tx + bx * knum_vectors_per_workgroup;

    //! Get pointers to current gpu source and destination buffers
    DataType_t* curr_dst_buff =
        reinterpret_cast<DataType_t*>(pcurr_track->dst_buffer);
    const DataType_t* curr_src_buff =
        reinterpret_cast<const DataType_t*>(pcurr_track->src_buffer);
    RingNode_t* pnext_track = pcurr_track->next_gpu;

    //! Iterate over all the gpus and gather data from them
    while (pnext_track->rank != rank) {
        //! Get pointer to peer gpu source buffer
        DataType_t* next_src_buff =
            reinterpret_cast<DataType_t*>(pnext_track->src_buffer);

        int curr_rank = pnext_track->rank;

        //! Read data from peer gpu and store it to current gpu destination
        //! buffer
        if (tid < count) {
            curr_dst_buff[tid + curr_rank * count] = next_src_buff[tid];
        }

        //! Get next gpu tracker
        pnext_track = pnext_track->next_gpu;
    }

    // copy self
    if (tid < count) {
        curr_dst_buff[tid + rank * count] = curr_src_buff[tid];
    }

    __syncthreads();
}
