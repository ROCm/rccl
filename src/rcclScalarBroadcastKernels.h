/*
Copyright (c) 2017 - Present Advanced Micro Devices, Inc.
All rights reserved.
*/

/**
 * @file rcclScalarBroadcastKernels.h
 * @brief Implementation of root copy kernel
 *
 * This file contains a kernel which reads data from root gpu
 *
 * @author Aditya Atluri
 */
#pragma once

//! @brief Definition of RcclKernelScalarCopyFromRoot
template <typename DataType_t>
__global__ void RcclKernelScalarCopyFromRoot(RingNode_t* proot_track,
                                             void* recv_buff, int count, int offset) {
    int tx = threadIdx.x;
    int bx = blockIdx.x;
    int tid = tx + bx * knum_vectors_per_workgroup;

    if (tid < count) {
        //! Copy data from root gpu source buffer to current gpu destination
        //! buffer
        int index = tid + offset;
        reinterpret_cast<DataType_t*>(recv_buff)[index] =
            reinterpret_cast<DataType_t*>(proot_track->src_buffer)[index];
    }
    __syncthreads();
}

//! @brief Definition of RcclKernelCopyRest
//! Gather data (which is not operated on by current gpu) from all gpus
template <typename DataType_t>
__global__ void RcclKernelBcastCopyRest(RingNode_t* pcurr_track, int num_gpus,
                                   int rank, int count_per_gpu,
                                   int max_count_per_gpu) {
    int tx = threadIdx.x;
    int bx = blockIdx.x;
    int tid = tx + bx * knum_workitems;

    RingNode_t* pnext_track = pcurr_track->next_gpu;

    //! Get pointer to current gpu destination buffer
    DataType_t* curr_dst_buff =
        reinterpret_cast<DataType_t*>(pcurr_track->dst_buffer);

    //! Iterate over all the gpus and gather data from them
    while (pnext_track->rank != rank) {
        //! Get pointer to peer gpu source buffer
        DataType_t* next_src_buff =
            reinterpret_cast<DataType_t*>(pnext_track->dst_buffer);

        int curr_rank = pnext_track->rank;

        int count = count_per_gpu;

        //! If the rank of peer gpu is last the last gpu, update the number of
        //! elements it operates on
        if (curr_rank == num_gpus - 1) {
            count = max_count_per_gpu;
        }

        //! Read data from peer gpu and store it to current gpu destination
        //! buffer
        if (tid < count) {
            curr_dst_buff[tid + curr_rank * count_per_gpu] =
                next_src_buff[tid + curr_rank * count_per_gpu];
        }

        //! Get next gpu tracker
        pnext_track = pnext_track->next_gpu;
    }
}
