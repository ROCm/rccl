/*
Copyright (c) 2017 - Present Advanced Micro Devices, Inc.
All rights reserved.
*/

#pragma once

/**
 * @file rcclScalarAllReduceKernels.h
 * @brief Kernels to implement allreduce operation
 *
 * This file contains implementation of kernels used by rcclAllReduce
 *
 * @author Aditya Atluri
 */

//! Definition of RcclKernelScalarAllReduce
template <typename DataType_t, rcclRedOp_t Op>
__global__ void RcclKernelScalarAllReduce(RingNode_t* pcurr_track,
                                          void* send_buff, void* recv_buff,
                                          int count, int offset) {
    int tx = threadIdx.x;
    int bx = blockIdx.x;
    int tid = tx + bx * knum_vectors_per_workgroup;

    // get pointers to current gpu source and destination buffers
    DataType_t* curr_dst_buff = reinterpret_cast<DataType_t*>(recv_buff);
    DataType_t* curr_src_buff = reinterpret_cast<DataType_t*>((void*)send_buff);

    // use only count number of workitems to do the reduction operation
    if (tid < count) {
        // get peer gpu tracker
        RingNode_t* pnext_track = pcurr_track->next_gpu;

        // find absolute index the gpu operates on
        int index = tid + offset;

        DataType_t result = curr_src_buff[index];

        while (pnext_track != pcurr_track) {
            DataType_t* next_src_buff =
                reinterpret_cast<DataType_t*>(std::atomic_load_explicit(
                    &(pnext_track->src_buffer), std::memory_order_seq_cst));

            if (Op == rcclSum) result = result + next_src_buff[index];
            if (Op == rcclProd) result = result * next_src_buff[index];
            if (Op == rcclMax)
                result = result > next_src_buff[index] ? result
                                                       : next_src_buff[index];
            if (Op == rcclMin)
                result = result < next_src_buff[index] ? result
                                                       : next_src_buff[index];

            pnext_track = pnext_track->next_gpu;
        }

        curr_dst_buff[index] = result;
    }

    __syncthreads();
}

//
// Gather data from all the other gpus
//
template <typename DataType_t>
__global__ void RcclKernelCopyRest(RingNode_t* pcurr_track, int num_gpus,
                                   int rank, int count_per_gpu,
                                   int max_count_per_gpu) {
    int tx = threadIdx.x;
    int bx = blockIdx.x;
    int tid = tx + bx * knum_workitems;

    RingNode_t* pnext_track = pcurr_track->next_gpu;

    DataType_t* curr_dst_buff =
        reinterpret_cast<DataType_t*>(std::atomic_load_explicit(
            &(pcurr_track->dst_buffer), std::memory_order_seq_cst));

    while (pnext_track->rank != rank) {
        DataType_t* next_src_buff =
            reinterpret_cast<DataType_t*>(std::atomic_load_explicit(
                &(pnext_track->dst_buffer), std::memory_order_seq_cst));

        int curr_rank = pnext_track->rank;

        int count = count_per_gpu;

        if (curr_rank == num_gpus - 1) {
            count = max_count_per_gpu;
        }

        if (tid < count) {
            curr_dst_buff[tid + curr_rank * count_per_gpu] =
                next_src_buff[tid + curr_rank * count_per_gpu];
        }

        pnext_track = pnext_track->next_gpu;
    }
}
