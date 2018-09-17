/*
Copyright (c) 2017 - Present Advanced Micro Devices, Inc.
All rights reserved.
*/

#pragma once

//
// This file declares kernels for allreduce op
//

//
// This kernel works on a portion of buffer
//
template <typename DataType_t, rcclRedOp_t Op>
__global__ void RcclKernelScalarAllReduce(RingNode_t* pcurr_track,
                                          void* send_buff, void* recv_buff,
                                          int count, int offset) {
    int tx = threadIdx.x;
    int bx = blockIdx.x;
    int tid = tx + bx * knum_vectors_per_workgroup;

    DataType_t* curr_dst_buff = reinterpret_cast<DataType_t*>(recv_buff);
    DataType_t* curr_src_buff = reinterpret_cast<DataType_t*>((void*)send_buff);

    if (tid < count) {
        // get peer gpu tracker
        RingNode_t* pnext_track = pcurr_track->next_gpu;

        int index = tid + offset;

        if (pnext_track != pcurr_track) {
            DataType_t* next_src_buff =
                reinterpret_cast<DataType_t*>(std::atomic_load_explicit(
                    &(pnext_track->src_buffer), std::memory_order_seq_cst));

            if (Op == rcclSum)
                curr_dst_buff[index] =
                    curr_src_buff[index] + next_src_buff[index];
            if (Op == rcclProd)
                curr_dst_buff[index] =
                    curr_src_buff[index] * next_src_buff[index];
            if (Op == rcclMax)
                curr_dst_buff[index] =
                    curr_src_buff[index] > next_src_buff[index]
                        ? curr_src_buff[index]
                        : next_src_buff[index];
            if (Op == rcclMin)
                curr_dst_buff[index] =
                    curr_src_buff[index] < next_src_buff[index]
                        ? curr_src_buff[index]
                        : next_src_buff[index];

            pnext_track = pnext_track->next_gpu;
        }

        while (pnext_track != pcurr_track) {
            DataType_t* next_src_buff =
                reinterpret_cast<DataType_t*>(std::atomic_load_explicit(
                    &(pnext_track->src_buffer), std::memory_order_seq_cst));

            if (Op == rcclSum)
                curr_dst_buff[index] =
                    curr_dst_buff[index] + next_src_buff[index];
            if (Op == rcclProd)
                curr_dst_buff[index] =
                    curr_dst_buff[index] * next_src_buff[index];
            if (Op == rcclMax)
                curr_dst_buff[index] =
                    curr_dst_buff[index] > next_src_buff[index]
                        ? curr_dst_buff[index]
                        : next_src_buff[index];
            if (Op == rcclMin)
                curr_dst_buff[index] =
                    curr_dst_buff[index] < next_src_buff[index]
                        ? curr_dst_buff[index]
                        : next_src_buff[index];

            pnext_track = pnext_track->next_gpu;
        }
    }
    __syncthreads();
}

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
