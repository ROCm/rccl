/*
Copyright (c) 2017 - Present Advanced Micro Devices, Inc.
All rights reserved.
*/

#pragma once

template <typename DataType_t, rcclRedOp_t Op>
__global__ void RcclKernelScalarReduce(RingNode_t* pcurr_track, void* send_buff,
                                       void* recv_buff, int count) {
    int tx = threadIdx.x;
    int bx = blockIdx.x;
    int tid = tx + bx * knum_vectors_per_workgroup;

    DataType_t* curr_dst_buff = reinterpret_cast<DataType_t*>(recv_buff);
    DataType_t* curr_src_buff = reinterpret_cast<DataType_t*>(send_buff);

    if (tid < count) {
        int index = tid;

        RingNode_t* pnext_track = pcurr_track->next_gpu;

        if (pnext_track != pcurr_track) {
            DataType_t* next_src_buff =
                reinterpret_cast<DataType_t*>(pnext_track->src_buffer);

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
                reinterpret_cast<DataType_t*>(pnext_track->src_buffer);

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
