/*
Copyright (c) 2017 - Present Advanced Micro Devices, Inc.
All rights reserved.
*/

#pragma once

//
// This file declares kernels for broadcast op
//

// TODO: DataType_t may not be useful - investigate
template<typename DataType_t, typename VectorType_t>
__global__ void RcclKernelBroadcast(DeviceControl_t* pcurr_track, void* send_buff) {
    unsigned tx = threadIdx.x;
    unsigned bx = blockIdx.x;
    unsigned tid = tx + bx * knum_vectors_per_workgroup;

    // get next gpu
    DeviceControl_t* pnext_track = pcurr_track->next_gpu;

    // get source buffer
    VectorType_t* curr_buff = reinterpret_cast<VectorType_t*>(send_buff);

    // iterate over all the trackers in the clique
    while(pnext_track != pcurr_track) {

        // wait until destination buffer in one of the peers tracker is set
        // once done, get its pointer and store data from present buffer to it
        while(std::atomic_load_explicit(&(pnext_track->dst_buffer), std::memory_order_seq_cst) == 0) {}
        VectorType_t* dest_buff = reinterpret_cast<VectorType_t*>(std::atomic_load_explicit(&(pnext_track->dst_buffer), std::memory_order_seq_cst));

        dest_buff[tid] = curr_buff[tid];

        // step to next gpu
        pnext_track = pnext_track->next_gpu;
    }
}
