/*
Copyright (c) 2017 - Present Advanced Micro Devices, Inc.
All rights reserved.
*/

#pragma once

//
// This file declares kernels for broadcast op
//

//
// This kernel is launched on root gpu,
// data from current buffer is transferred to
// buffer in all gpus.
//
template<typename DataType_t, typename VectorType_t>
__global__ void RcclKernelBroadcast(DeviceControl_t* pcurr_track, void* send_buff, unsigned num_vector_workgroups, unsigned num_scalars) {
    unsigned tx = threadIdx.x;
    unsigned bx = blockIdx.x;
    unsigned tid = tx + bx * knum_vectors_per_workgroup;

    const int knum_elements_per_vector = sizeof(VectorType_t) / sizeof(DataType_t);
    const int knum_elements_per_workgroup = knum_elements_per_vector * knum_workitems;
    int total_elements = knum_elements_per_workgroup * num_vector_workgroups;

    if(bx < num_vector_workgroups) {
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
    // operate on scalars, algorithm used is same as VectorType_t
    else {
        DeviceControl_t* pnext_track = pcurr_track->next_gpu;

        DataType_t* curr_buff = reinterpret_cast<DataType_t*>(send_buff) + total_elements;

        while(pnext_track != pcurr_track) {

            while(std::atomic_load_explicit(&(pnext_track->dst_buffer), std::memory_order_seq_cst) == 0) {}
            DataType_t* dest_buff = reinterpret_cast<DataType_t*>(std::atomic_load_explicit(&(pnext_track->dst_buffer), std::memory_order_seq_cst)) + total_elements;

            for(unsigned id = tx; id < num_scalars; id = id + knum_workitems) {
                dest_buff[id] = curr_buff[id];
            }

            pnext_track = pnext_track->next_gpu;
        }
    }
}
