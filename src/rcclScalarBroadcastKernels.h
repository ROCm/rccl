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
__global__ void RcclKernelScalarBroadcast(DeviceControl_t* pcurr_track, void* send_buff, int count) {
    unsigned tx = threadIdx.x;
    unsigned bx = blockIdx.x;
    unsigned tid = tx + bx * knum_workitems;

    if(tid < count) {

        DeviceControl_t* pnext_track = pcurr_track->next_gpu;

        DataType_t* curr_buff = reinterpret_cast<DataType_t*>(send_buff);

        while(pnext_track != pcurr_track) {

            while(std::atomic_load_explicit(&(pnext_track->dst_buffer), std::memory_order_seq_cst) == 0) {}

            DataType_t* dest_buff = reinterpret_cast<DataType_t*>(std::atomic_load_explicit(&(pnext_track->dst_buffer), std::memory_order_seq_cst));

            dest_buff[tid] = curr_buff[tid];

            pnext_track = pnext_track->next_gpu;
        }
    }
}

__global__ void RcclKernelWait(DeviceControl_t* pcurr_track) {
    int tx = threadIdx.x;
    while(std::atomic_load_explicit(&(pcurr_track->wait_signal), std::memory_order_seq_cst) != 1) {}
    std::atomic_store_explicit(&(pcurr_track->wait_signal), int(0), std::memory_order_seq_cst);
}

__global__ void RcclKernelSet(DeviceControl_t* pcurr_track) {
    DeviceControl_t* pnext_track = pcurr_track->next_gpu;
    while(pnext_track != pcurr_track) {
        std::atomic_store_explicit(&(pnext_track->wait_signal), int(1), std::memory_order_seq_cst);
        pnext_track = pnext_track->next_gpu;
    }
}
