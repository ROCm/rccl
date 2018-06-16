/*
Copyright (c) 2017-Present Advanced Micro Devices, Inc. 
All rights reserved.
*/

#pragma once

//
// DeviceControl_t::src_buffer is set to buffer pointers for a gpu
//
__global__ void RcclKernelSetSrcPtr(DeviceControl_t* pcurr_track, const void* send_buff) {
    unsigned tx = threadIdx.x;
    if(tx == 0) {
        std::atomic_store_explicit(&(pcurr_track->src_buffer), (void*)send_buff, std::memory_order_seq_cst);
    }
}

//
// DeviceControl_t::dst_buffer is set to buffer pointer for a gpu
//
__global__ void RcclKernelSetDstPtr(DeviceControl_t* pcurr_track, void* recv_buff) {
    unsigned tx = threadIdx.x;
    if(tx == 0) {
        std::atomic_store_explicit(&(pcurr_track->dst_buffer), recv_buff, std::memory_order_seq_cst);
    }
}
