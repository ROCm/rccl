/*
Copyright (c) 2017-Present Advanced Micro Devices, Inc. 
All rights reserved.
*/

#pragma once

//
// This file declares kernels for allgather
//

// 
// This kernel does data copy from all peer gpu buffers to current gpu buffer
//
template<typename DataType_t, typename VectorType_t>
__global__ void RcclKernelScalarAllGather(DeviceControl_t* pcurr_track, const void* send_buff, void* recv_buff, int rank, int count) {
    unsigned tx = threadIdx.x;
    unsigned bx = blockIdx.x;
    unsigned tid = tx + bx * knum_workitems;

    // add recv_buff to tracker on current gpu
    if(tid == 0) {
        std::atomic_store_explicit(&(pcurr_track->dst_buffer), (void*)recv_buff, std::memory_order_seq_cst);
    }
    __syncthreads();

    if(tid < count) {

        // typecast source buffer
        DataType_t* curr_buff = reinterpret_cast<DataType_t*>((void*)send_buff);
        // get destination buffer with offset at rank * count
        DataType_t* dest_buff = reinterpret_cast<DataType_t*>(recv_buff) + rank * count;

        dest_buff[tid] = curr_buff[tid];

        DeviceControl_t* pnext_track = pcurr_track->next_gpu;

        while(pnext_track != pcurr_track) {

            // wait until peer gpus destination buffer is set
            while(std::atomic_load_explicit(&(pnext_track->dst_buffer), std::memory_order_seq_cst) == 0) {}
            __syncthreads();

            DataType_t* next_buff = reinterpret_cast<DataType_t*>(std::atomic_load_explicit(&(pnext_track->dst_buffer), std::memory_order_seq_cst)) + rank * count;

            next_buff[tid] = curr_buff[tid];
            __syncthreads();

            pnext_track = pnext_track->next_gpu;

        }
    }
}
