/*
Copyright (c) 2017-Present Advanced Micro Devices, Inc. 
All rights reserved.
*/

#pragma once

#include "rcclReduceDeviceOps.h"

//
// This file declares kernels for reduce-scatter op
//

//
// This kernel is launched on all gpus.
// Data is read from all peer gpus based on rank of gpu, do reduction ops
// and store it in current gpu buffer
//
template<typename DataType_t, typename VectorType_t, rcclRedOp_t Op>
__global__ void RcclKernelScalarReduceScatter(DeviceControl_t* pcurr_track, const void* send_buff, void* recv_buff, int rank, int count) {
    unsigned tx = threadIdx.x;
    unsigned bx = blockIdx.x;
    unsigned tid = tx + bx * knum_workitems;

    // add send_buff, recv_buff to tracker
    if(tid == 0) {
        std::atomic_store_explicit(&(pcurr_track->src_buffer), (void*)send_buff, std::memory_order_seq_cst);
        std::atomic_store_explicit(&(pcurr_track->dst_buffer), recv_buff, std::memory_order_seq_cst);
    }
    __syncthreads();

    if(tid < count) {

        DataType_t* curr_buff = reinterpret_cast<DataType_t*>((void*)send_buff) + rank * count;
        DataType_t* dest_buff = reinterpret_cast<DataType_t*>(recv_buff);

        DeviceControl_t* pnext_track = pcurr_track->next_gpu;

        if(pnext_track != pcurr_track) {

            while(std::atomic_load_explicit(&(pnext_track->src_buffer), std::memory_order_seq_cst) == 0) {}
            __syncthreads();

            DataType_t* next_buff = reinterpret_cast<DataType_t*>(std::atomic_load_explicit(&(pnext_track->src_buffer), std::memory_order_seq_cst)) + rank * count;

            if(Op == rcclSum)   dest_buff[tid] = curr_buff[tid] + next_buff[tid];
            if(Op == rcclProd)  dest_buff[tid] = curr_buff[tid] * next_buff[tid];
            if(Op == rcclMax)   dest_buff[tid] = curr_buff[tid] > next_buff[tid] ? curr_buff[tid] : next_buff[tid];
            if(Op == rcclMin)   dest_buff[tid] = curr_buff[tid] < next_buff[tid] ? curr_buff[tid] : next_buff[tid];

            pnext_track = pnext_track->next_gpu;
        }

        while(pnext_track != pcurr_track) {

            while(std::atomic_load_explicit(&(pnext_track->src_buffer), std::memory_order_seq_cst) == 0) {}
            __syncthreads();

            DataType_t* next_buff = reinterpret_cast<DataType_t*>(std::atomic_load_explicit(&(pnext_track->src_buffer), std::memory_order_seq_cst)) + rank * count;

            if(Op == rcclSum)   dest_buff[tid] = dest_buff[tid] + next_buff[tid];
            if(Op == rcclProd)  dest_buff[tid] = dest_buff[tid] * next_buff[tid];
            if(Op == rcclMax)   dest_buff[tid] = dest_buff[tid] > next_buff[tid] ? dest_buff[tid] : next_buff[tid];
            if(Op == rcclMin)   dest_buff[tid] = dest_buff[tid] < next_buff[tid] ? dest_buff[tid] : next_buff[tid];

            pnext_track = pnext_track->next_gpu;
        }

        __syncthreads();
    }
}
