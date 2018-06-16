/*
Copyright (c) 2017-Present Advanced Micro Devices, Inc. 
All rights reserved.
*/

#pragma once

//
// This file declares kernels for reduce-scatter op
//

template<typename DataType_t, typename VectorType_t, rcclRedOp_t Op>
__global__ void RcclKernelReduceScatter(DeviceControl_t* pcurr_track, const void* send_buff, void* recv_buff, int rank, int count) {
    unsigned tx = threadIdx.x;
    unsigned bx = blockIdx.x;
    unsigned tid = tx + bx * knum_vectors_per_workgroup;

    // add send_buff, recv_buff to tracker
    if(tid == 0) {
        std::atomic_store_explicit(&(pcurr_track->src_buffer), (void*)send_buff, std::memory_order_seq_cst);
        std::atomic_store_explicit(&(pcurr_track->dst_buffer), recv_buff, std::memory_order_seq_cst);
    }
    __syncthreads();

    // get source buffer from offset of rank * count on current gpu
    VectorType_t* curr_buff = reinterpret_cast<VectorType_t*>(reinterpret_cast<DataType_t*>((void*)send_buff) + rank * count);
    // get destination buffer on current gpu
    VectorType_t* dest_buff = reinterpret_cast<VectorType_t*>(recv_buff);

    // get peer gpu tracker
    DeviceControl_t* pnext_track = pcurr_track->next_gpu;

    // 1. Get data from peer gpu
    // 2. Do op on data
    // 3. Store data to destination buffer
    if(pnext_track != pcurr_track) {

        // wait until peer gpu source buffer is set
        while(std::atomic_load_explicit(&(pnext_track->src_buffer), std::memory_order_seq_cst) == 0) {}
        __syncthreads();

        // load pper gpu source buffer at offset of rank * count
        VectorType_t* next_buff = reinterpret_cast<VectorType_t*>(reinterpret_cast<DataType_t*>(std::atomic_load_explicit(&(pnext_track->src_buffer), std::memory_order_seq_cst)) + rank * count);

        if(Op == rcclSum) {
            dest_buff[tid] = curr_buff[tid] + next_buff[tid];
        }
        if(Op == rcclProd) {
            dest_buff[tid] = curr_buff[tid] * next_buff[tid];
        }
        if(Op == rcclMax) {
            OpMax<DataType_t, VectorType_t>(dest_buff[tid], curr_buff[tid], next_buff[tid]);
        }
        if(Op == rcclMin) {
            OpMin<DataType_t, VectorType_t>(dest_buff[tid], curr_buff[tid], next_buff[tid]);
        }

        // move on to next gpu
        pnext_track = pnext_track->next_gpu;
    }

    // start traveling along the ring (clique) until current tracker is reached
    // 1. Get source buffer from peer gpu
    // 2. Get data from destination buffer
    // 3. Do op
    // 4. Store data to destination buffer 
    while(pnext_track != pcurr_track) {

        // wait until peer gpus source buffer is set
        while(std::atomic_load_explicit(&(pnext_track->src_buffer), std::memory_order_seq_cst) == 0) {}
        __syncthreads();

        // get source buffer from peer gpu at offset rank * count
        VectorType_t* next_buff = reinterpret_cast<VectorType_t*>(reinterpret_cast<DataType_t*>(std::atomic_load_explicit(&(pnext_track->src_buffer), std::memory_order_seq_cst)) + rank * count);

        if(Op == rcclSum) {
            dest_buff[tid] = dest_buff[tid] + next_buff[tid];
        }
        if(Op == rcclProd) {
            dest_buff[tid] = dest_buff[tid] * next_buff[tid];
        }
        if(Op == rcclMax) {
            OpMax<DataType_t, VectorType_t>(dest_buff[tid], dest_buff[tid], next_buff[tid]);
        }
        if(Op == rcclMin) {
            OpMin<DataType_t, VectorType_t>(dest_buff[tid], dest_buff[tid], next_buff[tid]);
        }

        // move on to next gpu
        pnext_track = pnext_track->next_gpu;
    }
    __syncthreads();
}
