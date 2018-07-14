/*
Copyright (c) 2017 - Present Advanced Micro Devices, Inc.
All rights reserved.
*/

#pragma once

//
// This file declares kernels for allreduce op
//

//
// This kernel is launched on all gpus.
// Data is read from all peer gpus and current gpu, do reduction ops
// and store it in current gpu buffer
//
template<typename DataType_t, rcclRedOp_t Op>
__global__ void RcclKernelScalarAllReduce(DeviceControl_t* pcurr_track, const void* send_buff, void* recv_buff, unsigned count) {
    unsigned tx = threadIdx.x;
    unsigned bx = blockIdx.x;
    unsigned tid = tx + bx * knum_vectors_per_workgroup;


    // add send_buff, recv_buff to tracker on current gpu
    if(tid == 0) {
        std::atomic_store_explicit(&(pcurr_track->src_buffer), (void*)send_buff, std::memory_order_seq_cst);
        std::atomic_store_explicit(&(pcurr_track->dst_buffer), recv_buff, std::memory_order_seq_cst);
    }
    __syncthreads();


    if(tid < count) {

        // destination buffer for op
        DataType_t* dest_buff = reinterpret_cast<DataType_t*>(recv_buff);
        // one operand for op
        DataType_t* curr_buff = reinterpret_cast<DataType_t*>((void*)send_buff);

        // get peer gpu tracker
        DeviceControl_t* pnext_track = pcurr_track->next_gpu;

        //
        if(pnext_track != pcurr_track) {

            // wait until peer gpus source buffer is set
            while(std::atomic_load_explicit(&(pnext_track->src_buffer), std::memory_order_seq_cst) == 0) {}

            // get peers source buffer
            DataType_t* next_buff = reinterpret_cast<DataType_t*>(std::atomic_load_explicit(&(pnext_track->src_buffer), std::memory_order_seq_cst));

            if(Op == rcclSum)   dest_buff[tid] = curr_buff[tid] + next_buff[tid];
            if(Op == rcclProd)  dest_buff[tid] = curr_buff[tid] * next_buff[tid];
            if(Op == rcclMax)   dest_buff[tid] = curr_buff[tid] > next_buff[tid] ? curr_buff[tid] : next_buff[tid];
            if(Op == rcclMin)   dest_buff[tid] = curr_buff[tid] < next_buff[tid] ? curr_buff[tid] : next_buff[tid];

            // move on to the next gpu
            pnext_track = pnext_track->next_gpu;
        }


        while(pnext_track != pcurr_track) {

            while(std::atomic_load_explicit(&(pnext_track->src_buffer), std::memory_order_seq_cst) == 0) {}

            DataType_t* next_buff = reinterpret_cast<DataType_t*>(std::atomic_load_explicit(&(pnext_track->src_buffer), std::memory_order_seq_cst));

            if(Op == rcclSum)   dest_buff[tid] = dest_buff[tid] + next_buff[tid];
            if(Op == rcclProd)  dest_buff[tid] = dest_buff[tid] * next_buff[tid];
            if(Op == rcclMax)   dest_buff[tid] = dest_buff[tid] > next_buff[tid] ? dest_buff[tid] : next_buff[tid];
            if(Op == rcclMin)   dest_buff[tid] = dest_buff[tid] < next_buff[tid] ? dest_buff[tid] : next_buff[tid];

            pnext_track = pnext_track->next_gpu;
        }
        __syncthreads();
    }
}
