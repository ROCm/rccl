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
__global__ void RcclKernelScalarAllReduce(RingNode_t* pcurr_track, const void* send_buff, void* recv_buff, unsigned count) {
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
        RingNode_t* pnext_track = pcurr_track->next_gpu;

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
    }
}

//
// This kernel works on a portion of buffer
//
template<typename DataType_t, rcclRedOp_t Op>
__global__ void RcclKernelScalarAllReduce(RingNode_t* pcurr_track, void* send_buff, void* recv_buff, int count, int offset) {
    int tx = threadIdx.x;
    int bx = blockIdx.x;
    int tid = tx + bx * knum_vectors_per_workgroup;

    if(tid == 0) {
        std::atomic_store_explicit(&(pcurr_track->src_buffer), (void*)send_buff, std::memory_order_seq_cst);
        std::atomic_store_explicit(&(pcurr_track->dst_buffer), recv_buff, std::memory_order_seq_cst);
    }
    __syncthreads();

    DataType_t* curr_dst_buff = reinterpret_cast<DataType_t*>(recv_buff);
    DataType_t* curr_src_buff = reinterpret_cast<DataType_t*>((void*)send_buff);

    if(tid < count) {

        // get peer gpu tracker
        RingNode_t* pnext_track = pcurr_track->next_gpu;

        int index = tid + offset;

        if(pnext_track != pcurr_track) {

            DataType_t* next_src_buff = reinterpret_cast<DataType_t*>(std::atomic_load_explicit(&(pnext_track->src_buffer), std::memory_order_seq_cst));

            if(Op == rcclSum)   curr_dst_buff[index] = curr_src_buff[index] + next_src_buff[index];
            if(Op == rcclProd)  curr_dst_buff[index] = curr_src_buff[index] * next_src_buff[index];
            if(Op == rcclMax)   curr_dst_buff[index] = curr_src_buff[index] > next_src_buff[index] ? curr_src_buff[index] : next_src_buff[index];
            if(Op == rcclMin)   curr_dst_buff[index] = curr_src_buff[index] < next_src_buff[index] ? curr_src_buff[index] : next_src_buff[index];

            pnext_track = pnext_track->next_gpu;
        }

        while(pnext_track != pcurr_track) {

            DataType_t* next_src_buff = reinterpret_cast<DataType_t*>(std::atomic_load_explicit(&(pnext_track->src_buffer), std::memory_order_seq_cst));


            if(Op == rcclSum)   curr_dst_buff[index] = curr_dst_buff[index] + next_src_buff[index];
            if(Op == rcclProd)  curr_dst_buff[index] = curr_dst_buff[index] * next_src_buff[index];
            if(Op == rcclMax)   curr_dst_buff[index] = curr_dst_buff[index] > next_src_buff[index] ? curr_dst_buff[index] : next_src_buff[index];
            if(Op == rcclMin)   curr_dst_buff[index] = curr_dst_buff[index] < next_src_buff[index] ? curr_dst_buff[index] : next_src_buff[index];

            pnext_track = pnext_track->next_gpu;
        }
    }
    __syncthreads();
}

__global__ void RcclKernelSetWaitReset(RingNode_t* pcurr_track, int num_gpus) {
    int tx = threadIdx.x;

    // set wait signal to peer gpus
    RingNode_t* pnext_track = pcurr_track->next_gpu;
    while(pnext_track != pcurr_track) {
        pnext_track->wait_signal.fetch_add(1, std::memory_order_seq_cst);
        pnext_track = pnext_track->next_gpu;
    }

    // wait on wait signal for current gpu
    while(std::atomic_load_explicit(&(pcurr_track->wait_signal), std::memory_order_seq_cst) != num_gpus - 1) {}

    // reset all fields in tracker for current gpu
    std::atomic_store_explicit(&(pcurr_track->wait_signal), int(0), std::memory_order_seq_cst);
    std::atomic_store_explicit(&(pcurr_track->dst_buffer), (void*)(0), std::memory_order_seq_cst);
    std::atomic_store_explicit(&(pcurr_track->src_buffer), (void*)(0), std::memory_order_seq_cst);
}


//
// New code for inplace fix
//

__global__ void RcclKernelWaitSignal(RingNode_t*, int);
__global__ void RcclKernelSetWaitSignal(RingNode_t*, int);
__global__ void RcclKernelResetAll(RingNode_t*);
__global__ void RcclKernelWaitForAllSignals(RingNode_t*, int);

__global__ void RcclKernelSetSrcDstBuffer(RingNode_t* pcurr_track, void* src_buffer, void* dst_buffer) {
    std::atomic_store_explicit(&(pcurr_track->src_buffer), src_buffer, std::memory_order_seq_cst);
    std::atomic_store_explicit(&(pcurr_track->dst_buffer), dst_buffer, std::memory_order_seq_cst);
    std::atomic_store_explicit(&(pcurr_track->wait_signal), 0, std::memory_order_seq_cst);
}

template<typename DataType_t>
__global__ void RcclKernelCopyRest(RingNode_t* pcurr_track, int num_gpus, int rank, int count_per_gpu, int max_count_per_gpu) {
    int tx = threadIdx.x;
    int bx = blockIdx.x;
    int tid = tx + bx * knum_workitems;

    RingNode_t* pnext_track = pcurr_track->next_gpu;

    DataType_t* curr_dst_buff = reinterpret_cast<DataType_t*>(std::atomic_load_explicit(&(pcurr_track->dst_buffer), std::memory_order_seq_cst));

    while(pnext_track->rank != rank) {
        DataType_t* next_src_buff = reinterpret_cast<DataType_t*>(std::atomic_load_explicit(&(pnext_track->dst_buffer), std::memory_order_seq_cst));

        int curr_rank = pnext_track->rank;

        int count = count_per_gpu;

        if(curr_rank == num_gpus - 1) {
            count = max_count_per_gpu;
        }

        if(tid < count) {

            curr_dst_buff[tid + curr_rank * count_per_gpu] = next_src_buff[tid + curr_rank * count_per_gpu];

        }

        pnext_track = pnext_track->next_gpu;
    }
}

__global__ void RcclKernelSetAndWait(RingNode_t* pcurr_track, int wait_signal) {
    std::atomic_store_explicit(&(pcurr_track->wait_signal), wait_signal, std::memory_order_seq_cst);
    RingNode_t* pnext_track = pcurr_track->next_gpu;
    while(pnext_track != pcurr_track) {
        while(std::atomic_load_explicit(&(pnext_track->wait_signal), std::memory_order_seq_cst) != wait_signal) {}
        pnext_track = pnext_track->next_gpu;
    }
}


__global__ void RcclKernelBarrierWait(RingNode_t* pcurr_track, int this_time, int get_here) {
    int val = 1;
    while(std::atomic_load_explicit(pcurr_track->times_done, std::memory_order_seq_cst) != this_time) {}

    int old_val = pcurr_track->bar_in->fetch_add(1, std::memory_order_seq_cst);

    while(std::atomic_load_explicit(pcurr_track->bar_in, std::memory_order_seq_cst) != get_here) {}
    pcurr_track->bar_out->fetch_add(1, std::memory_order_seq_cst);

    if(old_val + val == get_here) {
        while(std::atomic_load_explicit(pcurr_track->bar_out, std::memory_order_seq_cst) != get_here) {}

        std::atomic_store_explicit(pcurr_track->bar_in, 0, std::memory_order_seq_cst);
        std::atomic_store_explicit(pcurr_track->bar_out, 0, std::memory_order_seq_cst);

        pcurr_track->times_done->fetch_add(1, std::memory_order_seq_cst);
    }
}

