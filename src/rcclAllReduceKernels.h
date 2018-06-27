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
template<typename DataType_t, typename VectorType_t, rcclRedOp_t Op>
__global__ void RcclKernelAllReduce(DeviceControl_t* pcurr_track, const void* send_buff, void* recv_buff, unsigned num_vector_workgroups, unsigned num_scalars) {
    unsigned tx = threadIdx.x;
    unsigned bx = blockIdx.x;
    unsigned tid = tx + bx * knum_vectors_per_workgroup;

    const int knum_elements_per_vector = sizeof(VectorType_t) / sizeof(DataType_t);
    const int knum_elements_per_workgroup = knum_elements_per_vector * knum_workitems;
    int total_elements = knum_elements_per_workgroup * num_vector_workgroups;

    // add send_buff, recv_buff to tracker on current gpu
    if(tid == 0) {
        std::atomic_store_explicit(&(pcurr_track->src_buffer), (void*)send_buff, std::memory_order_seq_cst);
        std::atomic_store_explicit(&(pcurr_track->dst_buffer), recv_buff, std::memory_order_seq_cst);
    }
    __syncthreads();


    if(bx < num_vector_workgroups) {

        // destination buffer for op
        VectorType_t* dest_buff = reinterpret_cast<VectorType_t*>(recv_buff);
        // one operand for op
        VectorType_t* curr_buff = reinterpret_cast<VectorType_t*>((void*)send_buff);

        // get peer gpu tracker
        DeviceControl_t* pnext_track = pcurr_track->next_gpu;


        // 1. Get data from current gpu
        // 2. Get data from peer gpu
        // 3. Do op on data
        // 4. Store data on current gpu
        if(pnext_track != pcurr_track) {

            // wait until peer gpus set their source buffers
            while(std::atomic_load_explicit(&(pnext_track->src_buffer), std::memory_order_seq_cst) == 0) {}

            // get peers source buffer
            VectorType_t* next_buff = reinterpret_cast<VectorType_t*>(std::atomic_load_explicit(&(pnext_track->src_buffer), std::memory_order_seq_cst));

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

        // start traveling along the ring (clique) until current track is reached
        // 1. Get destination buffer from current gpu (as it stores intermediate values)
        // 2. Get data from peer gpu
        // 3. Do op
        // 4. Store data to destination buffer
        while(pnext_track != pcurr_track) {

            // wait until the peer gpus source buffer is set
            while(std::atomic_load_explicit(&(pnext_track->src_buffer), std::memory_order_seq_cst) == 0) {}

            // get source buffer from peer gpu
            VectorType_t* next_buff = reinterpret_cast<VectorType_t*>(std::atomic_load_explicit(&(pnext_track->src_buffer), std::memory_order_seq_cst));

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

    // operate on scalars, algorithm used for VectorType_t is used
    else {
        // destination buffer for op
        DataType_t* dest_buff = reinterpret_cast<DataType_t*>(recv_buff) + total_elements;
        // one operand for op
        DataType_t* curr_buff = reinterpret_cast<DataType_t*>((void*)send_buff) + total_elements;

        // get peer gpu tracker
        DeviceControl_t* pnext_track = pcurr_track->next_gpu;

        if(pnext_track != pcurr_track) {

            // wait until peer gpus set their source buffers
            while(std::atomic_load_explicit(&(pnext_track->src_buffer), std::memory_order_seq_cst) == 0) {}

            // get peers source buffer
            DataType_t* next_buff = reinterpret_cast<DataType_t*>(std::atomic_load_explicit(&(pnext_track->src_buffer), std::memory_order_seq_cst)) + total_elements;

            for(unsigned id = tx; id < num_scalars; id = id + knum_workitems) {

                if(Op == rcclSum) {
                    dest_buff[id] = curr_buff[id] + next_buff[id];
                }
                if(Op == rcclProd) {
                    dest_buff[id] = curr_buff[id] * next_buff[id];
                }
                if(Op == rcclMax) {
                    dest_buff[id] = curr_buff[id] > next_buff[id] ? curr_buff[id] : next_buff[id];
                }
                if(Op == rcclMin) {
                    dest_buff[id] = curr_buff[id] < next_buff[id] ? curr_buff[id] : next_buff[id];
                }
            }

        }
        // move on to next gpu
        pnext_track = pnext_track->next_gpu;

        while(pnext_track != pcurr_track) {

            // wait until the peer gpus source buffer is set
            while(std::atomic_load_explicit(&(pnext_track->src_buffer), std::memory_order_seq_cst) == 0) {}

            // get source buffer from peer gpu
            DataType_t* next_buff = reinterpret_cast<DataType_t*>(std::atomic_load_explicit(&(pnext_track->src_buffer), std::memory_order_seq_cst)) + total_elements;

            for(unsigned id = tx; id < num_scalars; id = id + knum_workitems) {

                if(Op == rcclSum) {
                    dest_buff[id] = dest_buff[id] + next_buff[id];
                }
                if(Op == rcclProd) {
                    dest_buff[id] = dest_buff[id] * next_buff[id];
                }
                if(Op == rcclMax) {
                    dest_buff[id] = dest_buff[id] > next_buff[id] ? dest_buff[id] : next_buff[id];
                }
                if(Op == rcclMin) {
                    dest_buff[id] = dest_buff[id] < next_buff[id] ? dest_buff[id] : next_buff[id];
                }
            }
            // move on to next gpu
            pnext_track = pnext_track->next_gpu;
        }
        __syncthreads();
    }

}
