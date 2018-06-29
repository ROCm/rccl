/*
Copyright (c) 2017 - Present Advanced Micro Devices, Inc.
All rights reserved.
*/

#pragma once

//
// This file declares kernels for reduction op
//

#include "rcclDataTypes.h"
#include "rcclReduceDeviceOps.h"

//
// This kernel does reduce on all source pointers from devices in clique and store in local buffer.
// Launched on only root gpu
//
template<typename DataType_t, typename VectorType_t, rcclRedOp_t Op>
__global__ void RcclKernelReduce(DeviceControl_t* pcurr_track, const void* send_buff, void* recv_buff, unsigned num_vector_workgroups, unsigned num_scalars) {
    unsigned tx = threadIdx.x;
    unsigned bx = blockIdx.x;
    unsigned tid = tx + bx * knum_vectors_per_workgroup;

    const int knum_elements_per_vector = sizeof(VectorType_t) / sizeof(DataType_t);
    const int knum_elements_per_workgroup = knum_elements_per_vector * knum_workitems;
    int total_elements = knum_elements_per_workgroup * num_vector_workgroups;

    if(bx < num_vector_workgroups) {

    // get source and destination buffers for current gpu
    VectorType_t* curr_buff = reinterpret_cast<VectorType_t*>((void*)send_buff);
    VectorType_t* dest_buff = reinterpret_cast<VectorType_t*>(recv_buff);

    // get next gpus tracker
    DeviceControl_t* pnext_track = pcurr_track->next_gpu;

    // get data from buffers in present gpu and immediate peer
    // do reduce op and store in destination buffer on present gpu
    if(pnext_track != pcurr_track) {

        // wait until the peer gpus source buffer is set
        while(std::atomic_load_explicit(&(pnext_track->src_buffer), std::memory_order_seq_cst) == 0) {}

        // get source buffer from peer gpu
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

    // start traveling along the ring (clique), get source buffer from gpu
    // and accumulate it to destination buffer
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

    // operate on scalars, algorithms used is same as VectorType_t
    else {

        DataType_t* curr_buff = reinterpret_cast<DataType_t*>((void*)send_buff) + total_elements;
        DataType_t* dest_buff = reinterpret_cast<DataType_t*>(recv_buff);

        DeviceControl_t* pnext_track = pcurr_track->next_gpu;

        if(pnext_track != pcurr_track) {
         // wait until the peer gpus source buffer is set
        while(std::atomic_load_explicit(&(pnext_track->src_buffer), std::memory_order_seq_cst) == 0) {}

        // get source buffer from peer gpu
        DataType_t* next_buff = reinterpret_cast<DataType_t*>(std::atomic_load_explicit(&(pnext_track->src_buffer), std::memory_order_seq_cst)) + total_elements;

        for(unsigned id = tx; id < num_scalars; id = id + knum_workitems) {

            if(Op == rcclSum) {
                dest_buff[id] = curr_buff[id] + next_buff[id];
            }
            if(Op == rcclProd) {
                dest_buff[id] = curr_buff[tid] * next_buff[id];
            }
            if(Op == rcclMax) {
                dest_buff[id] = curr_buff[id] > next_buff[id] ? curr_buff[id] : next_buff[id];
            }
            if(Op == rcclMin) {
                dest_buff[id] = curr_buff[id] < next_buff[id] ? curr_buff[id] : next_buff[id];
            }
        }
        // move on to next gpu
        pnext_track = pnext_track->next_gpu;
        }

        // start traveling along the ring (clique), get source buffer from gpu
        // and accumulate it to destination buffer
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
                    dest_buff[id] = dest_buff[id] > next_buff[id] ? dest_buff[id] : next_buff[id];
                }
            }

            // move on to next gpu
            pnext_track = pnext_track->next_gpu;
        }
        __syncthreads();
    }
}

// Disabled as its hard to do reduction on trail data
#if 0
template<typename DataType_t, typename VectorType_t, rcclRedOp_t Op>
__global__ void RcclKernelReduce(DeviceControl_t* pcurr_track, const void* send_buff, void* recv_buff, unsigned trail, unsigned num_blocks) {
    unsigned tx = threadIdx.x;
    unsigned bx = blockIdx.x;

    constexpr unsigned knum_iters = (knum_vectors_per_workgroup / knum_workitems);

    VectorType_t *curr_buff = reinterpret_cast<VectorType_t*>((void*)send_buff);
    VectorType_t *dest_buff = reinterpret_cast<VectorType_t*>(recv_buff);

    DeviceControl_t *pnext_track = pcurr_track->next_gpu;

    if(pnext_track != pcurr_track) {
        while(std::atomic_load_explicit(&(pnext_track->src_buffer), std::memory_order_seq_cst) == 0) {}
        const VectorType_t *next_buff = reinterpret_cast<VectorType_t*>(std::atomic_load_explicit(&(pnext_track->src_buffer), std::memory_order_seq_cst));
        for(unsigned i = 0; i < knum_iters; i++) {
            unsigned index = tx + i * knum_workitems + bx * knum_vectors_per_workgroup;
            dest_buff[index] = curr_buff[index] + next_buff[index];
        }
        pnext_track = pnext_track->next_gpu;
    }

    while(pnext_track != pcurr_track) {
        while(std::atomic_load_explicit(&(pnext_track->src_buffer), std::memory_order_seq_cst) == 0) {}
        const VectorType_t *next_buff = reinterpret_cast<VectorType_t*>(std::atomic_load_explicit(&(pnext_track->src_buffer), std::memory_order_seq_cst));
        for(unsigned i = 0; i < knum_iters; i++) {
            unsigned index = tx + i * knum_workitems + bx * knum_vectors_per_workgroup;
            if(Op == rcclSum) dest_buff[index] = dest_buff[index] + next_buff[index];
        }
        pnext_track = pnext_track->next_gpu;
    }
/*
    if(trail != 0) {
        if(bx == num_blocks) {
        unsigned offset = (num_blocks) * knum_vectors_per_workgroup;
        DataType_t *scurr_buff = reinterpret_cast<DataType_t*>(reinterpret_cast<VectorType_t*>(((void*)send_buff)) + offset);
        DataType_t *sdest_buff = reinterpret_cast<DataType_t*>(reinterpret_cast<VectorType_t*>(recv_buff) + offset);

        pnext_track = pcurr_track->next_gpu;

    if(pnext_track != pcurr_track) {
        DataType_t *snext_buff = reinterpret_cast<DataType_t*>(reinterpret_cast<VectorType_t*>(std::atomic_load_explicit(&(pnext_track->src_buffer), std::memory_order_seq_cst)) + offset);
        for(unsigned i = tx; i < trail; i += knum_workitems) {
            sdest_buff[i] = scurr_buff[i] + snext_buff[i];
        }
        pnext_track = pnext_track->next_gpu;
    }

    while(pnext_track != pcurr_track) {
        DataType_t *snext_buff = reinterpret_cast<DataType_t*>(reinterpret_cast<VectorType_t*>(std::atomic_load_explicit(&(pnext_track->src_buffer), std::memory_order_seq_cst)) + offset);
        for(unsigned i = tx; i < trail; i += knum_workitems) {
            if(Op == rcclSum) sdest_buff[i] = sdest_buff[i] + snext_buff[i];
        }
        pnext_track = pnext_track->next_gpu;
    }


        }
    }
*/
//
// Optimized to store intermediate values in registers
//
/*
    VectorType_t r_val[knum_iters]; // put current buffer in to registers and accumulate

    for(unsigned i = 0; i < knum_iters; i++) {
        unsigned index = tx + i * knum_workitems + bx * knum_vectors_per_workgroup;
        r_val[i] = curr_buff[index];
    }

    while(pnext_track != pcurr_track) {
        while(std::atomic_load_explicit(&(pnext_track->src_buffer), std::memory_order_seq_cst) == 0) {}
        const VectorType_t *next_buff = reinterpret_cast<VectorType_t*>(std::atomic_load_explicit(&(pnext_track->src_buffer), std::memory_order_seq_cst));
        for(unsigned i = 0; i < knum_iters; i++) {
            unsigned index = tx + i * knum_workitems + bx * knum_vectors_per_workgroup;
            if(Op == rcclSum) r_val[i] = r_val[i] + next_buff[index];
        }
        pnext_track = pnext_track->next_gpu;
    }

    for(unsigned i = 0; i < knum_iters; i++) {
        unsigned index = tx + i * knum_workitems + bx * knum_vectors_per_workgroup;
        curr_buff[index] = r_val[i];
    }
*/
}
#endif
