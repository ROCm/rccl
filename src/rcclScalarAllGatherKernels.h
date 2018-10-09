/*
Copyright (c) 2017 - Present Advanced Micro Devices, Inc.
All rights reserved.
*/

#pragma once

/**
 * @file rcclScalarAllGatherKernels.h
 * @brief Kernels to implement allgather operation
 *
 * This file contains implementation of kernels used by rcclAllGather
 *
 */
template<class T>
__device__ bool is_aligned(const void * ptr) noexcept {
    auto iptr = reinterpret_cast<std::uintptr_t>(ptr);
    return !(iptr % alignof(T));
}

//! @brief Definition of RcclKernelScalarAllGather
//! Gather data from all gpus and store to current gpu destination buffer
template <typename DataType_t>
__global__ void RcclKernelScalarAllGather(RingNode_t* pcurr_track, int rank,
        int count) {
    int tid = threadIdx.x;
    RingNode_t* pnext_track = pcurr_track->next_gpu;
    bool use_64b_copy = 0;
    int packed_count = count * sizeof(DataType_t) / sizeof(unsigned long long);
    unsigned long long * curr_dst_buff_64b;
    const unsigned long long * curr_src_buff_64b;
    //! Get pointers to current gpu source and destination buffers
    DataType_t* curr_dst_buff =
        reinterpret_cast<DataType_t*>(pcurr_track->dst_buffer);
    const DataType_t* curr_src_buff =
        reinterpret_cast<const DataType_t*>(pcurr_track->src_buffer);

    //! we can use 64b copying if following conditions are met
    //! data size in byte is greate than and exact multiple of 64b
    //! both src and dst buffers are 64b aligned
    if(count*sizeof(DataType_t)%alignof(unsigned long long) == 0 &&
            count*sizeof(DataType_t) > alignof(unsigned long long) &&
            is_aligned<unsigned long long>(curr_dst_buff) &&
            is_aligned<unsigned long long>(curr_src_buff))
        use_64b_copy = 1;

    if (curr_dst_buff + rank * count != curr_src_buff ) {
        //! copy self first
        if(use_64b_copy) {
            //! Get pointers to current gpu source and destination buffers
            curr_dst_buff_64b =
                reinterpret_cast<unsigned long long *>(pcurr_track->dst_buffer);
            curr_src_buff_64b =
                reinterpret_cast<const unsigned long long *>(pcurr_track->src_buffer);
            for(int i = 0; i + tid < packed_count; i += knum_workitems) {
                curr_dst_buff_64b[tid + rank * packed_count + i] =
                    curr_src_buff_64b[tid + i];
            }
        }
        else {
            for (int i = 0; i + tid < count; i += knum_workitems) {
                curr_dst_buff[tid + rank * count + i] =
                    curr_src_buff[tid + i];
            }
        }
    }

    //! Iterate over all the gpus and gather data from them
    while (pnext_track->rank != rank) {
        //! Get pointer to peer gpu source buffer
        int curr_rank = pnext_track->rank;
        DataType_t* next_src_buff =
            reinterpret_cast<DataType_t*>(pnext_track->src_buffer);

        //! check new src buffer alignment
        if(!is_aligned<unsigned long long>(next_src_buff))
            use_64b_copy = 0;
        //! Read data from peer gpu and store it to current gpu destination
        //! buffer
        if(use_64b_copy) {
            unsigned long long * next_src_buff_64b =
                reinterpret_cast<unsigned long long *>(pnext_track->src_buffer);
            if(curr_rank == 0)
                curr_src_buff_64b = reinterpret_cast<unsigned long long *>(pnext_track->src_buffer);
            else {
                //! Read data from peer gpu and store it to current gpu destination buffer
                for(int i = 0; i + tid < packed_count; i += knum_workitems)
                    curr_dst_buff_64b [tid + curr_rank * packed_count + i] =
                        next_src_buff_64b[tid + i];
            }
        }
        else {
            if(curr_rank == 0)
                curr_src_buff = reinterpret_cast<DataType_t*>(pnext_track->src_buffer);
            else {
                //! Read data from peer gpu and store it to current gpu destination buffer
                for (int i = 0; i + tid < count; i += knum_workitems) {
                    curr_dst_buff[tid + curr_rank * count + i] =
                        next_src_buff[tid + i];
                }
            }
        }
        //! Get next gpu tracker
        pnext_track = pnext_track->next_gpu;
    }
    __syncthreads();

    //! copy rank0 except on rank0 itself
    if(use_64b_copy) {
        if (rank != 0 && curr_dst_buff_64b != curr_src_buff_64b) {
            for(int i = 0; i + tid < packed_count; i += knum_workitems)
                curr_dst_buff_64b[tid + i] = curr_src_buff_64b[tid + i];
        }
    }
    else {
        if (rank != 0 && curr_dst_buff != curr_src_buff) {
            for (int i = 0; i + tid < count; i += knum_workitems)
                curr_dst_buff[tid + i] = curr_src_buff[tid + i];
        }
    }
    __syncthreads();
}
