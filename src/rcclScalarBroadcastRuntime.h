/*
Copyright (c) 2017-Present Advanced Micro Devices, Inc.
All rights reserved.
*/

/**
 * @file rcclScalarBroadcastRuntime.h
 * @brief Implemenation of rcclBcast internally
 *
 * This file contains internal implementation of rcclBcast
 *
 * @author Aditya Atluri
 */

#pragma once

#include "rcclBarrierKernels.h"
#include "rcclScalarBroadcastKernels.h"

//! @brief Definition of RcclInternalBroadcastRoot
//! This function is called on root gpu and it does not do the copy
void RcclInternalBroadcastRoot(RingNode_t* pcurr_track, hipStream_t stream,
                               void* send_buff, int* this_time, int num_gpus) {
    //! Set source pointer on root gpu
    hipLaunchKernelGGL(RcclKernelSetSrcDstPtr, dim3(1, 1, 1), dim3(1, 1, 1), 0,
                       stream, pcurr_track, (void*)send_buff, send_buff);

    //! Get the barrier instance used count
    int barrier_value = *this_time;

    //! Wait until root gpu sets its source pointer
    hipLaunchKernelGGL((RcclKernelBarrierWait), dim3(1, 1, 1), dim3(1, 1, 1), 0,
                       stream, pcurr_track, barrier_value++, num_gpus);

    //! Wait until everyone finished reading chunked data from source buffer
    hipLaunchKernelGGL((RcclKernelBarrierWait), dim3(1, 1, 1), dim3(1, 1, 1), 0,
                       stream, pcurr_track, barrier_value++, num_gpus);

    //! Wait until everyone finishes reading rest chunked data
    hipLaunchKernelGGL((RcclKernelBarrierWait), dim3(1, 1, 1), dim3(1, 1, 1), 0,
                       stream, pcurr_track, barrier_value++, num_gpus);

    //! Update how many times barrier is used
    *this_time = barrier_value;
}

//! @brief Definition of RcclInternalBroadcast
//! This function is called on all gpus except root gpu
template <typename DataType_t>
void RcclInternalBroadcast(RingNode_t* pcurr_track, RingNode_t* proot_track,
                           int count, hipStream_t stream, void* recv_buff,
                           int* this_time, int num_gpus, hipEvent_t event) {
    int num_workitems = 0, num_workgroups = 0;
    int rank = pcurr_track->rank;
    int offset = (count / num_gpus) * rank;

    //! Three counts are required to implement chunked broadcast
    //! - op_gpu_count stores how many elements each gpu operates on,
    //! depending on rank of gpu.
    //! - regular_gpu_count stores how many elements each gpu holds,
    //! except for the highest ranking gpu
    //! - last_gpu_count stores how many elements last ranked gpu holds
    int regular_gpu_count = count / num_gpus;
    int last_gpu_count = ((count / num_gpus) + (count % num_gpus));
    int op_gpu_count =
        (rank == num_gpus - 1) ? last_gpu_count : regular_gpu_count;

    //! Explain why you need last_gpu_count number of workitems
    if (last_gpu_count < knum_workitems) {
        num_workitems = last_gpu_count;
        num_workgroups = 1;
    } else {
        num_workitems = knum_workitems;
        num_workgroups = (last_gpu_count / knum_workitems) + 1;
    }

    //! Set source and destination buffers for current gpu
    hipLaunchKernelGGL(RcclKernelSetSrcDstPtr, dim3(1, 1, 1), dim3(1, 1, 1), 0,
                       stream, pcurr_track, (void*)recv_buff, recv_buff);

    //! Get barrier instance used count
    int barrier_value = *this_time;

    //! Wait until root gpu sets its source pointer
    hipLaunchKernelGGL((RcclKernelBarrierWait), dim3(1, 1, 1), dim3(1, 1, 1), 0,
                       stream, pcurr_track, barrier_value++, num_gpus);

    //! Read chunked data from root gpu
    hipLaunchKernelGGL((RcclKernelScalarCopyFromRoot<DataType_t>),
                       dim3(num_workgroups, 1, 1), dim3(num_workitems, 1, 1), 0,
                       stream, proot_track, recv_buff, op_gpu_count, offset);

    //! Flush gpu l2 cache
    hipEventRecord(event, stream);

    //! Wait until all gpus have finished reading chunked data
    hipLaunchKernelGGL((RcclKernelBarrierWait), dim3(1, 1, 1), dim3(1, 1, 1), 0,
                       stream, pcurr_track, barrier_value++, num_gpus);

    //! Once all gpus have done copy, gather result from all gpus to
    //! current gpu destination buffer
    hipLaunchKernelGGL((RcclKernelBcastCopyRest<DataType_t>),
                       dim3(num_workgroups, 1, 1), dim3(num_workitems, 1, 1), 0,
                       stream, pcurr_track, num_gpus, rank, regular_gpu_count,
                       last_gpu_count);

    //! Flush gpu l2 cache
    hipEventRecord(event, stream);


    //! Wait until everyone finishes reading rest chunked data
    hipLaunchKernelGGL((RcclKernelBarrierWait), dim3(1, 1, 1), dim3(1, 1, 1), 0,
                       stream, pcurr_track, barrier_value++, num_gpus);

    //! Update how many times barrier is used
    *this_time = barrier_value;
}
