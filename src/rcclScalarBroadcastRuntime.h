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
    hipLaunchKernelGGL((RcclKernelSetSrcPtr), dim3(1, 1, 1), dim3(1, 1, 1), 0,
                       stream, pcurr_track, send_buff);

    //! Get the barrier instance used count
    int barrier_value = *this_time;

    //! Wait until root gpu sets its source pointer
    hipLaunchKernelGGL((RcclKernelBarrierWait), dim3(1, 1, 1), dim3(1, 1, 1), 0,
                       stream, pcurr_track, barrier_value++, num_gpus);

    //! Wait until everyone finished reading from source buffer
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
                           int* this_time, int num_gpus) {
    bool check_count = count > knum_workitems;
    int num_workitems = check_count ? knum_workitems : count;
    int num_workgroups = check_count ? count / knum_workitems + 1 : 1;

    //! Get barrier instance used count
    int barrier_value = *this_time;

    //! Wait until root gpu sets its source pointer
    hipLaunchKernelGGL((RcclKernelBarrierWait), dim3(1, 1, 1), dim3(1, 1, 1), 0,
                       stream, pcurr_track, barrier_value++, num_gpus);

    //! Read data from root gpu
    hipLaunchKernelGGL((RcclKernelScalarCopyFromRoot<DataType_t>),
                       dim3(num_workgroups, 1, 1), dim3(num_workitems, 1, 1), 0,
                       stream, proot_track, recv_buff, count);

    //! Wait until everyone finishes reading
    hipLaunchKernelGGL((RcclKernelBarrierWait), dim3(1, 1, 1), dim3(1, 1, 1), 0,
                       stream, pcurr_track, barrier_value++, num_gpus);

    //! Update how many times barrier is used
    *this_time = barrier_value;
}
