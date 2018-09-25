/*
Copyright (c) 2017 - Present Advanced Micro Devices, Inc.
All rights reserved.
*/

/**
 * @file rcclScalarReduceRuntime.h
 * @brief Host code which launches kernels to do rcclReduce
 *
 * This file contains host code which launches kernels implementing rcclReduce
 *
 * @author Aditya Atluri
 */

#pragma once

#include "rcclBarrierKernels.h"
#include "rcclScalarReduceKernels.h"

extern int RCCL_TRACE_RT;

//! @brief Definition of RcclInternalReduce
//! This function is launched on root gpus
//! This function launches kernel on root gpu which gathers data from buffers on
//! all gpus, do reduction op and store it in root gpu destination buffer
template <typename DataType_t, typename VectorType_t, rcclRedOp_t Op>
void RcclInternalReduce(RingNode_t* pcurr_track, int count, hipStream_t stream,
                        const void* send_buff, void* recv_buff, int* this_time,
                        int num_gpus) {
    bool check_count = count > knum_workitems;

    int num_workitems = check_count ? knum_workitems : count;
    int num_workgroups = check_count ? count / knum_workitems + 1 : 1;

    //! Get how many times barrier is used
    int barrier_value = *this_time;

    //! Wait until non-root gpus set their source pointers
    hipLaunchKernelGGL(RcclKernelBarrierWait, dim3(1, 1, 1), dim3(1, 1, 1), 0,
                       stream, pcurr_track, barrier_value++, num_gpus);

    //! Once all the gpus set their source pointers do reduction on them and
    //! store the result to recv_buff
    hipLaunchKernelGGL((RcclKernelScalarReduce<DataType_t, Op>),
                       dim3(num_workgroups, 1, 1), dim3(num_workitems, 1, 1), 0,
                       stream, pcurr_track, send_buff, recv_buff, count);

    //! Make all gpus to wait until reduction is done. Once done, all gpus exit
    //! op
    hipLaunchKernelGGL(RcclKernelBarrierWait, dim3(1, 1, 1), dim3(1, 1, 1), 0,
                       stream, pcurr_track, barrier_value++, num_gpus);

    //! Store back how many times the barrier is used so far
    *this_time = barrier_value;
}

//! @brief Definition of RcclInternalReduceNotRoot
//! This function is launched on gpus which are not roots
void RcclInternalReduceNotRoot(RingNode_t* pcurr_track, hipStream_t stream,
                               const void* send_buff, int* this_time,
                               int num_gpus) {
    //! Get how many times barrier is used
    int barrier_value = *this_time;

    //! Set source pointer to RingNode_t so that other gpus and see them
    hipLaunchKernelGGL(RcclKernelSetSrcPtr, dim3(1, 1, 1), dim3(1, 1, 1), 0,
                       stream, pcurr_track, (void*)send_buff);

    //! Wait until all gpus have set their source pointers
    hipLaunchKernelGGL(RcclKernelBarrierWait, dim3(1, 1, 1), dim3(1, 1, 1), 0,
                       stream, pcurr_track, barrier_value++, num_gpus);

    //! Wait until all the gpus have finished the op
    hipLaunchKernelGGL(RcclKernelBarrierWait, dim3(1, 1, 1), dim3(1, 1, 1), 0,
                       stream, pcurr_track, barrier_value++, num_gpus);

    //! Store back how many times the barrier is used so far
    *this_time = barrier_value;
}
