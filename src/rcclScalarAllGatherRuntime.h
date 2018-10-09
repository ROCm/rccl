/*
Copyright (c) 2017 - Present Advanced Micro Devices, Inc.
All rights reserved.
*/

/**
 * @file rcclScalarAllGatherRuntime.h
 * @brief Host code which launches kernels to do rcclAllGather
 *
 * This file contains host code which launches kernels implementing
 * rcclAllGather
 *
 */

#pragma once

#include "rcclBarrierKernels.h"
#include "rcclScalarAllGatherKernels.h"

extern int RCCL_TRACE_RT;

//! @brief Definition of RcclInternalAllGather
//! Once all gpus have setup their buffers, each gpu gathers rest
//! of the data from other gpus.
template <typename DataType_t, typename VectorType_t>
void RcclInternalAllGather(RingNode_t* pcurr_track, const void* send_buff,
                           void* recv_buff, hipStream_t stream, int count,
                           int num_gpus, int rank, hipEvent_t event,
                           int* this_time) {
    int barrier_value = *this_time;

    //! Set source and destination buffers for current gpu
    hipLaunchKernelGGL(RcclKernelSetSrcDstPtr, dim3(1, 1, 1), dim3(1, 1, 1), 0,
                       stream, pcurr_track, (void*)send_buff, recv_buff);

    //! Wait using multi-gpu barrier until all the gpus set their source and
    //! destination buffers
    hipLaunchKernelGGL(RcclKernelBarrierWait, dim3(1, 1, 1), dim3(1, 1, 1), 0,
                       stream, pcurr_track, barrier_value++, num_gpus);

    //! Once all gpus have done buffer setup, gather result from all gpus to
    //! current gpu destination buffer
    hipLaunchKernelGGL((RcclKernelScalarAllGather<DataType_t>),
                       dim3(1, 1, 1), dim3(knum_workitems, 1, 1), 0,
                       stream, pcurr_track, rank, count);
    //! Flush gpu l2 cache
    hipEventRecord(event, stream);

    //! Update communicator with update barrier count
    *this_time = barrier_value;
}
