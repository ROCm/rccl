/*
Copyright (c) 2017 - Present Advanced Micro Devices, Inc.
All rights reserved.
*/

/**
 * @file rcclScalarReduceScatterRuntime.h
 * @brief Host code which launches kernels to do rcclReduceScatter
 *
 * This file contains host code which launches kernels implementing
 * rcclReduceScatter
 *
 */

#pragma once

#include "rcclBarrierKernels.h"
#include "rcclScalarReduceScatterKernels.h"

extern int RCCL_TRACE_RT;

//! @brief Definition of RcclInternalReduceScatter
//! We split source and destination buffer into n chunks where n is number of
//! gpus. Then, we assign each chunk to each gpu depending on the rank. For
//! example, all-reduce (sum) operation is requested on a 1024 float buffer
//! across 4 gpus, now each gpu operates on (1024/4 = 256) elements. Gpu 2
//! operates on buffer from index 512 to 767. Each gathers data from all the
//! gpus into its registers and does reduction operation on them. Using the
//! above example, gpu 2 gather data from index 512 to 767 from all the gpus
//! into its registers and does floating point addition on them. The final
//! result is stored into local destination buffer.
template <typename DataType_t, typename VectorType_t, rcclRedOp_t Op>
void RcclInternalReduceScatter(RingNode_t* pcurr_track, const void* send_buff,
                           void* recv_buff, hipStream_t stream, int count,
                           int num_gpus, int rank, hipEvent_t event,
                           int* this_time) {
    int num_workitems = 0, num_workgroups = 0;

    int offset = (count / num_gpus) * rank;

    //! Three counts are required to implement chunked allreduce
    //! - op_gpu_count stores how many elements each gpu operates on,
    //! depending on rank of gpu. This is used to launched reduction op
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

    int barrier_value = *this_time;

    //! Set source and destination buffers for current gpu
    hipLaunchKernelGGL(RcclKernelSetSrcDstPtr, dim3(1, 1, 1), dim3(1, 1, 1), 0,
                       stream, pcurr_track, (void*)send_buff, recv_buff);

    //! Wait using multi-gpu barrier until all the gpus set their source and
    //! destination buffers
    hipLaunchKernelGGL(RcclKernelBarrierWait, dim3(1, 1, 1), dim3(1, 1, 1), 0,
                       stream, pcurr_track, barrier_value++, num_gpus);

    //! Once all the gpus have set their buffer, do reduction on portion of the
    //! buffer depending on rank of the gpu
    hipLaunchKernelGGL((RcclKernelScalarReduceScatter<DataType_t, Op>),
                       dim3(num_workgroups, 1, 1), dim3(num_workitems, 1, 1), 0,
                       stream, pcurr_track, (void*)send_buff, recv_buff,
                       op_gpu_count, offset);

    //! Flush gpu l2 cache
    hipEventRecord(event, stream);

    //! Wait until all gpus have finished doing reduction on their respective
    //! portions
    hipLaunchKernelGGL(RcclKernelBarrierWait, dim3(1, 1, 1), dim3(1, 1, 1), 0,
                       stream, pcurr_track, barrier_value++, num_gpus);

    // handling in-place which need to move data
    if (rank && send_buff == recv_buff) {
      //! Once all gpus have done reduction, move data except rank 0 in-place
      hipLaunchKernelGGL((RcclKernelMoveRest<DataType_t>),
                         dim3(num_workgroups, 1, 1), dim3(num_workitems, 1, 1), 0,
                         stream, pcurr_track, num_gpus, rank, regular_gpu_count,
                         last_gpu_count);
      //! Flush gpu l2 cache
      hipEventRecord(event, stream);
    }
    //! Wait until all gpus have finished copied data from other gpus, don't
    //! exit from stream
    hipLaunchKernelGGL(RcclKernelBarrierWait, dim3(1, 1, 1), dim3(1, 1, 1), 0,
                       stream, pcurr_track, barrier_value++, num_gpus);

    //! Update communicator with update barrier count
    *this_time = barrier_value;
}
