/*
Copyright (c) 2017 - Present Advanced Micro Devices, Inc.
All rights reserved.
*/

#pragma once

#include "rcclScalarAllReduceKernels.h"

//
// The code here figures out the launch parameters for allreduce ops
//

extern int RCCL_TRACE_RT;
/*
static int this_time = 0;

template<typename DataType_t, typename VectorType_t, rcclRedOp_t Op>
void RcclInternalAllReduce(DeviceControl_t *pcurr_track, const void* send_buff, void* recv_buff, hipStream_t stream, int count, int num_gpus, int rank, hipEvent_t event) {

    int num_workitems = 0, num_workgroups = 0;

    int offset = (count / num_gpus) * rank;
    int max_count_per_gpu = (count / num_gpus) + (count % num_gpus);
    int count_per_gpu = (rank == num_gpus - 1) ? ((count / num_gpus) + (count % num_gpus)) : (count / num_gpus);

    if(count_per_gpu > knum_workitems) { // knum_workitems = 1024
        num_workitems = knum_workitems; 
        num_workgroups = max_count_per_gpu / knum_workitems + 1;
    } else {
        num_workitems = max_count_per_gpu;
        num_workgroups = 1;
    }

    hipLaunchKernelGGL(RcclKernelSetSrcDstBuffer, dim3(1,1,1), dim3(1,1,1), 0, stream, pcurr_track, (void*)send_buff, recv_buff);

    hipLaunchKernelGGL(RcclKernelBarrierWait, dim3(1,1,1), dim3(1,1,1), 0, stream, pcurr_track, this_time++, num_gpus);

    hipLaunchKernelGGL((RcclKernelScalarAllReduce<DataType_t, Op>), dim3(num_workgroups,1,1), dim3(num_workitems,1,1), 0, stream, pcurr_track, (void*)send_buff, recv_buff, count_per_gpu, offset);
    hipEventRecord(event, stream);

    hipLaunchKernelGGL(RcclKernelBarrierWait, dim3(1,1,1), dim3(1,1,1), 0, stream, pcurr_track, this_time++, num_gpus);

    hipLaunchKernelGGL((RcclKernelCopyRest<DataType_t>), dim3(num_workgroups,1,1), dim3(num_workitems,1,1), 0, stream, pcurr_track, num_gpus, rank, count_per_gpu, max_count_per_gpu);
    hipEventRecord(event, stream);

    hipLaunchKernelGGL(RcclKernelBarrierWait, dim3(1,1,1), dim3(1,1,1), 0, stream, pcurr_track, this_time++, num_gpus);
}
*/
template<typename DataType_t, typename VectorType_t, rcclRedOp_t Op>
void RcclInternalAllReduce(DeviceControl_t *pcurr_track, const void* send_buff, void* recv_buff, hipStream_t stream, int count, int num_gpus, int rank, hipEvent_t event, int* this_time) {
/*
    int num_workitems = 0, num_workgroups = 0;

    int offset = (count / num_gpus) * rank;
    int max_count_per_gpu = (count / num_gpus) + (count % num_gpus);
    int count_per_gpu = (rank == num_gpus - 1) ? ((count / num_gpus) + (count % num_gpus)) : (count / num_gpus);

    if(count_per_gpu > knum_workitems) { // knum_workitems = 1024
        num_workitems = knum_workitems; 
        num_workgroups = max_count_per_gpu / knum_workitems + 1;
    } else {
        num_workitems = max_count_per_gpu;
        num_workgroups = 1;
    }

    hipLaunchKernelGGL(RcclKernelSetSrcDstBuffer, dim3(1,1,1), dim3(1,1,1), 0, stream, pcurr_track, (void*)send_buff, recv_buff);

    int val = *this_time;

    hipLaunchKernelGGL(RcclKernelBarrierWait, dim3(1,1,1), dim3(1,1,1), 0, stream, pcurr_track, val++, num_gpus);


    hipLaunchKernelGGL((RcclKernelScalarAllReduce<DataType_t, Op>), dim3(num_workgroups,1,1), dim3(num_workitems,1,1), 0, stream, pcurr_track, (void*)send_buff, recv_buff, count_per_gpu, offset);
    hipEventRecord(event, stream);

    hipLaunchKernelGGL(RcclKernelBarrierWait, dim3(1,1,1), dim3(1,1,1), 0, stream, pcurr_track, val++, num_gpus);

    hipLaunchKernelGGL((RcclKernelCopyRest<DataType_t>), dim3(num_workgroups,1,1), dim3(num_workitems,1,1), 0, stream, pcurr_track, num_gpus, rank, count_per_gpu, max_count_per_gpu);
    hipEventRecord(event, stream);
    hipLaunchKernelGGL(RcclKernelBarrierWait, dim3(1,1,1), dim3(1,1,1), 0, stream, pcurr_track, val++, num_gpus);
    *this_time = val;
*/

       int num_workitems = 0, num_workgroups = 0;

       int offset = (count / num_gpus) * rank;
       int regular_gpu_count = count / num_gpus;
       int last_gpu_count = ((count / num_gpus) + (count % num_gpus));

       int op_gpu_count = (rank == num_gpus - 1) ? last_gpu_count : regular_gpu_count;

       if(last_gpu_count < knum_workitems) {
               num_workitems = last_gpu_count;
               num_workgroups = 1;
       } else {
               num_workitems = knum_workitems;
               num_workgroups = (last_gpu_count / knum_workitems) + 1;
       }

    hipLaunchKernelGGL(RcclKernelSetSrcDstBuffer, dim3(1,1,1), dim3(1,1,1), 0, stream, pcurr_track, (void*)send_buff, recv_buff);

    int val = *this_time;

    hipLaunchKernelGGL(RcclKernelBarrierWait, dim3(1,1,1), dim3(1,1,1), 0, stream, pcurr_track, val++, num_gpus);


       hipLaunchKernelGGL((RcclKernelScalarAllReduce<DataType_t, Op>), dim3(num_workgroups, 1, 1), dim3(num_workitems, 1, 1), 0, stream, pcurr_track, (void*)send_buff, recv_buff, op_gpu_count, offset);
    hipEventRecord(event, stream);

    hipLaunchKernelGGL(RcclKernelBarrierWait, dim3(1,1,1), dim3(1,1,1), 0, stream, pcurr_track, val++, num_gpus);

    hipLaunchKernelGGL((RcclKernelCopyRest<DataType_t>), dim3(num_workgroups,1,1), dim3(num_workitems,1,1), 0, stream, pcurr_track, num_gpus, rank, regular_gpu_count, last_gpu_count);

    hipEventRecord(event, stream);
    hipLaunchKernelGGL(RcclKernelBarrierWait, dim3(1,1,1), dim3(1,1,1), 0, stream, pcurr_track, val++, num_gpus);
    *this_time = val;

}

