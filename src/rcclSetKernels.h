/*
Copyright (c) 2017-Present Advanced Micro Devices, Inc.
All rights reserved.
*/

//
// This file contains kernels which sets source and destination pointer of
// current gpu. All the kernels are launched with one workitem and one
// workgroup.
//

#pragma once

#include "rcclTracker.h"

//
// RingNode_t::src_buffer is set
//
__global__ void RcclKernelSetSrcPtr(RingNode_t* pcurr_track, void* send_buff) {
    pcurr_track->src_buffer = send_buff;
}

//
// RingNode_t::dst_buffer is set
//
__global__ void RcclKernelSetDstPtr(RingNode_t* pcurr_track, void* recv_buff) {
    pcurr_track->dst_buffer = recv_buff;
}

//
// RingNode_t::src_buffer and RingNode_t::dst_buffer is set
//
__global__ void RcclKernelSetSrcDstPtr(RingNode_t* pcurr_track, void* send_buff,
                                       void* recv_buff) {
    pcurr_track->src_buffer = send_buff;
    pcurr_track->dst_buffer = recv_buff;
}
