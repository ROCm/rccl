/*
Copyright (c) 2017-Present Advanced Micro Devices, Inc.
All rights reserved.
*/

#pragma once

#include "rcclTracker.h"

//
// RingNode_t::src_buffer is set to buffer pointers for a gpu
//
__global__ void RcclKernelSetSrcPtr(RingNode_t* pcurr_track, void* send_buff) {
    pcurr_track->src_buffer = send_buff;
}

//
// RingNode_t::dst_buffer is set to buffer pointer for a gpu
//
__global__ void RcclKernelSetDstPtr(RingNode_t* pcurr_track, void* recv_buff) {
    pcurr_track->dst_buffer = recv_buff;
}

__global__ void RcclKernelSetSrcDstPtr(RingNode_t* pcurr_track, void* send_buff,
                                       void* recv_buff) {
    pcurr_track->src_buffer = send_buff;
    pcurr_track->dst_buffer = recv_buff;
}
