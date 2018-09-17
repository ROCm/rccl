/*
Copyright (c) 2017 - Present Advanced Micro Devices, Inc.
All rights reserved.
*/

#pragma once

//
// This file declares kernels for barriers
//


__global__ void RcclKernelBarrierWait(RingNode_t* pcurr_track, int this_time, int get_here) {
    int val = 1;
    while(std::atomic_load_explicit(pcurr_track->times_done, std::memory_order_seq_cst) != this_time) {}

    int old_val = pcurr_track->bar_in->fetch_add(1, std::memory_order_seq_cst);

    while(std::atomic_load_explicit(pcurr_track->bar_in, std::memory_order_seq_cst) != get_here) {}
    pcurr_track->bar_out->fetch_add(1, std::memory_order_seq_cst);

    if(old_val + val == get_here) {
        while(std::atomic_load_explicit(pcurr_track->bar_out, std::memory_order_seq_cst) != get_here) {}

        std::atomic_store_explicit(pcurr_track->bar_in, 0, std::memory_order_seq_cst);
        std::atomic_store_explicit(pcurr_track->bar_out, 0, std::memory_order_seq_cst);

        pcurr_track->times_done->fetch_add(1, std::memory_order_seq_cst);
    }
}
