/*
Copyright (c) 2017 - Present Advanced Micro Devices, Inc.
All rights reserved.
*/

/**
 * @file rcclBarrierKernels.h
 * @brief Barriers implementation in kernels
 *
 * This file contains kernels to implement multi-gpu barriers
 *
 * @author Aditya Atluri
 */

#pragma once

#include "rcclTracker.h"

//! @brief Definition of RcclKernelBarrierWait
//! This kernel acts a barrier between stream on multiple gpus. In rccl, this
//! kernel is used to make sure all the gpus done executing the previous
//! kernels so that the data they updated is visible to all the gpus. The data
//! structure Barrier_t present in rcclTracker.h is created once and shared
//! across all the gpus. The elements inside the structure are:
//! 1. bar_in, it stores the count of number of gpus that entered the barrier
//! 2. bar_out, it stores the count of number of gpus that exited the barrier
//! 3. times_done, it stores how many times the barrier has been used
//! In the kernel, first we check if all the gpus arrived at the same barrier
//! instance. For example, if one gpu (lets say A) exited the barrier and try to
//! use the same barrier before other gpus exited, 'A' gpu will wait until all
//! other gpus have exited the barrier and their local counter to number of
//! times used is incremented by 1. Once all the gpus are entering the same
//! instance of barrier, they increment bar_in atomic signaling other gpus that
//! it entered the barrier and get the value of how many gpus entered the
//! barrier before it entered. Now, it waits until all the gpus have entered the
//! barrier. Once all the gpus have entered the barrier, exit the barrier by
//! incremenet bar_out. Check if final gpu is exiting the barrier, this done by
//! checking if the last gpus old_val is one less than number of gpus. If yes,
//! then we wait until all the gpus have exited the barrier and reset all the
//! barrier in and out flags. Finally, increment the barrier instance usage
//! count by 1.
__global__ void RcclKernelBarrierWait(RingNode_t* pcurr_track, int this_time,
                                      int get_here) {
    int val = 1;
    while (std::atomic_load_explicit(&(pcurr_track->barrier->times_done),
                                     std::memory_order_seq_cst) != this_time) {
    }

    int old_val =
        pcurr_track->barrier->bar_in.fetch_add(val, std::memory_order_seq_cst);

    while (std::atomic_load_explicit(&(pcurr_track->barrier->bar_in),
                                     std::memory_order_seq_cst) != get_here) {
    }
    pcurr_track->barrier->bar_out.fetch_add(val, std::memory_order_seq_cst);

    if (old_val + val == get_here) {
        while (std::atomic_load_explicit(&(pcurr_track->barrier->bar_out),
                                         std::memory_order_seq_cst) !=
               get_here) {
        }

        std::atomic_store_explicit(&(pcurr_track->barrier->bar_in), 0,
                                   std::memory_order_seq_cst);
        std::atomic_store_explicit(&(pcurr_track->barrier->bar_out), 0,
                                   std::memory_order_seq_cst);

        pcurr_track->barrier->times_done.fetch_add(val,
                                                   std::memory_order_seq_cst);
    }
}
