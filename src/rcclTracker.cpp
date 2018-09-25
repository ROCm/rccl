/*
Copyright (c) 2017 - Present Advanced Micro Devices, Inc.
All rights reserved.
*/

/**
 * @file rcclTracker.cpp
 * @brief Implementation of rcclTracker.h
 *
 * This file contains implementation of data structures and classes declared in
 * rcclTracker.h
 *
 * @author Aditya Atluri
 */

#include "rcclTracker.h"

//! @brief Default constructor
//! Allocate new barrier_t at initialization
RingNodePool_t::RingNodePool_t() {
    num_devices_ = 0;
    active_devices_ = 0;
    device_indices_ = nullptr;

    HIPCHECK(
        hipHostMalloc(&barrier_, sizeof(Barrier_t), hipHostMallocCoherent));

    std::atomic_store_explicit(&(barrier_->bar_in), 0,
                               std::memory_order_seq_cst);
    std::atomic_store_explicit(&(barrier_->bar_out), 0,
                               std::memory_order_seq_cst);
    std::atomic_store_explicit(&(barrier_->times_done), 0,
                               std::memory_order_seq_cst);
}

//! @brief Default destructor
//! Free barrier and device indices
RingNodePool_t::~RingNodePool_t() {
    if (device_indices_ != nullptr) {
        delete device_indices_;
        device_indices_ = nullptr;
    }
    HIPCHECK(hipHostFree(barrier_));
}

//! @brief Construct device pool
//! Construct RingNode pool from device list and number of devices
RingNodePool_t::RingNodePool_t(const int* device_indices, int num_devices)
    : num_devices_(num_devices), active_devices_(num_devices) {
    //! Store users device index and restore before exiting function
    int user_device_index;
    HIPCHECK(hipGetDevice(&user_device_index));

    //! Allocate memory to store device indices provided by user
    device_indices_ = new int[num_devices_];
    memcpy(device_indices_, device_indices, num_devices_ * sizeof(int));

    struct RingNode_t* pdctl;

    //! Allocate Barrier_t
    HIPCHECK(
        hipHostMalloc(&barrier_, sizeof(Barrier_t), hipHostMallocCoherent));

    //! Reset fields in Barrier_t
    std::atomic_store_explicit(&(barrier_->bar_in), 0,
                               std::memory_order_seq_cst);
    std::atomic_store_explicit(&(barrier_->bar_out), 0,
                               std::memory_order_seq_cst);
    std::atomic_store_explicit(&(barrier_->times_done), 0,
                               std::memory_order_seq_cst);

    //! Allocate RingNode_t as system pinned memory for gpu and add its hip
    //! device index
    for (int i = 0; i < num_devices_; i++) {
        HIPCHECK(
            hipHostMalloc(&pdctl, sizeof(RingNode_t), hipHostMallocCoherent));
        pool_[i] = pdctl;
        pool_[i]->prev_gpu = nullptr;
        pool_[i]->next_gpu = nullptr;
        pool_[i]->hip_current_device_index = device_indices_[i];
        pool_[i]->src_buffer = nullptr;
        pool_[i]->dst_buffer = nullptr;
        pool_[i]->barrier = barrier_;
        pool_[i]->rank = i;
    }

    //! Reset all the nodes in the pool to create a ring
    ResetGpuRing();

    //! restore users hip device index
    HIPCHECK(hipSetDevice(user_device_index));
}

//! @brief Add new gpu to pool of RingNode_t
//! This function adds new gpu to RingNode_t pool
RcclComm_t* RingNodePool_t::AddDevice(int device, int rank, int ndev) {
    //! If device_indices_ is not allocated, create a new buffer
    if (device_indices_ == nullptr) {
        device_indices_ = new int[ndev];
    }

    device_indices_[rank] = device;

    //! Create new communicator
    RcclComm_t* ret_comm = new RcclComm_t;

    //! Increment the number of active devices in the clique
    active_devices_++;
    num_devices_ = ndev;

    //! Populate new communicator created
    ret_comm->num_devices_ = ndev;
    ret_comm->device_ = device;
    ret_comm->rank_ = rank;
    ret_comm->stream_ = NULL;
    ret_comm->this_time_ = 0;

    //! Create new RingNode_t
    struct RingNode_t* pdctl;
    HIPCHECK(hipHostMalloc(&pdctl, sizeof(RingNode_t), hipHostMallocCoherent));

    //! Create hipEvent_t for each gpu in the clique
    HIPCHECK(
        hipEventCreateWithFlags(&ret_comm->event_, hipEventReleaseToSystem));

    //! Populate RingNode_t
    pdctl->prev_gpu = nullptr;
    pdctl->next_gpu = nullptr;

    pdctl->src_buffer = nullptr;
    pdctl->dst_buffer = nullptr;

    pdctl->hip_current_device_index = device;

    pdctl->barrier = barrier_;

    pdctl->rank = rank;

    //! Check if RingNode_t is already created for current gpu
    if (pool_.find(rank) != pool_.end()) {
        // clean existing entry
    } else {
        pool_[rank] = pdctl;
    }

    //! Reset the gpu RingNode_t ring
    ResetGpuRing();

    //! Add RingNode_t to rccl communicator
    ret_comm->track_ = pdctl;
    return ret_comm;
}

//! @brief Resets all RingNode_t in pool
//! This method resets all RingNode_t structures in pool to form a ring
void RingNodePool_t::ResetGpuRing() {
    auto iter_before = pool_.begin();
    auto iter_after = iter_before;
    for (iter_after++; iter_after != pool_.end(); iter_before++, iter_after++) {
        iter_before->second->next_gpu = iter_after->second;
        iter_after->second->prev_gpu = iter_before->second;
    }

    pool_.rbegin()->second->next_gpu = pool_.begin()->second;
    pool_.begin()->second->prev_gpu = pool_.rbegin()->second;
}

//! @brief Removes device from clique and pool
//! This method removes RingNode_t, rcclComm_t from pool and reset gpu tracker
//! ring
void RingNodePool_t::RemoveDevice(RcclComm_t* pcomm) {
    for (auto iter = pool_.begin(); iter != pool_.end(); iter++) {
        std::cout << iter->first << std::endl;
    }
    int rank = pcomm->rank_;
    pool_.erase(rank);
    ResetGpuRing();
}

//! @brief Print elements of all nodes in ring
void RingNodePool_t::PrintAll() {
    for (int i = 0; i < num_devices_; i++) {
        std::cout << "On Device: " << device_indices_[i] << std::endl;
        std::cout << pool_[i]->prev_gpu << std::endl;
        std::cout << pool_[i]->next_gpu << std::endl;
        std::cout << pool_[i]->dst_buffer << std::endl;
        std::cout << pool_[i]->src_buffer << std::endl;
    }
}

//! @brief Get RingNode_t from hip device index
struct RingNode_t* RingNodePool_t::GetPoolByDeviceIndex(int device_index) {
    for (int i = 0; i < num_devices_; i++) {
        if (device_index == device_indices_[i]) {
            return pool_[i];
        }
    }
    return nullptr;
}
