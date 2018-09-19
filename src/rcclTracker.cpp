/*
Copyright (c) 2017 - Present Advanced Micro Devices, Inc.
All rights reserved.
*/

#include "rcclTracker.h"

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

RingNodePool_t::~RingNodePool_t() {
    if (device_indices_ != nullptr) {
        delete device_indices_;
        device_indices_ = nullptr;
    }
    HIPCHECK(hipHostFree(barrier_));
}

RingNodePool_t::RingNodePool_t(const int* device_indices, int num_devices)
    : num_devices_(num_devices), active_devices_(num_devices) {
    // store users device index
    // restore before exiting function
    int user_device_index;
    HIPCHECK(hipGetDevice(&user_device_index));

    // allocate memory to store device indices provided by user
    device_indices_ = new int[num_devices_];
    memcpy(device_indices_, device_indices, num_devices_ * sizeof(int));

    struct RingNode_t* pdctl;

    HIPCHECK(
        hipHostMalloc(&barrier_, sizeof(Barrier_t), hipHostMallocCoherent));

    std::atomic_store_explicit(&(barrier_->bar_in), 0,
                               std::memory_order_seq_cst);
    std::atomic_store_explicit(&(barrier_->bar_out), 0,
                               std::memory_order_seq_cst);
    std::atomic_store_explicit(&(barrier_->times_done), 0,
                               std::memory_order_seq_cst);

    // allocate RingNode_t as system pinned memory for gpu
    // and add its hip device index
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

    ResetGpuRing();

    // restore users hip device index
    HIPCHECK(hipSetDevice(user_device_index));
}

RcclComm_t* RingNodePool_t::AddDevice(int device, int rank, int ndev) {
    RcclComm_t* ret_comm = new RcclComm_t;
    active_devices_++;
    num_devices_ = ndev;
    ret_comm->num_devices_ = ndev;
    ret_comm->device_ = device;
    ret_comm->rank_ = rank;
    ret_comm->stream_ = NULL;
    ret_comm->this_time_ = 0;
    struct RingNode_t* pdctl;
    HIPCHECK(hipHostMalloc(&pdctl, sizeof(RingNode_t), hipHostMallocCoherent));
    HIPCHECK(
        hipEventCreateWithFlags(&ret_comm->event_, hipEventReleaseToSystem));

    pdctl->prev_gpu = nullptr;
    pdctl->next_gpu = nullptr;

    pdctl->src_buffer = nullptr;
    pdctl->dst_buffer = nullptr;

    pdctl->hip_current_device_index = device;

    pdctl->barrier = barrier_;

    pdctl->rank = rank;

    if (pool_.find(rank) != pool_.end()) {
        // clean existing entry
    } else {
        pool_[rank] = pdctl;
    }

    ResetGpuRing();

    ret_comm->track_ = pdctl;
    return ret_comm;
}

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

void RingNodePool_t::RemoveDevice(RcclComm_t* pcomm) {
    for (auto iter = pool_.begin(); iter != pool_.end(); iter++) {
        std::cout << iter->first << std::endl;
    }
    int rank = pcomm->rank_;
    pool_.erase(rank);
    ResetGpuRing();
}

void RingNodePool_t::PrintAll() {
    for (int i = 0; i < num_devices_; i++) {
        std::cout << "On Device: " << device_indices_[i] << std::endl;
        std::cout << pool_[i]->prev_gpu << std::endl;
        std::cout << pool_[i]->next_gpu << std::endl;
        std::cout << pool_[i]->dst_buffer << std::endl;
        std::cout << pool_[i]->src_buffer << std::endl;
    }
}

struct RingNode_t* RingNodePool_t::GetPoolByDeviceIndex(int device_index) {
    for (int i = 0; i < num_devices_; i++) {
        if (device_index == device_indices_[i]) {
            return pool_[i];
        }
    }
    return nullptr;
}
