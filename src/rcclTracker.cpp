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

    // create a ring of trackers
    pdctl = nullptr;  // reuse later.
    HIPCHECK(hipHostGetDevicePointer((void**)&pdctl, pool_[0], 0));
    if (num_devices_ != 1) {
        pool_[1]->prev_gpu = pdctl;
    } else {
        pool_[0]->prev_gpu = pdctl;
    }

    pool_[num_devices_ - 1]->next_gpu = pdctl;

    for (unsigned i = 1; i < num_devices_; i++) {
        HIPCHECK(hipSetDevice(device_indices_[i]));
        HIPCHECK(hipHostGetDevicePointer((void**)&pdctl, pool_[i], 0));
        pool_[(i + 1) % num_devices_]->prev_gpu = pdctl;
        pool_[(i - 1) % num_devices_]->next_gpu = pdctl;
    }

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

    if (pool_.size() == ndev) {
        pool_[1]->prev_gpu = pool_[0];
        pool_[ndev - 1]->next_gpu = pool_[0];
        for (int i = 1; i < ndev; i++) {
            pool_[(i + 1) % ndev]->prev_gpu = pool_[i];
            pool_[(i - 1) % ndev]->next_gpu = pool_[i];
        }
    }

    ret_comm->track_ = pdctl;
    return ret_comm;
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
