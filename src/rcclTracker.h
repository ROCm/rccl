/*
Copyright (c) 2017 - Present Advanced Micro Devices, Inc.
All rights reserved.
*/

/**
 * @file rcclTracker.h
 * @brief Header containing helper data structures
 *
 * This header contains all data structures required for tracking states of
 * different RCCL public data structures when used across different RCCL APIs.
 *
 * @author Aditya Atluri
 */

#pragma once

#include <hip/hip_runtime.h>
#include <atomic>
#include <map>
#include "rcclCheck.h"

#define KNRM "\x1B[0m"
#define KRED "\x1B[31m"
#define KGRN "\x1B[32m"
#define KYEL "\x1B[33m"
#define KBLU "\x1B[34m"
#define KMAG "\x1B[35m"
#define KCYN "\x1B[36m"
#define CWHT "\x1B[37m"

const char* API_COLOR = KGRN;
const char* API_COLOR_END = KNRM;

//! @brief Flags to print different debug levels
//! Enable debug log for rccl api calls
constexpr int krccl_print_api = 1 << 0;
//! Enable debug log for internal function calls
constexpr int krccl_print_internal = 1 << 1;
//! Enable debug log for printing kernel launches
constexpr int krccl_print_kernel = 1 << 2;

//! Limit the total number of workitems launched to 1024
constexpr unsigned knum_workitems = 1024;
//! Limit the number of elements operated on per workgroup
constexpr unsigned knum_vectors_per_workgroup = 1024;

//! @brief Multi-GPU barrier
//! Barrier structure is used to sync kernels from same rccl call across
//! multiple gpus. times_done is used to track how many times a gpu used the
//! barrier. bar_in tracks how many gpus have entered the barrier. bar_out
//! tracks how many gpus have exited. Owned by rcclUniqueId or RingNodePool_t
struct Barrier_t {
    std::atomic<int> bar_in, bar_out, times_done;
};

//! @brief Node for each gpu
//! Data structure used to track details about current gpu. Multiple structures
//! form a ring where RCCL API kernels use them to access data on gpus in
//! clique. It is allocated as pinned host memory visible to all the gpus. The
//! memory is uncached on gpus.
struct RingNode_t {
    //! Point to RingNode_t owned by previous gpu in clique
    struct RingNode_t* prev_gpu;
    //! Point to RingNode_t owned by next gpu in clique
    struct RingNode_t* next_gpu;

    //! We use atomic data type to store pointer to buffers on a gpu, because
    //! there are multiple readers (all peer gpus) and single writer (current
    //! gpu)

    //! Stores source buffer on current gpu
    std::atomic<void*> src_buffer;
    //! Stores destination buffer on current gpu
    std::atomic<void*> dst_buffer;

    //! Stores device index according to hip programming model
    uint32_t hip_current_device_index;

    //! Barrier is allocated once per rcclUniqueId, owned by Rccl
    Barrier_t* barrier;

    //! Holds rank of each gpu
    int rank;
};

struct RcclComm_t;

//! @brief
//! Pool data structure used to store all RingNode_t data structures and track
//! rcclComm_t accordingly
class RingNodePool_t {
  private:
    //! Stores an array of device indices user provided we allocate memory, do
    //! memcpy from user buffer deleted at destruction
    int* device_indices_;
    //! Number of devices in current pool
    int num_devices_;
    //! Barrier used by all devices in pool
    Barrier_t* barrier_;
    //! Reset the ring from the trackers in the pool
    void ResetGpuRing();

  public:
    //! Counter to track how many devices are active in pool. Used to know when
    //! we can destroy the pool and all data structures
    int active_devices_;
    //! Used to track RingNode_t structures for each gpu
    //! key -> rank of the gpu
    //! value -> RingNode_t* of respective gpu
    std::map<int, RingNode_t*> pool_;
    //! Destroy all the elements in pool_, barrier_ and device_indices_
    ~RingNodePool_t();
    //! Construction
    RingNodePool_t();
    RingNodePool_t(const int* device_indices_, int num_devices_);
    //! Adds new gpu to clique, return corresponding RcclComm_t structure
    RcclComm_t* AddDevice(int device, int rank, int ndev);
    //! Remove device from pool_ based on RcclComm_t data
    void RemoveDevice(RcclComm_t* pcomm);
    //! Get number of devices the pool is allocated for
    int GetNumDevices() const { return num_devices_; }
    //! Print data in pool
    void PrintAll();
    //! Given a device index, get RingNode_t structure
    RingNode_t* GetPoolByDeviceIndex(int device_index);
};

//! Internal representation of rcclComm_t structure, which is allocated for each
//! gpu.
struct RcclComm_t {
  public:
    //! Pool of gpus rcclComm_t is created with
    RingNodePool_t* pool_;
    //! RingNode_t* corresponding to current gpu
    RingNode_t* track_;
    //! The stream on which the last rccl call is made using same rcclComm_t
    hipStream_t stream_;
    //! Event with which inter-stream synchronization is done. Also, used to
    //! flush L2 caches after copy and reduction operation. Event is created by
    //! AddDevice method in RingNodePool_t and destroyed by destructor
    hipEvent_t event_;
    //! Variable to track how many times barrier is used by the gpu
    int this_time_;
    //! Number of devices the communicator is created with
    int num_devices_;
    //! Device index of a gpu
    int device_;
    //! Rank of current gpu
    int rank_;
    // Destroy hipEvent_t at deletion of current object
    ~RcclComm_t() { HIPCHECK(hipEventDestroy(event_)); }
};
