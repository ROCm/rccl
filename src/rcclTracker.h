/*
Copyright (c) 2017 - Present Advanced Micro Devices, Inc.
All rights reserved.
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

//
constexpr int krccl_print_api = 1 << 0;
constexpr int krccl_print_internal = 1 << 1;
constexpr int krccl_print_kernel = 1 << 2;

// we always launch 1024 workitems per workgroup
// in our kernels
constexpr unsigned knum_workitems = 1024;
// we use 1024 vectors per workgroup
// where single workitem does op on single vector
constexpr unsigned knum_vectors_per_workgroup = 1024;

struct Barrier_t {
    std::atomic<int> bar_in, bar_out, times_done;
};

//
// data structure used to track details about peer gpus.
// It is allocated as pinned host memory visible to all the gpus
//
struct RingNode_t {
    // point to RingNode_t owned by previous gpu in clique
    struct RingNode_t* prev_gpu;
    // point to RingNode_t owned by next gpu in clique
    struct RingNode_t* next_gpu;
    // we use atomic data type to store pointer to buffers
    // on a gpu, as there are multiple readers (all peer gpus)
    // and single writer (current gpu)

    // stores source buffer on current gpu
    void* src_buffer;
    // stores destination buffer on current gpu
    void* dst_buffer;
    // stores device index according to hip programming model
    uint32_t hip_current_device_index;

    Barrier_t* barrier;

    int rank;
};

struct RcclComm_t;

//
// pool data structure used to store all RingNode_t
// data structures and track rcclComm_t accordingly
//
class RingNodePool_t {
  private:
    // stores an array of device indices user provided
    // we allocate memory, do memcpy from user buffer
    // deleted at destruction
    int* device_indices_;
    // number of devices a pool is created for
    int num_devices_;

    Barrier_t* barrier_;

    void ResetGpuRing();

  public:
    // counter to track how many devices are
    // active in pool. Used to know when we can
    // destroy the pool and all data structures
    int active_devices_;
    // used to track RingNode_t structures for each gpu
    std::map<int, RingNode_t*> pool_;

    ~RingNodePool_t();
    // construction is initialization
    RingNodePool_t(const int* device_indices_, int num_devices_);

    RingNodePool_t();

    // when a new device is added to clique,
    // return corresponding RcclComm_t structure
    RcclComm_t* AddDevice(int device, int rank, int ndev);

    void RemoveDevice(RcclComm_t* pcomm);

    void PrintAll();
    // given a device index, get RingNode_t structure
    RingNode_t* GetPoolByDeviceIndex(int device_index);
};

// internal representation of rcclComm_t
// for structure is allocated for each gpu,
// tracks,
// 1. the RingNode_t pool where it belongs to
// 2. tracker corresponding to gpu
// 3. number of devices in the pool
// 4. device index it is associated to
// 5. rank of gpu in clique
struct RcclComm_t {
  public:
    RingNodePool_t* pool_;
    RingNode_t* track_;
    hipStream_t stream_;
    hipEvent_t event_;
    int this_time_;
    int num_devices_;
    int device_;
    int rank_;
    ~RcclComm_t() { HIPCHECK(hipEventDestroy(event_)); }
};
