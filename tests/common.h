/*
Copyright (c) 2017-Present Advanced Micro Devices, Inc.
All rights reserved.
*/

#pragma once

#include <unordered_map>
#include <vector>

#include "rccl/rccl.h"

#define HIPCHECK(status)                                             \
    if (status != hipSuccess) {                                      \
        std::cout << "Got: " << hipGetErrorString(status)            \
                  << " at: " << __LINE__ << " in file: " << __FILE__ \
                  << std::endl;                                      \
    }

#define RCCLCHECK(status)                                            \
    if (status != rcclSuccess) {                                     \
        std::cout << "Got: " << rcclGetErrorString(status)           \
                  << " at: " << __LINE__ << " in file: " << __FILE__ \
                  << std::endl;                                      \
    }

#define MAKE_UMAP_VALS(val) \
    { #val, val }

//
// Buffers allocated in tests shouldn't exceed
// the following limit: 256MB (irrespective of data type)
//
constexpr size_t kmax_buffer_size = 1024 * 1024 * 256;  // 256MB

//
// The values are accessed using device-id as index
// Limiting to 16 gpus
//
const std::vector<unsigned> kbuffer_values = {1, 2,  3,  4,  5,  6,  7,  8,
                                              9, 10, 11, 12, 13, 14, 15, 16};

//
// Create a hash map to get rcclOp from string provided by user
//
std::unordered_map<std::string, rcclRedOp_t> umap_rccl_op = {
    MAKE_UMAP_VALS(rcclSum), MAKE_UMAP_VALS(rcclProd), MAKE_UMAP_VALS(rcclMax),
    MAKE_UMAP_VALS(rcclMin)};

//
// Create a hash map to get rcclDataType from string provided by user
//
std::unordered_map<std::string, rcclDataType_t> umap_rccl_dtype = {
    MAKE_UMAP_VALS(rcclChar),   MAKE_UMAP_VALS(rcclInt),
    MAKE_UMAP_VALS(rcclHalf),   MAKE_UMAP_VALS(rcclFloat),
    MAKE_UMAP_VALS(rcclDouble), MAKE_UMAP_VALS(rcclInt64),
    MAKE_UMAP_VALS(rcclUint64)};

//
// Used to print multi-argument values
//
template <typename T>
void print_out(T val) {
    std::cout << val << std::endl;
}

template <typename T, typename... Args>
void print_out(T val, Args... args) {
    std::cout << val << " ";
    print_out(args...);
}

//
// Helps restore gpu device user set before
// any operation which may change current gpu.
// Will restore gpu device when the instance
// goes out of scope
//
class CurrDeviceGuard_t {
  private:
    int curr_device_index_;

  public:
    CurrDeviceGuard_t() { HIPCHECK(hipGetDevice(&curr_device_index_)); }
    ~CurrDeviceGuard_t() { HIPCHECK(hipSetDevice(curr_device_index_)); }
};

//
// RandomSizeGen_t is used to generate random numbers
// which are used as sizes for allocation buffers to
// do rccl ops
//
class RandomSizeGen_t {
  private:
    int seed_;
    int min_limit_;
    int max_limit_;

  public:
    RandomSizeGen_t(int seed, int min_limit, int max_limit)
        : seed_(seed), min_limit_(min_limit), max_limit_(max_limit) {
        std::srand(seed_);
    }
    size_t GetSize() const {  // never return 0
        return ((std::rand() - min_limit_) % max_limit_) + 1;
    }
};

//
// Given a list of gpu devices, this function
// enables peer access between them. Each gpu
// will be peer to every gpu in the list
//
void EnableDevicePeerAccess(std::vector<int>& device_list) {
    CurrDeviceGuard_t g;
    for (auto iter1 = device_list.begin(); iter1 != device_list.end();
         iter1++) {
        HIPCHECK(hipSetDevice(*iter1));
        for (auto iter2 = device_list.begin(); iter2 != device_list.end();
             iter2++) {
            if (*iter1 != *iter2) {
                HIPCHECK(hipDeviceEnablePeerAccess(*iter2, 0));
            }
        }
    }
}
