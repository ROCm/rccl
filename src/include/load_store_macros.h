/*************************************************************************
 * Copyright (c) 2015-2022, NVIDIA CORPORATION. All rights reserved.
 * Modifications Copyright (c) 2019-2022 Advanced Micro Devices, Inc. All rights reserved.
 * Modifications Copyright (c) Microsoft Corporation. Licensed under the MIT License.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

#ifndef NCCL_LOAD_STORE_MACROS_H_
#define NCCL_LOAD_STORE_MACROS_H_

#include <stdint.h>
#include <sys/types.h>

#if 0

#define SGLOBAL
#define SLOCAL 
#define SCONSTANT
#define SPRIVATE

#define TGLOBAL
#define TLOCAL
#define TCONSTANT

#define GLOBAL_LOAD(addr) __builtin_nontemporal_load(addr)
#define GLOBAL_STORE(x, addr) __builtin_nontemporal_store((x), (addr))

#define MAYBE_XINLINE __attribute__((noinline))

template<typename T>
__device__ __host__ inline static  T* Tglobal(T* ptr) 
{ return ptr; }

template<typename T>
__device__ __host__ inline static  T* Tlocal(T* ptr) 
{ return ptr; }

template<typename T>
__device__ __host__ inline static auto Xglobal(T* ptr) 
{ return ptr; }

template<typename T>
__device__ __host__ inline static auto Xlocal(T* ptr) 
{ return ptr; }

template<typename T> 
__device__ __host__ inline static auto Xprivate(T* ptr) 
{ return ptr; }

// ---------------------------------------------------------------------------
#else
// ---------------------------------------------------------------------------

// Accessing shared mem pointer with global_load/store -> results in seg fault
// HSA_STATUS_ERROR_MEMORY_APERTURE_VIOLATION: The agent attempted to access memory beyond the largest legal address. code: 0x2

// Accessing global mem pointer with ds_read/write -> seems to be OK but results are wrong of course

#define MAYBE_XINLINE __forceinline__

#define SGLOBAL    __attribute__((address_space(1)))
#define SLOCAL     __attribute__((address_space(3))) // fast LDS memory
#define SCONSTANT  __attribute__((address_space(4)))
#define SPRIVATE   __attribute__((address_space(5)))

#define TGLOBAL   __global
#define TLOCAL    __local
#define TCONSTANT __constant

// NOTE: if data size is small, maybe makes sense to use just normal load/store?
#if 0
#define GLOBAL_LOAD(addr) __builtin_nontemporal_load(Xglobal(addr))
#else
#define GLOBAL_LOAD(addr) Xglobal(addr)[0]
#endif
// it seems that loading with cache and storing without it gives the best results
#if 1
#define GLOBAL_STORE(x, addr) __builtin_nontemporal_store((x), Xglobal(addr))
#else
#define GLOBAL_STORE(x, addr) Xglobal(addr)[0] = (x)
#endif

template<typename T, 
    typename T2 = typename std::remove_volatile<T>::type >
__device__ __host__ inline static  T2* Tglobal(T* ptr) 
{ 
  return (T2*)(T2 SGLOBAL *)reinterpret_cast<uintptr_t>(ptr); 
}

template<typename T, 
    typename T2 = typename std::remove_volatile<T>::type >
__device__ __host__ inline static  T* Tlocal(T* ptr) 
{ return (T2*)(T2 SLOCAL *)reinterpret_cast<uintptr_t>(ptr); }

template<typename T, 
    typename T2 = typename std::remove_volatile<T>::type >
__device__ __host__ inline static auto Xglobal(T* ptr) 
{ return (T2 SGLOBAL *)reinterpret_cast<uintptr_t>(ptr); }

template<typename T, 
    typename T2 = typename std::remove_volatile<T>::type >
__device__ __host__ inline static auto Xlocal(T* ptr) 
{ return (T2 SLOCAL *)reinterpret_cast<uintptr_t>(ptr); }

template<typename T, 
    typename T2 = typename std::remove_volatile<T>::type >
__device__ __host__ inline static auto Xprivate(T* ptr) 
{ return (T2 SPRIVATE *)reinterpret_cast<uintptr_t>(ptr); }

#endif

#endif // NCCL_LOAD_STORE_MACROS_H_
