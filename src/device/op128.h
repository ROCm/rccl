/*************************************************************************
 * Copyright (c) 2016-2019, NVIDIA CORPORATION. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

#ifndef OP128_H_
#define OP128_H_

#include <type_traits>
#include "load_store_macros.h"

inline __device__ void load128(const uint64_t* ptr, uint64_t &v0, uint64_t &v1) {
  v0 = __builtin_nontemporal_load(Tglobal(ptr));
  v1 = __builtin_nontemporal_load(Tglobal(ptr)+1);
}

inline __device__ void store128(uint64_t* ptr, uint64_t v0, uint64_t v1) {
  __builtin_nontemporal_store(v0, Tglobal(ptr));
  __builtin_nontemporal_store(v1, Tglobal(ptr)+1);
}

inline __device__ uint64_t* shmemCvtPtr(volatile uint64_t* shmemGenericPtr) {
  return Tlocal((uint64_t*)shmemGenericPtr);
}

inline __device__ void loadShmem128(uint64_t* shmemAsmPtr, uint64_t &v0, uint64_t &v1) {
  v0 = *Tlocal(shmemAsmPtr);
  v1 = *Tlocal(shmemAsmPtr+1);
}

inline __device__ void storeShmem128(uint64_t* shmemAsmPtr, uint64_t v0, uint64_t v1) {
  *Tlocal(shmemAsmPtr) = v0;
  *Tlocal(shmemAsmPtr+1) = v1;
}

template<typename T>
inline __device__ void loadShmemMisaligned128(T *ptr, uint64_t &v0, uint64_t &v1) {
  union {
    uint32_t tmp4[4];
    uint64_t tmp8[2];
  };
  if(sizeof(T) < 4) {
    uint32_t *ptr4 = reinterpret_cast<uint32_t*>(reinterpret_cast<uintptr_t>(ptr) & -uintptr_t(4));
    #pragma unroll
    for(int e=0; e < 4; e++) {
      // Produce 4 bytes of sub-register type by reading 2 4-byte
      // aligned values and shifting.
      uint32_t lo, hi;
      
      // non-temporal load ???
      lo = __builtin_nontemporal_load(Tlocal((uint32_t *)ptr4+e+0));
      hi = __builtin_nontemporal_load(Tlocal((uint32_t *)ptr4+e+1));
      tmp4[e] = __funnelshift_r(lo, hi, 8*(int(reinterpret_cast<uintptr_t>(ptr))%4));
    }
  }
  else if(sizeof(T) == 4) {
    #pragma unroll
    for(int e=0; e < 4; e++)
      tmp4[e] = __builtin_nontemporal_load(Tlocal(reinterpret_cast<uint32_t  *>(ptr)+e));
  }
  else /*sizeof(T)==8*/ {
    #pragma unroll
    for(int e=0; e < 2; e++)
      tmp8[e] = __builtin_nontemporal_load(Tlocal(reinterpret_cast<uint64_t  *>(ptr)+e));
  }
  v0 = tmp8[0];
  v1 = tmp8[1];
}


template<typename T>
__device__ __forceinline__ uint32_t cvta_to_shared(T* ptr) {
  return (uint32_t)(uint64_t)(ptr);
}
template<typename T>
__device__ __forceinline__ uintptr_t cvta_to_global(T* ptr) {
  return (uintptr_t)(ptr);
}

template<typename T>
__device__ __forceinline__ T* cvta_from_shared(uint32_t shptr) {
  return (T*)shptr;
}
template<typename T>
__device__ __forceinline__ T* cvta_from_global(uintptr_t gptr) {
  return (T*)gptr;
}

////////////////////////////////////////////////////////////////////////////////
// BytePack<Size>: struct of bytes.

template<int Size>
union BytePack;
template<>
union BytePack<0> {};
template<>
union BytePack<1> {
  uint8_t u8, native;
};
template<>
union BytePack<2> {
  BytePack<1> half[2];
  uint8_t u8[2];
  uint16_t u16, native;
};
template<>
union BytePack<4> {
  BytePack<2> half[2];
  uint8_t u8[4];
  uint16_t u16[2];
  uint32_t u32, native;
};
template<>
union BytePack<8> {
  BytePack<4> half[2];
  uint8_t u8[8];
  uint16_t u16[4];
  uint32_t u32[2];
  uint64_t u64, native;
};
template<>
union alignas(16) BytePack<16> {
  BytePack<8> half[2];
  uint8_t u8[16];
  uint16_t u16[8];
  uint32_t u32[4];
  uint64_t u64[2];
  ulong2 ul2, native;
#if !defined(USE_INDIRECT_FUNCTION_CALL) || defined(__gfx942__) || defined(__gfx950__)
  inline __device__ BytePack<16>() = default;
  inline __device__ BytePack<16>(const BytePack<16>& other) {
    *this = other;
  }
  inline __device__ BytePack<16>& operator=(const BytePack<16>& other) {
    u64[0] = other.u64[0];
    u64[1] = other.u64[1];
    return *this;
  }
#endif
};

template<typename T>
struct BytePackOf {
  static constexpr int Size = sizeof(T);
  using Pack = BytePack<Size>;
};
template<>
struct BytePackOf<BytePack<0>> {
  static constexpr int Size = 0;
  using Pack = BytePack<0>;
};

template<typename T>
__device__ __forceinline__ typename BytePackOf<T>::Pack toPack(T value)  {
  union { typename BytePackOf<T>::Pack p; T v; };
  // Coverity recommends the use of std::move here but, given that T is a POD
  // scalar, a plain copy will be just as efficient.
  // coverity[copy_assignment_call]
  v = value;
  return p;
}

template<typename T>
__device__ __forceinline__ T fromPack(typename BytePackOf<T>::Pack pack)  {
  union { typename BytePackOf<T>::Pack p; T v; };
  p = pack;
  return v;
}

////////////////////////////////////////////////////////////////////////////////
// Load/store of BytePack<?> using integral addresses.

template<int Size> __device__ BytePack<Size> ld_volatile_global(uintptr_t addr);
template<int Size> __device__ void st_global(uintptr_t addr, BytePack<Size> value);

template<> __device__ __forceinline__ BytePack<0> ld_volatile_global<0>(uintptr_t addr) { return {}; }
template<> __device__ __forceinline__ void st_global<0>(uintptr_t addr, BytePack<0> value) {}

// Used to define implementations for above prototypes.
#define DEFINE_ld_st__size_space(bytes, data_cxx_ty, data_ptx_ty, data_reg_ty, space, addr_cxx_ty, addr_reg_ty) \
  template<> \
  __device__ __forceinline__ BytePack<bytes> ld_volatile_##space<bytes>(addr_cxx_ty addr) { \
    data_cxx_ty tmp; \
    tmp =  __builtin_nontemporal_load(Tglobal((data_cxx_ty *)addr)); \
    BytePack<bytes> ans; \
    ans.native = tmp; \
    return ans; \
  } \
  template<> \
  __device__ __forceinline__ void st_##space<bytes>(addr_cxx_ty addr, BytePack<bytes> value) { \
    __builtin_nontemporal_store(value.native, Tglobal((data_cxx_ty *)addr)); \
  }

#define DEFINE_ld_st__size(bytes, data_cxx_ty, data_ptx_ty, data_reg_ty) \
  DEFINE_ld_st__size_space(bytes, data_cxx_ty, data_ptx_ty, data_reg_ty, global, uintptr_t, l)

// Single-byte types use 4-byte registers since there is no 1-byte register
// character for asm blocks. See https://docs.nvidia.com/cuda/inline-ptx-assembly/index.html#constraints
DEFINE_ld_st__size(1, uint8_t, b8, r)
DEFINE_ld_st__size(2, uint16_t, b16, h)
DEFINE_ld_st__size(4, uint32_t, b32, r)
DEFINE_ld_st__size(8, uint64_t, b64, l)

#undef DEFINE_ld_st__size_space
#undef DEFINE_ld_st__size

#define DEFINE_ld_st_16__space(space, addr_cxx_ty, addr_reg_ty) \
  template<> \
  __device__ __forceinline__ BytePack<16> ld_volatile_##space<16>(addr_cxx_ty addr) { \
    BytePack<16> ans; \
    ans.u64[0] = __builtin_nontemporal_load(Tglobal((uint64_t *)addr)); \
    ans.u64[1] = __builtin_nontemporal_load(Tglobal((uint64_t *)addr+1)); \
    return ans; \
  } \
  template<> \
  __device__ __forceinline__ void st_##space<16>(addr_cxx_ty addr, BytePack<16> value) { \
    __builtin_nontemporal_store(value.u64[0], Tglobal((uint64_t *)addr)); \
    __builtin_nontemporal_store(value.u64[1], Tglobal((uint64_t *)addr+1)); \
  }

DEFINE_ld_st_16__space(global, uintptr_t, l)
// DEFINE_ld_st_16__space(shared, uint32_t, r)
#undef DEFINE_ld_st_16

// #undef PTX_relaxed_gpu

////////////////////////////////////////////////////////////////////////////////
// Atomic load/store using c++ pointers.

template <typename T>
__device__ __forceinline__ T ld_uncached_global(T *ptr) {
  return __atomic_load_n(Tglobal(ptr) ,__ATOMIC_RELAXED);
}

template <typename T>
__device__ __forceinline__ T ld_relaxed_sys_global(T *ptr) {
  return __hip_atomic_load(Tglobal(ptr), __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_SYSTEM);
}

template <typename T>
__device__ __forceinline__ T ld_acquire_sys_global(T *ptr) {
  return __hip_atomic_load(Tglobal(ptr), __ATOMIC_ACQUIRE, __HIP_MEMORY_SCOPE_SYSTEM);
}

template <typename T>
__device__ __forceinline__ T ld_seq_sys_global(T *ptr) {
  // NOTE this seems to generate buffer_inv sc0 sc1
  return __atomic_load_n(Tglobal(ptr), __ATOMIC_SEQ_CST);
  //return __hip_atomic_load(Tglobal(ptr), __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_SYSTEM);
}

//////////////////////////////////////////////////////////////////////////////
template <typename T, typename C >
__device__ inline static void st_uncached_global(T* ptr, C val) {
  static_assert(sizeof(C) <= sizeof(T), "Types must be convertible C -> T");
  __atomic_store_n(Tglobal(ptr), static_cast<T>(val), __ATOMIC_RELAXED);
}

template <typename T, typename C>
__device__ inline static void st_relaxed_sys_global(T* ptr, C val) {
  static_assert(sizeof(C) <= sizeof(T), "Types must be convertible C -> T");
  __hip_atomic_store(Tglobal(ptr), static_cast<T>(val), __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_SYSTEM);
}

template <typename T, typename C>
__device__ __forceinline__ void st_release_sys_global(T *ptr, C val) {
  static_assert(sizeof(C) <= sizeof(T), "Types must be convertible C -> T");
  __hip_atomic_store(Tglobal(ptr), static_cast<T>(val), __ATOMIC_RELEASE, __HIP_MEMORY_SCOPE_SYSTEM);
  //__atomic_store_n((uint64_t SGLOBAL *)ptr, val, __ATOMIC_SEQ_CST);
}

template <typename T, typename C>
__device__ __forceinline__ void st_seq_sys_global(T *ptr, C val) {
  static_assert(sizeof(C) <= sizeof(T), "Types must be convertible C -> T");
  __atomic_store_n(Tglobal(ptr), static_cast<T>(val), __ATOMIC_SEQ_CST);
}

__device__ __forceinline__ void fence_acq_rel_sys() {
    //asm volatile("membar.sys;" ::: "memory");
}
__device__ __forceinline__ void fence_acq_rel_gpu() {
    //asm volatile("membar.gl;" ::: "memory");
}

////////////////////////////////////////////////////////////////////////////////
// Multimem stores of BytePack<?>.

template<int Size>
__device__ __forceinline__ void multimem_st_global(uintptr_t addr, BytePack<Size> val) {
  // nop
}

template<int EltSize, int MaxBytes, bool Multimem, typename IntBytes>
__device__ __forceinline__ void copyGlobalShared_WarpUnrolled(
    int lane, uintptr_t dstAddr, uint32_t srcAddr, IntBytes nBytesAhead
  ) {
  // nop
}

#endif
