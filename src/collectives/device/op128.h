/*************************************************************************
 * Copyright (c) 2016-2019, NVIDIA CORPORATION. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

#ifndef OP128_H_
#define OP128_H_

#include <type_traits>

inline __device__ void load128(const uint64_t* ptr, uint64_t &v0, uint64_t &v1) {
  v0 = __builtin_nontemporal_load(ptr);
  v1 = __builtin_nontemporal_load(ptr+1);
}

inline __device__ void store128(uint64_t* ptr, uint64_t v0, uint64_t v1) {
  __builtin_nontemporal_store(v0, ptr);
  __builtin_nontemporal_store(v1, ptr+1);
}

inline __device__ uint64_t* shmemCvtPtr(volatile uint64_t* shmemGenericPtr) {
  return (uint64_t*)shmemGenericPtr;
}

inline __device__ void loadShmem128(uint64_t* shmemAsmPtr, uint64_t &v0, uint64_t &v1) {
  v0 = *(shmemAsmPtr);
  v1 = *(shmemAsmPtr+1);
}

inline __device__ void storeShmem128(uint64_t* shmemAsmPtr, uint64_t v0, uint64_t v1) {
  *(shmemAsmPtr) = v0;
  *(shmemAsmPtr+1) = v1;
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
      lo = __builtin_nontemporal_load(ptr4+e+0);
      hi = __builtin_nontemporal_load(ptr4+e+1);
      tmp4[e] = __funnelshift_r(lo, hi, 8*(int(reinterpret_cast<uintptr_t>(ptr))%4));
    }
  }
  else if(sizeof(T) == 4) {
    #pragma unroll
    for(int e=0; e < 4; e++)
      tmp4[e] = __builtin_nontemporal_load(ptr+e);
  }
  else /*sizeof(T)==8*/ {
    #pragma unroll
    for(int e=0; e < 2; e++)
      tmp8[e] = __builtin_nontemporal_load(ptr+e);
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
#if !defined(USE_INDIRECT_FUNCTION_CALL) || defined(__gfx940__) || defined(__gfx941__) || defined(__gfx942__)
  inline __device__ BytePack<16>& operator=(BytePack<16> other) {
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

template<int Size> __device__ BytePack<Size> ld_global(uintptr_t addr);
template<int Size> __device__ BytePack<Size> ld_volatile_global(uintptr_t addr);
//template<int Size> __device__ BytePack<Size> ld_shared(uint32_t addr);
//template<int Size> __device__ BytePack<Size> ld_volatile_shared(uint32_t addr);
template<int Size> __device__ void st_global(uintptr_t addr, BytePack<Size> value);
//template<int Size> __device__ void st_shared(uint32_t addr, BytePack<Size> value);

template<> __device__ __forceinline__ BytePack<0> ld_global<0>(uintptr_t addr) { return {}; }
template<> __device__ __forceinline__ BytePack<0> ld_volatile_global<0>(uintptr_t addr) { return {}; }
//template<> __device__ __forceinline__ BytePack<0> ld_shared<0>(uint32_t addr) { return {}; }
//template<> __device__ __forceinline__ BytePack<0> ld_volatile_shared<0>(uint32_t addr) { return {}; }
template<> __device__ __forceinline__ void st_global<0>(uintptr_t addr, BytePack<0> value) {}
//template<> __device__ __forceinline__ void st_shared<0>(uint32_t addr, BytePack<0> value) {}

// Used to define implementations for above prototypes.
#define DEFINE_ld_st(bytes, data_cxx_ty, data_ptx_ty, data_reg_ty, space, addr_cxx_ty, addr_reg_ty) \
  template<> \
  __device__ __forceinline__ BytePack<bytes> ld_##space<bytes>(addr_cxx_ty addr) { \
    data_cxx_ty tmp; \
    tmp = *((data_cxx_ty *)addr); \
    BytePack<bytes> ans; \
    ans.native = tmp; \
    return ans; \
  } \
  template<> \
  __device__ __forceinline__ BytePack<bytes> ld_volatile_##space<bytes>(addr_cxx_ty addr) { \
    data_cxx_ty tmp; \
    tmp =  __builtin_nontemporal_load((data_cxx_ty *)addr); \
    BytePack<bytes> ans; \
    ans.native = tmp; \
    return ans; \
  } \
  template<> \
  __device__ __forceinline__ void st_##space<bytes>(addr_cxx_ty addr, BytePack<bytes> value) { \
    data_cxx_ty tmp = value.native; \
    *((data_cxx_ty *)addr) = tmp; \
  }
// Single-byte types use 4-byte registers since there is no 1-byte register
// character for asm blocks. See https://docs.nvidia.com/cuda/inline-ptx-assembly/index.html#constraints
DEFINE_ld_st(1, uint8_t, b8, r, global, uintptr_t, l)
//DEFINE_ld_st(1, uint32_t, b8, r, shared, uint32_t, r)
DEFINE_ld_st(2, uint16_t, b16, h, global, uintptr_t, l)
//DEFINE_ld_st(2, uint16_t, b16, h, shared, uint32_t, r)
DEFINE_ld_st(4, uint32_t, b32, r, global, uintptr_t, l)
//DEFINE_ld_st(4, uint32_t, b32, r, shared, uint32_t, r)
DEFINE_ld_st(8, uint64_t, b64, l, global, uintptr_t, l)
//DEFINE_ld_st(8, uint64_t, b64, l, shared, uint32_t, r)
#undef DEFINE_ld_st

#define DEFINE_ld_st_16(space, addr_cxx_ty, addr_reg_ty) \
  template<> \
  __device__ __forceinline__ BytePack<16> ld_##space<16>(addr_cxx_ty addr) { \
    BytePack<16> ans; \
    ans.u64[0] = *((uint64_t*)addr); \
    ans.u64[1] = *((uint64_t*)addr+1); \
    return ans; \
  } \
  template<> \
  __device__ __forceinline__ BytePack<16> ld_volatile_##space<16>(addr_cxx_ty addr) { \
    BytePack<16> ans; \
    ans.u64[0] = __builtin_nontemporal_load((uint64_t*)addr); \
    ans.u64[1] = __builtin_nontemporal_load((uint64_t*)addr+1); \
    return ans; \
  } \
  template<> \
  __device__ __forceinline__ void st_##space<16>(addr_cxx_ty addr, BytePack<16> value) { \
    __builtin_nontemporal_store(value.u64[0], (uint64_t*)addr); \
    __builtin_nontemporal_store(value.u64[1], (uint64_t*)addr+1); \
  }
DEFINE_ld_st_16(global, uintptr_t, l)
//DEFINE_ld_st_16(shared, uint32_t, r)
#undef DEFINE_ld_st_16

////////////////////////////////////////////////////////////////////////////////
// Atomic load/store using c++ pointers.

__device__ __forceinline__ uint64_t ld_volatile_global(uint64_t *ptr) {
  uint64_t ans;
  ans = __builtin_nontemporal_load(ptr);
  return ans;
}
__device__ __forceinline__ uint64_t ld_relaxed_sys_global(uint64_t *ptr) {
  uint64_t ans;
  ans = __builtin_nontemporal_load(ptr);
  return ans;
}
__device__ __forceinline__ uint64_t ld_acquire_sys_global(uint64_t *ptr) {
  uint64_t ans;
  ans = __atomic_load_n(ptr ,__ATOMIC_SEQ_CST);
  return ans;
}

__device__ __forceinline__ void st_volatile_global(uint64_t *ptr, uint64_t val) {
  __builtin_nontemporal_store(val, ptr);
}
__device__ __forceinline__ void st_relaxed_sys_global(uint64_t *ptr, uint64_t val) {
  __builtin_nontemporal_store(val, ptr);
}
__device__ __forceinline__ void st_release_sys_global(uint64_t *ptr, uint64_t val) {
  __atomic_store_n(ptr, val, __ATOMIC_SEQ_CST);
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
__device__ __forceinline__ void multimem_st_global(uintptr_t addr, BytePack<Size> val);

#if __CUDA_ARCH__ >= 900 && CUDART_VERSION >= 12010
template<>
__device__ __forceinline__ void multimem_st_global<0>(uintptr_t addr, BytePack<0> val) {
  // nop
}
template<>
__device__ __forceinline__ void multimem_st_global<1>(uintptr_t addr, BytePack<1> val) {
  asm volatile("st.global.b8 [%0], %1;" :: "l"(addr), "r"((uint32_t)val.u8) : "memory");
}
template<>
__device__ __forceinline__ void multimem_st_global<2>(uintptr_t addr, BytePack<2> val) {
  asm volatile("st.global.b16 [%0], %1;" :: "l"(addr), "h"(val.u16) : "memory");
}
template<>
__device__ __forceinline__ void multimem_st_global<4>(uintptr_t addr, BytePack<4> val) {
  asm volatile("multimem.st.global.b32 [%0], %1;" :: "l"(addr), "r"(val.u32) : "memory");
}
template<>
__device__ __forceinline__ void multimem_st_global<8>(uintptr_t addr, BytePack<8> val) {
  asm volatile("multimem.st.global.b64 [%0], %1;" :: "l"(addr), "l"(val.u64) : "memory");
}
template<>
__device__ __forceinline__ void multimem_st_global<16>(uintptr_t addr, BytePack<16> val) {
  asm volatile("multimem.st.global.v4.f32 [%0], {%1,%2,%3,%4};"
    :: "l"(addr), "r"(val.u32[0]), "r"(val.u32[1]), "r"(val.u32[2]), "r"(val.u32[3])
    : "memory");
}
#else
template<int Size>
__device__ __forceinline__ void multimem_st_global(uintptr_t addr, BytePack<Size> val) {
  // nop
}
#endif

#if __CUDA_ARCH__ >= 900 && CUDART_VERSION >= 12010
// Warp-uniform memory copy from shared address (not generic) to global memory.
// The number of bytes copied is `min(MaxBytes, nBytesAhead)`, a negative value
// is interpeted as zero. EltSize is the guaranteed alignment of the addresses and sizes.
template<int EltSize, int MaxBytes, bool Multimem, typename IntBytes>
__device__ __forceinline__ void copyGlobalShared_WarpUnrolled(
    int lane, uintptr_t dstAddr, uint32_t srcAddr, IntBytes nBytesAhead
  ) {
  static_assert(std::is_signed<IntBytes>::value, "`IntBytes` must be a signed integral type.");
  int nBytes = min(nBytesAhead, (IntBytes)MaxBytes);
  int nFrontBytes = min(nBytes, (16 - int(dstAddr%16))%16);
  int nMiddleBytes = (nBytes-nFrontBytes) & -16;
  int nBackBytes = (nBytes-nFrontBytes) % 16;

  { int backLane = WARP_SIZE-1 - lane;
    bool hasFront = lane*EltSize < nFrontBytes;
    bool hasBack = backLane*EltSize < nBackBytes;
    int offset = hasFront ? lane*EltSize : (nBytes - (backLane+1)*EltSize);
    if (hasFront | hasBack) {
      BytePack<EltSize> tmp = ld_shared<EltSize>(srcAddr+offset);
      // Can't use multimem_st since it doesn't support EltSize==2
      st_global<EltSize>(dstAddr+offset, tmp);
    }
  }

  srcAddr += nFrontBytes;
  int srcMisalign = EltSize < 4 ? (srcAddr%4) : 0;
  srcAddr += -srcMisalign + lane*16;
  dstAddr += nFrontBytes + lane*16;
  nMiddleBytes -= lane*16;
  #pragma unroll
  for (int u=0; u < divUp(MaxBytes, WARP_SIZE*16); u++) {
    if (nMiddleBytes <= 0) break;
    union {
      BytePack<4> b4[4];
      BytePack<16> b16;
    };
    b4[0] = ld_shared<4>(srcAddr + 0*4);
    b4[1] = ld_shared<4>(srcAddr + 1*4);
    b4[2] = ld_shared<4>(srcAddr + 2*4);
    b4[3] = ld_shared<4>(srcAddr + 3*4);
    if (srcMisalign != 0) {
      BytePack<4> b4_4 = ld_shared<4>(srcAddr + 4*4);
      b4[0].u32 = __funnelshift_r(b4[0].u32, b4[1].u32, srcMisalign*8);
      b4[1].u32 = __funnelshift_r(b4[1].u32, b4[2].u32, srcMisalign*8);
      b4[2].u32 = __funnelshift_r(b4[2].u32, b4[3].u32, srcMisalign*8);
      b4[3].u32 = __funnelshift_r(b4[3].u32, b4_4.u32, srcMisalign*8);
    }
    if (Multimem) multimem_st_global<16>(dstAddr, b16);
    else          st_global<16>(dstAddr, b16);

    srcAddr += WARP_SIZE*16;
    dstAddr += WARP_SIZE*16;
    nMiddleBytes -= WARP_SIZE*16;
  }
}
#else
template<int EltSize, int MaxBytes, bool Multimem, typename IntBytes>
__device__ __forceinline__ void copyGlobalShared_WarpUnrolled(
    int lane, uintptr_t dstAddr, uint32_t srcAddr, IntBytes nBytesAhead
  ) {
  // nop
}
#endif

#endif
