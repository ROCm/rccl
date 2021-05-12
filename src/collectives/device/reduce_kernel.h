/*************************************************************************
 * Copyright (c) 2015-2019, NVIDIA CORPORATION. All rights reserved.
 * Modifications Copyright (c) 2019-2020 Advanced Micro Devices, Inc. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/


#ifndef NCCL_REDUCE_KERNEL_H_
#define NCCL_REDUCE_KERNEL_H_

#include "common_kernel.h"
#include <limits>

template<typename T>
struct FuncNull {
  __device__ T operator()(const T x, const T y) const {
    return 0;
  }
};

#if defined(__HIP_PLATFORM_HCC__) || defined(__HCC__) || defined(__HIPCC__)

//we really don't need any specializations and we don't need
//to break things into uint32_t
template<typename T>
__device__ inline T ncclMinFunc(T x, T y) { return y < x ? y : x; }

template<typename T>
__device__ inline T ncclMaxFunc(T x, T y) { return y < x ? x : y; }

template<typename T>
class FuncBase {
protected:
  static constexpr auto n = sizeof(PackType) / sizeof(T);

  union Cvt {
    using Vec = T __attribute__((ext_vector_type(n)));

    PackType data;
    Vec vec;

    static_assert(sizeof(Vec) == sizeof(data), "Vec must be the same size of data.");
  };
};

template<>
class FuncBase<half> {
protected:
  static constexpr auto n = sizeof(PackType) / sizeof(_Float16);
  union Cvt {
    using Vec = _Float16 __attribute__((ext_vector_type(n)));

    PackType data;
    Vec vec;

    static_assert(sizeof(Vec) == sizeof(data), "Vec must be the same size of data.");
  };
};

template<typename T>
struct FuncSum : private FuncBase<T> {
  __device__ PackType operator()(PackType x, PackType y) const
  {
    using Cvt = typename FuncBase<T>::Cvt;

    Cvt tmp_x{x};
    tmp_x.vec += Cvt{y}.vec;

    return tmp_x.data;
  }
  template<typename U = T, typename std::enable_if<!std::is_same<T, U>{}>* = nullptr>
  __device__ T operator()(const T x, const T y) const {
    return x + y;
  }
};

template<typename T>
struct FuncProd : private FuncBase<T> {
  __device__ PackType operator()(PackType x, PackType y) const
  {
    using Cvt = typename FuncBase<T>::Cvt;

    Cvt tmp_x{x};
    tmp_x.vec *= Cvt{y}.vec;

    return tmp_x.data;
  }
  template<typename U = T, typename std::enable_if<!std::is_same<T, U>{}>* = nullptr>
  __device__ T operator()(const T x, const T y) const {
    return x * y;
  }
};

template<typename T>
struct FuncMax : private FuncBase<T> {
  __device__ PackType operator()(PackType x, PackType y) const
  {
    using Cvt = typename FuncBase<T>::Cvt;

    Cvt tmp_x{x};
    Cvt tmp_y{y};

    for (auto i = 0u; i != FuncBase<T>::n; ++i) {
        tmp_x.vec[i] = ncclMaxFunc(tmp_x.vec[i], tmp_y.vec[i]);
    }

    return tmp_x.data;
  }
  template<typename U = T, typename std::enable_if<!std::is_same<T, U>{}>* = nullptr>
  __device__ T operator()(const T x, const T y) const {
    return (x < y) ? y : x;
  }
};

template<typename T>
struct FuncMin : private FuncBase<T> {
  __device__ PackType operator()(PackType x, PackType y) const
  {
    using Cvt = typename FuncBase<T>::Cvt;

    Cvt tmp_x{x};
    Cvt tmp_y{y};

    for (auto i = 0u; i != FuncBase<T>::n; ++i) {
        tmp_x.vec[i] = ncclMinFunc(tmp_x.vec[i], tmp_y.vec[i]);
    }

    return tmp_x.data;
  }
  template<typename U = T, typename std::enable_if<!std::is_same<T, U>{}>* = nullptr>
  __device__ T operator()(const T x, const T y) const {
    return (x < y) ? x : y;
  }
};

template<>
struct FuncSum<rccl_bfloat16> {
  static constexpr auto n = sizeof(PackType) / sizeof(rccl_bfloat16);
  __device__ PackType operator()(PackType x, PackType y) const
  {
    union converter { PackType storage; rccl_bfloat16 vec[n]; };
    static_assert(sizeof(PackType) == sizeof(converter), "PackType must be the same size of converter.");
    converter cx, cy, cr;
    cx.storage = x;
    cy.storage = y;
    for (auto i = 0u; i != n; ++i) {
      cr.vec[i] = cx.vec[i] + cy.vec[i];
    }
    return cr.storage;
  }
  __device__ rccl_bfloat16 operator()(const rccl_bfloat16 x, const rccl_bfloat16 y) const {
    return x + y;
  }
};

template<>
struct FuncProd<rccl_bfloat16> {
  static constexpr auto n = sizeof(PackType) / sizeof(rccl_bfloat16);
  __device__ PackType operator()(PackType x, PackType y) const
  {
    union converter { PackType storage; rccl_bfloat16 vec[n]; };
    static_assert(sizeof(PackType) == sizeof(converter), "PackType must be the same size of converter.");
    converter cx, cy, cr;
    cx.storage = x;
    cy.storage = y;
    for (auto i = 0u; i != n; ++i) {
      cr.vec[i] = cx.vec[i] * cy.vec[i];
    }
    return cr.storage;
  }
  __device__ rccl_bfloat16 operator()(const rccl_bfloat16 x, const rccl_bfloat16 y) const {
    return x * y;
  }
};

template<>
struct FuncMax<rccl_bfloat16> {
  static constexpr auto n = sizeof(PackType) / sizeof(rccl_bfloat16);
  __device__ PackType operator()(PackType x, PackType y) const
  {
    union converter { PackType storage; rccl_bfloat16 vec[n]; };
    static_assert(sizeof(PackType) == sizeof(converter), "PackType must be the same size of converter.");
    converter cx, cy, cr;
    cx.storage = x;
    cy.storage = y;
    for (auto i = 0u; i != n; ++i) {
      cr.vec[i] = cx.vec[i] < cy.vec[i] ? cy.vec[i] : cx.vec[i];
    }
    return cr.storage;
  }
  __device__ rccl_bfloat16 operator()(const rccl_bfloat16 x, const rccl_bfloat16 y) const {
    return x < y ? y : x;
  }
};

template<>
struct FuncMin<rccl_bfloat16> {
  static constexpr auto n = sizeof(PackType) / sizeof(rccl_bfloat16);
  __device__ PackType operator()(PackType x, PackType y) const
  {
    union converter { PackType storage; rccl_bfloat16 vec[n]; };
    static_assert(sizeof(PackType) == sizeof(converter), "PackType must be the same size of converter.");
    converter cx, cy, cr;
    cx.storage = x;
    cy.storage = y;
    for (auto i = 0u; i != n; ++i) {
      cr.vec[i] = cx.vec[i] < cy.vec[i] ? cx.vec[i] : cy.vec[i];
    }
    return cr.storage;
  }
  __device__ rccl_bfloat16 operator()(const rccl_bfloat16 x, const rccl_bfloat16 y) const {
    return x < y ? x : y;
  }
};

#else

template<typename T>
struct FuncSum {
  __device__ T operator()(const T x, const T y) const {
    return x + y;
  }
};

template<typename T>
struct FuncProd {
  __device__ T operator()(const T x, const T y) const {
    return x * y;
  }
};

template<typename T>
struct FuncMax {
  __device__ T operator()(const T x, const T y) const {
    return (x < y) ? y : x;
  }
};

template<typename T>
struct FuncMin {
  __device__ T operator()(const T x, const T y) const {
    return (x < y) ? x : y;
  }
};

#define MASK0 0x00ff00ff
#define MASK1 0xff00ff00
static __device__ uint32_t addChar4(const uint32_t x, const uint32_t y) {
  /* This can be used both for signed and unsigned 8-bit addition */
  const uint32_t x0 = x & MASK0;
  const uint32_t x1 = x & MASK1;
  const uint32_t y0 = y & MASK0;
  const uint32_t y1 = y & MASK1;
  const uint32_t r0 = (x0+y0);
  const uint32_t r1 = (x1+y1);
  return (r0 & MASK0) | (r1 & MASK1);
}

template<>
struct FuncSum<int8_t> {
  __device__ uint32_t operator()(const uint32_t x, const uint32_t y) const {
#if (__CUDA_ARCH__ >= 300) && (__CUDA_ARCH__ < 500)
    int32_t rv, z=0;
    asm("vadd4.s32.s32.s32 %0, %1, %2, %3;" : "=r"(rv) : "r"(x), "r"(y), "r"(z));
    return rv;
#else
    return addChar4(x, y);
#endif
  }
  __device__ int8_t operator()(const int8_t x, const int8_t y) const {
    return x+y;
  }
};
template<>
struct FuncSum<uint8_t> {
  __device__ uint32_t operator()(const uint32_t x, const uint32_t y) const {
#if (__CUDA_ARCH__ >= 300) && (__CUDA_ARCH__ < 500)
    int32_t rv, z=0;
    asm("vadd4.u32.u32.u32 %0, %1, %2, %3;" : "=r"(rv) : "r"(x), "r"(y), "r"(z));
    return rv;
#else
    return addChar4(x, y);
#endif
  }
  __device__ uint8_t operator()(const uint8_t x, const uint8_t y) const {
    return x+y;
  }
};

static __device__ uint32_t mulChar4(const uint32_t x, const uint32_t y) {
  /* This can be used both for signed and unsigned 8-bit multiplication */
  union converter { uint32_t storage; char4 a; };
  converter cx, cy, cr;
  cx.storage = x;
  cy.storage = y;
  cr.a.x = cx.a.x * cy.a.x;
  cr.a.y = cx.a.y * cy.a.y;
  cr.a.z = cx.a.z * cy.a.z;
  cr.a.w = cx.a.w * cy.a.w;
  return cr.storage;
}

template<>
struct FuncProd<int8_t> {
  __device__ uint32_t operator()(const uint32_t x, const uint32_t y) const {
    return mulChar4(x, y);
  }
  __device__ int8_t operator()(const int8_t x, const int8_t y) const {
    return x*y;
  }
};
template<>
struct FuncProd<uint8_t> {
  __device__ uint32_t operator()(const uint32_t x, const uint32_t y) const {
    return mulChar4(x, y);
  }
  __device__ uint8_t operator()(const uint8_t x, const uint8_t y) const {
    return x*y;
  }
};

template<>
struct FuncMax<int8_t> {
  union converter { uint32_t storage; char4 a; };
  __device__ uint32_t operator()(const uint32_t x, const uint32_t y) const {
#if (__CUDA_ARCH__ >= 300) && (__CUDA_ARCH__ < 500)
    int32_t rv, z=0;
    asm("vmax4.s32.s32.s32 %0, %1, %2, %3;" : "=r"(rv) : "r"(x), "r"(y), "r"(z));
    return rv;
#else
    converter cx, cy, cr;
    cx.storage = x;
    cy.storage = y;
    cr.a.x = max(cx.a.x, cy.a.x);
    cr.a.y = max(cx.a.y, cy.a.y);
    cr.a.z = max(cx.a.z, cy.a.z);
    cr.a.w = max(cx.a.w, cy.a.w);
    return cr.storage;
#endif
  }
  __device__ int8_t operator()(const int8_t x, const int8_t y) const {
    return (x>y) ? x : y;
  }
};
template<>
struct FuncMax<uint8_t> {
  union converter { uint32_t storage; uchar4 a; };
  __device__ uint32_t operator()(const uint32_t x, const uint32_t y) const {
#if (__CUDA_ARCH__ >= 300) && (__CUDA_ARCH__ < 500)
    int32_t rv, z=0;
    asm("vmax4.u32.u32.u32 %0, %1, %2, %3;" : "=r"(rv) : "r"(x), "r"(y), "r"(z));
    return rv;
#else
    converter cx, cy, cr;
    cx.storage = x;
    cy.storage = y;
    cr.a.x = max(cx.a.x, cy.a.x);
    cr.a.y = max(cx.a.y, cy.a.y);
    cr.a.z = max(cx.a.z, cy.a.z);
    cr.a.w = max(cx.a.w, cy.a.w);
    return cr.storage;
#endif
  }
  __device__ uint8_t operator()(const uint8_t x, const uint8_t y) const {
    return (x>y) ? x : y;
  }
};

template<>
struct FuncMin<int8_t> {
  union converter { uint32_t storage; char4 a; };
  __device__ uint32_t operator()(const uint32_t x, const uint32_t y) const {
#if (__CUDA_ARCH__ >= 300) && (__CUDA_ARCH__ < 500)
    int32_t rv, z=0;
    asm("vmin4.s32.s32.s32 %0, %1, %2, %3;" : "=r"(rv) : "r"(x), "r"(y), "r"(z));
    return rv;
#else
    converter cx, cy, cr;
    cx.storage = x;
    cy.storage = y;
    cr.a.x = min(cx.a.x, cy.a.x);
    cr.a.y = min(cx.a.y, cy.a.y);
    cr.a.z = min(cx.a.z, cy.a.z);
    cr.a.w = min(cx.a.w, cy.a.w);
    return cr.storage;
#endif
  }
  __device__ int8_t operator()(const int8_t x, const int8_t y) const {
    return (x<y) ? x : y;
  }
};
template<>
struct FuncMin<uint8_t> {
  union converter { uint32_t storage; uchar4 a; };
  __device__ uint32_t operator()(const uint32_t x, const uint32_t y) const {
#if (__CUDA_ARCH__ >= 300) && (__CUDA_ARCH__ < 500)
    int32_t rv, z=0;
    asm("vmin4.u32.u32.u32 %0, %1, %2, %3;" : "=r"(rv) : "r"(x), "r"(y), "r"(z));
    return rv;
#else
    converter cx, cy, cr;
    cx.storage = x;
    cy.storage = y;
    cr.a.x = min(cx.a.x, cy.a.x);
    cr.a.y = min(cx.a.y, cy.a.y);
    cr.a.z = min(cx.a.z, cy.a.z);
    cr.a.w = min(cx.a.w, cy.a.w);
    return cr.storage;
#endif
  }
  __device__ uint8_t operator()(const uint8_t x, const uint8_t y) const {
    return (x<y) ? x : y;
  }
};

template<>
struct FuncSum<half> {
  __device__ half2 operator()(const half2 x, const half2 y) const {
#if __CUDA_ARCH__ >= 530 && __CUDA_ARCH__ != 610
    return __hadd2(x, y);
#else
    float2 fx, fy, fr;
    fx = __half22float2(x);
    fy = __half22float2(y);
    fr.x = fx.x + fy.x;
    fr.y = fx.y + fy.y;
    return __float22half2_rn(fr);
#endif
  }
  __device__ half operator()(const half x, const half y) const {
#if __CUDA_ARCH__ >= 530 && __CUDA_ARCH__ != 610
    return __hadd(x, y);
#else
    return __float2half( __half2float(x) + __half2float(y) );
#endif
  }
};

template<>
struct FuncProd<half> {
  __device__ half2 operator()(const half2 x, const half2 y) const {
#if __CUDA_ARCH__ >= 530 && __CUDA_ARCH__ != 610
    return __hmul2(x, y);
#else
    float2 fx, fy, fr;
    fx = __half22float2(x);
    fy = __half22float2(y);
    fr.x = fx.x * fy.x;
    fr.y = fx.y * fy.y;
    return __float22half2_rn(fr);
#endif
  }
  __device__ half operator()(const half x, const half y) const {
#if __CUDA_ARCH__ >= 530 && __CUDA_ARCH__ != 610
    return __hmul(x, y);
#else
    return __float2half( __half2float(x) * __half2float(y) );
#endif
  }
};

template<>
struct FuncMax<half> {
  __device__ half2 operator()(const half2 x, const half2 y) const {
    float2 fx, fy, fr;
    fx = __half22float2(x);
    fy = __half22float2(y);
    fr.x = fmaxf(fx.x, fy.x);
    fr.y = fmaxf(fx.y, fy.y);
    return __float22half2_rn(fr);
  }
  __device__ half operator()(const half x, const half y) const {
    float fx, fy, fm;
    fx = __half2float(x);
    fy = __half2float(y);
    fm = fmaxf(fx, fy);
    return __float2half(fm);
  }
};

template<>
struct FuncMin<half> {
  __device__ half2 operator()(const half2 x, const half2 y) const {
    float2 fx, fy, fr;
    fx = __half22float2(x);
    fy = __half22float2(y);
    fr.x = fminf(fx.x, fy.x);
    fr.y = fminf(fx.y, fy.y);
    return __float22half2_rn(fr);
  }
  __device__ half operator()(const half x, const half y) const {
    float fx, fy, fm;
    fx = __half2float(x);
    fy = __half2float(y);
    fm = fminf(fx, fy);
    return __float2half(fm);
  }
};

#endif // defined(__HIP_PLATFORM_HCC__) || defined(__HCC__) || defined(__HIPCC__)

#endif // REDUCE_KERNEL_H_
