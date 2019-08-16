/*************************************************************************
 * Copyright (c) 2015, NVIDIA CORPORATION. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/


#ifndef COPY_KERNEL_H_
#define COPY_KERNEL_H_
#include <cstdio>
#include <cstdint>

// Define min for ssize_t
static __device__ int min(int a, ssize_t b) { return (a < b) ? a : b; }

typedef uint64_t PackType;

template<class FUNC, typename T>
struct MULTI {
    __device__ PackType operator()(const PackType x, const PackType y) const
    {
        return FUNC()(x, y);
    }
};

#define ALIGNUP(x, a)   ((((x)-1) & ~((a)-1)) + (a))

template<typename T>
__device__ inline volatile T* AlignUp(volatile T * ptr, size_t align) {
  size_t ptrval = reinterpret_cast<size_t>(ptr);
  return reinterpret_cast<volatile T*>(ALIGNUP(ptrval, align));
}

template<typename T> inline __device__
T vFetch(const volatile T* ptr) {
  return *ptr;
}

template<typename T> inline __device__
void vStore(volatile T* ptr, const T val) {
  *ptr = val;
}

template<class FUNC, typename T, bool TWO_INPUTS, bool TWO_OUTPUTS>
__attribute__((noinline))
__device__ inline void ReduceCopy(
    const int tid, const int nthreads,
    const volatile T * __restrict__ const src0,
    const volatile T * __restrict__ const src1,
    volatile T * __restrict__ const dest0,
    volatile T * __restrict__ const dest1, const int N) {
  for (int idx = tid; idx < N; idx += nthreads) {
    T val = vFetch(src0+idx);
    if (TWO_INPUTS) {
      val = FUNC()(val, vFetch(src1+idx));
    }
    vStore(dest0+idx, val);
    if (TWO_OUTPUTS) {
      vStore(dest1+idx, val);
    }
  }
}

template<typename T>
struct FuncPassA {
  __device__ T operator()(const T x, const T y) const {
    return x;
  }
};

template<typename T>
struct FuncSum {
  __device__ T operator()(const T x, const T y) const {
    return x + y;
  }
};

template<class FUNC>
struct MULTI<FUNC, float> {
  static_assert(sizeof(PackType) == 2 * sizeof(float),
      "PackType must be twice the size of float.");
  union converter {
    PackType storage;
    struct {
      float a, b;
    };
  };

  __device__ PackType operator()(const PackType x, const PackType y) const {
    converter cx, cy, cr;
    cx.storage = x;
    cy.storage = y;

    cr.a = FUNC()(cx.a, cy.a);
    cr.b = FUNC()(cx.b, cy.b);

    return cr.storage;
  }
};


typedef ulong2 Pack128;

template<class FUNC, typename T>
struct MULTI128 {
  __device__ void operator()(Pack128& x, Pack128& y) {
    x.x = MULTI<FUNC, T>()(x.x, y.x);
    x.y = MULTI<FUNC, T>()(x.y, y.y);
  }
};

inline __device__ void Fetch128(Pack128& v, const Pack128* p) {
  v.x = p->x;
  v.y = p->y;
}
inline __device__ void Store128(Pack128* p, Pack128& v) {
  p->x = v.x;
  p->y = v.y;
}

template<class FUNC, typename T, int MINSRCS, int MAXSRCS, int MINDSTS, int MAXDSTS>
__device__ void ReduceCopyMulti(const int tid, const int nthreads,
    int nsrcs, const T* srcs[MAXSRCS], int ndsts, T* dsts[MAXDSTS],
    const int offset, const int N) {
  for (int idx = offset+tid; idx < offset+N; idx += nthreads) {
    T val = vFetch(srcs[0]+idx);
    #pragma unroll
    for (int i=1; i<MINSRCS; i++) val = FUNC()(val, vFetch(srcs[i]+idx));
    #pragma unroll 1
    for (int i=MINSRCS; i<MAXSRCS && i<nsrcs; i++) val = FUNC()(val, vFetch(srcs[i]+idx));

    #pragma unroll
    for (int i=0; i<MINDSTS; i++) vStore(dsts[i]+idx, val);
    #pragma unroll 1
    for (int i=MINDSTS; i<MAXDSTS && i<ndsts; i++) vStore(dsts[i]+idx, val);
  }
}

#define WARP_SIZE 64

template<class FUNC, typename T, int UNROLL, int MINSRCS, int MAXSRCS, int MINDSTS, int MAXDSTS>
__device__ void ReduceCopy128bMulti( const int w, const int nw, const int t,
    int nsrcs, const T* s[MAXSRCS], int ndsts, T* d[MAXDSTS],
    const int elemOffset, const int Npack) {
  const int inc = nw * UNROLL * WARP_SIZE;
  int offset = w * UNROLL * WARP_SIZE + t;

  const Pack128* srcs[MAXSRCS];
  for (int i=0; i<MAXSRCS; i++) srcs[i] = ((const Pack128*)(s[i]+elemOffset))+offset;
  Pack128* dsts[MAXDSTS];
  for (int i=0; i<MAXDSTS; i++) dsts[i] = ((Pack128*)(d[i]+elemOffset))+offset;

  while (offset < Npack) {
    Pack128 vals[UNROLL];
    // Load and reduce
    for (int u = 0; u < UNROLL; ++u) Fetch128(vals[u], srcs[0]+u*WARP_SIZE);

    for (int i=1; i<MINSRCS; i++) {
      Pack128 vals2[UNROLL];
      for (int u = 0; u < UNROLL; ++u) Fetch128(vals2[u], srcs[i]+u*WARP_SIZE);
      for (int u = 0; u < UNROLL; ++u) MULTI128<FUNC, T>()(vals[u], vals2[u]);
    }
    #pragma unroll 1
    for (int i=MINSRCS; i<MAXSRCS && i<nsrcs; i++) {
      Pack128 vals2[UNROLL];
      for (int u = 0; u < UNROLL; ++u) Fetch128(vals2[u], srcs[i]+u*WARP_SIZE);
      for (int u = 0; u < UNROLL; ++u) MULTI128<FUNC, T>()(vals[u], vals2[u]);
    }

    // Store
    for (int i = 0; i < MINDSTS; i++) {
      for (int u = 0; u < UNROLL; ++u) Store128(dsts[i]+u*WARP_SIZE, vals[u]);
    }
    #pragma unroll 1
    for (int i=MINDSTS; i<MAXDSTS && i<ndsts; i++) {
      for (int u = 0; u < UNROLL; ++u) Store128(dsts[i]+u*WARP_SIZE, vals[u]);
    }
    for (int i=0; i<MAXSRCS; i++) srcs[i] += inc;
    for (int i=0; i<MAXDSTS; i++) dsts[i] += inc;
    offset += inc;
  }
}

template <typename T>
__device__ int ptrAlign128(T* ptr) { return (uint64_t)ptr % alignof(Pack128); }

// Try to limit consecutive load/stores to 8.
// Use UNROLL 8 when we have a single source and a single destination, 4 otherwise
#define AUTOUNROLL (UNROLL*(4/(MINDSTS+MINSRCS)))

template<int UNROLL, class FUNC, typename T, int MINSRCS, int MAXSRCS, int MINDSTS, int MAXDSTS>
__device__ void ReduceOrCopyMulti(const int tid, const int nthreads,
    int nsrcs, const T* srcs[MAXSRCS], int ndsts, T* dsts[MAXDSTS],
    int N) {
  int Nrem = N;
  if (Nrem <= 0) return;

  int alignDiff = 0;
  int align = ptrAlign128(srcs[0]);
  #pragma unroll
  for (int i=1; i<MINSRCS; i++) alignDiff |= (align ^ ptrAlign128(srcs[i]));
  for (int i=MINSRCS; i<MAXSRCS && i<nsrcs; i++) alignDiff |= (align ^ ptrAlign128(srcs[i]));
  #pragma unroll
  for (int i=0; i<MINDSTS; i++) alignDiff |= (align ^ ptrAlign128(dsts[i]));
  for (int i=MINDSTS; i<MAXDSTS && i<ndsts; i++) alignDiff |= (align ^ ptrAlign128(dsts[i]));

  int Npreamble = alignDiff ? Nrem :
    N < alignof(Pack128) ? N :
    (alignof(Pack128) - align) % alignof(Pack128);

  // stage 1: preamble: handle any elements up to the point of everything coming
  // into alignment
  if (Npreamble) {
    ReduceCopyMulti<FUNC, T, MINSRCS, MAXSRCS, MINDSTS, MAXDSTS>(tid, nthreads, nsrcs, srcs, ndsts, dsts, 0, Npreamble);
    Nrem -= Npreamble;
    if (Nrem == 0) return;
  }
  int offset = Npreamble;

  // stage 2: fast path: use 128b loads/stores to do the bulk of the work,
  // assuming the pointers we have are all 128-bit alignable.
  int w = tid / WARP_SIZE;       // Warp number
  int nw = nthreads / WARP_SIZE; // Number of warps
  int t = tid % WARP_SIZE;       // Thread (inside the warp)

  const int packFactor = sizeof(Pack128) / sizeof(T);

  // stage 2a: main loop
  int Npack2a = (Nrem / (packFactor * AUTOUNROLL * WARP_SIZE))
      * (AUTOUNROLL * WARP_SIZE); // round down
  int Nelem2a = Npack2a * packFactor;

  ReduceCopy128bMulti<FUNC, T, AUTOUNROLL, MINSRCS, MAXSRCS, MINDSTS, MAXDSTS>(w, nw, t, nsrcs, srcs, ndsts, dsts, offset, Npack2a);

  Nrem -= Nelem2a;
  if (Nrem == 0) return;
  offset += Nelem2a;

  // stage 2b: slightly less optimized for section when we don't have full
  // unrolling

  int Npack2b = Nrem / packFactor;
  int Nelem2b = Npack2b * packFactor;

  ReduceCopy128bMulti<FUNC, T, 1, MINSRCS, MAXSRCS, MINDSTS, MAXDSTS>(w, nw, t, nsrcs, srcs, ndsts, dsts, offset, Npack2b);

  Nrem -= Nelem2b;
  if (Nrem == 0) return;
  offset += Nelem2b;

  // stage 2c: tail
  ReduceCopyMulti<FUNC, T, MINSRCS, MAXSRCS, MINDSTS, MAXDSTS>(tid, nthreads, nsrcs, srcs, ndsts, dsts, offset, Nrem);
}

// Assumptions:
// - there is exactly 1 block
// - THREADS is the number of producer threads
// - this function is called by all producer threads
template<int UNROLL, int THREADS, typename T>
__device__ void Copy(volatile T * __restrict__ const dest,
    const volatile T * __restrict__ const src, const int N) {
  const T* srcs[2];
  T* dsts[2];
  srcs[0] = (const T*)src;
  dsts[0] = (T*)dest;
  ReduceOrCopyMulti<UNROLL, FuncPassA<T>, T, 1, 2, 1, 2>(threadIdx.x, THREADS,
      1, srcs, 1, dsts, N);
}

template<int UNROLL, int THREADS, typename T>
__device__ void DoubleCopy(volatile T * __restrict__ const dest0,
    volatile T * __restrict__ const dest1,
    const volatile T * __restrict__ const src, const int N) {
  const T* srcs[2];
  T* dsts[2];
  srcs[0] = (const T*)src;
  dsts[0] = (T*)dest0;
  dsts[1] = (T*)dest1;
  ReduceOrCopyMulti<UNROLL, FuncPassA<T>, T, 1, 2, 1, 2>(threadIdx.x, THREADS,
      1, srcs, 2, dsts, N);
}

template<int UNROLL, int THREADS, typename T>
__device__ void Reduce(volatile T * __restrict__ const dest,
    const volatile T * __restrict__ const src0,
    const volatile T * __restrict__ const src1, const int N) {
  const T* srcs[2];
  T* dsts[2];
  srcs[0] = (const T*)src0;
  srcs[1] = (const T*)src1;
  dsts[0] = (T*)dest;
  ReduceOrCopyMulti<UNROLL, FuncPassA<T>, T, 1, 2, 1, 2>(threadIdx.x, THREADS,
      2, srcs, 1, dsts, N);
}

template<int UNROLL, int THREADS, typename T>
__device__ void ReduceCopy(volatile T * __restrict__ const dest0,
    volatile T * __restrict__ const dest1,
    const volatile T * __restrict__ const src0,
    const volatile T * __restrict__ const src1, const int N) {
  const T* srcs[2];
  T* dsts[2];
  srcs[0] = (const T*)src0;
  srcs[1] = (const T*)src1;
  dsts[0] = (T*)dest0;
  dsts[1] = (T*)dest1;
  ReduceOrCopyMulti<UNROLL, FuncPassA<T>, T, 1, 2, 1, 2>(threadIdx.x, THREADS,
      2, srcs, 2, dsts, N);
}
#endif // COPY_KERNEL_H_
