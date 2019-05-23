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

typedef ulong2 Pack128;

template<class FUNC, typename T>
struct MULTI128 {
  __device__ void operator()(Pack128& x, Pack128& y) {
    x.x = MULTI<FUNC, T>()(x.x, y.x);
    x.y = MULTI<FUNC, T>()(x.y, y.y);
  }
};

inline __device__ void Fetch128(Pack128& v, Pack128* p) {
  v.x = p->x;
  v.y = p->y;
}

inline __device__ void Store128(Pack128* p, Pack128& v) {
  p->x = v.x;
  p->y = v.y;
}

#define WARP_SIZE 32
template<class FUNC, typename T, bool TWO_INPUTS, bool TWO_OUTPUTS, int UNROLL>
__attribute__((noinline))
__device__ inline void ReduceCopy128b( const int w, const int nw, const int t,
    Pack128 * src0, Pack128 * src1, Pack128 * dest0, Pack128 * dest1,
    const int N) {
  Pack128 t0[UNROLL];
  Pack128 t1[UNROLL];
  const Pack128* src0_end = src0 + N;
  const int inc = nw * UNROLL * WARP_SIZE;
  const int offset = w * UNROLL * WARP_SIZE + t;
  src0 += offset;  if (TWO_INPUTS)  src1 += offset;
  dest0 += offset; if (TWO_OUTPUTS) dest1 += offset;

  while (src0 < src0_end) {
#pragma unroll
    for (int u = 0; u < UNROLL; ++u) {
      Fetch128(t0[u], src0+u*WARP_SIZE);
      if (TWO_INPUTS) Fetch128(t1[u], src1+u*WARP_SIZE);
    }
#pragma unroll
    for (int u = 0; u < UNROLL; ++u) {
      if (TWO_INPUTS) MULTI128<FUNC, T>()(t0[u], t1[u]);
      Store128(dest0+u*WARP_SIZE, t0[u]);
      if (TWO_OUTPUTS) Store128(dest1+u*WARP_SIZE, t0[u]);
    }
    src0 += inc;  if (TWO_INPUTS)  src1 += inc;
    dest0 += inc; if (TWO_OUTPUTS) dest1 += inc;
  }
}

template<int UNROLL, class FUNC, typename T, bool HAS_DEST1, bool HAS_SRC1>
__attribute__((noinline))
__device__ inline void ReduceOrCopy(const int tid, const int nthreads,
    volatile T * __restrict__ dest0, volatile T * __restrict__ dest1,
    const volatile T * __restrict__ src0, const volatile T * __restrict__ src1,
    int N) {
  int Nrem = N;
  if (Nrem <= 0) return;

  int Npreamble = (Nrem<alignof(Pack128)) ? Nrem : AlignUp(dest0, alignof(Pack128)) - dest0;

  // stage 0: check if we'll be able to use the fast, 128-bit aligned path.
  // If not, we'll just use the slow preamble path for the whole operation
  bool alignable = (((AlignUp(src0,  alignof(Pack128)) == src0  + Npreamble)) &&
          (!HAS_DEST1 || (AlignUp(dest1, alignof(Pack128)) == dest1 + Npreamble)) &&
          (!HAS_SRC1  || (AlignUp(src1,  alignof(Pack128)) == src1  + Npreamble)));

  if (!alignable) {
    Npreamble = Nrem;
  }

  // stage 1: preamble: handle any elements up to the point of everything coming
  // into alignment
  ReduceCopy<FUNC, T, HAS_SRC1, HAS_DEST1>(tid, nthreads, src0, src1, dest0, dest1, Npreamble);

  Nrem -= Npreamble;
  if (Nrem == 0) return;

  dest0 += Npreamble; if (HAS_DEST1) { dest1 += Npreamble; }
  src0  += Npreamble; if (HAS_SRC1)  { src1  += Npreamble; }

  // stage 2: fast path: use 128b loads/stores to do the bulk of the work,
  // assuming the pointers we have are all 128-bit alignable.
  int w = tid / WARP_SIZE;       // Warp number
  int nw = nthreads / WARP_SIZE; // Number of warps
  int t = tid % WARP_SIZE;       // Thread (inside the warp)

  const int PackFactor = sizeof(Pack128) / sizeof(T);

  // stage 2a: main loop
  int Nalign2a = (Nrem / (PackFactor * UNROLL * nthreads))
      * (UNROLL * nthreads); // round down

  ReduceCopy128b<FUNC, T, HAS_SRC1, HAS_DEST1, UNROLL>(w, nw, t, (Pack128*)src0, (Pack128*)src1, (Pack128*)dest0, (Pack128*)dest1, Nalign2a);

  int Ndone2a = Nalign2a * PackFactor;
  Nrem -= Ndone2a;
  if (Nrem == 0) return;
  dest0 += Ndone2a; if (HAS_DEST1) { dest1 += Ndone2a; }
  src0  += Ndone2a; if (HAS_SRC1)  { src1  += Ndone2a; }

  // stage 2b: slightly less optimized for section when we don't have full
  // UNROLLs

  int Nalign2b = Nrem / PackFactor;

  ReduceCopy128b<FUNC, T, HAS_SRC1, HAS_DEST1, 1>(w, nw, t, (Pack128*)src0, (Pack128*)src1, (Pack128*)dest0, (Pack128*)dest1, Nalign2b);

  int Ndone2b = Nalign2b * PackFactor;
  Nrem -= Ndone2b;
  if (Nrem == 0) return;
  dest0 += Ndone2b; if (HAS_DEST1) { dest1 += Ndone2b; }
  src0  += Ndone2b; if (HAS_SRC1)  { src1  += Ndone2b; }

  // stage 2c: tail
  ReduceCopy<FUNC, T, HAS_SRC1, HAS_DEST1>(tid, nthreads, src0, src1, dest0, dest1, Nrem);
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

// Assumptions:
// - there is exactly 1 block
// - THREADS is the number of producer threads
// - this function is called by all producer threads
template<int UNROLL, int THREADS, typename T>
__device__ void Copy(volatile T * __restrict__ const dest,
    const volatile T * __restrict__ const src, const int N) {
  ReduceOrCopy<UNROLL, FuncPassA<T>, T, false, false>(threadIdx.x, THREADS,
      dest, nullptr, src, nullptr, N);
}

template<int UNROLL, int THREADS, typename T>
__device__ void DoubleCopy(volatile T * __restrict__ const dest0,
    volatile T * __restrict__ const dest1,
    const volatile T * __restrict__ const src, const int N) {
  ReduceOrCopy<UNROLL, FuncPassA<T>, T, true, false>(threadIdx.x, THREADS,
      dest0, dest1, src, nullptr, N);
}

template<int UNROLL, int THREADS, typename T>
__device__ void Reduce(volatile T * __restrict__ const dest,
    const volatile T * __restrict__ const src0,
    const volatile T * __restrict__ const src1, const int N) {
  ReduceOrCopy<UNROLL, FuncSum<T>, T, false, true>(threadIdx.x, THREADS,
      dest, nullptr, src0, src1, N);
}

template<int UNROLL, int THREADS, typename T>
__device__ void ReduceCopy(volatile T * __restrict__ const dest0,
    volatile T * __restrict__ const dest1,
    const volatile T * __restrict__ const src0,
    const volatile T * __restrict__ const src1, const int N) {
  ReduceOrCopy<UNROLL, FuncSum<T>, T, true, true>(threadIdx.x, THREADS,
      dest0, dest1, src0, src1, N);
}
#endif // COPY_KERNEL_H_
