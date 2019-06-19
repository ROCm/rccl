/*************************************************************************
 * Copyright (c) 2016-2018, NVIDIA CORPORATION. All rights reserved.
 * Modifications Copyright (c) 2019 Advanced Micro Devices, Inc. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

#ifndef NCCL_PRIMITIVES_H_
#define NCCL_PRIMITIVES_H_

#include <type_traits>
#include "reduce_kernel.h" // for reduction funcs


/* Defines primitive operations: Copy, Reduce, DoubleCopy, and ReduceCopy.
 *
 * In order to reduce the reptetion of template arguments, the operations
 * are bundled as static methods of the Primitives class.
 *
 * Each primitive operation copies/reduces a contiguous buffer and syncs
 * an optional set of flags against a sub-step counter. The sync value is
 * based on the step parameter. Sync flags must be of type WaitFlag or
 * PostFlag. The primitive routines wait for all WaitFlag args to attain
 * at least a value of SUBSTEPS*(step-1)+substep+1 (i.e. completion of
 * corresponding substep by previous step) before executing the transfer.
 * After each substep is transfered, all PostFlag arguments get updated to
 * the value SUBSTEPS*step+substep+1.
 */


class WaitFlag {
  volatile uint64_t * const flag;
  const int shift;
 public:
  __device__
  WaitFlag(volatile uint64_t * const flag, const int shift) : flag(flag), shift(shift) { }
  __device__
  void wait(uint64_t val) { while ((LOAD(flag) + shift) < val) /*SPIN*/; }
};


class PostFlag {
  volatile uint64_t * const flag;
  const int shift;
  volatile int * const fifo;
  const int fifo_size;
  uint32_t * hdp_reg;
 public:
  __device__
  PostFlag(volatile uint64_t* const flag, const int shift, volatile int* const fifo, const int fifo_size, uint32_t* hdp_reg = NULL)
    : flag(flag), shift(shift), fifo(fifo), fifo_size(fifo_size), hdp_reg(hdp_reg) { }
  // remote writes can be reordered if we don't do s_waitcnt 0 + store to HDP between the data and flag
  __device__
  void post(uint64_t val) { if (hdp_reg != NULL) STORE(hdp_reg, 0x1); STORE(flag, (val - shift)); }
  __device__
  void postSize(uint64_t step, int size) { if (fifo != NULL) STORE(fifo + step%fifo_size, size); };
};


// Helper to check if any argument is of type T.
// e.g. AnyAre<WaitFlag>(Flag1, Flag2, ...)
template<typename T> __device__
bool AnyAre() { return false; }

template<typename T, typename FIRST_T, typename... TAIL_Ts>
__device__
bool AnyAre(FIRST_T first, TAIL_Ts... tail) {
  return std::is_same<T, FIRST_T>::value || AnyAre<T>(tail...);
}


// Wait on all WaitFlags, ignore PostFlags
__device__
static void WaitOnFlags(uint64_t val) { }

template <typename... TAIL_Ts> __device__
static void WaitOnFlags(uint64_t val, WaitFlag flag, TAIL_Ts... tail) {
  flag.wait(val);
  WaitOnFlags(val, tail...);
}

template <typename... TAIL_Ts> __device__
static void WaitOnFlags(uint64_t val, PostFlag, TAIL_Ts... tail) {
  WaitOnFlags(val, tail...);
}


// Post all PostFlags, ignore WaitFlags
__device__
static void PostToFlags(uint64_t val) { }

template <typename... TAIL_Ts> __device__
static void PostToFlags(uint64_t val, WaitFlag flag, TAIL_Ts... tail) {
  PostToFlags(val, tail...);
}

template <typename... TAIL_Ts> __device__
static void PostToFlags(uint64_t val, PostFlag flag, TAIL_Ts... tail) {
  flag.post(val);
  PostToFlags(val, tail...);
}


// Post sizes for PostFlags, ignore WaitFlags
__device__
static void PostSizeToFlags(uint64_t step, int size) { }

template <typename... TAIL_Ts> __device__
static void PostSizeToFlags(uint64_t step, int size, WaitFlag flag, TAIL_Ts... tail) {
  PostSizeToFlags(step, size, tail...);
}

template <typename... TAIL_Ts> __device__
static void PostSizeToFlags(uint64_t step, int size, PostFlag flag, TAIL_Ts... tail) {
  flag.postSize(step, size);
  PostSizeToFlags(step, size, tail...);
}


// Create pointer arithmetic syntax that doesn't break for std::nullptr_t
template <typename Tptr> __device__
static Tptr ptradd(Tptr ptr, int i) {
  return ptr + i;
}

__device__
static std::nullptr_t ptradd(std::nullptr_t ptr, int i) {
  return nullptr;
}

// use different unroll numbers for all primitives for best throughput
#define COPY_UNROLL       4
#define REDUCE_UNROLL     2
#define DOUBLECOPY_UNROLL 2
#define REDUCECOPY_UNROLL 2

// Implementation of primitive types
template <int, int SUBSTEPS, typename T, typename REDOP=FuncSum<T> >
class Primitives {
 private:
  template <int UNROLL,
      typename SRC2_T, // either T* or std::nullptr_t
      typename DST2_T, // either T* or std::nullptr_t
      typename... SYNC_Ts> // either WaitFunc or PostFunc
  static __device__ __attribute__((noinline)) void
  GenericOp(const int tid, const int nthreads,
      const T*     src1,
      const SRC2_T src2,
      T*     dst1,
      DST2_T dst2,
      int len, int maxoffset, uint64_t step, SYNC_Ts... flags) {

    enum { noSrc2 = std::is_same<SRC2_T, std::nullptr_t>::value };
    enum { noDst2 = std::is_same<DST2_T, std::nullptr_t>::value };
    static_assert(noSrc2 || std::is_same<SRC2_T, const T*>::value,
        "src2 must be of type T* or std::nullptr_t");
    static_assert(noDst2 || std::is_same<DST2_T, T*>::value,
        "dst2 must be of type T* or std::nullptr_t");

    using OpType = typename std::conditional<noSrc2, FuncSum<T>, REDOP>::type;

    int sliceSize = len / SUBSTEPS;
    int sliceOffset = 0;

#pragma unroll 1
    for (int sub=0; sub<SUBSTEPS; ++sub) {
      int realSize = max(0, min(sliceSize, maxoffset-sliceOffset));
      if (tid < nthreads) {
        if (AnyAre<WaitFlag>(flags...)) {
          if (tid == 0) {
            WaitOnFlags(SUBSTEPS*step + sub + 1, flags...);
          }
#if defined(__HIP_PLATFORM_HCC__) || defined(__HCC__)
          __syncthreads();
#else
          asm volatile ("bar.sync 1, %0;" :: "r"(nthreads));
#endif
        }
        ReduceOrCopy
        <
        UNROLL,
        OpType,
        T,
        !std::is_same<DST2_T, std::nullptr_t>::value, // HAS_DEST1
        !std::is_same<SRC2_T, std::nullptr_t>::value  // HAS_SRC1
        >
        (
            tid, nthreads,
            ptradd(dst1, sliceOffset),
            ptradd(dst2, sliceOffset),
            ptradd(src1, sliceOffset),
            ptradd(src2, sliceOffset),
            realSize
        );
        if (AnyAre<PostFlag>(flags...)) {
          __syncthreads();
          if(tid == 0)
            PostSizeToFlags(SUBSTEPS*step+sub, realSize*sizeof(T), flags...);
          __threadfence_system();
          if(tid == 0)
            PostToFlags(SUBSTEPS*step + sub + 1, flags...);
        }
      }
      sliceOffset += sliceSize;
    }
  }

 public:
  template <typename... SYNC_Ts>
  static __device__ void
  Copy(const int tid, const int nthreads, const T* src, T* dst,
      int len, int maxOffset, uint64_t step, SYNC_Ts... flags) {
    GenericOp<COPY_UNROLL>(tid, nthreads, src, nullptr, dst, nullptr, len, maxOffset, step, flags...);
  }

  template <typename... SYNC_Ts>
  static __device__ void
  DoubleCopy(const int tid, const int nthreads, const T* src, T* dst1, T* dst2,
      int len, int maxOffset, uint64_t step, SYNC_Ts... flags) {
    GenericOp<DOUBLECOPY_UNROLL>(tid, nthreads, src, nullptr, dst1, dst2, len, maxOffset, step, flags...);
  }

  template <typename... SYNC_Ts>
  static __device__ void
  Reduce(const int tid, const int nthreads, const T* src1, const T* src2, T* dst,
      int len, int maxOffset, uint64_t step, SYNC_Ts... flags) {
    GenericOp<REDUCE_UNROLL>(tid, nthreads, src1, src2, dst, nullptr, len, maxOffset, step, flags...);
  }

  template <typename... SYNC_Ts>
  static __device__ void
  ReduceCopy(const int tid, const int nthreads, const T* src1, const T* src2, T* dst1, T* dst2,
      int len, int maxOffset, uint64_t step, SYNC_Ts... flags) {
    GenericOp<REDUCECOPY_UNROLL>(tid, nthreads, src1, src2, dst1, dst2, len, maxOffset, step, flags...);
  }
};

#endif // end include guard
