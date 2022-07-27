/*************************************************************************
 * Copyright (c) 2016-2022, NVIDIA CORPORATION. All rights reserved.
 * Modifications Copyright (c) 2019-2022 Advanced Micro Devices, Inc. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

#ifndef NCCL_PRIMITIVES_H_
#define NCCL_PRIMITIVES_H_

#include <type_traits>
#include "reduce_kernel.h" // for reduction funcs
#include "common.h"

#define NCCL_SPINS_BEFORE_CHECK_ABORT 1000000

#define barrier_by_group() do { \
  if (nthreads == NCCL_MAX_NTHREADS) \
    __syncthreads(); \
  else { \
    const int w = threadIdx.x/WARP_SIZE; \
    const int wid = threadIdx.x%WARP_SIZE; \
    if (wid == 0) { \
      barrier_next[w] += nthreads/WARP_SIZE; \
      atomicAdd((unsigned long long *)barriers, 1); \
      while (atomicAdd((unsigned long long *)barriers, 0) < barrier_next[w]) __builtin_amdgcn_s_sleep(8); \
      __asm__ __volatile__("s_wakeup"); \
    } \
  } \
} while (0)

/* Protocol classes: ProtoSimple, ProtoLL, ProtoLL128
 * We use these as template args to the Primtiives class instead of integral
 * enums (e.g. NCCL_PROTO_LL) because for SIMPLE we need to carry a few extra
 * numbers. Also these types hold methods which let us compute numbers important
 * to how that protocol operates with a consistent interface so that our
 * algorithm code can operate protocol parametrically.
 */
template<int SlicePerChunk_1, int StepPerSlice_1, int Unroll_1 = COLL_UNROLL>
struct ProtoSimple {
  static constexpr int Id = NCCL_PROTO_SIMPLE;
  static constexpr int SlicePerChunk = SlicePerChunk_1;
  static constexpr int StepPerSlice = StepPerSlice_1;
  static constexpr int Unroll = Unroll_1;

  // Data bytes (no flags etc) in one step of the fifo queue.
  __device__ static int calcBytePerStep() {
    return ncclShmem->comm.buffSizes[NCCL_PROTO_SIMPLE]/NCCL_STEPS;
  }
  // Granularity of data bytes transferred per thread.
  __device__ static int calcBytePerGrain() {
    return sizeof(uint64_t); // Bogus value? Nobody queries this metric for simple.
  }
  // Group width is how many consecutive group values a subchannel occupies.
  static constexpr int MaxGroupWidth = 1;
  __device__ static int calcGroupWidth(bool send, int nthreads) {
    return 1;
  }
};

struct ProtoLL {
  static constexpr int Id = NCCL_PROTO_LL;

  // Data bytes (no flags etc) in one step of the fifo queue.
  __device__ static int calcBytePerStep() {
    return ncclShmem->comm.buffSizes[NCCL_PROTO_LL]/NCCL_STEPS/2; // Half is data
  }
  // Granularity of data bytes transferred per thread.
  __device__ static int calcBytePerGrain() {
    return sizeof(uint64_t); // One 16-byte line has 8-bytes of data
  }
  // Group width is how many consecutive group values a subchannel occupies.
  static constexpr int MaxGroupWidth = 1;
  __device__ static int calcGroupWidth(bool send, int nthreads) {
    return 1;
  }
};

struct ProtoLL128 {
  static constexpr int Id = NCCL_PROTO_LL128;

  // Data bytes (no flags etc) in one step of the fifo queue.
  __device__ static int calcBytePerStep() {
    return (ncclShmem->comm.buffSizes[NCCL_PROTO_LL128]/NCCL_STEPS)*NCCL_LL128_DATAELEMS/NCCL_LL128_LINEELEMS;
  }
  // Granularity of data bytes transferred per thread.
  __device__ static int calcBytePerGrain() {
    return NCCL_LL128_SHMEM_ELEMS_PER_THREAD*NCCL_LL128_DATAELEMS*sizeof(uint64_t)/NCCL_LL128_LINEELEMS;
  }
  // Group width is how many consecutive group values a subchannel occupies.
  static constexpr int MaxGroupWidth = 1;
  __device__ static int calcGroupWidth(bool send, int nthreads) {
    return 1;
  }
};

/* Fan (as in fan-in & fan-out) classes hold recv and send counts. The template
 * arguments are static bounds on the maximum values. Asymmetric counts are
 * independent. Symmetric is a static guarantee that nrecv==nsend, so it only
 * stores one value at runtime. This optimization save 32-bit register, but more
 * importantly uses fewer predicate registers when unrolling loops.
 */
template<int MaxRecv_, int MaxSend_>
struct FanAsymmetric {
  static constexpr int MaxRecv = MaxRecv_, MaxSend = MaxSend_;
  int nr, ns;
  FanAsymmetric() = default;
  __device__ FanAsymmetric(int nrecv, int nsend): nr(nrecv), ns(nsend) {
    // assert(nrecv <= MaxRecv && nsend <= MaxSend);
  }
  __device__ int nrecv() const { return MaxRecv ? nr : 0; }
  __device__ int nsend() const { return MaxSend ? ns : 0; }
};

template<int MaxArity>
struct FanSymmetric {
  static constexpr int MaxRecv = MaxArity, MaxSend = MaxArity;
  int n;
  FanSymmetric() = default;
  __device__ FanSymmetric(int nrecv, int nsend): n(nrecv) {
    // assert(nrecv == nsend && nrecv <= MaxArity);
  }
  __device__ int nrecv() const { return n; }
  __device__ int nsend() const { return n; }
};

// The primitives class. Specialized per protocol in the other headers.
template<typename T, typename RedOp, typename Fan, int Direct, typename Proto, int P2p>
class Primitives;

// Used by LL & LL128 to implement direct members in the naive way.
template<typename RealPrimitives>
struct PrimitivesWithoutDirect {
  __device__ void directSend(intptr_t inpIx, intptr_t remoteOutIx, int eltN) {
    static_cast<RealPrimitives*>(this)->send(inpIx, eltN);
  }
  __device__ void directSendFromOutput(intptr_t outIx, intptr_t remoteOutIx, int eltN) {
    static_cast<RealPrimitives*>(this)->sendFromOutput(outIx, eltN);
  }
  __device__ void directRecv(intptr_t outIx, int eltN) {
    static_cast<RealPrimitives*>(this)->recv(outIx, eltN, /*postOp=*/false);
  }
  __device__ void directCopySend(intptr_t inpIx, intptr_t outIx, intptr_t remoteOutIx, int eltN, bool postOp=false) {
    static_cast<RealPrimitives*>(this)->copySend(inpIx, outIx, eltN, postOp);
  }
  __device__ void directRecvCopySend(intptr_t outIx, intptr_t remoteOutIx, int eltN) {
    static_cast<RealPrimitives*>(this)->recvCopySend(outIx, eltN, /*postOp=*/false);
  }
  __device__ void directRecvReduceCopySend(intptr_t inpIx, intptr_t outIx, intptr_t remoteOutIx, int eltN, bool postOp=false) {
    // Direct is only for the send part
    static_cast<RealPrimitives*>(this)->recvReduceCopySend(inpIx, outIx, eltN, postOp);
  }
};

#include "prims_simple.h"
#include "prims_ll.h"
#include "prims_ll128.h"

#ifdef ENABLE_PROFILING
#define INIT_COUNTER \
  if (tid == 0) { struct ncclProfElem *elem = devProf.elems+args->opCount%PROFILE_NUM_ITEMS; t0 = __builtin_amdgcn_s_memrealtime(); ws = elem->elem[blockIdx.x].wait_cycle; }
#define ACCUMULATE_COUNTER(prim) \
  if (tid == 0) { struct ncclProfElem *elem = devProf.elems+args->opCount%PROFILE_NUM_ITEMS; elem->elem[blockIdx.x].prim##_cycle += (__builtin_amdgcn_s_memrealtime() - t0 \
    + ws - elem->elem[blockIdx.x].wait_cycle); \
    elem->elem[blockIdx.x].prim##_byte += nelem * sizeof(T); elem->elem[blockIdx.x].opCount = args->opCount;}
#define ACCUMULATE_PRIM_COUNTER(prim) \
  if (tid == 0) { struct ncclProfElem *elem = devProf.elems+args->opCount%PROFILE_NUM_ITEMS; elem->elem[blockIdx.x].prim##_cycle += (__builtin_amdgcn_s_memrealtime() - t0 \
    + ws - elem->elem[blockIdx.x].wait_cycle); elem->elem[blockIdx.x].opCount = args->opCount;}
#else
#define INIT_COUNTER
#define ACCUMULATE_COUNTER(prim)
#define ACCUMULATE_PRIM_COUNTER(prim)
#endif

#endif
