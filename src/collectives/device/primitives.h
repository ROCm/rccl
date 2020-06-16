/*************************************************************************
 * Copyright (c) 2016-2020, NVIDIA CORPORATION. All rights reserved.
 * Modifications Copyright (c) 2019-2020 Advanced Micro Devices, Inc. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

#ifndef NCCL_PRIMITIVES_H_
#define NCCL_PRIMITIVES_H_

#include <type_traits>
#include "reduce_kernel.h" // for reduction funcs
#include "common.h"

#define SPINS_BEFORE_CHECK_ABORT 1000000

// Unroll unconditionally the first send/recv since nsend/nrecv should be at
// least 1 if SEND/RECV is set.
#define FOR_SEND(func, ...) do { \
  if (SEND) { \
    /* Send to far first, then close */ \
    for (int i=1; i<NSEND && i<nsend; i++) func(i, ##__VA_ARGS__); \
    func(0, ##__VA_ARGS__); \
  } \
} while (0)

#define FOR_RECV(func, ...) do { \
  if (RECV) { \
    /* Recv from close first, then far */ \
    func(0, ##__VA_ARGS__); \
    for (int i=1; i<NRECV && i<nrecv; i++) func(i, ##__VA_ARGS__); \
  } \
} while (0)

template <typename T, int NRECV>
class ncclPrimitivesRecvData {
public:
  struct ncclConnInfo* recvConn = NULL;
  volatile uint64_t* recvConnHeadPtr = NULL;
  uint64_t recvConnHead;
  volatile uint64_t* recvConnTailPtr = NULL;
  uint64_t recvConnTail;
  uint64_t recvConnTailCache; // Cache last seen value

  uint64_t recvStep[NRECV];
#if defined(RCCL_USE_DIRECT_BUFFER)
  const T* recvDirectBuff[NRECV];
#endif
  const T* recvBuff[NRECV];
};

template <typename T, int NSEND>
class ncclPrimitivesSendData {
public:
  struct ncclConnInfo* sendConn = NULL;
  volatile int* sendConnFifoPtr = NULL;
  volatile uint64_t* sendConnTailPtr = NULL;
  uint64_t sendConnTail;
  volatile uint64_t* sendConnHeadPtr = NULL;
  uint64_t sendConnHead;
  uint64_t sendConnHeadCache; // Cache last seen value

  uint64_t sendStep[NSEND];
#if defined(RCCL_USE_DIRECT_BUFFER)
  const T* sendDirectBuff[NSEND];
#endif
  T* sendBuff[NSEND];
};

// Implementation of primitive types
template <int UNROLL, int SLICESPERCHUNK, int SLICESTEPS, typename T, int NRECV, int NSEND, int DIRECT, class FUNC>
class ncclPrimitives {
 private:
  const int tid;
  const int nthreads;
  const int wid;
  const int stepSize;
  int nrecv = 0;
  int nsend = 0;

  typename std::conditional<NRECV == NCCL_MAX_TREE_ARITY,
    ncclPrimitivesRecvData<T, NRECV>&, ncclPrimitivesRecvData<T, NRECV>>::type r;
  typename std::conditional<NSEND == NCCL_MAX_TREE_ARITY,
    ncclPrimitivesSendData<T, NSEND>&, ncclPrimitivesSendData<T, NSEND>>::type s;

  const struct ncclDevComm* comm;

  inline __device__ int recvOffset(int i) { return (r.recvStep[i]%NCCL_STEPS)*stepSize; }
  inline __device__ int sendOffset(int i) { return (s.sendStep[i]%NCCL_STEPS)*stepSize; }
  inline __device__ const T* recvPtr(int i) { return ((const T*)r.recvBuff[i])+recvOffset(i); }
  inline __device__ T* sendPtr(int i) { return ((T*)s.sendBuff[i])+sendOffset(i); }

  inline __device__ void barrier() {
#if defined(__HIP_PLATFORM_HCC__) || defined(__HCC__) || defined(__HIPCC__)
    __syncthreads();
#else
    asm volatile ("bar.sync 1, %0;" :: "r"(nthreads+WARP_SIZE));
#endif
  }

  inline __device__ void subBarrier() {
#if defined(__HIP_PLATFORM_HCC__) || defined(__HCC__) || defined(__HIPCC__)
    __syncthreads();
#else
    asm volatile ("bar.sync 2, %0;" :: "r"(nthreads));
#endif
  }

  uint32_t mismatch = 0;
  const uint64_t opCount;

  inline __device__ void checkMismatch(struct ncclConnInfo* conn) {
    if (mismatch) {
      // In non-LL, we use _threadfence_system before incrementing opCount, yet we are still waiting for credits here, so there must be a size mismatch
      STORE(comm->fatalDevError, ncclDevAssertedMismatch);
    } else if (conn && LOAD(conn->opCountRem) > opCount+1) {
      mismatch += 1;
    }
  }

  uint32_t spins = 0;
  uint32_t abort = 0;

  inline __device__ int checkAbort(int i, int send) {
    spins++;
    if (abort == 0 && spins == SPINS_BEFORE_CHECK_ABORT) {
      abort = LOAD(comm->abortFlag);
      if (wid == i) checkMismatch(send ? s.sendConn : r.recvConn);
      spins = 0;
    }
    return abort;
  }

  inline __device__ void waitSend(int nbytes) {
    spins = 0;
    mismatch = 0;
    if (s.sendConnHeadPtr) {
      while (s.sendConnHeadCache + NCCL_STEPS < s.sendConnHead + SLICESTEPS) {
        s.sendConnHeadCache = LOAD(s.sendConnHeadPtr);
        if (checkAbort(wid, 1)) break;
      }
      if (s.sendConnFifoPtr) {
        STORE(s.sendConnFifoPtr+s.sendConnHead%NCCL_STEPS, nbytes);
      }
      s.sendConnHead += SLICESTEPS;
    }
  }

  inline __device__ void waitRecv() {
    spins = 0;
    mismatch = 0;
    if (r.recvConnTailPtr) {
#ifdef ENABLE_PROFILING
      uint64_t t0 = __rtc64();
#endif
      while (r.recvConnTailCache < r.recvConnTail + SLICESTEPS) {
        r.recvConnTailCache = LOAD(r.recvConnTailPtr);
        if (checkAbort(wid, 0)) break;
      }
#ifdef ENABLE_PROFILING
      if (opCount > 0) __atomic_fetch_add(&comm->devProf->wait_recv_cycle[blockIdx.x], __rtc64() - t0, __ATOMIC_SEQ_CST);
#endif
      r.recvConnTail += SLICESTEPS;
    }
  }

  inline __device__ void incRecv(int i) {
    r.recvStep[i] += SLICESTEPS;
  }
  inline __device__ void postRecv() {
    if (r.recvConnHeadPtr) STORE(r.recvConnHeadPtr, r.recvConnHead += SLICESTEPS);
  }

  inline __device__ void incSend(int i) {
    s.sendStep[i] += SLICESTEPS;
  }
  inline __device__ void postSend() {
    if (s.sendConnTailPtr) {
      if (s.sendConn->next_hdp_reg) STORE(s.sendConn->next_hdp_reg, 0x1);
      STORE(s.sendConnTailPtr, s.sendConnTail += SLICESTEPS);
    }
  }

  template <int DIRECTRECV>
  inline __device__ const T* directRecvPtr(int i, ssize_t directOffset) {
#if defined(RCCL_USE_DIRECT_BUFFER)
    return DIRECTRECV && r.recvDirectBuff[i] ? r.recvDirectBuff[i]+directOffset : recvPtr(i);
#else
    return recvPtr(i);
#endif
  }

  template <int DIRECTSEND>
  inline __device__ T* directSendPtr(int i, ssize_t directOffset) {
#if defined(RCCL_USE_DIRECT_BUFFER)
    return DIRECTSEND && s.sendDirectBuff[i] ? s.sendDirectBuff[i]+directOffset : sendPtr(i);
#else
    return sendPtr(i);
#endif
  }

template <int DIRECTRECV>
inline __device__ int directRecvInc(int i, int directInc, int sliceInc) {
#if defined(RCCL_USE_DIRECT_BUFFER)
  return DIRECTRECV && r.recvDirectBuff[i] ? directInc : sliceInc;
#else
  return sliceInc;
#endif
}

template <int DIRECTSEND>
inline __device__ int directSendInc(int i, int directInc, int sliceInc) {
#if defined(RCCL_USE_DIRECT_BUFFER)
  return DIRECTSEND && s.sendDirectBuff[i] ? directInc : sliceInc;
#else
  return sliceInc;
#endif
}

  template <int DIRECTRECV, int DIRECTSEND, int RECV, int SEND, int SRC, int DST>
  inline __device__ void
  GenericOp(const T* srcPtr, T* dstPtr, int nelem, ssize_t directOffset) {
    int offset = 0;
    int sliceSize = stepSize*SLICESTEPS;
    int dataSize = max(DIVUP(nelem, 16*SLICESPERCHUNK)*16, sliceSize/32);

    const T* srcs[RECV*NRECV+SRC];
    srcs[0] = SRC ? srcPtr : directRecvPtr<DIRECTRECV>(0, directOffset);
    if (RECV) {
      if (SRC) srcs[1] = recvPtr(0);
      for (int i=1; i<NRECV && i<nrecv; i++) srcs[SRC+i] = recvPtr(i);
    }

    T* dsts[SEND*NSEND+DST];
    dsts[0] = DST ? dstPtr : directSendPtr<DIRECTSEND>(0, directOffset);
    if (SEND) {
      if (DST) dsts[1] = directSendPtr<DIRECTSEND>(0, directOffset);
      for (int i=1; i<NSEND && i<nsend; i++) dsts[DST+i] = directSendPtr<DIRECTSEND>(i, directOffset);
    }

    #pragma unroll
    for (int slice=0; slice<SLICESPERCHUNK; ++slice) {
      int realSize = max(0, min(dataSize, nelem-offset));
#ifdef ENABLE_PROFILING
      uint64_t t0 = __rtc64();
#endif
      if (SEND) waitSend(realSize*sizeof(T));
      if (RECV) waitRecv();
      if (realSize > 0) {
        barrier();
#ifdef ENABLE_PROFILING
        if (tid == 0  && opCount > 0) __atomic_fetch_add(&comm->devProf->wait_cycle[blockIdx.x], __rtc64() - t0, __ATOMIC_SEQ_CST);
#endif
#if defined(RCCL_USE_DIRECT_BUFFER)
        if (DIRECTRECV && r.recvDirectBuff[0]) {
          // We can only have one direct receive. Since srcs[0] == dstPtr+offset, skip one copy
          if (SEND) {
            ReduceOrCopyMulti<UNROLL, FUNC, T, 1, 1, 1, NSEND>(tid, nthreads, 1, srcs, nsend, dsts+1, realSize);
          }
        } else {
          ReduceOrCopyMulti<UNROLL, FUNC, T, RECV+SRC, RECV*NRECV+SRC, SEND+DST, SEND*NSEND+DST>(tid, nthreads, RECV*nrecv+SRC, srcs, SEND*nsend+DST, dsts, realSize);
        }
#else
        ReduceOrCopyMulti<UNROLL, FUNC, T, RECV+SRC, RECV*NRECV+SRC, SEND+DST, SEND*NSEND+DST>(tid, nthreads, RECV*nrecv+SRC, srcs, SEND*nsend+DST, dsts, realSize);
#endif
      }
      barrier();
      FOR_SEND(incSend);
      FOR_RECV(incRecv);
      if (tid >= nthreads-WARP_SIZE) {
        if (SEND) {
          if (realSize > 0 && wid == 0) __threadfence_system();
          __syncwarp();
          postSend();
        }
        if (RECV) postRecv();
      }
      srcs[0] += SRC ? realSize : directRecvInc<DIRECTRECV>(0, realSize, sliceSize);
      for (int i=1-SRC; i<RECV*NRECV; i++) srcs[SRC+i] += sliceSize;
      dsts[0] += DST ? realSize : directSendInc<DIRECTSEND>(0, realSize, sliceSize);
      for (int i=1-DST; i<SEND*NSEND; i++) dsts[DST+i] += directSendInc<DIRECTSEND>(i, realSize, sliceSize);
      offset += realSize;
    }
  }

  __device__ __forceinline__ void loadRecvConn(struct ncclConnInfo* conn, int i, T* directBuff) {
    r.recvBuff[i] = (const T*)LOAD(conn->buffs+NCCL_PROTO_SIMPLE);
    r.recvStep[i] = LOAD(&conn->step);
    r.recvStep[i] = ROUNDUP(r.recvStep[i], SLICESPERCHUNK*SLICESTEPS);
#if defined(RCCL_USE_DIRECT_BUFFER)
    r.recvDirectBuff[i] = NULL;
    if (DIRECT && LOAD((&conn->direct) & NCCL_DIRECT_GPU)) {
      r.recvDirectBuff[i] = directBuff;
      if (tid == 0) STORE(conn->ptrExchange, directBuff);
    }
#endif
    if (wid == i) r.recvConn = conn;
    if (wid == i) r.recvConnTail = r.recvConnHead = r.recvStep[i]; // Make sure we set this after rounding up
    nrecv++;
  }

  __device__ __forceinline__ void loadRecvSync() {
    if (tid >= WARP_SIZE && tid < 2*WARP_SIZE && wid<nrecv) {
      r.recvConnTailPtr = LOAD(&r.recvConn->tail);
      r.recvConnTailCache = LOAD(r.recvConnTailPtr);
    }
    if (tid >= nthreads-WARP_SIZE && wid < nrecv) {
      r.recvConnHeadPtr = LOAD(&r.recvConn->head);
      // Return credits in case we rounded up.
      STORE(r.recvConnHeadPtr, r.recvConnHead);
      // Update opCount in case we skipped some operations
      STORE(r.recvConn->opCountLoc, opCount);
    }
  }

  __device__ __forceinline__ void loadSendConn(struct ncclConnInfo* conn, int i) {
    s.sendBuff[i] = (T*)LOAD(conn->buffs+NCCL_PROTO_SIMPLE);
    s.sendStep[i] = LOAD(&conn->step);
    s.sendStep[i] = ROUNDUP(s.sendStep[i], SLICESPERCHUNK*SLICESTEPS);
#if defined(RCCL_USE_DIRECT_BUFFER)
    s.sendDirectBuff[i] = NULL;
    if (DIRECT && LOAD((&conn->direct) & NCCL_DIRECT_GPU)) {
      void* volatile* ptr = LOAD(&conn->ptrExchange);
      while ((s.sendDirectBuff[i] = (T*)(LOAD(ptr))) == NULL);
      barrier();
      if (tid == 0) STORE(ptr, NULL);
    }
#endif
    if (wid == i) s.sendConn = conn;
    if (wid == i) s.sendConnTail = s.sendConnHead = s.sendStep[i]; // Make sure we set this after rounding up
    nsend++;
  }
  __device__ __forceinline__ void loadSendSync() {
    if (tid < nsend) {
      s.sendConnHeadPtr = LOAD(&s.sendConn->head);
      s.sendConnHeadCache = LOAD(s.sendConnHeadPtr);
      s.sendConnFifoPtr = LOAD(&s.sendConn->fifo);
      STORE(s.sendConn->opCountLoc, opCount);
    }
    if (tid >= nthreads-WARP_SIZE && wid < nsend) {
      s.sendConnTailPtr = LOAD(&s.sendConn->tail);
    }
  }

  __device__ __forceinline__ void saveRecvSync() {
    if (tid >= nthreads-WARP_SIZE && wid < nrecv) {
      STORE(&r.recvConn->step, r.recvConnHead);
      STORE(r.recvConn->opCountLoc, opCount+1);
      __threadfence_system();
    }
  }

  __device__ __forceinline__ void saveSendSync() {
    if (tid < nsend) {
      STORE(&s.sendConn->step, s.sendConnHead);
      STORE(s.sendConn->opCountLoc, opCount+1);
      __threadfence_system();
    }
  }

  inline __device__ void init(int* recvPeers, int* sendPeers, struct ncclChannel* channel) {
    // Make sure step is updated before we read it.
    barrier();

    for (int i=0; i<NRECV && recvPeers[i] >= 0; i++) loadRecvConn(&channel->devPeers[recvPeers[i]].recv.conn, i, 0);
    for (int i=0; i<NSEND && sendPeers[i] >= 0; i++) loadSendConn(&channel->devPeers[sendPeers[i]].send.conn, i);
    loadRecvSync();
    loadSendSync();
  }

 public:
  __device__ __forceinline__
  ncclPrimitives(const int tid, const int nthreads, int* recvPeers, int* sendPeers, T* directBuff, int stepSize, struct ncclChannel* channel, struct ncclDevComm* comm, const uint64_t opCount)
    : comm(comm), tid(tid), nthreads(nthreads), wid(tid%WARP_SIZE), stepSize(stepSize), opCount(opCount) {
    init(recvPeers, sendPeers, channel);
  }

  __device__ __forceinline__
  ncclPrimitives(const int tid, const int nthreads, int* recvPeers, int* sendPeers, T* directBuff, int stepSize, struct ncclChannel* channel, struct ncclDevComm* comm, const uint64_t opCount, ncclPrimitivesRecvData<T, NRECV>& r)
    : comm(comm), tid(tid), nthreads(nthreads), wid(tid%WARP_SIZE), stepSize(stepSize), opCount(opCount), r(r) {
    init(recvPeers, sendPeers, channel);
  }

  __device__ __forceinline__
  ncclPrimitives(const int tid, const int nthreads, int* recvPeers, int* sendPeers, T* directBuff, int stepSize, struct ncclChannel* channel, struct ncclDevComm* comm, const uint64_t opCount, ncclPrimitivesSendData<T, NSEND>& s)
    : comm(comm), tid(tid), nthreads(nthreads), wid(tid%WARP_SIZE), stepSize(stepSize), opCount(opCount), s(s) {
    init(recvPeers, sendPeers, channel);
  }

  __device__ __forceinline__ void
  send(const T* src, int nelem) {
    GenericOp<0, 0, 0, 1, 1, 0>(src, NULL, nelem, 0);
  }
  __device__ __forceinline__ void
  directSend(const T* src, ssize_t directOffset, int nelem) {
    GenericOp<0, 1, 0, 1, 1, 0>(src, NULL, nelem, directOffset);
  }

  __device__ __forceinline__ void
  recv(T* dst, int nelem) {
    GenericOp<0, 0, 1, 0, 0, 1>(NULL, dst, nelem, 0);
  }
  __device__ __forceinline__ void
  directRecv(T* dst, ssize_t directOffset, int nelem) {
    GenericOp<1, 0, 1, 0, 0, 1>(NULL, dst, nelem, directOffset);
  }

  __device__ __forceinline__ void
  copySend(const T* src, T* dst, int nelem) {
    GenericOp<0, 0, 0, 1, 1, 1>(src, dst, nelem, 0);
  }
  __device__ __forceinline__ void
  directCopySend(const T* src, T* dst, ssize_t directOffset, int nelem) {
    GenericOp<0, 1, 0, 1, 1, 1>(src, dst, nelem, directOffset);
  }

  __device__ __forceinline__ void
  recvCopySend(T* dst, int nelem) {
    GenericOp<0, 0, 1, 1, 0, 1>(NULL, dst, nelem, 0);
  }
  __device__ __forceinline__ void
  directRecvCopySend(T* dst, ssize_t directOffset, int nelem) {
    GenericOp<1, 1, 1, 1, 0, 1>(NULL, dst, nelem, directOffset);
  }

  __device__ __forceinline__ void
  recvReduceCopy(const T* src, T* dst, int nelem) {
    GenericOp<0, 0, 1, 0, 1, 1>(src, dst, nelem, 0);
  }

  __device__ __forceinline__ void
  recvReduceSend(const T* src, int nelem) {
    GenericOp<0, 0, 1, 1, 1, 0>(src, NULL, nelem, 0);
  }

  __device__ __forceinline__ void
  recvReduceCopySend(const T* src, T* dst, int nelem) {
    GenericOp<0, 0, 1, 1, 1, 1>(src, dst, nelem, 0);
  }
  __device__ __forceinline__ void
  directRecvReduceCopySend(const T* src, T* dst, ssize_t directOffset, int nelem) {
    // Direct is only for the send part
    GenericOp<0, 1, 1, 1, 1, 1>(src, dst, nelem, directOffset);
  }

  __device__ __forceinline__ ~ncclPrimitives() {
    // Save steps for the next operation
    saveRecvSync();
    saveSendSync();
  }
};

#include "prims_ll.h"
//#include "prims_ll128.h"

#ifdef ENABLE_PROFILING
#define INIT_COUNTER \
  if (tid == 0) { t0 = __rtc64(); ws = LOAD(&(devProf->wait_cycle[blockIdx.x])); }

#define ACCUMULATE_COUNTER(prim) \
  if (tid == 0 && args->opCount > 0) { __atomic_fetch_add(&(devProf->prim##_cycle), __rtc64() - t0 \
    + ws - LOAD(&(devProf->wait_cycle[blockIdx.x])), __ATOMIC_SEQ_CST); \
    __atomic_fetch_add(&(devProf->prim##_byte), nelem * sizeof(T), __ATOMIC_SEQ_CST); }
#else
#define INIT_COUNTER
#define ACCUMULATE_COUNTER(prim)
#endif

#endif
