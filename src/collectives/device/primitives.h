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

#define barrier_by_id(id) do { \
  const int w = threadIdx.x/WARP_SIZE; \
  barrier_next[id*MAXWARPS+w] += nthreads/WARP_SIZE; \
  __atomic_fetch_add(barriers+id, 1, __ATOMIC_SEQ_CST); \
  while (LOAD(barriers+id) < barrier_next[id*MAXWARPS+w]) /* spin */; \
} while (0)

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
  struct ncclConnInfo* recvConn = NULL;
  volatile uint64_t* recvConnHeadPtr = NULL;
  uint64_t recvConnHead;
  volatile uint64_t* recvConnTailPtr = NULL;
  uint64_t recvConnTail;
  uint64_t recvConnTailCache; // Cache last seen value

  struct ncclConnInfo* sendConn = NULL;
  volatile int* sendConnFifoPtr = NULL;
  volatile uint64_t* sendConnTailPtr = NULL;
  uint64_t sendConnTail;
  volatile uint64_t* sendConnHeadPtr = NULL;
  uint64_t sendConnHead;
  uint64_t sendConnHeadCache; // Cache last seen value

  uint64_t recvStep[NRECV];
  uint64_t sendStep[NSEND];
#if defined(RCCL_USE_DIRECT_BUFFER)
  const T* recvDirectBuff[NRECV];
  T* sendDirectBuff[NSEND];
#endif
  const T* recvBuff[NRECV];
  T* sendBuff[NSEND];
  struct ncclDevComm* comm;

  inline __device__ int recvOffset(int i) { return (recvStep[i]%NCCL_STEPS)*stepSize; }
  inline __device__ int sendOffset(int i) { return (sendStep[i]%NCCL_STEPS)*stepSize; }
  inline __device__ const T* recvPtr(int i) { return ((const T*)recvBuff[i])+recvOffset(i); }
  inline __device__ T* sendPtr(int i) { return ((T*)sendBuff[i])+sendOffset(i); }

  uint64_t* barriers;
  uint64_t* barrier_next;

  inline __device__ void barrier() {
#if defined(__HIP_PLATFORM_HCC__) || defined(__HCC__) || defined(__HIPCC__)
    if (wid == 0) {
      if (NRECV < NSEND) barrier_by_id(0);
      else barrier_by_id(1);
    }
#else
    if (NSEND>NRECV) {
      asm volatile ("bar.sync 1, %0;" :: "r"(nthreads+WARP_SIZE));
    } else {
      asm volatile ("bar.sync 2, %0;" :: "r"(nthreads+WARP_SIZE));
    }
#endif
  }

  inline __device__ void subBarrier() {
#if defined(__HIP_PLATFORM_HCC__) || defined(__HCC__) || defined(__HIPCC__)
    __syncthreads();
#else
    if (NSEND>NRECV) {
      asm volatile ("bar.sync 3, %0;" :: "r"(nthreads));
    } else {
      asm volatile ("bar.sync 4, %0;" :: "r"(nthreads));
    }
#endif
  }

  uint32_t spins = 0;
  uint32_t abort = 0;

  inline __device__ int checkAbort(int i, int send) {
    spins++;
    if (abort == 0 && spins == SPINS_BEFORE_CHECK_ABORT) {
      abort = LOAD(comm->abortFlag);
      spins = 0;
    }
    return abort;
  }

  inline __device__ void waitSend(int nbytes) {
    spins = 0;
    if (sendConnHeadPtr) {
      while (sendConnHeadCache + NCCL_STEPS < sendConnHead + SLICESTEPS) {
        sendConnHeadCache = LOAD(sendConnHeadPtr);
        if (checkAbort(wid, 1)) break;
      }
      if (sendConnFifoPtr) {
        STORE(sendConnFifoPtr+sendConnHead%NCCL_STEPS, nbytes);
      }
      sendConnHead += SLICESTEPS;
    }
  }

  inline __device__ void waitRecv() {
    spins = 0;
    if (recvConnTailPtr) {
#ifdef ENABLE_PROFILING
      uint64_t t0 = __rtc64();
#endif
      while (recvConnTailCache < recvConnTail + SLICESTEPS) {
        recvConnTailCache = LOAD(recvConnTailPtr);
        if (checkAbort(wid, 0)) break;
      }
#ifdef ENABLE_PROFILING
      if (opCount > 0) __atomic_fetch_add(&comm->devProf->wait_recv_cycle[blockIdx.x], __rtc64() - t0, __ATOMIC_SEQ_CST);
#endif
      recvConnTail += SLICESTEPS;
    }
  }

  inline __device__ void incRecv(int i) {
    recvStep[i] += SLICESTEPS;
  }
  inline __device__ void postRecv() {
    if (recvConnHeadPtr) STORE(recvConnHeadPtr, recvConnHead += SLICESTEPS);
  }

  inline __device__ void incSend(int i) {
    sendStep[i] += SLICESTEPS;
  }
  inline __device__ void postSend() {
    if (sendConnTailPtr) {
      if (sendConn->next_hdp_reg) STORE(sendConn->next_hdp_reg, 0x1);
      STORE(sendConnTailPtr, sendConnTail += SLICESTEPS);
    }
  }

  template <int DIRECTRECV>
  inline __device__ const T* directRecvPtr(int i, ssize_t directOffset) {
#if defined(RCCL_USE_DIRECT_BUFFER)
    return DIRECTRECV && recvDirectBuff[i] ? recvDirectBuff[i]+directOffset : recvPtr(i);
#else
    return recvPtr(i);
#endif
  }

  template <int DIRECTSEND>
  inline __device__ T* directSendPtr(int i, ssize_t directOffset) {
#if defined(RCCL_USE_DIRECT_BUFFER)
    return DIRECTSEND && sendDirectBuff[i] ? sendDirectBuff[i]+directOffset : sendPtr(i);
#else
    return sendPtr(i);
#endif
  }

template <int DIRECTRECV>
inline __device__ int directRecvInc(int i, int directInc, int sliceInc) {
#if defined(RCCL_USE_DIRECT_BUFFER)
  return DIRECTRECV && recvDirectBuff[i] ? directInc : sliceInc;
#else
  return sliceInc;
#endif
}

template <int DIRECTSEND>
inline __device__ int directSendInc(int i, int directInc, int sliceInc) {
#if defined(RCCL_USE_DIRECT_BUFFER)
  return DIRECTSEND && sendDirectBuff[i] ? directInc : sliceInc;
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
        if (DIRECTRECV && recvDirectBuff[0]) {
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
    recvBuff[i] = (const T*)LOAD(conn->buffs+NCCL_PROTO_SIMPLE);
    recvStep[i] = LOAD(&conn->step);
    recvStep[i] = ROUNDUP(recvStep[i], SLICESPERCHUNK*SLICESTEPS);
#if defined(RCCL_USE_DIRECT_BUFFER)
    recvDirectBuff[i] = NULL;
    if (DIRECT && LOAD((&conn->direct) & NCCL_DIRECT_GPU)) {
      recvDirectBuff[i] = directBuff;
      if (tid == 0) STORE(conn->ptrExchange, directBuff);
    }
#endif
    if (wid == i) recvConn = conn;
    if (wid == i) recvConnTail = recvConnHead = recvStep[i]; // Make sure we set this after rounding up
    nrecv++;
  }

  __device__ __forceinline__ void loadRecvSync() {
    if (tid >= WARP_SIZE && tid < 2*WARP_SIZE && wid<nrecv) {
      recvConnTailPtr = LOAD(&recvConn->tail);
      recvConnTailCache = LOAD(recvConnTailPtr);
    }
    if (tid >= nthreads-WARP_SIZE && wid < nrecv) {
      recvConnHeadPtr = LOAD(&recvConn->head);
      // Return credits in case we rounded up.
      STORE(recvConnHeadPtr, recvConnHead);
    }
  }

  __device__ __forceinline__ void loadSendConn(struct ncclConnInfo* conn, int i) {
    sendBuff[i] = (T*)LOAD(conn->buffs+NCCL_PROTO_SIMPLE);
    sendStep[i] = LOAD(&conn->step);
    sendStep[i] = ROUNDUP(sendStep[i], SLICESPERCHUNK*SLICESTEPS);
#if defined(RCCL_USE_DIRECT_BUFFER)
    sendDirectBuff[i] = NULL;
    if (DIRECT && LOAD((&conn->direct) & NCCL_DIRECT_GPU)) {
      void* volatile* ptr = LOAD(&conn->ptrExchange);
      while ((sendDirectBuff[i] = (T*)(LOAD(ptr))) == NULL);
      barrier();
      if (tid == 0) STORE(ptr, NULL);
    }
#endif
    if (wid == i) sendConn = conn;
    if (wid == i) sendConnTail = sendConnHead = sendStep[i]; // Make sure we set this after rounding up
    nsend++;
  }
  __device__ __forceinline__ void loadSendSync() {
    if (tid < nsend) {
      sendConnHeadPtr = LOAD(&sendConn->head);
      sendConnHeadCache = LOAD(sendConnHeadPtr);
      sendConnFifoPtr = LOAD(&sendConn->fifo);
    }
    if (tid >= nthreads-WARP_SIZE && wid < nsend) {
      sendConnTailPtr = LOAD(&sendConn->tail);
    }
  }

  __device__ __forceinline__ void saveRecvSync() {
    if (tid >= nthreads-WARP_SIZE && wid < nrecv) {
      STORE(&recvConn->step, recvConnHead);
      __threadfence_system();
    }
  }

  __device__ __forceinline__ void saveSendSync() {
    if (tid < nsend) {
      STORE(&sendConn->step, sendConnHead);
      __threadfence_system();
    }
  }

 public:
  __device__ __forceinline__
  ncclPrimitives(const int tid, const int nthreads, int* recvPeers, int* sendPeers, T* directBuff, int stepSize, struct ncclChannel* channel, struct ncclDevComm* comm)
    : comm(comm), tid(tid), nthreads(nthreads), wid(tid%WARP_SIZE), stepSize(stepSize) {
    barriers = channel->barrier;
    barrier_next = channel->barrier_next;
    // Make sure step is updated before we read it.
    barrier();

    for (int i=0; i<NRECV && recvPeers[i] >= 0; i++) loadRecvConn(&channel->devPeers[recvPeers[i]].recv.conn, i, 0);
    for (int i=0; i<NSEND && sendPeers[i] >= 0; i++) loadSendConn(&channel->devPeers[sendPeers[i]].send.conn, i);
    loadRecvSync();
    loadSendSync();
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
