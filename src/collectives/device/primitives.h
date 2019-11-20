/*************************************************************************
 * Copyright (c) 2016-2019, NVIDIA CORPORATION. All rights reserved.
 * Modifications Copyright (c) 2019 Advanced Micro Devices, Inc. All rights reserved.
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

// Implementation of primitive types
template <int UNROLL, int SLICESPERCHUNK, int SLICESTEPS, typename T, int NRECV, int NSEND, class FUNC>
class ncclPrimitives {
 private:
  const int tid;
  const int nthreads;
  int nrecv = 0;
  int nsend = 0;
  const int stepSize;
  struct ncclConnInfo* recvConn[NRECV];
  struct ncclConnInfo* sendConn[NSEND];
  volatile uint64_t* waitPtr;
  uint64_t recvStep[NRECV];
  uint64_t sendStep[NSEND];
  uint64_t sendConnHead[NSEND];
#if defined(RCCL_USE_DIRECT_BUFFER)
  const T* recvDirectBuff[NRECV];
  T* sendDirectBuff[NSEND];
#endif
  const T* recvBuff[NRECV];
  T* sendBuff[NSEND];
  struct ncclDevComm* comm;
  uint32_t* abortCount;

  __device__ int recvOffset(int i) { return (recvStep[i]%NCCL_STEPS)*stepSize; }
  __device__ int sendOffset(int i) { return (sendStep[i]%NCCL_STEPS)*stepSize; }
  __device__ const T* recvPtr(int i) { return ((const T*)recvBuff[i])+recvOffset(i); }
  __device__ T* sendPtr(int i) { return ((T*)sendBuff[i])+sendOffset(i); }

  __device__ void barrier() {
#if defined(__HIP_PLATFORM_HCC__) || defined(__HCC__) || defined(__HIPCC__)
    __syncthreads();
#else
    asm volatile ("bar.sync 1, %0;" :: "r"(nthreads));
#endif
  }

  uint32_t mismatch = 0;
  const uint64_t opCount;

  __device__ void checkMismatch(volatile uint64_t* remoteOpCount) {
    if (mismatch) {
      // In non-LL, we use _threadfence_system before incrementing opCount, yet we are still waiting for credits here, so there must be a size mismatch
      STORE(comm->fatalDevError, ncclDevAssertedMismatch);
    } else if (remoteOpCount && LOAD(remoteOpCount) > opCount) {
      mismatch += 1;
    }
  }

  uint32_t spins = 0;
  uint32_t abort = 0;

  __device__ int checkAbort(volatile uint64_t* remoteOpCount) {
    spins++;
    if (spins == SPINS_BEFORE_CHECK_ABORT) {
      abort = LOAD(comm->abortFlag);
      checkMismatch(remoteOpCount);
      spins = 0;
    }
    return abort;
  }

  __device__ void waitRecv(int i) {
    spins = 0;
    mismatch = 0;
    recvStep[i] += SLICESTEPS;
    if (tid == i) {
#ifdef ENABLE_PROFILING
      auto devProf = comm->devProf;
      uint64_t t0 = clock64();
#endif
      while (LOAD(waitPtr) < recvStep[i]) {
        if (checkAbort(recvConn[i]->opCountRem)) break;
      }
#ifdef ENABLE_PROFILING
      __atomic_fetch_add(&devProf->wait_recv_cycle[blockIdx.x], clock64() - t0, __ATOMIC_SEQ_CST);
#endif
    }
  }

  __device__ void waitSend(int i) {
    spins = 0;
    mismatch = 0;
    sendStep[i] += SLICESTEPS;
    if (tid == WARP_SIZE+i) {
#ifdef ENABLE_PROFILING
      auto devProf = comm->devProf;
      uint64_t t0 = clock64();
#endif
      while (sendConnHead[i] + NCCL_STEPS < sendStep[i]) {
        sendConnHead[i] = LOAD(waitPtr);
        if (checkAbort(sendConn[i]->opCountRem)) break;
      }
#ifdef ENABLE_PROFILING
      __atomic_fetch_add(&devProf->wait_send_cycle[blockIdx.x], clock64() - t0, __ATOMIC_SEQ_CST);
#endif
    }
  }

  inline __device__ void postRecv(int i) {
    STORE(recvConn[i]->head, recvStep[i]);
  }

  inline __device__ void postSend(int i) {
    if (sendConn[i]->next_hdp_reg) STORE(sendConn[i]->next_hdp_reg, 0x1);
    STORE(sendConn[i]->tail, sendStep[i]);
  }

  __device__ void postSendSize(int i, int size) {
    if (sendConn[i]->fifo) STORE(sendConn[i]->fifo+((sendStep[i]-SLICESTEPS)%NCCL_STEPS), size);
  }

  template <int DIRECTRECV>
  __device__ const T* directRecvPtr(int i, int directOffset) {
#if defined(RCCL_USE_DIRECT_BUFFER)
    return DIRECTRECV && recvDirectBuff[i] ? recvDirectBuff[i]+directOffset : recvPtr(i);
#else
    return recvPtr(i);
#endif
  }

  template <int DIRECTSEND>
  __device__ T* directSendPtr(int i, int directOffset) {
  #if defined(RCCL_USE_DIRECT_BUFFER)
    return DIRECTSEND && sendDirectBuff[i] ? sendDirectBuff[i]+directOffset : sendPtr(i);
  #else
    return sendPtr(i);
  #endif
  }

  template <int DIRECTRECV, int DIRECTSEND, int RECV, int SEND, int SRC, int DST>
  __device__ void
  GenericOp(const T* srcPtr, T* dstPtr, int nelem, int directOffset) {
    int offset = 0;
    int sliceSize = stepSize * SLICESTEPS;

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

    #pragma unroll 1
    for (int slice=0; slice<SLICESPERCHUNK; ++slice) {
      int realSize = max(0, min(sliceSize, nelem-offset));
      FOR_SEND(waitSend);
      FOR_RECV(waitRecv);
      if (realSize > 0) {
        barrier();
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
      exitIfAbortBarrier(abort, abortCount);
      if (tid == 0)
      {
        FOR_SEND(postSendSize, realSize*sizeof(T));
        __threadfence_system();
        FOR_SEND(postSend);
        FOR_RECV(postRecv);
      }

      for (int i=0; i<RECV*NRECV+SRC; i++) srcs[i] += sliceSize;
      for (int i=0; i<SEND*NSEND+DST; i++) dsts[i] += sliceSize;
      offset += sliceSize;
    }
  }

  __device__ void loadRecvConn(struct ncclConnInfo* conn, int i, T* directBuff) {
    recvConn[i] = conn;
    recvBuff[i] = (const T*)LOAD(&recvConn[i]->buff);
    recvStep[i] = LOAD(&recvConn[i]->step);
    recvStep[i] = ROUNDUP(recvStep[i], SLICESPERCHUNK*SLICESTEPS);
    // Return credits in case we rounded up.
    if (tid == 0) STORE(recvConn[i]->head, recvStep[i]);
    if (tid == i) {
      waitPtr = LOAD(&recvConn[i]->tail);
      STORE(recvConn[i]->opCountLoc, opCount);
    }
#if defined(RCCL_USE_DIRECT_BUFFER)
    recvDirectBuff[i] = NULL;
    if (directBuff && recvConn[i]->direct) {
      recvDirectBuff[i] = directBuff;
      if (tid == 0) STORE(recvConn[i]->ptrExchange, directBuff);
    }
#endif
    nrecv++;
  }

  __device__ void loadSendConn(struct ncclConnInfo* conn, int i, T* directBuff) {
    sendConn[i] = conn;
    sendBuff[i] = (T*)LOAD(&sendConn[i]->buff);
    sendStep[i] = LOAD(&sendConn[i]->step);
    sendStep[i] = ROUNDUP(sendStep[i], SLICESPERCHUNK*SLICESTEPS);
    if (tid == WARP_SIZE+i) {
      waitPtr = LOAD(&sendConn[i]->head);
      sendConnHead[i] = LOAD(waitPtr);
      STORE(sendConn[i]->opCountLoc, opCount);
    }
#if defined(RCCL_USE_DIRECT_BUFFER)
    sendDirectBuff[i] = NULL;
    if (directBuff && sendConn[i]->direct) {
      void* volatile* ptr = sendConn[i]->ptrExchange;
      while ((sendDirectBuff[i] = (T*)(LOAD(ptr))) == NULL);
      __syncthreads();
      if (tid == 0) STORE(ptr, NULL);
    }
#endif
    nsend++;
  }

  __device__ void saveRecvConn(int i) {
    if (tid == i) {
      STORE(&recvConn[i]->step, recvStep[i]);
      __threadfence_system();
      __atomic_fetch_add(recvConn[i]->opCountLoc, 1, __ATOMIC_SEQ_CST);
    }
  }

  __device__ void saveSendConn(int i) {
    if (tid == WARP_SIZE+i) {
      STORE(&sendConn[i]->step, sendStep[i]);
      __threadfence_system();
      __atomic_fetch_add(sendConn[i]->opCountLoc, 1, __ATOMIC_SEQ_CST);
    }
  }

 public:
  __device__
  ncclPrimitives(const int tid, const int nthreads, int* recvPeers, int* sendPeers, T* directBuff, int stepSize, struct ncclChannel* channel, struct ncclDevComm* comm, const uint64_t opCount)
    : comm(comm), tid(tid), nthreads(nthreads), stepSize(stepSize), opCount(opCount) {
    // Make sure step is updated before we read it
    abortCount = channel->abortCount;
    __syncthreads();

    // disable directBuff
    for (int i=0; i<NRECV && recvPeers[i] >= 0; i++) loadRecvConn(&channel->devPeers[recvPeers[i]].recv.conn, i, 0);
    for (int i=0; i<NSEND && sendPeers[i] >= 0; i++) loadSendConn(&channel->devPeers[sendPeers[i]].send.conn, i, 0);
  }

  __device__ void
  send(const T* src, int nelem) {
    GenericOp<0, 0, 0, 1, 1, 0>(src, NULL, nelem, 0);
  }
  __device__ void
  directSend(const T* src, int directOffset, int nelem) {
    GenericOp<0, 1, 0, 1, 1, 0>(src, NULL, nelem, directOffset);
  }

  __device__ void
  recv(T* dst, int nelem) {
    GenericOp<0, 0, 1, 0, 0, 1>(NULL, dst, nelem, 0);
  }
  __device__ void
  directRecv(T* dst, int directOffset, int nelem) {
    GenericOp<1, 0, 1, 0, 0, 1>(NULL, dst, nelem, directOffset);
  }

  __device__ void
  copySend(const T* src, T* dst, int nelem) {
    GenericOp<0, 0, 0, 1, 1, 1>(src, dst, nelem, 0);
  }
  __device__ void
  directCopySend(const T* src, T* dst, int directOffset, int nelem) {
    GenericOp<0, 1, 0, 1, 1, 1>(src, dst, nelem, directOffset);
  }

  __device__ void
  recvCopySend(T* dst, int nelem) {
    GenericOp<0, 0, 1, 1, 0, 1>(NULL, dst, nelem, 0);
  }
  __device__ void
  directRecvCopySend(T* dst, int directOffset, int nelem) {
    GenericOp<1, 1, 1, 1, 0, 1>(NULL, dst, nelem, directOffset);
  }

  __device__ void
  recvReduceCopy(const T* src, T* dst, int nelem) {
    GenericOp<0, 0, 1, 0, 1, 1>(src, dst, nelem, 0);
  }

  __device__ void
  recvReduceSend(const T* src, int nelem) {
    GenericOp<0, 0, 1, 1, 1, 0>(src, NULL, nelem, 0);
  }

  __device__ void
  recvReduceCopySend(const T* src, T* dst, int nelem) {
    GenericOp<0, 0, 1, 1, 1, 1>(src, dst, nelem, 0);
  }
  __device__ void
  directRecvReduceCopySend(const T* src, T* dst, int directOffset, int nelem) {
    // Direct is only for the send part
    GenericOp<0, 1, 1, 1, 1, 1>(src, dst, nelem, directOffset);
  }

  __device__ ~ncclPrimitives() {
    // Save steps for next collective. Have thread 0 do it to be compatible
    // with the way LL works.
    for (int i=0; i<NRECV && i<nrecv; i++) saveRecvConn(i);
    for (int i=0; i<NSEND && i<nsend; i++) saveSendConn(i);
  }
};

template <typename T, class FUNC, int NRECV, int NSEND>
class ncclLLPrimitives {
 private:
  const int tid;
  const int nthreads;
  int nrecv = 0;
  int nsend = 0;
  struct ncclConnInfo* recvConn[NRECV];
  struct ncclConnInfo* sendConn[NSEND];
  volatile uint64_t* waitPtr;
  volatile uint64_t* postPtr;
  volatile int* fifoPtr;
  uint64_t recvStep[NRECV];
  uint64_t sendStep[NSEND];
  uint64_t sendConnHead;
  union ncclLLFifoLine* recvBuff[NRECV];
  union ncclLLFifoLine* sendBuff[NSEND];
  struct ncclDevComm* comm;
  uint32_t* abortCount;

  __device__ int recvOffset(int i) { return (recvStep[i]%NCCL_STEPS)*NCCL_LL_SLICE_LINES; }
  __device__ int sendOffset(int i) { return (sendStep[i]%NCCL_STEPS)*NCCL_LL_SLICE_LINES; }
  __device__ union ncclLLFifoLine* recvPtr(int i) { return recvBuff[i]+recvOffset(i); }
  __device__ union ncclLLFifoLine* sendPtr(int i) { return sendBuff[i]+sendOffset(i); }
  __device__ uint32_t recvFlag(int i) { return NCCL_LL_FLAG(recvStep[i]+1); }
  __device__ uint32_t sendFlag(int i) { return NCCL_LL_FLAG(sendStep[i]+1); }

#if defined(__HIP_PLATFORM_HCC__) || defined(__HCC__) || defined(__HIPCC__)
#else
  // Exit If Abort Barrier : make sure all threads exit consistently
  // Each thread sets a predicate to true if val == 1
  // all CTA's threads enter the barrier and do a popc on their predicates being True
  // If any of the thread's predicate was True, all the threads call exit()
  __device__ void exitIfAbortLocalBarrier() {
    uint32_t popc;
    asm ("{");
    asm volatile ("   .reg .pred barr_pred;");
    asm volatile ("   setp.eq.u32 barr_pred,%0,1;" :: "r"(abort));
    asm volatile ("   bar.red.popc.u32 %0, 14, %1, barr_pred;" : "=r"(popc) : "r"(nthreads));
    asm ("}");
    if (popc) {
      // Make sure threads not participating in the operation get the abort and all threads exit
      exitIfAbortBarrier(1);
    }
  }
#endif

  __device__ void barrier() {
#if defined(__HIP_PLATFORM_HCC__) || defined(__HCC__) || defined(__HIPCC__)
    __syncthreads();
#else
    asm volatile ("bar.sync 1, %0;" :: "r"(nthreads));
#endif
  }

  uint32_t mismatch = 0;
  const uint64_t opCount;

  __device__ void checkMismatch(volatile uint64_t* remoteOpCount) {
    if (mismatch > 20) {
      // We have seen that the peer advanced opcount so many times yet we are still waiting for credit of current op, so it is _most likely_ a mismatch
      // Note that we are not using _threadfence_system in LL so the error cannot be asserted
      STORE(comm->fatalDevError, ncclDevSuspectedMismatch);
    } else if (remoteOpCount && LOAD(remoteOpCount) > opCount) {
      mismatch += 1;
    }
  }

  uint32_t spins = 0;
  uint32_t abort = 0;

  __device__ int checkAbort(volatile uint64_t* remoteOpCount) {
    spins++;
    if (spins == SPINS_BEFORE_CHECK_ABORT) {
      abort = LOAD(comm->abortFlag);
      checkMismatch(remoteOpCount);
      spins = 0;
    }
    return abort;
  }

  __device__ void waitSend(int i, int nbytes) {
    spins = 0;
    mismatch = 0;
    if (tid == WARP_SIZE+i) {
      while (sendConnHead + NCCL_STEPS < sendStep[i] + 1) {
        sendConnHead = LOAD(waitPtr);
        if (checkAbort(sendConn[i]->opCountRem)) break;
      }
      if (fifoPtr) {
        int size = ((sendStep[i] & NCCL_LL_CLEAN_MASK) == NCCL_LL_CLEAN_MASK) ? NCCL_LL_SLICE_LINES*sizeof(union ncclLLFifoLine) : nbytes;
        STORE(fifoPtr+sendStep[i]%NCCL_STEPS, size);
      }
    }
  }

  __device__ void postRecv(int i) {
    recvStep[i]++;
    if (tid == i) STORE(postPtr, recvStep[i]);
  }

  __device__ void postSend(int i, int offset) {
    // LL Cleanup : write all flags in the slice to make sure we don't have
    // data corruption when flag loops over.
    if ((sendStep[i] & NCCL_LL_CLEAN_MASK) == NCCL_LL_CLEAN_MASK) {
      for (int o = offset; o<NCCL_LL_SLICE_LINES; o+=nthreads) storeLL(sendPtr(i)+o, 0, sendFlag(i));
    }
    sendStep[i]++;
  }

  __device__ __attribute__((noinline)) uint64_t readLL(int i, int offset) {
    union ncclLLFifoLine* src = recvPtr(i) + offset;
    uint32_t flag = recvFlag(i);
    uint32_t data1, flag1, data2, flag2;
    spins = 0;
    mismatch = 0;
#if defined(__HIP_PLATFORM_HCC__) || defined(__HCC__) || defined(__HIPCC__)
    using Vec = uint32_t __attribute__((ext_vector_type(4)));
    Vec i4;
    do {
      asm volatile ("flat_load_dwordx4 %0, %1, glc\n"
        "s_waitcnt vmcnt(0)\n"
        "buffer_wbinvl1_vol\n" : "=v"(i4) : "v"(src));
      if (i4[1] == flag && i4[3] == flag) break;
    } while (!checkAbort(recvConn[i]->opCountRem));
    uint64_t val64 = (uint64_t)(i4[0]) + (((uint64_t)i4[2]) << 32);
#else
    do {
      asm volatile("ld.volatile.global.v4.u32 {%0,%1,%2,%3}, [%4];" : "=r"(data1), "=r"(flag1), "=r"(data2), "=r"(flag2) : "l"(&src->i4));
      if (checkAbort(recvConn[i]->opCountRem)) break;
    } while ((flag1 != flag) || (flag2 != flag));
    uint64_t val64 = data1 + (((uint64_t)data2) << 32);
#endif
    return val64;
  }

  __device__ __attribute__((noinline)) void storeLL(union ncclLLFifoLine* dst, uint64_t val, uint32_t flag) {
#if defined(__HIP_PLATFORM_HCC__) || defined(__HCC__) || defined(__HIPCC__)
  using Vec = uint32_t __attribute__((ext_vector_type(4)));
  Vec i4;
  i4[0] = val & 0xffffffff;
  i4[1] = flag;
  i4[2] = (val >> 32);
  i4[3] = flag;
  asm volatile ("flat_store_dwordx4 %0, %1, glc\n"
    "s_waitcnt vmcnt(0)\n"
    "buffer_wbinvl1_vol\n" : : "v"(dst), "v"(i4));
#else
    asm volatile("st.volatile.global.v4.u32 [%0], {%1,%2,%3,%4};" :: "l"(&dst->i4), "r"((uint32_t)val), "r"(flag), "r"((uint32_t)(val >> 32)), "r"(flag));
#endif
  }

  // Using memcpy handles misaligned pointers.
  __device__ uint64_t readAL(uint64_t* src) {
    uint64_t val;
    memcpy((char*)&val, (char*)src, sizeof(uint64_t));
    return val;
  }

  __device__ void storeAL(uint64_t* dst, uint64_t val, uint32_t nbytes) {
    memcpy((char*)dst, (char*)&val, nbytes);
  }

  template <int RECV, int SEND, int SRC, int DST>
  __device__ void LLGenericOp(const T* srcPtr, T* dstPtr, int nelem) {
    uint32_t nbytes = nelem < 0 ? 0 : nelem*sizeof(T);
    FOR_SEND(waitSend, nbytes*2);
    barrier();
    uint32_t npack = DIVUP(nbytes, sizeof(uint64_t));
    uint64_t* srcPack = (uint64_t*)srcPtr;
    uint64_t* dstPack = (uint64_t*)dstPtr;
    int offset = tid;
    // Do multiples of 64 bits
    #pragma unroll 1
    for (; offset<npack; offset+=nthreads) {
      // Recv : local, then intra-node, then inter-node
      uint64_t val = SRC ? readAL(srcPack+offset) : readLL(0, offset);
      if (RECV) {
        if (SRC) val = MULTI<FUNC, T>()(readLL(0, offset), val);
        for (int i=1; i<NRECV && i<nrecv; i++) {
          val = MULTI<FUNC, T>()(readLL(i, offset), val);
        }
      }

      // Send : inter-node, then intra-node, then local
      if (SEND) {
        for (int i=1; i<NSEND && i<nsend; i++) storeLL(sendPtr(i)+offset, val, sendFlag(i));
        storeLL(sendPtr(0)+offset, val, sendFlag(0));
      }
      if (DST) {
        if (((offset*sizeof(uint64_t)) ^ nbytes) < sizeof(uint64_t)) {
          // Last incomplete word
          storeAL(dstPack+offset, val, nbytes & 0x7);
        } else {
          storeAL(dstPack+offset, val, sizeof(uint64_t));
        }
      }
    }
#if defined(__HIP_PLATFORM_HCC__) || defined(__HCC__) || defined(__HIPCC__)
    exitIfAbortBarrier(abort, abortCount);
#else
    exitIfAbortLocalBarrier();
#endif
    FOR_RECV(postRecv);
    FOR_SEND(postSend, offset);
  }

  __device__ void loadRecvConn(struct ncclConnInfo* conn, int i) {
    recvConn[i] = conn;
    recvBuff[i] = recvConn[i]->llBuff;
    recvStep[i] = recvConn[i]->step;
    if (tid == i) {
      postPtr = recvConn[i]->head;
      STORE(recvConn[i]->opCountLoc, opCount);
    }
    nrecv++;
  }

  __device__ void loadSendConn(struct ncclConnInfo* conn, int i) {
    sendConn[i] = conn;
    sendBuff[i] = sendConn[i]->llBuff;
    sendStep[i] = sendConn[i]->step;
    if (tid == WARP_SIZE+i) {
      waitPtr = sendConn[i]->head;
      fifoPtr = sendConn[i]->fifo;
      sendConnHead = LOAD(waitPtr);
      STORE(sendConn[i]->opCountLoc, opCount);
    }
    nsend++;
  }

  __device__ void saveRecvConn(int i) {
    if (tid == i) {
      recvConn[i]->step = recvStep[i];
      __atomic_fetch_add(recvConn[i]->opCountLoc, 1, __ATOMIC_SEQ_CST);
      __threadfence_block();
    }
  }

  __device__ void saveSendConn(int i) {
    if (tid == WARP_SIZE+i) {
      sendConn[i]->step = sendStep[i];
      __atomic_fetch_add(sendConn[i]->opCountLoc, 1, __ATOMIC_SEQ_CST);
      __threadfence_block();
    }
  }

 public:
  __device__
  ncclLLPrimitives(const int tid, const int nthreads, int* recvPeers, int* sendPeers, struct ncclChannel* channel, struct ncclDevComm* comm, const uint64_t opCount)
    : comm(comm), tid(tid), nthreads(nthreads), opCount(opCount) {
    // Make sure step is updated before we read it.
    abortCount = channel->abortCount;
    barrier();

    for (int i=0; i<NRECV && recvPeers[i] >= 0; i++) loadRecvConn(&channel->devPeers[recvPeers[i]].recv.conn, i);
    for (int i=0; i<NSEND && sendPeers[i] >= 0; i++) loadSendConn(&channel->devPeers[sendPeers[i]].send.conn, i);
  }

  __device__ void send(const T* src, int nelem) {
    return LLGenericOp<0, 1, 1, 0>(src, NULL, nelem);
  }

  __device__ void recv(T* dst, int nelem) {
    return LLGenericOp<1, 0, 0, 1>(NULL, dst, nelem);
  }

  __device__ void recvReduceSend(const T* src, int nelem) {
    return LLGenericOp<1, 1, 1, 0>(src, NULL, nelem);
  }

  __device__ void recvReduceCopy(const T* src, T* dst, int nelem) {
    return LLGenericOp<1, 0, 1, 1>(src, dst, nelem);
  }

  __device__ void copySend(const T* src, T* dst, int nelem) {
    return LLGenericOp<0, 1, 1, 1>(src, dst, nelem);
  }

  __device__ void recvCopySend(T* dst, int nelem) {
    return LLGenericOp<1, 1, 0, 1>(NULL, dst, nelem);
  }

  __device__ void recvReduceCopySend(const T* src, T* dst, int nelem) {
    return LLGenericOp<1, 1, 1, 1>(src, dst, nelem);
  }

  __device__ ~ncclLLPrimitives() {
    // Save steps for the next operation
    for (int i=0; i<NRECV && i<nrecv; i++) saveRecvConn(i);
    for (int i=0; i<NSEND && i<nsend; i++) saveSendConn(i);
  }
};

#ifdef ENABLE_PROFILING
#define INIT_COUNTER \
  if (tid==0) { t0 = clock64(); ws = LOAD(&(devProf->wait_send_cycle[blockIdx.x])); \
    wr = LOAD(&(devProf->wait_recv_cycle[blockIdx.x])); }

#define ACCUMULATE_COUNTER(prim) \
  if (tid==0) { __atomic_fetch_add(&(devProf->prim##_cycle), clock64() - t0 \
    + ws - LOAD(&(devProf->wait_send_cycle[blockIdx.x])) \
    + wr - LOAD(&(devProf->wait_recv_cycle[blockIdx.x])), \
    __ATOMIC_SEQ_CST); \
    __atomic_fetch_add(&(devProf->prim##_byte), nelem * sizeof(T), __ATOMIC_SEQ_CST); }
#else
#define INIT_COUNTER
#define ACCUMULATE_COUNTER(prim)
#endif

#endif
