/*************************************************************************
 * Copyright (c) 2016-2021, NVIDIA CORPORATION. All rights reserved.
 * Modifications Copyright (c) 2019-2021 Advanced Micro Devices, Inc. All rights reserved.
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

#define barrier_by_group() do { \
  const int w = threadIdx.x/WARP_SIZE; \
  const int wid = threadIdx.x%WARP_SIZE; \
  if (wid == 0) { \
    barrier_next[w] += nthreads/WARP_SIZE; \
    __atomic_fetch_add(barriers, 1, __ATOMIC_SEQ_CST); \
    while (LOAD(barriers) < barrier_next[w]) /* spin */; \
  } \
} while (0)

#define ROLE_SRC       0x01
#define ROLE_DST       0x02
#define ROLE_WAIT_RECV 0x04
#define ROLE_WAIT_SEND 0x08
#define ROLE_POST_SEND 0x10
#define ROLE_POST_RECV 0x20

// Connection index is used to select P2P and NET and needs to be passed into ncclPrimitives constructor.
// To avoid adding another parameter which requires changes to every places ncclPrimitives are constructed,
// we pack group (max 7) and connection index (max 2) to original group which is 32-bit.
#define PACK_GROUP(gr, idx) (gr | (idx<<16))
#define TO_GR(group) (group&0xffff)
#define TO_IDX(group) (group>>16)

// Implementation of primitive types
template <int UNROLL, int SLICESPERCHUNK, int SLICESTEPS, typename T, int NRECV, int NSEND, int DIRECT, class FUNC>
class ncclPrimitives {
 private:
  const int tid;
  int nthreads;
  int nworkers;
  const int stepSize;
  int nrecv = 0;
  int nsend = 0;
  struct ncclConnInfo* conn = NULL;
  volatile int* connSizesFifoPtr = NULL;
  void** connPtrsFifoPtr = NULL;
  volatile uint64_t* connHeadPtr = NULL;
  volatile uint64_t* connTailPtr = NULL;
  uint64_t connTailCache; // Cache last seen value
  uint64_t connHeadCache; // Cache last seen value

  int index; // Peer index I'm responsible for
  int peer = -1;
  int role = 0;
  int group;
  uint64_t step;
  T* direct = NULL;
  T* buff;
  struct ncclDevComm* comm;
  const int connIndex;

  const T** srcs;
  T** dsts;

  uint64_t* barriers;
  uint64_t* barrier_next;
  // Don't use barrier 0 as it's used by the final sync
  inline __device__ void barrier() {
#if defined(__HIP_PLATFORM_HCC__) || defined(__HCC__) || defined(__HIPCC__)
    if (nthreads == WARP_SIZE) __syncwarp();
    else barrier_by_group();
#else
    if (nthreads == WARP_SIZE) __syncwarp();
    else asm volatile ("bar.sync %0, %1;" :: "r"(group+1), "r"(nthreads));
#endif
  }

  inline __device__ void subBarrier() {
#if defined(__HIP_PLATFORM_HCC__) || defined(__HCC__) || defined(__HIPCC__)
    barrier();
#else
    if (nworkers == nthreads) barrier();
    else asm volatile ("bar.sync %0, %1;" :: "r"(group+2), "r"(nworkers));
#endif
  }

  uint32_t spins = 0;
  uint32_t abort = 0;

  inline __device__ int checkAbort() {
    spins++;
    if (abort == 0 && spins == SPINS_BEFORE_CHECK_ABORT) {
      abort = LOAD(comm->abortFlag);
      spins = 0;
    }
    return abort;
  }

  template <int DIRECTPTR>
  inline __device__ T* directPtr(ssize_t directOffset) {
    return DIRECTPTR && direct ? direct+directOffset : buff+(step%NCCL_STEPS)*stepSize;
  }

  template <int DST, int DIRECTSEND>
  inline __device__ void waitSend(ssize_t directOffset, int nbytes) {
    spins = 0;
    while (connHeadCache + NCCL_STEPS < step + SLICESTEPS) {
      connHeadCache = LOAD(connHeadPtr);
      if (checkAbort()) break;
    }
    if (connSizesFifoPtr) {
      STORE(connSizesFifoPtr+step%NCCL_STEPS, nbytes);
    }

    if (connPtrsFifoPtr) dsts[DST+index] = (T *)LOAD(connPtrsFifoPtr+step%NCCL_STEPS);
    else dsts[DST+index] = directPtr<DIRECTSEND>(directOffset);
    step += SLICESTEPS;
  }

  template <int SRC, int DIRECTRECV>
  inline __device__ void waitRecv(ssize_t directOffset) {
    spins = 0;
#ifdef ENABLE_PROFILING
    uint64_t t0 = __builtin_amdgcn_s_memrealtime();
#endif
    while (connTailCache < step + SLICESTEPS) {
      connTailCache = LOAD(connTailPtr);
      if (checkAbort()) break;
    }
#ifdef ENABLE_PROFILING
    if (tid == 0) __atomic_fetch_add(&comm->devProf->wait_recv_cycle[blockIdx.x], __builtin_amdgcn_s_memrealtime() - t0, __ATOMIC_SEQ_CST);
#endif
    if (connPtrsFifoPtr) srcs[SRC+index] = (const T *)LOAD(connPtrsFifoPtr+step%NCCL_STEPS);
    else srcs[SRC+index] = directPtr<DIRECTRECV>(directOffset);
    step += SLICESTEPS;
  }

  inline __device__ void postRecv() {
    STORE(connHeadPtr, step += SLICESTEPS);
  }

  inline __device__ void postSend() {
    if (conn->next_hdp_reg) STORE(conn->next_hdp_reg, 0x1);
    STORE(connTailPtr, step += SLICESTEPS);
  }

  template <int DIRECTRECV, int DIRECTSEND, int RECV, int SEND, int SRC, int DST>
  inline __device__ void
  GenericOp(const T* srcPtr, T* dstPtr, int nelem, ssize_t directOffset) {
    int offset = 0;
    int sliceSize = stepSize*SLICESTEPS;
    int dataSize = max(DIVUP(nelem, 16*SLICESPERCHUNK)*16, sliceSize/32);

    #pragma unroll
    for (int slice=0; slice<SLICESPERCHUNK; ++slice) {
      int realSize = max(0, min(dataSize, nelem-offset));
#ifdef ENABLE_PROFILING
      uint64_t t0 = __builtin_amdgcn_s_memrealtime();
#endif
      if (tid < nworkers) {
        if (SRC && (role & ROLE_SRC)) srcs[0] = srcPtr+offset;
        if (RECV && (role & ROLE_WAIT_RECV)) waitRecv<SRC, DIRECTRECV>(directOffset+offset);
        if (DST && (role & ROLE_DST)) dsts[0] = dstPtr+offset;
        if (SEND && (role & ROLE_WAIT_SEND)) waitSend<DST, DIRECTSEND>(directOffset+offset, realSize*sizeof(T));
        if (realSize > 0) {
#ifdef ENABLE_PROFILING
          if (tid == 0) __atomic_fetch_add(&comm->devProf->wait_cycle[blockIdx.x], __builtin_amdgcn_s_memrealtime() - t0, __ATOMIC_SEQ_CST);
#endif
          subBarrier();
          ReduceOrCopyMulti<UNROLL, FUNC, T, RECV+SRC, RECV*NRECV+SRC, SEND+DST, SEND*NSEND+DST>(tid, nworkers, RECV*nrecv+SRC, srcs, SEND*nsend+DST, dsts, realSize);
        }
      }
      barrier();
      if (SEND && (role & ROLE_POST_SEND) && realSize > 0 && index == 0) __threadfence_system();
      __syncwarp();
      if (SEND && (role & ROLE_POST_SEND)) postSend();
      if (RECV && (role & ROLE_POST_RECV)) postRecv();
      offset += realSize;
    }
  }

  // Scatter and gather do not support DIRECT
  template <int RECV, int SEND>
  inline __device__ void
  ScatterGatherOp(const T* srcPtr, T* dstPtr, int totalElem, int peerElem, int skip, int shift) {
    int offset = 0; // slice offset
    int sliceSize = stepSize*SLICESTEPS;
    int dataSize = max(DIVUP(peerElem, 16*SLICESPERCHUNK)*16, sliceSize/32);  // per-peer slice size

    #pragma unroll
    for (int slice=0; slice<SLICESPERCHUNK; ++slice) {
      int realSize = max(0, min(dataSize, peerElem-offset));
      if (tid < nworkers) {
        if (RECV && (role & ROLE_WAIT_RECV)) waitRecv<0, 0>(0);
        // realSize is not accurate here; but intra-node does not rely on sizes FIFO
        if (SEND && (role & ROLE_WAIT_SEND)) waitSend<0, 0>(0, realSize*sizeof(T));
        subBarrier();
        if (SEND) {
          #pragma unroll 1
          for (int j=0; j<nsend; j++) {
            int i = (j+shift)%nsend;
            int peerOffset = i*peerElem + offset;
            if (skip >=0 && i >= skip) peerOffset += peerElem;
            const T* src0 = srcPtr + peerOffset;
            int realPeerSize = min(realSize, totalElem-peerOffset);
            if (realPeerSize > 0) ReduceOrCopyMulti<UNROLL, FUNC, T, 1, 1, 1, 1>(tid, nworkers, 1, &src0, 1, dsts+i, realPeerSize);
          }
        } else if (RECV) {
          #pragma unroll 1
          for (int j=0; j<nrecv; j++) {
            int i = (j+shift)%nrecv;
            int peerOffset = i*peerElem + offset;
            if (skip >= 0 && i >= skip) peerOffset += peerElem;
            T* dst0 = dstPtr + peerOffset;
            int realPeerSize = min(realSize, totalElem-peerOffset);
            if (realPeerSize > 0) ReduceOrCopyMulti<UNROLL, FUNC, T, 1, 1, 1, 1>(tid, nworkers, 1, srcs+i, 1, &dst0, realPeerSize);
          }
        }
      }
      barrier();
      if (SEND && (role & ROLE_POST_SEND) && realSize > 0 && index == 0) __threadfence_system();
      __syncwarp();
      if (SEND && (role & ROLE_POST_SEND)) postSend();
      if (RECV && (role & ROLE_POST_RECV)) postRecv();
      offset += realSize;
    }
  }

  __device__ __forceinline__ void loadRecvConn(struct ncclChannel* channel, T* directBuff) {
    if (role & (ROLE_WAIT_RECV|ROLE_POST_RECV)) {
      // For oneshot: groups 0,1 use conn 0, groups 2,3 use conn 1
      conn = &channel->devPeers[peer].recv[connIndex].conn;
      step = conn->step;
      step = ROUNDUP(step, SLICESPERCHUNK*SLICESTEPS);
      if (role & ROLE_POST_RECV) {
        connHeadPtr = conn->head;
        // Return credits in case we rounded up.
        STORE(connHeadPtr, step);
      }
      if (role & ROLE_WAIT_RECV) {
        buff = (T*)conn->buffs[NCCL_PROTO_SIMPLE];
        //if (DIRECT && (conn->direct & NCCL_DIRECT_GPU)) {
        //  direct = directBuff;
        //  *conn->ptrExchange = directBuff;
        //}
        connTailPtr = conn->tail;
        connTailCache = LOAD(connTailPtr);
        connPtrsFifoPtr = conn->ptrsFifo;
      }
    }
  }

  __device__ __forceinline__ void loadSendConn(struct ncclChannel* channel) {
    if (role & (ROLE_WAIT_SEND|ROLE_POST_SEND)) {
      // For oneshot: groups 0,1 use conn 0, groups 2,3 use conn 1
      conn = &channel->devPeers[peer].send[connIndex].conn;
      step = conn->step;
      step = ROUNDUP(step, SLICESPERCHUNK*SLICESTEPS);
      if (role & ROLE_POST_SEND) {
        connTailPtr = conn->tail;
      }
      if (role & ROLE_WAIT_SEND) {
        buff = (T*)conn->buffs[NCCL_PROTO_SIMPLE];
#if 0
        if (DIRECT && (conn->direct & NCCL_DIRECT_GPU)) {
          void* volatile* ptr = conn->ptrExchange;
          while ((direct = (T*)(*ptr)) == NULL) { if (checkAbort()) break; }
          *ptr = NULL;
        }
#endif
        connHeadPtr = conn->head;
        connHeadCache = LOAD(connHeadPtr);
        connSizesFifoPtr = conn->sizesFifo;
        connPtrsFifoPtr = conn->ptrsFifo;
      }
    }
  }

  __device__ __forceinline__ void saveSync() {
    if (role & (ROLE_POST_SEND|ROLE_POST_RECV)) {
      conn->step = step;
      __threadfence_system();
    }
  }

 public:
  __device__ __forceinline__
  ncclPrimitives(const int tid, const int nworkers, int* recvPeers, int* sendPeers, T* directBuff, int stepSize, struct ncclChannel* channel, struct ncclDevComm* comm, struct ncclShmemPtrs* ptrs, int group)
    : comm(comm), tid(tid), nworkers(nworkers), stepSize(stepSize), srcs((const T**)ptrs[TO_GR(group)].srcs), dsts((T**)ptrs[TO_GR(group)].dsts),
    group(TO_GR(group)), barriers(&ptrs[TO_GR(group)].barrier), barrier_next(ptrs[TO_GR(group)].barrier_next),
    connIndex((NSEND == NCCL_MAX_DIRECT_ARITY || NRECV == NCCL_MAX_DIRECT_ARITY) ? TO_GR(group)/2 : TO_IDX(group)) {
    nthreads = nworkers;
    // For send operations, we need an extra warp to overlap the threadfence and the copy
    // int postThreads = NSEND && nworkers >= 64 ? WARP_SIZE : 0;
    // nthreads += postThreads;

    // Make sure step is updated before we read it.
    barrier();

    for (int i=0; i<NRECV; i++) if (recvPeers[i] != -1) nrecv++;
    for (int i=0; i<NSEND; i++) if (sendPeers[i] != -1) nsend++;

    #define SYNC_GROUP 8
    static_assert(NSEND < SYNC_GROUP && NRECV < SYNC_GROUP, "Not enough threads to cover all peers");

    int g = tid / SYNC_GROUP;
    int ng = nthreads / SYNC_GROUP;
    index = tid % SYNC_GROUP;

    if (g == 0) {
      if (index < nrecv) role |= ROLE_WAIT_RECV;
      if (index == nrecv) role |= ROLE_SRC;
    } else if (g == 1) {
      if (index < nsend) role |= ROLE_WAIT_SEND;
      if (index == nsend) role |= ROLE_DST;
    } else if (g == ng - 2) {
      if (index < nrecv) role |= ROLE_POST_RECV;
    } else if (g == ng - 1) {
      if (index < nsend) role |= ROLE_POST_SEND;
    }

    if (role & (ROLE_WAIT_RECV|ROLE_POST_RECV)) peer = recvPeers[index];
    if (role & (ROLE_WAIT_SEND|ROLE_POST_SEND)) peer = sendPeers[index];

    loadRecvConn(channel, directBuff);
    loadSendConn(channel);
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

  __device__ __forceinline__ void
  scatter(const T* src, int totalElem, int peerElem, int skip, int shift) {
    ScatterGatherOp<0, 1>(src, NULL, totalElem, peerElem, skip, shift);
  }

  __device__ __forceinline__ void
  gather(T* dst, int totalElem, int peerElem, int skip, int shift) {
    ScatterGatherOp<1, 0>(NULL, dst, totalElem, peerElem, skip, shift);
  }

  __device__ __forceinline__ ~ncclPrimitives() {
    // Save steps for the next operation
    saveSync();
  }
};

#include "prims_ll.h"
//#include "prims_ll128.h"

#ifdef ENABLE_PROFILING
#define INIT_COUNTER \
  if (tid == 0) { t0 = __builtin_amdgcn_s_memrealtime(); ws = LOAD(&(devProf->wait_cycle[blockIdx.x])); }

#define ACCUMULATE_COUNTER(prim) \
  if (tid == 0) { __atomic_fetch_add(&(devProf->prim##_cycle), __builtin_amdgcn_s_memrealtime() - t0 \
    + ws - LOAD(&(devProf->wait_cycle[blockIdx.x])), __ATOMIC_SEQ_CST); \
    __atomic_fetch_add(&(devProf->prim##_byte), nelem * sizeof(T), __ATOMIC_SEQ_CST); }
#else
#define INIT_COUNTER
#define ACCUMULATE_COUNTER(prim)
#endif

#endif
