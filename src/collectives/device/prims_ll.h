/*************************************************************************
 * Copyright (c) 2016-2019, NVIDIA CORPORATION. All rights reserved.
 * Modifications Copyright (c) 2019 Advanced Micro Devices, Inc. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

template <typename T, int NRECV>
class ncclLLPrimitivesRecvData {
public:
  struct ncclConnInfo* recvConn = NULL;
  volatile uint64_t* recvConnHeadPtr = NULL;
  uint64_t recvConnHead;
  uint64_t recvStep[NRECV];
  union ncclLLFifoLine* recvBuff[NRECV];
};

template <typename T, int NSEND>
class ncclLLPrimitivesSendData {
public:
  struct ncclConnInfo* sendConn = NULL;
  volatile int* sendConnFifoPtr = NULL;
  volatile uint64_t* sendConnHeadPtr = NULL;
  uint64_t sendConnHead;
  uint64_t sendConnHeadCache; // Cache last seen value
  uint64_t sendStep[NSEND];
  union ncclLLFifoLine* sendBuff[NSEND];
};

template <typename T, class FUNC, int NRECV, int NSEND>
class ncclLLPrimitives {
 private:
  const int tid;
  const int nthreads;
  const int wid;
  int nrecv = 0;
  int nsend = 0;
  struct ncclDevComm* comm;

  typename std::conditional<NRECV == NCCL_MAX_TREE_ARITY,
    ncclLLPrimitivesRecvData<T, NRECV>&, ncclLLPrimitivesRecvData<T, NRECV>>::type r;
  typename std::conditional<NSEND == NCCL_MAX_TREE_ARITY,
    ncclLLPrimitivesSendData<T, NSEND>&, ncclLLPrimitivesSendData<T, NSEND>>::type s;

  inline __device__ int recvOffset(int i) { return (r.recvStep[i]%NCCL_STEPS)*NCCL_LL_SLICE_LINES; }
  inline __device__ int sendOffset(int i) { return (s.sendStep[i]%NCCL_STEPS)*NCCL_LL_SLICE_LINES; }
  inline __device__ union ncclLLFifoLine* recvPtr(int i) { return r.recvBuff[i]+recvOffset(i); }
  inline __device__ union ncclLLFifoLine* sendPtr(int i) { return s.sendBuff[i]+sendOffset(i); }
  inline __device__ uint32_t recvFlag(int i) { return NCCL_LL_FLAG(r.recvStep[i]+1); }
  inline __device__ uint32_t sendFlag(int i) { return NCCL_LL_FLAG(s.sendStep[i]+1); }

  inline __device__ void barrier() {
#if defined(__HIP_PLATFORM_HCC__) || defined(__HCC__) || defined(__HIPCC__)
    __syncthreads();
#else
    asm volatile ("bar.sync 1, %0;" :: "r"(nthreads));
#endif
  }

  uint32_t mismatch = 0;
  const uint64_t opCount;

  inline __device__ void checkMismatch(struct ncclConnInfo* conn) {
    if (mismatch > 20) {
      // We have seen that the peer advanced opcount so many times yet we are still waiting for credit of current op, so it is _most likely_ a mismatch
      // Note that we are not using _threadfence_system in LL so the error cannot be asserted
      STORE(comm->fatalDevError, ncclDevSuspectedMismatch);
    } else if (conn && LOAD(conn->opCountRem) > opCount) {
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
      while (s.sendConnHeadCache + NCCL_STEPS < s.sendConnHead + 1) {
        s.sendConnHeadCache = LOAD(s.sendConnHeadPtr);
        if (checkAbort(wid, 1)) break;
      }
      if (s.sendConnFifoPtr) {
        int size = ((s.sendConnHead & NCCL_LL_CLEAN_MASK) == NCCL_LL_CLEAN_MASK) ? NCCL_LL_SLICE_LINES*sizeof(union ncclLLFifoLine) : nbytes;
        STORE(s.sendConnFifoPtr+s.sendConnHead%NCCL_STEPS, size);
      }
      s.sendConnHead += 1;
    }
    barrier();
  }

  inline __device__ void incRecv(int i) {
    r.recvStep[i] += 1;
  }
  inline __device__ void postRecv() {
    barrier();
    if (r.recvConnHeadPtr) STORE(r.recvConnHeadPtr, r.recvConnHead += 1);
  }

  inline __device__ void incSend(int i, int offset) {
    // LL Cleanup : write all flags in the slice to make sure we don't have
    // data corruption when flag loops over.
    if ((s.sendStep[i] & NCCL_LL_CLEAN_MASK) == NCCL_LL_CLEAN_MASK) {
      for (int o = offset; o<NCCL_LL_SLICE_LINES; o+=nthreads) storeLL(sendPtr(i)+o, 0, sendFlag(i));
    }
    s.sendStep[i]++;
  }

  __device__ uint64_t readLL(int i, int offset) {
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
    } while ((i4[1] != flag) || (i4[3] != flag));
    uint64_t val64 = (uint64_t)(i4[0]) + (((uint64_t)i4[2]) << 32);
#else
    do {
      asm volatile("ld.volatile.global.v4.u32 {%0,%1,%2,%3}, [%4];" : "=r"(data1), "=r"(flag1), "=r"(data2), "=r"(flag2) : "l"(&src->i4));
      if (checkAbort(i, 0)) break;
    } while ((flag1 != flag) || (flag2 != flag));
    uint64_t val64 = data1 + (((uint64_t)data2) << 32);
#endif
    return val64;
  }

  __device__ void storeLL(union ncclLLFifoLine* dst, uint64_t val, uint32_t flag) {
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
    uint32_t npack = DIVUP(nbytes, sizeof(uint64_t));
    uint64_t* srcPack = (uint64_t*)srcPtr;
    uint64_t* dstPack = (uint64_t*)dstPtr;
    int offset = tid;

    // Always waitSend in case of cleanup
    if (SEND) waitSend(npack*sizeof(union ncclLLFifoLine));

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
    FOR_RECV(incRecv); if (RECV) postRecv();
    FOR_SEND(incSend, offset);
  }

  __device__ __forceinline__ void loadRecvConn(struct ncclConnInfo* conn, int i) {
    r.recvBuff[i] = LOAD(&conn->llBuff);
    r.recvStep[i] = LOAD(&conn->step);
    if (wid == i) r.recvConn = conn;
    nrecv++;
  }
  __device__ __forceinline__ void loadRecvSync() {
    if (tid >= nthreads-WARP_SIZE && wid < nrecv) {
      r.recvConnHeadPtr = LOAD(&r.recvConn->head);
      r.recvConnHead = LOAD(&r.recvConn->step);
      // Update opCount in case we skipped some operations
      STORE(r.recvConn->opCountLoc, opCount);
    }
  }

  __device__ __forceinline__ void loadSendConn(struct ncclConnInfo* conn, int i) {
    s.sendBuff[i] = LOAD(&conn->llBuff);
    s.sendStep[i] = LOAD(&conn->step);
    if (wid == i) s.sendConn = conn;
    nsend++;
  }
  __device__ __forceinline__ void loadSendSync() {
    if (tid < nsend) {
      s.sendConnHeadPtr = LOAD(&s.sendConn->head);
      s.sendConnHeadCache = LOAD(s.sendConnHeadPtr);
      s.sendConnHead = LOAD(&s.sendConn->step);
      s.sendConnFifoPtr = LOAD(&s.sendConn->fifo);
      STORE(s.sendConn->opCountLoc, opCount);
    }
  }

  __device__ __forceinline__ void saveRecvSync() {
    if (tid >= nthreads-WARP_SIZE && wid < nrecv) {
      STORE(&r.recvConn->step, r.recvConnHead);
      STORE(r.recvConn->opCountLoc, opCount+1);
      __threadfence_block();
    }
  }

  __device__ __forceinline__ void saveSendSync() {
    if (tid < nsend) {
      STORE(&s.sendConn->step, s.sendConnHead);
      STORE(s.sendConn->opCountLoc, opCount+1);
      __threadfence_block();
    }
  }

  inline __device__ void init(int* recvPeers, int* sendPeers, struct ncclChannel* channel) {
    // Make sure step is updated before we read it.
    barrier();

    for (int i=0; i<NRECV && recvPeers[i] >= 0; i++) loadRecvConn(&channel->devPeers[recvPeers[i]].recv.conn, i);
    for (int i=0; i<NSEND && sendPeers[i] >= 0; i++) loadSendConn(&channel->devPeers[sendPeers[i]].send.conn, i);
    loadRecvSync();
    loadSendSync();
  }

 public:
  __device__ __forceinline__
  ncclLLPrimitives(const int tid, const int nthreads, int* recvPeers, int* sendPeers, struct ncclChannel* channel, struct ncclDevComm* comm, const uint64_t opCount)
    : comm(comm), tid(tid), nthreads(nthreads), wid(tid%WARP_SIZE), opCount(opCount) {
    init(recvPeers, sendPeers, channel);
  }

  __device__ __forceinline__
  ncclLLPrimitives(const int tid, const int nthreads, int* recvPeers, int* sendPeers, struct ncclChannel* channel, struct ncclDevComm* comm, const uint64_t opCount, ncclLLPrimitivesRecvData<T, NRECV>& r)
    : comm(comm), tid(tid), nthreads(nthreads), wid(tid%WARP_SIZE), opCount(opCount), r(r) {
    init(recvPeers, sendPeers, channel);
  }

  __device__ __forceinline__
  ncclLLPrimitives(const int tid, const int nthreads, int* recvPeers, int* sendPeers, struct ncclChannel* channel, struct ncclDevComm* comm, const uint64_t opCount, ncclLLPrimitivesSendData<T, NSEND>& s)
    : comm(comm), tid(tid), nthreads(nthreads), wid(tid%WARP_SIZE), opCount(opCount), s(s) {
    init(recvPeers, sendPeers, channel);
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

  __device__ __forceinline__ ~ncclLLPrimitives() {
    // Save steps for the next operation
    saveRecvSync();
    saveSendSync();
  }
};
