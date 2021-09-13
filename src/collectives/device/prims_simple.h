/*************************************************************************
 * Copyright (c) 2016-2021, NVIDIA CORPORATION. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

template<typename T, typename RedOp, typename Fan, int Direct,
         int SlicePerChunk, int StepPerSlice, int Unroll>
class Primitives<
    T, RedOp, Fan, Direct, ProtoSimple<SlicePerChunk, StepPerSlice, Unroll>
  > {
  static constexpr int MaxRecv = Fan::MaxRecv, MaxSend = Fan::MaxSend;
  static constexpr int Input=0, Output=1;
  static constexpr int RoleInput = 0x01,
                       RoleOutput = 0x02,
                       RoleWaitRecv = 0x04,
                       RoleWaitSend = 0x08,
                       RolePostSend = 0x10,
                       RolePostRecv = 0x20,
                       Aborted = 0x40,
                       PtrsFifoEnabled = 0x80,
                       SizesFifoEnabled = 0x100,
                       DirectEnabled = 0x200,
                       ThreadsSynced = 0x400;
  const int tid;
  int nthreads;
  int nworkers;
  const int stepSize;
  Fan fan;
  RedOp const redOp;
  int index; // Peer index I'm responsible for
  int flags;
  int group;
  uint64_t step;
  union {
    void **connPtrsFifoPtr; // (flags & PtrsFifoEnabled)
    T *userBuff;            // (flags & (RoleInput|RoleOutput))
    T *connEltsFifo;        // !(flags & (PtrsFifoEnabled|RoleInput|RoleOutput))
  };
  union {
    int volatile *connSizesFifoPtr; //  (flags & SizesFifoEnabled)
    T *directBuff;                  // !(flags & SizesFifoEnabled)
  };
  uint64_t volatile *connStepPtr;
  uint64_t connStepCache; // Cache last seen value of (*connStepPtr)
  uint64_t* barriers;
  uint64_t* barrier_next;
  const int connIndex;

  // Don't use barrier 0 as it's used by the final sync
  inline __device__ void barrier() {
#if defined(__HIP_PLATFORM_HCC__) || defined(__HCC__) || defined(__HIPCC__)
    if (nthreads == WARP_SIZE)
      __syncwarp();
    else
      barrier_by_group();
#else
    if (nthreads == WARP_SIZE)
      __syncwarp();
    else
      asm volatile("bar.sync %0, %1;" :: "r"(group+1), "r"(nthreads));
#endif
    flags |= ThreadsSynced;
  }
  inline __device__ void subBarrier() {
#if defined(__HIP_PLATFORM_HCC__) || defined(__HCC__) || defined(__HIPCC__)
    barrier();
#else
    if (nworkers == nthreads)
      barrier();
    else
      asm volatile("bar.sync %0, %1;" :: "r"(group+2), "r"(nworkers));
#endif
  }

  inline __device__ bool checkAbort(int &spins) {
    spins++;
    if (!(flags & Aborted) && spins == NCCL_SPINS_BEFORE_CHECK_ABORT) {
      flags |= LOAD(ncclShmem->comm.abortFlag) ? Aborted : 0;
      spins = 0;
    }
    return flags & Aborted;
  }

  template <int DirectRecv, int DirectSend, int Recv, int Send, int Src, int Dst>
  inline __device__ void waitPeer(intptr_t dstIx, intptr_t remoteOutIx, int offset, int nelts) {
    if (flags & (Recv*RoleWaitRecv | Send*RoleWaitSend)) {
      bool const isSendNotRecv = (Send && Recv) ? (flags & RoleWaitSend) : Send;
      int spins = 0;
      while (connStepCache + (isSendNotRecv ? NCCL_STEPS : 0) < step + StepPerSlice) {
        connStepCache = LOAD(connStepPtr);
        if (checkAbort(spins)) break;
        //if (spins == 0) printf("r=%d b=%d t=%d SPUN OUT got=%d want=%d\n", ncclShmem->comm.rank, blockIdx.x, threadIdx.x, int(connStepCache + (isSendNotRecv ? NCCL_STEPS : 0)), int(step+StepPerSlice));
      }

      if (isSendNotRecv && (flags & SizesFifoEnabled))
        STORE(connSizesFifoPtr+step%NCCL_STEPS, nelts*sizeof(T));

      void **ptrs = isSendNotRecv ? (ncclShmem->groups[group].dsts + Dst)
                                  : (ncclShmem->groups[group].srcs + Src);
      if (flags & PtrsFifoEnabled)
        loadPtr(connPtrsFifoPtr + step%NCCL_STEPS, ptrs[index]);
      else if ((isSendNotRecv ? DirectSend : DirectRecv) && (flags & DirectEnabled))
        ptrs[index] = directBuff + (isSendNotRecv ? remoteOutIx : dstIx) + offset;
      else
        ptrs[index] = connEltsFifo + (step%NCCL_STEPS)*stepSize;
      step += StepPerSlice;
    }
  }

  template<int Recv, int Send>
  inline __device__ void postPeer() {
    if (flags & (Recv*RolePostRecv | Send*RolePostSend)) {
      step += StepPerSlice;
      STORE(connStepPtr, step);
    }
  }

  template <int DirectRecv1, int DirectSend1, int Recv, int Send, int SrcBuf, int DstBuf>
  inline __device__ void genericOp(
      intptr_t srcIx, intptr_t dstIx, intptr_t remoteOutIx, int nelem, bool postOp
    ) {
    constexpr int DirectRecv = 1 && Direct && DirectRecv1;
    constexpr int DirectSend = 1 && Direct && DirectSend1;
    constexpr int Src = SrcBuf != -1;
    constexpr int Dst = DstBuf != -1;

    nelem = nelem < 0 ? 0 : nelem;
    int sliceSize = stepSize*StepPerSlice;
    sliceSize = max(divUp(nelem, 16*SlicePerChunk)*16, sliceSize/32);
    int slice = 0;
    int offset = 0;

    if (tid < nworkers && offset < nelem) {
      // Worker-only loop for non-empty slices. Non-workers and empty slices are
      // processed in the loop following this if block. The benefit of splitting
      // the loop like this is we pull two branches out of the critical path.
      // Using "number of branch insns (taken or not) encountered dynamically"
      // as the performance metric, then:
      //   perf_orig = 2*numslices
      //   perf_new = 2+numslices
      // So the new code and old code behave the same for numslices=2, and for
      // numslices>2 the new code is superior. And note that in the case
      // numslices=1, the loop is trivially unrollable (single iteration) so we
      // don't incur that that tail branch and we still have perf_new=2.
      //
      // ORIGINAL CODE:
      //   unrolled for(slices) {
      //     if(worker) { // This branch removed
      //       wait();
      //       subBarrier();
      //       if(slice not empty) // This branch removed
      //         ReduceCopyMulti();
      //     }
      //     barrier();
      //     post();
      //   } // Since we no longer unroll, new branch added here
      #pragma unroll 1
      do {
        sliceSize = sliceSize < nelem-offset ? sliceSize : nelem-offset;
        if (Src && (flags & (SrcBuf==Input ? RoleInput : RoleOutput)))
          ncclShmem->groups[group].srcs[0] = userBuff + srcIx + offset;
        if (Dst && (flags & (DstBuf==Input ? RoleInput : RoleOutput)))
          ncclShmem->groups[group].dsts[0] = userBuff + dstIx + offset;
#ifdef ENABLE_PROFILING
        uint64_t t0;
        if (tid == 0) t0 = __builtin_amdgcn_s_memrealtime();
#endif
        waitPeer<DirectRecv, DirectSend, Recv, Send, Src, Dst>(dstIx, remoteOutIx, offset, sliceSize);
        subBarrier();
#ifdef ENABLE_PROFILING
        if (tid == 0) ncclShmem->comm.devProf->elems[blockIdx.x].wait_cycle += (__builtin_amdgcn_s_memrealtime() - t0);
#endif
        if (DirectRecv && ncclShmem->groups[group].srcs[0] == ncclShmem->groups[group].dsts[0]) {
          // We can only have one direct receive. Since srcs[0] == dstPtr+offset, skip one copy
          if (Send) {
            // (1-Send) is only there to avoid compilation errors in case MaxSend=0 (and Send=0).
            ReduceOrCopyMulti<Unroll, RedOp, T, 1, 1, 1, (1-Send)+MaxSend>
              (tid, nworkers, redOp, 0, false,
               1, (T const**)ncclShmem->groups[group].srcs,
               fan.nsend(), (T**)ncclShmem->groups[group].dsts+1,
               sliceSize);
          }
        } else {
          ReduceOrCopyMulti<Unroll, RedOp, T, Recv+Src, Recv*MaxRecv+Src, Send+Dst, Send*MaxSend+Dst>
            (tid, nworkers, redOp, SrcBuf==Input ? 1 : 0, postOp,
             Recv*fan.nrecv()+Src, (T const**)ncclShmem->groups[group].srcs,
             Send*fan.nsend()+Dst, (T**)ncclShmem->groups[group].dsts,
             sliceSize);
        }
        barrier(); // This barrier has a counterpart in following loop
        if (Send && (flags & RolePostSend) && index == 0) __threadfence_system();
        __syncwarp();
        postPeer<Recv, Send>();
        offset += sliceSize;
        slice += 1;
      } while (slice < SlicePerChunk && offset < nelem);
    }

    // Non-workers come straight here. Workers too but only once the remaining
    // slices are all empty. Since empty slices are the uncommon case, and
    // worker perf is the limiter, perf-wise this loop is effectively unentered,
    // hence just a single branch insn.
    #pragma unroll 1
    while (slice < SlicePerChunk) {
      sliceSize = sliceSize < nelem-offset ? sliceSize : nelem-offset;
      { // Only workers could have Wait roles so we know the slice must be empty
        // since we've exited the loop above.
        waitPeer<DirectRecv, DirectSend, Recv, Send, Src, Dst>(0, 0, 0, 0);
      }
      barrier(); // Has couterpart in preceding worker-only loop.
      if (Send && (flags & RolePostSend) && sliceSize > 0 && index == 0) __threadfence_system();
      __syncwarp();
      postPeer<Recv, Send>();
      offset += sliceSize;
      slice += 1;
    }
  }

  // Scatter and gather do not support Direct
  template <int Recv, int Send>
  inline __device__ void
  ScatterGatherOp(intptr_t inpIx, intptr_t outIx, int totalElem, int peerElem, int skip, int shift, bool postOp) {
    int offset = 0; // slice offset
    int sliceSize = stepSize*StepPerSlice;
    int dataSize = max(DIVUP(peerElem, 16*SlicePerChunk)*16, sliceSize/32);  // per-peer slice size

    #pragma unroll 1
    for (int slice=0; slice<SlicePerChunk; ++slice) {
      int realSize = max(0, min(dataSize, peerElem-offset));
      if (tid < nworkers) {
        if (Send && (flags & RoleInput)) ncclShmem->groups[group].srcs[0] = userBuff + inpIx + offset;
        if (Recv && (flags & RoleOutput)) ncclShmem->groups[group].dsts[0] = userBuff + outIx + offset;
        // realSize is not accurate here; but intra-node does not rely on sizes FIFO
        waitPeer<0, 0, Recv, Send, 0, 0>(0, 0, 0, realSize);
        subBarrier();
        if (Send) {
          #pragma unroll 1
          for (int j=0; j<fan.nsend(); j++) {
            int i = (j+shift)%fan.nsend();
            int peerOffset = i*peerElem;
            if (skip >= 0 && i >= skip) peerOffset += peerElem;
            const T* src0 = (T*)ncclShmem->groups[group].srcs[0] + peerOffset;
            int realPeerSize = min(realSize, totalElem-peerOffset);
            if (realPeerSize > 0) ReduceOrCopyMulti<Unroll, RedOp, T, 1, 1, 1, 1>(tid, nworkers, redOp, 1, false, 1, &src0, 1, (T**)ncclShmem->groups[group].dsts+i, realPeerSize);
          }
        } else if (Recv) {
          #pragma unroll 1
          for (int j=0; j<fan.nrecv(); j++) {
            int i = (j+shift)%fan.nrecv();
            int peerOffset = i*peerElem;
            if (skip >= 0 && i >= skip) peerOffset += peerElem;
            T* dst0 = (T*)ncclShmem->groups[group].dsts[0] + peerOffset;
            int realPeerSize = min(realSize, totalElem-peerOffset);
            if (realPeerSize > 0) ReduceOrCopyMulti<Unroll, RedOp, T, 1, 1, 1, 1>(tid, nworkers, redOp, 0, postOp, 1, (T const**)ncclShmem->groups[group].srcs+i, 1, &dst0, realPeerSize);
          }
        }
      }
      barrier();
      if (Send && (flags & RolePostSend) && realSize > 0 && index == 0) __threadfence_system();
      __syncwarp();
      postPeer<Recv, Send>();
      offset += realSize;
    }
  }

  __device__ __forceinline__ void loadRecvConn(ncclPeer *peer) {
    if (flags & (RoleWaitRecv|RolePostRecv)) {
      // For other colls: group <= 2, hence always use conn 0
      // For P2P: Direct is set to 1, hence always use conn 0
      // Ideally we should be accepting connIndex from the constructor!
      auto *conn = &peer->recv[connIndex].conn;
      step = conn->step;
      step = roundUp(step, SlicePerChunk*StepPerSlice);
      if (flags & RolePostRecv) {
        connStepPtr = conn->head;
        STORE(connStepPtr, step); // Return credits in case we rounded up.
      }
      if (flags & RoleWaitRecv) {
        ncclShmem->groups[group].recvConns[index] = conn; // WaitRecv role saves since that's who needs it in setDataPtrs()
        connStepPtr = conn->tail;
        connStepCache = LOAD(connStepPtr);
        flags |= (conn->ptrsFifo != nullptr) ? PtrsFifoEnabled : 0;
        flags |= (Direct && (conn->direct & NCCL_DIRECT_GPU)) ? DirectEnabled : 0;
        if (flags & PtrsFifoEnabled)
          connPtrsFifoPtr = conn->ptrsFifo;
        else
          connEltsFifo = (T*)conn->buffs[NCCL_PROTO_SIMPLE];
      }
    }
  }

  __device__ __forceinline__ void loadSendConn(ncclPeer *peer) {
    if (flags & (RoleWaitSend|RolePostSend)) {
      // For other colls: group <= 2, hence always use conn 0
      // For P2P: Direct is set to 1, hence always use conn 0
      // Ideally we should be accepting connIndex from the constructor!
      auto *conn = &peer->send[connIndex].conn;
      step = conn->step;
      step = roundUp(step, SlicePerChunk*StepPerSlice);
      if (flags & RolePostSend) {
        connStepPtr = conn->tail;
      }
      if (flags & RoleWaitSend) {
        ncclShmem->groups[group].sendConns[index] = conn; // WaitSend role saves since that's who needs it in setDataPtrs()
        connStepPtr = conn->head;
        connStepCache = LOAD(connStepPtr);
        flags |= (conn->ptrsFifo != nullptr) ? PtrsFifoEnabled : 0;
        if (flags & PtrsFifoEnabled)
          connPtrsFifoPtr = conn->ptrsFifo;
        else
          connEltsFifo = (T*)conn->buffs[NCCL_PROTO_SIMPLE];

        if (conn->sizesFifo != nullptr) {
          flags |= SizesFifoEnabled;
          connSizesFifoPtr = conn->sizesFifo;
        }
        else if (Direct && (conn->direct & NCCL_DIRECT_GPU))
          flags |= DirectEnabled;
      }
    }
  }

 public:
  __device__ Primitives(
      int tid, int nthreads, int const *recvPeers, int const *sendPeers,
      void const *inputBuf, void *outputBuf, int group=0, int connIndex=0
    ):
    tid(tid),
    stepSize(ncclShmem->comm.buffSizes[NCCL_PROTO_SIMPLE]/NCCL_STEPS/sizeof(T)),
    redOp(FuncTraits<RedOp>::make(ncclShmem->comm.nRanks)),
    connIndex((NCCL_MAX_DIRECT_ARITY==Fan::MaxSend || NCCL_MAX_DIRECT_ARITY==Fan::MaxRecv)?(group/2):connIndex),
    barriers(&ncclShmem->groups[group].barrier), barrier_next(ncclShmem->groups[group].barrier_next) {

    // For send operations, we need an extra warp to overlap the threadfence and the copy
    this->nthreads = nthreads;
    this->nworkers = nthreads;
    this->group = group;

    int nrecv=0, nsend=0;
    while (nrecv < MaxRecv && recvPeers[nrecv] != -1) nrecv++;
    while (nsend < MaxSend && sendPeers[nsend] != -1) nsend++;
    this->fan = Fan(nrecv, nsend);

    constexpr int ThreadPerSync = 8;
    static_assert(MaxSend < ThreadPerSync && MaxRecv < ThreadPerSync, "Not enough threads to cover all peers");

    int g = tid / ThreadPerSync;
    int ng = nthreads / ThreadPerSync;
    index = tid % ThreadPerSync;
    flags = 0;
    if (g == 0) {
      if (index < nrecv) flags |= RoleWaitRecv;
      if (index == nrecv) flags |= RoleInput;
    } else if (g == 1) {
      if (index < nsend) flags |= RoleWaitSend;
      if (index == nsend) flags |= RoleOutput;
    } else if (g == ng - 2) {
      if (index < nrecv) flags |= RolePostRecv;
    } else if (g == ng - 1) {
      if (index < nsend) flags |= RolePostSend;
    }

    int peer = 0;
    if (flags & (RoleWaitRecv|RolePostRecv)) peer = recvPeers[index];
    if (flags & (RoleWaitSend|RolePostSend)) peer = sendPeers[index];

    loadRecvConn(&ncclShmem->channel.devPeers[peer]);
    loadSendConn(&ncclShmem->channel.devPeers[peer]);

    setDataPtrs(inputBuf, outputBuf);
  }

  __device__ ~Primitives() {
    // Ensure ncclShmem->groups[].send/recvConns are available
    if (!(flags & ThreadsSynced))
      barrier();
    // Save steps for the next operation
    if (flags & (RolePostSend|RolePostRecv)) {
      auto *conns = (flags & RolePostSend) ? ncclShmem->groups[group].sendConns : ncclShmem->groups[group].recvConns;
      STORE(&conns[index]->step, step);
    }
    // Make sure all threads are done writing back conn->step and done using
    // ncclShmem->groups[group]
    barrier();
  }

  __device__ void setDataPtrs(void const *inputBuf, void *outputBuf) {
    if (flags & RoleInput) userBuff = (T*)inputBuf;
    if (flags & RoleOutput) userBuff = (T*)outputBuf;
    if (Direct && flags == (flags|RoleWaitRecv|DirectEnabled)) {
      int spins = 0;
      void *volatile *slot = ncclShmem->groups[group].recvConns[index]->ptrExchange;
      // Wait for consumer to consume previous value before trampling it.
      while (LOAD(slot) != nullptr && !checkAbort(spins));
      directBuff = (T*)outputBuf;
      // Encode pointer by XOR'ing against some address they definitely wouldn't send
      // since we want to allow them sending us nullptr while not colliding with
      // the empty slot value.
      *slot = reinterpret_cast<T*>(reinterpret_cast<uintptr_t>(outputBuf) ^ reinterpret_cast<uintptr_t>(slot));
    }
    if (Direct && flags == (flags|RoleWaitSend|DirectEnabled)) {
      int spins = 0;
      void *volatile *slot = ncclShmem->groups[group].sendConns[index]->ptrExchange;
      void *ptr;
      while (true) {
        ptr = LOAD(slot);
        if (ptr != nullptr || checkAbort(spins)) break;
      }
      directBuff = reinterpret_cast<T*>(reinterpret_cast<uintptr_t>(ptr) ^ reinterpret_cast<uintptr_t>(slot));
      *slot = nullptr;
    }
  }

  __device__ void moveDataPtrs(intptr_t delta) {
    if (flags & (RoleInput|RoleOutput))
      userBuff += delta;
  }

  __device__ __forceinline__ void send(intptr_t inpIx, int eltN) {
    genericOp<0, 0, 0, 1, Input, -1>(inpIx, -1, -1, eltN, false);
  }
  __device__ __forceinline__ void sendFromOutput(intptr_t outIx, int eltN) {
    genericOp<0, 0, 0, 1, Output, -1>(outIx, -1, -1, eltN, false);
  }
  __device__ __forceinline__ void directSend(intptr_t inpIx, intptr_t remoteOutIx, int eltN) {
    genericOp<0, 1, 0, 1, Input, -1>(inpIx, -1, remoteOutIx, eltN, false);
  }
  __device__ __forceinline__ void directSendFromOutput(intptr_t outIx, intptr_t remoteOutIx, int eltN) {
    genericOp<0, 1, 0, 1, Output, -1>(outIx, -1, remoteOutIx, eltN, false);
  }

  __device__ __forceinline__ void recv(intptr_t outIx, int eltN, bool postOp=false) {
    genericOp<0, 0, 1, 0, -1, Output>(-1, outIx, -1, eltN, postOp);
  }
  __device__ __forceinline__ void directRecv(intptr_t outIx, int eltN) {
    genericOp<1, 0, 1, 0, -1, Output>(-1, outIx, -1, eltN, /*postOp=*/false);
  }

  __device__ __forceinline__ void copySend(intptr_t inpIx, intptr_t outIx, int eltN, bool postOp=false) {
    genericOp<0, 0, 0, 1, Input, Output>(inpIx, outIx, -1, eltN, postOp);
  }
  __device__ __forceinline__ void directCopySend(intptr_t inpIx, intptr_t outIx, intptr_t remoteOutIx, int eltN, bool postOp=false) {
    genericOp<0, 1, 0, 1, Input, Output>(inpIx, outIx, remoteOutIx, eltN, postOp);
  }

  __device__ __forceinline__ void recvCopySend(intptr_t outIx, int eltN, bool postOp=false) {
    genericOp<0, 0, 1, 1, -1, Output>(-1, outIx, -1, eltN, postOp);
  }
  __device__ __forceinline__ void directRecvCopySend(intptr_t outIx, intptr_t remoteOutIx, int eltN) {
    genericOp<1, 1, 1, 1, -1, Output>(-1, outIx, remoteOutIx, eltN, false);
  }

  __device__ __forceinline__ void recvReduceCopy(intptr_t inpIx, intptr_t outIx, int eltN, bool postOp=false) {
    genericOp<0, 0, 1, 0, Input, Output>(inpIx, outIx, -1, eltN, postOp);
  }

  __device__ __forceinline__ void recvReduceSend(intptr_t inpIx, int eltN, bool postOp=false) {
    genericOp<0, 0, 1, 1, Input, -1>(inpIx, -1, -1, eltN, postOp);
  }

  __device__ __forceinline__ void recvReduceCopySend(intptr_t inpIx, intptr_t outIx, int eltN, bool postOp=false) {
    genericOp<0, 0, 1, 1, Input, Output>(inpIx, outIx, -1, eltN, postOp);
  }
  __device__ __forceinline__ void directRecvReduceCopySend(intptr_t inpIx, intptr_t outIx, intptr_t remoteOutIx, int eltN, bool postOp=false) {
    // Direct is only for the send part
    genericOp<0, 1, 1, 1, Input, Output>(inpIx, outIx, remoteOutIx, eltN, postOp);
  }

  __device__ __forceinline__ void
  scatter(intptr_t inpIx, int totalElem, int peerElem, int skip, int shift) {
    ScatterGatherOp<0, 1>(inpIx, -1, totalElem, peerElem, skip, shift, /*postOp=*/false);
  }

  __device__ __forceinline__ void
  gather(intptr_t outIx, int totalElem, int peerElem, int skip, int shift, bool postOp=false) {
    ScatterGatherOp<1, 0>(-1, outIx, totalElem, peerElem, skip, shift, postOp);
  }
};
