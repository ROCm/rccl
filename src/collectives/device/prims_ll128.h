/*************************************************************************
 * Copyright (c) 2016-2022, NVIDIA CORPORATION. All rights reserved.
 * Modifications Copyright (c) 2019-2022 Advanced Micro Devices, Inc. All rights reserved.
 * Modifications Copyright (c) Microsoft Corporation. Licensed under the MIT License.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

#if defined(ENABLE_NPKIT)
#include "npkit/npkit.h"
#endif

#define NCCL_LL128_FLAGTHREAD (NCCL_LL128_LINEELEMS-1)

#ifndef RCCL_USE_WBINVL1_VOL
#if defined(__GFX8__) || defined(__GFX9__)
#define RCCL_USE_WBINVL1_VOL 1
#else
#define RCCL_USE_WBINVL1_VOL 0
#endif
#endif

template<typename T, typename RedOp, typename Fan, int Direct, int P2p>
class Primitives<T, RedOp, Fan, Direct, ProtoLL128, P2p>:
  public PrimitivesWithoutDirect<Primitives<T, RedOp, Fan, Direct, ProtoLL128, P2p>> {

  static constexpr int MaxRecv = Fan::MaxRecv, MaxSend = Fan::MaxSend;
  static constexpr int Input=0, Output=1;
  RedOp redOp;
  const int tid;
  const int nthreads;
  const int wid;
  const int stepSize;
  const int warp;
  const int warpInBlock; // warp index in thread block
  const bool flagThread;
  const int group;
  Fan fan;
  T *userBufs[2];
  struct ncclConnInfo* recvConn = NULL;
  volatile uint64_t* recvConnHeadPtr = NULL;
  uint64_t recvConnHead;

  struct ncclConnInfo* sendConn = NULL;
  volatile int* sendConnFifoPtr = NULL;
  volatile uint64_t* sendConnTailPtr = NULL;
  uint64_t sendConnTail;
  volatile uint64_t* sendConnHeadPtr = NULL;
  uint64_t sendConnHead;
  uint64_t sendConnHeadCache; // Cache last seen value

  uint64_t recvStep[MaxRecv];
  uint64_t sendStep[MaxSend];
  uint64_t* recvBuff[MaxRecv];
  uint64_t* sendBuff[MaxSend];

  inline __device__ int recvOffset(int i) { return (recvStep[i]%NCCL_STEPS)*stepSize; }
  inline __device__ int sendOffset(int i) { return (sendStep[i]%NCCL_STEPS)*stepSize; }
  inline __device__ uint64_t* recvPtr(int i) { return recvBuff[i]+recvOffset(i); }
  inline __device__ uint64_t* sendPtr(int i) { return sendBuff[i]+sendOffset(i); }
  inline __device__ uint64_t recvFlag(int i) { return recvStep[i]+1; }
  inline __device__ uint64_t sendFlag(int i) { return sendStep[i]+1; }

  uint64_t* barriers;
  uint64_t* barrier_next;

#if defined(ENABLE_NPKIT)
public:
  int npKitCtxIdx = 0;
  uint64_t npKitDataProcessEntryTime = 0;
  uint64_t npKitDataProcessExitTime = 0;
  uint64_t npKitDataProcessTotalTime = 0;
private:
#endif

  inline __device__ void barrier() {
#if defined(__HIP_PLATFORM_HCC__) || defined(__HCC__) || defined(__HIPCC__)
  if (nthreads != WARP_SIZE)
    barrier_by_group();
#else
   asm volatile ("bar.sync %1, %0;" :: "r"(nthreads), "r"(15-group));
#endif
  }

  uint32_t abort = 0;
  uint32_t* sync;

  inline __device__ int checkAbort(int &spins, int i, int send) {
    spins++;
    if (abort == 0 && spins == NCCL_SPINS_BEFORE_CHECK_ABORT) {
      abort = __atomic_load_n(ncclShmem.comm.abortFlag, __ATOMIC_SEQ_CST);
      spins = 0;
    }
    return abort;
  }

  inline __device__ void waitSend(int nbytes) {
    if (sendConnHeadPtr) {
      int spins = 0;
      while (sendConnHeadCache + NCCL_STEPS < sendConnHead + 1) {
        __builtin_amdgcn_s_sleep(1);
        sendConnHeadCache = atomicAdd_system((unsigned long long *)sendConnHeadPtr, 0);
        if (checkAbort(spins, wid, 1)) break;
      }
      __asm__ __volatile__("s_wakeup");
      if (sendConnFifoPtr) {
        __atomic_store_n(sendConnFifoPtr+sendStep[wid]%NCCL_STEPS, nbytes, __ATOMIC_SEQ_CST);
      }
      sendConnHead += 1;
    }
  }

  inline __device__ void postRecv() {
    if (recvConnHeadPtr) STORE(recvConnHeadPtr, recvConnHead += 1);
  }
  inline __device__ void postSend() {
    if (sendConnTailPtr) { __threadfence(); STORE((unsigned long long *)sendConnTailPtr, sendConnTail += 1); }
  }

  template<int WordPerThread>
  __device__ __forceinline__ void loadRegsBegin(uint64_t(&regs)[WordPerThread], T const *src, int eltN) {
    constexpr int EltPer16B = 16/sizeof(T);
    /* We are aligned to 16 bytes, so load directly to registers no shmem.
     * Flag threads load half as much data which gets shuffled to the even
     * registers during Finish. The point of splitting into two phases is to
     * defer that shuffle, which incurs a dependency stall, until after other
     * memops are launched by the caller.
     */
    #pragma unroll
    for(int g=0; g < WordPerThread/2; g++) {
      int ix = g*WARP_SIZE - 16*(g/2) + wid - (g%2)*(wid/4);
      if(!flagThread || g%2==0) {
        if(ix*EltPer16B < eltN) {
          if(reinterpret_cast<uintptr_t>(src)%4 == 0) {
            regs[2*g+0] = __builtin_nontemporal_load((uint64_t*)(src + ix*EltPer16B));
            regs[2*g+1] = __builtin_nontemporal_load((uint64_t*)(src + ix*EltPer16B)+1);
          } else {
            union {
              uint64_t regs64[WordPerThread];
              uint32_t regs32[WordPerThread*2];
              uint16_t regs16[WordPerThread*4];
              uint8_t regs8[WordPerThread*8];
            };
            if (sizeof(T) == 8) {
              uint64_t *src64 = (uint64_t*)(src+ix*EltPer16B);
              for (int i=0; i < 2; i++)
                regs64[2*g+i] = __builtin_nontemporal_load(src64+i);
            } else if (sizeof(T) == 4) {
              uint32_t *src32 = (uint32_t*)(src+ix*EltPer16B);
              for (int i=0; i < 2*sizeof(uint64_t)/sizeof(T); i++)
                regs32[2*g+i] = __builtin_nontemporal_load(src32+i);
            } else if (sizeof(T) == 2) {
              uint16_t *src16 = (uint16_t*)(src+ix*EltPer16B);
              for (int i=0; i < 2*sizeof(uint64_t)/sizeof(T); i++)
                regs16[2*g+i] = __builtin_nontemporal_load(src16+i);
            } else if (sizeof(T) == 1) {
              uint8_t *src8 = (uint8_t*)(src+ix*EltPer16B);
              for (int i=0; i < 2*sizeof(uint64_t)/sizeof(T); i++)
                regs8[2*g+i] = __builtin_nontemporal_load(src8+i);
            }
            regs[2*g+0] = regs64[2*g+0];
            regs[2*g+1] = regs64[2*g+1];
          }
        }
      }
    }
  }

  template<int WordPerThread>
  __device__ __forceinline__ void loadRegsFinish(uint64_t(&regs)[WordPerThread]) {
    // Move data out of flag registers into the vacant registers.
    #pragma unroll
    for (int g=1; g < WordPerThread/2; g+=2) {
      if (flagThread) regs[2*g] = regs[2*g-1];
    }
  }

  template<int WordPerThread>
  __device__ __forceinline__ void storeRegs(T *dst, uint64_t(&regs)[WordPerThread], int eltN) {
    constexpr int EltPer16B = 16/sizeof(T);
    // Reverse Finish() register permuatation.
    #pragma unroll
    for (int g=1; g < WordPerThread/2; g+=2) {
      if (flagThread) regs[2*g-1] = regs[2*g];
    }
    // Write to dst if 4-byte aligned, shmem otherwise.
    int misalignment = reinterpret_cast<uintptr_t>(dst)%4;
    #pragma unroll
    for(int g=0; g < WordPerThread/2; g++) {
      int ix = g*WARP_SIZE - 16*(g/2) + wid - (g%2)*(wid/4);
      if (!flagThread || g%2==0) {
        if(misalignment == 0 && (ix+1)*EltPer16B <= eltN) {
          __builtin_nontemporal_store(regs[2*g+0], (uint64_t*)(dst + ix*EltPer16B));
          __builtin_nontemporal_store(regs[2*g+1], (uint64_t*)(dst + ix*EltPer16B)+1);
        } else {
          union {
            uint64_t regs64[WordPerThread];
            uint32_t regs32[WordPerThread*2];
            uint16_t regs16[WordPerThread*4];
            uint8_t regs8[WordPerThread*8];
          };
          regs64[2*g+0] = regs[2*g+0];
          regs64[2*g+1] = regs[2*g+1];
          int remaining = eltN - ix*EltPer16B;
          if (sizeof(T) == 8) {
            uint64_t *dst64 = (uint64_t*)(dst+ix*EltPer16B);
            for (int i=0; i < 2 && i < remaining; i++)
              __builtin_nontemporal_store(regs64[2*g+i], dst64+i);
          } else if (sizeof(T) == 4) {
            uint32_t *dst32 = (uint32_t*)(dst+ix*EltPer16B);
            for (int i=0; i < 2*sizeof(uint64_t)/sizeof(T) && i < remaining; i++)
              __builtin_nontemporal_store(regs32[2*g+i], dst32+i);
          } else if (sizeof(T) == 2) {
            uint16_t *dst16 = (uint16_t*)(dst+ix*EltPer16B);
            for (int i=0; i < 2*sizeof(uint64_t)/sizeof(T) && i < remaining; i++)
              __builtin_nontemporal_store(regs16[2*g+i], dst16+i);
          } else if (sizeof(T) == 1) {
            uint8_t *dst8 = (uint8_t*)(dst+ix*EltPer16B);
            for (int i=0; i < 2*sizeof(uint64_t)/sizeof(T) && i < remaining; i++)
              __builtin_nontemporal_store(regs8[2*g+i], dst8+i);
          }
        }
      }
    }
  }

  #define WARP_MASK 0xffffffff

  template <int ELEMS_PER_THREAD, int RECV, int SEND, int SrcBuf, int DstBuf>
  __device__ __forceinline__ void recvReduceSendCopy(uint64_t(&v)[ELEMS_PER_THREAD], int ll128Offset, bool postOp) {
    constexpr int SRC = SrcBuf != -1 ? 1 : 0;
    uint64_t vr[ELEMS_PER_THREAD];

    __syncwarp();
    /************************ Wait first recv ********************/
    if (RECV) {
      uint64_t* ptr = recvPtr(0)+ll128Offset;
      uint64_t flag = recvFlag(0);
      bool needReload;
      int spins = 0;
      do {
        needReload = false;
        #pragma unroll
        for (int u=0; u<ELEMS_PER_THREAD; u+=2) {
          vr[u] = __builtin_nontemporal_load(ptr+u*WARP_SIZE);
          vr[u+1] = __builtin_nontemporal_load(ptr+u*WARP_SIZE+1);
          needReload |= flagThread && (vr[u+1] != flag);
        }
        needReload &= (0 == checkAbort(spins, 0, 0));
      } while (__any(needReload));
    }

    /************* Finish register load **************/
    if (SRC) {
      // By deferring register shuffle here we've overlapped spinning on first
      // peer's data with memory loads of src data.
      loadRegsFinish(v);
      if (SrcBuf == Input) {
        #pragma unroll
        for (int u=0; u<ELEMS_PER_THREAD; u+=2) {
          v[u] = applyPreOp(redOp, v[u]);
          if (!flagThread)
            v[u+1] = applyPreOp(redOp, v[u+1]);
        }
      }
    }

    /************************ Recv rest *********************/
    if (RECV) {
      { // Consume data from first recv
        uint64_t* ptr = recvPtr(0)+ll128Offset;
        #pragma unroll
        for (int u=0; u<ELEMS_PER_THREAD; u+=2) {
          v[u]   = SRC ? applyReduce(redOp, vr[u], v[u]) : vr[u];
          v[u+1] = SRC ? applyReduce(redOp, vr[u+1], v[u+1]) : vr[u+1];
        }
      }

      for (int i=1; i<MaxRecv && i<fan.nrecv(); i++) {
        uint64_t flag = recvFlag(i);
        uint64_t* ptr = recvPtr(i)+ll128Offset;
        bool needReload;
        int spins = 0;
        do {
          needReload = false;
          #pragma unroll
          for (int u=0; u<ELEMS_PER_THREAD; u+=2) {
            vr[u] = __builtin_nontemporal_load(ptr+u*WARP_SIZE);
            vr[u+1] = __builtin_nontemporal_load(ptr+u*WARP_SIZE+1);
            needReload |= flagThread && (vr[u+1] != flag);
          }
          needReload &= (0 == checkAbort(spins, i, 0));
        } while (__any(needReload));

        #pragma unroll
        for (int u=0; u<ELEMS_PER_THREAD; u+=2) {
          v[u]   = applyReduce(redOp, vr[u], v[u]);
          v[u+1] = applyReduce(redOp, vr[u+1], v[u+1]);
        }
      }
    }
    /********************** End Recv ************************/

    if (postOp) {
      #pragma unroll
      for (int u=0; u<ELEMS_PER_THREAD; u+=2) {
        v[u]   = applyPostOp(redOp, v[u]);
        v[u+1] = applyPostOp(redOp, v[u+1]);
      }
    }

#if RCCL_USE_WBINVL1_VOL
    if (tid == 0) __builtin_amdgcn_buffer_wbinvl1();
#endif
    /************************ Send **************************/
    if (SEND) {
      for (int i=1; i<MaxSend && i<fan.nsend(); i++) {
        uint64_t flag = sendFlag(i);
        uint64_t* ptr = sendPtr(i)+ll128Offset;
        #pragma unroll
        for (int u=0; u<ELEMS_PER_THREAD; u+=2) {
          __builtin_nontemporal_store(v[u], ptr+u*WARP_SIZE);
          __builtin_nontemporal_store(flagThread ? flag : v[u+1], ptr+u*WARP_SIZE+1);
        }
      }
      uint64_t flag = sendFlag(0);
      uint64_t* ptr = sendPtr(0)+ll128Offset;
      #pragma unroll
      for (int u=0; u<ELEMS_PER_THREAD; u+=2) {
        __builtin_nontemporal_store(v[u], ptr+u*WARP_SIZE);
        __builtin_nontemporal_store(flagThread ? flag : v[u+1], ptr+u*WARP_SIZE+1);
      }
    }
    /********************** End Send ************************/
  }

  static constexpr int WireWordPerSlice = WARP_SIZE*NCCL_LL128_SHMEM_ELEMS_PER_THREAD;
  static constexpr int DataEltPerSlice = (WireWordPerSlice - WireWordPerSlice/NCCL_LL128_LINEELEMS)*(sizeof(uint64_t)/sizeof(T));

  template <int RECV, int SEND, int SrcBuf, int DstBuf>
  __device__ __forceinline__ void GenericOp(intptr_t srcIx, intptr_t dstIx, int nelem, bool postOp) {
    constexpr int SRC = SrcBuf != -1 ? 1 : 0;
    constexpr int DST = DstBuf != -1 ? 1 : 0;
    T const *srcPtr = SrcBuf == -1 ? nullptr : userBufs[SrcBuf] + srcIx;
    T       *dstPtr = DstBuf == -1 ? nullptr : userBufs[DstBuf] + dstIx;
    int wireOffset = WireWordPerSlice*warp + 2*wid;
    const int nwarps = nthreads/WARP_SIZE;
    nelem = nelem < 0 ? 0 : nelem;

    if (SEND) waitSend(divUp(nelem, DataEltPerSlice)*WireWordPerSlice*sizeof(uint64_t));
    barrier();
    nelem -= DataEltPerSlice*warp;
    srcPtr += DataEltPerSlice*warp;
    dstPtr += DataEltPerSlice*warp;
    while (nelem > 0) {
      const int eltInSlice = min(nelem, DataEltPerSlice);
      uint64_t regs[NCCL_LL128_SHMEM_ELEMS_PER_THREAD];
      if (SRC) loadRegsBegin(regs, srcPtr, eltInSlice);
      recvReduceSendCopy<NCCL_LL128_SHMEM_ELEMS_PER_THREAD, RECV, SEND, SrcBuf, DstBuf>(regs, wireOffset, postOp);
      if (DST) storeRegs(dstPtr, regs, eltInSlice);

      wireOffset += WireWordPerSlice*nwarps;
      srcPtr += DataEltPerSlice*nwarps;
      dstPtr += DataEltPerSlice*nwarps;
      nelem -= DataEltPerSlice*nwarps;
    }

    barrier();
    if (SEND) for (int i=0; i < MaxSend; i++) sendStep[i] += 1;
    if (SEND) postSend();
    if (RECV) for (int i=0; i < MaxRecv; i++) recvStep[i] += 1;
    if (RECV) postRecv();
  }

  template <int REDUCE, int COPY, int MULTISRCS, int MULTIDSTS>
  __device__ __forceinline__ void mscclGenericOp(T** srcs, int nsrcs, T** dsts, int ndsts, int nelem) {
#if defined(ENABLE_NPKIT) && defined(ENABLE_NPKIT_EVENT_MSCCL_GENERIC_OP_ENTRY)
    if (tid == 0) {
      NpKit::CollectGpuEvent(NPKIT_EVENT_MSCCL_GENERIC_OP_ENTRY, nelem*sizeof(T), 0, NPKIT_GET_GPU_TIMESTAMP(),
          ncclShmem.comm.npKitEventCollectContexts + npKitCtxIdx);
    }
#endif

    T const *srcPtr = srcs[0];
    T       *dstPtr = dsts[0];
    int wireOffset = WireWordPerSlice*warp + 2*wid;
    const int nwarps = nthreads/WARP_SIZE;
    nelem = nelem < 0 ? 0 : nelem;

    nelem -= DataEltPerSlice*warp;
    srcPtr += DataEltPerSlice*warp;
    dstPtr += DataEltPerSlice*warp;
    if (MULTISRCS){
      for (int i = 1; i < nsrcs; i++){
        srcs[i] += DataEltPerSlice*warp;
      }
    }
    if (MULTIDSTS){
      for (int i = 1; i < ndsts; i++){
        dsts[i] += DataEltPerSlice*warp;
      }
    }
    while (nelem > 0) {
      const int eltInSlice = min(nelem, DataEltPerSlice);
      uint64_t regs[NCCL_LL128_SHMEM_ELEMS_PER_THREAD];
      loadRegsBegin(regs, srcPtr, eltInSlice);
      loadRegsFinish(regs);
      if (REDUCE){
        uint64_t regsD[NCCL_LL128_SHMEM_ELEMS_PER_THREAD];
        loadRegsBegin(regsD, dstPtr, eltInSlice);
        loadRegsFinish(regsD);
        #pragma unroll
        for (int u=0; u<NCCL_LL128_SHMEM_ELEMS_PER_THREAD; u+=2) {
          regsD[u] = applyReduce(redOp, regs[u], regsD[u]);
          if (!flagThread)
            regsD[u+1] = applyReduce(redOp, regs[u+1], regsD[u+1]);
        }
        if (MULTISRCS){
          for (int i = 1; i < nsrcs; i++){
            loadRegsBegin(regs, srcs[i], eltInSlice);
            loadRegsFinish(regs);
            for (int u=0; u<NCCL_LL128_SHMEM_ELEMS_PER_THREAD; u+=2) {
              regsD[u] = applyReduce(redOp, regs[u], regsD[u]);
              if (!flagThread)
                regsD[u+1] = applyReduce(redOp, regs[u+1], regsD[u+1]);
            }
          }
        }
        storeRegs(dstPtr, regsD, eltInSlice);
      }
      if (COPY){
        storeRegs(dstPtr, regs, eltInSlice);
        if (MULTIDSTS){
          for (int i = 1; i < nsrcs; i++){
            loadRegsBegin(regs, srcs[i], eltInSlice);
            loadRegsFinish(regs);
            storeRegs(dsts[i], regs, eltInSlice);
          }
        }
      }

      wireOffset += WireWordPerSlice*nwarps;
      srcPtr += DataEltPerSlice*nwarps;
      dstPtr += DataEltPerSlice*nwarps;
      if (MULTISRCS){
        for (int i = 1; i < nsrcs; i++){
          srcs[i] += DataEltPerSlice*nwarps;
        }
      }
      if (MULTIDSTS){
        for (int i = 1; i < ndsts; i++){
          dsts[i] += DataEltPerSlice*nwarps;
        }
      }
      nelem -= DataEltPerSlice*nwarps;
    }

#if defined(ENABLE_NPKIT) && defined(ENABLE_NPKIT_EVENT_MSCCL_GENERIC_OP_EXIT)
    if (tid == 0) {
      NpKit::CollectGpuEvent(NPKIT_EVENT_MSCCL_GENERIC_OP_EXIT, nelem*sizeof(T), 0, NPKIT_GET_GPU_TIMESTAMP(),
          ncclShmem.comm.npKitEventCollectContexts + npKitCtxIdx);
    }
#endif

    barrier();
  }

  __device__ __forceinline__ void loadRecvConn(struct ncclConnInfo* conn, int i) {
    recvBuff[i] = (uint64_t*)conn->buffs[NCCL_PROTO_LL128];
    recvStep[i] = conn->step;
    if (wid == i) recvConn = conn;
  }
  __device__ __forceinline__ void loadRecvSync() {
    if (tid >= nthreads-WARP_SIZE && wid < fan.nrecv()) {
      recvConnHeadPtr = recvConn->head;
      recvConnHead = recvConn->step;
    }
  }

  __device__ __forceinline__ void loadSendConn(struct ncclConnInfo* conn, int i) {
    sendBuff[i] = (uint64_t*)conn->buffs[NCCL_PROTO_LL128];
    sendStep[i] = conn->step;
    if (wid == i) sendConn = conn;
  }
  __device__ __forceinline__ void loadSendSync() {
    if (tid < fan.nsend()) {
      sendConnHeadPtr = sendConn->head;
      sendConnHeadCache = *sendConnHeadPtr;
      sendConnHead = sendConn->step;
      sendConnFifoPtr = sendConn->sizesFifo;
    }
    if (tid >= nthreads-WARP_SIZE && wid<fan.nsend()) {
      if (sendConn->sizesFifo) {
        sendConnTailPtr = sendConn->tail;
        sendConnTail = sendConn->step;
      }
    }
  }

public:
  __device__ Primitives(
      const int tid, const int nthreads, int const *recvPeers, int const *sendPeers,
      void const *inputBuf, void *outputBuf, uint64_t redOpArg, uint8_t group=0,
      uint8_t connIndexRecv=0, uint8_t connIndexSend=0
    ):
    redOp(redOpArg),
    tid(tid), nthreads(nthreads), wid(tid%WARP_SIZE), warp(tid/WARP_SIZE),
    warpInBlock(threadIdx.x/WARP_SIZE),
    flagThread((tid%4)==3), group(group),
    stepSize(ncclShmem.comm.buffSizes[NCCL_PROTO_LL128]/NCCL_STEPS/sizeof(uint64_t)) {
    auto *channel = &ncclShmem.channel;
    barriers = &ncclShmem.groups[group].barrier;
    barrier_next = ncclShmem.groups[group].barrier_next;
    int nrecv=0, nsend=0;
    while (nrecv < MaxRecv && recvPeers[nrecv] >= 0) {
      loadRecvConn(&channel->peers[recvPeers[nrecv]]->recv[connIndexRecv], nrecv);
      nrecv++;
    }
    while (nsend < MaxSend && sendPeers[nsend] >= 0) {
      loadSendConn(&channel->peers[sendPeers[nsend]]->send[connIndexSend], nsend);
      nsend++;
    }
    this->fan = Fan(nrecv, nsend);
    loadRecvSync();
    loadSendSync();
    setDataPtrs(inputBuf, outputBuf);
  }

  __device__ ~Primitives() {
    // Save steps for the next operation
    if (tid >= nthreads-WARP_SIZE && wid < fan.nrecv())
      recvConn->step = recvConnHead;
    if (tid < fan.nsend())
      sendConn->step = sendConnHead;
    // Ensure all steps written back
    barrier();
  }

  __device__ void setDataPtrs(void const *inputBuf, void *outputBuf) {
    userBufs[Input] = (T*)inputBuf;
    userBufs[Output] = (T*)outputBuf;
  }

  __device__ void moveDataPtrs(intptr_t delta) {
    userBufs[Input] += delta;
    userBufs[Output] += delta;
  }

  __device__ void send(intptr_t inpIx, int eltN) {
    return GenericOp<0, 1, Input, -1>(inpIx, -1, eltN, false);
  }
  __device__ void sendFromOutput(intptr_t outIx, int eltN) {
    return GenericOp<0, 1, Output, -1>(outIx, -1, eltN, false);
  }
  __device__ void recv(intptr_t outIx, int eltN, bool postOp=false) {
    return GenericOp<1, 0, -1, Output>(-1, outIx, eltN, postOp);
  }
  __device__ void recvReduceSend(intptr_t inpIx, int eltN) {
    return GenericOp<1, 1, Input, -1>(inpIx, -1, eltN, false);
  }
  __device__ void recvReduceCopy(intptr_t inpIx, intptr_t outIx, int eltN, bool postOp=false) {
    return GenericOp<1, 0, Input, Output>(inpIx, outIx, eltN, postOp);
  }
  __device__ void copySend(intptr_t inpIx, intptr_t outIx, int eltN, bool postOp=false) {
    return GenericOp<0, 1, Input, Output>(inpIx, outIx, eltN, postOp);
  }
  __device__ void recvCopySend(intptr_t outIx, int eltN, bool postOp=false) {
    return GenericOp<1, 1, -1, Output>(-1, outIx, eltN, postOp);
  }
  __device__ void recvReduceCopySend(intptr_t inpIx, intptr_t outIx, int eltN, bool postOp=false) {
    return GenericOp<1, 1, Input, Output>(inpIx, outIx, eltN, postOp);
  }
  __device__ void recvSend(int eltN) {
    return GenericOp<1, 1, -1, -1>(-1, -1, eltN, false);
  }

  // MSCCL primitives
  __device__ void sendWithBarrier(intptr_t inpIx, int eltN) {
    send(inpIx, eltN);
  }
  __device__ void localCopy(T* srcs, T* dsts, int eltN) {
    return mscclGenericOp<0,1,0,0>(&srcs, 1, &dsts, 1, eltN);
  }
  __device__ void reduce(T** srcs, int nsrcs, T** dsts, int ndsts, int eltN) {
    if (nsrcs == 1) {
      return mscclGenericOp<1,0,0,0>(srcs, 1, dsts, 1, eltN);
    } else {
      return mscclGenericOp<1,0,1,0>(srcs, nsrcs, dsts, 1, eltN);
    }
  }
};
