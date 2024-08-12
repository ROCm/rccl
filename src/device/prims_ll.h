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

template<typename T, typename RedOp, typename Fan, int Direct, int P2p>
class Primitives<T, RedOp, Fan, Direct, ProtoLL, P2p>:
  public PrimitivesWithoutDirect<Primitives<T, RedOp, Fan, Direct, ProtoLL, P2p>> {

  // In the case of Fan::MaxRecv == 0, we need to force MaxRecv to 1 for this to compile
  // This is because of a recv buffer which is allocated to MaxRecv length in send-only cases
  static constexpr int MaxRecv = Fan::MaxRecv > 1 ? Fan::MaxRecv : 1;
  static constexpr int MaxSend = Fan::MaxSend;
  static constexpr int Input=0, Output=1;
  RedOp redOp;
  const int tid;
  const int nthreads;
  const int wid;
  const int group;
  const int stepLines;
  Fan fan;
  T *userBufs[2];
  struct ncclConnInfo* recvConn = NULL;
  volatile uint64_t* recvConnHeadPtr = NULL;
  uint64_t recvConnHead;

  struct ncclConnInfo* sendConn = NULL;
  volatile struct ncclConnFifo* sendConnFifo = NULL;
  volatile uint64_t* sendConnHeadPtr = NULL;
  uint64_t sendConnHead;
  uint64_t sendConnHeadCache; // Cache last seen value

  uint64_t recvStep[MaxRecv];
  uint64_t sendStep[MaxSend];
  union ncclLLFifoLine* recvBuff[MaxRecv];
  union ncclLLFifoLine* sendBuff[MaxSend];

#if defined(ENABLE_NPKIT)
public:
  int npKitCtxIdx = 0;
  uint64_t npKitDataProcessEntryTime = 0;
  uint64_t npKitDataProcessExitTime = 0;
  uint64_t npKitDataProcessTotalTime = 0;
private:
#endif

#if defined(ENABLE_NPKIT) && (defined(ENABLE_NPKIT_EVENT_PRIM_LL_DATA_PROCESS_ENTRY) && defined(ENABLE_NPKIT_EVENT_PRIM_LL_DATA_PROCESS_EXIT) || defined(ENABLE_NPKIT_PRIM_COLLECT_DATA_PROCESS_TIME))
  uint64_t npKitWaitRecvDataProcessSize = 0;
  uint64_t npKitWaitRecvEntryTime = 0;
  uint64_t npKitWaitRecvExitTime = 0;
  uint64_t npKitWaitRecvTotalTime = 0;
#endif

  inline __device__ int recvOffset(int i) { return (recvStep[i]%NCCL_STEPS)*stepLines; }
  inline __device__ int sendOffset(int i) { return (sendStep[i]%NCCL_STEPS)*stepLines; }
  inline __device__ union ncclLLFifoLine* recvPtr(int i) { return recvBuff[i]+recvOffset(i); }
  inline __device__ union ncclLLFifoLine* sendPtr(int i) { return sendBuff[i]+sendOffset(i); }
  inline __device__ uint32_t recvFlag(int i) { return NCCL_LL_FLAG(recvStep[i]+1); }
  inline __device__ uint32_t sendFlag(int i) { return NCCL_LL_FLAG(sendStep[i]+1); }

  uint64_t* barriers;
  uint64_t* barrier_next;

  inline __device__ void barrier() {
#if defined(__HIP_PLATFORM_AMD__) || defined(__HIPCC__)
    if (nthreads != WARP_SIZE)
      barrier_by_group();
#else
    asm volatile ("bar.sync %1, %0;" :: "r"(nthreads), "r"(15-group));
#endif
  }

  uint32_t abort = 0;

  inline __device__ int checkAbort(int &spins, int send) {
    spins++;
    if (abort == 0 && spins == NCCL_SPINS_BEFORE_CHECK_ABORT) {
      abort = __atomic_load_n((ncclShmem.comm.abortFlag), __ATOMIC_SEQ_CST);
      spins = 0;
    }
    return abort;
  }

  inline __device__ void waitSend(int nbytes) {
#if defined(ENABLE_NPKIT) && defined(ENABLE_NPKIT_EVENT_PRIM_LL_WAIT_SEND_ENTRY)
    if (tid == 0) {
      NpKit::CollectGpuEvent(NPKIT_EVENT_PRIM_LL_WAIT_SEND_ENTRY, nbytes, 0, NPKIT_GET_GPU_TIMESTAMP(),
          ncclShmem.comm.npKitEventCollectContexts + npKitCtxIdx);
    }
#endif
    if (sendConnHeadPtr) {
      int spins = 0;
      while (sendConnHeadCache + NCCL_STEPS < sendConnHead + 1) {
        __builtin_amdgcn_s_sleep(1);
        sendConnHeadCache = atomicAdd((unsigned long long *)sendConnHeadPtr, 0);
        if (checkAbort(spins, 1)) break;
      }
      if (sendConnFifo) {
        int size = ((sendConnHead & NCCL_LL_CLEAN_MASK) == NCCL_LL_CLEAN_MASK) ? stepLines*sizeof(union ncclLLFifoLine) : nbytes;
        sendConnFifo[sendConnHead%NCCL_STEPS].size = size;
      }
      sendConnHead += 1;
    }
    barrier();
#if defined(ENABLE_NPKIT) && defined(ENABLE_NPKIT_EVENT_PRIM_LL_WAIT_SEND_EXIT)
    if (tid == 0) {
      NpKit::CollectGpuEvent(NPKIT_EVENT_PRIM_LL_WAIT_SEND_EXIT, nbytes, 0, NPKIT_GET_GPU_TIMESTAMP(),
          ncclShmem.comm.npKitEventCollectContexts + npKitCtxIdx);
    }
#endif
  }

  inline __device__ void incRecv(int i) {
    recvStep[i] += 1;
  }
  inline __device__ void postRecv() {
    barrier();
    if (recvConnHeadPtr) STORE(recvConnHeadPtr, recvConnHead += 1);
  }

  inline __device__ void incSend(int i, int offset) {
    // LL Cleanup : write all flags in the slice to make sure we don't have
    // data corruption when flag loops over.
    if ((sendStep[i] & NCCL_LL_CLEAN_MASK) == NCCL_LL_CLEAN_MASK) {
      for (int o = offset; o<stepLines; o+=nthreads) storeLL(sendPtr(i)+o, 0, sendFlag(i));
    }
    sendStep[i]++;
  }

  __device__ uint64_t readLL(int offset, int i) {
    union ncclLLFifoLine* src = recvPtr(i) + offset;
    uint32_t flag = recvFlag(i);
    uint32_t data1, flag1, data2, flag2;
    int spins = 0;

#if defined(ENABLE_NPKIT) && (defined(ENABLE_NPKIT_EVENT_PRIM_LL_DATA_PROCESS_ENTRY) && defined(ENABLE_NPKIT_EVENT_PRIM_LL_DATA_PROCESS_EXIT) || defined(ENABLE_NPKIT_PRIM_COLLECT_DATA_PROCESS_TIME))
    int npkitWaitRecvSpins = 0;
    if (tid == 0) {
      npKitWaitRecvEntryTime = NPKIT_GET_GPU_TIMESTAMP();
    }
#endif

#if defined(__HIP_PLATFORM_AMD__) || defined(__HIPCC__)
    union ncclLLFifoLine i4;
    do {
#ifdef __GFX11__
      asm volatile ("global_load_b128 %0, %1, off glc slc dlc\n"
        "s_waitcnt vmcnt(0)\n" : "=v"(i4.i4) : "v"(&src->i4));
#else
      i4.v[0] = __builtin_nontemporal_load(src->v);
      i4.v[1] = __builtin_nontemporal_load(src->v+1);
#endif
#if defined(ENABLE_NPKIT) && (defined(ENABLE_NPKIT_EVENT_PRIM_LL_DATA_PROCESS_ENTRY) && defined(ENABLE_NPKIT_EVENT_PRIM_LL_DATA_PROCESS_EXIT) || defined(ENABLE_NPKIT_PRIM_COLLECT_DATA_PROCESS_TIME))
      npkitWaitRecvSpins++;
#endif
      if (checkAbort(spins, 0)) break;
    } while ((i4.flag1 != flag) || (i4.flag2 != flag));
    uint64_t val64 = (uint64_t)(i4.data1) + (((uint64_t)i4.data2) << 32);
#else
    do {
      asm("ld.volatile.global.v4.u32 {%0,%1,%2,%3}, [%4];" : "=r"(data1), "=r"(flag1), "=r"(data2), "=r"(flag2) : "l"(&src->i4));
#if defined(ENABLE_NPKIT) && (defined(ENABLE_NPKIT_EVENT_PRIM_LL_DATA_PROCESS_ENTRY) && defined(ENABLE_NPKIT_EVENT_PRIM_LL_DATA_PROCESS_EXIT) || defined(ENABLE_NPKIT_PRIM_COLLECT_DATA_PROCESS_TIME))
      npkitWaitRecvSpins++;
#endif
      if (checkAbort(spins, 0)) break;
    } while ((flag1 != flag) || (flag2 != flag));
    uint64_t val64 = data1 + (((uint64_t)data2) << 32);
#endif

#if defined(ENABLE_NPKIT) && (defined(ENABLE_NPKIT_EVENT_PRIM_LL_DATA_PROCESS_ENTRY) && defined(ENABLE_NPKIT_EVENT_PRIM_LL_DATA_PROCESS_EXIT) || defined(ENABLE_NPKIT_PRIM_COLLECT_DATA_PROCESS_TIME))
    if (tid == 0) {
      npKitWaitRecvExitTime = NPKIT_GET_GPU_TIMESTAMP();
      npKitWaitRecvTotalTime += (npKitWaitRecvExitTime - npKitWaitRecvEntryTime) * (npkitWaitRecvSpins - 1) / npkitWaitRecvSpins;
    }
#endif

    return val64;
  }

  template<int BeginIx>
  __device__ void readLLBeginAll(int offset, ncclLLFifoLine(&line)[MaxRecv]) {
    #pragma unroll
    for (int i=BeginIx; i < MaxRecv; i++) {
      if (i < fan.nrecv()) {
        union ncclLLFifoLine* src = recvPtr(i) + offset;
#if defined(__HIP_PLATFORM_AMD__) || defined(__HIPCC__)
#ifdef __GFX11__
        asm volatile ("global_load_b128 %0, %1, off glc slc dlc\n"
          "s_waitcnt vmcnt(0)\n" : "=v"(line[i].i4) : "v"(&src->i4));
#else
        line[i].v[0] = __builtin_nontemporal_load(src->v);
        line[i].v[1] = __builtin_nontemporal_load(src->v+1);
#endif
#else
        asm("ld.volatile.global.v4.u32 {%0,%1,%2,%3}, [%4];" : "=r"(line[i].data1), "=r"(line[i].flag1), "=r"(line[i].data2), "=r"(line[i].flag2) : "l"(&src->i4));
#endif
      }
    }
  }
  __device__ uint64_t readLLFinish(int offset, ncclLLFifoLine(&line)[MaxRecv], int i) {
    union ncclLLFifoLine* src = recvPtr(i) + offset;
    uint32_t flag = recvFlag(i);
    int spins = 0;

#if defined(ENABLE_NPKIT) && (defined(ENABLE_NPKIT_EVENT_PRIM_LL_DATA_PROCESS_ENTRY) && defined(ENABLE_NPKIT_EVENT_PRIM_LL_DATA_PROCESS_EXIT) || defined(ENABLE_NPKIT_PRIM_COLLECT_DATA_PROCESS_TIME))
    int npkitWaitRecvSpins = 0;
    if (tid == 0) {
      npKitWaitRecvEntryTime = NPKIT_GET_GPU_TIMESTAMP();
    }
#endif

    do {
#if defined(__HIP_PLATFORM_AMD__) || defined(__HIPCC__)
#ifdef __GFX11__
      asm volatile ("global_load_b128 %0, %1, off glc slc dlc\n"
        "s_waitcnt vmcnt(0)\n" : "=v"(line[i].i4) : "v"(&src->i4));
#else
      line[i].v[0] = __builtin_nontemporal_load(src->v);
      line[i].v[1] = __builtin_nontemporal_load(src->v+1);
#endif
#else
      asm("ld.volatile.global.v4.u32 {%0,%1,%2,%3}, [%4];" : "=r"(line[i].data1), "=r"(line[i].flag1), "=r"(line[i].data2), "=r"(line[i].flag2) : "l"(&src->i4));
#endif
#if defined(ENABLE_NPKIT) && (defined(ENABLE_NPKIT_EVENT_PRIM_LL_DATA_PROCESS_ENTRY) && defined(ENABLE_NPKIT_EVENT_PRIM_LL_DATA_PROCESS_EXIT) || defined(ENABLE_NPKIT_PRIM_COLLECT_DATA_PROCESS_TIME))
      npkitWaitRecvSpins++;
#endif
      if (checkAbort(spins, 0)) break;
    } while(line[i].flag1 != flag || line[i].flag2 != flag);
    uint64_t val64 = line[i].data1 + (((uint64_t)line[i].data2) << 32);

#if defined(ENABLE_NPKIT) && (defined(ENABLE_NPKIT_EVENT_PRIM_LL_DATA_PROCESS_ENTRY) && defined(ENABLE_NPKIT_EVENT_PRIM_LL_DATA_PROCESS_EXIT) || defined(ENABLE_NPKIT_PRIM_COLLECT_DATA_PROCESS_TIME))
    if (tid == 0) {
      npKitWaitRecvExitTime = NPKIT_GET_GPU_TIMESTAMP();
      npKitWaitRecvTotalTime += (npKitWaitRecvExitTime - npKitWaitRecvEntryTime) * (npkitWaitRecvSpins - 1) / npkitWaitRecvSpins;
    }
#endif

    return val64;
  }

  __device__ void storeLL(union ncclLLFifoLine* dst, uint64_t val, uint32_t flag) {
#if defined(__HIP_PLATFORM_AMD__) || defined(__HIPCC__)
    union ncclLLFifoLine i4;
    i4.data1 = val & 0xffffffff;
    i4.flag1 = flag;
    i4.data2 = (val >> 32);
    i4.flag2 = flag;
    __builtin_nontemporal_store(i4.v[0], dst->v);
    __builtin_nontemporal_store(i4.v[1], dst->v+1);
#else
    asm volatile("st.volatile.global.v4.u32 [%0], {%1,%2,%3,%4};" :: "l"(&dst->i4), "r"((uint32_t)val), "r"(flag), "r"((uint32_t)(val >> 32)), "r"(flag));
#endif
  }

  static constexpr int EltPerLine = sizeof(uint64_t)/sizeof(T);

  template<typename U>
  __device__ static U load(U *src) {
    union {
      U elt;
      uint8_t u1;
      uint16_t u2;
      uint32_t u4;
      uint64_t u8;
    };
#if defined(__HIP_PLATFORM_AMD__) || defined(__HIPCC__)
    if(sizeof(U) == 1)
#ifdef __GFX11__
      u1 = __atomic_load_n((uint8_t*)src, __ATOMIC_RELAXED);
#else
      u1 = __builtin_nontemporal_load((uint8_t*)src);
#endif
    else if(sizeof(U) == 2)
#ifdef __GFX11__
      u2 = __atomic_load_n((uint16_t*)src, __ATOMIC_RELAXED);
#else
      u2 = __builtin_nontemporal_load((uint16_t*)src);
#endif
    else if(sizeof(U) == 4)
#ifdef __GFX11__
      u4 = __atomic_load_n((uint32_t*)src, __ATOMIC_RELAXED);
#else
      u4 = __builtin_nontemporal_load((uint32_t*)src);
#endif
    else
#ifdef __GFX11__
      u8 = __atomic_load_n((uint64_t*)src, __ATOMIC_RELAXED);
#else
      u8 = __builtin_nontemporal_load((uint64_t*)src);
#endif
#else
    if(sizeof(U) == 1)
      asm("ld.volatile.global.b8 %0,[%1];" : "=r"(u4) : "l"(src));
    else if(sizeof(U) == 2)
      asm("ld.volatile.global.b16 %0,[%1];" : "=h"(u2) : "l"(src));
    else if(sizeof(U) == 4)
      asm("ld.volatile.global.b32 %0,[%1];" : "=r"(u4) : "l"(src));
    else
      asm("ld.volatile.global.b64 %0,[%1];" : "=l"(u8) : "l"(src));
#endif
    return elt;
  }

  template<typename U>
  __device__ static void store(U *dst, U val) {
    union {
      U elt;
      uint8_t u1;
      uint16_t u2;
      uint32_t u4;
      uint64_t u8;
    };
    elt = val;
#if defined(__HIP_PLATFORM_AMD__) || defined(__HIPCC__)
    if(sizeof(U) == 1)
      __builtin_nontemporal_store(u1, (uint8_t*)dst);
    else if(sizeof(U) == 2)
      __builtin_nontemporal_store(u2, (uint16_t*)dst);
    else if(sizeof(U) == 4)
      __builtin_nontemporal_store(u4, (uint32_t*)dst);
    else
      __builtin_nontemporal_store(u8, (uint64_t*)dst);
#else
    if(sizeof(U) == 1)
      asm("st.volatile.global.b8 [%0],%1;" :: "l"(dst), "r"(u4));
    else if(sizeof(U) == 2)
      asm("st.volatile.global.b16 [%0],%1;" :: "l"(dst), "h"(u2));
    else if(sizeof(U) == 4)
      asm("st.volatile.global.b32 [%0],%1;" :: "l"(dst), "r"(u4));
    else
      asm("st.volatile.global.b64 [%0],%1;" :: "l"(dst), "l"(u8));
#endif
  }

  struct DataLoader {
    int misalign;
    union {
      uint32_t u4[sizeof(T) <= 2 ? 3 : 2];
      uint64_t u8;
      T elt[EltPerLine];
    };

    __device__ void loadBegin(T *src, int eltN) {
      if (sizeof(T) <= 2) {
        misalign = reinterpret_cast<uintptr_t>(src)%4;
        uint32_t *p = reinterpret_cast<uint32_t*>(reinterpret_cast<uintptr_t>(src) & -uintptr_t(4));
        u4[0] = load(p+0);
        u4[1] = misalign + eltN*sizeof(T) > 4 ? load(p+1) : 0;
        // u4[2] would be simpler, but that throws warnings on some compilers
        u4[sizeof(T) <= 2 ? 2 : 0] = misalign + eltN*sizeof(T) > 8 ? load(p+2) : 0;
      }
      else {
        #pragma unroll
        for(int i=0; i < EltPerLine; i++) {
          if(i==0 || i < eltN)
            elt[i] = load(src + i);
        }
      }
    }

    __device__ uint64_t loadFinish() {
      if (sizeof(T) <= 2) {
        u4[0] = __funnelshift_r(u4[0], u4[1], 8*misalign);
        // u4[2] would be simpler, but that throws warnings on some compilers
        u4[1] = __funnelshift_r(u4[1], u4[sizeof(T) <= 2 ? 2 : 0], 8*misalign);
      }
      return u8;
    }
  };

  __device__ void storeData(T *dst, uint64_t val, int eltN) {
    union {
      uint64_t u8;
      T elt[EltPerLine];
    };
    u8 = val;
    #pragma unroll
    for(int i=0; i < EltPerLine; i++) {
      if (i==0 || i < eltN)
        //store(dst+i, elt[i]);
        dst[i] = elt[i];
    }
  }

  __device__ void mscclStoreData(T *dst, uint64_t val, int eltN) {
    union {
      uint64_t u8;
      T elt[EltPerLine];
    };
    u8 = val;
    #pragma unroll
    for(int i=0; i < EltPerLine; i++) {
      if (i==0 || i < eltN)
        store(dst+i, elt[i]);
        // dst[i] = elt[i];
    }
  }

  template <int RECV, int SEND, int SrcBuf, int DstBuf>
  __device__ void LLGenericOp(intptr_t srcIx, intptr_t dstIx, int nelem, bool postOp) {
    constexpr int SRC = SrcBuf != -1 ? 1 : 0;
    constexpr int DST = DstBuf != -1 ? 1 : 0;
    T *srcElts = SrcBuf == -1 ? nullptr : userBufs[SrcBuf] + srcIx;
    T *dstElts = DstBuf == -1 ? nullptr : userBufs[DstBuf] + dstIx;

    // Always waitSend in case of cleanup
    nelem = nelem < 0 ? 0 : nelem;
    if (SEND) waitSend(divUp(nelem, EltPerLine)*sizeof(ncclLLFifoLine));

#if defined(ENABLE_NPKIT) && defined(ENABLE_NPKIT_EVENT_PRIM_LL_DATA_PROCESS_ENTRY) && defined(ENABLE_NPKIT_EVENT_PRIM_LL_DATA_PROCESS_EXIT)
    if (tid == 0) {
      npKitWaitRecvTotalTime = 0;
      npKitWaitRecvDataProcessSize = nelem*sizeof(T);
      NpKit::CollectGpuEvent(NPKIT_EVENT_PRIM_LL_DATA_PROCESS_ENTRY,
          npKitWaitRecvDataProcessSize, 0, NPKIT_GET_GPU_TIMESTAMP(), ncclShmem.comm.npKitEventCollectContexts + npKitCtxIdx);
    }
#endif

#if defined(ENABLE_NPKIT) && defined(ENABLE_NPKIT_PRIM_COLLECT_DATA_PROCESS_TIME)
    if (tid == 0) {
      npKitWaitRecvTotalTime = 0;
      npKitDataProcessEntryTime = NPKIT_GET_GPU_TIMESTAMP();
    }
#endif

    nelem -= tid*EltPerLine;
    srcElts += tid*EltPerLine;
    dstElts += tid*EltPerLine;
    int offset = tid;
    int eltPerTrip = nthreads*EltPerLine;
    while (nelem > 0) {
      int eltInLine = EltPerLine < nelem ? EltPerLine : nelem;

      DataLoader dl;
      ncclLLFifoLine line[MaxRecv];
      uint64_t data, peerData;
      if (SRC) {
        dl.loadBegin(srcElts, eltInLine);
        srcElts += eltPerTrip;
      }
      if (RECV) {
        readLLBeginAll<1>(offset, line);
        peerData = readLL(offset, 0);
      }
      if (SRC) {
        data = dl.loadFinish();
        if (SrcBuf == Input) data = applyPreOp(redOp, data);
      }
      if (RECV) {
        data = !SRC ? peerData : applyReduce(redOp, peerData, data);
        #pragma unroll MaxRecv
        for (int i=1; i < MaxRecv && i < fan.nrecv(); i++) {
          peerData = readLLFinish(offset, line, i);
          data = applyReduce(redOp, peerData, data);
        }
      }

      if (postOp) data = applyPostOp(redOp, data);

      // Send : inter-node, then intra-node, then local
      if (SEND) {
        for (int i=1; i < MaxSend && i < fan.nsend(); i++)
          storeLL(sendPtr(i)+offset, data, sendFlag(i));
        storeLL(sendPtr(0)+offset, data, sendFlag(0));
      }
      if (DST) {
        storeData(dstElts, data, eltInLine);
        dstElts += eltPerTrip;
      }
      nelem -= eltPerTrip;
      offset += nthreads;
    }

#if defined(ENABLE_NPKIT) && defined(ENABLE_NPKIT_PRIM_COLLECT_DATA_PROCESS_TIME)
    if (tid == 0) {
      npKitDataProcessExitTime = NPKIT_GET_GPU_TIMESTAMP();
      npKitDataProcessTotalTime += npKitDataProcessExitTime - npKitDataProcessEntryTime - npKitWaitRecvTotalTime;
    }
#endif

#if defined(ENABLE_NPKIT) && defined(ENABLE_NPKIT_EVENT_PRIM_LL_DATA_PROCESS_ENTRY) && defined(ENABLE_NPKIT_EVENT_PRIM_LL_DATA_PROCESS_EXIT)
    if (tid == 0) {
      NpKit::CollectGpuEvent(NPKIT_EVENT_PRIM_LL_DATA_PROCESS_EXIT,
          npKitWaitRecvDataProcessSize, npKitWaitRecvTotalTime, NPKIT_GET_GPU_TIMESTAMP(),
          ncclShmem.comm.npKitEventCollectContexts + npKitCtxIdx);
    }
#endif

    if (RECV) {
      for (int i=0; i < MaxRecv; i++) incRecv(i);
      postRecv();
    }
    if (SEND) {
      for (int i=1; i < MaxSend && i < fan.nsend(); i++)
        incSend(i, offset);
      incSend(0, offset);
    }
  }

  template <int REDUCE, int COPY, int MULTISRCS, int MULTIDSTS>
  __device__ __forceinline__ void mscclGenericOp(T** srcs, int nsrcs, T** dsts, int ndsts, int nelem) {
#if defined(ENABLE_NPKIT) && defined(ENABLE_NPKIT_EVENT_MSCCL_GENERIC_OP_ENTRY)
    if (tid == 0) {
      NpKit::CollectGpuEvent(NPKIT_EVENT_MSCCL_GENERIC_OP_ENTRY, nelem*sizeof(T), 0, NPKIT_GET_GPU_TIMESTAMP(),
          ncclShmem.comm.npKitEventCollectContexts + npKitCtxIdx);
    }
#endif

    nelem = nelem < 0 ? 0 : nelem;
    T *srcElts = srcs[0];
    T *dstElts = dsts[0];
    nelem -= tid*EltPerLine;
    srcElts += tid*EltPerLine;
    dstElts += tid*EltPerLine;
    if (MULTISRCS){
      for (int i = 1; i < nsrcs; i++){
        srcs[i] += tid*EltPerLine;
      }
    }
    if (MULTIDSTS){
      for (int i = 1; i < ndsts; i++){
        dsts[i] += tid*EltPerLine;
      }
    }
    int eltPerTrip = nthreads*EltPerLine;
    while (nelem > 0) {
      int eltInLine = EltPerLine < nelem ? EltPerLine : nelem;

      DataLoader dl;
      uint64_t data;
      dl.loadBegin(srcElts, eltInLine);
      srcElts += eltPerTrip;
      data = dl.loadFinish();
      if (REDUCE) {
        uint64_t dataD;
        dl.loadBegin(dstElts, eltInLine);
        dataD = dl.loadFinish();
        dataD = applyReduce(redOp, dataD, data);
        if (MULTISRCS){
          for (int i = 1; i < nsrcs; i++){
            dl.loadBegin(srcs[i], eltInLine);
            srcs[i] += eltPerTrip;
            data = dl.loadFinish();
            dataD = applyReduce(redOp, dataD, data);
          }
        }
        mscclStoreData(dstElts, dataD, eltInLine);
        dstElts += eltPerTrip;
      }
      if (COPY){
        mscclStoreData(dstElts, data, eltInLine);
        dstElts += eltPerTrip;
        if (MULTIDSTS){
          for (int i = 1; i < ndsts; i++){
            dl.loadBegin(srcs[i], eltInLine);
            srcs[i] += eltPerTrip;
            data = dl.loadFinish();
            mscclStoreData(dsts[i], data, eltInLine);
            dsts[i] += eltPerTrip;
          }
        }
      }
      nelem -= eltPerTrip;
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
    recvBuff[i] = (union ncclLLFifoLine*)conn->buffs[NCCL_PROTO_LL];
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
    sendBuff[i] = (union ncclLLFifoLine*)conn->buffs[NCCL_PROTO_LL];
    sendStep[i] = conn->step;
    if (wid == i) sendConn = conn;
  }
  __device__ __forceinline__ void loadSendSync() {
    if (tid < fan.nsend()) {
      sendConnHeadPtr = sendConn->head;
      sendConnHeadCache = *sendConnHeadPtr;
      sendConnHead = sendConn->step;
      sendConnFifo = sendConn->connFifo;
    }
  }

 public:
  __device__  Primitives(
      const int tid, const int nthreads, int const *recvPeers, int const *sendPeers,
      void const *inputBuf, void *outputBuf, uint64_t redOpArg, uint8_t group=0,
      uint8_t connIndexRecv=0, uint8_t connIndexSend=0, struct ncclWorkElem* e = nullptr, struct ncclWorkElemP2p* p2p = nullptr, int stepSize_=0
    ):
    redOp(redOpArg),
    tid(tid), nthreads(nthreads), wid(tid%WARP_SIZE), group(group),
    stepLines(ncclShmem.comm.buffSizes[NCCL_PROTO_LL]/NCCL_STEPS/sizeof(ncclLLFifoLine)) {
    auto *channel = &ncclShmem.channel;
    barriers = &ncclShmem.groups[group].barrier;
    barrier_next = ncclShmem.groups[group].barrier_next;
    // If we are going to support oneshot collNet + LL, then we would need to add connector index here
    int nrecv=0, nsend=0;
    // We compare with Fan::MaxRecv here because this->MaxRecv is always at least 1
    while (nrecv < Fan::MaxRecv && recvPeers[nrecv] >= 0) {
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
#if defined(ENABLE_NPKIT) && defined(ENABLE_NPKIT_EVENT_SEND_ENTRY)
    if (tid == 0) {
      NpKit::CollectGpuEvent(NPKIT_EVENT_SEND_ENTRY, eltN*sizeof(T), 0, NPKIT_GET_GPU_TIMESTAMP(),
          ncclShmem.comm.npKitEventCollectContexts + npKitCtxIdx);
    }
#endif
    LLGenericOp<0, 1, Input, -1>(inpIx, -1, eltN, false);
#if defined(ENABLE_NPKIT) && defined(ENABLE_NPKIT_EVENT_SEND_EXIT)
    if (tid == 0) {
      NpKit::CollectGpuEvent(NPKIT_EVENT_SEND_EXIT, eltN*sizeof(T), 0, NPKIT_GET_GPU_TIMESTAMP(),
          ncclShmem.comm.npKitEventCollectContexts + npKitCtxIdx);
    }
#endif
  }
  __device__ void sendFromOutput(intptr_t outIx, int eltN) {
#if defined(ENABLE_NPKIT) && defined(ENABLE_NPKIT_EVENT_SEND_FROM_OUTPUT_ENTRY)
    if (tid == 0) {
      NpKit::CollectGpuEvent(NPKIT_EVENT_SEND_FROM_OUTPUT_ENTRY, eltN*sizeof(T), 0, NPKIT_GET_GPU_TIMESTAMP(),
          ncclShmem.comm.npKitEventCollectContexts + npKitCtxIdx);
    }
#endif
    LLGenericOp<0, 1, Output, -1>(outIx, -1, eltN, false);
#if defined(ENABLE_NPKIT) && defined(ENABLE_NPKIT_EVENT_SEND_FROM_OUTPUT_EXIT)
    if (tid == 0) {
      NpKit::CollectGpuEvent(NPKIT_EVENT_SEND_FROM_OUTPUT_EXIT, eltN*sizeof(T), 0, NPKIT_GET_GPU_TIMESTAMP(),
          ncclShmem.comm.npKitEventCollectContexts + npKitCtxIdx);
    }
#endif
  }
  __device__ void recv(intptr_t outIx, int eltN, bool postOp=false) {
#if defined(ENABLE_NPKIT) && defined(ENABLE_NPKIT_EVENT_RECV_ENTRY)
    if (tid == 0) {
      NpKit::CollectGpuEvent(NPKIT_EVENT_RECV_ENTRY, eltN*sizeof(T), 0, NPKIT_GET_GPU_TIMESTAMP(),
          ncclShmem.comm.npKitEventCollectContexts + npKitCtxIdx);
    }
#endif
    LLGenericOp<1, 0, -1, Output>(-1, outIx, eltN, postOp);
#if defined(ENABLE_NPKIT) && defined(ENABLE_NPKIT_EVENT_RECV_EXIT)
    if (tid == 0) {
      NpKit::CollectGpuEvent(NPKIT_EVENT_RECV_EXIT, eltN*sizeof(T), 0, NPKIT_GET_GPU_TIMESTAMP(),
          ncclShmem.comm.npKitEventCollectContexts + npKitCtxIdx);
    }
#endif
  }
  __device__ void recvReduceSend(intptr_t inpIx, int eltN) {
#if defined(ENABLE_NPKIT) && defined(ENABLE_NPKIT_EVENT_RECV_REDUCE_SEND_ENTRY)
    if (tid == 0) {
      NpKit::CollectGpuEvent(NPKIT_EVENT_RECV_REDUCE_SEND_ENTRY, eltN*sizeof(T), 0, NPKIT_GET_GPU_TIMESTAMP(),
          ncclShmem.comm.npKitEventCollectContexts + npKitCtxIdx);
    }
#endif
    LLGenericOp<1, 1, Input, -1>(inpIx, -1, eltN, false);
#if defined(ENABLE_NPKIT) && defined(ENABLE_NPKIT_EVENT_RECV_REDUCE_SEND_EXIT)
    if (tid == 0) {
      NpKit::CollectGpuEvent(NPKIT_EVENT_RECV_REDUCE_SEND_EXIT, eltN*sizeof(T), 0, NPKIT_GET_GPU_TIMESTAMP(),
          ncclShmem.comm.npKitEventCollectContexts + npKitCtxIdx);
    }
#endif
  }
  __device__ void recvReduceCopy(intptr_t inpIx, intptr_t outIx, int eltN, bool postOp=false) {
#if defined(ENABLE_NPKIT) && defined(ENABLE_NPKIT_EVENT_RECV_REDUCE_COPY_ENTRY)
    if (tid == 0) {
      NpKit::CollectGpuEvent(NPKIT_EVENT_RECV_REDUCE_COPY_ENTRY, eltN*sizeof(T), 0, NPKIT_GET_GPU_TIMESTAMP(),
          ncclShmem.comm.npKitEventCollectContexts + npKitCtxIdx);
    }
#endif
    LLGenericOp<1, 0, Input, Output>(inpIx, outIx, eltN, postOp);
#if defined(ENABLE_NPKIT) && defined(ENABLE_NPKIT_EVENT_RECV_REDUCE_COPY_EXIT)
    if (tid == 0) {
      NpKit::CollectGpuEvent(NPKIT_EVENT_RECV_REDUCE_COPY_EXIT, eltN*sizeof(T), 0, NPKIT_GET_GPU_TIMESTAMP(),
          ncclShmem.comm.npKitEventCollectContexts + npKitCtxIdx);
    }
#endif
  }
  __device__ void copySend(intptr_t inpIx, intptr_t outIx, int eltN, bool postOp=false) {
#if defined(ENABLE_NPKIT) && defined(ENABLE_NPKIT_EVENT_COPY_SEND_ENTRY)
    if (tid == 0) {
      NpKit::CollectGpuEvent(NPKIT_EVENT_COPY_SEND_ENTRY, eltN*sizeof(T), 0, NPKIT_GET_GPU_TIMESTAMP(),
          ncclShmem.comm.npKitEventCollectContexts + npKitCtxIdx);
    }
#endif
    LLGenericOp<0, 1, Input, Output>(inpIx, outIx, eltN, postOp);
#if defined(ENABLE_NPKIT) && defined(ENABLE_NPKIT_EVENT_COPY_SEND_EXIT)
    if (tid == 0) {
      NpKit::CollectGpuEvent(NPKIT_EVENT_COPY_SEND_EXIT, eltN*sizeof(T), 0, NPKIT_GET_GPU_TIMESTAMP(),
          ncclShmem.comm.npKitEventCollectContexts + npKitCtxIdx);
    }
#endif
  }
  __device__ void recvCopySend(intptr_t outIx, int eltN, bool postOp=false) {
#if defined(ENABLE_NPKIT) && defined(ENABLE_NPKIT_EVENT_RECV_COPY_SEND_ENTRY)
    if (tid == 0) {
      NpKit::CollectGpuEvent(NPKIT_EVENT_RECV_COPY_SEND_ENTRY, eltN*sizeof(T), 0, NPKIT_GET_GPU_TIMESTAMP(),
          ncclShmem.comm.npKitEventCollectContexts + npKitCtxIdx);
    }
#endif
    LLGenericOp<1, 1, -1, Output>(-1, outIx, eltN, postOp);
#if defined(ENABLE_NPKIT) && defined(ENABLE_NPKIT_EVENT_RECV_COPY_SEND_EXIT)
    if (tid == 0) {
      NpKit::CollectGpuEvent(NPKIT_EVENT_RECV_COPY_SEND_EXIT, eltN*sizeof(T), 0, NPKIT_GET_GPU_TIMESTAMP(),
          ncclShmem.comm.npKitEventCollectContexts + npKitCtxIdx);
    }
#endif
  }
  __device__ void recvReduceCopySend(intptr_t inpIx, intptr_t outIx, int eltN, bool postOp=false) {
#if defined(ENABLE_NPKIT) && defined(ENABLE_NPKIT_EVENT_RECV_REDUCE_COPY_SEND_ENTRY)
    if (tid == 0) {
      NpKit::CollectGpuEvent(NPKIT_EVENT_RECV_REDUCE_COPY_SEND_ENTRY, eltN*sizeof(T), 0, NPKIT_GET_GPU_TIMESTAMP(),
          ncclShmem.comm.npKitEventCollectContexts + npKitCtxIdx);
    }
#endif
    LLGenericOp<1, 1, Input, Output>(inpIx, outIx, eltN, postOp);
#if defined(ENABLE_NPKIT) && defined(ENABLE_NPKIT_EVENT_RECV_REDUCE_COPY_SEND_EXIT)
    if (tid == 0) {
      NpKit::CollectGpuEvent(NPKIT_EVENT_RECV_REDUCE_COPY_SEND_EXIT, eltN*sizeof(T), 0, NPKIT_GET_GPU_TIMESTAMP(),
          ncclShmem.comm.npKitEventCollectContexts + npKitCtxIdx);
    }
#endif
  }
  __device__ void recvSend(int eltN) {
    return LLGenericOp<1, 1, -1, -1>(-1, -1, eltN, false);
  }

  // MSCCL primitives
  __device__ void sendWithBarrier(intptr_t inpIx, int eltN) {
    send(inpIx, eltN);
    // This is the only primitive.instruction where there is no barrier at the end, add it
    barrier();
  }
  __device__ void localCopy(T* srcs, T* dsts, int eltN) {
    return mscclGenericOp<0,1,0,0>(&srcs, 1, &dsts, 1, eltN);
  }
};
