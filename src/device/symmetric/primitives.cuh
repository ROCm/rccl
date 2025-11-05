// Modification Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT 

#ifndef NCCL_DEVICE_SYMMETRIC_PRIMITIVES_H_
#define NCCL_DEVICE_SYMMETRIC_PRIMITIVES_H_

#include "sym_kernels.h"
#include "bitops.h"
#include "collectives.h"
<<<<<<< HEAD
#include "op128.h"
#include "reduce_kernel.h"
#include "common.h"
=======
#include "../op128.h"
#include "../reduce_kernel.h"
>>>>>>> f1308997d0420148b1be1c24d63f19d902ae589b

#if __CUDA_ARCH__ >= 700
// __grid_constant__ appears to break cuda-gdb
#define NCCL_GRID_CONSTANT __grid_constant__
#else
#define NCCL_GRID_CONSTANT
#endif

// flattenIx(pos0, dim0, pos1, dim1, pos2, dim2, ...)
// Given a position vector `pos` in a rectangular index space with lengths in the `dim`
// vector, flatten that down to a linear index. The fastest moving dimension is given first.
__device__ __forceinline__ int flattenIx() { return 0; }

template<typename Int0, typename Int1, typename ...Ints>
static __device__ Int0 flattenIx(Int0 pos, Int1 size, Ints ...more) {
  return pos + size*flattenIx(more...);
}

namespace {
struct ncclSymkArgsHandler {
  ncclDevComm const& comm;
  ncclLLA2AHandle const& lsaLLA2A;
  struct ncclSymkChannelWorkRange* channelWorkRange;
  struct ncclSymkDevWork* devWork;
  uint32_t nRanks_rcp32;

  __device__ ncclSymkArgsHandler(ncclSymkDevWorkArgs const* args):
    comm(args->kcomm.devComm),
    lsaLLA2A(args->kcomm.lsaLLA2A) {
    channelWorkRange = args->getWorkRange();

    devWork = args->getWorks(args->nMaxChannels);
    nRanks_rcp32 = comm.nRanks_rcp32;
  }

  template<typename T>
    __device__ void getWorkRange(int block,
                                 uint16_t& workLo, size_t& indexLo, uint16_t& workHi, size_t& indexHi) {
    constexpr int EltPerCell = NCCL_SYM_KERNEL_CELL_SIZE / sizeof(T);
    uint32_t fracLo, fracHi;

    // Where the work begins
    workLo = (block==0) ? 0 : channelWorkRange[block-1].workHi; // start where predecessor ends
    fracLo = (block==0) ? 0 : channelWorkRange[block-1].fracHi + 1;
    // If the predecessor ended on the work boundary, then we step to the beginning of the next work.
    // This ensures we never have empty parts.
    if (fracLo == 0x10000) {
      workLo++;
      fracLo = 0;
    }
    struct ncclSymkDevWork const& dw = devWork[workLo];
    indexLo = ((fracLo * divUp(dw.nElts, EltPerCell)) >> 16) * EltPerCell;

    // Where the work ends
    workHi = channelWorkRange[block].workHi;
    fracHi = channelWorkRange[block].fracHi + 1;
    indexHi = min(((fracHi * divUp(dw.nElts, EltPerCell)) >> 16) * EltPerCell, dw.nElts);
  }

  template<typename T>
    __device__ void getWorkRangeFused(int blockIdx, int w,
                                      int& block, int& nBlocks, size_t& indexLo, size_t& indexHi) {
    constexpr int EltPerCell = NCCL_SYM_KERNEL_CELL_SIZE / sizeof(T);
    struct ncclSymkDevWork const& dw = devWork[w];
    uint32_t fracLo, fracHi;
    int lastBlock;

    block = blockIdx - dw.sChannelId;
    nBlocks = dw.nChannels;
    lastBlock = dw.sChannelId+dw.nChannels-1;

    // Where the work begins
    fracLo = (dw.sChannelId==0) ? 0 : ((channelWorkRange[dw.sChannelId-1].fracHi + 1) & 0xFFFF);
    indexLo = ((fracLo * divUp(dw.nElts, EltPerCell)) >> 16) * EltPerCell;
    fracHi = (channelWorkRange[lastBlock].workHi == w) ? channelWorkRange[lastBlock].fracHi + 1 : 0x10000;
    indexHi = min(((fracHi * divUp(dw.nElts, EltPerCell)) >> 16) * EltPerCell, dw.nElts);
  }

  template<typename T, typename Fn>
    __device__ void forEachWork(Fn const& fn) {
      uint16_t workLo, workHi;
      size_t indexLo, indexHi;

      getWorkRange<T>(blockIdx.x, workLo, indexLo, workHi, indexHi);

      size_t currentIndexLo = indexLo;
      #pragma unroll 1
      for (int w = workLo; w <= workHi; w++) {
        struct ncclSymkDevWork const& dw = devWork[w];
        size_t const& nAllElts = dw.nElts;
        size_t currentIndexHi;
        int block, nBlocks;
        if (blockIdx.x >= dw.sChannelId && blockIdx.x < dw.sChannelId + dw.nChannels) {
          getWorkRangeFused<T>(blockIdx.x, w, block, nBlocks, currentIndexLo, currentIndexHi);
        } else {
          currentIndexHi = (w < workHi) ? nAllElts : indexHi;
          block = 0;
          nBlocks = 1;
        }

        fn(block, nBlocks, currentIndexHi - currentIndexLo, nAllElts,
           ncclSymPtr<T>(dw.inputWin, dw.inputOff) + currentIndexLo,
           ncclSymPtr<T>(dw.outputWin, dw.outputOff) + currentIndexLo);

        currentIndexLo = 0;
      }
<<<<<<< HEAD
    #endif
    } else {
      int r = cta.self();
      if (r != rank && r < nRanks) {
        uint32_t* inbox = &peerPtr(r, base)->barInboxPerPeer[block*nRanks + rank];
        #if __CUDA_ARCH__ >= 700
          if (release) {
            asm volatile("st.release.sys.u32 [%0],%1;" :: "l"(inbox), "r"(barEpoch+1));
          } else {
            asm volatile("st.relaxed.sys.u32 [%0],%1;" :: "l"(inbox), "r"(barEpoch+1));
          }
        #else
          if (release) {
            __atomic_store_n(inbox, barEpoch + 1, __ATOMIC_RELEASE);
          } else {
            __atomic_store_n(inbox, barEpoch + 1, __ATOMIC_RELAXED);
          }
          // asm volatile("st.volatile.u32 [%0],%1;" :: "l"(inbox), "r"(barEpoch+1));
        #endif
      }
    }
  }

  __device__  void barrierWait(ncclCoopCta cta, bool acquire) {
    if (flags & ncclSymPrims_UseMultimem) {
    #if __CUDA_ARCH__ >= 900
      if (cta.self() == 0) {
        uint32_t* inbox = &base->barInboxMc[block];
        while (true) {
          uint32_t got;
          if (acquire) {
            asm volatile("ld.acquire.sys.u32 %0,[%1];" : "=r"(got) : "l"(inbox));
          } else {
            asm volatile("ld.relaxed.sys.u32 %0,[%1];" : "=r"(got) : "l"(inbox));
          }
          if (got-(barEpoch+nRanks) <= uint32_t(-1)>>1) break;
        }
        barEpoch += nRanks;
      }
    #endif
    } else {
      int r = cta.self();
      if (r != rank && r < nRanks) {
        uint32_t* inbox = &base->barInboxPerPeer[block*nRanks + r];
        while (true) {
          uint32_t got;
          #if __CUDA_ARCH__ >= 700
            if (acquire) {
              asm volatile("ld.acquire.sys.u32 %0,[%1];" : "=r"(got) : "l"(inbox));
            } else {
              asm volatile("ld.relaxed.sys.u32 %0,[%1];" : "=r"(got) : "l"(inbox));
            }
          #else
            if (acquire) {
              got = __atomic_load_n(inbox, __ATOMIC_ACQUIRE);
            } else {
              got = __atomic_load_n(inbox, __ATOMIC_RELAXED);
            }
            // asm volatile("ld.volatile.u32 %0,[%1];" : "=r"(got) : "l"(inbox));
          #endif
          if (got-(barEpoch+1) <= uint32_t(-1)>>1) break;
        }
      }
      #if __CUDA_ARCH__ < 700
        if (acquire) {
          cta.sync();
          if (cta.self() == 0) __threadfence();
        }
      #endif
      barEpoch += 1;
    }
    cta.sync();
  }
=======
  }

  template<typename T, typename Fn>
    __device__ void singleWork(Fn const& fn) {
      uint16_t w;
      size_t indexLo, indexHi;
>>>>>>> f1308997d0420148b1be1c24d63f19d902ae589b

      getWorkRange<T>(blockIdx.x, w, indexLo, w, indexHi);

<<<<<<< HEAD
  template<typename T>
  __device__ void sendLL(int peer, int slot, T val) {
    union { T tmp; uint32_t u32[divUp(sizeof(T),8)][2]; };
    tmp = val;
    uint4* buf = ncclSymDevBase_getLLBuf(peerPtr(peer, base), nRanks, block, llEpoch) + slot;
    #pragma unroll
    for (int u=0; u < divUp(sizeof(T),8); u++) {
      using Vec = uint32_t __attribute__((ext_vector_type(4)));
      Vec i4;
      i4[0] = u32[u][0];
      i4[1] = llEpoch;
      i4[2] = u32[u][1];
      i4[3] = llEpoch;
#if defined(__gfx950__)
      asm volatile ("flat_store_dwordx4 %0, %1 sc0 sc1 nt" :: "v"(buf + ncclSymLLMaxSlots(sizeof(T))*u), "v"(i4));
#else
      __builtin_nontemporal_store(i4, (Vec*)(buf + ncclSymLLMaxSlots(sizeof(T))*u));
#endif
      // asm volatile("st.volatile.v4.u32 [%0],{%1,%3,%2,%3};" :: "l"(buf + ncclSymLLMaxSlots(sizeof(T))*u), "r"(u32[u][0]), "r"(u32[u][1]), "r"(llEpoch));
    }
  }

  template<typename T>
  __device__ void bcastLL(int slot, T val) {
    if (flags & ncclSymPrims_UseMultimem) {
      union { T tmp; uint32_t u32[divUp(sizeof(T),8)][2]; };
      tmp = val;
      uint4* bufmc = ncclSymDevBase_getLLBuf(multimemPtr(base), nRanks, block, llEpoch) + slot;
      #pragma unroll
      for (int u=0; u < divUp(sizeof(T),8); u++) {
        using Vec = uint32_t __attribute__((ext_vector_type(4)));
        Vec i4;
        i4[0] = u32[u][0];
        i4[1] = llEpoch;
        i4[2] = u32[u][1];
        i4[3] = llEpoch;
#if defined(__gfx950__)
        asm volatile ("flat_store_dwordx4 %0, %1 sc0 sc1 nt" :: "v"(bufmc + ncclSymLLMaxSlots(sizeof(T))*u), "v"(i4));
#else
        __builtin_nontemporal_store(i4, (Vec*)(bufmc + ncclSymLLMaxSlots(sizeof(T))*u));
#endif
        // asm volatile("st.volatile.v4.u32 [%0],{%1,%3,%2,%3};" :: "l"(bufmc + ncclSymLLMaxSlots(sizeof(T))*u), "r"(u32[u][0]), "r"(u32[u][1]), "r"(llEpoch));
      }
    } else {
      union { T tmp; uint32_t u32[divUp(sizeof(T),8)][2]; };
      tmp = val;
      uint4* buf0 = ncclSymDevBase_getLLBuf(peerPtr(0, base), nRanks, block, llEpoch) + slot;
      int dr = 0;
      int r = rank;
      #pragma unroll 1
      for (; dr+8 <= nRanks; dr += 8) {
        #pragma unroll
        for (int ur=0; ur < 8; ur++) {
          uint4* buf = add4G(buf0, r*stride4G);
          #pragma unroll
          for (int u=0; u < divUp(sizeof(T),8); u++) {
            using Vec = uint32_t __attribute__((ext_vector_type(4)));
            Vec i4;
            i4[0] = u32[u][0];
            i4[1] = llEpoch;
            i4[2] = u32[u][1];
            i4[3] = llEpoch;
#if defined(__gfx950__)
            asm volatile ("flat_store_dwordx4 %0, %1 sc0 sc1 nt" :: "v"(buf + ncclSymLLMaxSlots(sizeof(T))*u), "v"(i4));
#else
            __builtin_nontemporal_store(i4, (Vec*)((buf + ncclSymLLMaxSlots(sizeof(T))*u)));
#endif
            // asm volatile("st.volatile.v4.u32 [%0],{%1,%3,%2,%3};" :: "l"(buf + ncclSymLLMaxSlots(sizeof(T))*u), "r"(u32[u][0]), "r"(u32[u][1]), "r"(llEpoch));
          }
          r += 1;
          if (r == nRanks) r = 0;
        }
      }
      #pragma unroll
      for (int ur=0; ur < 8; ur++, dr++) {
        if (dr == nRanks) break;
        uint4* buf = add4G(buf0, r*stride4G);
        #pragma unroll
        for (int u=0; u < divUp(sizeof(T),8); u++) {
          using Vec = uint32_t __attribute__((ext_vector_type(4)));
          Vec i4;
          i4[0] = u32[u][0];
          i4[1] = llEpoch;
          i4[2] = u32[u][1];
          i4[3] = llEpoch;
#if defined(__gfx950__)
          asm volatile ("flat_store_dwordx4 %0, %1 sc0 sc1 nt" :: "v"(buf + ncclSymLLMaxSlots(sizeof(T))*u), "v"(i4));
#else
          __builtin_nontemporal_store(i4, (Vec*)(buf + ncclSymLLMaxSlots(sizeof(T))*u));
#endif
          // asm volatile("st.volatile.v4.u32 [%0],{%1,%3,%2,%3};" :: "l"(buf + ncclSymLLMaxSlots(sizeof(T))*u), "r"(u32[u][0]), "r"(u32[u][1]), "r"(llEpoch));
        }
        r += 1;
        if (r == nRanks) r = 0;
      }
    }
  }

  template<int nSlotsMin, int nSlotsMax, typename T>
  __device__ void recvLL(int slot0, int nSlots, int stride, T(&elts)[nSlotsMax]) {
    uint4* buf = ncclSymDevBase_getLLBuf(base, nRanks, block, llEpoch) + slot0;
    uint4 tmp[nSlotsMax][divUp(sizeof(T),8)];
    //int spins=0;
    while (true) {
      #pragma unroll
      for (int u=0; u < nSlotsMax; u++) {
        if (u < nSlotsMin || u < nSlots) {
          #pragma unroll
          for (int v=0; v < divUp(sizeof(T),8); v++) {
            tmp[u][v] = *(buf + u * stride + v * ncclSymLLMaxSlots(sizeof(T)));
            // asm volatile("ld.volatile.v4.u32 {%0,%1,%2,%3},[%4];" : "=r"(tmp[u][v].x), "=r"(tmp[u][v].y), "=r"(tmp[u][v].z), "=r"(tmp[u][v].w) : "l"(buf + u*stride + v*ncclSymLLMaxSlots(sizeof(T))));
          }
        }
      }
      bool okAll = true;
      #pragma unroll
      for (int u=0; u < nSlotsMax; u++) {
        #pragma unroll
        for (int v=0; v < divUp(sizeof(T),8); v++) {
          if (u < nSlotsMin || u < nSlots) {
            bool ok = tmp[u][v].y == llEpoch &&
                      tmp[u][v].w == llEpoch;
            okAll &= ok;
          }
        }
      }
      if (__builtin_expect(okAll, true)) break;
      //if (spins++ == 10<<20) spins=0;
    }
    #pragma unroll
    for (int u=0; u < nSlotsMax; u++) {
      if (nSlotsMin <= u && u == nSlots) break;
      union { T val; uint32_t u32[divUp(sizeof(T),8)][2]; };
      #pragma unroll
      for (int v=0; v < divUp(sizeof(T),8); v++) {
        u32[v][0] = tmp[u][v].x;
        u32[v][1] = tmp[u][v].z;
      }
      elts[u] = val;
    }
  }

  template<typename Pack, typename T, typename Red, int Unroll=8>
  __device__ Pack recvReduceLL(int slot, int stride, Red red) {
    using Acc = typename Red::EltType;
    using AccPack = BytePack<sizeof(Pack)*sizeof(Acc)/sizeof(T)>;
    AccPack acc;
    bool first = true;
    int r = 0;
    #pragma unroll 1
    for (; r+Unroll <= nRanks; r += Unroll) {
      Pack got[Unroll];
      this->template recvLL</*Min=*/Unroll>(slot + r*stride, Unroll, stride, got);
      AccPack acc0 = applyCast<T, Acc>(got[0]);
      acc = first ? acc0 : applyReduce(red, acc, acc0);
      first = false;
      #pragma unroll
      for (int i=1; i < Unroll; i++) acc = applyReduce(red, acc, applyCast<T, Acc>(got[i]));
    }
    if (r < nRanks) {
      Pack got[Unroll];
      this->template recvLL</*Min=*/1>(slot + r*stride, nRanks-r, stride, got);
      AccPack acc0 = applyCast<T, Acc>(got[0]);
      acc = first ? acc0 : applyReduce(red, acc, acc0);
      #pragma unroll
      for (int i=1; i < Unroll-1; i++) {
        if (r+i < nRanks) acc = applyReduce(red, acc, applyCast<T, Acc>(got[i]));
      }
    }
    return applyCast<Acc, T>(acc);
  }

  template<typename T>
  __device__ T recvLL(int slot) {
    T one[1];
    this->template recvLL<1, 1, T>(slot, 1, 0, one);
    return one[0];
  }

  template<typename Coop, typename T>
  __device__ void coopRecvLL(Coop coop, int slot0, int nSlots, T* dst) {
    int me = coop.self();
    if (me < nSlots) {
      uint4* buf = ncclSymDevBase_getLLBuf(base, nRanks, block, llEpoch) + slot0 + me;
      uint4 got[divUp(sizeof(T), 8)];
      //int spins=0;
      #pragma unroll 1
      while (true) {
        #pragma unroll
        for (int u=0; u < divUp(sizeof(T), 8); u++) {
          got[u] = *((buf + u * ncclSymLLMaxSlots(sizeof(T))));
          // asm volatile("ld.volatile.v4.u32 {%0,%1,%2,%3},[%4];" : "=r"(got[u].x), "=r"(got[u].y), "=r"(got[u].z), "=r"(got[u].w) : "l"(buf + u*ncclSymLLMaxSlots(sizeof(T))));
        }
        bool ok = true;
        #pragma unroll
        for (int u=0; u < divUp(sizeof(T), 8); u++) {
          ok &= got[u].y == llEpoch;
          ok &= got[u].w == llEpoch;
        }
        if (__builtin_expect(ok, true)) break;
        //if (++spins == 10<<20) { spins=0; printf("r=%d LL spin @ ix=%d got=%d want=%d\n", rank, slot0+me, got[0].y, llEpoch); }
      }
      union { T val; uint32_t u32[divUp(sizeof(T), 8)][2]; };
      #pragma unroll
      for (int u=0; u < divUp(sizeof(T), 8); u++) {
        u32[u][0] = got[u].x;
        u32[u][1] = got[u].z;
      }
      dst[slot0 + me] = val;
    }
=======
      struct ncclSymkDevWork const& dw = devWork[w];

      fn(indexHi - indexLo, dw.nElts,
         ncclSymPtr<T>(dw.inputWin, dw.inputOff) + indexLo,
         ncclSymPtr<T>(dw.outputWin, dw.outputOff) + indexLo);
>>>>>>> f1308997d0420148b1be1c24d63f19d902ae589b
  }
};
}

template<template<typename> typename Red, typename T, bool nvls>
struct ncclSymkAccumType { using Type = T; };

// Only Red's whose opArg is invariant w.r.t. the datatype can have a different
// accumulator type. At the moment this excludes integer min/max, sumpostdiv,
// and premulsum.
template<> struct ncclSymkAccumType<FuncSum, __half, false> { using Type = float; };
#if defined(__CUDA_BF16_TYPES_EXIST__)
template<> struct ncclSymkAccumType<FuncSum, __nv_bfloat16, false> { using Type = float; };
#endif
#if defined(__CUDA_FP8_TYPES_EXIST__)
template<> struct ncclSymkAccumType<FuncSum, __nv_fp8_e4m3, false> { using Type = float; };
template<> struct ncclSymkAccumType<FuncSum, __nv_fp8_e5m2, false> { using Type = float; };
#endif
#endif
