/*************************************************************************
 * Copyright (c) 2017-2022, NVIDIA CORPORATION. All rights reserved.
 * Modifications Copyright (c) 2019-2023 Advanced Micro Devices, Inc. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

#ifndef NCCL_DEVICE_COMMON_H_
#define NCCL_DEVICE_COMMON_H_

#include "collectives.h"
#include "devcomm.h"

#if defined(__gfx908__)
#define COLL_UNROLL 2
#else
#define COLL_UNROLL 4
#endif

#define NCCL_MAX_DEV_ARITY (NCCL_MAX_TREE_ARITY-1)  // Using balanced tree instead of split tree

#define __syncwarp()

#define __synclds() \
  asm volatile("s_waitcnt lgkmcnt(0) \n s_barrier");

#ifdef __GFX9__
#define STORE(DST, SRC) \
  { __threadfence(); __atomic_store_n((DST), (SRC), __ATOMIC_RELAXED); }
#else
#define STORE(DST, SRC) \
  { __atomic_store_n((DST), (SRC), __ATOMIC_SEQ_CST); }
#endif

#ifdef ENABLE_LL128
#define NCCL_FUNC5(func, algo, devredop, type, nullify) \
  MACRO_IF(nullify, nullptr, NCCL_FUNC_NAME(func, algo, LL,     devredop, type)), \
  MACRO_IF(nullify, nullptr, NCCL_FUNC_NAME(func, algo, LL128,  devredop, type)), \
  MACRO_IF(nullify, nullptr, NCCL_FUNC_NAME(func, algo, SIMPLE, devredop, type))
#else
#define NCCL_FUNC5(func, algo, devredop, type, nullify) \
  MACRO_IF(nullify, nullptr, NCCL_FUNC_NAME(func, algo, LL,     devredop, type)), \
  MACRO_IF(nullify, nullptr, NCCL_FUNC_NAME(func, algo, LL,     devredop, type)), \
  MACRO_IF(nullify, nullptr, NCCL_FUNC_NAME(func, algo, SIMPLE, devredop, type))
#endif

#define NCCL_FUNC4(func, devredop, type, nullify) \
  NCCL_FUNC5(func, TREE,    devredop, type, nullify), \
  NCCL_FUNC5(func, RING,    devredop, type, nullify), \
  NCCL_FUNC5(func, COLLNET_DIRECT, devredop, type, nullify), \
  NCCL_FUNC5(func, COLLNET_CHAIN, devredop, type, nullify), \
  NCCL_FUNC5(func, NVLS, devredop, type, nullify), \
  NCCL_FUNC5(func, NVLS_TREE, devredop, type, nullify)

// Must be consistent with ncclDataType_t
#define NCCL_FUNCS3A(func, devredop, nullForFloat) \
  NCCL_FUNC4(func, devredop, int8_t, 0), \
  NCCL_FUNC4(func, devredop, uint8_t, 0), \
  NCCL_FUNC4(func, devredop, int32_t, 0), \
  NCCL_FUNC4(func, devredop, uint32_t, 0), \
  NCCL_FUNC4(func, devredop, int64_t, 0), \
  NCCL_FUNC4(func, devredop, uint64_t, 0), \
  NCCL_FUNC4(func, devredop, half, nullForFloat), \
  NCCL_FUNC4(func, devredop, float, nullForFloat), \
  NCCL_FUNC4(func, devredop, double, nullForFloat), \
  NCCL_FUNC4(func, devredop, rccl_bfloat16, nullForFloat)
#define NCCL_FUNCS3B(func, devredop) \
  NCCL_FUNC4(func, devredop, int8_t, 0), \
  NCCL_FUNC4(func, devredop, int8_t, 0), \
  NCCL_FUNC4(func, devredop, int8_t, 0), \
  NCCL_FUNC4(func, devredop, int8_t, 0), \
  NCCL_FUNC4(func, devredop, int8_t, 0), \
  NCCL_FUNC4(func, devredop, int8_t, 0), \
  NCCL_FUNC4(func, devredop, int8_t, 0), \
  NCCL_FUNC4(func, devredop, int8_t, 0), \
  NCCL_FUNC4(func, devredop, int8_t, 0), \
  NCCL_FUNC4(func, devredop, int8_t, 0)

// Must be consistent with ncclRedOp_t
#define NCCL_FUNCS2A(func) \
  NCCL_FUNCS3A(func, Sum,        /*nullForFloat=*/0), \
  NCCL_FUNCS3A(func, Prod,       /*nullForFloat=*/0), \
  NCCL_FUNCS3A(func, Max,        /*nullForFloat=*/0), \
  NCCL_FUNCS3A(func, Min,        /*nullForFloat=*/0), \
  NCCL_FUNCS3A(func, PreMulSum,  /*nullForFloat=*/0), \
  NCCL_FUNCS3A(func, SumPostDiv, /*nullForFloat=*/1)

#define NCCL_FUNCS2B(func) \
  NCCL_FUNCS3B(func, Sum), \
  NCCL_FUNCS3B(func, Sum), \
  NCCL_FUNCS3B(func, Sum), \
  NCCL_FUNCS3B(func, Sum), \
  NCCL_FUNCS3B(func, Sum), \
  NCCL_FUNCS3B(func, Sum)

// Must be consistent with the ncclFuncSet enum
using ncclKernelFunc_t = void (*)();

static const __device__ constexpr ncclKernelFunc_t ncclFuncs[]{
// Don't try to initialize the host shadow copy of this device-side global
// variable. There is no host pointer to a device-side function, which
// confuses clang. This will be fixed in the next clang release.
#if defined(__HIP_DEVICE_COMPILE__)
#if defined(BUILD_ALLREDUCE_ONLY)
  NCCL_FUNC4(AllReduce, Sum, float, 0),
#else
  NCCL_FUNCS2B(Broadcast),
  NCCL_FUNCS2A(Reduce),
  NCCL_FUNCS2B(AllGather),
  NCCL_FUNCS2A(ReduceScatter),
  NCCL_FUNCS2A(AllReduce),
  NCCL_ONERANK_REDUCE_NAME(PreMulSum, int8_t),
  NCCL_ONERANK_REDUCE_NAME(PreMulSum, uint8_t),
  NCCL_ONERANK_REDUCE_NAME(PreMulSum, int32_t),
  NCCL_ONERANK_REDUCE_NAME(PreMulSum, uint32_t),
  NCCL_ONERANK_REDUCE_NAME(PreMulSum, int64_t),
  NCCL_ONERANK_REDUCE_NAME(PreMulSum, uint64_t),
  NCCL_ONERANK_REDUCE_NAME(PreMulSum, half),
  NCCL_ONERANK_REDUCE_NAME(PreMulSum, float),
  NCCL_ONERANK_REDUCE_NAME(PreMulSum, double),
#if defined(RCCL_BFLOAT16)
  NCCL_ONERANK_REDUCE_NAME(PreMulSum, rccl_bfloat16),
#endif
  NCCL_FUNC_NAME(SendRecv, RING, SIMPLE, Sum, int8_t),
  NCCL_FUNC_NAME(AllToAllPivot, RING, SIMPLE, Sum, int8_t),
#endif
#endif
};

static_assert(FUNC_INDEX_P2P == 5410, "Wrong P2P function index");
static_assert(FUNC_INDEX_ALLTOALL_PIVOT == 5411, "Wrong AllToAllPivot function index");

#if !defined(USE_INDIRECT_FUNCTION_CALL) || defined(__gfx940__) || defined(__gfx941__) || defined(__gfx942__)
template<unsigned short f, unsigned short l, bool u>
struct Caller {
  static __forceinline__ __device__ __host__
  void call(unsigned short funcIndex) noexcept
  {
    constexpr unsigned short m = f + (l - f) / 2;

     return (funcIndex < m) ? Caller<f, m, u>::call(funcIndex) : Caller<m, l, u>::call(funcIndex);
  }
};

template<unsigned short f, bool u>
struct Caller<f, f + 1, u>{
  static __forceinline__ __device__ __host__
  void call(unsigned short funcIndex) noexcept { ncclFuncs[f](); }
};

template<bool USING_LL128>
__forceinline__
__device__
void NCCL_CALL_FUNCTIONS(unsigned short funcIndex) noexcept {
#if defined(BUILD_ALLREDUCE_ONLY)
  if (funcIndex == FUNC_INDEX(ncclFuncAllReduce, ncclSum, ncclFloat32, NCCL_ALGO_RING, NCCL_PROTO_SIMPLE))
    ncclFunction_AllReduce_RING_SIMPLE_Sum_float();
  else if (funcIndex == FUNC_INDEX(ncclFuncAllReduce, ncclSum, ncclFloat32, NCCL_ALGO_RING, NCCL_PROTO_LL))
    ncclFunction_AllReduce_RING_LL_Sum_float();
  else if (USING_LL128 && funcIndex == FUNC_INDEX(ncclFuncAllReduce, ncclSum, ncclFloat32, NCCL_ALGO_RING, NCCL_PROTO_LL128))
    ncclFunction_AllReduce_RING_LL128_Sum_float();
  else if (!USING_LL128 && funcIndex == FUNC_INDEX(ncclFuncAllReduce, ncclSum, ncclFloat32, NCCL_ALGO_RING, NCCL_PROTO_LL128))
    ncclFunction_AllReduce_RING_LL_Sum_float();
  else if (funcIndex == FUNC_INDEX(ncclFuncAllReduce, ncclSum, ncclFloat32, NCCL_ALGO_TREE, NCCL_PROTO_SIMPLE))
    ncclFunction_AllReduce_TREE_SIMPLE_Sum_float();
  else if (funcIndex == FUNC_INDEX(ncclFuncAllReduce, ncclSum, ncclFloat32, NCCL_ALGO_TREE, NCCL_PROTO_LL))
    ncclFunction_AllReduce_TREE_LL_Sum_float();
  else if (USING_LL128 && funcIndex == FUNC_INDEX(ncclFuncAllReduce, ncclSum, ncclFloat32, NCCL_ALGO_TREE, NCCL_PROTO_LL128))
    ncclFunction_AllReduce_TREE_LL128_Sum_float();
  else if (!USING_LL128 && funcIndex == FUNC_INDEX(ncclFuncAllReduce, ncclSum, ncclFloat32, NCCL_ALGO_TREE, NCCL_PROTO_LL128))
    ncclFunction_AllReduce_TREE_LL_Sum_float();
  else if (funcIndex == FUNC_INDEX(ncclFuncAllReduce, ncclSum, ncclFloat32, NCCL_ALGO_COLLNET_DIRECT, NCCL_PROTO_SIMPLE))
    ncclFunction_AllReduce_COLLNET_DIRECT_SIMPLE_Sum_float();
  else if (funcIndex == FUNC_INDEX(ncclFuncAllReduce, ncclSum, ncclFloat32, NCCL_ALGO_COLLNET_DIRECT, NCCL_PROTO_LL))
    ncclFunction_AllReduce_COLLNET_DIRECT_LL_Sum_float();
  else if (USING_LL128 && funcIndex == FUNC_INDEX(ncclFuncAllReduce, ncclSum, ncclFloat32, NCCL_ALGO_COLLNET_DIRECT, NCCL_PROTO_LL128))
    ncclFunction_AllReduce_COLLNET_DIRECT_LL128_Sum_float();
  else if (!USING_LL128 && funcIndex == FUNC_INDEX(ncclFuncAllReduce, ncclSum, ncclFloat32, NCCL_ALGO_COLLNET_DIRECT, NCCL_PROTO_LL128))
    ncclFunction_AllReduce_COLLNET_DIRECT_LL_Sum_float();
  else if (funcIndex == FUNC_INDEX(ncclFuncAllReduce, ncclSum, ncclFloat32, NCCL_ALGO_COLLNET_CHAIN, NCCL_PROTO_SIMPLE))
    ncclFunction_AllReduce_COLLNET_CHAIN_SIMPLE_Sum_float();
  else if (funcIndex == FUNC_INDEX(ncclFuncAllReduce, ncclSum, ncclFloat32, NCCL_ALGO_COLLNET_CHAIN, NCCL_PROTO_LL))
    ncclFunction_AllReduce_COLLNET_CHAIN_LL_Sum_float();
  else if (USING_LL128 && funcIndex == FUNC_INDEX(ncclFuncAllReduce, ncclSum, ncclFloat32, NCCL_ALGO_COLLNET_CHAIN, NCCL_PROTO_LL128))
    ncclFunction_AllReduce_COLLNET_CHAIN_LL128_Sum_float();
  else if (!USING_LL128 && funcIndex == FUNC_INDEX(ncclFuncAllReduce, ncclSum, ncclFloat32, NCCL_ALGO_COLLNET_CHAIN, NCCL_PROTO_LL128))
    ncclFunction_AllReduce_COLLNET_CHAIN_LL_Sum_float();
  else
    assert("Unsupported function index");
#else
  if (funcIndex < 1080) {
    if (funcIndex % 18 == 0) ncclFunction_Broadcast_TREE_LL_Sum_int8_t();
    else if (USING_LL128 && funcIndex % 18 == 1) ncclFunction_Broadcast_TREE_LL128_Sum_int8_t();
    else if (!USING_LL128 && funcIndex % 18 == 1) ncclFunction_Broadcast_TREE_LL_Sum_int8_t();
    else if (funcIndex % 18 == 2) ncclFunction_Broadcast_TREE_SIMPLE_Sum_int8_t();
    else if (funcIndex % 18 == 3) ncclFunction_Broadcast_RING_LL_Sum_int8_t();
    else if (USING_LL128 && funcIndex % 18 == 4) ncclFunction_Broadcast_RING_LL128_Sum_int8_t();
    else if (!USING_LL128 && funcIndex % 18 == 4) ncclFunction_Broadcast_RING_LL_Sum_int8_t();
    else if (funcIndex % 18 == 5) ncclFunction_Broadcast_RING_SIMPLE_Sum_int8_t();
    else if (funcIndex % 18 == 6) ncclFunction_Broadcast_COLLNET_DIRECT_LL_Sum_int8_t();
    else if (USING_LL128 && funcIndex % 18 == 7) ncclFunction_Broadcast_COLLNET_DIRECT_LL128_Sum_int8_t();
    else if (!USING_LL128 && funcIndex % 18 == 7) ncclFunction_Broadcast_COLLNET_DIRECT_LL_Sum_int8_t();
    else if (funcIndex % 18 == 8) ncclFunction_Broadcast_COLLNET_DIRECT_SIMPLE_Sum_int8_t();
    else if (funcIndex % 18 == 9) ncclFunction_Broadcast_COLLNET_CHAIN_LL_Sum_int8_t();
    else if (USING_LL128 && funcIndex % 18 == 10) ncclFunction_Broadcast_COLLNET_CHAIN_LL128_Sum_int8_t();
    else if (!USING_LL128 && funcIndex % 18 == 10) ncclFunction_Broadcast_COLLNET_CHAIN_LL_Sum_int8_t();
    else ncclFunction_Broadcast_COLLNET_CHAIN_SIMPLE_Sum_int8_t();
  }
  else if (funcIndex < 2160) Caller<1080, 2160, USING_LL128>::call(funcIndex);
  else if (funcIndex < 3240) {
    if (funcIndex % 18 == 0) ncclFunction_AllGather_TREE_LL_Sum_int8_t();
    else if (USING_LL128 && funcIndex % 18 == 1) ncclFunction_AllGather_TREE_LL128_Sum_int8_t();
    else if (!USING_LL128 && funcIndex % 18 == 1) ncclFunction_AllGather_TREE_LL_Sum_int8_t();
    else if (funcIndex % 18 == 2) ncclFunction_AllGather_TREE_SIMPLE_Sum_int8_t();
    else if (funcIndex % 18 == 3) ncclFunction_AllGather_RING_LL_Sum_int8_t();
    else if (USING_LL128 && funcIndex % 18 == 4) ncclFunction_AllGather_RING_LL128_Sum_int8_t();
    else if (!USING_LL128 && funcIndex % 18 == 4) ncclFunction_AllGather_RING_LL_Sum_int8_t();
    else if (funcIndex % 18 == 5) ncclFunction_AllGather_RING_SIMPLE_Sum_int8_t();
    else if (funcIndex % 18 == 6) ncclFunction_AllGather_COLLNET_DIRECT_LL_Sum_int8_t();
    else if (USING_LL128 && funcIndex % 18 == 7) ncclFunction_AllGather_COLLNET_DIRECT_LL128_Sum_int8_t();
    else if (!USING_LL128 && funcIndex % 18 == 7) ncclFunction_AllGather_COLLNET_DIRECT_LL_Sum_int8_t();
    else if (funcIndex % 18 == 8) ncclFunction_AllGather_COLLNET_DIRECT_SIMPLE_Sum_int8_t();
    else if (funcIndex % 18 == 9) ncclFunction_AllGather_COLLNET_CHAIN_LL_Sum_int8_t();
    else if (USING_LL128 && funcIndex % 18 == 10) ncclFunction_AllGather_COLLNET_CHAIN_LL128_Sum_int8_t();
    else if (!USING_LL128 && funcIndex % 18 == 10) ncclFunction_AllGather_COLLNET_CHAIN_LL_Sum_int8_t();
    else ncclFunction_AllGather_COLLNET_CHAIN_SIMPLE_Sum_int8_t();
  }
  else if (funcIndex < 5400) Caller<3240, 5400, USING_LL128>::call(funcIndex);
  else {
    switch (funcIndex - 5400) {
      case 0:
        ncclFunction_OneRankReduce_PreMulSum_int8_t();
        break;
      case 1:
        ncclFunction_OneRankReduce_PreMulSum_uint8_t();
        break;
      case 2:
        ncclFunction_OneRankReduce_PreMulSum_int32_t();
        break;
      case 3:
        ncclFunction_OneRankReduce_PreMulSum_uint32_t();
        break;
      case 4:
        ncclFunction_OneRankReduce_PreMulSum_int64_t();
        break;
      case 5:
        ncclFunction_OneRankReduce_PreMulSum_uint64_t();
        break;
      case 6:
        ncclFunction_OneRankReduce_PreMulSum_half();
        break;
      case 7:
        ncclFunction_OneRankReduce_PreMulSum_float();
        break;
      case 8:
        ncclFunction_OneRankReduce_PreMulSum_double();
        break;
      case 9:
        ncclFunction_OneRankReduce_PreMulSum_rccl_bfloat16();
        break;
      case 10:
        ncclFunction_SendRecv_RING_SIMPLE_Sum_int8_t();
        break;
      case 11:
        ncclFunction_AllToAllPivot_RING_SIMPLE_Sum_int8_t();
      default:
        break;
    }
  }
#endif
}
#endif

template <ncclFunc_t FUNCTION, int ALGO, int PROTO, class REDOP, typename T, int UNROLL>
class ncclFunction {
  public:
#if defined(USE_INDIRECT_FUNCTION_CALL) && !defined(__gfx940__) && !defined(__gfx941__) && !defined(__gfx942__)
  __device__ __attribute__((noinline)) void run(struct ncclWorkElem* args) {}
#else
  __device__ void run(struct ncclWorkElem* args) {}
#endif
};

#if defined(__gfx1100__) || defined(__gfx1101__) || defined(__gfx1102__)
#define __trace_hwreg()
#else
#define __trace_hwreg() \
  asm volatile ("s_getreg_b32 %0, hwreg(HW_REG_HW_ID)" : "=s" (collTrace->data_0));
#endif
#ifdef ENABLE_COLLTRACE
  #define INC_COLL_TRACE \
    uint32_t pos = atomicAdd(&ncclShmem.collTraceTail->tail, 1)%COLLTRACE_NUM_ITEMS; \
    struct ncclCollTrace* collTrace = ncclShmem.collTrace+pos; \
    collTrace->timeStamp = wall_clock64(); \
    collTrace->bid = blockIdx.x;
    // TODO: switch to atomicInc after llvm crash is fixed
    // uint32_t pos = atomicInc(&ncclShmem.collTraceTail->tail, COLLTRACE_NUM_ITEMS)

  #define traceKernelLaunch(launch_type) { \
    INC_COLL_TRACE \
    collTrace->funcIndex = ncclShmem.work.header.funcIndex; \
    __trace_hwreg()\
    if (ncclShmem.work.header.type == ncclWorkTypeP2p) { \
      struct ncclWorkElemP2p *p2pElems = ncclShmem.work.p2pElems; \
      collTrace->p2p[0].connIndex = 0; \
      collTrace->p2pOpCount[0] = p2pElems[0].opCount; \
      collTrace->p2p[0].ngroups = p2pElems[0].ngroups; \
      collTrace->p2p[0].nWarps = p2pElems[0].nWarps; \
      collTrace->p2p[0].warpStart = p2pElems[0].warpStart; \
      collTrace->p2p[0].peer = p2pElems[0].p2pType == ncclWorkP2pTypeRecv ? (uint16_t)(p2pElems[0].peer) : -1; \
      collTrace->p2p[1].connIndex = 0; \
      collTrace->p2pOpCount[1] = p2pElems[1].opCount; \
      collTrace->p2p[1].ngroups = p2pElems[1].ngroups; \
      collTrace->p2p[1].nWarps = p2pElems[1].nWarps; \
      collTrace->p2p[1].warpStart = p2pElems[1].warpStart; \
      collTrace->p2p[1].peer = p2pElems[1].p2pType == ncclWorkP2pTypeSend ? (uint16_t)(p2pElems[1].peer) : -1; \
      collTrace->type = (launch_type) | ncclCollTraceP2pElemType; \
    } else if (ncclShmem.work.header.type == ncclWorkTypeColl) { \
      struct ncclWorkElem *elems = ncclShmem.work.elems; \
      collTrace->opCount = elems[0].opCount; \
      collTrace->coll.nWarps = elems[0].nWarps; \
      collTrace->coll.bid = elems[0].bid; \
      collTrace->coll.nChannels = elems[0].nChannels; \
      collTrace->type = (launch_type) | ncclCollTraceCollElemType; \
    } \
  }
  #define traceKernelEnd(end_type)  { \
    INC_COLL_TRACE \
    if (ncclShmem.work.header.type == ncclWorkTypeP2p) { \
      struct ncclWorkElemP2p *p2pElems = ncclShmem.work.p2pElems; \
      collTrace->p2pOpCount[0] = p2pElems[0].opCount; \
      collTrace->p2pOpCount[1] = p2pElems[1].opCount; \
    } else if (ncclShmem.work.header.type == ncclWorkTypeColl) { \
      struct ncclWorkElem *elems = ncclShmem.work.elems; \
      collTrace->opCount = elems[0].opCount; \
    } \
    collTrace->type = end_type; \
  }
  #define traceData(data2, data4, data8_0, data8_1) { \
    INC_COLL_TRACE \
    collTrace->funcIndex = data2; \
    collTrace->data_0 = data4; \
    collTrace->opCount = data8_0; \
    collTrace->data_1 = data8_1; \
    collTrace->type = ncclCollTraceDataType; \
  }
#else
#define traceKernelLaunch(launch_type)
#define traceKernelEnd(end_type)
#define traceData(data2, data4, data8_0, data8_1)
#endif

struct ncclShmemGroup {
  ncclConnInfo *recvConns[NCCL_MAX_NVLS_ARITY];
  ncclConnInfo *sendConns[NCCL_MAX_NVLS_ARITY];
  void* srcs[NCCL_MAX_NVLS_ARITY+1];
  void* dsts[NCCL_MAX_NVLS_ARITY+1];
  uint64_t barrier;
  uint64_t barrier_next[NCCL_MAX_GROUPS];
};

#define LDS_NUM_EVENTS 64

struct ncclShmemData {
  struct ncclShmemGroup groups[NCCL_MAX_GROUPS];
  uint64_t redOpArgs[NCCL_MAX_NVLS_ARITY+1];
  int channelId;
  int aborted;
  alignas(16) struct ncclDevComm comm;
  alignas(16) struct ncclDevChannel channel;
  alignas(16) struct ncclWork work;
#ifdef ENABLE_COLLTRACE
  struct ncclCollTrace* collTrace;
  union ncclCollTraceTail* collTraceTail;
#endif
#ifdef ENABLE_PROFILING
  struct ncclProf prof;
#endif
#if defined(ENABLE_NPKIT)
  NpKitEvent event_buffer[LDS_NUM_EVENTS];
  uint64_t event_buffer_head;
#endif
};
static_assert(offsetof(struct ncclShmemData, work)%16 == 0, "ncclShmem.work needs to be 16B aligned");

extern __shared__ ncclShmemData ncclShmem;
#if __CUDA_ARCH__ >= 700
  extern __shared__ ulong2 ncclShmemPerWarp[/*ncclShmemDynamicSize()/sizeof(ulong2)*/];
#else
  extern __shared__ ulong2 ncclShmemPerWarp[ncclShmemScratchWarpSize()*(NCCL_MAX_NTHREADS/WARP_SIZE)/sizeof(ulong2)];
#endif

__device__ inline void* ncclScratchForWarp(int warp) {
  return (char*)ncclShmemPerWarp + warp*ncclShmemScratchWarpSize();
}

#ifdef ENABLE_PROFILING
#define __insert_timestamp(line_num) do { \
      if (ncclShmem.prof.count < PROFILE_NUM_ITEMS) { \
        ncclShmem.prof.elem[ncclShmem.prof.count].line = line_num; \
        ncclShmem.prof.elem[ncclShmem.prof.count].timeStamp = wall_clock64(); \
        ncclShmem.prof.count++; \
      } \
    } while(0);
#else
#define __insert_timestamp(line_num)
#endif

// Copy 16-byte aligned data. You must call with at least `(bytes+15)/16` threads.
inline __device__ void copyToShmem16(int tid, void* dst, void const* src, int bytes) {
  int offset = 16*tid;
  if (offset < bytes) {
    ulong2 *src2, *dst2;
    src2 = (ulong2*)((char const*)src + offset);
    dst2 = (ulong2*)((char*)dst + offset);
    dst2->x = src2->x;
    dst2->y = src2->y;
  }
}

template<ncclFunc_t Fn, typename T, typename RedOp, int Algo, int Proto>
struct RunWorkElement {
  __device__ void run(ncclWorkElem*) {
    // Put NOT IMPLEMENTED behavior here.
  }
};

template<ncclFunc_t Fn, typename T, typename RedOp, int Algo, int Proto>
struct RunWork {
  // This __forceinline__ is necessary. The compiler was inserting a function call
  // here from the LL ncclKernel.
  __device__ __forceinline__ void run(ncclWork *w) {
    int wid = threadIdx.x / WARP_SIZE;
    ncclWorkElem* we = w->header.type == ncclWorkTypeRegColl ? &w->regElems[0].elem : &w->elems[0];
    int stride = w->header.type == ncclWorkTypeRegColl ? sizeof(ncclWorkElemReg) : sizeof(ncclWorkElem);
    #pragma unroll 1
    while ((char*)we + stride <= (char*)(w+1) && we->isUsed) {
      if (wid < we->nWarps) {
        RunWorkElement<Fn, T, RedOp, Algo, Proto>().run(we);
      }
      we = (ncclWorkElem*)((char*)we + stride);
    }
  }
};

static __forceinline__ __device__ void ncclRedopPtrDeref(struct ncclWorkElem* we) {
  if (we->isUsed && we->redOpArgIsPtr) {
    /* redOpArg is a pointer to the scalar value, so we'll dereference it
     * here so that redOpArg holds the bits of the scalar going forward.
     * The tricky thing is we don't know its type T since that's encoded in
     * the funcIndex. Because it would be difficult to get sizeof(T) from
     * funcIndex, we'll cheat and just dereference the largest possible size
     * given the alignment of the pointer. We might be reading in more bytes
     * than we need but that's harmless.
     */
    if (we->redOpArg%2 != 0)
      we->redOpArg = *reinterpret_cast<uint8_t*>(we->redOpArg);
    else if (we->redOpArg%4 != 0)
      we->redOpArg = *reinterpret_cast<uint16_t*>(we->redOpArg);
    else if (we->redOpArg%8 != 0)
      we->redOpArg = *reinterpret_cast<uint32_t*>(we->redOpArg);
    else
      we->redOpArg = *reinterpret_cast<uint64_t*>(we->redOpArg);
  }
}

template<ncclFunc_t Fn, typename T, typename RedOp, int Algo, int Proto, int FnIndex, bool COLLTRACE>
__forceinline__ __device__ void ncclKernel(
    struct ncclDevComm* comm, uint64_t channelMask, struct ncclWork* workHead
  )  {
  const int tid = threadIdx.x;
  int x = tid;
  switch (tid/WARP_SIZE) {
  case 0:
    if (channelMask & (1ull<<x)) {
      int y = __popcll(channelMask & ((1ull<<x)-1));
      if (blockIdx.x == y) ncclShmem.channelId = x;
    }
    if (32 < MAXCHANNELS) {
      x = 32 + tid;
      if (channelMask & (1ull<<x)) {
        int y = __popcll(channelMask & ((1ull<<x)-1));
        if (blockIdx.x == y) ncclShmem.channelId = x;
      }
    }
    break;
  case 1:
    if (tid < WARP_SIZE + NCCL_MAX_GROUPS)
      ncclShmem.groups[tid-WARP_SIZE].barrier = 0;
    break;
  case 2:
    if (tid < 2*WARP_SIZE + NCCL_MAX_GROUPS*NCCL_MAX_GROUPS)
      ncclShmem.groups[(tid-2*WARP_SIZE)/NCCL_MAX_GROUPS].barrier_next[(tid-2*WARP_SIZE)%NCCL_MAX_GROUPS] = 0;
    break;
  case 3:
    /* set abort flag to 0 */
    if (tid == 3*WARP_SIZE) ncclShmem.aborted = 0;
    break;
  default:
    break;
  }
  __synclds(); // publish ncclShmem.channelId
  // To map blockId to channelId, we need the n'th set bit of channelMask which
  // is the inverse of counting the number of set bits among the the first n.
  int channelId = ncclShmem.channelId;

  if (true) {
    void *dst, *src;
    int bytes;
    // Use first 3 warps to load comm, channel, and work into shmem
    switch (tid/WARP_SIZE) {
    case 0:
      dst = &ncclShmem.comm;
      src = comm;
      bytes = sizeof(ncclDevComm);
      static_assert(sizeof(ncclDevComm) <= 16*WARP_SIZE, "ncclDevComm cannot be loaded by a single warp in one insn.");
      break;
    case 1:
      // Get address of channel without incurring indirect load from ncclDevComm::channels
      dst = &ncclShmem.channel;
      src = &((ncclDevCommAndChannels*)comm)->channels[channelId];
      bytes = sizeof(ncclDevChannel);
      static_assert(sizeof(ncclDevChannel) <= 16*WARP_SIZE, "ncclDevChannel cannot be loaded by a single warp in one insn.");
      break;
    case 2:
      dst = &ncclShmem.work;
      src = workHead + blockIdx.x;
      bytes = sizeof(ncclWork);
      static_assert(sizeof(ncclWork) <= 16*WARP_SIZE, "ncclWork cannot be loaded by a single warp in one insn.");
      break;
    default:
      bytes = 0;
      break;
    }
    copyToShmem16(tid%WARP_SIZE, dst, src, bytes);
  }
#ifdef ENABLE_COLLTRACE
  if (tid == 0) {
    ncclShmem.collTrace = comm->collTrace + COLLTRACE_NUM_ITEMS*ncclShmem.channelId;
    ncclShmem.collTraceTail = comm->collTraceTail + ncclShmem.channelId;
  }
#endif
  __synclds(); // publish shmem
#ifdef ENABLE_PROFILING
  if (tid == 0) {
    ncclShmem.prof.count = 0;
    ncclShmem.prof.seq = ncclShmem.comm.devProf[blockIdx.x].seq;
  }
#endif
  if (tid == 0) __insert_timestamp(__LINE__);
  if (COLLTRACE && tid == 0) traceKernelLaunch(ncclCollTraceKernelLaunchType);

  while (true) {
    // Notify host that all fifo reads are complete.
    if (tid == 0 && ncclShmem.work.header.isLast && ncclShmem.work.header.inFifo) {
      *ncclShmem.channel.workFifoDone = ncclShmem.work.header.doneAcks;
    }

    __syncwarp();
    if (ncclShmem.work.header.type == ncclWorkTypeColl) {
      if (tid < NCCL_MAX_WORK_ELEMENTS) ncclRedopPtrDeref(&ncclShmem.work.elems[tid]);
    } else if (ncclShmem.work.header.type == ncclWorkTypeRegColl) {
      if (tid < NCCL_MAX_WORK_ELEMENTS_REG) ncclRedopPtrDeref(&ncclShmem.work.regElems[tid].elem);
    }
    __synclds();

    if (tid == 0) __insert_timestamp(__LINE__);
    if (ncclShmem.work.header.funcIndex == FnIndex) {
      RunWork<Fn, T, RedOp, Algo, Proto>().run(&ncclShmem.work);
    } else {
#if defined(USE_INDIRECT_FUNCTION_CALL) && !defined(__gfx940__) && !defined(__gfx941__) && !defined(__gfx942__)
      ncclFuncs[ncclShmem.work.header.funcIndex]();
#else
#ifdef ENABLE_LL128
      NCCL_CALL_FUNCTIONS<1>(ncclShmem.work.header.funcIndex);
#else
      NCCL_CALL_FUNCTIONS<0>(ncclShmem.work.header.funcIndex);
#endif
#endif
    }

    int workIxNext = ncclShmem.work.header.workNext;
    __synclds();
    if (ncclShmem.work.header.isLast) break;

    copyToShmem16(tid, &ncclShmem.work, workHead + workIxNext, sizeof(ncclWork));

    { // Check whether the last operation was aborted and make sure all threads exit
      int aborted = tid == 0 ? *comm->abortFlag : 0;
      if (__any(aborted)) { // publish ncclShmem.work
        traceKernelEnd(ncclCollTraceAbortType);
        break;
      }
    }
    if (COLLTRACE && tid == 0) traceKernelLaunch(ncclCollTraceCollLaunchType);
  }
  if (COLLTRACE && tid == 0) traceKernelEnd(ncclCollTraceKernelEndType);

#ifdef ENABLE_PROFILING
  if (ncclShmem.comm.devProf->seq < PROFILE_NUM_LAUNCHES) {
    __synclds();
    copyToShmem16(tid, ncclShmem.comm.devProf+MAXCHANNELS*ncclShmem.prof.seq+blockIdx.x, &ncclShmem.prof, sizeof(struct ncclProf));
    if (tid == 0) ncclShmem.comm.devProf[blockIdx.x].seq++;
  }
#endif
}

#ifdef ENABLE_COLLTRACE
#define IMPL_COLL_KERN(func, algo, proto, devredop, type, fIndex) \
__launch_bounds__(NCCL_MAX_NTHREADS, 1) \
__global__ void NCCL_KERN_NAME(func, algo, proto, devredop, type)(struct ncclDevComm* comm, uint64_t channelMask, struct ncclWork* workHead) { \
  ncclKernel<ncclFunc##func, type, Func##devredop<type>, NCCL_ALGO_##algo, NCCL_PROTO_##proto, fIndex, false>(comm, channelMask, workHead); \
} \
 \
__launch_bounds__(NCCL_MAX_NTHREADS, 1) \
__global__ void NCCL_KERN_NAME_DEBUG(func, algo, proto, devredop, type)(struct ncclDevComm* comm, uint64_t channelMask, struct ncclWork* workHead) { \
  ncclKernel<ncclFunc##func, type, Func##devredop<type>, NCCL_ALGO_##algo, NCCL_PROTO_##proto, fIndex, true>(comm, channelMask, workHead); \
}
#else
#define IMPL_COLL_KERN(func, algo, proto, devredop, type, fIndex) \
__launch_bounds__(NCCL_MAX_NTHREADS, 1) \
__global__ void NCCL_KERN_NAME(func, algo, proto, devredop, type)(struct ncclDevComm* comm, uint64_t channelMask, struct ncclWork* workHead) { \
  ncclKernel<ncclFunc##func, type, Func##devredop<type>, NCCL_ALGO_##algo, NCCL_PROTO_##proto, fIndex, false>(comm, channelMask, workHead); \
}
#endif

// Examples :     AllReduce, RING, LL,    Sum,   uint8
/* Functions for aggregation case */

#if defined(USE_INDIRECT_FUNCTION_CALL) && !defined(__gfx940__) && !defined(__gfx941__) && !defined(__gfx942__)
#define IMPL_COLL_FUNC(func, algo, proto, devredop, type) \
__device__  void NCCL_FUNC_NAME(func, algo, proto, devredop, type)() { \
  RunWork<ncclFunc##func, type, Func##devredop<type>, NCCL_ALGO_##algo, NCCL_PROTO_##proto>().run(&ncclShmem.work); \
}
#else
#define IMPL_COLL_FUNC(func, algo, proto, devredop, type) \
__device__  __attribute__((noinline)) void NCCL_FUNC_NAME(func, algo, proto, devredop, type)() { \
  RunWork<ncclFunc##func, type, Func##devredop<type>, NCCL_ALGO_##algo, NCCL_PROTO_##proto>().run(&ncclShmem.work); \
}
#endif

// Only generate inline kernels for LL
#ifdef ENABLE_LL128
#define IMPL_COLL4(func, algo, devredop, type) \
  IMPL_COLL_FUNC(func, algo, LL,     devredop, type) \
  IMPL_COLL_FUNC(func, algo, LL128,  devredop, type) \
  IMPL_COLL_FUNC(func, algo, SIMPLE, devredop, type)
#else
#define IMPL_COLL4(func, algo, devredop, type) \
  IMPL_COLL_FUNC(func, algo, LL,     devredop, type) \
  IMPL_COLL_FUNC(func, algo, SIMPLE, devredop, type)
#endif

#define IMPL_COLL3(func, devredop, type) \
  IMPL_COLL4(func, TREE,    devredop, type) \
  IMPL_COLL4(func, RING,    devredop, type) \
  IMPL_COLL4(func, COLLNET_DIRECT, devredop, type) \
  IMPL_COLL4(func, COLLNET_CHAIN, devredop, type) \
  IMPL_COLL4(func, NVLS, devredop, type) \
  IMPL_COLL4(func, NVLS_TREE, devredop, type)

#define IMPL_COLL2(func, devredop) \
  IMPL_COLL3(func, devredop, int8_t) \
  IMPL_COLL3(func, devredop, uint8_t) \
  IMPL_COLL3(func, devredop, int32_t) \
  IMPL_COLL3(func, devredop, uint32_t) \
  IMPL_COLL3(func, devredop, int64_t) \
  IMPL_COLL3(func, devredop, uint64_t) \
  IMPL_COLL3(func, devredop, half) \
  IMPL_COLL3(func, devredop, float) \
  IMPL_COLL3(func, devredop, double) \
  IMPL_COLL3(func, devredop, rccl_bfloat16)

#define IMPL_COLL2A(func, devredop) \
  IMPL_COLL3(func, devredop, int8_t) \
  IMPL_COLL3(func, devredop, uint8_t) \
  IMPL_COLL3(func, devredop, int32_t) \
  IMPL_COLL3(func, devredop, uint32_t) \
  IMPL_COLL3(func, devredop, int64_t) \
  IMPL_COLL3(func, devredop, uint64_t)

// Reduction define all functions
#define IMPL_COLL_R(func) \
  IMPL_COLL2(func, Sum) \
  IMPL_COLL2(func, Prod) \
  IMPL_COLL2(func, Min) \
  IMPL_COLL2(func, Max) \
  IMPL_COLL2(func, PreMulSum) \
  IMPL_COLL2A(func, SumPostDiv)

// Copy primitives only define one function for copy
#define IMPL_COLL_C(func) IMPL_COLL3(func, Sum, int8_t);

// Point-to-point primitives only have one function/kernel.
#define IMPL_COLL_P(func) \
  IMPL_COLL_FUNC(func, RING, SIMPLE, Sum, int8_t); \
  IMPL_COLL_KERN(func, RING, SIMPLE, Sum, int8_t, FUNC_INDEX_P2P);

// AllToAll Pivot primitive only has one function.
#define IMPL_COLL_F(func) \
  IMPL_COLL_FUNC(func, RING, SIMPLE, Sum, int8_t);

#define NCCL_NVLS_ENABLED (__CUDA_ARCH__ >= 900 && NCCL_NVLS_SUPPORTS(NCCL_TYPE, NCCL_OP))

#endif
