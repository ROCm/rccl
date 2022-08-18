/*************************************************************************
 * Copyright (c) 2017-2022, NVIDIA CORPORATION. All rights reserved.
 * Modifications Copyright (c) 2019-2022 Advanced Micro Devices, Inc. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

#ifndef NCCL_DEVICE_COMMON_H_
#define NCCL_DEVICE_COMMON_H_

#include "collectives.h"
#include "devcomm.h"
#include "op128.h"

#define COLL_UNROLL 2
#define NCCL_MAX_DEV_ARITY (NCCL_MAX_TREE_ARITY-1)  // Using balanced tree instead of split tree

#define __syncwarp()

#define NCCL_FUNC5(func, algo, devredop, type, nullify) \
  MACRO_IF(nullify, nullptr, NCCL_FUNC_NAME(func, algo, LL,     devredop, type)), \
  MACRO_IF(nullify, nullptr, NCCL_FUNC_NAME(func, algo, LL,  devredop, type)), \
  MACRO_IF(nullify, nullptr, NCCL_FUNC_NAME(func, algo, SIMPLE, devredop, type))

#define NCCL_FUNC4(func, devredop, type, nullify) \
  NCCL_FUNC5(func, TREE,    devredop, type, nullify), \
  NCCL_FUNC5(func, RING,    devredop, type, nullify), \
  NCCL_FUNC5(func, COLLNET, devredop, type, nullify)

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

template<unsigned short f, unsigned short l>
struct Caller {
  static __device__ __host__
  void call(unsigned short funcIndex) noexcept
  {
    constexpr unsigned short m = f + (l - f) / 2;

     return (funcIndex < m) ? Caller<f, m>::call(funcIndex) : Caller<m, l>::call(funcIndex);
  }
};

template<unsigned short f>
struct Caller<f, f + 1>{
  static __device__ __host__
  void call(unsigned short funcIndex) noexcept { ncclFuncs[f](); }
};

static_assert(FUNC_INDEX_P2P == 2710, "Wrong P2P function index");
static_assert(FUNC_INDEX_ALLTOALL_PIVOT == 2711, "Wrong AllToAllPivot function index");

inline
__device__
void NCCL_CALL_FUNCTIONS(unsigned short funcIndex) noexcept {
#if defined(BUILD_ALLREDUCE_ONLY)
  if (funcIndex == FUNC_INDEX(ncclFuncAllReduce, ncclSum, ncclFloat32, NCCL_ALGO_RING, NCCL_PROTO_SIMPLE))
    ncclFunction_AllReduce_RING_SIMPLE_Sum_float();
  else if (funcIndex == FUNC_INDEX(ncclFuncAllReduce, ncclSum, ncclFloat32, NCCL_ALGO_RING, NCCL_PROTO_LL))
    ncclFunction_AllReduce_RING_LL_Sum_float();
  else if (funcIndex == FUNC_INDEX(ncclFuncAllReduce, ncclSum, ncclFloat32, NCCL_ALGO_RING, NCCL_PROTO_LL128))
    ncclFunction_AllReduce_RING_LL_Sum_float();
  else if (funcIndex == FUNC_INDEX(ncclFuncAllReduce, ncclSum, ncclFloat32, NCCL_ALGO_TREE, NCCL_PROTO_SIMPLE))
    ncclFunction_AllReduce_TREE_SIMPLE_Sum_float();
  else if (funcIndex == FUNC_INDEX(ncclFuncAllReduce, ncclSum, ncclFloat32, NCCL_ALGO_TREE, NCCL_PROTO_LL))
    ncclFunction_AllReduce_TREE_LL_Sum_float();
  else if (funcIndex == FUNC_INDEX(ncclFuncAllReduce, ncclSum, ncclFloat32, NCCL_ALGO_COLLNET, NCCL_PROTO_SIMPLE))
    ncclFunction_AllReduce_COLLNET_SIMPLE_Sum_float();
  else if (funcIndex == FUNC_INDEX(ncclFuncAllReduce, ncclSum, ncclFloat32, NCCL_ALGO_COLLNET, NCCL_PROTO_LL))
    ncclFunction_AllReduce_COLLNET_LL_Sum_float();
  else
    assert("Unsupported function index");
#else
  if (funcIndex < 540) {
    if (funcIndex % 9 == 0) ncclFunction_Broadcast_TREE_LL_Sum_int8_t();
    else if (funcIndex % 9 == 1) ncclFunction_Broadcast_TREE_LL_Sum_int8_t();
    else if (funcIndex % 9 == 2) ncclFunction_Broadcast_TREE_SIMPLE_Sum_int8_t();
    else if (funcIndex % 9 == 3) ncclFunction_Broadcast_RING_LL_Sum_int8_t();
    else if (funcIndex % 9 == 4) ncclFunction_Broadcast_RING_LL_Sum_int8_t();
    else if (funcIndex % 9 == 5) ncclFunction_Broadcast_RING_SIMPLE_Sum_int8_t();
    else if (funcIndex % 9 == 6) ncclFunction_Broadcast_COLLNET_LL_Sum_int8_t();
    else if (funcIndex % 9 == 7) ncclFunction_Broadcast_COLLNET_LL_Sum_int8_t();
    else ncclFunction_Broadcast_COLLNET_SIMPLE_Sum_int8_t();
  }
  else if (funcIndex < 1080) Caller<540, 1080>::call(funcIndex);
  else if (funcIndex < 1620) {
    if (funcIndex % 9 == 0) ncclFunction_AllGather_TREE_LL_Sum_int8_t();
    else if (funcIndex % 9 == 1) ncclFunction_AllGather_TREE_LL_Sum_int8_t();
    else if (funcIndex % 9 == 2) ncclFunction_AllGather_TREE_SIMPLE_Sum_int8_t();
    else if (funcIndex % 9 == 3) ncclFunction_AllGather_RING_LL_Sum_int8_t();
    else if (funcIndex % 9 == 4) ncclFunction_AllGather_RING_LL_Sum_int8_t();
    else if (funcIndex % 9 == 5) ncclFunction_AllGather_RING_SIMPLE_Sum_int8_t();
    else if (funcIndex % 9 == 6) ncclFunction_AllGather_COLLNET_LL_Sum_int8_t();
    else if (funcIndex % 9 == 7) ncclFunction_AllGather_COLLNET_LL_Sum_int8_t();
    else ncclFunction_AllGather_COLLNET_SIMPLE_Sum_int8_t();
  }
  else if (funcIndex < 2700) Caller<1620, 2700>::call(funcIndex);
  else {
    switch (funcIndex - 2700) {
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

template <ncclFunc_t FUNCTION, int ALGO, int PROTO, class REDOP, typename T, int UNROLL>
class ncclFunction {
  public:
  __device__ __attribute__((noinline)) void run(struct ncclWorkElem* args) {}
};

#ifdef ENABLE_COLLTRACE
#define traceColl(elem,launch_type) \
    uint32_t pos = __atomic_fetch_add(ncclShmem->comm.collTraceTail, 1, __ATOMIC_SEQ_CST)%COLLTRACE_NUM_ITEMS; \
    ncclShmem->comm.collTrace[pos].timeStamp = __builtin_amdgcn_s_memrealtime(); \
    ncclShmem->comm.collTrace[pos].bid = blockIdx.x; \
    ncclShmem->comm.collTrace[pos].funcIndex = ncclShmem->work.header.funcIndex; \
    asm volatile ("s_getreg_b32 %0, hwreg(HW_REG_HW_ID)" : "=s" (ncclShmem->comm.collTrace[pos].data_0)); \
    if (elem.header.type == ncclWorkTypeP2p) { \
      struct ncclWorkElemP2p *p2pElems = (struct ncclWorkElemP2p *)&elem; \
      ncclShmem->comm.collTrace[pos].p2p[0].connIndex = p2pElems[0].connIndex; \
	    ncclShmem->comm.collTrace[pos].p2pOpCount[0] = p2pElems[0].opCount; \
      ncclShmem->comm.collTrace[pos].p2p[0].ngroups = p2pElems[0].ngroups; \
      ncclShmem->comm.collTrace[pos].p2p[0].nWarps = p2pElems[0].nWarps; \
      ncclShmem->comm.collTrace[pos].p2p[0].warpStart = p2pElems[0].warpStart; \
      ncclShmem->comm.collTrace[pos].p2p[0].peer = (uint16_t)(p2pElems[0].peer); \
	    ncclShmem->comm.collTrace[pos].p2p[1].connIndex = p2pElems[1].connIndex; \
      ncclShmem->comm.collTrace[pos].p2pOpCount[1] = p2pElems[1].opCount; \
      ncclShmem->comm.collTrace[pos].p2p[1].ngroups = p2pElems[1].ngroups; \
      ncclShmem->comm.collTrace[pos].p2p[1].nWarps = p2pElems[1].nWarps; \
      ncclShmem->comm.collTrace[pos].p2p[1].warpStart = p2pElems[1].warpStart; \
      ncclShmem->comm.collTrace[pos].p2p[1].peer = (uint16_t)(p2pElems[1].peer); \
      ncclShmem->comm.collTrace[pos].type = (ncclCollTraceP2pElemType|launch_type); \
    } else { \
      ncclShmem->comm.collTrace[pos].opCount = elem.opCount; \
      ncclShmem->comm.collTrace[pos].coll.nWarps = elem.header.nWarps; \
      ncclShmem->comm.collTrace[pos].coll.bid = elem.bid; \
      ncclShmem->comm.collTrace[pos].coll.nChannels = elem.nChannels; \
      ncclShmem->comm.collTrace[pos].type = (ncclCollTraceCollElemType|launch_type); \
    }

#define traceKernelLaunch(elem,firstLaunch)  { \
    traceColl(elem,(firstLaunch?ncclCollTraceKernelLaunchType:ncclCollTraceCollLaunchType)); \
  }
#define traceKernelEnd()  { \
    uint32_t pos = __atomic_fetch_add(ncclShmem->comm.collTraceTail, 1, __ATOMIC_SEQ_CST)%COLLTRACE_NUM_ITEMS; \
    ncclShmem->comm.collTrace[pos].timeStamp = __builtin_amdgcn_s_memrealtime(); \
    ncclShmem->comm.collTrace[pos].bid = bid; \
    ncclShmem->comm.collTrace[pos].type = ncclCollTraceKernelEndType; \
  }
#define traceAbort()  { \
    uint32_t pos = __atomic_fetch_add(ncclShmem->comm.collTraceTail, 1, __ATOMIC_SEQ_CST)%COLLTRACE_NUM_ITEMS; \
    ncclShmem->comm.collTrace[pos].timeStamp = __builtin_amdgcn_s_memrealtime(); \
    ncclShmem->comm.collTrace[pos].bid = bid; \
    ncclShmem->comm.collTrace[pos].type = ncclCollTraceAbortType; \
  }
//  traceData(int16_t data2, uint32_t data4, uint64_t data8_0, uint64_t data8_1)
#define traceData(data2, data4, data8_0, data8_1) { \
    uint32_t pos = __atomic_fetch_add(ncclShmem->comm.collTraceTail, 1, __ATOMIC_SEQ_CST)%COLLTRACE_NUM_ITEMS; \
    ncclShmem->comm.collTrace[pos].bid = blockIdx.x; \
    ncclShmem->comm.collTrace[pos].timeStamp = __builtin_amdgcn_s_memrealtime(); \
    ncclShmem->comm.collTrace[pos].funcIndex = data2; \
    ncclShmem->comm.collTrace[pos].data_0 = data4; \
    ncclShmem->comm.collTrace[pos].opCount = data8_0; \
    ncclShmem->comm.collTrace[pos].data_1 = data8_1; \
    ncclShmem->comm.collTrace[pos].type = ncclCollTraceDataType; \
  }
#else
#define traceKernelLaunch()
#define traceAbort()
#define traceData(data2, data4, data8_0, data8_1)
#endif

#ifdef ENABLE_PROFILING
#define __insert_timestamp(line_num) do { \
      if (ncclShmem->prof.count < PROFILE_NUM_ITEMS) { \
        ncclShmem->prof.elem[ncclShmem->prof.count].line = line_num; \
        ncclShmem->prof.elem[ncclShmem->prof.count].timeStamp = __builtin_amdgcn_s_memrealtime(); \
        ncclShmem->prof.count++; \
      } \
    } while(0);
#else
#define __insert_timestamp(line_num)
#endif

__device__ inline bool barrierReduceAny(int bit, uint32_t* abortCount) {
  if (bit) atomicAdd(abortCount, 1); \
  __syncthreads(); \
  return atomicAdd(abortCount, 0) != 0;
}

// Copy src to dst and fill extra size with zeroes
template<typename Tdst, typename Tsrc>
__device__ void copyToShmem(Tdst *dst, Tsrc const *src, int tid, int nthreads) {
  static_assert(sizeof(Tdst)%(2*sizeof(uint64_t)) == 0 && sizeof(Tsrc)%(2*sizeof(uint64_t)) == 0,
      "copyToShmem needs sizes which are multiple of 16B");
  static_assert(sizeof(Tdst) >= sizeof(Tsrc), "Tdst size is too small");
  static_assert(sizeof(Tdst) <= WARP_SIZE*2*sizeof(uint64_t), "copyToShmem limited to 512B to make sure it can always be done in one cycle");
  uint64_t *d = reinterpret_cast<uint64_t*>(dst);
  uint64_t const *s = reinterpret_cast<uint64_t const*>(src);
  uint64_t *shmemPtr = d;
  int offset = 2*tid;
  uint64_t v0, v1;
  if (offset >= sizeof(Tsrc)/sizeof(uint64_t)) {
    v0 = v1 = 0ULL;
  } else {
    v0 = s[offset] ; v1 = s[offset+1];
  }
  if (offset < sizeof(Tdst)/sizeof(uint64_t)) {
    shmemPtr[offset] = v0; shmemPtr[offset+1] = v1;
  }
}

template<typename T>
__device__ int copyToShmem(T *dst, T const *src, int turn=0) {
  static_assert(sizeof(uint64_t) <= alignof(T), "Uhoh");
  uint64_t *d = reinterpret_cast<uint64_t*>(dst);
  uint64_t const *s = reinterpret_cast<uint64_t const*>(src);
  int t = threadIdx.x - turn;
  if (t < 0) t += blockDim.x;
  int n = sizeof(T)/sizeof(uint64_t);

  int delta = (n + WARP_SIZE-1) & -WARP_SIZE; // round up to warp lane 0
  if (delta < blockDim.x) {
    turn += delta;
    if (turn >= blockDim.x) turn -= blockDim.x;
  }
  else
    turn = 0;

  n -= t;
  d += t;
  s += t;
  #pragma unroll
  for (int i=0; i < divUp(sizeof(T), WARP_SIZE*sizeof(uint64_t)); i++) {
    if (n > 0) {
      *d = *s;
      d += blockDim.x;
      s += blockDim.x;
      n -= blockDim.x;
    }
  }
  return turn;
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
    int inc = w->header.type == ncclWorkTypeRegColl ? sizeof(ncclWorkElemReg) / sizeof(ncclWorkElem) : 1;
    #pragma unroll 1
    for(int e=0; e < NCCL_MAX_WORK_ELEMENTS && w->elems[e].header.type != ncclWorkTypeUnused; e += inc) {
      if (wid < w->header.nWarps)
        RunWorkElement<Fn, T, RedOp, Algo, Proto>().run(&w->elems[e]);
    }
  }
};

struct ncclShmemGroup {
  ncclConnInfo *recvConns[NCCL_MAX_DIRECT_ARITY];
  ncclConnInfo *sendConns[NCCL_MAX_DIRECT_ARITY];
  void* srcs[NCCL_MAX_DIRECT_ARITY+1];
  void* dsts[NCCL_MAX_DIRECT_ARITY+1];
  int totalSendSize[NCCL_MAX_SLICE_PER_CHUNK];
  uint64_t barrier;
  uint64_t barrier_next[NCCL_MAX_GROUPS];
};

struct ncclShmemData {
  union {
    uint64_t ll128warp[NCCL_MAX_GROUPS][NCCL_MAX_GROUPS];
    struct ncclShmemGroup groups[NCCL_MAX_GROUPS];
  };
  uint32_t sync[NCCL_MAX_GROUPS];
  uint64_t redOpArgs[NCCL_MAX_DIRECT_ARITY+1];
  struct ncclDevComm comm;
  struct ncclChannel channel;
  uint64_t pad;
  struct ncclWork work;
#ifdef ENABLE_PROFILING
  struct ncclProf prof;
#endif
};

static __device__ void ncclRedopPtrDeref(struct ncclWorkElem* we) {
  if (we->header.type != ncclWorkTypeUnused && we->redOpArgIsPtr) {
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

extern __device__ struct ncclShmemData *ncclShmem;

template<ncclFunc_t Fn, typename T, typename RedOp, int Algo, int Proto, int FnIndex, bool COLLTRACE>
__device__ void ncclKernel(struct ncclDevComm* comm, ncclWorkElem first)  {
  int tid = threadIdx.x;
  int nthreads = blockDim.x;
  int bid = blockIdx.x;
  __shared__ struct ncclShmemData shmem;
  ncclShmem = &shmem;
  __shared__ uint32_t abortCount;
  if (tid == 0) {
    abortCount = 0;
    for (auto i = 0; i < NCCL_MAX_GROUPS; i++) {
      shmem.groups[i].barrier = 0;
      for (auto j = 0; j < NCCL_MAX_GROUPS; j++) shmem.groups[i].barrier_next[j] = 0;
    }
 }
  __syncthreads();

  int turn = copyToShmem(&ncclShmem->comm, comm);
#ifdef ENABLE_PROFILING
  if (tid == 0) {
    ncclShmem->prof.count = 0;
    ncclShmem->prof.seq = ncclShmem->comm.devProf[bid].seq;
  }
#endif
  if (tid == 0) __insert_timestamp(__LINE__);
  // get address of channel without incurring indirect load from ncclDevCom::channels
  ncclChannel *channel = &((ncclDevCommAndChannels*)comm)->channels[bid];
  turn = copyToShmem(&ncclShmem->channel, channel, turn);

  // To optimize for latency, (only) the first operation is passed as argument.
  if (bid == 0 && first.header.type != ncclWorkTypeUnused) {
    // Copy first elem to work and zero out the rest
    copyToShmem(&ncclShmem->work, &first, tid, nthreads);
  }
  __syncthreads(); // publish ncclShmem
  if (tid == 0) __insert_timestamp(__LINE__);
  if (tid == 0) __insert_timestamp(__LINE__);

  ncclWork *workFifoHost = ncclShmem->channel.workFifo;
  ncclWork *workFifoDev = ncclShmem->channel.workFifoDev;
  int workFifoIx = ncclShmem->channel.index;

  bool skipLoadWork = false, firstLaunch = true;
  if (bid == 0 && first.header.type != ncclWorkTypeUnused)
    skipLoadWork = true;

  while (true) {
    if (!skipLoadWork) {
      copyToShmem(&ncclShmem->work, &workFifoDev[workFifoIx], tid, nthreads);
      if (tid == 0) __insert_timestamp(__LINE__);
      { // Check whether the last operation was aborted and make sure all threads exit
        int aborted = tid == 0 ? *comm->abortFlag : 0;
        if (barrierReduceAny(aborted, &abortCount)) { // publish ncclShmem->work
          if (COLLTRACE && tid == 0) traceAbort();
          break;
        }
        if (tid == 0)
          workFifoHost[workFifoIx].header.type = ncclWorkTypeUnused;
      }
    }
    if (tid == 0) __insert_timestamp(__LINE__);

    workFifoIx = (workFifoIx + 1)%NCCL_MAX_OPS;
    if (tid == 0)
      channel->index = workFifoIx; // write back to real channel, not shmem shadow

    __syncwarp();
    if (ncclShmem->work.header.type == ncclWorkTypeColl) {
      if (tid < NCCL_MAX_WORK_ELEMENTS) ncclRedopPtrDeref(&ncclShmem->work.elems[tid]);
    } else if (ncclShmem->work.header.type == ncclWorkTypeRegColl) {
      if (tid < NCCL_MAX_WORK_ELEMENTS_REG) ncclRedopPtrDeref(&ncclShmem->work.regElems[tid].elem);
    }
    __syncthreads();

    if (COLLTRACE && tid == 0) {
      traceKernelLaunch(ncclShmem->work.elems[0],firstLaunch);
      firstLaunch = false;
      #pragma unroll 1
      for(int e=1; e < NCCL_MAX_WORK_ELEMENTS && ncclShmem->work.elems[e].header.type != ncclWorkTypeUnused; e ++) {
        traceColl(ncclShmem->work.elems[e], 0);
      }
    }
    if (tid == 0) __insert_timestamp(__LINE__);
    if (ncclShmem->work.header.funcIndex == FnIndex)
      RunWork<Fn, T, RedOp, Algo, Proto>().run(&ncclShmem->work);
    else
      NCCL_CALL_FUNCTIONS(ncclShmem->work.header.funcIndex);

    if (ncclShmem->work.header.isLast) break;
    __syncthreads();
    skipLoadWork = false;
  }
  if (COLLTRACE && tid == 0) traceKernelEnd()
#ifdef ENABLE_PROFILING
  if (ncclShmem->comm.devProf->seq < PROFILE_NUM_LAUNCHES) {
    copyToShmem(ncclShmem->comm.devProf+MAXCHANNELS*ncclShmem->prof.seq+blockIdx.x, &ncclShmem->prof);
    if (tid == 0) ncclShmem->comm.devProf[bid].seq++;
  }
#endif
}

#define IMPL_COLL_KERN(func, algo, proto, devredop, type, fIndex) \
__launch_bounds__(NCCL_MAX_NTHREADS, 1) \
__global__ void NCCL_KERN_NAME(func, algo, proto, devredop, type)(struct ncclDevComm* comm, ncclWorkElem first) { \
  if (comm->collTraceThread) \
    ncclKernel<ncclFunc##func, type, Func##devredop<type>, NCCL_ALGO_##algo, NCCL_PROTO_##proto, fIndex, true>(comm, first); \
  else \
    ncclKernel<ncclFunc##func, type, Func##devredop<type>, NCCL_ALGO_##algo, NCCL_PROTO_##proto, fIndex, false>(comm, first); \
}

// Examples :     AllReduce, RING, LL,    Sum,   uint8
/* Functions for aggregation case */
#define IMPL_COLL_FUNC(func, algo, proto, devredop, type) \
__device__  __attribute__((noinline)) void NCCL_FUNC_NAME(func, algo, proto, devredop, type)() { \
  RunWork<ncclFunc##func, type, Func##devredop<type>, NCCL_ALGO_##algo, NCCL_PROTO_##proto>().run(&ncclShmem->work); \
}

// Only generate inline kernels for LL
#define IMPL_COLL4(func, algo, devredop, type, ncclType) \
  IMPL_COLL_FUNC(func, algo, LL,     devredop, type) \
  IMPL_COLL_FUNC(func, algo, SIMPLE, devredop, type) \

#define IMPL_COLL3(func, devredop, type, ncclType) \
  IMPL_COLL4(func, TREE,    devredop, type, ncclType) \
  IMPL_COLL4(func, RING,    devredop, type, ncclType) \
  IMPL_COLL4(func, COLLNET, devredop, type, ncclType)

#define IMPL_COLL2(func, devredop) \
  IMPL_COLL3(func, devredop, int8_t,   ncclInt8) \
  IMPL_COLL3(func, devredop, uint8_t,  ncclUint8) \
  IMPL_COLL3(func, devredop, int32_t,  ncclInt32) \
  IMPL_COLL3(func, devredop, uint32_t, ncclUint32) \
  IMPL_COLL3(func, devredop, int64_t,  ncclInt64) \
  IMPL_COLL3(func, devredop, uint64_t, ncclUint64) \
  IMPL_COLL3(func, devredop, half,     ncclFloat16) \
  IMPL_COLL3(func, devredop, float,    ncclFloat32) \
  IMPL_COLL3(func, devredop, double,   ncclFloat64) \
  IMPL_COLL3(func, devredop, rccl_bfloat16, ncclBfloat16)

#define IMPL_COLL2A(func, devredop) \
  IMPL_COLL3(func, devredop, int8_t,   ncclInt8) \
  IMPL_COLL3(func, devredop, uint8_t,  ncclUint8) \
  IMPL_COLL3(func, devredop, int32_t,  ncclInt32) \
  IMPL_COLL3(func, devredop, uint32_t, ncclUint32) \
  IMPL_COLL3(func, devredop, int64_t,  ncclInt64) \
  IMPL_COLL3(func, devredop, uint64_t, ncclUint64)

// Reduction define all functions
#define IMPL_COLL_R(func) \
  IMPL_COLL2(func, Sum) \
  IMPL_COLL2(func, Prod) \
  IMPL_COLL2(func, Min) \
  IMPL_COLL2(func, Max) \
  IMPL_COLL2(func, PreMulSum) \
  IMPL_COLL2A(func, SumPostDiv)

// Copy primitives only define one function for copy
#define IMPL_COLL_C(func) IMPL_COLL3(func, Sum, int8_t, ncclInt8);

// Point-to-point primitives only have one function/kernel.
#define IMPL_COLL_P(func) \
  IMPL_COLL_FUNC(func, RING, SIMPLE, Sum, int8_t); \
  IMPL_COLL_KERN(func, RING, SIMPLE, Sum, int8_t, FUNC_INDEX_P2P);

// AllToAll Pivot primitive only has one function.
#define IMPL_COLL_ALLTOALL_PIVOT(func) \
  IMPL_COLL_FUNC(func, RING, SIMPLE, Sum, int8_t);

#endif
