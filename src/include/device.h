/*************************************************************************
 * Copyright (c) 2015-2022, NVIDIA CORPORATION. All rights reserved.
 * Modifications Copyright (c) 2019-2022 Advanced Micro Devices, Inc. All rights reserved.
 * Modifications Copyright (c) Microsoft Corporation. Licensed under the MIT License.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

#ifndef NCCL_DEVICE_H_
#define NCCL_DEVICE_H_

#include "nccl.h"
#include "rccl_float8.h"
#include <hip/hip_bfloat16.h>
#include "nccl_common.h"
#include "align.h"
#include "collectives.h"
#if defined(ENABLE_NPKIT)
#include "npkit/npkit_struct.h"
#endif
#include <stdint.h>

extern const char* ncclFuncStr[NCCL_NUM_FUNCTIONS+2];

extern const char* ncclAlgoStr[NCCL_NUM_ALGORITHMS];

extern const char* ncclProtoStr[NCCL_NUM_PROTOCOLS];

extern const char* funcNames[FUNC_INDEX_TOTAL];

#define NCCL_MAX_OPS 2048
#define NCCL_STEPS 8

#include "net_device.h"

enum ncclDevRedOp_t {
  ncclDevSum, ncclDevProd, ncclDevMinMax,
  ncclDevPreMulSum, ncclDevSumPostDiv,
  ncclNumDevRedOps
};
struct ncclDevRedOpFull {
  ncclDevRedOp_t op;
  ncclRedOp_t proxyOp;
  bool scalarArgIsPtr;
  uint64_t scalarArg;
};

union ncclLLFifoLine {
  /* Flags have to be *after* data, because otherwise, an incomplete receive
     from the network may receive the flag but not the data.
     Note this is assuming that either we receive contiguous chunks of data
     (sockets) or data is written with an atomicity of 8 bytes (IB/RDMA). */
  struct {
    uint32_t data1;
    uint32_t flag1;
    uint32_t data2;
    uint32_t flag2;
  };
  uint64_t v[2];
  int4 i4;
};

#define WARP_SIZE warpSize

#define MAXCHANNELS 128
#define CHANNEL_LIMIT 16

#define NCCL_MAX_NTHREADS 256
#define NCCL_SIMPLE_MAX_NTHREADS NCCL_MAX_NTHREADS
#define NCCL_LL_MAX_NTHREADS NCCL_MAX_NTHREADS
#define NCCL_LL_LINES_PER_THREAD 8
#ifdef TEST_LL_CLEANUP
#define NCCL_LL_CLEAN_MASK 0x078 // Set to 0x100 to disable cleanup
#define NCCL_LL_FLAG_MAX   0x100
#define NCCL_LL_FLAG(a) ((uint32_t)((a) % NCCL_LL_FLAG_MAX))
#else
#define NCCL_LL_CLEAN_MASK 0x7ffffff8
#define NCCL_LL_FLAG(a) ((uint32_t)(a))
#endif
// Make sure the clean mask will last for at least NCCL_NSTEPS
static_assert(NCCL_LL_CLEAN_MASK % NCCL_STEPS == 0, "Invalid NCCL_LL_CLEAN_MASK value");

#define NCCL_LL128_LINESIZE 64
#define NCCL_LL128_LINEELEMS (NCCL_LL128_LINESIZE/sizeof(uint64_t))
#define NCCL_LL128_DATAELEMS (NCCL_LL128_LINEELEMS-1)

#define NCCL_LL128_MAX_NTHREADS 256
#define NCCL_LL128_ELEMS_PER_THREAD 28

#define NCCL_LL128_SHMEM_ELEMS_PER_THREAD 4
#define NCCL_LL128_SHMEM_SIZE (NCCL_LL128_SHMEM_ELEMS_PER_THREAD*NCCL_LL128_MAX_NTHREADS)

#define NCCL_DIRECT_WRITE 0x01
#define NCCL_DIRECT_READ  0x02
#define NCCL_DIRECT_NIC   0x04
#define NCCL_IPC_WRITE    0x08
#define NCCL_IPC_READ     0x10
#define NCCL_NVLS_MIN_POLL 0x20

#define NCCL_MAX_COLLNET_SIZE (1L << 29)

enum ncclRegBufferType {
  NCCL_REGULAR_BUFFER = 0,
  NCCL_IPC_REG_BUFFER = 1,
  NCCL_NVLS_REG_BUFFER = 2,
  NCCL_COLLNET_REG_BUFFER = 3
};

struct ncclConnInfo {
  // Regular comm mechanism
  char *buffs[NCCL_NUM_PROTOCOLS]; // Local for recv, remote for send
  void* mhandles[NCCL_NUM_PROTOCOLS];
  uint64_t *tail;     // Local for recv, remote for send
  uint64_t *head;     // Local for send, remote for recv

  int flags;          // Direct communication / other flags
  int shared;         // Buffers are shared
  int stepSize;       // Step size for the SIMPLE buffer
  void **ptrExchange; // Pointer exchange for direct communication
  uint64_t* redOpArgExchange; // PreOp scaler exchange for direct pull case

  struct ncclConnFifo* connFifo; // Used for GPU - Proxy communication

  uint64_t step;      // Keep where we are
  uint64_t llLastCleaning;

  // GPU's HDP_MEM_FLUSH_ADDR: HDP Memory Coherency Flush Control. This register
  // allows software to explicitly initiate a flush read to HDP memory. See more
  // descriptions in primitives.h.
  uint32_t* next_hdp_reg;  // Next GPU in ring (for p2p transport use only)
  uint32_t* curr_hdp_reg;  // Current GPU's HDP register
  ncclNetDeviceHandle_t netDeviceHandle;
};

struct ncclProxyConnector {
  int tpRank;
  int tpLocalRank;
  int sameProcess;
  struct ncclProxyConnection* connection;
  ncclResult_t (*proxyProgress)(struct ncclProxyState* proxyState, struct ncclProxyArgs*); // Copied from transport if necessary
};

struct ncclConnector {
  int connected;
  struct ncclProxyConnector proxyConn;
  struct ncclTransportComm* transportComm;
  void* transportResources;
  struct ncclConnInfo conn;
};

struct ncclRing {
  // Shortcuts for userRanks[1] and userRanks[n-1]
  int prev;
  int next;

  // Maps an internal nccl index to user-specified rank order. This is necessary
  // since we need to know how the user expects data to be ordered across
  // devices. Ordered from current device.
  int* userRanks;

  int index; // This rank's index in the ring
};


// The root of each tree only has one node down (+1 intra-node).
#define NCCL_MAX_TREE_ARITY_TOP 2
// Nodes inside the binary tree can have to two nodes down (+1 intra-node).
#define NCCL_MAX_TREE_ARITY 3
struct ncclTree {
  int depth;
  int up;
  int down[NCCL_MAX_TREE_ARITY];
};

#define NCCL_MAX_DIRECT_ARITY 7
struct ncclDirect {
  int depth;
  int out;
  int nHeads;   // Number of parallel N<->1<->net operations we'll do in parallel; size of up/down
  int headRank; // Index in 0..nHeads-1 I am the head rank of. -1 if I'm not a head rank (no local NIC)
  int shift;    // Shuffling of send/recv for scatter/gather operations, basically localRank%nHeads
  // The heads[...] are guaranteed to be in rotated order start with self:
  //   headRank, (headRank+1)%nHeads, (headRank+2)%nHeads, ...
  int heads[NCCL_MAX_DIRECT_ARITY+1];
  int up[NCCL_MAX_DIRECT_ARITY];
  int down[NCCL_MAX_DIRECT_ARITY];
};

#define NCCL_CONN_IDX_P2P_NET 2
#define NCCL_MAX_NVLS_ARITY 32
#define NCCL_MAX_NVLS_TREE_ARITY 3
struct ncclNvls {
  int out;
  int nHeads;   // Number of parallel N<->1<->net operations we'll do in parallel; size of up/down
  int headRank; // Index in 0..nHeads-1 I am the head rank of. -1 if I'm not a head rank (no local NIC)
  int up[NCCL_MAX_NVLS_ARITY];
  int down;
  int treeUp;
  int treeDown[NCCL_MAX_NVLS_TREE_ARITY];
  int node;
  int nNodes;
};

#if __CUDA_ARCH__ >= 900
#define NCCL_MAX_ARITY NCCL_MAX_NVLS_ARITY
#else
#define NCCL_MAX_ARITY NCCL_MAX_DIRECT_ARITY
#endif

#define NCCL_MAX_CONNS 3
struct ncclChannelPeer {
  struct ncclConnector send[NCCL_MAX_CONNS];
  struct ncclConnector recv[NCCL_MAX_CONNS];
  int refCount;
};

struct ncclDevComm;

#pragma pack(push)  /* push current alignment to stack */
#pragma pack(8)     /* set alignment to 8 bytes boundary */
/* ncclWork is to be a power of two, currently 8x64 bytes, */
/* to make sure reads to host from the CUDA kernel are aligned. */
/* Make sure to adjust padding at the end of ncclWorkElem. */
#define NCCL_WORK_SIZE 256

enum ncclWorkType : uint8_t {
   ncclWorkTypeUnused=0,
   ncclWorkTypeColl=1,
   ncclWorkTypeP2p=2,
   ncclWorkTypeRegColl=3
};
enum ncclWorkP2PType : uint8_t {
  ncclWorkP2pTypeUnused=0,
  ncclWorkP2pTypeSend,
  ncclWorkP2pTypeRecv
};

struct ncclWorkHeader {
  union {
    int32_t workNext;  // when isLast=0: Offset from kernel argument workHead
    uint32_t doneAcks; // when isLast=1: Monotonic (mod 1<<32) ack value to send back.
  };
  uint16_t funcIndex;
  uint8_t isLast:1; // last work for this kernel
  uint8_t inFifo:1; // is this work in the fifo
  enum ncclWorkType type;
};

struct ncclWorkElem {
  union {
    uint8_t flagBits;
    struct {
      uint8_t isUsed:1, redOpArgIsPtr:1, oneNode:1;
    };
  };
  uint8_t regUsed;
  uint8_t nWarps;
  uint8_t direct;

  uint32_t root:30, connIndex:2;
  const void *sendbuff;
  void *recvbuff;

  uint64_t count:39, opCount:25;
  uint64_t redOpArg;
  uint64_t chunkCount:25, workCount:39;
  union {
    struct {
      uint64_t lastChunkCount:25;
      uint64_t workOffset:39;
    };
    struct {
      uint32_t nChannels;
      uint16_t bid;
      // Pivot A2A kernel computes chunk size itself.
      // Instead, it needs the number of bidirectional rings.
      uint16_t pivotA2ANumBiRings;
    };
  };
};

static_assert((NCCL_WORK_SIZE - alignUp(sizeof(ncclWorkHeader), alignof(ncclWorkElem)))/sizeof(ncclWorkElem) == 4, "Sanity check: NCCL_MAX_WORK_ELEMENTS == 4");
#define NCCL_MAX_WORK_ELEMENTS 1

struct ncclWorkElemP2p {
  struct {
    int32_t peer:28;
    uint32_t connIndex:2;
    int32_t proto:2;
  };
  union {
    uint16_t flagBit;
    struct {
      enum ncclWorkP2PType p2pType:4;
      uint8_t nWarps:4;
      uint8_t warpStart:4;
      uint8_t ngroups:4;
    };
  };
  uint8_t reg:1;
  uint16_t opCount:12;
  // Important not to use any fields with greater than 4-byte alignment since
  // we need sizeof(ncclWorkElemP2p)==28, but that would be padded up to 32 if
  // there were 8-byte fields.
  //void* buff;
  uint32_t buffHi32, buffLo32; // buff = buffHi32<<32 | buffLo32;
  //size_t count;
  uint32_t countHi32, countLo32; // count = countHi32<<32 | countLo32;
  int chunkSize;
};

static_assert(((NCCL_WORK_SIZE - alignUp(sizeof(ncclWorkHeader), alignof(ncclWorkElemP2p)))/sizeof(ncclWorkElemP2p)) == 8, "Sanity check: NCCL_MAX_WORK_ELEMENTS_P2P == 8");
#define NCCL_MAX_WORK_ELEMENTS_P2P 2

struct ncclWorkElemReg {
  struct ncclWorkElem elem;
  void* dnInputs[NCCL_MAX_DIRECT_ARITY+1];
  void* dnOutputs[NCCL_MAX_DIRECT_ARITY+1];
  void* upOutputs[NCCL_MAX_DIRECT_ARITY+1];
};

#define NCCL_MAX_WORK_ELEMENTS_REG ((NCCL_WORK_SIZE - alignUp(sizeof(ncclWorkHeader), alignof(ncclWorkElemReg)))/sizeof(ncclWorkElemReg))
static_assert(NCCL_MAX_WORK_ELEMENTS_REG == 1, "Sanity check: NCCL_MAX_WORK_ELEMENTS_REG == 1");

// Number of named barriers supported by CUDA
#define NCCL_MAX_GROUPS (NCCL_MAX_NTHREADS/WARP_SIZE)

struct ncclWork {
  struct ncclWorkHeader header;
  union {
    char pad[NCCL_WORK_SIZE - sizeof(struct ncclWorkHeader)];
    struct ncclWorkElem elems[NCCL_MAX_WORK_ELEMENTS];
    struct ncclWorkElemP2p p2pElems[NCCL_MAX_WORK_ELEMENTS_P2P];
    struct ncclWorkElemReg regElems[NCCL_MAX_WORK_ELEMENTS_REG];
  };
};
static_assert(sizeof(struct ncclWork) == NCCL_WORK_SIZE, "Sanity check: sizeof(struct ncclWork) == NCCL_WORK_SIZE");
static_assert(sizeof(struct ncclWork)%16 == 0, "Sanity check: sizeof(struct ncclWork)%16 == 0");

struct ncclDevChannelPeer {
  // Stripped version of ncclChannelPeer where we only keep the ncclConnInfo
  // instead of the full ncclConnector.
  struct ncclConnInfo send[NCCL_MAX_CONNS];
  struct ncclConnInfo recv[NCCL_MAX_CONNS];

};
#pragma pack(pop)   /* restore original alignment from stack */

#ifdef ENABLE_PROFILING
#define PROFILE_NUM_ITEMS 31
#define PROFILE_NUM_LAUNCHES 1024

struct ncclProf {
  uint32_t count;
  uint32_t seq; // only entry from first launch is used
  struct {
    uint64_t line:16;
    uint64_t timeStamp:48;
  } elem[PROFILE_NUM_ITEMS];
};
static_assert(sizeof(struct ncclProf) == 256, "ncclProf must have size of 256");
#endif

#ifdef ENABLE_COLLTRACE
typedef enum {
  ncclCollTraceNotReady = 0,
  ncclCollTraceKernelLaunchType = 1,
  ncclCollTraceKernelEndType = 2,
  ncclCollTraceCollLaunchType = 3,
  ncclCollTraceAbortType = 4,
  ncclCollTraceDataType = 5,
  ncclCollTraceCollElemType = (1<<4),
  ncclCollTraceP2pElemType = (1<<5),
} ncclCollTraceDataType_t;

struct ncclCollTrace {
  uint8_t type;
  uint8_t bid;
  int16_t funcIndex;
  uint32_t data_0;
  uint64_t timeStamp;
  union {
    uint64_t opCount;
    uint32_t p2pOpCount[2];
  };
  union {
    uint64_t data_1;
    struct {
      uint8_t nWarps;
      uint8_t bid;
      uint8_t nChannels;
    } coll;
    struct {
      int16_t peer;
      uint8_t ngroups:4;
      uint8_t connIndex:4;
      uint8_t warpStart:4;
      uint8_t nWarps:4;
    } p2p[2];
  };
};
static_assert(sizeof(struct ncclCollTrace) == 8*sizeof(int), "ncclCollTrace must have a pow2 size");

union ncclCollTraceTail{
  uint32_t tail;
  char padding[4096];
};

#define COLLTRACE_NUM_ITEMS 8192
#endif

struct alignas(16) ncclDevChannel {
  struct ncclDevChannelPeer** peers;
  struct ncclRing ring;
  struct ncclTree tree;
  struct ncclTree collnetChain;
  struct ncclDirect collnetDirect;
  struct ncclTree binTree;
  struct ncclNvls nvls;
  uint32_t* workFifoDone; // Location of done counter, device writes index+1 of last work processed
};

struct ncclDevComm {
  int rank;
  int nRanks;
  int node;
  int nNodes;
  int buffSizes[NCCL_NUM_PROTOCOLS];
  int p2pChunkSize;

  // Operation list for aggregation
  int workFifoDepth;
  struct ncclWork* workFifoHeap; // may be cudaHost or GDR memory

  int* collNetDenseToUserRank;

  // Flag to ask NCCL kernels to abort
  volatile uint32_t* abortFlag;

  // Channels, device side
  struct ncclDevChannel* channels/*[MAXCHANNELS]*/;

#if defined(ENABLE_NPKIT)
  NpKitEventCollectContext* npKitEventCollectContexts;
  uint64_t* cpuTimestamp;
#endif

#ifdef ENABLE_COLLTRACE
  struct ncclCollTrace* collTrace;
  union ncclCollTraceTail *collTraceTail;
  pthread_t collTraceThread;
#endif

#ifdef ENABLE_PROFILING
  struct ncclProf* devProf;
#endif
};

struct alignas(16) ncclDevCommAndChannels {
  struct ncclDevComm comm;
  struct ncclDevChannel channels[MAXCHANNELS];
};

#ifdef __CUDA_ARCH__
  #define NCCL_CUDA_ARCH __CUDA_ARCH__
#else
  #define NCCL_CUDA_ARCH 0
#endif

template<typename T>
__host__ __device__ constexpr T min_constexpr(T a) { return a; }
template<typename T, typename ...Ts>
__host__ __device__ constexpr T min_constexpr(T a, T b, Ts ...c) {
  return min_constexpr<T>((a < b ? a : b), c...);
}

template<typename T>
__host__ __device__ constexpr T max_constexpr(T a) { return a; }
template<typename T, typename ...Ts>
__host__ __device__ constexpr T max_constexpr(T a, T b, Ts ...c) {
  return max_constexpr<T>((a > b ? a : b), c...);
}

// Calculate the unroll factor given:
// * bytePerPack: number of bytes accessed per instruction
// * insns: max permissible unroll value
// * bytes: desired number of in-flight bytes per iteration ( = unroll*bytePerPack)
__host__ __device__ constexpr int ncclCalcUnroll(int bytePerPack, int insns, int bytes) {
  return min_constexpr(insns, (bytes + bytePerPack-1)/bytePerPack);
}

// Note that all unroll value logic should depend on a given cudaArch argument
// and not __CUDA_ARCH__ since these need to be host-side executable where the
// arch value is strictly runtime only. By defaulting to NCCL_CUDA_ARCH, device
// side code can elide passing the arch for brevity.

__host__ __device__ constexpr int ncclCollUnroll(int cudaArch = NCCL_CUDA_ARCH) {
  // Our collective unroll should move to the same bytes&insns model as NVLS.
  return cudaArch >= 800 ? 8 : 4;
}

__host__ __device__ constexpr int ncclNvlsUnrollBytes(int cudaArch = NCCL_CUDA_ARCH) { return 4*16; }
__host__ __device__ constexpr int ncclNvlsUnrollInsns(int cudaArch = NCCL_CUDA_ARCH) { return 16; }

__host__ __device__ constexpr int ncclNvlsUnroll(int bytePerPack, int cudaArch = NCCL_CUDA_ARCH) {
  return ncclCalcUnroll(bytePerPack, ncclNvlsUnrollInsns(cudaArch), ncclNvlsUnrollBytes(cudaArch));
}

// The amount of dynamic shmem per warp
__host__ __device__ constexpr int ncclShmemScratchWarpSize(int cudaArch = NCCL_CUDA_ARCH) {
  return (max_constexpr<int>(
      /*LL    */0,
      /*LL128 */(NCCL_LL128_SHMEM_ELEMS_PER_THREAD*WARP_SIZE)*sizeof(uint64_t),
      /*SIMPLE*/(ncclCollUnroll(cudaArch)*WARP_SIZE + 1)*16,
      // NVLS needs an extra 16B to read unaligned data.
      /*NVLS  */WARP_SIZE*(cudaArch >= 900 ? ncclNvlsUnrollBytes(cudaArch) : 0) + 16
    ) + 15) & -16; // pad to 16 bytes
}

// The amount of dynamic shmem per block
__host__ __device__ constexpr int ncclShmemDynamicSize(int cudaArch = NCCL_CUDA_ARCH) {
  return cudaArch < 700 ? 0 : ncclShmemScratchWarpSize(cudaArch)*(NCCL_MAX_NTHREADS/WARP_SIZE);
}

// Launch a one-rank reduction on stream.
ncclResult_t ncclLaunchOneRank(void* dst, void const* src, size_t nElts, struct ncclDevRedOpFull redOp, ncclDataType_t type, cudaStream_t stream);

// `ncclNvlsSupported()` needs to be in sync with "func_valid" in "src/device/generate.py"
inline bool ncclNvlsSupported(int devRedOp, int type) {
  switch (type) {
  case ncclInt32:
  case ncclUint32:
  case ncclInt64:
  case ncclUint64:
  case ncclFloat16:
#if defined(RCCL_BFLOAT16)
  case ncclBfloat16:
#endif
#if defined(RCCL_FLOAT8)
  case ncclFp8E4M3:
  case ncclFp8E5M2:
#endif
    return devRedOp == ncclDevSum || devRedOp == ncclDevMinMax;
  case ncclFloat:
  case ncclDouble:
    return devRedOp == ncclDevSum;
  default:
    return false;
  }
}

// Map the rowIdx to funcIdx
extern int const ncclDevFuncRowToId[];

// `ncclFuncIndex()` needs to be in sync with 'ALL_COLLS' in Generate.cmake
inline int ncclDevFuncId(int coll, int devRedOp, int type, int algo, int proto) {
  int row = 0;
  do {
    // RING / <all_protos> / Sum / int8_t
    if (coll == ncclFuncAllGather) {
      row += proto;
      break;
    }
    row += NCCL_NUM_PROTOCOLS;

    // <all_algos> / <all_protos> / <all_redops> / <all_types>
    if (coll == ncclFuncAllReduce) {
      row += (((algo * NCCL_NUM_PROTOCOLS + proto) * ncclNumDevRedOps + devRedOp) * ncclNumTypes + type) - NCCL_NUM_FLOATS * (algo * NCCL_NUM_PROTOCOLS + proto);
      break;
    }
    row += (NCCL_NUM_ALGORITHMS - 2) * NCCL_NUM_PROTOCOLS * (ncclNumDevRedOps * ncclNumTypes - NCCL_NUM_FLOATS);

    // RING / SIMPLE / Sum / int8_t
    if (coll == ncclFuncAllToAllPivot) break;
    row += 1;

    // RING / <all_protos> / Sum / int8_t
    if (coll == ncclFuncBroadcast) {
      row += proto;
      break;
    }
    row += NCCL_NUM_PROTOCOLS;

    // RING / <all_protos> / <all_redops> / <all_types>
    if (coll == ncclFuncReduce) {
      row += ((proto * ncclNumDevRedOps + devRedOp) * ncclNumTypes + type) - NCCL_NUM_FLOATS * proto; 
      break;
    }
    row += NCCL_NUM_PROTOCOLS * (ncclNumDevRedOps * ncclNumTypes - NCCL_NUM_FLOATS);

    // RING / <all_protos> / <all_redops> / <all_types>
    if (coll == ncclFuncReduceScatter) {
      row += ((proto * ncclNumDevRedOps + devRedOp) * ncclNumTypes + type) - NCCL_NUM_FLOATS * proto;
      break;
    }
    row += NCCL_NUM_PROTOCOLS * (ncclNumDevRedOps * ncclNumTypes - NCCL_NUM_FLOATS);

    // RING / SIMPLE / Sum / int8_t
    if (coll == ncclFuncSendRecv) break;
    row += 1;

  } while (false);

  return ncclDevFuncRowToId[row];
}

inline int ncclDevFuncId_P2p() { return ncclDevFuncRowToId[FUNC_INDEX_TOTAL - NCCL_NUM_ONERANK - 1]; }

#endif
