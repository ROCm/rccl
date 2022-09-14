/*************************************************************************
 * Copyright (c) 2015-2022, NVIDIA CORPORATION. All rights reserved.
 * Modifications Copyright (c) 2019-2022 Advanced Micro Devices, Inc. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

#ifndef NCCL_DEVICE_H_
#define NCCL_DEVICE_H_

#include "nccl.h"
#include "rccl_bfloat16.h"
#include "align.h"
#if defined(ENABLE_NPKIT)
#include "npkit/npkit_struct.h"
#endif
#include <stdint.h>


#define NCCL_NUM_FUNCTIONS 5 // SendRecv and AllToAllPivot not included for now
typedef enum { ncclFuncBroadcast, ncclFuncReduce, ncclFuncAllGather, ncclFuncReduceScatter, ncclFuncAllReduce, ncclFuncSendRecv, ncclFuncSend, ncclFuncRecv, ncclFuncAllToAllPivot, ncclNumFuncs} ncclFunc_t;
extern const char* ncclFuncStr[NCCL_NUM_FUNCTIONS+2];

#define NCCL_NUM_ALGORITHMS 3 // Tree/Ring/CollNet
#define NCCL_ALGO_TREE 0
#define NCCL_ALGO_RING 1
#define NCCL_ALGO_COLLNET 2
extern const char* ncclAlgoStr[NCCL_NUM_ALGORITHMS];

#define NCCL_NUM_PROTOCOLS 3 // Simple/LL/LL128
#define NCCL_PROTO_LL 0
#define NCCL_PROTO_LL128 1
#define NCCL_PROTO_SIMPLE 2
extern const char* ncclProtoStr[NCCL_NUM_PROTOCOLS];

#define NCCL_MAX_OPS 2048
#define NCCL_STEPS 8

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

#if defined(__gfx1030__)
#define WARP_SIZE 32
#else
#define WARP_SIZE 64
#endif
#define MAXCHANNELS 32
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

struct ncclConnInfo {
  // Regular comm mechanism
  char *buffs[NCCL_NUM_PROTOCOLS]; // Local for recv, remote for send
  uint64_t *tail;     // Local for recv, remote for send
  uint64_t *head;     // Local for send, remote for recv

  int direct;         // Direct communication
  int shared;         // Buffers are shared
  void **ptrExchange; // Pointer exchange for direct communication
  uint64_t* redOpArgExchange; // PreOp scaler exchange for direct pull case

  int *sizesFifo;     // Sizes fifo from GPU to proxy
  int *offsFifo;      // Buffer fifo from proxy to GPU

  uint64_t step;      // Keep where we are
  uint64_t llLastCleaning;

  // GPU's HDP_MEM_FLUSH_ADDR: HDP Memory Coherency Flush Control. This register
  // allows software to explicitly initiate a flush read to HDP memory. See more
  // descriptions in primitives.h.
  uint32_t* next_hdp_reg;  // Next GPU in ring (for p2p transport use only)
  uint32_t* curr_hdp_reg;  // Current GPU's HDP register
};

struct ncclProxyConnector {
  int rank;
  int localRank;
  struct ncclProxyConnection* connection;
  struct ncclComm* comm;
};

struct ncclConnector {
  int connected;
  struct ncclProxyConnector proxyConn;
  struct ncclTransportComm* transportComm;
  void* transportResources;
  struct ncclConnInfo conn;
  struct ncclComm *comm;
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
  int nHeads;
  int headRank;
  int shift;
  int up[NCCL_MAX_DIRECT_ARITY];
  int down[NCCL_MAX_DIRECT_ARITY];
};

#define NCCL_CONN_IDX_P2P_NET 2
#define NCCL_MAX_CONNS 3
struct ncclChannelPeer {
  struct ncclConnector send[NCCL_MAX_CONNS];
  struct ncclConnector recv[NCCL_MAX_CONNS];
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
      uint8_t isUsed:1, redOpArgIsPtr:1, regUsed:1, nWarps:5;
    };
  };
  uint8_t direct;
  uint8_t bid;
  uint8_t nChannels;
  struct {
    uint32_t root:28;
    uint32_t pad_0:2;
    uint32_t connIndex:2;
  };

  const void * sendbuff;
  void * recvbuff;

  size_t count;
  union {
    size_t lastChunkSize;
    // Pivot A2A kernel computes chunk size itself.
    // Instead, it needs the number of bidirectional rings.
    size_t pivotA2ANumBiRings;
  };
  uint64_t redOpArg;
  uint64_t opCount;
};

static_assert((NCCL_WORK_SIZE - alignUp(sizeof(ncclWorkHeader), alignof(ncclWorkElem)))/sizeof(ncclWorkElem) == 4, "Sanity check: NCCL_MAX_WORK_ELEMENTS == 4");
#define NCCL_MAX_WORK_ELEMENTS 1

struct ncclWorkElemP2p {
  struct {
    int32_t peer:30;
    uint32_t connIndex:2;
  };
  union {
    uint16_t flagBits;
    struct {
      enum ncclWorkP2PType p2pType:4;
      uint16_t nWarps:4;
      uint16_t warpStart:4;
      uint16_t ngroups:4;
    };
  };
  uint16_t opCount;
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

#define COLLTRACE_NUM_ITEMS 8192
#endif

struct alignas(16) ncclDevChannel {
  struct ncclDevChannelPeer *peers;
  struct ncclRing ring;
  struct ncclTree tree;
  struct ncclTree binTree;
  struct ncclDirect collTree;
  uint32_t* workFifoDone; // Location of done counter, device writes index+1 of last work processed
};

struct ncclDevComm {
  int rank;
  int nRanks;
  int buffSizes[NCCL_NUM_PROTOCOLS];

  // Operation list for aggregation
  int workFifoDepth;
  struct ncclWork* workFifoHeap; // may be cudaHost or GDR memory

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
  volatile uint32_t *collTraceTail;
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

#endif
