/*************************************************************************
 * Copyright (c) 2015-2020, NVIDIA CORPORATION. All rights reserved.
 * Modifications Copyright (c) 2019-2021 Advanced Micro Devices, Inc. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

#ifndef NCCL_DEVICE_H_
#define NCCL_DEVICE_H_

#include "nccl.h"
#include "rccl_bfloat16.h"
#include "align.h"
#include <stdint.h>
// [RCCL] Support for clique-based kernels
#include "clique/CliqueCommon.h"
// [/RCCL]

// Convert volatile access to atomic
#if defined(__HIP_PLATFORM_HCC__) || defined(__HCC__) || defined(__HIPCC__)
#define LOAD(VAR) __atomic_load_n((VAR), __ATOMIC_SEQ_CST)
#define STORE(DST, SRC) __atomic_store_n((DST), (SRC), __ATOMIC_SEQ_CST)
#else
#define LOAD(VAR) *(VAR)
#define STORE(DST, SRC) *(DST) = (SRC)
#endif


#define NCCL_NUM_FUNCTIONS 5 // SendRecv not included for now
typedef enum { ncclFuncBroadcast, ncclFuncReduce, ncclFuncAllGather, ncclFuncReduceScatter, ncclFuncAllReduce, ncclFuncSendRecv} ncclFunc_t;
extern const char* ncclFuncStr[NCCL_NUM_FUNCTIONS+1];

#define NCCL_NUM_ALGORITHMS 3 // Tree/Ring/CollNet
#define NCCL_ALGO_TREE 0
#define NCCL_ALGO_RING 1
#define NCCL_ALGO_COLLNET 2
extern const char* ncclAlgoStr[NCCL_NUM_ALGORITHMS];

#define NCCL_NUM_PROTOCOLS 3 // Simple/LL/LL128
#define NCCL_PROTO_LL 0
#define NCCL_PROTO_LL128 1
#define NCCL_PROTO_CLIQUE 1  // [RCCL] Clique takes up same protocol as unused LL128
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

#define NCCL_LL128_LINESIZE 128
#define NCCL_LL128_LINEELEMS (NCCL_LL128_LINESIZE/sizeof(uint64_t))
#define NCCL_LL128_DATAELEMS (NCCL_LL128_LINEELEMS-1)

#define NCCL_LL128_MAX_NTHREADS 256
#define NCCL_LL128_ELEMS_PER_THREAD 120

// Receiving from up to 3 sources is more compute intensive than sending
// to 3 dests. Use 70% for reduce and 30% for bcast.
#define NCCL_LL128_SPLIT(nt) ((nt*7/(10*32))*32)

#define NCCL_LL128_SHMEM_ELEMS_PER_THREAD 2
#define NCCL_LL128_SHMEM_SIZE (NCCL_LL128_SHMEM_ELEMS_PER_THREAD*NCCL_LL128_MAX_NTHREADS)

#define NCCL_DIRECT_GPU 0x01
#define NCCL_DIRECT_NIC 0x10

struct ncclConnInfo {
  // Regular comm mechanism
  char *buffs[NCCL_NUM_PROTOCOLS]; // Local for recv, remote for send
  uint64_t *tail;     // Local for recv, remote for send
  uint64_t *head;     // Local for send, remote for recv

  int direct;         // Direct communication
  int shared;         // Buffers are shared
  void **ptrExchange; // Pointer exchange for direct communication

  int *sizesFifo;     // Sizes fifo from GPU to proxy
  void* *ptrsFifo;      // Buffer fifo from proxy to GPU

  uint64_t step;      // Keep where we are
  uint64_t llLastCleaning;

  // GPU's HDP_MEM_FLUSH_ADDR: HDP Memory Coherency Flush Control. This register
  // allows software to explicitly initiate a flush read to HDP memory. See more
  // descriptions in primitives.h.
  uint32_t* next_hdp_reg;  // Next GPU in ring (for p2p transport use only)
  uint32_t* curr_hdp_reg;  // Current GPU's HDP register
};

struct ncclConnector {
  int connected;
  struct ncclProxyArgs *proxyAppend;
  struct ncclTransportComm* transportComm;
  void* transportResources; // Host-side resources
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
  int* devUserRanks;
};


#define NCCL_MAX_TREE_ARITY 3
struct ncclTree {
  int depth;
  int up;
  int down[NCCL_MAX_TREE_ARITY];
};

struct ncclPeer {
  struct ncclConnector send;
  struct ncclConnector recv;
  struct ncclConnector p2pSend;
  struct ncclConnector p2pRecv;
};

struct ncclDevComm;

#pragma pack(push)  /* push current alignment to stack */
#pragma pack(4)     /* set alignment to 4 bytes boundary */
#define NCCL_MAX_WORK_ELEMENTS 2
#define NCCL_MAX_GROUPS (NCCL_MAX_WORK_ELEMENTS*2)

/* ncclWork is to be a power of two, currently 8x64 bytes, */
/* to make sure reads to host from the CUDA kernel are aligned. */
/* Make sure to adjust padding at the end of ncclWorkElem. */
struct ncclWorkElem {
  // Header
  struct ncclDevComm* comm;
  uint16_t nThreads;
  uint16_t funcIndex;
  uint16_t index;
  uint16_t active;

  const void * sendbuff;
  void * recvbuff;

  uint64_t opCount;
  // Op-specific fields.
  union {
    struct {
      size_t count;
      size_t lastChunkSize;
      uint32_t root;
      uint8_t bid;
      uint8_t nChannels;
    } coll;
    struct {
      size_t sendCount;
      size_t recvCount;
      int32_t delta;
      uint16_t nThreads;
    } p2p;
    // [RCCL] Clique-based arguments
    //        NOTE: Follows same field structure as coll
    //              because nChannels is accessed from "coll" struct.
    struct {
      size_t count;
      cliqueDevicePtrs_t* ptrs;
      uint32_t unused;
      uint8_t bid;
      uint8_t nChannels;
    } clique;
    // [/RCCL]
    uint64_t align[3];
  };
};
struct ncclWork {
  struct ncclWorkElem elems[NCCL_MAX_WORK_ELEMENTS];
};
static_assert(sizeof(struct ncclWorkElem) == (0x10*sizeof(int)), "ncclWorkElem must have a pow2 size");

struct ncclChannel {
  union {
    struct {
      struct ncclRing ring;
      struct ncclTree tree;
      struct ncclTree collTree;

      int id;

      // Communication structures
      struct ncclPeer* peers;
      struct ncclPeer* devPeers;

      // Operation list for aggregation
      struct ncclWork* workFifo;
      int workCount;
      uint64_t workFifoTail; // Only used by CPU

#ifdef ENABLE_PROFILING
      struct timeval tvs;
      uint64_t sizes;
      int active_req;
      uint64_t send_byte;
      uint64_t recv_byte;
      float bw_cumulative;
      int bw_count;
#endif
    };
    int data[0x80];
  };
};
static_assert(sizeof(struct ncclChannel) == 0x80*sizeof(int), "ncclChannel must have a pow2 size");
#pragma pack(pop)   /* restore original alignment from stack */

#ifdef ENABLE_PROFILING
struct ncclProf {
  union {
    struct {
      uint64_t total_cycle;
      uint64_t wait_cycle[MAXCHANNELS];      // total wait cycle
      uint64_t wait_recv_cycle[MAXCHANNELS]; // recv wait cycle
      // primtive cycles
      uint64_t send_cycle;
      uint64_t directSend_cycle;
      uint64_t recv_cycle;
      uint64_t directRecv_cycle;
      uint64_t copySend_cycle;
      uint64_t directCopySend_cycle;
      uint64_t recvCopySend_cycle;
      uint64_t directRecvCopySend_cycle;
      uint64_t recvReduceCopy_cycle;
      uint64_t recvReduceSend_cycle;
      uint64_t recvReduceCopySend_cycle;
      uint64_t directRecvReduceCopySend_cycle;
      // primitive bytes
      uint64_t send_byte;
      uint64_t directSend_byte;
      uint64_t recv_byte;
      uint64_t directRecv_byte;
      uint64_t copySend_byte;
      uint64_t directCopySend_byte;
      uint64_t recvCopySend_byte;
      uint64_t directRecvCopySend_byte;
      uint64_t recvReduceCopy_byte;
      uint64_t recvReduceSend_byte;
      uint64_t recvReduceCopySend_byte;
      uint64_t directRecvReduceCopySend_byte;
    };
    int data[0x80];
  };
};
#endif

#ifdef ENABLE_COLLTRACE
typedef enum {
  ncclCollTraceNotReady,
  ncclCollTraceKernelLaunchType,
  ncclCollTraceCollEndType,
  ncclCollTraceAbortType,
  ncclCollTraceDataType
} ncclCollTraceDataType_t;

struct ncclCollTrace {
  uint8_t type;
  uint8_t bid;
  int16_t funcIndex;
  uint32_t data_0;
  uint64_t timeStamp;
  uint64_t opCount;
  union {
    uint64_t data_1;
    struct {
      uint16_t nThreads;
      uint8_t bid;
      uint8_t nChannels;
    } coll;
    struct {
      uint16_t nThreads;
      uint16_t delta;
    } p2p;
  };
};
static_assert(sizeof(struct ncclCollTrace) == 8*sizeof(int), "ncclCollTrace must have a pow2 size");

#define COLLTRACE_NUM_ITEMS 8192
#endif

struct ncclDevComm {
  int rank;
  int nRanks;
  int buffSizes[NCCL_NUM_PROTOCOLS];

  // Flag to ask NCCL kernels to abort
  volatile uint32_t *abortFlag;

  // Channels, device side
  struct ncclChannel* channels;

  // Flags for enable P2P NET
  uint32_t *p2pNet;

#ifdef ENABLE_PROFILING
  // Profiling counters
  struct ncclProf* devProf;
#endif

#ifdef ENABLE_COLLTRACE
  struct ncclCollTrace* collTrace;
  uint32_t collTraceHead, *collTraceTail;
  pthread_t collTraceThread;
  bool collTraceExit;
#endif
};

#endif
