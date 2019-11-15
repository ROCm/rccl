/*************************************************************************
 * Copyright (c) 2015-2019, NVIDIA CORPORATION. All rights reserved.
 * Modifications Copyright (c) 2019 Advanced Micro Devices, Inc. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

#ifndef NCCL_DEVICE_H_
#define NCCL_DEVICE_H_

#include "nccl.h"
#include "rccl_bfloat16.h"
#include <stdint.h>

// Convert volatile access to atomic
#if defined(__HIP_PLATFORM_HCC__) || defined(__HCC__) || defined(__HIPCC__)
#define LOAD(VAR) __atomic_load_n((VAR), __ATOMIC_SEQ_CST)
#define STORE(DST, SRC) __atomic_store_n((DST), (SRC), __ATOMIC_SEQ_CST)
#else
#define LOAD(VAR) *(VAR)
#define STORE(DST, SRC) *(DST) = (SRC)
#endif

#define NCCL_MAX_OPS 2048
#define NCCL_STEPS 8

typedef enum { ncclCollBroadcast, ncclCollReduce, ncclCollAllGather, ncclCollReduceScatter, ncclCollAllReduce, ncclCollCount } ncclColl_t;

#define DIVUP(x, y) \
    (((x)+(y)-1)/(y))
#define ROUNDUP(x, y) \
    (DIVUP((x), (y))*(y))

#define ALIGN_SIZE(size, align) \
  size = ((size + (align) - 1) / (align)) * (align);

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

#define MAXTHREADS 256
#define NCCL_LL_MAX_NTHREADS MAXTHREADS
#define NUM_LINES_PER_THREAD 8
#define NCCL_LL_SLICE_LINES (NUM_LINES_PER_THREAD*NCCL_LL_MAX_NTHREADS)
#define NCCL_LL_BUFF_LINES (NCCL_LL_SLICE_LINES*NCCL_STEPS)
#define NCCL_LL_BUFF_SIZE (NCCL_LL_BUFF_LINES*sizeof(union ncclLLFifoLine))
#ifdef DEBUG_LL
#define NCCL_LL_CLEAN_MASK 0x00000ff8
#define NCCL_LL_FLAG_MAX   0x00001000
#define NCCL_LL_FLAG(a) ((uint32_t)(a % NCCL_LL_FLAG_MAX))
#else
#define NCCL_LL_CLEAN_MASK 0x7ffffff8
#define NCCL_LL_FLAG(a) ((uint32_t)(a))
#endif
// Make sure the clean mask will last for at least NCCL_NSTEPS
static_assert(NCCL_LL_CLEAN_MASK % NCCL_STEPS == 0, "Invalid NCCL_LL_CLEAN_MASK value");

struct ncclConnInfo {
  // Regular comm mechanism
  char *buff;         // Local for recv, remote for send
  uint64_t *tail;     // Local for recv, remote for send
  uint64_t *head;     // Local for send, remote for recv
  uint64_t *opCountLoc; // opCount of local rank
  uint64_t *opCountRem; // opCount of remote rank

  int direct;         // Direct communication
  void **ptrExchange; // Pointer exchange for direct communication

  int *fifo;          // Size fifo for proxy

  uint64_t step;      // Keep where we are

  // Low latency mechanism
  union ncclLLFifoLine *llBuff; // Local for recv, remote for send
  uint64_t llLastCleaning;

  // GPU's HDP_MEM_FLUSH_ADDR: HDP Memory Coherency Flush Control. This register
  // allows software to explicitly initiate a flush read to HDP memory. See more
  // descriptions in primitives.h.
  uint32_t* next_hdp_reg;  // Next GPU in ring (for p2p transport use only)
  uint32_t* curr_hdp_reg;  // Curr GPU in ring (for rdma transport use only)
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
};

struct ncclDevComm;

#pragma pack(push)  /* push current alignment to stack */
#pragma pack(4)     /* set alignment to 4 bytes boundary */
/* CollectiveArgs + ncclColl are to be a power of two, currently 64 bytes, */
/* to make sure reads to host from the CUDA kernel are aligned. */
/* Make sure to adjust padding at the end of ncclColl. */
struct CollectiveArgs {
  struct ncclDevComm* comm;
  uint64_t opCount;

  // local and remote input, output, and buffer
  const void * ThisInput;
  void * ThisOutput;

  // general parameters
  size_t N;
  uint32_t root;
  uint8_t bid;
  uint8_t nChannels;
  uint16_t nThreads;

  int lastChunkSize;
};
struct ncclColl {
  union {
    struct {
      struct CollectiveArgs args;
      uint16_t funcIndex;
      uint16_t nextIndex;
      uint8_t  active;
    };
    int data[0x10];
  };
};
static_assert(sizeof(struct ncclColl) == (0x10*sizeof(int)), "ncclColl must have a pow2 size");

struct ncclChannel {
  union {
    struct {
      struct ncclRing ring;
      struct ncclTree tree;

      int id;
      int nthreads;
      int buffSize;

      // Communication structures
      struct ncclPeer* peers;
      struct ncclPeer* devPeers;

      // Operation list for aggregation
      struct ncclColl* collectives;
      struct ncclColl* devCollectives;
      int collStart;
      int collCount;
      int collFifoHead; // Only used by GPU
      int collFifoTail; // Only used by CPU

      uint32_t* abortCount;
    };
    int data[0x80];
  };
};
static_assert(sizeof(struct ncclChannel) == 0x80*sizeof(int), "ncclChannel must have a pow2 size");
#pragma pack(pop)   /* restore original alignment from stack */

#define MAXCHANNELS 16

#ifdef ENABLE_PROFILING
struct ncclProf {
  union {
    struct {
      uint64_t total_cycle;
      uint64_t wait_send_cycle[MAXCHANNELS];
      uint64_t wait_recv_cycle[MAXCHANNELS];
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

typedef enum {
  ncclDevSuccess,
  ncclDevAssertedMismatch,
  ncclDevSuspectedMismatch
} ncclDevError_t;

struct ncclDevComm {
  int rank;
  int nRanks;

  // Flag to ask NCCL kernels to abort
  volatile uint32_t *abortFlag;
  volatile ncclDevError_t *fatalDevError;

  // Channels, device side
  struct ncclChannel* channels;

#ifdef ENABLE_PROFILING
  // Profiling counters
  struct ncclProf* devProf;
#endif
};

#endif
