/*************************************************************************
 * Copyright (c) 2023, NVIDIA CORPORATION. All rights reserved.
 * Copyright (c) 2023, Meta Platforms, Inc. and affiliates.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

#ifndef NCCL_TUNER_H_
#define NCCL_TUNER_H_

#include "nccl.h"

typedef enum {NCCL_LOG_NONE=0, NCCL_LOG_VERSION=1, NCCL_LOG_WARN=2, NCCL_LOG_INFO=3, NCCL_LOG_ABORT=4, NCCL_LOG_TRACE=5} ncclDebugLogLevel;
typedef enum {NCCL_INIT=1, NCCL_COLL=2, NCCL_P2P=4, NCCL_SHM=8, NCCL_NET=16, NCCL_GRAPH=32, NCCL_TUNING=64, NCCL_ENV=128, NCCL_ALLOC=256, NCCL_CALL=512, NCCL_PROXY=1024, NCCL_NVLS=2048, NCCL_ALL=~0} ncclDebugLogSubSys;

typedef void (*ncclDebugLogger_t)(ncclDebugLogLevel level, unsigned long flags, const char *file, int line, const char *fmt, ...);

#define NCCL_NUM_FUNCTIONS 5 // Send/Recv not included for now
typedef enum { ncclFuncBroadcast, ncclFuncReduce, ncclFuncAllGather, ncclFuncReduceScatter, ncclFuncAllReduce, ncclFuncSendRecv, ncclFuncSend, ncclFuncRecv, ncclNumFuncs} ncclFunc_t;

#define NCCL_NUM_ALGORITHMS 6 // Tree/Ring/CollNet*
#define NCCL_ALGO_UNDEF -1
#define NCCL_ALGO_TREE 0
#define NCCL_ALGO_RING 1
#define NCCL_ALGO_COLLNET_DIRECT 2
#define NCCL_ALGO_COLLNET_CHAIN 3
#define NCCL_ALGO_NVLS 4
#define NCCL_ALGO_NVLS_TREE 5

#define NCCL_NUM_PROTOCOLS 3 // Simple/LL/LL128
#define NCCL_PROTO_UNDEF -1
#define NCCL_PROTO_LL 0
#define NCCL_PROTO_LL128 1
#define NCCL_PROTO_SIMPLE 2

#define HOPPER_COMPCAP_IDX 2
// NVLink, PCI, Network
#define NCCL_HW_NVLINK 0
#define NCCL_HW_PCI 1
#define NCCL_HW_NET 2

static long log2i(long n) {
 long l = 0;
 while (n>>=1) l++;
 return l;
}


// Latencies in us, Bandwidths in GB/s
// Tree { LL, LL128, Simple } , Ring { LL, LL128, Simple }
static const float baseLat  [NCCL_NUM_ALGORITHMS][NCCL_NUM_PROTOCOLS] = { 
       { 12.0, 12.0, 17.0 }, { 12.0, 12.0, 17.0 },   // Tree, Ring
       { 12.0, 12.0, 17.0 }, { 12.0, 12.0, 17.0 },   // Collnet Direct, Chain
       {    0,    0,    0 }, {    0,    0,    0 }};  // NVLS, NVLS Tree

struct tuningModel {
  float hwLat[3][NCCL_NUM_ALGORITHMS][NCCL_NUM_PROTOCOLS];
  float bwRatio[2][NCCL_NUM_ALGORITHMS][NCCL_NUM_PROTOCOLS];
  float treeCorrectionFactor[NCCL_NUM_PROTOCOLS][27];
  float ringCorrectionFactor[NCCL_NUM_PROTOCOLS][27];
};

static struct tuningModel tuning_model = {
  {
    /* NVLINK */
    { /* Tree (LL/LL128/Simple)*/ { 0.8, 0.0, 2.5 }, /* Ring (LL/LL128/Simple)*/ { 0.8, 0.0, 3.6 }, /* CollNetDirect (Simple)*/ { 0.0, 0.0, 0.8 }, /* CollNetChain (Simple)*/ { 0.0, 0.0, 0.0 }, /* NVLS */ { 0, 0, 0 }, /* NVLS Tree */ { 0, 0, 0 } },
    /* PCI */
    { /* Tree (LL/LL128/Simple)*/ { 2.2, 2.2, 5.7 }, /* Ring (LL/LL128/Simple)*/ { 2.2, 2.2, 5.7 }, /* CollNetDirect (Simple)*/ { 0.0, 0.0, 5.7 }, /* CollNetChain (Simple)*/ { 0.0, 0.0, 5.7 }, /* NVLS */ { 0, 0, 0 }, /* NVLS Tree */ { 0, 0, 0 } },
    /* NET */
    { /* Tree (LL/LL128/Simple)*/ { 12.5, 0.0, 22.4 }, /* Ring (LL/LL128/Simple)*/ { 9.5, 0.0, 19.8 }, /* CollNetDirect (Simple)*/ { 0.0, 0.0, 12.5 }, /* CollNetChain (Simple)*/ { 0.0, 0.0, 0.0 }, /* NVLS */ { 0, 0, 0 }, /* NVLS Tree */ { 0, 0, 0 } },
  },

  {
    /* 2 nodes */
    { /* Tree (LL/LL128/Simple)*/ { 0.41, 0.00, 1.00 }, /* Ring (LL/LL128/Simple)*/ { 0.41, 0.00, 1.00 }, /* CollNetDirect (Simple)*/ { 0.00, 0.00, 1.00 }, /* CollNetChain (Simple)*/ { 0.00, 0.00, 1.00 }, /* NVLS */ { 0, 0, 0 }, /* NVLS Tree */ { 0, 0, 0 } },
    /* more than 2 nodes */
    { /* Tree (LL/LL128/Simple)*/ { 0.41, 0.00, 0.86 }, /* Ring (LL/LL128/Simple)*/ { 0.41, 0.00, 1.00 }, /* CollNetDirect (Simple)*/ { 0.00, 0.00, 1.00 }, /* CollNetChain (Simple)*/ { 0.00, 0.00, 1.00 }, /* NVLS */ { 0, 0, 0 }, /* NVLS Tree */ { 0, 0, 0 } },
  },

  {
    { 0.1, 0.1, 0.1, 0.1, 0.1, 1.0, 1.0, 0.8, 0.1, 0.4, 0.5, 1.0, 0.6, 0.4, 0.6, 0.1, 0.3, 0.4, 0.4, 0.3, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, },
    { 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, },
    { 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 1.0, 1.0, 1.0, 0.4, 1.0, 1.0, 1.0, 0.2, 0.7, 1.0, 1.0, 1.0, 0.8, 0.7, 0.7, 0.8, 0.8, 0.8, 0.9, },
  },

  {
    { 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 1.0, 0.1, 0.2, 0.2, 0.1, 0.5, 0.8, 1.0, 0.2, 0.4, 0.5, 0.4, 0.4, 0.3, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, },
    { 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, },
    { 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.7, 0.1, 0.1, 0.1, 0.1, 0.1, 1.0, 1.0, 1.0, 0.9, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, },
  },
};

float latencies[NCCL_NUM_FUNCTIONS][NCCL_NUM_ALGORITHMS][NCCL_NUM_PROTOCOLS];
float bandwidths[NCCL_NUM_FUNCTIONS][NCCL_NUM_ALGORITHMS][NCCL_NUM_PROTOCOLS];

// API to be implemented by external tuner
typedef struct {
  // Name of the tuner
  const char* name;

  // Initializes tuner states.
  // nRanks: number of ranks in current communicator. Each communicator initialize its own tuner.
  // nNodes: number of nodes in current communicator.
  // logFunction: a logFunction can be useful to integrate logging together with NCCL core.
  ncclResult_t (*init)(size_t nRanks, size_t nNodes, ncclDebugLogger_t logFunction);

  // Gets info (algo, protocol, number of ctas and threads) for a given collective.
  // Inputs:
  //   - collType: collective type , e.g., allreduce, allgather…
  //   - nBytes: collective size in bytes
  //   - collNetSupport: whether collnet supports this type
  //   - nvlsSupport: whether nvlink sharp supports this time
  //   - numPipeOps: number of operations in the group
  //
  // Outputs:
  //   - algorithm: selected algorithm to be used for the given collective
  //   - protocol: selected protocol to be used for the given collective
  //   - nChannels: number of channels (hence SMs) to be used.
  //
  // If getCollInfo() does not return ncclSuccess, NCCL will fall back to the
  // default tuning for the given collective.
  // Also, the plugin is allowed to not set any output, or set only the
  // algorithm and protocol, but not only the algorithm or only the protocol.
  // Unset fields will be set automatically by NCCL.
  ncclResult_t (*getCollInfo)(ncclFunc_t collType, size_t nBytes,
                              int collNetSupport, int nvlsSupport, int numPipeOps,
                              int *algorithm, int *protocol, int* nChannels);

  // Terminates the plugin and cleans up any resources that the plugin allocated.
  ncclResult_t (*destroy)();
} ncclTuner_v1_t;

typedef ncclTuner_v1_t ncclTuner_t;

#define NCCL_TUNER_PLUGIN_SYMBOL "ncclTunerPlugin_v1"

#endif
