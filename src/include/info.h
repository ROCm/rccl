/*************************************************************************
 * Copyright (c) 2019-2020, NVIDIA CORPORATION. All rights reserved.
 * Modifications Copyright (c) 2019-2020 Advanced Micro Devices, Inc. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

#ifndef NCCL_INFO_H_
#define NCCL_INFO_H_

#include "nccl.h"
#include "devcomm.h"

typedef enum {
  ncclPatternRing,
  ncclPatternRingTwice,
  ncclPatternPipelineFrom,
  ncclPatternPipelineTo,
  ncclPatternTreeUp,
  ncclPatternTreeDown,
  ncclPatternTreeUpDown,
  ncclPatternCollTreeUp,
  ncclPatternCollTreeDown,
  ncclPatternAll
} ncclPattern_t;

// Used to pass NCCL call information between functions
struct ncclInfo {
  ncclFunc_t coll;
  const char* opName;
  // NCCL Coll Args
  const void* sendbuff;
  void* recvbuff;
  size_t count;
  ncclDataType_t datatype;
  ncclRedOp_t op;
  int root;
  ncclComm_t comm;
  hipStream_t stream;
  // Algorithm details
  int chunkSteps;
  int sliceSteps;
  // For alltoallv
  const size_t *sendcounts;
  const size_t *sdispls;
  const size_t *recvcounts;
  const size_t *rdispls;
  // Computed later
  int algorithm;
  int protocol;
  ncclPattern_t pattern;
  int nChannels;
  int nThreads;
  size_t nBytes;
  int nstepsPerLoop;
  int nchunksPerLoop;
  ssize_t sendbytes;
  ssize_t recvbytes;
  uint32_t delta;
  int channelId;
};

#endif
