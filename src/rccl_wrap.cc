/*
Copyright (c) 2025 Advanced Micro Devices, Inc. All rights reserved.

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in
all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
THE SOFTWARE.
*/

#include "rccl_common.h"
#include "comm.h"
#include "graph/topo.h"
#include "enqueue.h"

void rcclUpdateCollectiveProtocol(struct ncclComm* comm, size_t const& nBytes, struct ncclTaskColl* info) {
  // Honor user input for protocol choice
  static int userProtocolInput = -2;
  if (userProtocolInput == -2) {
    const char *protoStr = getenv("NCCL_PROTO");
    userProtocolInput = !protoStr ? 0 : 1;
  }

  if(!userProtocolInput && comm->nNodes >= 2 && (info->func == ncclFuncReduceScatter || info->func == ncclFuncAllGather || info->func == ncclFuncAllReduce || info->func == ncclFuncBroadcast)) {
    auto tunableIndex = rcclGetTunableIndex(info->func);
    auto llMin = comm->minMaxLLRange[tunableIndex][NCCL_PROTO_LL][RCCL_PROTOCOL_MIN_IDX];
    auto llMax = comm->minMaxLLRange[tunableIndex][NCCL_PROTO_LL][RCCL_PROTOCOL_MAX_IDX];

    auto ll128Min = comm->minMaxLLRange[tunableIndex][NCCL_PROTO_LL128][RCCL_PROTOCOL_MIN_IDX];
    auto ll128Max = comm->minMaxLLRange[tunableIndex][NCCL_PROTO_LL128][RCCL_PROTOCOL_MAX_IDX];

    // Only override model choices if min/max cutoff points are set in the tuning models
    if ((ll128Max != RCCL_LL_LIMITS_UNDEFINED) || (llMax != RCCL_LL_LIMITS_UNDEFINED)) {
      // Keep it simple unless otherwise required
      info->protocol = NCCL_PROTO_SIMPLE;
      size_t sizePerRank = rcclGetSizePerRank(info->func, nBytes, comm->nRanks);
      if (sizePerRank <= llMax && sizePerRank > llMin) {
        info->protocol = NCCL_PROTO_LL;
      }
#if defined(ENABLE_LL128)
      // When LL128 is performant, the next condition overrides the previous LL choice
      if (comm->topo->ll128Enabled) {
        if (info->func == ncclFuncAllReduce) {
          if(comm->nNodes > 2) {
            ll128Max *= 3.8; // Scale max message size for n > 2 since Tree has special behavior at 2 nodes
          }
          // ll128Max += (log2i(comm->nNodes) - 1) * comm->minMaxLLRange[tunableIndex][NCCL_PROTO_LL128][RCCL_PROTOCOL_FACTOR_IDX];
        }
        if (sizePerRank <= ll128Max && sizePerRank > ll128Min) {
          info->protocol = NCCL_PROTO_LL128;
        }
      }
#endif
    } else if (IsArchMatch(comm->topo->nodes[GPU].nodes[0].gpu.gcn, "gfx942") ||
               IsArchMatch(comm->topo->nodes[GPU].nodes[0].gpu.gcn, "gfx950")) {
      // Warn that model detection for the above listed architectures did not work as expected
      // Add supported archs to this condition as they come
      // Also make sure the tuning_model and model detection are updated for new archs
      static bool failedWarn = false;
      if (!failedWarn) {
        WARN("LL cutoff points not detected for a supported arch %s", comm->topo->nodes[GPU].nodes[0].gpu.gcn);
        failedWarn = true;
      }
    }
  }
}

void rcclUpdateThreadThreshold(struct ncclComm* comm, size_t const& nBytes, struct ncclTaskColl* info, int& threadThreshold) {
  // Honor user input for thread thresholds
  static int userChannelControlInput = -2;
  if (userChannelControlInput == -2) {
    const char *inputStr = getenv("NCCL_THREAD_THRESHOLDS");
    if (!inputStr) {
      inputStr = getenv("NCCL_MAX_NCHANNELS");
    }
    if (!inputStr) {
      inputStr = getenv("NCCL_MIN_NCHANNELS");
    }
    userChannelControlInput = !inputStr ? 0 : 1;
  }

  if(!userChannelControlInput && comm->nNodes >= 2 && (info->func == ncclFuncReduceScatter || info->func == ncclFuncAllGather)) {
    auto tunableIndex = rcclGetTunableIndex(info->func);
    auto tunedThreshold = comm->minMaxLLRange[tunableIndex][info->protocol][RCCL_PROTOCOL_THREAD_THRESHOLD_IDX];
    if(tunedThreshold != RCCL_LL_LIMITS_UNDEFINED) {
      threadThreshold = tunedThreshold * comm->nRanks;
    }
  }
}

extern ncclResult_t getAlgoInfo(
    struct ncclComm* comm, struct ncclTaskColl* task,
    int collNetSupport, int nvlsSupport, int numPipeOps, ncclSimInfo_t* simInfo = NULL
);

ncclResult_t rcclGetAlgoInfo(struct ncclComm* comm, ncclFunc_t coll, uint64_t count, ncclDataType_t dataType,
                             int collNetSupport, int nvlsSupport, int numPipeOps,
                             int* algo, int* protocol, int* maxChannels) {
  RCCL_STATIC_EXPOSE_CHECK();
  struct ncclTaskColl task;
  task.func = coll;
  task.count = count;
  task.datatype = dataType;
  NCCLCHECK(getAlgoInfo(comm, &task, collNetSupport, nvlsSupport, numPipeOps));
  *algo = task.algorithm;
  *protocol = task.protocol;
  *maxChannels = task.nMaxChannels;
  return ncclSuccess;
}

void rcclSetPxn(struct ncclComm* comm,  int& rcclPxnDisable) {
  static int pxnDisable = RCCL_VALUE_UNSET;
  if(pxnDisable == RCCL_VALUE_UNSET) {
    const char *inputStr = getenv("NCCL_PXN_DISABLE");
    const bool archGfx942 = IsArchMatch(comm->topo->nodes[GPU].nodes[0].gpu.gcn, "gfx942");
    const bool archGfx950 = IsArchMatch(comm->topo->nodes[GPU].nodes[0].gpu.gcn, "gfx950");
    if((!archGfx942 && !archGfx950) || inputStr) {
      rcclPxnDisable = pxnDisable = RCCL_VALUE_INVALID;
      return;
    }
    const int ranksThreshold = (archGfx942)? 64 : 32;
    pxnDisable = (comm->nRanks >= ranksThreshold)? 0 : 1;
    INFO(NCCL_INIT, "RCCL PXN set as %s", !pxnDisable? "enabled" : "disabled");
  }
  rcclPxnDisable = pxnDisable;
}

void rcclSetP2pNetChunkSize(struct ncclComm* comm,  int& rcclP2pNetChunkSize) {
  static int p2pNetChunkSize = RCCL_VALUE_UNSET;
  if(p2pNetChunkSize == RCCL_VALUE_UNSET) {
    const char *inputStr = getenv("NCCL_P2P_NET_CHUNKSIZE");
    const bool archGfx942 = IsArchMatch(comm->topo->nodes[GPU].nodes[0].gpu.gcn, "gfx942");
    const bool archGfx950 = IsArchMatch(comm->topo->nodes[GPU].nodes[0].gpu.gcn, "gfx950");
    if((!archGfx942 && !archGfx950) || inputStr) {
      rcclP2pNetChunkSize = p2pNetChunkSize = RCCL_VALUE_INVALID;
      return;
    }

    if(archGfx942)
      p2pNetChunkSize = (comm->nRanks >= 64)? (1 << 19) : (1 << 17);
    else  if(archGfx950)
      p2pNetChunkSize = (comm->nRanks >= 32) ? (1 << 19) : (comm->nRanks >= 16 ? (1 << 18) : (1 << 17));
    else
      WARN("RCCL P2P attempt to set P2P net chunk size for unsupported arch: %s", comm->topo->nodes[GPU].nodes[0].gpu.gcn);
    INFO(NCCL_INIT, "RCCL P2P net chunk size default set to: %d", p2pNetChunkSize);
  }
  rcclP2pNetChunkSize = p2pNetChunkSize;
}

ncclResult_t rcclFuncMaxSendRecvCount(ncclFunc_t func, int nRanks, size_t count, size_t& maxCount) {
  RCCL_STATIC_EXPOSE_CHECK();
  maxCount = ncclFuncMaxSendRecvCount(func, nRanks, count);
  return ncclSuccess;
}

ncclResult_t commSetUnrollFactor(struct ncclComm* comm) {
  hipDeviceProp_t devProp;
  CUDACHECK(hipGetDeviceProperties(&devProp, comm->cudaDev));
  if(IsArchMatch(devProp.gcnArchName, "gfx950")) {
    if(comm->nNodes == 1)
      comm->unroll = NCCL_UNROLL_1;
    else
      comm->unroll = NCCL_UNROLL_2;
  }
  else if(IsArchMatch(devProp.gcnArchName, "gfx908") || ((IsArchMatch(devProp.gcnArchName, "gfx942") && devProp.multiProcessorCount > 80)))
    comm->unroll = NCCL_UNROLL_2;
  else
    comm->unroll = NCCL_UNROLL_4;

  INFO(NCCL_INIT, "RCCL Unroll Factor (pre-set): %d", comm->unroll+1);
  return ncclSuccess;
}
