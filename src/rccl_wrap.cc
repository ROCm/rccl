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
#include "rocm_smi/rocm_smi.h"
#include <algorithm>
// Use this param to experiment pipelining new data types besides bfloat16
// Make sure you generate the device code with the new data type (i.e. in generate.py)
RCCL_PARAM(PipelineAllDTypes, "PIPELINE_ALL_DATA_TYPES", 0);

// Use this to assess impact of pipelining on performance.
// Otherwise, it is automatically set for certain archs, datatypes and reduction collectives
RCCL_PARAM(disableReduceCopyPipelining, "DISABLE_REDUCE_COPY_PIPELINING", 0);
RCCL_PARAM(DirectAllGatherThreshold, "DIRECT_ALLGATHER_THRESHOLD", 75497472);

void rcclUpdateCollectiveProtocol(struct ncclComm* comm, size_t const& nBytes, struct ncclTaskColl* info) {
  // Honor user input for protocol choice
  static int userProtocolInput = -2;
  if (userProtocolInput == -2) {
    const char *protoStr = getenv("NCCL_PROTO");
    userProtocolInput = !protoStr ? 0 : 1;
  }
  if (!userProtocolInput && IsArchMatch(comm->topo->nodes[GPU].nodes[0].gpu.gcn, "gfx950") && comm->nNodes == 1 && (info->func == ncclFuncAllGather) && nBytes <= 524288) {
    // Change LL protocol threshold
    info->protocol = NCCL_PROTO_LL;

  } else if(!userProtocolInput && comm->nNodes >= 2 && (info->func == ncclFuncReduceScatter || info->func == ncclFuncAllGather || info->func == ncclFuncAllReduce || info->func == ncclFuncBroadcast || info->func == ncclFuncReduce)) {
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

ncclResult_t rcclGetAlgoProtoIndex(const char *envStr, const char* algoProtoString[], int nEntries, int& result) {
  if(envStr) {
    for (int i = 0; i < nEntries; ++i) {
      if (strcasecmp(envStr, algoProtoString[i]) == 0) {
        result = i;
        return ncclSuccess;
      }
    }
    static bool failedProtoWarn = false;
    if (!failedProtoWarn) {
      WARN("Invalid algo or protocol string passed %s", envStr);
      failedProtoWarn = true;
      return ncclInvalidUsage;
    }
  }
  return ncclInvalidUsage;
}

extern int64_t ncclParamMinNchannels();
extern int64_t ncclParamMaxNchannels();
RCCL_PARAM(ChannelTuningEnable, "CHANNEL_TUNING_ENABLE", 1);

ncclResult_t rcclOverrideChannels(struct ncclComm* comm, ncclFunc_t coll, size_t nBytes, int& nc){
  if(comm->nNodes < 2 || !rcclParamChannelTuningEnable()){
    INFO(NCCL_TUNING, "RCCL Channel Tuning not applied");
    return ncclSuccess;
  }

  auto tunableIndex = rcclGetTunableIndex(coll);
  if(tunableIndex == RCCL_UNSUPPORTED_TUNABLE){
    INFO(NCCL_TUNING, "tunableIndex:%i not supported", tunableIndex);
    return ncclSuccess;
  }

  int minCTAs = comm->config.minCTAs;
  int maxCTAs = comm->config.maxCTAs;
  int minNChannels = ncclParamMinNchannels();
  int maxNChannels = std::max(comm->nChannels, static_cast<int>(ncclParamMaxNchannels()));
  size_t bytesPerRank = divUp(nBytes, comm->nRanks);

  for(int channelCountIndex = 0; channelCountIndex < RCCL_CHANNELS_TUNABLE_ENTRIES; ++channelCountIndex){    
    size_t minByteThreshold = comm->minMaxChannelThresholds[tunableIndex][channelCountIndex][0];
    size_t maxByteThreshold = comm->minMaxChannelThresholds[tunableIndex][channelCountIndex][1];
    INFO(NCCL_TUNING, "nBytes:%lu bytesPerRank:%lu minByteThreshold:%lu maxByteThreshold:%lu  NCCL_MIN_NCHANNELS:%i or NCCL_MAX_NCHANNELS:%i minCTAs:%i maxCTAs:%i", nBytes, bytesPerRank, minByteThreshold, maxByteThreshold, minNChannels, maxNChannels, minCTAs, maxCTAs);
    if(minByteThreshold == CHAN_THRESHOLDS_UNDEFINED || maxByteThreshold == CHAN_THRESHOLDS_UNDEFINED) {
      INFO(NCCL_TUNING, "RCCL tuning model does not define threshold for coll:%i and nbytes:%lu", coll, nBytes);
      break; // Skip undefined thresholds
    }
    
    if(bytesPerRank > minByteThreshold && bytesPerRank <= maxByteThreshold){
      int channelCount = comm->minMaxChannelThresholds[tunableIndex][channelCountIndex][2];

      //honor user's min/max channels defined through NCCL_MIN_NCHANNELS and NCCL_MAX_NCHANNELS
      if(channelCount >= minNChannels && channelCount <= maxNChannels && channelCount >= minCTAs && channelCount <= maxCTAs){
        nc = comm->minMaxChannelThresholds[tunableIndex][channelCountIndex][2];
        INFO(NCCL_TUNING, "RCCL tuning model overrides nchannels to %i, channels may be decreased further due to MinTrafficPerchannel thresholds", channelCount);
      }
      else{
        INFO(NCCL_TUNING, "RCCL tuning model cannot override nchannels to %i due to conflicting NCCL_MIN_NCHANNELS:%i or NCCL_MAX_NCHANNELS:%i minCTAs:%i maxCTAs:%i", channelCount, minNChannels, maxNChannels, minCTAs, maxCTAs);
      }

      break;
    }

  }
  return ncclSuccess;
}

ncclResult_t rcclOverrideProtocol(const char* ncclProtoStr[], float table[][NCCL_NUM_PROTOCOLS], struct ncclTaskColl* info) {
  static const char* protoOverrideEnv = ncclGetEnv("RCCL_OVERRIDE_PROTO");
  static bool validInput = true;
  if (!validInput) return ncclInvalidUsage;

  if (protoOverrideEnv) {
    static int protoVal = NCCL_PROTO_UNDEF;
    if (protoVal == NCCL_PROTO_UNDEF) {
      if (rcclGetAlgoProtoIndex(protoOverrideEnv, ncclProtoStr, NCCL_NUM_PROTOCOLS, protoVal) != ncclSuccess) {
        validInput = false;
        return ncclInvalidUsage;
      }
    }
    if (protoVal > NCCL_PROTO_UNDEF) {
      if (table[info->algorithm][protoVal] == NCCL_ALGO_PROTO_IGNORE) {
        WARN("Failed to force unsupported protocol %s for function %s with datatype %s", protoOverrideEnv, ncclFuncToString(info->func), ncclDatatypeToString(info->datatype));
        return ncclInternalError;
      } else {
        info->protocol = protoVal;
      }
    }
  }
  return ncclSuccess;
}

ncclResult_t rcclOverrideAlgorithm(const char* ncclAlgoStr[], float table[][NCCL_NUM_PROTOCOLS], struct ncclTaskColl* info) {
  static const char* algoOverrideEnv = ncclGetEnv("RCCL_OVERRIDE_ALGO");
  static bool validInput = true;
  if (!validInput) return ncclInvalidUsage;

  if (algoOverrideEnv) {
    static int algoVal = NCCL_ALGO_UNDEF;
    if (algoVal == NCCL_ALGO_UNDEF) {
      if (rcclGetAlgoProtoIndex(algoOverrideEnv, ncclAlgoStr, NCCL_NUM_ALGORITHMS, algoVal) != ncclSuccess) {
        validInput = false;
        return ncclInvalidUsage;
      }
    }
    if (algoVal > NCCL_ALGO_UNDEF) {
      if (table[algoVal][info->protocol] == NCCL_ALGO_PROTO_IGNORE) {
        WARN("Failed to force unsupported algorithm %s for function %s with datatype %s", algoOverrideEnv, ncclFuncToString(info->func), ncclDatatypeToString(info->datatype));
        return ncclInternalError;
      } else {
        info->algorithm = algoVal;
      }
    }
  }
  return ncclSuccess;
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

void rcclSetPipelining(struct ncclComm* comm, size_t const& nBytes, struct ncclTaskColl* info) {
  info->pipeline = 0; // Default to no pipelining
  if (rcclParamdisableReduceCopyPipelining()) {
    return;
  }
  const bool dtypeOK = (info->datatype == ncclBfloat16) || rcclParamPipelineAllDTypes();

  if (IsArchMatch(comm->topo->nodes[GPU].nodes[0].gpu.gcn, "gfx950") && dtypeOK) {
    if (comm->nNodes > 1) {
      switch (info->func) {
        case ncclFuncAllReduce:
        case ncclFuncReduceScatter:
        case ncclFuncReduce:
          // Enable for multi-node
          info->pipeline = 1;
          break;
        default:
          break;
      }
    }
    return;
  }

  if (IsArchMatch(comm->topo->nodes[GPU].nodes[0].gpu.gcn, "gfx942") && dtypeOK) {
    switch (info->func) {
      // For multi-node case, we check if the number of bytes (`nBytes`) satisfies
      // the Bf16 Limit Equation for bf16 all_reduce on MI300:
      // 512MB × 2^(log2[nNodes] - 1), nNodes > 1
      // The above equation is derived from the tuning results of the bf16 all_reduce on MI300.
      case ncclFuncAllReduce:
        if ( comm->nNodes == 1 ||
             ((comm->nNodes > 1) &&
               nBytes <= (1ULL << 29 /*512MB*/) * (1ULL << (log2i(comm->nNodes) - 1))) ) {
          info->pipeline = 1;
        }
        break;

      case ncclFuncReduceScatter:
      case ncclFuncReduce:
        info->pipeline = 1;
        break;

      default:
        break;
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
  int nRanks;
  NCCLCHECK(ncclCommCount(comm, &nRanks));
  size_t msgSize = count * ncclTypeSize(dataType) * nRanks;
  if (coll == ncclFuncAllGather && rcclUseAllGatherDirect(comm, msgSize)) {
    *algo = rcclAddonAlgos_t::RCCL_DIRECT_ALLGATHER;
    *protocol = NCCL_PROTO_SIMPLE; // TODO: consider LL for small messages
    *maxChannels = comm->nChannels;
    return ncclSuccess;
  }
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

ncclResult_t rcclGetAlgoName(int algo, const char** algoName) {
  if (algo < 0 || algo >= RCCL_ALGO_COUNT) {
    WARN("Invalid algorithm value: %d", algo);
    return ncclInvalidArgument;
  }
  if(algo >= NCCL_NUM_ALGORITHMS) {
    switch(algo) {
      case rcclAddonAlgos_t::RCCL_DIRECT_ALLGATHER:
        *algoName = "Direct";
        break;
      case rcclAddonAlgos_t::RCCL_MSCCL:
        *algoName = "MSCCL";
        break;
      case rcclAddonAlgos_t::RCCL_MSCCLPP:
        *algoName = "MSCCLPP";
        break;
      default:
        WARN("Invalid algorithm value: %d", algo);
        return ncclInvalidArgument;
    }
    return ncclSuccess;
  }
  *algoName = ncclAlgoToString(algo);
  return ncclSuccess;
}

ncclResult_t rcclGetProtocolName(int protocol, const char** protocolName) {
  if (protocol < 0 || protocol >= NCCL_NUM_PROTOCOLS) {
    WARN("Invalid protocol value: %d", protocol);
    return ncclInvalidArgument;
  }
  *protocolName = ncclProtoToString(protocol);
  return ncclSuccess;
}

bool rcclUseAllGatherDirect(struct ncclComm* comm, size_t& msgSize) {
  size_t threshold = rcclParamDirectAllGatherThreshold();

  if (IsArchMatch(comm->topo->nodes[GPU].nodes[0].gpu.gcn, "gfx950")) {
     if (comm->nNodes == 1 && threshold != -1) {
        threshold = 8388608;
     } else if (comm->nNodes < 64 && threshold != -1) {
        threshold = comm->nNodes * 2097152;
     }
  } else if (IsArchMatch(comm->topo->nodes[GPU].nodes[0].gpu.gcn, "gfx942")) {
	threshold = 4194304;	
  }

  comm->enableCustColl = IsArchMatch(comm->topo->nodes[GPU].nodes[0].gpu.gcn, "gfx950") || IsArchMatch(comm->topo->nodes[GPU].nodes[0].gpu.gcn, "gfx942");

  int rankMultiple = comm->nRanks % 8;
  
  //return (comm->enableCustColl && (comm->nNodes > 1) && (msgSize <= threshold) && (threshold != -1))
  return (comm->enableCustColl && (msgSize <= threshold) && (threshold != -1) && !rankMultiple)
    ;
}

void rcclSetPxn(struct ncclComm* comm,  int& rcclPxnDisable) {
  static int pxnDisable = RCCL_VALUE_UNSET;
  comm->enableCustColl = false;
  if(pxnDisable == RCCL_VALUE_UNSET) {
    const char *inputStr = getenv("NCCL_PXN_DISABLE");
    const bool archGfx942 = IsArchMatch(comm->topo->nodes[GPU].nodes[0].gpu.gcn, "gfx942");
    const bool archGfx950 = IsArchMatch(comm->topo->nodes[GPU].nodes[0].gpu.gcn, "gfx950");
    comm->enableCustColl = (archGfx942 || archGfx950) && (inputStr && !atoi(inputStr));

    if((!archGfx942 && !archGfx950) || inputStr) {
      rcclPxnDisable = pxnDisable = RCCL_VALUE_INVALID;
      return;
    }
    const int ranksThreshold = (archGfx942)? 64 : 32;
    pxnDisable = (comm->nRanks >= ranksThreshold)? 0 : 1;
    INFO(NCCL_INIT, "RCCL PXN set as %s", !pxnDisable? "enabled" : "disabled");
  }
  rcclPxnDisable = pxnDisable;
  comm->enableCustColl = !pxnDisable;
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

  INFO(NCCL_INIT, "RCCL Unroll Factor (pre-set): %d", (int) (pow(2.0, (double)comm->unroll)));
  return ncclSuccess;
}

std::string trimString(const std::string& s) {
  int sz = s.size();
  int b = 0;
  int e = sz - 1;
  while (b < sz && isspace(s[b])) {
    b++;
  }
  if (b >= sz) {
    return "";
  }

  while (e >= b && e < sz && isspace(s[e])) {
    e--;
  }
  if (b > e) {
    return "";
  }
  return s.substr(b, e - b + 1);
}

std::vector<std::string> splitString(const std::string& s, char delimiter) {
  std::vector<std::string> tokens;
  std::stringstream ss(s);
  std::string token;

  while (std::getline(ss, token, delimiter)) {
    tokens.push_back(trimString(token));
  }
  return tokens;
}

int parseFirmwareVersionImpl() {
  uint64_t fw_version = -1;

  // using rocm-smi APIs for now to query MEC FW version
  // will switch to amd-smi APIs soon
  rsmi_status_t ret;
  ret = rsmi_init(0);
  if (ret != RSMI_STATUS_SUCCESS) return -1;
  ret = rsmi_dev_firmware_version_get(0, RSMI_FW_BLOCK_MEC, &fw_version);
  if (ret != RSMI_STATUS_SUCCESS) return -1;

  return fw_version;
}

int parseFirmwareVersion() {
  int version = -1;
  try {
    version = parseFirmwareVersionImpl();
  } catch (const std::exception& ex) {
  }
  return version;
}

bool validHsaScratchEnvSetting(const char*hsaScratchEnv, int hipRuntimeVersion, int firmwareVersion, char const* archName) {
  bool hsaScratchEnvSet = (hsaScratchEnv && strcmp(hsaScratchEnv,"1") == 0);
  if (hsaScratchEnvSet) {
    return true;
  }
  if (IsArchMatch(archName, "gfx950")) {
    return (hipRuntimeVersion >= 60443484 && firmwareVersion >= 24);
  }
  if (IsArchMatch(archName, "gfx942")) {
    return (hipRuntimeVersion >= 60443484 && firmwareVersion >= 177);
  }
  return true;
}


/* PCIe Graph Printing Functions */
static void int64ToBusIdShort(int64_t id, char* busId) {
  sprintf(busId, "%02lx",(id & 0xff000) >> 12);
}

static ncclResult_t rcclNetDevToIndex(struct ncclTopoSystem* system, int netDev, int* index) {
  *index = -1;
  for (int i=0; i<system->nodes[NET].count; i++) {
    if (system->nodes[NET].nodes[i].net.dev == netDev) {
      *index = i;
      return ncclSuccess;
    }
  }
  return ncclInternalError;
}

static void rcclGetNicBusId(struct ncclTopoSystem* system, int netDevId, std::string& busId) {
  if(netDevId < 0) return;
  char busIdStr[32];
  int netDevIdx = -1;
  rcclNetDevToIndex(system, netDevId, &netDevIdx);
  if(netDevIdx >= 0) {
    int64ToBusIdShort(system->nodes[NET].nodes[netDevIdx].net.busId, busIdStr);
  }
  busId =  std::to_string(netDevId)+ ":[0x" + std::string(busIdStr)+"]";
}

static void rcclGetGpuBusId(struct ncclTopoSystem* system, int rank, std::string& busId, int localRank = -1) {
  if(rank < 0) return;
  char busIdStr[32] = "";
  int topoIndex = -1;
  if (ncclTopoRankToIndex(system, rank, &topoIndex, false) == ncclSuccess && topoIndex >= 0) {
    int64ToBusIdShort(system->nodes[GPU].nodes[topoIndex].id, busIdStr);
    busId = std::to_string(rank) + ":[0x" + std::string(busIdStr) + "]";
  } else {
    busId = std::to_string(rank)+ ":[remote]";
  }
}

void rcclLogGraph(struct ncclTopoSystem* system,
                  struct ncclTopoGraph* graph, int rank) {
  int ngpus = system->nodes[GPU].count;
  // Print graph only once per node
  if(rank%ngpus != 0) return;
  for (int c = 0; c < graph->nChannels; c++) {
    char line[2048];
    int offset = (graph->pattern == NCCL_TOPO_PATTERN_RING)? sprintf(line, "RING_GRAPH ch=%d: hostid=%d:", c, system->hostIdx) : sprintf(line, "TREE_GRAPH ch=%d: hostid=%d:", c, system->hostIdx);

    if (system->nodes[NET].count > 0 &&
        system->nodes[GPU].count != system->nRanks &&
        !graph->nIntraChannels) {
      int n = graph->inter[2*c] - 'N';
      if (n >= 0 && n < system->nodes[NET].count) {
        offset += sprintf(line+offset, " NET/%d", n);
      }
    }

    for (int i = 0; i < ngpus; i++) {
      int n;

      n = graph->intraNets[(ngpus*c+i)*2] - 'N';
      if (n >= 0 && n < system->nodes[NET].count) {
        offset += sprintf(line+offset, " NET/%d", n);
      }

      std::string busId;
      rcclGetGpuBusId(system, graph->intra[ngpus*c+i], busId);
      offset += sprintf(line+offset, " GPU/%s", busId.c_str());

      n = graph->intraNets[(ngpus*c+i)*2+1] - 'N';
      if (n >= 0 && n < system->nodes[NET].count) {
        offset += sprintf(line+offset, " NET/%d", n);
      }
    }

    if (system->nodes[NET].count > 0 &&
        system->nodes[GPU].count != system->nRanks &&
        !graph->nIntraChannels) {
      int n = graph->inter[2*c+1] - 'N';
      if (n >= 0 && n < system->nodes[NET].count) {
        offset += sprintf(line+offset, " NET/%d", n);
      }
    }

    INFO(RCCL_GRAPHV2, "%s", line);
  }
}

void rcclLogGraphSegmentForChannels(struct ncclComm* comm) {
  for (int c = 0; c < comm->nChannels; ++c) {
    std::string curBus;
    rcclGetGpuBusId(comm->topo, comm->rank, curBus);

    const ncclTree* tree = &comm->channels[c].tree;

    auto busOrDash = [&](int rankIdx) -> const char* {
      std::string tmp;
      if (rankIdx < 0) return "-";
      rcclGetGpuBusId(comm->topo, rankIdx, tmp);
      return tmp.c_str();
    };

    const char* up  = busOrDash(tree->up);
    const char* d0  = busOrDash(tree->down[0]);
    const char* d1  = busOrDash(tree->down[1]);
    const char* d2  = busOrDash(tree->down[2]);

    INFO(RCCL_GRAPHV2, "TREE ch=%d: GPU/%s/GPU/%s/GPU/%s -> GPU/%s -> GPU/%s", c, d0, d1, d2, curBus.c_str(), up);

    const char* prev =  busOrDash(comm->channels[c].ring.prev);
    const char* next =  busOrDash(comm->channels[c].ring.next);

    INFO(RCCL_GRAPHV2, "RING ch=%d: GPU/%s -> GPU/%s -> GPU/%s", c, prev, curBus.c_str(), next);
  }
}

void rcclLogSendRecvNic(struct ncclTopoSystem* system, int rank, int netDev, int channelId, bool isSend, int useGdr, int peerRank, int proxyRank) {
  std::string gpuBusId, nicBusId;
  rcclGetGpuBusId(system, rank, gpuBusId);
  rcclGetNicBusId(system, netDev, nicBusId);

  const char* gdrSuffix = (useGdr ? "/GDRDMA" : "");
  const char* gdrMode   = (useGdr == ncclTopoGdrModePci ? "(PCI)" : "");

  if (rank != proxyRank && proxyRank >= 0) {
    std::string proxyGpuBusId;
    rcclGetGpuBusId(system, proxyRank, proxyGpuBusId);
    INFO(RCCL_GRAPHV2, "NICMAP ch=%d %s GPU/%s NIC/%s%s%s PEER_GPU/%d PROXY_GPU/%s",
           channelId,
           isSend ? "SEND" : "RECV",
           gpuBusId.c_str(),
           nicBusId.c_str(),
           gdrSuffix,
           gdrMode,
           peerRank,
           proxyGpuBusId.c_str());
  } else {
   INFO(RCCL_GRAPHV2, "NICMAP ch=%d %s GPU/%s NIC/%s%s%s PEER_GPU/%d",
           channelId,
           isSend ? "SEND" : "RECV",
           gpuBusId.c_str(),
           nicBusId.c_str(),
           gdrSuffix,
           gdrMode,
           peerRank);
  }
}

void rcclLogP2p(struct ncclTopoSystem* system, int channelId, int connIndex, int rank, int peerRank, const char* descriptor) {
  std::string gpuBusId, peerBusId;
  rcclGetGpuBusId(system, rank, gpuBusId);
  rcclGetGpuBusId(system, peerRank, peerBusId);

  INFO(RCCL_GRAPHV2, "P2PMAP ch=%d connIndex=%d %s GPU/%s PEER_GPU/%s",
           channelId,
           connIndex,
           descriptor,
           gpuBusId.c_str(),
           peerBusId.c_str());
}