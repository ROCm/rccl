/*************************************************************************
 * Copyright (c) 2016-2019, NVIDIA CORPORATION. All rights reserved.
 * Modifications Copyright (c) 2019-2023 Advanced Micro Devices, Inc. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

#ifndef UTILS_H_
#define UTILS_H_

struct graphInfo {
  int pattern;
  int nChannels;
  int sameChannels;
  float bwIntra;
  float bwInter;
  int typeIntra;
  int typeInter;
};

struct allGatherInfo {
  struct graphInfo graphInfo[NCCL_NUM_ALGORITHMS];
  struct ncclTopoRanks topoRanks;
  int nc;
  bool pivotA2AEnabled;
  bool ll128Enabled;
  bool mscclEnabled;
};

struct QPTracker;
struct NodeQPStats;
struct RankQPStats;
struct ChannelQPStats;

QPTracker* getDeviceTracker(int deviceId);
NodeQPStats* getNodeStats(int nodeId);
RankQPStats* getRankStats(int rankId, int nodeId);
ChannelQPStats* getChannelStats(int channelId);
void printQPStatistics();
void printChannelMappings();
void resetQPStatistics();
int getDeviceQPCount(int deviceId);
int getNodeQPCount(int nodeId);
int getRankQPCount(int rankId);
bool isDeviceQPLimitReached(int deviceId);
void trackQPWithChannelInfo(int rank, int nodeId, int deviceId, int channelId, int peerRank, bool isSend);

void initCollNet();

ncclResult_t ncclTopoGetSystem(const char* xmlTopoFile, struct ncclTopoSystem** system);

ncclResult_t ncclTopoGetSystemFromXml(struct ncclXml* xml, struct ncclTopoSystem** topoSystem);

ncclResult_t fillInfo(struct ncclComm* comm, struct ncclPeerInfo* info, uint64_t commHash);

ncclResult_t initTransportsRank_1(struct ncclComm* comm, struct allGatherInfo *allGather3Data,
  struct ncclTopoGraph& treeGraph, struct ncclTopoGraph& ringGraph, struct ncclTopoGraph& collNetGraph, struct ncclTopoGraph& nvlsGraph, struct ncclComm* parent = NULL);

ncclResult_t initTransportsRank_3(struct ncclComm* comm, struct allGatherInfo *allGather3Data,
  struct ncclTopoGraph& treeGraph, struct ncclTopoGraph& ringGraph, struct ncclTopoGraph& collNetGraph, struct ncclTopoGraph& nvlsGraph);

#define TIME_START(index)

#define TIME_STOP(index)

#define TIME_CANCEL(index)

#define TIME_PRINT(name)

#endif
