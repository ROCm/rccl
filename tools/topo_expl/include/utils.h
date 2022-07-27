/*************************************************************************
 * Copyright (c) 2016-2019, NVIDIA CORPORATION. All rights reserved.
 * Modifications Copyright (c) 2019-2020 Advanced Micro Devices, Inc. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

#ifndef UTILS_H_
#define UTILS_H_

// AllGather3 - begin
struct ncclGraphInfo {
  int pattern;
  int nChannels;
  int sameChannels;
  float speedIntra;
  float speedInter;
  int typeIntra;
  int typeInter;
};

struct allGather3Data_t{
  int netDev;
  int collNetSupport;
  int nc;
  struct ncclGraphInfo tree;
  struct ncclGraphInfo ring;
  struct ncclGraphInfo collNet;
  struct ncclTopoRanks topoRanks;
};

void initCollNet();

ncclResult_t ncclTopoGetSystem(const char* xmlTopoFile, struct ncclTopoSystem** system);

ncclResult_t ncclTopoGetSystemFromXml(struct ncclXml* xml, struct ncclTopoSystem** topoSystem);

ncclResult_t fillInfo(struct ncclComm* comm, struct ncclPeerInfo* info, uint64_t commHash);

ncclResult_t initTransportsRank_1(struct ncclComm* comm, struct allGather3Data_t *allGather3Data,
  struct ncclTopoGraph& treeGraph, struct ncclTopoGraph& ringGraph, struct ncclTopoGraph& collNetGraph);

ncclResult_t initTransportsRank_3(struct ncclComm* comm, struct allGather3Data_t *allGather3Data,
  struct ncclTopoGraph& treeGraph, struct ncclTopoGraph& ringGraph, struct ncclTopoGraph& collNetGraph);

#endif