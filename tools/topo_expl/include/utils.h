/*************************************************************************
 * Copyright (c) 2016-2019, NVIDIA CORPORATION. All rights reserved.
 * Modifications Copyright (c) 2019-2020 Advanced Micro Devices, Inc. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

#ifndef UTILS_H_
#define UTILS_H_

struct allGather1Data_t {
  struct ncclPeerInfo peerInfo;
  struct ncclComm* comm;
};

struct allGather3Data_t {
  int cudaCompCap;
  int fullCudaCompCap;
  int nvlink;
  int nChannels;
  struct {
    int sameChannels;
    int speedIntra;
    int speedInter;
    int nvlink;
  } tree;
  struct {
    int sameChannels;
    int speedIntra;
    int speedInter;
    int nvlink;
  } ring;
  struct ncclTopoRanks topoRanks;
};

ncclResult_t bootstrapAllGather(struct ncclComm* comm, struct allGather1Data_t * allGather1Data);

ncclResult_t initTransportsRank_1(struct ncclComm* comm, struct allGather1Data_t *allGather1Data,
  struct allGather3Data_t *allGather3Data, struct ncclTopoGraph& treeGraph, struct ncclTopoGraph& ringGraph);

ncclResult_t initTransportsRank_3(struct ncclComm* comm, struct allGather3Data_t *allGather3Data,
  struct ncclTopoGraph& treeGraph, struct ncclTopoGraph& ringGraph);

#endif