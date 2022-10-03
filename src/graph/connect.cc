/*************************************************************************
 * Copyright (c) 2016-2022, NVIDIA CORPORATION. All rights reserved.
 * Modifications Copyright (c) 2019-2022 Advanced Micro Devices, Inc. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

/*
 * Code for binary tree based on the same function available in Open MPI
 * File: ompi/mca/coll/base/coll_base_topo.c
 * 
 * Copyright (c) 2004-2005 The Trustees of Indiana University and Indiana
 *                         University Research and Technology
 *                         Corporation.  All rights reserved.
 * Copyright (c) 2004-2015 The University of Tennessee and The University
 *                         of Tennessee Research Foundation.  All rights
 *                         reserved.
 * Copyright (c) 2004-2005 High Performance Computing Center Stuttgart,
 *                         University of Stuttgart.  All rights reserved.
 * Copyright (c) 2004-2005 The Regents of the University of California.
 *                         All rights reserved.
 * Copyright (c) 2015      Research Organization for Information Science
 *                         and Technology (RIST). All rights reserved.
 */


#include "comm.h"
#include "graph.h"
#include "trees.h"
#include "rings.h"
#include "topo.h"

/******************************************************************/
/********************* Internode connection ***********************/
/******************************************************************/

ncclResult_t ncclTopoPreset(struct ncclComm* comm,
    struct ncclTopoGraph* treeGraph, struct ncclTopoGraph* ringGraph,
    struct ncclTopoRanks* topoRanks) {
  int rank = comm->rank;
  int nChannels = comm->nChannels;
  int localRanks = 0;
  for (int i=0; i<comm->topo->nodes[GPU].count; i++) {
    localRanks += comm->topo->nodes[GPU].nodes[i].gpu.nRanksPerGpu;
  }

  for (int c=0; c<nChannels; c++) {
    struct ncclChannel* channel = comm->channels+c;
    channel->ring.prev = channel->ring.next = -1;
    channel->tree.up = -1;
    for (int i=0; i<NCCL_MAX_TREE_ARITY; i++) channel->tree.down[i] = -1;
    channel->collTree.out = -1;
    channel->collTree.headRank = -1;
    channel->collTree.nHeads = 0;
    channel->collTree.shift = 0;
    for (int i=0; i<NCCL_MAX_DIRECT_ARITY; i++) channel->collTree.up[i] = -1;
    for (int i=0; i<NCCL_MAX_DIRECT_ARITY; i++) channel->collTree.down[i] = -1;

    int* ringIntra = ringGraph->intra+c*localRanks;
    int* treeIntra = treeGraph->intra+c*localRanks;

    for (int i=0; i<localRanks; i++) {
      if (ringIntra[i] == rank) {
        topoRanks->ringRecv[c] = ringIntra[0];
        topoRanks->ringSend[c] = ringIntra[localRanks-1];
        channel->ring.prev = (i == 0) ? -1 : ringIntra[i-1];
        channel->ring.next = (i == localRanks-1) ? -1 : ringIntra[i+1];
      }
      if (treeIntra[i] == rank) {
        int parentIndex = 0;
        int child0Index = treeGraph->pattern == NCCL_TOPO_PATTERN_TREE ? 0 : 1;
        int child1Index = treeGraph->pattern == NCCL_TOPO_PATTERN_SPLIT_TREE ? 1 : 0;

        topoRanks->treeToParent[c] = treeIntra[parentIndex];
        topoRanks->treeToChild0[c] = treeIntra[child0Index];
        topoRanks->treeToChild1[c] = treeIntra[child1Index];
        channel->tree.up         = i == 0 ? -1 : treeIntra[i-1];
        channel->tree.down[0]    = i == localRanks-1 ? -1 : treeIntra[i+1];
      }
    }
    topoRanks->ringPrev[c] = channel->ring.prev;
    topoRanks->ringNext[c] = channel->ring.next;
  }
  // Duplicate channels rings/trees
  struct ncclChannel* channel0 = comm->channels;
  struct ncclChannel* channel1 = (nChannels > MAXCHANNELS/2) ? 0 : channel0+nChannels;
  if (channel1) memcpy(channel1, channel0, nChannels*sizeof(struct ncclChannel));
  return ncclSuccess;
}

static int calculate_level (int rank)
{
    int level, num;
    if( rank < 0 ) return -1;
    for( level = 0, num = 0; num <= rank; level++ ) {
      num += 1<<level;
    }
    return level-1;
}

static int calculate_num_nodes_up_to_level (int level)
{
  return ((1<<level) - 1);
}

ncclResult_t ncclBinaryTreePostset(struct ncclComm* comm,
    struct ncclTopoGraph* treeGraph) {
  int nChannels = comm->nChannels;
  int localRanks = 0;
  for (int i=0; i<comm->topo->nodes[GPU].count; i++) {
    localRanks += comm->topo->nodes[GPU].nodes[i].gpu.nRanksPerGpu;
  }

  for (int c=0; c<nChannels; c++) {
    struct ncclChannel* channel = comm->channels+c;
    // Only the first rank on a GPU can be a treeRoot
    int treeRoot = comm->topo->nodes[GPU].nodes[c%comm->topo->nodes[GPU].count].gpu.rank[0];

    channel->binTree.up      = -1;
    channel->binTree.down[0] = -1;
    channel->binTree.down[1] = -1;
    channel->binTree.down[2] = -1;

    /*
     * Shift all ranks by root, so that the algorithm can be
     * designed as if root would be always 0
     * shiftedrank should be used in calculating distances
     * and position in tree
     */
    int shiftedrank = comm->rank - treeRoot;
    if (shiftedrank < 0 ) {
      shiftedrank += localRanks;
    }

    /* calculate my level */
    int level = calculate_level (shiftedrank);
    int delta = 1<<level;

    /* find my children */
    for (int i = 0; i < 2; i++) {
      int schild = shiftedrank + delta * (i+1);
      if (schild < localRanks) {
	channel->binTree.down[i] = (schild+treeRoot)%localRanks;
      }
    }

    /* find my parent */
    int slimit = calculate_num_nodes_up_to_level (level);
    int sparent = shiftedrank;
    if (sparent < 2) {
      sparent = 0;
    }
    else {
      while (sparent >= slimit) {
	sparent -= delta/2;
      }
    }
    if (comm->rank != treeRoot) {
      channel->binTree.up = (sparent+treeRoot)%localRanks;
    }
  }

  return ncclSuccess;
}

#define NUM_HAYABUSA_TREES 2
static bool hayabusa_tree_matrix_is_init=false;
static int hayabusa_tree_matrix[NUM_HAYABUSA_TREES][16][4];

static void hayabusa_tree_matrix_init()
{
  if (hayabusa_tree_matrix_is_init)
    return;

  // index = rank of proc, child0, child1, child2, parent
  // channel  0: root is 15
  hayabusa_tree_matrix[0][0][0]  = 1;
  hayabusa_tree_matrix[0][0][1]  = -1;
  hayabusa_tree_matrix[0][0][2]  = -1;
  hayabusa_tree_matrix[0][0][3]  = 4;

  hayabusa_tree_matrix[0][1][0]  = -1;
  hayabusa_tree_matrix[0][1][1]  = -1;
  hayabusa_tree_matrix[0][1][2]  = -1;
  hayabusa_tree_matrix[0][1][3]  = 0;

  hayabusa_tree_matrix[0][2][0]  = 3;
  hayabusa_tree_matrix[0][2][1]  = -1;
  hayabusa_tree_matrix[0][2][2]  = -1;
  hayabusa_tree_matrix[0][2][3]  = 6;

  hayabusa_tree_matrix[0][3][0]  = -1;
  hayabusa_tree_matrix[0][3][1]  = -1;
  hayabusa_tree_matrix[0][3][2]  = -1;
  hayabusa_tree_matrix[0][3][3]  = 2;

  hayabusa_tree_matrix[0][4][0]  = 0;
  hayabusa_tree_matrix[0][4][1]  = -1;
  hayabusa_tree_matrix[0][4][2]  = -1;
  hayabusa_tree_matrix[0][4][3]  = 5;

  hayabusa_tree_matrix[0][5][0]  = 4;
  hayabusa_tree_matrix[0][5][1]  = -1;
  hayabusa_tree_matrix[0][5][2]  = -1;
  hayabusa_tree_matrix[0][5][3]  = 14;

  hayabusa_tree_matrix[0][6][0]  = 2;
  hayabusa_tree_matrix[0][6][1]  = 7;
  hayabusa_tree_matrix[0][6][2]  = -1;
  hayabusa_tree_matrix[0][6][3]  = 14;

  hayabusa_tree_matrix[0][7][0]  = -1;
  hayabusa_tree_matrix[0][7][1]  = -1;
  hayabusa_tree_matrix[0][7][2]  = -1;
  hayabusa_tree_matrix[0][7][3]  = 6;

  hayabusa_tree_matrix[0][8][0]  = -1;
  hayabusa_tree_matrix[0][8][1]  = -1;
  hayabusa_tree_matrix[0][8][2]  = -1;
  hayabusa_tree_matrix[0][8][3]  = 9;

  hayabusa_tree_matrix[0][9][0]  = 13;
  hayabusa_tree_matrix[0][9][1]  = 8;
  hayabusa_tree_matrix[0][9][2]  = -1;
  hayabusa_tree_matrix[0][9][3]  = 11;

  hayabusa_tree_matrix[0][10][0] = -1;
  hayabusa_tree_matrix[0][10][1] = -1;
  hayabusa_tree_matrix[0][10][2] = -1;
  hayabusa_tree_matrix[0][10][3] = 11;

  hayabusa_tree_matrix[0][11][0] = 9;
  hayabusa_tree_matrix[0][11][1] = 10;
  hayabusa_tree_matrix[0][11][2] = -1;
  hayabusa_tree_matrix[0][11][3] = 15;

  hayabusa_tree_matrix[0][12][0] = -1;
  hayabusa_tree_matrix[0][12][1] = -1;
  hayabusa_tree_matrix[0][12][2] = -1;
  hayabusa_tree_matrix[0][12][3] = 13;

  hayabusa_tree_matrix[0][13][0] = 12;
  hayabusa_tree_matrix[0][13][1] = -1;
  hayabusa_tree_matrix[0][13][2] = -1;
  hayabusa_tree_matrix[0][13][3] = 9;

  hayabusa_tree_matrix[0][14][0] = 5;
  hayabusa_tree_matrix[0][14][1] = 6;
  hayabusa_tree_matrix[0][14][2] = -1;
  hayabusa_tree_matrix[0][14][3] = 15;

  hayabusa_tree_matrix[0][15][0] = 14;
  hayabusa_tree_matrix[0][15][1] = 11;
  hayabusa_tree_matrix[0][15][2] = -1;
  hayabusa_tree_matrix[0][15][3] = -1;

  //Channel 1: root is 6
  hayabusa_tree_matrix[1][0][0]  = -1;
  hayabusa_tree_matrix[1][0][1]  = -1;
  hayabusa_tree_matrix[1][0][2]  = -1;
  hayabusa_tree_matrix[1][0][3]  = 1;

  hayabusa_tree_matrix[1][1][0]  = 5;
  hayabusa_tree_matrix[1][1][1]  = 0;
  hayabusa_tree_matrix[1][1][2]  = -1;
  hayabusa_tree_matrix[1][1][3]  = 3;

  hayabusa_tree_matrix[1][2][0]  = -1;
  hayabusa_tree_matrix[1][2][1]  = -1;
  hayabusa_tree_matrix[1][2][2]  = -1;
  hayabusa_tree_matrix[1][2][3]  = 3;

  hayabusa_tree_matrix[1][3][0]  = 1;
  hayabusa_tree_matrix[1][3][1]  = 2;
  hayabusa_tree_matrix[1][3][2]  = -1;
  hayabusa_tree_matrix[1][3][3]  = 7;

  hayabusa_tree_matrix[1][4][0]  = -1;
  hayabusa_tree_matrix[1][4][1]  = -1;
  hayabusa_tree_matrix[1][4][2]  = -1;
  hayabusa_tree_matrix[1][4][3]  = 5;

  hayabusa_tree_matrix[1][5][0]  = 4;
  hayabusa_tree_matrix[1][5][1]  = -1;
  hayabusa_tree_matrix[1][5][2]  = -1;
  hayabusa_tree_matrix[1][5][3]  = 1;

  hayabusa_tree_matrix[1][6][0]  = 7;
  hayabusa_tree_matrix[1][6][1]  = 13;
  hayabusa_tree_matrix[1][6][2]  = -1;
  hayabusa_tree_matrix[1][6][3]  = -1;

  hayabusa_tree_matrix[1][7][0]  = 3;
  hayabusa_tree_matrix[1][7][1]  = 15;
  hayabusa_tree_matrix[1][7][2]  = -1;
  hayabusa_tree_matrix[1][7][3]  = 6;

  hayabusa_tree_matrix[1][8][0]  = 9;
  hayabusa_tree_matrix[1][8][1]  = -1;
  hayabusa_tree_matrix[1][8][2]  = -1;
  hayabusa_tree_matrix[1][8][3]  = 12;

  hayabusa_tree_matrix[1][9][0]  = -1;
  hayabusa_tree_matrix[1][9][1]  = -1;
  hayabusa_tree_matrix[1][9][2]  = -1;
  hayabusa_tree_matrix[1][9][3]  = 8;

  hayabusa_tree_matrix[1][10][0] = -1;
  hayabusa_tree_matrix[1][10][1] = -1;
  hayabusa_tree_matrix[1][10][2] = -1;
  hayabusa_tree_matrix[1][10][3] = 11;

  hayabusa_tree_matrix[1][11][0] = 10;
  hayabusa_tree_matrix[1][11][1] = -1;
  hayabusa_tree_matrix[1][11][2] = -1;
  hayabusa_tree_matrix[1][11][3] = 15;

  hayabusa_tree_matrix[1][12][0] = 8;
  hayabusa_tree_matrix[1][12][1] = -1;
  hayabusa_tree_matrix[1][12][2] = -1;
  hayabusa_tree_matrix[1][12][3] = 13;

  hayabusa_tree_matrix[1][13][0] = 12;
  hayabusa_tree_matrix[1][13][1] = -1;
  hayabusa_tree_matrix[1][13][2] = -1;
  hayabusa_tree_matrix[1][13][3] = 6;

  hayabusa_tree_matrix[1][14][0] = -1;
  hayabusa_tree_matrix[1][14][1] = -1;
  hayabusa_tree_matrix[1][14][2] = -1;
  hayabusa_tree_matrix[1][14][3] = 15;

  hayabusa_tree_matrix[1][15][0] = 11;
  hayabusa_tree_matrix[1][15][1] = 14;
  hayabusa_tree_matrix[1][15][2] = -1;
  hayabusa_tree_matrix[1][15][3] = 7;

  hayabusa_tree_matrix_is_init = true;
}

static void set_channel_info(int c, int rank, struct ncclChannel *channel)
{
  channel->binTree.down[0] = hayabusa_tree_matrix[c%NUM_HAYABUSA_TREES][rank][0];
  channel->binTree.down[1] = hayabusa_tree_matrix[c%NUM_HAYABUSA_TREES][rank][1];
  channel->binTree.down[2] = hayabusa_tree_matrix[c%NUM_HAYABUSA_TREES][rank][2];
  channel->binTree.up      = hayabusa_tree_matrix[c%NUM_HAYABUSA_TREES][rank][3];
}

ncclResult_t ncclBinaryTreeHayabusaPostset(struct ncclComm* comm,
    struct ncclTopoGraph* treeGraph) {
  int nChannels = comm->nChannels;

  hayabusa_tree_matrix_init();

  for (int c=0; c<nChannels; c++) {
    struct ncclChannel* channel = comm->channels+c;

    set_channel_info(c, comm->localRank, channel);
  }

  return ncclSuccess;
}

ncclResult_t ncclTreeBasePostset(struct ncclComm* comm,
    struct ncclTopoGraph* treeGraph) {
  int nChannels = comm->nChannels;
  int localRanks = 0;
  for (int i=0; i<comm->topo->nodes[GPU].count; i++) {
    localRanks += comm->topo->nodes[GPU].nodes[i].gpu.nRanksPerGpu;
  }
  //new tree
  for (int c=0; c<nChannels; c++) {
    int* treeIntra = treeGraph->intra+c%3*localRanks;
    int buff = ((c%6)*localRanks)/6 + ((localRanks/4)*(c/6));

    int tempArray[256];
    for (int ko=0; ko < localRanks; ko++){
      tempArray[ko] = treeIntra[(ko+buff)%localRanks];
    }
    struct ncclChannel* channel = comm->channels+c;

    int curRank = comm->rank;
    int arrayIndex;

    for (int i=0; i<localRanks; i++) {
      if (tempArray[i] == curRank) {
        if (i == 0) {
          channel->tree.up = -1;
          channel->tree.down[0] = tempArray[i+1];
          channel->tree.down[1] = tempArray[localRanks-1];
          channel->tree.down[2] = -1;
        }
        else {
          channel->tree.up = i > localRanks/2 ?  tempArray[(i+1)%localRanks] : tempArray[i-1];
          channel->tree.down[0] = i > localRanks/2 ?  tempArray[i-1] : tempArray[i+1];
          if ((i == localRanks/2) || (i == (localRanks/2 + 1))) {
            channel->tree.down[0] = -1;
          }
          channel->tree.down[1] = -1;
          channel->tree.down[2] = -1;
        }
      }
    }
  }
  return ncclSuccess;
}

static ncclResult_t connectRings(struct ncclComm* comm, int* ringRecv, int* ringSend, int* ringPrev, int* ringNext, int* firstRanks) {
  int nChannels = comm->nChannels;
  int nNodes = comm->nNodes;
  for (int c=0; c<nChannels; c++) {
    int* recv = ringRecv+c*comm->nRanks;
    int* send = ringSend+c*comm->nRanks;
    int* prev = ringPrev+c*comm->nRanks;
    int* next = ringNext+c*comm->nRanks;
    struct ncclChannel* channel0 = comm->channels+c;
    struct ncclChannel* channel1 = (nChannels > MAXCHANNELS/2) ? 0 : channel0+nChannels;
    for (int n=0; n<nNodes; n++) {
      int recvRank = recv[firstRanks[n]];
      int prevSendRank = send[firstRanks[(n-1+nNodes)%nNodes]];
      prev[recvRank] = prevSendRank;
      if (comm->rank == recvRank) {
        channel0->ring.prev = prevSendRank;
        if (channel1) channel1->ring.prev = prevSendRank;
      }
      int sendRank = send[firstRanks[n]];
      int nextRecvRank = recv[firstRanks[(n+1)%nNodes]];
      next[sendRank] = nextRecvRank;
      if (comm->rank == sendRank) {
        channel0->ring.next = nextRecvRank;
        if (channel1) channel1->ring.next = nextRecvRank;
      }
    }
    TRACE(NCCL_GRAPH, "Ring %d : %d -> %d -> %d", c, channel0->ring.prev, comm->rank, channel0->ring.next);
    if (channel1) TRACE(NCCL_GRAPH, "Ring %d : %d -> %d -> %d", c+nChannels, channel1->ring.prev, comm->rank, channel1->ring.next);
  }
  return ncclSuccess;
}

static ncclResult_t getIndexes(int* ranks, int* indexes, int nNodes, int* firstRanks) {
 for (int n=0; n<nNodes; n++) indexes[n] = ranks[firstRanks[n]];
 return ncclSuccess;
}

static ncclResult_t setTreeUp(struct ncclTree* tree, int* indexes, int u) {
  if (u == -1) return ncclSuccess;
  tree->up = indexes[u];
  return ncclSuccess;
}

static ncclResult_t setTreeDown(struct ncclTree* tree, int* indexes, int d) {
  if (d == -1) return ncclSuccess;
  int x = 0;
  while (x < NCCL_MAX_TREE_ARITY && tree->down[x] >= 0) x++;
  if (x == NCCL_MAX_TREE_ARITY) {
    WARN("Internal error : tree already has %d children (%d %d %d)", x, tree->down[0], tree->down[1], tree->down[2]);
    return ncclInternalError;
  }
  tree->down[x] = indexes[d];
  return ncclSuccess;
}

static ncclResult_t connectTrees(struct ncclComm* comm, int* treeToParent, int* treeToChild0, int* treeToChild1, int* firstRanks, int* treePatterns) {
  const int nChannels = (comm->nChannels > MAXCHANNELS/2) ? comm->nChannels/2 : comm->nChannels, nNodes = comm->nNodes, node = comm->node;
  int* ranksToParent, *ranksToChild0, *ranksToChild1;
  NCCLCHECK(ncclCalloc(&ranksToParent, nNodes));
  NCCLCHECK(ncclCalloc(&ranksToChild0, nNodes));
  NCCLCHECK(ncclCalloc(&ranksToChild1, nNodes));

  // Compute tree depth. Not an exact value but a good approximation in most
  // cases
  int depth = comm->nRanks/nNodes - 1 + log2i(nNodes);

  int t0u, t0d0, t0d1, t0ChildType, t1u, t1d0, t1d1, t1ChildType;
  NCCLCHECK(ncclGetDtree(nNodes, node, &t0u, &t0d0, &t0d1, &t0ChildType, &t1u, &t1d0, &t1d1, &t1ChildType));

  if (comm->nChannels <= MAXCHANNELS/2) {
    for (int c=0; c<nChannels; c++) {
       struct ncclChannel* channel0 = comm->channels+c;
       struct ncclChannel* channel1 = channel0+nChannels;
       NCCLCHECK(getIndexes(treeToParent+c*comm->nRanks, ranksToParent, nNodes, firstRanks));
       NCCLCHECK(getIndexes(treeToChild0+c*comm->nRanks, ranksToChild0, nNodes, firstRanks));
       NCCLCHECK(getIndexes(treeToChild1+c*comm->nRanks, ranksToChild1, nNodes, firstRanks));
       if (comm->rank == ranksToParent[node]) {
         NCCLCHECK(setTreeUp(&channel0->tree, t0ChildType == 0 ? ranksToChild0 : ranksToChild1, t0u));
         NCCLCHECK(setTreeUp(&channel1->tree, t1ChildType == 0 ? ranksToChild0 : ranksToChild1, t1u));
       }
       if (comm->rank == ranksToChild0[node]) {
         NCCLCHECK(setTreeDown(&channel0->tree, ranksToParent, t0d0));
         NCCLCHECK(setTreeDown(&channel1->tree, ranksToParent, t1d0));
       }
       if (comm->rank == ranksToChild1[node]) {
         NCCLCHECK(setTreeDown(&channel0->tree, ranksToParent, t0d1));
         NCCLCHECK(setTreeDown(&channel1->tree, ranksToParent, t1d1));
       }
       if (comm->rank == ranksToParent[node] ||
           comm->rank == ranksToChild0[node] ||
           comm->rank == ranksToChild1[node]) {
         INFO(NCCL_GRAPH, "Tree %d : %d -> %d -> %d/%d/%d", c,           channel0->tree.up, comm->rank, channel0->tree.down[0], channel0->tree.down[1], channel0->tree.down[2]);
         INFO(NCCL_GRAPH, "Tree %d : %d -> %d -> %d/%d/%d", c+nChannels, channel1->tree.up, comm->rank, channel1->tree.down[0], channel1->tree.down[1], channel1->tree.down[2]);
       }
       channel0->tree.depth = channel1->tree.depth = depth;
    }
  } else {
    for (int c=0; c<nChannels; c++) {
       struct ncclChannel* channel0 = comm->channels+c;
       NCCLCHECK(getIndexes(treeToParent+c*comm->nRanks, ranksToParent, nNodes, firstRanks));
       NCCLCHECK(getIndexes(treeToChild0+c*comm->nRanks, ranksToChild0, nNodes, firstRanks));
       NCCLCHECK(getIndexes(treeToChild1+c*comm->nRanks, ranksToChild1, nNodes, firstRanks));
       if (comm->rank == ranksToParent[node]) {
         NCCLCHECK(setTreeUp(&channel0->tree, t0ChildType == 0 ? ranksToChild0 : ranksToChild1, t0u));
       }
       if (comm->rank == ranksToChild0[node]) {
         NCCLCHECK(setTreeDown(&channel0->tree, ranksToParent, t0d0));
       }
       if (comm->rank == ranksToChild1[node]) {
         NCCLCHECK(setTreeDown(&channel0->tree, ranksToParent, t0d1));
       }
       if (comm->rank == ranksToParent[node] ||
           comm->rank == ranksToChild0[node] ||
           comm->rank == ranksToChild1[node]) {
         INFO(NCCL_GRAPH, "Tree %d : %d -> %d -> %d/%d/%d", c, channel0->tree.up, comm->rank, channel0->tree.down[0], channel0->tree.down[1], channel0->tree.down[2]);
       }
       channel0->tree.depth = depth;
    }
    for (int c=nChannels; c<nChannels*2; c++) {
       struct ncclChannel* channel1 = comm->channels+c;
       NCCLCHECK(getIndexes(treeToParent+c*comm->nRanks, ranksToParent, nNodes, firstRanks));
       NCCLCHECK(getIndexes(treeToChild0+c*comm->nRanks, ranksToChild0, nNodes, firstRanks));
       NCCLCHECK(getIndexes(treeToChild1+c*comm->nRanks, ranksToChild1, nNodes, firstRanks));
       if (comm->rank == ranksToParent[node]) {
         NCCLCHECK(setTreeUp(&channel1->tree, t1ChildType == 0 ? ranksToChild0 : ranksToChild1, t1u));
       }
       if (comm->rank == ranksToChild0[node]) {
         NCCLCHECK(setTreeDown(&channel1->tree, ranksToParent, t1d0));
       }
       if (comm->rank == ranksToChild1[node]) {
         NCCLCHECK(setTreeDown(&channel1->tree, ranksToParent, t1d1));
       }
       if (comm->rank == ranksToParent[node] ||
           comm->rank == ranksToChild0[node] ||
           comm->rank == ranksToChild1[node]) {
         INFO(NCCL_GRAPH, "Tree %d : %d -> %d -> %d/%d/%d", c, channel1->tree.up, comm->rank, channel1->tree.down[0], channel1->tree.down[1], channel1->tree.down[2]);
       }
       channel1->tree.depth = depth;
    }
  }
  free(ranksToParent);
  free(ranksToChild0);
  free(ranksToChild1);
  return ncclSuccess;
}

static ncclResult_t connectCollNet(struct ncclComm* comm, struct ncclTopoGraph* collNetGraph) {
  int rank = comm->rank;
  int localRanks = comm->localRanks;
  int nHeads = collNetGraph->nChannels;
  int *heads;
  NCCLCHECK(ncclCalloc(&heads, nHeads));
  // Find all head ranks
  // Head index is always 0
  for (int c=0; c<nHeads; c++) {
    int* collNetIntra = collNetGraph->intra+c*localRanks;
    heads[c] = collNetIntra[0];
  }
  // For all channels
  for (int c=0; c<comm->nChannels; c++) {
    struct ncclChannel* channel = comm->channels+c;
    char line[1024];
    sprintf(line, "CollNet channel %d rank %d ", c, rank);
    int nDown = 0;
    for (int i=0; i<nHeads; i++) {
      if (rank == heads[i]) { // is head
        channel->collTree.headRank = i; // Mark the index for deciding offset in the CUDA kernel
        channel->collTree.out = comm->nRanks; // Set root of collTree to id nranks
        int* collNetIntra = collNetGraph->intra+i*localRanks;
        sprintf(line+strlen(line), "down ");
        for (int r=0; r<localRanks; r++) {
          if (collNetIntra[r] == rank) continue;
          channel->collTree.down[nDown++] = collNetIntra[r];  // connect to all peers
          sprintf(line+strlen(line), " %d ", collNetIntra[r]);
        }
        sprintf(line+strlen(line), "nDown %d ", nDown);
        break;
      }
    }
    // Connect to all heads
    int nUp = 0;
    sprintf(line+strlen(line), "up ");
    for (int h=0; h<nHeads; h++) {
      if (rank == heads[h]) continue;
      channel->collTree.up[nUp++] = heads[h];
      sprintf(line+strlen(line), " %d ", heads[h]);
    }
    channel->collTree.nHeads = nHeads;
    channel->collTree.shift = (rank%localRanks)%nHeads; // Shift by intraRank so that leaves don't send to same head simultaneously
    channel->collTree.depth = (nUp == 0 && nDown == 0) ? 1 : 2;
    sprintf(line+strlen(line), "nUp %d nHeads %d ", nUp, nHeads);
    sprintf(line+strlen(line), "headRank %d out %d shift %d", channel->collTree.headRank, channel->collTree.out, channel->collTree.shift);
    INFO(NCCL_GRAPH, "%s", line);
  }
  free(heads);
  return ncclSuccess;
}

// Legacy naming
NCCL_PARAM(MinNrings, "MIN_NRINGS", -2);
NCCL_PARAM(MaxNrings, "MAX_NRINGS", -2);
// New naming
NCCL_PARAM(MinNchannels, "MIN_NCHANNELS", -2);
NCCL_PARAM(MaxNchannels, "MAX_NCHANNELS", -2);

int ncclMinNchannels() {
  int minNchannels = 2;
  if (ncclParamMinNrings() != -2) minNchannels = ncclParamMinNrings();
  if (ncclParamMinNchannels() != -2) minNchannels = ncclParamMinNchannels();
  if (minNchannels > MAXCHANNELS) {
    WARN("User asked for a minimum of %d channels, limiting to %d", minNchannels, MAXCHANNELS);
    minNchannels = MAXCHANNELS;
  }
  if (minNchannels < 0) minNchannels = 0;
  return minNchannels;
}
int ncclMaxNchannels() {
  int maxNchannels = MAXCHANNELS;
  if (ncclParamMaxNrings() != -2) maxNchannels = ncclParamMaxNrings();
  if (ncclParamMaxNchannels() != -2) maxNchannels = ncclParamMaxNchannels();
  if (maxNchannels > MAXCHANNELS) maxNchannels = MAXCHANNELS;
  if (maxNchannels < 1) {
    WARN("User asked for a maximum of %d channels, setting it to 1", maxNchannels);
    maxNchannels = 1;
  }
  return maxNchannels;
}

static int copyChannels(struct ncclComm* comm, int start, int end, int* ringPrev, int* ringNext) {
  int nranks = comm->nRanks;
  int c;
  for (c=start; c<end; c++) {
    memcpy(ringPrev+c*nranks, ringPrev+(c-start)*nranks, nranks*sizeof(int));
    memcpy(ringNext+c*nranks, ringNext+(c-start)*nranks, nranks*sizeof(int));
    memcpy(comm->channels+c, comm->channels+c-start, sizeof(struct ncclChannel));
  }
  return c;
}

ncclResult_t ncclTopoPostset(struct ncclComm* comm, int* firstRanks, int* treePatterns, struct ncclTopoRanks** allTopoRanks, int* rings, struct ncclTopoGraph* collNetGraph, int nc) {
  // Gather data from all ranks
  int *ringRecv, *ringSend, *ringPrev, *ringNext, *treeToParent, *treeToChild0, *treeToChild1;
  int nranks = comm->nRanks;
  int nChannels = comm->nChannels;
  NCCLCHECK(ncclCalloc(&ringRecv, nranks*MAXCHANNELS));
  NCCLCHECK(ncclCalloc(&ringSend, nranks*MAXCHANNELS));
  NCCLCHECK(ncclCalloc(&ringPrev, nranks*MAXCHANNELS));
  NCCLCHECK(ncclCalloc(&ringNext, nranks*MAXCHANNELS));
  NCCLCHECK(ncclCalloc(&treeToParent, nranks*MAXCHANNELS));
  NCCLCHECK(ncclCalloc(&treeToChild0, nranks*MAXCHANNELS));
  NCCLCHECK(ncclCalloc(&treeToChild1, nranks*MAXCHANNELS));
  for (int i=0; i<nranks; i++) {
    for (int c=0; c<nChannels;c++) {
      ringRecv[c*nranks+i] = allTopoRanks[i]->ringRecv[c];
      ringSend[c*nranks+i] = allTopoRanks[i]->ringSend[c];
      ringPrev[c*nranks+i] = allTopoRanks[i]->ringPrev[c];
      ringNext[c*nranks+i] = allTopoRanks[i]->ringNext[c];
      treeToParent[c*nranks+i] = allTopoRanks[i]->treeToParent[c];
      treeToChild0[c*nranks+i] = allTopoRanks[i]->treeToChild0[c];
      treeToChild1[c*nranks+i] = allTopoRanks[i]->treeToChild1[c];
    }
  }

  // Connect rings and trees. This should also duplicate the channels.
  NCCLCHECK(connectRings(comm, ringRecv, ringSend, ringPrev, ringNext, firstRanks));
  NCCLCHECK(connectTrees(comm, treeToParent, treeToChild0, treeToChild1, firstRanks, treePatterns));

  // Duplicate ringPrev/ringNext for ncclBuildRing
  if (nChannels <= MAXCHANNELS/2) memcpy(ringPrev+nChannels*nranks, ringPrev, nChannels*nranks*sizeof(int));
  if (nChannels <= MAXCHANNELS/2) memcpy(ringNext+nChannels*nranks, ringNext, nChannels*nranks*sizeof(int));

  // Get number of channels after duplication
  nc *= comm->nChannels;
  nc = std::min((int)ncclMaxNchannels(), nc);
  // Duplication should be complete now
  nChannels = comm->nChannels = std::min(MAXCHANNELS, (nChannels <= MAXCHANNELS/2) ? nChannels*2 : nChannels);

  // Setup CollNet
  if (comm->collNetSupport == 1) {
    // Add more channels to saturate intra-node bandwidth, except the 1 PPN case
    if (collNetGraph->speedIntra > collNetGraph->speedInter && comm->nRanks > comm->nNodes) {
      int collNetNchannels = std::min(MAXCHANNELS, nChannels+nChannels/2);
      nChannels = comm->nChannels = copyChannels(comm, nChannels, collNetNchannels, ringPrev, ringNext);
    }
    NCCLCHECK(connectCollNet(comm, collNetGraph));
  }

  // Honor NCCL_MIN_NRINGS/NCCL_MAX_NRINGS.
  // We permit combining max, then min, to only use the first channels, then duplicate them.
  nChannels = comm->nChannels = std::min((int)ncclMaxNchannels(), nChannels);
  nChannels = comm->nChannels = copyChannels(comm, nChannels, std::max(nc, ncclMinNchannels()), ringPrev, ringNext);

  // Create rings array and check all is fine
  NCCLCHECK(ncclBuildRings(nChannels, rings, comm->rank, comm->nRanks, ringPrev, ringNext));

  free(ringRecv);
  free(ringSend);
  free(ringPrev);
  free(ringNext);
  free(treeToParent);
  free(treeToChild0);
  free(treeToChild1);

  return ncclSuccess;
}
