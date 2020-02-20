/*************************************************************************
 * Copyright (c) 2016-2019, NVIDIA CORPORATION. All rights reserved.
 * Modifications Copyright (c) 2019-2020 Advanced Micro Devices, Inc. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

#include "nccl.h"
#include "channel.h"
#include "nvmlwrap.h"
#include "bootstrap.h"
#include "transport.h"
#include "group.h"
#include "net.h"
#include "graph.h"
#include "argcheck.h"
#include <sched.h>
#include <fcntl.h>
#include <unistd.h>
#include <hip/hip_runtime.h>
#include <string.h>
#include <errno.h>
#include <assert.h>
#include <dlfcn.h>
#include <sys/types.h>
#include <sys/stat.h>
#include <unistd.h>
#include "model.h"
#include "utils.h"

extern NodeModel *node_model;

NCCL_PARAM(CrossNic, "CROSS_NIC", 2);

// Get current Compute Capability
int ncclCudaCompCap() {
  int ccMajor = 1, ccMinor = 0;
  return ccMajor*10+ccMinor;
}

ncclResult_t int64ToBusId(int64_t id, char* busId) {
  sprintf(busId, "%04lx:%02lx:%02lx.%01lx", (id) >> 20, (id & 0xff000) >> 12, (id & 0xff0) >> 4, (id & 0xf));
  return ncclSuccess;
}

ncclResult_t busIdToInt64(char* busId, int64_t* id) {
  const int size = strlen(busId);
  char* hexStr;
  NCCLCHECK(ncclCalloc(&hexStr, size));
  int hexOffset = 0;
  for (int i=0; i<size; i++) {
    char c = busId[i];
    if (c == '.' || c == ':') continue;
    if ((c >= '0' && c <= '9') ||
        (c >= 'A' && c <= 'F') ||
        (c >= 'a' && c <= 'f')) {
      hexStr[hexOffset++] = busId[i];
    } else break;
  }
  hexStr[hexOffset] = '\0';
  *id = strtol(hexStr, NULL, 16);
  free(hexStr);
  return ncclSuccess;
}

int ncclDebugLevel = -1;

void ncclDebugInit() {
  if (ncclDebugLevel != -1) return;
  const char* nccl_debug = getenv("NCCL_DEBUG");
  if (nccl_debug == NULL) {
    ncclDebugLevel = NCCL_LOG_NONE;
  } else if (strcasecmp(nccl_debug, "VERSION") == 0) {
    ncclDebugLevel = NCCL_LOG_VERSION;
  } else if (strcasecmp(nccl_debug, "WARN") == 0) {
    ncclDebugLevel = NCCL_LOG_WARN;
  } else if (strcasecmp(nccl_debug, "INFO") == 0) {
    ncclDebugLevel = NCCL_LOG_INFO;
  } else if (strcasecmp(nccl_debug, "ABORT") == 0) {
    ncclDebugLevel = NCCL_LOG_ABORT;
  } else if (strcasecmp(nccl_debug, "TRACE") == 0) {
    ncclDebugLevel = NCCL_LOG_TRACE;
  }
}

void ncclDebugLog(ncclDebugLogLevel level, unsigned long flags, const char *filefunc, int line, const char *fmt, ...) {
  if (ncclDebugLevel == -1) ncclDebugInit();
  if (level == NCCL_LOG_TRACE && ncclDebugLevel != NCCL_LOG_TRACE) return;
  char buffer[1024];
  size_t len;
  len = snprintf(buffer, sizeof(buffer),
                   "[%d:%d] ", node_model->nodeId, node_model->currRank);
  va_list args;
  va_start(args, fmt);
  vsprintf(buffer+len, fmt, args);
  va_end(args);
  printf("%s\n", buffer);
  if (level == NCCL_LOG_WARN) {
    fprintf(stderr,"[%d:%d] %s:%d TOPO EXPL ABORT\n",
            node_model->nodeId, node_model->currRank, filefunc, line);
    abort();
  }
}

ncclResult_t bootstrapAllGather(struct ncclComm* comm, struct allGather1Data_t * allGather1Data) {
  // AllGather1 - begin
  allGather1Data[comm->rank].peerInfo.rank = comm->rank;
  allGather1Data[comm->rank].peerInfo.cudaDev = node_model->rankToCudaDev(comm->rank);
  allGather1Data[comm->rank].peerInfo.gdrSupport = 1;
  allGather1Data[comm->rank].peerInfo.hostHash = node_model->hostHash;
  allGather1Data[comm->rank].peerInfo.pidHash = node_model->pidHash;
  allGather1Data[comm->rank].peerInfo.shmDev = 0x19;
  allGather1Data[comm->rank].peerInfo.busId = node_model->getGpuBusId(node_model->rankToCudaDev(comm->rank));
  return ncclSuccess;
}

ncclResult_t initTransportsRank_1(struct ncclComm* comm, struct allGather1Data_t *allGather1Data,
  struct allGather3Data_t *allGather3Data, struct ncclTopoGraph& treeGraph, struct ncclTopoGraph& ringGraph) {
  // We use 3 AllGathers
  // 1. { peerInfo, comm }
  // 2. ConnectTransport[nranks], ConnectValue[nranks]
  // 3. { nThreads, nrings, compCap, prev[MAXCHANNELS], next[MAXCHANNELS] }

  int rank = comm->rank;
  int nranks = comm->nRanks;
  //uint64_t commHash = getHash(commId->internal, NCCL_UNIQUE_ID_BYTES);
  //TRACE(NCCL_INIT, "comm %p, commHash %lx, rank %d nranks %d - BEGIN", comm, commHash, rank, nranks);
  //NCCLCHECK(bootstrapInit(commId, rank, nranks, &comm->bootstrap));

  // AllGather1 - begin
  //struct allGather1Data_t *allGather1Data;
  //NCCLCHECK(ncclCalloc(&allGather1Data, nranks));
  //allGather1Data[rank].comm = comm;
  struct ncclPeerInfo* myInfo = &allGather1Data[rank].peerInfo;
  //NCCLCHECK(fillInfo(comm, myInfo, commHash));
  //NCCLCHECK(bootstrapAllGather(comm, allGather1Data));

  NCCLCHECK(ncclCalloc(&comm->peerInfo, nranks));
  for (int i = 0; i < nranks; i++) {
    memcpy(comm->peerInfo+i, &allGather1Data[i].peerInfo, sizeof(struct ncclPeerInfo));
    if ((i != rank) && (comm->peerInfo[i].hostHash == myInfo->hostHash) && (comm->peerInfo[i].busId == myInfo->busId)) {
      WARN("Duplicate GPU detected : rank %d and rank %d both on CUDA device %x", rank, i, myInfo->busId);
      return ncclInvalidUsage;
    }
  }
  // AllGather1 data is used again below
  // AllGather1 - end

  // Topo detection / System graph creation
  NCCLCHECK(ncclTopoGetSystem(comm, &comm->topo));
  // Compute paths between GPUs and NICs
  NCCLCHECK(ncclTopoComputePaths(comm->topo, comm->peerInfo));
  // Remove inaccessible GPUs and unused NICs
  NCCLCHECK(ncclTopoTrimSystem(comm->topo, comm));
  // Recompute paths after trimming
  NCCLCHECK(ncclTopoComputePaths(comm->topo, comm->peerInfo));
  // Compute max speed to accelerate search
  NCCLCHECK(ncclTopoGetMaxSpeed(comm->topo));
  // Print final topology
  NCCLCHECK(ncclTopoPrint(comm->topo));

  // Get rings and trees
  //struct ncclTopoGraph treeGraph;
  treeGraph.pattern = NCCL_TOPO_PATTERN_SPLIT_TREE;
  treeGraph.crossNic = ncclParamCrossNic();
  // We communicate only half the data between node with trees on 2 nodes.
  NCCLCHECK(ncclTopoCompute(comm->topo, &treeGraph));
  NCCLCHECK(ncclTopoPrintGraph(comm->topo, &treeGraph));
  //struct ncclTopoGraph ringGraph;
  ringGraph.pattern = NCCL_TOPO_PATTERN_RING;
  ringGraph.crossNic = ncclParamCrossNic();
  NCCLCHECK(ncclTopoCompute(comm->topo, &ringGraph));
  NCCLCHECK(ncclTopoPrintGraph(comm->topo, &ringGraph));

  // AllGather3 - begin
  allGather3Data[rank].cudaCompCap = ncclCudaCompCap();
  allGather3Data[rank].nvlink = treeGraph.nvlink;
  allGather3Data[rank].nChannels = comm->nChannels = std::min(treeGraph.nChannels, ringGraph.nChannels);
  allGather3Data[rank].tree.sameChannels = treeGraph.sameChannels;
  allGather3Data[rank].tree.speedIntra = treeGraph.speedIntra;
  allGather3Data[rank].tree.speedInter = treeGraph.speedInter;
  allGather3Data[rank].tree.nvlink = treeGraph.nvlink;
  allGather3Data[rank].ring.sameChannels = ringGraph.sameChannels;
  allGather3Data[rank].ring.speedIntra = ringGraph.speedIntra;
  allGather3Data[rank].ring.speedInter = ringGraph.speedInter;
  allGather3Data[rank].ring.nvlink = ringGraph.nvlink;

  NCCLCHECK(ncclTopoPreset(comm, &treeGraph, &ringGraph, &allGather3Data[rank].topoRanks));
  //INFO(NCCL_GRAPH, "%d: nvlink %d nChannels %d tree.sameChannels %d tree.speedIntra %d tree.speedInter %d tree.nvlink %d ring.sameChannels %d ring.speedIntra %d ring.speedInter %d ring.nvlink %d",
  //  rank, allGather3Data[rank].nvlink, allGather3Data[rank].nChannels, allGather3Data[rank].tree.sameChannels, allGather3Data[rank].tree.speedIntra, allGather3Data[rank].tree.speedInter, allGather3Data[rank].tree.nvlink,
  //  allGather3Data[rank].ring.sameChannels, allGather3Data[rank].ring.speedIntra, allGather3Data[rank].ring.speedInter, allGather3Data[rank].ring.nvlink);
  //INFO(NCCL_GRAPH, "  ringRecv %d ringSend %d ringPrev %d ringNext %d treeUpRecv %d treeUpSend %d treeDnRecv %d treeDnSend %d",
  //  allGather3Data[rank].topoRanks.ringRecv[0], allGather3Data[rank].topoRanks.ringSend[0], allGather3Data[rank].topoRanks.ringPrev[0], allGather3Data[rank].topoRanks.ringNext[0],
  //  allGather3Data[rank].topoRanks.treeUpRecv[0], allGather3Data[rank].topoRanks.treeUpSend[0], allGather3Data[rank].topoRanks.treeDnRecv[0], allGather3Data[rank].topoRanks.treeDnSend[0]);
  return ncclSuccess;
}

template <int type>
static ncclResult_t selectTransport(struct ncclTopoSystem* topo, struct ncclTopoGraph* graph, struct ncclPeerInfo* myInfo, struct ncclPeerInfo* peerInfo, struct ncclConnect* connect, struct ncclConnector* connector, int buffSize, int channelId) {
  for (int t=0; t<NTRANSPORTS; t++) {
    struct ncclTransport *transport = ncclTransports+t;
    struct ncclTransportComm* transportComm = type == 1 ? &transport->send : &transport->recv;
    int ret = 0;
    NCCLCHECK(transport->canConnect(&ret, topo, graph, myInfo, peerInfo));
    if (ret) {
      //cpu_set_t affinitySave;
      //sched_getaffinity(0, sizeof(cpu_set_t), &affinitySave);
      //int cudaDev;
      //CUDACHECK(hipGetDevice(&cudaDev));
      //setCpuAffinity(cudaDev);
      connector->transportComm = transportComm;
      NCCLCHECK(transportComm->setup(topo, graph, myInfo, peerInfo, connect, connector, buffSize, channelId));
      //sched_setaffinity(0, sizeof(cpu_set_t), &affinitySave);
      return ncclSuccess;
    }
  }
  WARN("No transport found !");
  return ncclInternalError;
}

static ncclResult_t p2pSetup(struct ncclComm* comm, struct ncclTopoGraph* graph, struct ncclChannel* channel, int nrecv, int* peerRecv, int nsend, int* peerSend) {
  TRACE(NCCL_INIT, "nsend %d nrecv %d", nsend, nrecv);
  uint32_t nSkippedSend = 0, nSkippedRecv = 0; /* for tracing */
  struct ncclConnect connect;
  struct ncclConnector* conn;
  for (int i=0; i<nrecv; i++) {
    int peer = peerRecv[i];
    if (peer == -1) continue;
    conn = &channel->peers[peer].recv;
    if (conn->connected) { ++nSkippedRecv; continue; }
    memset(&connect, 0, sizeof(connect));
    NCCLCHECK(selectTransport<0>(comm->topo, graph, comm->peerInfo+comm->rank, comm->peerInfo+peer, &connect, conn, channel->buffSize, channel->id));
    //NCCLCHECK(bootstrapSend(comm->bootstrap, peer, &connect, sizeof(struct ncclConnect)));
  }
  for (int i=0; i<nsend; i++) {
    int peer = peerSend[i];
    if (peer == -1) continue;
    conn = &channel->peers[peer].send;
    if (conn->connected) { ++nSkippedSend; continue; }
    memset(&connect, 0, sizeof(connect));
    NCCLCHECK(selectTransport<1>(comm->topo, graph, comm->peerInfo+comm->rank, comm->peerInfo+peer, &connect, conn, channel->buffSize, channel->id));
    //NCCLCHECK(bootstrapSend(comm->bootstrap, peer, &connect, sizeof(struct ncclConnect)));
  }
  for (int i=0; i<nsend; i++) {
    int peer = peerSend[i];
    if (peer == -1) continue;
    conn = &channel->peers[peer].send;
    if (conn->connected) {++nSkippedSend; continue; }
    memset(&connect, 0, sizeof(connect));
    //NCCLCHECK(bootstrapRecv(comm->bootstrap, peer, &connect, sizeof(struct ncclConnect)));
    //NCCLCHECK(conn->transportComm->connect(&connect, conn));
    conn->connected = 1;
  }
  for (int i=0; i<nrecv; i++) {
    int peer = peerRecv[i];
    if (peer == -1) continue;
    conn = &channel->peers[peer].recv;
    if (conn->connected) {++nSkippedRecv; continue; }
    memset(&connect, 0, sizeof(connect));
    //CCLCHECK(bootstrapRecv(comm->bootstrap, peer, &connect, sizeof(struct ncclConnect)));
    //NCCLCHECK(conn->transportComm->connect(&connect, conn));
    conn->connected = 1;
  }
  TRACE(NCCL_INIT, "nsend %d nrecv %d nSkippedSend %u nSkippedRecv %u - DONE", nsend, nrecv, nSkippedSend, nSkippedRecv);
  return ncclSuccess;
}

ncclResult_t initChannel(struct ncclComm* comm, int channelid) {
  struct ncclChannel* channel = comm->channels+channelid;
  channel->id = channelid;

  // Setup intermediate buffering
  //channel->buffSize = ncclParamBuffsize();

  // Ring index to user rank table.
  //NCCLCHECK(ncclCudaCalloc(&channel->ring.devUserRanks, comm->nRanks));
  NCCLCHECK(ncclCalloc(&channel->ring.userRanks, comm->nRanks));

  // Communication structures with peers.
  //NCCLCHECK(ncclCudaCalloc(&channel->devPeers, comm->nRanks));
  NCCLCHECK(ncclCalloc(&channel->peers, comm->nRanks));
  for (size_t i=0; i<comm->nRanks; ++i) {
    channel->peers[i].send.comm = comm;
    channel->peers[i].recv.comm = comm;
  }

  // Per-channel operation list.
  //NCCLCHECK(ncclCudaHostAlloc((void**)&channel->collectives, (void**)&channel->devCollectives, sizeof(struct ncclColl)*NCCL_MAX_OPS));
  return ncclSuccess;
}

static ncclResult_t setupChannel(struct ncclComm* comm, int channelId, int rank, int nranks, int* ringRanks) {
  TRACE(NCCL_INIT, "rank %d nranks %d", rank, nranks);
  NCCLCHECK(initChannel(comm, channelId));

  struct ncclRing* ring = &comm->channels[channelId].ring;
  // Reorganize ranks to start with rank.
  int shift;
  for (shift = 0; shift<nranks; shift++) {
    if (ringRanks[shift] == rank) {
      break;
    }
  }
  for (int i=0; i<nranks; i++) {
    ring->userRanks[i] = ringRanks[(i+shift)%nranks];
  }
  return ncclSuccess;
}

ncclResult_t initTransportsRank_3(struct ncclComm* comm, struct allGather3Data_t *allGather3Data, struct ncclTopoGraph& treeGraph,
  struct ncclTopoGraph& ringGraph) {
  int rank = comm->rank;
  int nranks = comm->nRanks;
  //NCCLCHECK(bootstrapAllGather(comm->bootstrap, allGather3Data, sizeof(*allGather3Data)));

  // Determine nNodes, firstRanks, ...
  int* nodesFirstRank;
  NCCLCHECK(ncclCalloc(&nodesFirstRank, nranks));
  for (int i=0; i<nranks; i++) {
    int node = -1;
    int firstRank = allGather3Data[i].topoRanks.ringRecv[0];
    for (int n=0; n<comm->nNodes; n++) {
      if (nodesFirstRank[n] == firstRank) node = n;
    }
    if (node == -1) {
      node = comm->nNodes++;
      nodesFirstRank[node] = firstRank;
    }
    if (i == comm->rank) comm->node = node;
  }

  // Determine the minimum CUDA Compute capability of all GPUs
  int myCompCap = allGather3Data[rank].cudaCompCap;
  int minCompCap = myCompCap, maxCompCap = myCompCap;
  for (int i = 0; i < nranks; i++) {
    minCompCap = std::min(allGather3Data[i].cudaCompCap, minCompCap);
    maxCompCap = std::max(allGather3Data[i].cudaCompCap, maxCompCap);
  }

  comm->nvlink = 1;
  for (int i = 0; i < nranks; i++) comm->nvlink &= allGather3Data[i].nvlink;

  int nChannelsOrig = comm->nChannels;
  struct ncclTopoRanks** allTopoRanks;
  NCCLCHECK(ncclCalloc(&allTopoRanks, comm->nRanks));
  for (int i=0; i<nranks; i++) {
    allTopoRanks[i] = &allGather3Data[i].topoRanks;
    // Make sure we align all ranks so that the tuning is consistent across ranks
    treeGraph.nChannels = ringGraph.nChannels = comm->nChannels = std::min(allGather3Data[i].nChannels, comm->nChannels);
    treeGraph.sameChannels = std::min(allGather3Data[i].tree.sameChannels, treeGraph.sameChannels);
    treeGraph.speedIntra = std::min(allGather3Data[i].tree.speedIntra, treeGraph.speedIntra);
    treeGraph.speedInter = std::min(allGather3Data[i].tree.speedInter, treeGraph.speedInter);
    treeGraph.nvlink = std::min(allGather3Data[i].tree.nvlink, treeGraph.nvlink);
    ringGraph.sameChannels = std::min(allGather3Data[i].ring.sameChannels, ringGraph.sameChannels);
    ringGraph.speedIntra = std::min(allGather3Data[i].ring.speedIntra, ringGraph.speedIntra);
    ringGraph.speedInter = std::min(allGather3Data[i].ring.speedInter, ringGraph.speedInter);
    ringGraph.nvlink = std::min(allGather3Data[i].ring.nvlink, ringGraph.nvlink);
  }

  if (comm->nChannels < nChannelsOrig) {
    // We started duplicating channels during Preset(), so we need to move the
    // duplicated channels since we have removed some.
    for (int i=0; i<comm->nChannels; i++) memcpy(comm->channels+comm->nChannels+i, comm->channels+nChannelsOrig+i, sizeof(struct ncclChannel));
  }

  int *rings;
  NCCLCHECK(ncclCalloc(&rings, nranks*MAXCHANNELS));

  char line[1024];
  sprintf(line, "nodesFirstRank: ");
  int offset = strlen(line);
  for (int i=0; i<comm->nNodes; i++) {
    sprintf(line+offset, "%d ", nodesFirstRank[i]);
    offset = strlen(line);
  }
  INFO(NCCL_INIT, "%s", line);

  NCCLCHECK(ncclTopoPostset(comm, nodesFirstRank, allTopoRanks, rings));

  free(allTopoRanks);
  free(nodesFirstRank);

  // AllGather3 - end

  TRACE(NCCL_INIT, "rank %d nranks %d - BUILT %d TREES/RINGS", rank, nranks, comm->nChannels);

  line[0]='\0';
  for (int c=0; c<comm->nChannels; c++) {
    struct ncclTree* treeUp = &comm->channels[c].treeUp;
    struct ncclTree* treeDn = &comm->channels[c].treeDn;
    snprintf(line+strlen(line), 1023-strlen(line), " [%d] %d/%d/%d->%d->%d|%d->%d->%d/%d/%d",
        c, treeUp->down[0], treeUp->down[1], treeUp->down[2], rank, treeUp->up,
        treeDn->up, rank, treeDn->down[0], treeDn->down[1], treeDn->down[2]);
  }
  line[1023] = '\0';
  INFO(NCCL_INIT, "Trees%s", line);

  free(rings);

  // Done with AllGather1 data
  //free(allGather1Data);

  TRACE(NCCL_INIT, "rank %d nranks %d - DONE", rank, nranks);

  // Connect with prev/next for each ring
  struct ncclConnect *connect;
  NCCLCHECK(ncclCalloc(&connect, 2));
  for (int c=0; c<comm->nChannels; c++) {
    struct ncclChannel* channel = comm->channels+c;
    NCCLCHECK(setupChannel(comm, c, rank, nranks, rings+c*nranks));
    if (comm->nRanks == 1) continue;
    NCCLCHECK(p2pSetup(comm, &ringGraph, channel, 1, &channel->ring.prev, 1, &channel->ring.next));
    NCCLCHECK(p2pSetup(comm, &treeGraph, channel, NCCL_MAX_TREE_ARITY, channel->treeUp.down, 1, &channel->treeUp.up));
    NCCLCHECK(p2pSetup(comm, &treeGraph, channel, 1, &channel->treeDn.up, NCCL_MAX_TREE_ARITY, channel->treeDn.down));
  }
  TRACE(NCCL_INIT, "rank %d nranks %d - CONNECTED %d RINGS AND TREES", rank, nranks, comm->nChannels);
  free(connect);

  return ncclSuccess;
}
