/*************************************************************************
 * Copyright (c) 2015-2022, NVIDIA CORPORATION. All rights reserved.
 * Modifications Copyright (c) 2019-2022 Advanced Micro Devices, Inc. All rights reserved.
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
#include "xml.h"
#include "coll_net.h"
#include "model.h"
#include "utils.h"
#include "rocm_smi/rocm_smi.h"

const char* ncclFuncStr[NCCL_NUM_FUNCTIONS+2] = { "Broadcast", "Reduce", "AllGather", "ReduceScatter", "AllReduce", "SendRecv", "AllToAllPivot" };
const char* ncclAlgoStr[NCCL_NUM_ALGORITHMS] = { "Tree", "Ring", "CollNet" };
const char* ncclProtoStr[NCCL_NUM_PROTOCOLS] = { "LL", "LL128", "Simple" };

extern NodeModel *node_model;

NCCL_PARAM(CollNetEnable, "COLLNET_ENABLE", 0);
NCCL_PARAM(GraphDumpFileRank, "GRAPH_DUMP_FILE_RANK", 0);
NCCL_PARAM(CollNetNodeThreshold, "COLLNET_NODE_THRESHOLD", 2);

thread_local int ncclDebugNoWarn = 0;
ncclCollNet_t* ncclCollNet = NULL;

// Get current Compute Capability
int ncclCudaCompCap() {
  int ccMajor = 1, ccMinor = 0;
  return ccMajor*10+ccMinor;
}

ncclResult_t int64ToBusId(int64_t id, char* busId) {
  sprintf(busId, "%04lx:%02lx:%02lx.%01lx", (id) >> 20, (id & 0xff000) >> 12, (id & 0xff0) >> 4, (id & 0xf));
  return ncclSuccess;
}

ncclResult_t busIdToInt64(const char* busId, int64_t* id) {
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
    ncclDebugLevel = NCCL_LOG_INFO;
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
  if (ncclDebugLevel < level || ((flags & (NCCL_INIT|NCCL_GRAPH|NCCL_TUNING)) == 0)) return;

  char buffer[1024];
  size_t len = 0;
  if (node_model) len = snprintf(buffer, sizeof(buffer),
    "[%d:%d] ", node_model->nodeId, node_model->currRank);
  va_list args;
  va_start(args, fmt);
  vsprintf(buffer+len, fmt, args);
  va_end(args);
  printf("%s\n", buffer);
#if 0
  if (level == NCCL_LOG_WARN) {
    fprintf(stderr,"[%d:%d] %s:%d TOPO EXPL ABORT\n",
            node_model->nodeId, node_model->currRank, filefunc, line);
    abort();
  }
#endif
}

ncclResult_t ncclTopoGetSystem(const char* xmlTopoFile, struct ncclTopoSystem** system) {
  struct ncclXml* xml;
  NCCLCHECK(ncclCalloc(&xml, 1));
  NCCLCHECK(ncclTopoGetXmlFromFile(xmlTopoFile, xml, 0));
  NCCLCHECK(ncclTopoGetSystemFromXml(xml, system));
  free(xml);
  return ncclSuccess;
}


void initCollNet() {
  if (ncclParamCollNetEnable() == 1 && ncclCollNet == 0)
    ncclCollNet = (ncclCollNet_t*)0x12345678;
}

ncclResult_t initChannel(struct ncclComm* comm, int channelid) {
  struct ncclChannel* channel = comm->channels+channelid;
  if (channel->id != -1) return ncclSuccess;
  channel->id = channelid;

  // Ring index to user rank table.
  //NCCLCHECK(ncclCudaCalloc(&channel->ring.devUserRanks, comm->nRanks));
  NCCLCHECK(ncclCalloc(&channel->ring.userRanks, comm->nRanks));

  // Communication structures with peers.
  //NCCLCHECK(ncclCudaCalloc(&channel->devPeers, comm->nRanks+1)); // The extra one rank is for collnet root (i.e. network)
  NCCLCHECK(ncclCalloc(&channel->peers, comm->nRanks+1));
  for (size_t i=0; i<comm->nRanks+1; ++i) {
    for (int b=0; b<NCCL_MAX_CONNS; b++) {
      channel->peers[i].send[b].comm = comm;
      channel->peers[i].recv[b].comm = comm;
    }
  }

  // Per-channel operation list.
  //NCCLCHECK(ncclCudaHostCalloc(&channel->workFifo, NCCL_MAX_OPS));
  //if (ncclGdrCopy != NULL && ncclParamGdrCopyFifoEnable() == 1) {
    // GDRCOPY support
    // We allocate a workFifo in GDR mapped CUDA memory
    // But we still allocate the Host workFifo so that we
    // can copy the work elements to CUDA memory on kernel launch
    //NCCLCHECK(ncclGdrCudaCalloc(&channel->workFifoGdr, &channel->workFifoDev, NCCL_MAX_OPS, &channel->gdrMemDesc));
  //} else {
    // The device workFifo is the Host one
    //channel->workFifoDev = channel->workFifo;
  //}

  return ncclSuccess;
}

ncclResult_t fillInfo(struct ncclComm* comm, struct ncclPeerInfo* info, uint64_t commHash) {
  info->rank = comm->rank;
  info->cudaDev = node_model->rankToCudaDev(comm->rank);
  info->hostHash = node_model->hostHash;
  info->pidHash = node_model->pidHash;

  // Get the device MAJOR:MINOR of /dev/shm so we can use that
  // information to decide whether we can use SHM for inter-process
  // communication in a container environment
  //struct stat statbuf;
  //SYSCHECK(stat("/dev/shm", &statbuf), "stat");
  info->shmDev = 0x19;

  info->busId = node_model->getGpuBusId(comm->rank);

  // detect if fine grained memory is available on this GPU
  info->hasFineGrain = true;
  info->gdrSupport = 1;

  info->comm = comm;
  info->cudaCompCap = 1;
  return ncclSuccess;
}

static ncclResult_t setupChannel(struct ncclComm* comm, int channelId, int rank, int nranks, int* ringRanks) {
  TRACE(NCCL_INIT, "rank %d nranks %d", rank, nranks);
  NCCLCHECK(initChannel(comm, channelId));

  struct ncclRing* ring = &comm->channels[channelId].ring;
  // Find our ring-distance from rank zero and reorganize ranks to start with rank.
  int ixZero=0, ixRank=0;
  for (int i=0; i < nranks; i++) {
    if (ringRanks[i] == 0) ixZero = i;
    if (ringRanks[i] == rank) ixRank = i;
  }
  ring->index = (ixRank-ixZero + nranks)%nranks;
  for (int i=0; i<nranks; i++) {
    ring->userRanks[i] = ringRanks[(i+ixRank)%nranks];
  }
  return ncclSuccess;
}

template <int type>
static ncclResult_t selectTransport(struct ncclComm* comm, struct ncclTopoGraph* graph, struct ncclConnect* connect, int channelId, int peer, int connIndex, int* transportType) {
  struct ncclPeerInfo* myInfo = comm->peerInfo+comm->rank;
  struct ncclPeerInfo* peerInfo = comm->peerInfo+peer;
  struct ncclConnector* connector = (type == 1) ? comm->channels[channelId].peers[peer].send + connIndex :
                                                  comm->channels[channelId].peers[peer].recv + connIndex;
  // handle intra-node network connections
  int n1 = -1, n2 = -1;
  if (connIndex == NCCL_CONN_IDX_P2P_NET) {
    NCCLCHECK(ncclTopoGetIntraNetDev(comm->topo, comm->rank, graph, channelId, (type == 1) ? 1 : 0, &n1));
    NCCLCHECK(ncclTopoGetIntraNetDev(comm->topo, peer, graph, channelId, (type == 1) ? 0 : 1, &n2));
  }
  bool xgmi;
  NCCLCHECK(ncclTopoGetLinkType(comm->topo, myInfo->cudaDev, peerInfo->cudaDev, &xgmi));

  for (int t=0; t<NTRANSPORTS; t++) {
    if (graph == NULL && connIndex == NCCL_CONN_IDX_P2P_NET && (t == TRANSPORT_SHM || (!xgmi && t == TRANSPORT_P2P))) continue;
    if (graph && n1 >= 0 && n2 >= 0 && t != TRANSPORT_NET) continue;
    struct ncclTransport *transport = ncclTransports[t];
    struct ncclTransportComm* transportComm = type == 1 ? &transport->send : &transport->recv;
    int ret = 0;
    NCCLCHECK(transport->canConnect(&ret, comm->topo, graph, myInfo, peerInfo));
    if (ret) {
      connector->transportComm = transportComm;
      NCCLCHECK(transportComm->setup(comm, graph, myInfo, peerInfo, connect, connector, channelId, connIndex));
      if (transportType) *transportType = t;
      return ncclSuccess;
    }
  }
  WARN("No transport found for rank %d[%lx] -> rank %d[%lx]", myInfo->rank, myInfo->busId, peerInfo->rank, peerInfo->busId);
  return ncclSystemError;
}

ncclResult_t ncclTransportP2pConnect(struct ncclComm* comm, int channelId, int nrecv, int* peerRecv, int nsend, int* peerSend, int connIndex) {
  TRACE(NCCL_INIT, "nsend %d nrecv %d", nsend, nrecv);
  struct ncclChannel* channel = &comm->channels[channelId];
  uint32_t mask = 1 << channelId;
  for (int i=0; i<nrecv; i++) {
    int peer = peerRecv[i];
    if (peer == -1 || peer >= comm->nRanks || peer == comm->rank || channel->peers[peer].recv[connIndex].connected) continue;
    comm->connectRecv[peer+comm->nRanks*(connIndex == NCCL_CONN_IDX_P2P_NET ? NCCL_CONN_IDX_P2P_NET : 0)] |= mask;
  }
  for (int i=0; i<nsend; i++) {
    int peer = peerSend[i];
    if (peer == -1 || peer >= comm->nRanks || peer == comm->rank || channel->peers[peer].send[connIndex].connected) continue;
    comm->connectSend[peer+comm->nRanks*(connIndex == NCCL_CONN_IDX_P2P_NET ? NCCL_CONN_IDX_P2P_NET : 0)] |= mask;
  }
  return ncclSuccess;
}

void dumpData(struct ncclConnect* data, int ndata) {
  for (int n=0; n<ndata; n++) {
    printf("[%d] ", n);
    uint8_t* d = (uint8_t*)data;
    for (int i=0; i<sizeof(struct ncclConnect); i++) printf("%02x", d[i]);
    printf("\n");
  }
}

ncclResult_t ncclTransportP2pSetup(struct ncclComm* comm, struct ncclTopoGraph* graph, int connIndex, int* highestTransportType/*=NULL*/) {
  // Stream used during transport setup; need for P2P pre-connect + CUDA Graph
  int highestType = TRANSPORT_P2P;  // track highest transport type

  //hipStream_t transportSetupStream;
  //CUDACHECK(hipStreamCreateWithFlags(&transportSetupStream, hipStreamNonBlocking));

  struct ncclConnect data[2*MAXCHANNELS];
  for (int i=1; i<comm->nRanks; i++) {
    int bootstrapTag = (i<<8) + (graph ? graph->id+1 : 0);
    int recvPeer = (comm->rank - i + comm->nRanks) % comm->nRanks;
    int sendPeer = (comm->rank + i) % comm->nRanks;
    uint32_t recvMask = comm->connectRecv[recvPeer+comm->nRanks*(connIndex == NCCL_CONN_IDX_P2P_NET ? NCCL_CONN_IDX_P2P_NET : 0)];
    uint32_t sendMask = comm->connectSend[sendPeer+comm->nRanks*(connIndex == NCCL_CONN_IDX_P2P_NET ? NCCL_CONN_IDX_P2P_NET : 0)];

    struct ncclConnect* recvData = data;
    int sendChannels = 0, recvChannels = 0;
    int type;
    for (int c=0; c<MAXCHANNELS; c++) {
      if (recvMask & (1<<c)) {
        NCCLCHECK(selectTransport<0>(comm, graph, recvData+recvChannels++, c, recvPeer, connIndex, &type));
        if (type > highestType) highestType = type;
      }
    }
    struct ncclConnect* sendData = recvData+recvChannels;
    for (int c=0; c<MAXCHANNELS; c++) {
      if (sendMask & (1<<c)) {
        NCCLCHECK(selectTransport<1>(comm, graph, sendData+sendChannels++, c, sendPeer, connIndex, &type));
        if (type > highestType) highestType = type;
      }
    }

    if (sendPeer == recvPeer) {
      if (recvChannels+sendChannels) {
         //NCCLCHECK(bootstrapSend(comm->bootstrap, recvPeer, bootstrapTag, data, sizeof(struct ncclConnect)*(recvChannels+sendChannels)));
         //NCCLCHECK(bootstrapRecv(comm->bootstrap, recvPeer, bootstrapTag, data, sizeof(struct ncclConnect)*(recvChannels+sendChannels)));
         sendData = data;
         recvData = data+sendChannels;
      }
    } else {
      //if (recvChannels) NCCLCHECK(bootstrapSend(comm->bootstrap, recvPeer, bootstrapTag, recvData, sizeof(struct ncclConnect)*recvChannels));
      //if (sendChannels) NCCLCHECK(bootstrapSend(comm->bootstrap, sendPeer, bootstrapTag, sendData, sizeof(struct ncclConnect)*sendChannels));
      //if (sendChannels) NCCLCHECK(bootstrapRecv(comm->bootstrap, sendPeer, bootstrapTag, sendData, sizeof(struct ncclConnect)*sendChannels));
      //if (recvChannels) NCCLCHECK(bootstrapRecv(comm->bootstrap, recvPeer, bootstrapTag, recvData, sizeof(struct ncclConnect)*recvChannels));
    }

    for (int c=0; c<MAXCHANNELS; c++) {
      if (sendMask & (1<<c)) {
        struct ncclConnector* conn = comm->channels[c].peers[sendPeer].send + connIndex;
        //NCCLCHECK(conn->transportComm->connect(comm, sendData++, 1, comm->rank, conn));
        conn->connected = 1;
        //CUDACHECK(hipMemcpyAsync(&comm->channels[c].devPeers[sendPeer].send[connIndex], &conn->conn, sizeof(struct ncclConnInfo), hipMemcpyHostToDevice, transportSetupStream));
      }
    }
    for (int c=0; c<MAXCHANNELS; c++) {
      if (recvMask & (1<<c)) {
        struct ncclConnector* conn = comm->channels[c].peers[recvPeer].recv + connIndex;
        //NCCLCHECK(conn->transportComm->connect(comm, recvData++, 1, comm->rank, conn));
        conn->connected = 1;
        //CUDACHECK(hipMemcpyAsync(&comm->channels[c].devPeers[recvPeer].recv[connIndex], &conn->conn, sizeof(struct ncclConnInfo), hipMemcpyHostToDevice, transportSetupStream));
      }
    }
    comm->connectRecv[recvPeer+comm->nRanks*(connIndex == NCCL_CONN_IDX_P2P_NET ? NCCL_CONN_IDX_P2P_NET : 0)] = comm->connectSend[sendPeer+comm->nRanks*(connIndex == NCCL_CONN_IDX_P2P_NET ? NCCL_CONN_IDX_P2P_NET : 0)] = 0;
  }
  //CUDACHECK(hipStreamSynchronize(transportSetupStream));
  //CUDACHECK(hipStreamDestroy(transportSetupStream));
  if (highestTransportType != NULL) *highestTransportType = highestType;
  return ncclSuccess;
}

extern struct ncclTransport collNetTransport;

// All ranks must participate in collNetSetup call
// We do not NCCLCHECK this call because we would fall back to P2P network in case CollNet setup fails
int ncclTransportCollNetSetup(struct ncclComm* comm, struct ncclTopoGraph* collNetGraph, struct ncclChannel* channel, int masterRank, int masterPeer, int collNetGraphChannelId, int type) {
  int fail = 1;
  int rank = comm->rank;
  int nranks = comm->nRanks;
  int nMasters = comm->nNodes;
  int rankInCollNet = -1;
  int isMaster = (rank == masterRank) ? 1 : 0;
  struct {
    int collNetRank;
    ncclConnect connect;
  } sendrecvExchange;

  // check if we can connect to collnet, whose root is the nranks-th rank
  struct ncclPeerInfo *myInfo = comm->peerInfo+rank, *peerInfo = comm->peerInfo+nranks;
  peerInfo->rank = nranks;

  // send master receives connect info from peer recv master
  if (isMaster && type == collNetSend) {
    //NCCLCHECK(bootstrapRecv(comm->bootstrap, masterPeer, collNetGraph->id, &sendrecvExchange, sizeof(sendrecvExchange)));
    rankInCollNet = sendrecvExchange.collNetRank;
    TRACE(NCCL_INIT, "CollNet [send] : rank %d collNetRank %d collNetNranks %d received connect from rank %d", rank, rankInCollNet, nMasters, masterPeer);
  }

  // select
  struct ncclChannelPeer* root = channel->peers+nranks;
  // connector index: 0 for recv, 1 for send
  struct ncclConnector* conn = (type == collNetRecv) ? root->recv+type : root->send+type;
  struct ncclTransportComm* transportComm = (type == collNetRecv) ? &(collNetTransport.recv) : &(collNetTransport.send);
  conn->transportComm = transportComm;
  // setup
  struct ncclConnect myConnect;
  if (isMaster) {
    NCCLCHECK(transportComm->setup(comm, collNetGraph, myInfo, peerInfo, &myConnect, conn, collNetGraphChannelId, type));
  }
  // prepare connect handles
  ncclResult_t res;
  struct {
    int isMaster;
    ncclConnect connect;
  } *allConnects = NULL;
  ncclConnect *masterConnects = NULL;
  NCCLCHECK(ncclCalloc(&masterConnects, nMasters));
  if (type == collNetRecv) {  // recv side: AllGather
    // all ranks must participate
    NCCLCHECK(ncclCalloc(&allConnects, nranks));
    allConnects[rank].isMaster = isMaster;
    memcpy(&(allConnects[rank].connect), &myConnect, sizeof(struct ncclConnect));
    //NCCLCHECKGOTO(bootstrapAllGather(comm->bootstrap, allConnects, sizeof(*allConnects)), res, cleanup);
    // consolidate
    int c = 0;
    for (int r = 0; r < nranks; r++) {
      if (allConnects[r].isMaster) {
        memcpy(masterConnects+c, &(allConnects[r].connect), sizeof(struct ncclConnect));
        if (r == rank) rankInCollNet = c;
        c++;
      }
    }
  } else { // send side : copy in connect info received from peer recv master
    //if (isMaster) memcpy(masterConnects+rankInCollNet, &(sendrecvExchange.connect), sizeof(struct ncclConnect));
  }
  // connect
  if (isMaster) {
    //NCCLCHECKGOTO(transportComm->connect(comm, masterConnects, nMasters, rankInCollNet, conn), res, cleanup);
    struct ncclDevChannelPeer* devRoot = channel->devPeers+nranks;
    struct ncclConnInfo* devConnInfo = (type == collNetRecv) ? devRoot->recv+type : devRoot->send+type;
    //CUDACHECKGOTO(hipMemcpy(devConnInfo, &conn->conn, sizeof(struct ncclConnInfo), hipMemcpyHostToDevice), res, cleanup);
  }
  // recv side sends connect info to send side
  if (isMaster && type == collNetRecv) {
    sendrecvExchange.collNetRank = rankInCollNet;
    //memcpy(&sendrecvExchange.connect, masterConnects+rankInCollNet, sizeof(struct ncclConnect));
    //NCCLCHECKGOTO(bootstrapSend(comm->bootstrap, masterPeer, collNetGraph->id, &sendrecvExchange, sizeof(sendrecvExchange)), res, cleanup);
    TRACE(NCCL_INIT, "CollNet [recv] : rank %d collNetRank %d collNetNranks %d sent connect to rank %d", rank, rankInCollNet, nMasters, masterPeer);
  }
  fail = 0;
cleanup:
  if (allConnects != NULL) free(allConnects);
  if (masterConnects != NULL) free(masterConnects);
  return fail;
}

ncclResult_t ncclTransportCollNetCheck(struct ncclComm* comm, int collNetSetupFail) {
  // AllGather collNet setup results
  int allGatherFailures[NCCL_MAX_LOCAL_RANKS] = {0};
  allGatherFailures[comm->localRank] = collNetSetupFail;
  //NCCLCHECK(bootstrapIntraNodeAllGather(comm->bootstrap, comm->localRankToRank, comm->localRank, comm->localRanks, allGatherFailures, sizeof(int)));
  for (int i=0; i<comm->localRanks; i++) {
    if (allGatherFailures[i] != 0) {
      collNetSetupFail = 1;
      break;
    }
  }
  if (collNetSetupFail) {
    if (comm->localRank == 0) WARN("Cannot initialize CollNet, using point-to-point network instead");
    return ncclSystemError;
  }
  return ncclSuccess;
}

ncclResult_t ncclTransportCollNetFree(struct ncclComm* comm) {
  // Free collNet resources
  for (int r=0; r<comm->nChannels; r++) {
    struct ncclChannel* channel = comm->channels+r;
    struct ncclChannelPeer* peer = channel->peers+comm->nRanks;
    for (int b=0; b<NCCL_MAX_CONNS; b++) {
      struct ncclConnector* send = peer->send + b;
      //if (send->transportResources && send->transportComm) NCCLCHECK(send->transportComm->free(send));
      send->transportResources = NULL; // avoid double free
    }
    for (int b=0; b<NCCL_MAX_CONNS; b++) {
      struct ncclConnector* recv = peer->recv + b;
      //if (recv->transportResources && recv->transportComm) NCCLCHECK(recv->transportComm->free(recv));
      recv->transportResources = NULL; // avoid double free
    }
  }
  return ncclSuccess;
}

RCCL_PARAM(P2pNetDisable, "P2P_NET_DISABLE", 0);
RCCL_PARAM(PivotAlltoallEnable, "PIVOT_ALLTOALL_ENABLE", 0);

ncclResult_t initTransportsRank_1(struct ncclComm* comm, struct allGather3Data_t *allGather3Data,
  struct ncclTopoGraph& treeGraph, struct ncclTopoGraph& ringGraph, struct ncclTopoGraph& collNetGraph) {
  // We use 2 AllGathers
  // 1. { peerInfo, comm, compCap}
  // 2. { nChannels, graphInfo, topoRanks }

  int rank = comm->rank;
  int nranks = comm->nRanks;
  //uint64_t commHash = getHash(commId->internal, NCCL_UNIQUE_ID_BYTES);
  //TRACE(NCCL_INIT, "comm %p, commHash %lx, rank %d nranks %d - BEGIN", comm, commHash, rank, nranks);
  // [RCCL] Collect the PID of the root
  int rootPid;
  //NCCLCHECK(bootstrapInit(commId, comm));
  // [/RCCL]

  // AllGather1 - begin
  //NCCLCHECK(ncclCalloc(&comm->peerInfo, nranks+1)); // Extra rank to represent CollNet root
  //NCCLCHECK(fillInfo(comm, comm->peerInfo+rank, comm->rank));
  //NCCLCHECK(bootstrapAllGather(comm->bootstrap, comm->peerInfo, sizeof(struct ncclPeerInfo)));

  for (int i = 0; i < nranks; i++) {
    if ((i != rank) && (comm->peerInfo[i].hostHash == comm->peerInfo[rank].hostHash) && (comm->peerInfo[i].busId == comm->peerInfo[rank].busId)) {
      WARN("Duplicate GPU detected : rank %d and rank %d both on CUDA device %lx", rank, i, comm->peerInfo[rank].busId);
      return ncclInvalidUsage;
    }
  }

  // AllGather1 - end

  // Topo detection / System graph creation
  //NCCLCHECK(ncclTopoGetSystem(comm, &comm->topo));
  // save nRanks to ncclTopoSystem as indicator of multi-node
  comm->topo->nRanks = comm->nRanks;
  // init netGdrLevel
  comm->topo->netGdrLevel = -2;
  // init Pivot A2A related fields
  comm->topo->pivotA2AEnabled = false;
  comm->topo->pivotA2ANumBiRings = 0;
  // Compute paths between GPUs and NICs
  NCCLCHECK(ncclTopoComputePaths(comm->topo, comm));
  // Remove inaccessible GPUs and unused NICs
  NCCLCHECK(ncclTopoTrimSystem(comm->topo, comm));
  // Recompute paths after trimming
  NCCLCHECK(ncclTopoComputePaths(comm->topo, comm));
  // Init search
  NCCLCHECK(ncclTopoSearchInit(comm->topo));
  // Print final topology
  NCCLCHECK(ncclTopoPrint(comm->topo));

  // Set Affinity to a CPU local the our GPU, so that all memory we allocate
  // on the host is local.
  //NCCLCHECK(ncclTopoGetCpuAffinity(comm->topo, comm->rank, &comm->cpuAffinity));
  //cpu_set_t affinitySave;
 // if (CPU_COUNT(&comm->cpuAffinity)) {
    //sched_getaffinity(0, sizeof(cpu_set_t), &affinitySave);
    //sched_setaffinity(0, sizeof(cpu_set_t), &comm->cpuAffinity);
  //}
  ncclResult_t ret;

  // Launch proxy service thread
  //NCCLCHECK(ncclProxyCreate(comm));

  // Get rings and trees
  //struct ncclTopoGraph ringGraph;
  ringGraph.id = 0;
  ringGraph.pattern = NCCL_TOPO_PATTERN_RING;
  ringGraph.collNet = 0;
  ringGraph.minChannels = 1;
  ringGraph.maxChannels = MAXCHANNELS/2;
  NCCLCHECK(ncclTopoCompute(comm->topo, &ringGraph));
  NCCLCHECK(ncclTopoPrintGraph(comm->topo, &ringGraph));

  //struct ncclTopoGraph treeGraph;
  treeGraph.id = 1;
  treeGraph.pattern = NCCL_TOPO_PATTERN_BALANCED_TREE;
  treeGraph.collNet = 0;
  treeGraph.minChannels = comm->topo->nodes[NET].count != 0 ? 1 : ringGraph.nChannels;
  treeGraph.maxChannels = ringGraph.nChannels;
  NCCLCHECK(ncclTopoCompute(comm->topo, &treeGraph));
  NCCLCHECK(ncclTopoPrintGraph(comm->topo, &treeGraph));

  //struct ncclTopoGraph collNetGraph;
  collNetGraph.id = 2;
  collNetGraph.pattern = NCCL_TOPO_PATTERN_TREE;
  collNetGraph.collNet = 1;
  collNetGraph.minChannels = collNetGraph.maxChannels = ringGraph.nChannels;
  NCCLCHECK(ncclTopoCompute(comm->topo, &collNetGraph));
  NCCLCHECK(ncclTopoPrintGraph(comm->topo, &collNetGraph));

  bool allXgmi = true, hasPeerAccess = true;
  // Check that all the GPUs have peer access to one another and are XGMI connected
  for (int i = 0; i < nranks && hasPeerAccess; i++) {
    int cudaDev1 = comm->peerInfo[i].cudaDev;
    for (int j = 0; j < nranks; j++) {
      if (i == j) continue;
      int cudaDev2 = comm->peerInfo[j].cudaDev;
      int p2p;
      if (hipDeviceCanAccessPeer(&p2p, cudaDev1, cudaDev2) != hipSuccess || !p2p)
      {
        hasPeerAccess = false;
        break;
      }

      bool isXGMI;
      // Limit to single intermediate GPU for enabling clique
      NCCLCHECK(ncclTopoGetLinkType(comm->topo, i, j, &isXGMI, 1));
      allXgmi &= isXGMI;
    }
  }

  if (comm->rank == ncclParamGraphDumpFileRank()) {
    struct ncclTopoGraph* graphs[3] = { &ringGraph, &treeGraph, &collNetGraph };
    NCCLCHECK(ncclTopoDumpGraphs(comm->topo, 3, graphs));
  }

  // Determine local CollNet support before all-gather
  if (collNetSupport(comm)) {
    char *collNetEnable = getenv("NCCL_COLLNET_ENABLE");
    if (collNetEnable != NULL) {
      INFO(NCCL_ALL, "NCCL_COLLNET_ENABLE set by environment to %s.", collNetEnable);
      if (strcmp(collNetEnable, "1") == 0) {
        comm->collNetSupport = 1;
      }
    }
  }
  if (comm->collNetSupport == 1 && collNetGraph.nChannels <= 0) comm->collNetSupport = 0;

  if ((comm->topo->type & RCCL_TOPO_4P2H_ROME) && (comm->topo->type & RCCL_TOPO_GDR_ALL)) {
    if (rcclParamP2pNetDisable() == 0) {
      if (!(comm->topo->type & RCCL_TOPO_FORCE_INTRA)) comm->p2pNet = 1;
      INFO(NCCL_INIT, "RCCL enabled same node P2P over network");
    }
    else
      INFO(NCCL_INIT, "RCCL force disabled same node P2P over network");
  }
  // AllGather3 - begin
#if 0
  struct ncclGraphInfo {
    int pattern;
    int nChannels;
    int sameChannels;
    float speedIntra;
    float speedInter;
    int typeIntra;
    int typeInter;
  };

  struct {
    int netDev;
    int collNetSupport;
    int nc;
    struct ncclGraphInfo tree;
    struct ncclGraphInfo ring;
    struct ncclGraphInfo collNet;
    struct ncclTopoRanks topoRanks;
    bool pivotA2AEnabled;
  } *allGather3Data;

  NCCLCHECK(ncclCalloc(&allGather3Data, nranks));
#endif
  int idx;
  NCCLCHECK(ncclTopoIdToIndex(comm->topo, GPU, comm->busId, &idx));
  allGather3Data[rank].nc = 2;
  if (comm->topo->nodes[GPU].count == comm->topo->nRanks && comm->topo->nodes[GPU].nodes[idx].gpu.gcn == 906 && allXgmi)
    allGather3Data[rank].nc = 4;
  if (comm->topo->nodes[GPU].nodes[idx].gpu.gcn == 908)
    allGather3Data[rank].nc = std::max(4/ringGraph.nChannels, 2);
  if (comm->topo->nodes[GPU].count == comm->topo->nRanks && (comm->topo->type & RCCL_TOPO_CR8G))
    allGather3Data[rank].nc = 4;
  if (comm->topo->nodes[GPU].count == comm->topo->nRanks && comm->topo->nodes[GPU].nodes[idx].gpu.gcn == 910)
    allGather3Data[rank].nc = 4;
  if (comm->topo->nodes[GPU].nodes[idx].gpu.gcn == 910)
    allGather3Data[rank].nc = std::max(allGather3Data[rank].nc, 4/ringGraph.nChannels);
  if (ringGraph.nChannels > MAXCHANNELS/2)
    allGather3Data[rank].nc = 1;
  NCCLCHECK(ncclTopoGetLocalNet(comm->topo, rank, &allGather3Data[rank].netDev));
  allGather3Data[rank].tree.pattern = treeGraph.pattern;
  allGather3Data[rank].tree.nChannels = treeGraph.nChannels;
  allGather3Data[rank].tree.sameChannels = treeGraph.sameChannels;
  allGather3Data[rank].tree.speedIntra = treeGraph.speedIntra;
  allGather3Data[rank].tree.speedInter = treeGraph.speedInter;
  allGather3Data[rank].tree.typeIntra = treeGraph.typeIntra;
  allGather3Data[rank].tree.typeInter = treeGraph.typeInter;
  allGather3Data[rank].ring.pattern = ringGraph.pattern;
  allGather3Data[rank].ring.nChannels = ringGraph.nChannels;
  allGather3Data[rank].ring.sameChannels = ringGraph.sameChannels;
  allGather3Data[rank].ring.speedIntra = ringGraph.speedIntra;
  allGather3Data[rank].ring.speedInter = ringGraph.speedInter;
  allGather3Data[rank].ring.typeIntra = ringGraph.typeIntra;
  allGather3Data[rank].ring.typeInter = ringGraph.typeInter;
  allGather3Data[rank].collNet.pattern = collNetGraph.pattern;
  allGather3Data[rank].collNet.nChannels = collNetGraph.nChannels;
  allGather3Data[rank].collNet.sameChannels = collNetGraph.sameChannels;
  allGather3Data[rank].collNet.speedIntra = collNetGraph.speedIntra;
  allGather3Data[rank].collNet.speedInter = collNetGraph.speedInter;
  allGather3Data[rank].collNet.typeIntra = collNetGraph.typeIntra;
  allGather3Data[rank].collNet.typeInter = collNetGraph.typeInter;
  allGather3Data[rank].collNetSupport = comm->collNetSupport;
  allGather3Data[rank].pivotA2AEnabled = comm->topo->pivotA2AEnabled && rcclParamPivotAlltoallEnable();

  comm->nChannels = (comm->topo->nodes[GPU].count != comm->topo->nRanks && comm->topo->nodes[NET].count)
    ? std::min(treeGraph.nChannels, ringGraph.nChannels) : ringGraph.nChannels;
  NCCLCHECK(ncclTopoPreset(comm, &treeGraph, &ringGraph, &allGather3Data[rank].topoRanks));
  return ncclSuccess;
}

ncclResult_t initTransportsRank_3(struct ncclComm* comm, struct allGather3Data_t *allGather3Data,
  struct ncclTopoGraph& treeGraph, struct ncclTopoGraph& ringGraph, struct ncclTopoGraph& collNetGraph) {
  int rank = comm->rank;
  int nranks = comm->nRanks;
  ncclResult_t ret;
  //NCCLCHECK(bootstrapAllGather(comm->bootstrap, allGather3Data, sizeof(*allGather3Data)));

  // Determine nNodes, firstRanks, ...
  int *nodesFirstRank, *nodesTreePatterns;
  NCCLCHECK(ncclCalloc(&nodesFirstRank, nranks));
  NCCLCHECK(ncclCalloc(&nodesTreePatterns, nranks));
  NCCLCHECK(ncclCalloc(&comm->rankToNode, comm->nRanks));
  for (int r=0; r<nranks; r++) {
    int node;
    int firstRank = allGather3Data[r].topoRanks.ringRecv[0];
    for (node=0; node<comm->nNodes && nodesFirstRank[node] != firstRank; node++);
    if (node == comm->nNodes) {
      comm->nNodes++;
      nodesFirstRank[node] = firstRank;
      // Record tree pattern of each node as they can be different depending on sm arch
      nodesTreePatterns[node] = allGather3Data[r].tree.pattern;
    }
    comm->rankToNode[r] = node;
  }
  // Now that we know nNodes, alloc nodeRanks and compute localRanks for each node
  NCCLCHECK(ncclCalloc(&comm->nodeRanks, comm->nNodes));
  NCCLCHECK(ncclCalloc(&comm->rankToLocalRank, comm->nRanks));
  for (int r=0; r<comm->nRanks; r++) {
    int node = comm->rankToNode[r];
    comm->rankToLocalRank[r] = comm->nodeRanks[node].localRanks;
    comm->nodeRanks[node].localRanks++;
  }
  // Allocate ranks arrays for each node
  for (int n=0; n<comm->nNodes; n++) {
    NCCLCHECK(ncclCalloc(&comm->nodeRanks[n].localRankToRank, comm->nodeRanks[n].localRanks));
    comm->maxLocalRanks = std::max(comm->maxLocalRanks, comm->nodeRanks[n].localRanks);
    comm->nodeRanks[n].localRanks = 0;
  }
  // And fill the ranks arrays
  for (int r=0; r<comm->nRanks; r++) {
    int node = comm->rankToNode[r];
    comm->nodeRanks[node].localRankToRank[comm->nodeRanks[node].localRanks++] = r;
  }
  comm->node = comm->rankToNode[rank];
  comm->localRankToRank = comm->nodeRanks[comm->node].localRankToRank;
  comm->localRank = comm->rankToLocalRank[rank];
  comm->localRanks = comm->nodeRanks[comm->node].localRanks;

  TRACE(NCCL_INIT,"hostHash[%d] %lx localRank %d localRanks %d localRank0 %d",
        rank, comm->peerInfo[rank].hostHash, comm->localRank, comm->localRanks, comm->localRankToRank[0]);
  if (comm->localRank == -1 || comm->localRankToRank[0] == -1 || comm->localRanks == 0) {
    WARN("Failed to determine local ranks rank %d hostHash %lx pidHash %lx localRank %d localRanks %d localRank0 %d",
         rank, comm->peerInfo[rank].hostHash, comm->peerInfo[rank].pidHash,
         comm->localRank, comm->localRanks, comm->localRankToRank[0]);
    return ncclInternalError;
  }

  int nChannelsOrig = comm->nChannels;
  struct ncclTopoRanks** allTopoRanks;
  NCCLCHECK(ncclCalloc(&allTopoRanks, comm->nRanks));
  int nc = allGather3Data[0].nc;
  for (int i=0; i<nranks; i++) {
    comm->peerInfo[i].netDev = allGather3Data[i].netDev;
    allTopoRanks[i] = &allGather3Data[i].topoRanks;
    nc = std::min(allGather3Data[i].nc, nc);
    // Make sure we align all ranks so that the tuning is consistent across ranks
    treeGraph.nChannels = std::min(allGather3Data[i].tree.nChannels, treeGraph.nChannels);
    treeGraph.sameChannels = std::min(allGather3Data[i].tree.sameChannels, treeGraph.sameChannels);
    treeGraph.speedIntra = std::min(allGather3Data[i].tree.speedIntra, treeGraph.speedIntra);
    treeGraph.speedInter = std::min(allGather3Data[i].tree.speedInter, treeGraph.speedInter);
    treeGraph.typeIntra = std::max(allGather3Data[i].tree.typeIntra, treeGraph.typeIntra);
    treeGraph.typeInter = std::max(allGather3Data[i].tree.typeInter, treeGraph.typeInter);
    ringGraph.nChannels = std::min(allGather3Data[i].ring.nChannels, ringGraph.nChannels);
    ringGraph.sameChannels = std::min(allGather3Data[i].ring.sameChannels, ringGraph.sameChannels);
    ringGraph.speedIntra = std::min(allGather3Data[i].ring.speedIntra, ringGraph.speedIntra);
    ringGraph.speedInter = std::min(allGather3Data[i].ring.speedInter, ringGraph.speedInter);
    ringGraph.typeIntra = std::max(allGather3Data[i].ring.typeIntra, ringGraph.typeIntra);
    ringGraph.typeInter = std::max(allGather3Data[i].ring.typeInter, ringGraph.typeInter);
    collNetGraph.nChannels = std::min(allGather3Data[i].collNet.nChannels, collNetGraph.nChannels);
    collNetGraph.sameChannels = std::min(allGather3Data[i].collNet.sameChannels, collNetGraph.sameChannels);
    collNetGraph.speedIntra = std::min(allGather3Data[i].collNet.speedIntra, collNetGraph.speedIntra);
    collNetGraph.speedInter = std::min(allGather3Data[i].collNet.speedInter, collNetGraph.speedInter);
    collNetGraph.typeIntra = std::max(allGather3Data[i].collNet.typeIntra, collNetGraph.typeIntra);
    collNetGraph.typeInter = std::max(allGather3Data[i].collNet.typeInter, collNetGraph.typeInter);
    comm->collNetSupport = std::min(allGather3Data[i].collNetSupport, comm->collNetSupport);
    comm->topo->pivotA2AEnabled = comm->topo->pivotA2AEnabled && allGather3Data[i].pivotA2AEnabled;
  }

  comm->nChannels = treeGraph.nChannels = ringGraph.nChannels =
    (comm->topo->nodes[GPU].count != comm->topo->nRanks && comm->topo->nodes[NET].count)
    ? std::min(treeGraph.nChannels, ringGraph.nChannels) : ringGraph.nChannels;
  if (comm->nChannels < nChannelsOrig) {
    // We started duplicating channels during Preset(), so we need to move the
    // duplicated channels since we have removed some.
    for (int i=0; i<comm->nChannels; i++) memcpy(comm->channels+comm->nChannels+i, comm->channels+nChannelsOrig+i, sizeof(struct ncclChannel));
  }

  // Determine CollNet support after all-gather now that we know nNodes and each node localRanks
  if (comm->collNetSupport == 1) {
    int collNetNodeThreshold = ncclParamCollNetNodeThreshold();
    if (comm->nNodes < collNetNodeThreshold) {
      INFO(NCCL_INIT, "Communicator has %d nodes which is less than CollNet node threshold %d, disabling CollNet", comm->nNodes, collNetNodeThreshold);
      comm->collNetSupport = 0;
    }
    for (int n=0; n<comm->nNodes; n++) {
      if (comm->nodeRanks[n].localRanks > NCCL_MAX_DIRECT_ARITY+1) {
        WARN("CollNet currently only supports up to %d GPUs per node, disabling CollNet", NCCL_MAX_DIRECT_ARITY+1);
        comm->collNetSupport = 0;
        break;
      }
    }
  }

  int *rings;
  NCCLCHECK(ncclCalloc(&rings, nranks*MAXCHANNELS));
  NCCLCHECK(ncclTopoPostset(comm, nodesFirstRank, nodesTreePatterns, allTopoRanks, rings, &collNetGraph, nc));

  if (comm->topo->pivotA2ANumBiRings == 3) NCCLCHECK(ncclTreeBasePostset(comm, &treeGraph));

  free(allTopoRanks);
  free(nodesTreePatterns);
  free(nodesFirstRank);
  //free(allGather3Data);

  // AllGather3 - end

  TRACE(NCCL_INIT, "rank %d nranks %d - BUILT %d TREES/RINGS", rank, nranks, comm->nChannels);

  char line[1024];
  line[0]='\0';
  for (int c=0; c<comm->nChannels; c++) {
    struct ncclTree* tree = &comm->channels[c].tree;
    snprintf(line+strlen(line), 1023-strlen(line), " [%d] %d/%d/%d->%d->%d",
        c, tree->down[0], tree->down[1], tree->down[2], rank, tree->up);
    INFO(NCCL_GRAPH, "Ring %d : %d -> %d -> %d", c, comm->channels[c].ring.prev, comm->rank, comm->channels[c].ring.next);
  }
  line[1023] = '\0';
  INFO(NCCL_INIT, "Trees%s", line);

  //NCCLCHECK(computeBuffSizes(comm));

  // Connect with prev/next for each ring
  for (int c=0; c<comm->nChannels; c++) {
    struct ncclChannel* channel = comm->channels+c;
    NCCLCHECKGOTO(setupChannel(comm, c, rank, nranks, rings+c*nranks), ret, affinity_restore);
    if (comm->nRanks == 1) continue;
    NCCLCHECKGOTO(ncclTransportP2pConnect(comm, c, 1, &channel->ring.prev, 1, &channel->ring.next, 0), ret, affinity_restore);
  }
  NCCLCHECKGOTO(ncclTransportP2pSetup(comm, &ringGraph, 0), ret, affinity_restore);
  if (ringGraph.nIntraChannels && rcclParamP2pNetDisable() == 0) {
    comm->useIntraNet = 1;
    // Connect NET for intranode use
    for (int c=0; c<comm->nChannels; c++) {
      struct ncclChannel* channel = comm->channels+c;
      if (comm->nRanks == 1) continue;
      NCCLCHECKGOTO(ncclTransportP2pConnect(comm, c, 1, &channel->ring.prev, 1, &channel->ring.next, NCCL_CONN_IDX_P2P_NET), ret, affinity_restore);
    }
    NCCLCHECKGOTO(ncclTransportP2pSetup(comm, &ringGraph, NCCL_CONN_IDX_P2P_NET), ret, affinity_restore);
  }
  free(rings);
  INFO(NCCL_INIT, "Connected all rings");

  // Connect Trees
  for (int c=0; c<comm->nChannels; c++) {
    struct ncclChannel* channel = comm->channels+c;
    if (comm->nRanks == 1) continue;
    NCCLCHECKGOTO(ncclTransportP2pConnect(comm, c, NCCL_MAX_TREE_ARITY, channel->tree.down, 1, &channel->tree.up, 0), ret, affinity_restore);
    NCCLCHECKGOTO(ncclTransportP2pConnect(comm, c, 1, &channel->tree.up, NCCL_MAX_TREE_ARITY, channel->tree.down, 0), ret, affinity_restore);
  }
  NCCLCHECKGOTO(ncclTransportP2pSetup(comm, &treeGraph, 0), ret, affinity_restore);
  INFO(NCCL_INIT, "Connected all trees");

  // Check if we can setup CollNet
  if (comm->collNetSupport > 0) {
    int collNetSetupFail = 0;
    int highestTypes[NCCL_MAX_LOCAL_RANKS] = {TRANSPORT_P2P};
    // Find all head ranks
    int nHeads = collNetGraph.nChannels;
    int *heads;
    NCCLCHECK(ncclCalloc(&heads, nHeads));
    // Head GPU index is always 0
    for (int c=0; c<nHeads; c++) {
      heads[c] = collNetGraph.intra[c*comm->localRanks+0];
    }
    for (int c=0; c<comm->nChannels; c++) {
      struct ncclChannel* channel = comm->channels+c;
      for (int h=0; h<nHeads; h++) {
        const int head = heads[h];
        collNetSetupFail = ncclTransportCollNetSetup(comm, &collNetGraph, channel, head, head, h, collNetRecv);
        collNetSetupFail += ncclTransportCollNetSetup(comm, &collNetGraph, channel, head, head, h, collNetSend);
      }
      // Verify CollNet setup across ranks after trying the first channel
      if (c == 0) {
        NCCLCHECKGOTO(ncclTransportCollNetCheck(comm, collNetSetupFail), ret, collnet_cleanup);
      }
    }
    // Verify CollNet setup across ranks after trying all channels
    NCCLCHECKGOTO(ncclTransportCollNetCheck(comm, collNetSetupFail), ret, collnet_cleanup);
    TRACE(NCCL_INIT, "rank %d Connected inter-node CollNet", rank);

    // Connect intra-node CollNet
    int highestTransportType0, highestTransportType1;
    for (int c=0; c<comm->nChannels; c++) {
      struct ncclChannel* channelRecv = comm->channels+c;
      NCCLCHECKGOTO(ncclTransportP2pConnect(comm, c, NCCL_MAX_DIRECT_ARITY, channelRecv->collTree.up, NCCL_MAX_DIRECT_ARITY, channelRecv->collTree.down, 0), ret, collnet_cleanup);
    }
    NCCLCHECKGOTO(ncclTransportP2pSetup(comm, &collNetGraph, 0, &highestTransportType0), ret, collnet_cleanup);
    for (int c=0; c<comm->nChannels; c++) {
      struct ncclChannel* channelSend = comm->channels+c;
      NCCLCHECKGOTO(ncclTransportP2pConnect(comm, c, NCCL_MAX_DIRECT_ARITY, channelSend->collTree.down, NCCL_MAX_DIRECT_ARITY, channelSend->collTree.up, 1), ret, collnet_cleanup);
    }
    NCCLCHECKGOTO(ncclTransportP2pSetup(comm, &collNetGraph, 1, &highestTransportType1), ret, collnet_cleanup);

    // Exchange highest intra-node transport type among ranks
    // because we need to know whether all ranks can p2p each other to determine whether we can directly read/write registered user buffer
    comm->intraHighestTransportType = highestTypes[comm->localRank] = highestTransportType0 > highestTransportType1 ? highestTransportType0 : highestTransportType1;
    //NCCLCHECK(bootstrapIntraNodeAllGather(comm->bootstrap, comm->localRankToRank, comm->localRank, comm->localRanks, highestTypes, sizeof(int)));
    for (int i=0; i<comm->localRanks; i++) {
      if (highestTypes[i] > comm->intraHighestTransportType)
        comm->intraHighestTransportType = highestTypes[i];
    }
    INFO(NCCL_INIT, "rank %d Connected CollNet comm %p nRanks %02d", rank, comm, comm->nRanks);

collnet_cleanup:
    free(heads);
    if (ret != ncclSuccess) {
      NCCLCHECK(ncclTransportCollNetFree(comm));
      comm->collNetSupport = 0;
      ret = ncclSuccess;
    }
  }
  TRACE(NCCL_INIT, "rank %d nranks %d - CONNECTED %d RINGS AND TREES", rank, nranks, comm->nChannels);

  // Compute time models for algorithm and protocol combinations
  do {
    int myCompCap = comm->peerInfo[rank].cudaCompCap;
    int minCompCap = myCompCap, maxCompCap = myCompCap;
    for (int i = 0; i < nranks; i++) {
      minCompCap = std::min(comm->peerInfo[i].cudaCompCap, minCompCap);
      maxCompCap = std::max(comm->peerInfo[i].cudaCompCap, maxCompCap);
    }
    NCCLCHECK(ncclTopoTuneModel(comm, minCompCap, maxCompCap, &treeGraph, &ringGraph, &collNetGraph));
  } while(0);

  // Compute nChannels per peer for p2p
  NCCLCHECK(ncclTopoComputeP2pChannels(comm));
#if 0
  do { // Setup p2p structures in comm->tasks
    struct ncclTasks* tasks = &comm->tasks;
    int nRanks = comm->nRanks;
    int node = comm->node;
    int nNodes = comm->nNodes;
    struct ncclNodeRanks *nodeRanks = comm->nodeRanks;
    int localRank = comm->localRank;
    tasks->peers = ncclMemoryStackAlloc<ncclTasks::Peer>(&comm->memPermanent, nRanks);
    tasks->p2pSendOrder = ncclMemoryStackAlloc<int>(&comm->memPermanent, nRanks);
    tasks->p2pRecvOrder = ncclMemoryStackAlloc<int>(&comm->memPermanent, nRanks);
    int s=0, r=0;
    // schedule delta 0, +1, -1, +2, -2, ...
    // also make sure we don't do 0 twice, nor +n/2 and -n/2 if n is even.
    for (int d=0; d <= nNodes/4; d++) {
      int deltas[4] = { d, (nNodes-d)%nNodes, nNodes/2-d, (nNodes-(nNodes/2-d))%nNodes };
      int index = 0;
      int delta = deltas[index];
    sched_delta:
      int recvNode = (node+nNodes-delta)%nNodes;
      int sendNode = (node+delta)%nNodes;
      int steps = comm->maxLocalRanks;
      for (int step=0; step < steps; step++) {
        int recvIndex = (localRank-step+steps)%steps;
        if (recvIndex < nodeRanks[recvNode].localRanks) {
          tasks->p2pRecvOrder[r] = nodeRanks[recvNode].localRankToRank[recvIndex];
          r++;
        }
        int sendIndex = (localRank+step)%steps;
        if (sendIndex < nodeRanks[sendNode].localRanks) {
          tasks->p2pSendOrder[s] = nodeRanks[sendNode].localRankToRank[sendIndex];
          s++;
        }
      }
      index++;
      if (index == 1 && deltas[1] == deltas[0]) index++;
      if (index == 2 && deltas[2] == deltas[0]) index++;
      if (index == 3 && deltas[3] == deltas[2]) index++;
      if (index == 3 && deltas[3] == deltas[1]) index++;
      if (index < 4) {
        delta = deltas[index];
        goto sched_delta;
      }
    }
    assert(s == nRanks && r == nRanks);
  } while (0);

  if (ncclParamNvbPreconnect()) {
    // Connect p2p when using NVB path
    int nvbNpeers;
    int* nvbPeers;
    NCCLCHECK(ncclTopoGetNvbGpus(comm->topo, comm->rank, &nvbNpeers, &nvbPeers));
    for (int r=0; r<nvbNpeers; r++) {
      int peer = nvbPeers[r];
      int channelId;
      for (int c=0; c<comm->p2pnChannelsPerPeer; c++) {
        NCCLCHECK(ncclChannelCompute(comm, peer, c, ncclFuncSend, &channelId));
        if (comm->channels[channelId].peers[peer].send[1].connected == 0) {
          comm->connectSend[peer] |= (1<<channelId);
        }
      }
      for (int c=0; c<comm->p2pnChannelsPerPeer; c++) {
        NCCLCHECK(ncclChannelCompute(comm, peer, c, ncclFuncRecv, &channelId));
        if (comm->channels[channelId].peers[peer].recv[1].connected == 0) {
          comm->connectRecv[peer] |= (1<<channelId);
        }
      }
    }
    NCCLCHECK(ncclTransportP2pSetup(comm, NULL, 1));
    free(nvbPeers);
  }
#endif
  // Connect to local net proxy
  struct ncclProxyConnector proxyConn;
  //NCCLCHECK(ncclProxyConnect(comm, TRANSPORT_NET, 1, comm->rank, &proxyConn));
  //NCCLCHECK(ncclProxyCall(&proxyConn, ncclProxyMsgSharedInit, &comm->p2pnChannels, sizeof(int), NULL, 0));

  // Then to remote ones when using PXN
  if (ncclPxnDisable(comm) == 0) {
    int nranks;
    int* pxnPeers;
    NCCLCHECK(ncclTopoGetPxnRanks(comm, &pxnPeers, &nranks));
    for (int r=0; r<nranks; r++) {
      //NCCLCHECK(ncclProxyConnect(comm, TRANSPORT_NET, 1, pxnPeers[r], &proxyConn));
     // NCCLCHECK(ncclProxyCall(&proxyConn, ncclProxyMsgSharedInit, &comm->p2pnChannels, sizeof(int), NULL, 0));
    }
    free(pxnPeers);
  }

  do {
    // Compute intra-process ranks
    int intraProcRank0 = -1, intraProcRank = -1, intraProcRanks = 0;
    for (int i = 0; i < nranks; i++) {
      if ((comm->peerInfo[i].hostHash == comm->peerInfo[rank].hostHash)
          && (comm->peerInfo[i].pidHash == comm->peerInfo[rank].pidHash)) {
        // Rank is in same process
        if (intraProcRanks == 0) intraProcRank0 = i;
        if (i == rank) intraProcRank = intraProcRanks;
        intraProcRanks++;
        if (intraProcRank0 == rank && rank != i) {
          comm->peerInfo[i].comm->intraNext = comm->intraNext;
          comm->intraNext = comm->peerInfo[i].comm;
        }
      }
    }
    TRACE(NCCL_INIT,"pidHash[%d] %lx intraProcRank %d intraProcRanks %d intraProcRank0 %d",
        rank, comm->peerInfo[rank].pidHash, intraProcRank, intraProcRanks, intraProcRank0);
    if (intraProcRank == -1 || intraProcRank0 == -1 || comm->peerInfo[intraProcRank0].comm == NULL) {
      WARN("Failed to determine intra proc ranks rank %d hostHash %lx pidHash %lx intraProcRank %d intraProcRanks %d intraProcRank0 %d",
          rank, comm->peerInfo[rank].hostHash, comm->peerInfo[rank].pidHash,
          intraProcRank, intraProcRanks, intraProcRank0);
      return ncclInternalError;
    }
    struct ncclComm* comm0 = comm->peerInfo[intraProcRank0].comm;
    assert(intraProcRank==0 ? comm==comm0 : true);
    comm->intraComm0 = comm0;
    comm->intraRefs = intraProcRank==0 ? intraProcRanks : 0;
    comm->intraRank = intraProcRank;
    comm->intraRanks = intraProcRanks;
    comm->intraBarrierPhase = 0;
    comm->intraBarrierCounter = 0;
    comm->intraBarrierGate = 0;
  } while(0);

#if 0
  if (comm->intraRank == 0) { // Load ncclParamLaunchMode
    char* str = getenv("NCCL_LAUNCH_MODE");
    enum ncclLaunchMode mode, modeOld;
    if (str && strcasecmp(str, "GROUP") == 0) {
      mode = ncclLaunchModeGroup;
    } else {
      mode = ncclLaunchModeParallel;
    }
    // In theory we could be racing with other communicators not associated with
    // this one if the user is connecting to multiple ncclUniqueId's concurrently.
    modeOld = __atomic_exchange_n(&ncclParamLaunchMode, mode, __ATOMIC_RELAXED);
    if (modeOld == ncclLaunchModeInvalid && str && str[0]!='\0') {
      INFO(NCCL_ENV, "NCCL_LAUNCH_MODE set by environment to %s", mode == ncclLaunchModeParallel ? "PARALLEL" : "GROUP");
    }
  }

  /* Local intra-node barrier */
  //NCCLCHECK(bootstrapBarrier(comm->bootstrap, comm->localRankToRank, comm->localRank, comm->localRanks, comm->localRankToRank[0]));

  // Unlink proxy shm to make sure it will be properly cleaned up.
  NCCLCHECK(ncclProxyShmUnlink(comm));
#endif

  // We should have allocated all buffers, collective fifos, ... we can
  // restore the affinity.
affinity_restore:
  //if (CPU_COUNT(&comm->cpuAffinity)) sched_setaffinity(0, sizeof(cpu_set_t), &affinitySave);
  if (ret != ncclSuccess) return ret;

  TRACE(NCCL_INIT, "rank %d nranks %d - DONE", rank, nranks);
  return ncclSuccess;
}

ncclResult_t rocm_smi_init() {
  return ncclSuccess;
}

ncclResult_t rocm_smi_getDeviceIndexByPciBusId(const char* pciBusId, uint32_t* deviceIndex) {
  return ncclSuccess;
}

ncclResult_t rocm_smi_getLinkInfo(int srcDev, int dstDev, RSMI_IO_LINK_TYPE* rsmi_type, int *hops, int *bw) {
  return ncclSuccess;
}

int ncclNetVersion(struct ncclComm* comm) {
  return 4;
}
