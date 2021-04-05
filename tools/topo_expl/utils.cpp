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
#include "xml.h"
#include "coll_net.h"
#include "model.h"
#include "utils.h"

const char* ncclFuncStr[NCCL_NUM_FUNCTIONS+1] = { "Broadcast", "Reduce", "AllGather", "ReduceScatter", "AllReduce", "SendRecv" };
const char* ncclAlgoStr[NCCL_NUM_ALGORITHMS] = { "Tree", "Ring", "CollNet" };
const char* ncclProtoStr[NCCL_NUM_PROTOCOLS] = { "LL", "LL128", "Simple" };

extern NodeModel *node_model;

NCCL_PARAM(CrossNic, "CROSS_NIC", 2);
NCCL_PARAM(CollNetEnable, "COLLNET_ENABLE", 0);
NCCL_PARAM(GraphDumpFileRank, "GRAPH_DUMP_FILE_RANK", 0);

thread_local int ncclDebugNoWarn = 0;
ncclCollNet_t* ncclCollNet = 0;

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
  size_t len = 0;
  if (node_model) len = snprintf(buffer, sizeof(buffer),
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

ncclResult_t ncclTopoGetSystem(const char* xmlTopoFile, struct ncclTopoSystem** system) {
  struct ncclXml* xml;
  NCCLCHECK(ncclCalloc(&xml, 1));
  NCCLCHECK(ncclTopoGetXmlFromFile(xmlTopoFile, xml));
  NCCLCHECK(ncclTopoGetSystemFromXml(xml, system));
  free(xml);
  return ncclSuccess;
}


ncclResult_t bootstrapAllGather(struct ncclComm* comm, struct allGather1Data_t * allGather1Data) {
  // AllGather1 - begin
  allGather1Data[comm->rank].comm = comm;
  allGather1Data[comm->rank].cudaCompCap = 1;
  allGather1Data[comm->rank].peerInfo.rank = comm->rank;
  allGather1Data[comm->rank].peerInfo.cudaDev = node_model->rankToCudaDev(comm->rank);
  allGather1Data[comm->rank].peerInfo.gdrSupport = 1;
  allGather1Data[comm->rank].peerInfo.hostHash = node_model->hostHash;
  allGather1Data[comm->rank].peerInfo.pidHash = node_model->pidHash;
  allGather1Data[comm->rank].peerInfo.shmDev = 0x19;
  allGather1Data[comm->rank].peerInfo.busId = node_model->getGpuBusId(comm->rank);
  return ncclSuccess;
}

extern struct ncclTransport collNetTransport;

// All ranks must participate in collNetSetup call
// type: 0 for send, 1 for recv
// return: 0 - unsupported, 1 - supported
// We do not NCCLCHECK this call because we would fall back to P2P network in case CollNet setup fails
static int collNetSetup(struct ncclComm* comm, struct ncclTopoGraph* collNetGraph, struct ncclChannel* channel, int rank, int nranks,  int masterRank, int masterPeer, int nMasters, int type) {
  int rankInCollNet = -1;
  int supported = 0;
  int isMaster = (rank == masterRank) ? 1 : 0;
  struct {
    int collNetRank;
    ncclConnect connect;
  } sendrecvExchange;

  // check if we can connect to collnet, whose root is the nranks-th rank
  struct ncclPeerInfo *myInfo = comm->peerInfo+rank, *peerInfo = comm->peerInfo+nranks;
  peerInfo->rank = nranks;
  int ret = 1;
  if (isMaster) {
    NCCLCHECK(collNetTransport.canConnect(&ret, comm->topo, collNetGraph, myInfo, peerInfo));
  }

  // send master receives connect info from peer recv master
  if (isMaster && type == 0) {
    //NCCLCHECK(bootstrapRecv(comm->bootstrap, masterPeer, &sendrecvExchange, sizeof(sendrecvExchange)));
    rankInCollNet = sendrecvExchange.collNetRank;
    INFO(NCCL_INIT, "CollNet [send] : rank %d collNetRank %d collNetNranks %d received connect from rank %d", rank, rankInCollNet, nMasters, masterPeer);
  }

  // select
  struct ncclPeer* root = channel->peers+nranks;
  struct ncclConnector* conn = (type == 1) ? &root->recv : &root->send;
  struct ncclTransportComm* transportComm = (type == 1) ? &(collNetTransport.recv) : &(collNetTransport.send);
  conn->transportComm = transportComm;
  // setup
  struct ncclConnect myConnect;
  if (isMaster && ret > 0) {
    NCCLCHECK(transportComm->setup(comm, collNetGraph, myInfo, peerInfo, &myConnect, conn, channel->id));
  }
  // prepare connect handles
  ncclResult_t res;
  struct {
    int isMaster;
    ncclConnect connect;
  } *allConnects = NULL;
  ncclConnect *masterConnects = NULL;
  //NCCLCHECK(ncclCalloc(&masterConnects, nMasters));
  if (type == 1) {  // recv side: AllGather
    // all ranks must participate
    //NCCLCHECK(ncclCalloc(&allConnects, nranks));
    //allConnects[rank].isMaster = isMaster;
    //memcpy(&(allConnects[rank].connect), &myConnect, sizeof(struct ncclConnect));
    //NCCLCHECKGOTO(bootstrapAllGather(comm->bootstrap, allConnects, sizeof(*allConnects)), res, cleanup);
    // consolidate
    //int c = 0;
    //for (int r = 0; r < nranks; r++) {
      //if (allConnects[r].isMaster) {
        //memcpy(masterConnects+c, &(allConnects[r].connect), sizeof(struct ncclConnect));
        //if (r == rank) rankInCollNet = c;
        //c++;
      //}
    //}
  } else { // send side : copy in connect info received from peer recv master
    //if (isMaster) memcpy(masterConnects+rankInCollNet, &(sendrecvExchange.connect), sizeof(struct ncclConnect));
  }
  // connect
  if (isMaster && ret > 0) {
    //NCCLCHECKGOTO(transportComm->connect(comm, masterConnects, nMasters, rankInCollNet, conn), res, cleanup);
    //struct ncclPeer* devRoot = channel->devPeers+nranks;
    //struct ncclConnector* devConn = (type == 1) ? &devRoot->recv : &devRoot->send;
    //CUDACHECKGOTO(hipMemcpy(devConn, conn, sizeof(struct ncclConnector), hipMemcpyHostToDevice), res, cleanup);
  }
  // recv side sends connect info to send side
  if (isMaster && type == 1) {
    //sendrecvExchange.collNetRank = rankInCollNet;
    //memcpy(&sendrecvExchange.connect, masterConnects+rankInCollNet, sizeof(struct ncclConnect));
    //NCCLCHECKGOTO(bootstrapSend(comm->bootstrap, masterPeer, &sendrecvExchange, sizeof(sendrecvExchange)), res, cleanup);
    INFO(NCCL_INIT, "CollNet [recv] : rank %d collNetRank %d collNetNranks %d sent connect to rank %d", rank, rankInCollNet, nMasters, masterPeer);
  }
  if (ret > 0) {
    supported = 1;
  }
cleanup:
  //if (allConnects != NULL) free(allConnects);
  //if (masterConnects != NULL) free(masterConnects);
  return supported;
}

static ncclResult_t checkCollNetSetup(struct ncclComm* comm, int rank, int collNetSetupFail) {
  comm->collNetSupport = 1;
  return ncclSuccess;
}

void initCollNet() {
  if (ncclParamCollNetEnable() == 1 && ncclCollNet == 0)
    ncclCollNet = (ncclCollNet_t*)0x12345678;
}

ncclResult_t initTransportsRank_1(struct ncclComm* comm, struct allGather1Data_t *allGather1Data, struct allGather3Data_t *allGather3Data,
  struct ncclTopoGraph& treeGraph, struct ncclTopoGraph& ringGraph, struct ncclTopoGraph& collNetGraph) {
  int rank = comm->rank;
  int nranks = comm->nRanks;

  //uint64_t commHash = getHash(commId->internal, NCCL_UNIQUE_ID_BYTES);
  //TRACE(NCCL_INIT, "comm %p, commHash %lx, rank %d nranks %d - BEGIN", comm, commHash, rank, nranks);
  //NCCLCHECK(bootstrapInit(commId, rank, nranks, &comm->bootstrap));

  // AllGather1 - begin
  //struct {
  //  struct ncclPeerInfo peerInfo;
  //  struct ncclComm* comm;
  //  int cudaCompCap;
  //} *allGather1Data;

  //NCCLCHECK(ncclCalloc(&allGather1Data, nranks));
  //allGather1Data[rank].comm = comm;
  //allGather1Data[rank].cudaCompCap = ncclCudaCompCap();
  struct ncclPeerInfo* myInfo = &allGather1Data[rank].peerInfo;
  //NCCLCHECK(fillInfo(comm, myInfo, commHash));
  //NCCLCHECK(bootstrapAllGather(comm->bootstrap, allGather1Data, sizeof(*allGather1Data)));

  NCCLCHECK(ncclCalloc(&comm->peerInfo, nranks+1)); // Extra rank to represent CollNet root
  for (int i = 0; i < nranks; i++) {
    memcpy(comm->peerInfo+i, &allGather1Data[i].peerInfo, sizeof(struct ncclPeerInfo));
    if ((i != rank) && (comm->peerInfo[i].hostHash == myInfo->hostHash) && (comm->peerInfo[i].busId == myInfo->busId)) {
      WARN("Duplicate GPU detected : rank %d and rank %d both on CUDA device %lx", rank, i, myInfo->busId);
      return ncclInvalidUsage;
    }
  }

  // Compute intra ranks and minimum CUDA Compute capabilities of intra-node GPUs and all GPUs
  int intraRank0 = -1, intraRank = -1, intraRanks = 0;
  int myCompCap = allGather1Data[rank].cudaCompCap;
  int minCompCap = myCompCap, maxCompCap = myCompCap;
  uint64_t otherHostHash;
  int tmpNnodes = 1;
  for (int i = 0; i < nranks; i++) {
    if (allGather1Data[i].peerInfo.hostHash == allGather1Data[rank].peerInfo.hostHash) {
      if (allGather1Data[i].peerInfo.pidHash == allGather1Data[rank].peerInfo.pidHash) {
        if (intraRanks == 0) intraRank0 = i;
        if (i == rank) intraRank = intraRanks;
        intraRanks++;
      }
    } else {  // Determine whether number of nodes is 2 (for use in tree pattern determination)
      if (tmpNnodes == 1) {
        otherHostHash = allGather1Data[i].peerInfo.hostHash;
        tmpNnodes = 2;
      } else if (tmpNnodes == 2 && otherHostHash != allGather1Data[i].peerInfo.hostHash) {
        tmpNnodes = 3;
      }
    }
    minCompCap = std::min(allGather1Data[i].cudaCompCap, minCompCap);
    maxCompCap = std::max(allGather1Data[i].cudaCompCap, maxCompCap);
  }
  TRACE(NCCL_INIT,"hostHash[%d] %lx intraRank %d intraRanks %d intraRank0 %d",
        rank, allGather1Data[rank].peerInfo.hostHash, intraRank, intraRanks, intraRank0);
  if (intraRank == -1 || intraRank0 == -1 || allGather1Data[intraRank0].comm == NULL) {
    WARN("Failed to determine intra ranks hostHash[%d] %lx intraRank %d intraRanks %d intraRank0 %d",
         rank, allGather1Data[rank].peerInfo.hostHash, intraRank, intraRanks, intraRank0);
    return ncclInternalError;
  }
  struct ncclComm* intraRank0Comm = allGather1Data[intraRank0].comm;

  //free(allGather1Data);

  // AllGather1 - end

  // Topo detection / System graph creation
  //NCCLCHECK(ncclTopoGetSystem(comm, &comm->topo));
  comm->topo->nRanks = comm->nRanks;
  // Compute paths between GPUs and NICs
  NCCLCHECK(ncclTopoComputePaths(comm->topo, comm->peerInfo));
  // Remove inaccessible GPUs and unused NICs
  NCCLCHECK(ncclTopoTrimSystem(comm->topo, comm));
  // Recompute paths after trimming
  NCCLCHECK(ncclTopoComputePaths(comm->topo, comm->peerInfo));
  // Init search
  NCCLCHECK(ncclTopoSearchInit(comm->topo));
  // Print final topology
  NCCLCHECK(ncclTopoPrint(comm->topo));

  // Get rings and trees
  //struct ncclTopoGraph ringGraph;
  ringGraph.id = 0;
  ringGraph.pattern = NCCL_TOPO_PATTERN_RING;
  ringGraph.crossNic = ncclParamCrossNic();
  ringGraph.collNet = 0;
  ringGraph.minChannels = 1;
  ringGraph.maxChannels = MAXCHANNELS/2;
  NCCLCHECK(ncclTopoCompute(comm->topo, &ringGraph));
  NCCLCHECK(ncclTopoPrintGraph(comm->topo, &ringGraph));

  //struct ncclTopoGraph treeGraph;
  treeGraph.id = 1;
  treeGraph.pattern = tmpNnodes <= 2 ? NCCL_TOPO_PATTERN_TREE : NCCL_TOPO_PATTERN_BALANCED_TREE;
  treeGraph.crossNic = ncclParamCrossNic();
  treeGraph.collNet = 0;
  treeGraph.minChannels = comm->topo->nodes[NET].count != 0 ? 1 : ringGraph.nChannels;
  treeGraph.maxChannels = ringGraph.nChannels;
  NCCLCHECK(ncclTopoCompute(comm->topo, &treeGraph));
  NCCLCHECK(ncclTopoPrintGraph(comm->topo, &treeGraph));

  //struct ncclTopoGraph collNetGraph;
  collNetGraph.id = 2;
  collNetGraph.pattern = NCCL_TOPO_PATTERN_TREE;
  collNetGraph.collNet = 1;
  collNetGraph.crossNic = ncclParamCrossNic();
  collNetGraph.minChannels = collNetGraph.maxChannels = ringGraph.nChannels;
  NCCLCHECK(ncclTopoCompute(comm->topo, &collNetGraph));
  NCCLCHECK(ncclTopoPrintGraph(comm->topo, &collNetGraph));

  if (comm->rank == ncclParamGraphDumpFileRank()) {
    struct ncclTopoGraph* graphs[3] = { &ringGraph, &treeGraph, &collNetGraph };
    NCCLCHECK(ncclTopoDumpGraphs(comm->topo, 3, graphs));
  }

  // AllGather3 - begin
#if 0
  struct ncclGraphInfo {
    int pattern;
    int sameChannels;
    float speedIntra;
    float speedInter;
    int typeIntra;
    int typeInter;
  };

  struct {
    int cudaCompCap;
    int fullCudaCompCap;
    int nChannels;
    int gcn;
    struct ncclGraphInfo tree;
    struct ncclGraphInfo ring;
    struct ncclGraphInfo collNet;
    struct ncclTopoRanks topoRanks;
  } *allGather3Data;

  NCCLCHECK(ncclCalloc(&allGather3Data, nranks));
#endif
  int idx;
  NCCLCHECK(ncclTopoIdToIndex(comm->topo, GPU, myInfo->busId, &idx));
  allGather3Data[rank].cudaCompCap = comm->topo->nodes[GPU].nodes[idx].gpu.cudaCompCap;
  allGather3Data[rank].gcn = comm->topo->nodes[GPU].nodes[idx].gpu.gcn;

  allGather3Data[rank].nChannels = comm->nChannels = treeGraph.nChannels = ringGraph.nChannels =
    std::min(treeGraph.nChannels, ringGraph.nChannels);
  allGather3Data[rank].tree.pattern = treeGraph.pattern;
  allGather3Data[rank].tree.sameChannels = treeGraph.sameChannels;
  allGather3Data[rank].tree.speedIntra = treeGraph.speedIntra;
  allGather3Data[rank].tree.speedInter = treeGraph.speedInter;
  allGather3Data[rank].tree.typeIntra = treeGraph.typeIntra;
  allGather3Data[rank].tree.typeInter = treeGraph.typeInter;
  allGather3Data[rank].ring.pattern = ringGraph.pattern;
  allGather3Data[rank].ring.sameChannels = ringGraph.sameChannels;
  allGather3Data[rank].ring.speedIntra = ringGraph.speedIntra;
  allGather3Data[rank].ring.speedInter = ringGraph.speedInter;
  allGather3Data[rank].ring.typeIntra = ringGraph.typeIntra;
  allGather3Data[rank].ring.typeInter = ringGraph.typeInter;
  allGather3Data[rank].collNet.pattern = collNetGraph.pattern;
  allGather3Data[rank].collNet.sameChannels = collNetGraph.sameChannels;
  allGather3Data[rank].collNet.speedIntra = collNetGraph.speedIntra;
  allGather3Data[rank].collNet.speedInter = collNetGraph.speedInter;
  allGather3Data[rank].collNet.typeIntra = collNetGraph.typeIntra;
  allGather3Data[rank].collNet.typeInter = collNetGraph.typeInter;

  // CollNet channels are already duplicated
  comm->collNetnChannels = 2*collNetGraph.nChannels;
  NCCLCHECK(ncclTopoPreset(comm, &treeGraph, &ringGraph, &collNetGraph, &allGather3Data[rank].topoRanks));

  return ncclSuccess;
}

ncclResult_t initChannel(struct ncclComm* comm, int channelid) {
  struct ncclChannel* channel = comm->channels+channelid;
  channel->id = channelid;

  // Setup intermediate buffering
  //int buffSize = ncclParamBuffsize();
  int cpuArch, cpuVendor, cpuModel;
  NCCLCHECK(ncclTopoCpuType(comm->topo, &cpuArch, &cpuVendor, &cpuModel));
  //channel->buffSize = buffSize != -2 ? buffSize :
  //  cpuArch == NCCL_TOPO_CPU_ARCH_ARM ? DEFAULT_BUFFER_SIZE_BYTES_ARM : DEFAULT_BUFFER_SIZE_BYTES;

  // Ring index to user rank table.
  //NCCLCHECK(ncclCudaCalloc(&channel->ring.devUserRanks, comm->nRanks));
  NCCLCHECK(ncclCalloc(&channel->ring.userRanks, comm->nRanks));

  // Communication structures with peers.
  //NCCLCHECK(ncclCudaCalloc(&channel->devPeers, comm->nRanks+1)); // The extra one rank is for collnet root (i.e. network)
  NCCLCHECK(ncclCalloc(&channel->peers, comm->nRanks+1));
  for (size_t i=0; i<comm->nRanks+1; ++i) {
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

template <int type>
static ncclResult_t selectTransport(struct ncclComm* comm, struct ncclTopoGraph* graph, struct ncclPeerInfo* myInfo, struct ncclPeerInfo* peerInfo, struct ncclConnect* connect, struct ncclConnector* connector, int channelId) {
  for (int t=0; t<NTRANSPORTS; t++) {
    struct ncclTransport *transport = ncclTransports+t;
    struct ncclTransportComm* transportComm = type == 1 ? &transport->send : &transport->recv;
    int ret = 0;
    NCCLCHECK(transport->canConnect(&ret, comm->topo, graph, myInfo, peerInfo));
    if (ret) {
      connector->transportComm = transportComm;
      NCCLCHECK(transportComm->setup(comm, graph, myInfo, peerInfo, connect, connector, channelId));
      return ncclSuccess;
    }
  }
  WARN("No transport found !");
  return ncclInternalError;
}

ncclResult_t ncclTransportP2pConnect(struct ncclComm* comm, struct ncclChannel* channel, int nrecv, int* peerRecv, int nsend, int* peerSend) {
  TRACE(NCCL_INIT, "nsend %d nrecv %d", nsend, nrecv);
  uint32_t mask = 1 << channel->id;
  for (int i=0; i<nrecv; i++) {
    int peer = peerRecv[i];
    if (peer == -1 || peer >= comm->nRanks || peer == comm->rank || channel->peers[peer].recv.connected) continue;
    comm->connectRecv[peer] |= mask;
  }
  for (int i=0; i<nsend; i++) {
    int peer = peerSend[i];
    if (peer == -1 || peer >= comm->nRanks || peer == comm->rank || channel->peers[peer].send.connected) continue;
    comm->connectSend[peer] |= mask;
  }
  return ncclSuccess;
}

ncclResult_t ncclTransportP2pSetup(struct ncclComm* comm, struct ncclTopoGraph* graph) {
  struct ncclConnect data[2*MAXCHANNELS];
  for (int i=1; i<comm->nRanks; i++) {
    int recvPeer = (comm->rank - i + comm->nRanks) % comm->nRanks;
    int sendPeer = (comm->rank + i) % comm->nRanks;
    uint32_t recvMask = comm->connectRecv[recvPeer];
    uint32_t sendMask = comm->connectSend[sendPeer];

    struct ncclConnect* recvData = data;
    int sendChannels = 0, recvChannels = 0;
    for (int c=0; c<MAXCHANNELS; c++) {
      if (recvMask & (1<<c)) {
        struct ncclConnector* conn = &comm->channels[c].peers[recvPeer].recv;
        NCCLCHECK(selectTransport<0>(comm, graph, comm->peerInfo+comm->rank, comm->peerInfo+recvPeer, recvData+recvChannels++, conn, c));
      }
    }
    struct ncclConnect* sendData = recvData+recvChannels;
    for (int c=0; c<MAXCHANNELS; c++) {
      if (sendMask & (1<<c)) {
        struct ncclConnector* conn = &comm->channels[c].peers[sendPeer].send;
        NCCLCHECK(selectTransport<1>(comm, graph, comm->peerInfo+comm->rank, comm->peerInfo+sendPeer, sendData+sendChannels++, conn, c));
      }
    }

    if (sendPeer == recvPeer) {
      if (recvChannels+sendChannels) {
         //NCCLCHECK(bootstrapSend(comm->bootstrap, recvPeer, data, sizeof(struct ncclConnect)*(recvChannels+sendChannels)));
         //NCCLCHECK(bootstrapRecv(comm->bootstrap, recvPeer, data, sizeof(struct ncclConnect)*(recvChannels+sendChannels)));
         sendData = data;
         recvData = data+sendChannels;
      }
    } else {
      //if (recvChannels) NCCLCHECK(bootstrapSend(comm->bootstrap, recvPeer, recvData, sizeof(struct ncclConnect)*recvChannels));
      //if (sendChannels) NCCLCHECK(bootstrapSend(comm->bootstrap, sendPeer, sendData, sizeof(struct ncclConnect)*sendChannels));
      //if (sendChannels) NCCLCHECK(bootstrapRecv(comm->bootstrap, sendPeer, sendData, sizeof(struct ncclConnect)*sendChannels));
      //if (recvChannels) NCCLCHECK(bootstrapRecv(comm->bootstrap, recvPeer, recvData, sizeof(struct ncclConnect)*recvChannels));
    }

    for (int c=0; c<MAXCHANNELS; c++) {
      if (sendMask & (1<<c)) {
        struct ncclConnector* conn = &comm->channels[c].peers[sendPeer].send;
        //NCCLCHECK(conn->transportComm->connect(comm, sendData++, 1, comm->rank, conn));
        conn->connected = 1;
        //CUDACHECK(hipMemcpy(&comm->channels[c].devPeers[sendPeer].send, conn, sizeof(struct ncclConnector), hipMemcpyHostToDevice));
      }
    }
    for (int c=0; c<MAXCHANNELS; c++) {
      if (recvMask & (1<<c)) {
        struct ncclConnector* conn = &comm->channels[c].peers[recvPeer].recv;
        //NCCLCHECK(conn->transportComm->connect(comm, recvData++, 1, comm->rank, conn));
        conn->connected = 1;
        //CUDACHECK(hipMemcpy(&comm->channels[c].devPeers[recvPeer].recv, conn, sizeof(struct ncclConnector), hipMemcpyHostToDevice));
      }
    }
    comm->connectRecv[recvPeer] = comm->connectSend[sendPeer] = 0;
  }
  return ncclSuccess;
}


ncclResult_t initTransportsRank_3(struct ncclComm* comm, struct allGather3Data_t *allGather3Data,
  struct ncclTopoGraph& treeGraph, struct ncclTopoGraph& ringGraph, struct ncclTopoGraph& collNetGraph) {
  int rank = comm->rank;
  int nranks = comm->nRanks;
  //NCCLCHECK(bootstrapAllGather(comm->bootstrap, allGather3Data, sizeof(*allGather3Data)));

  // Determine nNodes, firstRanks, ...
  int *nodesFirstRank, *nodesTreePatterns;
  NCCLCHECK(ncclCalloc(&nodesFirstRank, nranks));
  NCCLCHECK(ncclCalloc(&nodesTreePatterns, nranks));
  for (int i=0; i<nranks; i++) {
    int node = -1;
    int firstRank = allGather3Data[i].topoRanks.ringRecv[0];
    for (int n=0; n<comm->nNodes; n++) {
      if (nodesFirstRank[n] == firstRank) node = n;
    }
    if (node == -1) {
      node = comm->nNodes++;
      nodesFirstRank[node] = firstRank;
      // Record tree pattern of each node as they can be different depending on sm arch
      nodesTreePatterns[node] = allGather3Data[i].tree.pattern;
    }
    if (i == comm->rank) comm->node = node;
  }

  int nChannelsOrig = comm->nChannels;
  struct ncclTopoRanks** allTopoRanks;
  NCCLCHECK(ncclCalloc(&allTopoRanks, comm->nRanks));
  int gcn = allGather3Data[0].gcn;
  for (int i=0; i<nranks; i++) {
    allTopoRanks[i] = &allGather3Data[i].topoRanks;
    gcn = std::min(allGather3Data[i].gcn, gcn);
    // Make sure we align all ranks so that the tuning is consistent across ranks
    treeGraph.nChannels = ringGraph.nChannels = comm->nChannels = std::min(allGather3Data[i].nChannels, comm->nChannels);
    treeGraph.sameChannels = std::min(allGather3Data[i].tree.sameChannels, treeGraph.sameChannels);
    treeGraph.speedIntra = std::min(allGather3Data[i].tree.speedIntra, treeGraph.speedIntra);
    treeGraph.speedInter = std::min(allGather3Data[i].tree.speedInter, treeGraph.speedInter);
    treeGraph.typeIntra = std::min(allGather3Data[i].tree.typeIntra, treeGraph.typeIntra);
    treeGraph.typeInter = std::min(allGather3Data[i].tree.typeInter, treeGraph.typeInter);
    ringGraph.sameChannels = std::min(allGather3Data[i].ring.sameChannels, ringGraph.sameChannels);
    ringGraph.speedIntra = std::min(allGather3Data[i].ring.speedIntra, ringGraph.speedIntra);
    ringGraph.speedInter = std::min(allGather3Data[i].ring.speedInter, ringGraph.speedInter);
    ringGraph.typeIntra = std::min(allGather3Data[i].ring.typeIntra, ringGraph.typeIntra);
    ringGraph.typeInter = std::min(allGather3Data[i].ring.typeInter, ringGraph.typeInter);
    collNetGraph.sameChannels = std::min(allGather3Data[i].collNet.sameChannels, collNetGraph.sameChannels);
    collNetGraph.speedIntra = std::min(allGather3Data[i].collNet.speedIntra, collNetGraph.speedIntra);
    collNetGraph.speedInter = std::min(allGather3Data[i].collNet.speedInter, collNetGraph.speedInter);
    collNetGraph.typeIntra = std::min(allGather3Data[i].collNet.typeIntra, collNetGraph.typeIntra);
    collNetGraph.typeInter = std::min(allGather3Data[i].collNet.typeInter, collNetGraph.typeInter);
  }

  // count NETs used by ring
  int nNets = 0;
  int nets[MAXCHANNELS*2];
  // do not count NETs in case of single node, i.e comm->topo->nodes[GPU].count == comm->topo->nRanks
  for (int i = 0; comm->topo->nodes[GPU].count != comm->topo->nRanks && i < ringGraph.nChannels; i++) {
    for (int j = 0; j < 2; j++) {
      int k;
      for (k = 0; k < nNets; k++)
        if (nets[k] == ringGraph.inter[i*2+j]) break;
      if (k >= nNets) {
        nets[nNets] = ringGraph.inter[i*2+j];
        nNets++;
      }
    }
  }

  if (comm->nChannels < nChannelsOrig) {
    // We started duplicating channels during Preset(), so we need to move the
    // duplicated channels since we have removed some.
    for (int i=0; i<comm->nChannels; i++) memcpy(comm->channels+comm->nChannels+i, comm->channels+nChannelsOrig+i, sizeof(struct ncclChannel));
  }

  int *rings;
  NCCLCHECK(ncclCalloc(&rings, nranks*MAXCHANNELS));

  NCCLCHECK(ncclTopoPostset(comm, nodesFirstRank, nodesTreePatterns, allTopoRanks, rings, gcn, nNets));
  if (comm->nNodes > 1 &&
      ncclParamCollNetEnable() == 1 &&
      collNetSupport() && collNetGraph.nChannels) {
    NCCLCHECK(ncclTopoConnectCollNet(comm, &collNetGraph, rank));
  } else {
    comm->collNetnChannels = 0;
  }

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

  // Set Affinity to a CPU local the our GPU, so that all memory we allocate
  // on the host is local.
  cpu_set_t affinitySave;
  sched_getaffinity(0, sizeof(cpu_set_t), &affinitySave);
  NCCLCHECK(ncclTopoSetAffinity(comm->topo, comm->rank));
  ncclResult_t ret;

  //NCCLCHECK(computeBuffSizes(comm));

  // Connect with prev/next for each ring
  for (int c=0; c<comm->nChannels; c++) {
    struct ncclChannel* channel = comm->channels+c;
    NCCLCHECKGOTO(setupChannel(comm, c, rank, nranks, rings+c*nranks), ret, affinity_restore);
    if (comm->nRanks == 1) continue;
    NCCLCHECKGOTO(ncclTransportP2pConnect(comm, channel, 1, &channel->ring.prev, 1, &channel->ring.next), ret, affinity_restore);
  }
  NCCLCHECKGOTO(ncclTransportP2pSetup(comm, &ringGraph), ret, affinity_restore);
  INFO(NCCL_INIT, "Connected all rings");

  // Connect Trees
  for (int c=0; c<comm->nChannels; c++) {
    struct ncclChannel* channel = comm->channels+c;
    if (comm->nRanks == 1) continue;
    NCCLCHECKGOTO(ncclTransportP2pConnect(comm, channel, NCCL_MAX_TREE_ARITY, channel->tree.down, 1, &channel->tree.up), ret, affinity_restore);
    NCCLCHECKGOTO(ncclTransportP2pConnect(comm, channel, 1, &channel->tree.up, NCCL_MAX_TREE_ARITY, channel->tree.down), ret, affinity_restore);
  }
  NCCLCHECKGOTO(ncclTransportP2pSetup(comm, &treeGraph), ret, affinity_restore);
  INFO(NCCL_INIT, "Connected all trees");

  // Check if we can setup CollNet
  if (comm->nNodes > 1 &&
      ncclParamCollNetEnable() == 1 &&
      collNetSupport() && collNetGraph.nChannels) {
    for (int c=comm->nChannels; c<comm->collNetnChannels; c++)
      NCCLCHECK(initChannel(comm, c));;
    int logicChannels = comm->collNetnChannels/2;
    int collNetSetupFail = 0;
    const int recvIndex = 0;  // recv GPU index is always 0
    const int sendIndex = collNetGraph.pattern == NCCL_TOPO_PATTERN_TREE ? 0 : 1;  // send GPU index depends on topo pattern
    for (int c=0; c<logicChannels; c++) {
      struct ncclChannel* channelRecv = comm->channels+logicChannels+c;
      struct ncclChannel* channelSend = comm->channels+c;
      NCCLCHECK(ncclTransportP2pConnect(comm, channelRecv, 1, &channelRecv->collTree.up, 1, channelRecv->collTree.down));
      NCCLCHECK(ncclTransportP2pConnect(comm, channelSend, 1, channelSend->collTree.down, 1, &channelSend->collTree.up));
      const int recvMaster = collNetGraph.intra[c*comm->localRanks+recvIndex];
      const int sendMaster = collNetGraph.intra[c*comm->localRanks+sendIndex];
      if (collNetSetup(comm, &collNetGraph, channelRecv, rank, nranks, recvMaster, sendMaster, comm->nNodes, 1) != 1)
        collNetSetupFail = 1;
      else if (collNetSetup(comm, &collNetGraph, channelSend, rank, nranks, sendMaster, recvMaster, comm->nNodes, 0) != 1)
        collNetSetupFail = 1;
    }
    NCCLCHECK(ncclTransportP2pSetup(comm, &collNetGraph));
    // Verify CollNet setup across ranks
    NCCLCHECK(checkCollNetSetup(comm, rank, collNetSetupFail));
  }
  TRACE(NCCL_INIT, "rank %d nranks %d - CONNECTED %d RINGS AND TREES", rank, nranks, comm->nChannels);
  free(rings);

  // Compute time models for algorithm and protocol combinations
  //NCCLCHECK(ncclTopoTuneModel(comm, minCompCap, maxCompCap, &treeGraph, &ringGraph, &collNetGraph));

  // Compute nChannels per peer for p2p
  NCCLCHECK(ncclTopoComputeP2pChannels(comm));

  //NCCLCHECK(ncclCommSetIntra(comm, intraRank, intraRanks, intraRank0Comm));

  //if (comm->nNodes) NCCLCHECK(ncclProxyCreate(comm));

  // We should have allocated all buffers, collective fifos, ... we can
  // restore the affinity.
affinity_restore:
  sched_setaffinity(0, sizeof(cpu_set_t), &affinitySave);
  if (ret != ncclSuccess) return ret;

  TRACE(NCCL_INIT, "rank %d nranks %d - DONE", rank, nranks);

  return ncclSuccess;
}
