/*************************************************************************
 * Copyright (c) 2016-2020, NVIDIA CORPORATION. All rights reserved.
 * Modifications Copyright (c) 2019-2020 Advanced Micro Devices, Inc. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

#include "comm.h"
#include "info.h"
#include "bootstrap.h"
#include "../graph/topo.h"

extern struct ncclTransport p2pTransport;
extern struct ncclTransport shmTransport;
extern struct ncclTransport netTransport;

struct ncclTransport ncclTransports[NTRANSPORTS] = {
  p2pTransport,
  shmTransport,
  netTransport,
};

static ncclResult_t connectedByXGMI(int* ret, struct ncclTopoSystem* system, struct ncclPeerInfo* info1, struct ncclPeerInfo* info2) {
  *ret = 0;
  if (info1->hostHash != info2->hostHash) return ncclSuccess;
  int g1, g2;
  NCCLCHECK(ncclTopoRankToIndex(system, info1->rank, &g1));
  NCCLCHECK(ncclTopoRankToIndex(system, info2->rank, &g2));
  if (system->nodes[GPU].nodes[g1].paths[GPU][g2].type == PATH_NVL) *ret = 1;
  return ncclSuccess;
}

template <int type>
static ncclResult_t selectTransportN(struct ncclComm* comm, struct ncclPeerInfo* myInfo, struct ncclPeerInfo* peerInfo, struct ncclConnect* connect, struct ncclConnector* connector, int channelId, int n) {
  for (int t=n; t<NTRANSPORTS; t++) {
    if (t == TRANSPORT_SHM) continue;
    struct ncclTransport *transport = ncclTransports+t;
    struct ncclTransportComm* transportComm = type == 1 ? &transport->send : &transport->recv;
    int ret = 0;
    NCCLCHECK(transport->canConnect(&ret, comm->topo, NULL, myInfo, peerInfo));
    if (ret) {
      connector->transportComm = transportComm;
      NCCLCHECK(transportComm->setup(comm, NULL, myInfo, peerInfo, connect, connector, channelId));
      return ncclSuccess;
    }
  }
  WARN("No transport found !");
  return ncclInternalError;
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

void dumpData(struct ncclConnect* data, int ndata) {
  for (int n=0; n<ndata; n++) {
    printf("[%d] ", n);
    uint8_t* d = (uint8_t*)data;
    for (int i=0; i<sizeof(struct ncclConnect); i++) printf("%02x", d[i]);
    printf("\n");
  }
}

ncclResult_t ncclTransportP2pSetup(struct ncclComm* comm, struct ncclTopoGraph* graph) {
  struct ncclConnect data[2*MAXCHANNELS];
  uint32_t p2pNet = LOAD(comm->p2pNet);
  for (int i=1; i<comm->nRanks; i++) {
    int recvPeer = (comm->rank - i + comm->nRanks) % comm->nRanks;
    int sendPeer = (comm->rank + i) % comm->nRanks;
    uint32_t recvMask = comm->connectRecv[recvPeer];
    uint32_t sendMask = comm->connectSend[sendPeer];

    struct ncclConnect* recvData = data;
    int sendChannels = 0, recvChannels = 0;
    for (int c=0; c<MAXCHANNELS; c++) {
      if (recvMask & (1<<c)) {
        int xgmi = 0;
        if (p2pNet && graph == NULL) {
          struct ncclConnector* conn = &comm->channels[c].peers[recvPeer].p2pRecv;
          NCCLCHECK(connectedByXGMI(&xgmi, comm->topo, comm->peerInfo+comm->rank, comm->peerInfo+recvPeer));
          if (xgmi) {
            NCCLCHECK(selectTransportN<0>(comm, comm->peerInfo+comm->rank, comm->peerInfo+recvPeer, recvData+recvChannels++, conn, c, TRANSPORT_P2P));
          }
          else {
            NCCLCHECK(selectTransportN<0>(comm, comm->peerInfo+comm->rank, comm->peerInfo+recvPeer, recvData+recvChannels++, conn, c, TRANSPORT_NET));
          }
        }
        else {
          struct ncclConnector* conn = &comm->channels[c].peers[recvPeer].recv;
          NCCLCHECK(selectTransport<0>(comm, graph, comm->peerInfo+comm->rank, comm->peerInfo+recvPeer, recvData+recvChannels++, conn, c));
        }
      }
    }
    struct ncclConnect* sendData = recvData+recvChannels;
    for (int c=0; c<MAXCHANNELS; c++) {
      if (sendMask & (1<<c)) {
        int xgmi = 0;
        if (p2pNet && graph == NULL) {
          struct ncclConnector* conn = &comm->channels[c].peers[sendPeer].p2pSend;
          NCCLCHECK(connectedByXGMI(&xgmi, comm->topo, comm->peerInfo+comm->rank, comm->peerInfo+sendPeer));
          if (xgmi) {
            NCCLCHECK(selectTransportN<1>(comm, comm->peerInfo+comm->rank, comm->peerInfo+sendPeer, sendData+sendChannels++, conn, c, TRANSPORT_P2P));
          }
          else {
            NCCLCHECK(selectTransportN<1>(comm, comm->peerInfo+comm->rank, comm->peerInfo+sendPeer, sendData+sendChannels++, conn, c, TRANSPORT_NET));
          }
        }
        else {
          struct ncclConnector* conn = &comm->channels[c].peers[sendPeer].send;
          NCCLCHECK(selectTransport<1>(comm, graph, comm->peerInfo+comm->rank, comm->peerInfo+sendPeer, sendData+sendChannels++, conn, c));
        }
      }
    }

    if (sendPeer == recvPeer) {
      if (recvChannels+sendChannels) {
         NCCLCHECK(bootstrapSend(comm->bootstrap, recvPeer, data, sizeof(struct ncclConnect)*(recvChannels+sendChannels)));
         NCCLCHECK(bootstrapRecv(comm->bootstrap, recvPeer, data, sizeof(struct ncclConnect)*(recvChannels+sendChannels)));
         sendData = data;
         recvData = data+sendChannels;
      }
    } else {
      if (recvChannels) NCCLCHECK(bootstrapSend(comm->bootstrap, recvPeer, recvData, sizeof(struct ncclConnect)*recvChannels));
      if (sendChannels) NCCLCHECK(bootstrapSend(comm->bootstrap, sendPeer, sendData, sizeof(struct ncclConnect)*sendChannels));
      if (sendChannels) NCCLCHECK(bootstrapRecv(comm->bootstrap, sendPeer, sendData, sizeof(struct ncclConnect)*sendChannels));
      if (recvChannels) NCCLCHECK(bootstrapRecv(comm->bootstrap, recvPeer, recvData, sizeof(struct ncclConnect)*recvChannels));
    }

    for (int c=0; c<MAXCHANNELS; c++) {
      if (sendMask & (1<<c)) {
        struct ncclConnector* conn = (p2pNet && graph == NULL) ? &comm->channels[c].peers[sendPeer].p2pSend
          : &comm->channels[c].peers[sendPeer].send;
        NCCLCHECK(conn->transportComm->connect(comm, sendData++, 1, comm->rank, conn));
        conn->connected = 1;
        if (p2pNet && graph == NULL) CUDACHECK(hipMemcpy(&comm->channels[c].devPeers[sendPeer].p2pSend, conn, sizeof(struct ncclConnector), hipMemcpyHostToDevice));
        else CUDACHECK(hipMemcpy(&comm->channels[c].devPeers[sendPeer].send, conn, sizeof(struct ncclConnector), hipMemcpyHostToDevice));
      }
    }
    for (int c=0; c<MAXCHANNELS; c++) {
      if (recvMask & (1<<c)) {
        struct ncclConnector* conn = (p2pNet && graph == NULL) ? &comm->channels[c].peers[recvPeer].p2pRecv
          : &comm->channels[c].peers[recvPeer].recv;
        NCCLCHECK(conn->transportComm->connect(comm, recvData++, 1, comm->rank, conn));
        conn->connected = 1;
        if (p2pNet && graph == NULL) CUDACHECK(hipMemcpy(&comm->channels[c].devPeers[recvPeer].p2pRecv, conn, sizeof(struct ncclConnector), hipMemcpyHostToDevice));
        else CUDACHECK(hipMemcpy(&comm->channels[c].devPeers[recvPeer].recv, conn, sizeof(struct ncclConnector), hipMemcpyHostToDevice));
      }
    }
    comm->connectRecv[recvPeer] = comm->connectSend[sendPeer] = 0;
  }
  return ncclSuccess;
}
