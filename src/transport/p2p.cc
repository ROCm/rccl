/*************************************************************************
 * Copyright (c) 2016-2021, NVIDIA CORPORATION. All rights reserved.
 * Modifications Copyright (c) 2019-2021 Advanced Micro Devices, Inc. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

#include "comm.h"
#include "graph.h"
#include "utils.h"
#include "bootstrap.h"
#include "rocm_smi_wrap.h"

struct p2pConnectInfo {
  int rank;
  int read;
  void* directPtr;
  hipIpcMemHandle_t devIpc;
};

struct p2pSendResources {
  struct ncclSendMem* devMem;
  void* ipcPtr;
  uint32_t* next_hdp_reg;  // Next GPU in ring (for p2p transport use only)
  int remoteId;
  int memRank;
  void* bootstrap;
};

struct p2pRecvResources {
  struct ncclRecvMem* devMem;
  void* ipcPtr;
  int remoteId;
  int memRank;
  void* bootstrap;
};

#include <sys/types.h>

/* Convert a PCI busId string into a local cudaDev device index (cf. CUDA_VISIBLE_DEVICES) */
int busIdToCudaDev(int64_t busId) {
  int ndev;
  if (hipGetDeviceCount(&ndev) != hipSuccess)
    return -1;
  for (int i = 0; i < ndev; i++) {
    char devBusIdStr[NVML_DEVICE_PCI_BUS_ID_BUFFER_SIZE];
    if (hipDeviceGetPCIBusId(devBusIdStr, NVML_DEVICE_PCI_BUS_ID_BUFFER_SIZE, i) != hipSuccess)
      return -1;
    int64_t devBusId;
    NCCLCHECK(busIdToInt64(devBusIdStr, &devBusId));
    if (busId == devBusId) return i;
  }
  // BusId was not found in our locally visible CUDA devices
  return -1;
}

/* Determine if two peers can communicate through p2p */
ncclResult_t p2pCanConnect(int* ret, struct ncclTopoSystem* topo, struct ncclTopoGraph* graph, struct ncclPeerInfo* info1, struct ncclPeerInfo* info2) {
#if defined(__HIP_PLATFORM_HCC__) || defined(__HCC__) || defined(__HIPCC__)
  if (!hasFineGrainVramPcie())  {
    *ret = 0;
    return ncclSuccess;
  }
#endif

  // Rule out different nodes
  if (info1->hostHash != info2->hostHash) {
    *ret = 0;
    return ncclSuccess;
  }

  // Check topology / p2p level.
  int intermediateRank;
  NCCLCHECK(ncclTopoCheckP2p(topo, info1->busId, info2->busId, ret, NULL, &intermediateRank));
  if (*ret == 0) return ncclSuccess;
  if (intermediateRank != -1) return ncclSuccess;

  // Convert the peer's busId into a local cudaDev index (cf. CUDA_VISIBLE_DEVICES)
  int cudaDev1 = busIdToCudaDev(info1->busId);
  int cudaDev2 = busIdToCudaDev(info2->busId);
  if (cudaDev1 == -1 || cudaDev2 == -1) {
#if CUDART_VERSION >= 10010
    // CUDA 10.1 and later can use P2P with invisible devices.
    return ncclSuccess;
#else
    // Peer's CUDA device is not visible in this process : we can't communicate with it.
    *ret = 0;
    return ncclSuccess;
#endif
  }

#if defined(__HIP_PLATFORM_HCC__) || defined(__HCC__) || defined(__HIPCC__)
  int dev;
  CUDACHECK(hipGetDevice(&dev));
  CUDACHECK(hipSetDevice(cudaDev2));
  if (!hasFineGrainVramPcie())  {
    *ret = 0;
    CUDACHECK(hipSetDevice(dev));
    return ncclSuccess;
  }
  CUDACHECK(hipSetDevice(dev));
#endif
  // Check that CUDA can do P2P
  int p2p;
  if (hipDeviceCanAccessPeer(&p2p, cudaDev1, cudaDev2) != hipSuccess) {
    INFO(NCCL_INIT|NCCL_P2P,"peer query failed between dev %d(=%lx) and dev %d(=%lx)",
         cudaDev1, info1->busId, cudaDev2, info2->busId);
    *ret = 0;
    return ncclSuccess;
  }
  if (p2p == 0) {
    INFO(NCCL_INIT|NCCL_P2P,"Could not enable P2P between dev %d(=%lx) and dev %d(=%lx)",
         cudaDev1, info1->busId, cudaDev2, info2->busId);
    *ret = 0;
    return ncclSuccess;
  }
  return ncclSuccess;
}

#define TRACE_DUMP_IPC(DEVIPC)                                                             \
  do {                                                                                     \
    unsigned long *devIpc = (unsigned long *) (DEVIPC);                                    \
    TRACE(P2P,"IPC: %016lx %016lx %016lx %016lx", devIpc[0], devIpc[1], devIpc[2], devIpc[3]); \
    TRACE(P2P,"IPC: %016lx %016lx %016lx %016lx", devIpc[4], devIpc[5], devIpc[6], devIpc[7]); \
  } while (0)

// Setting this to non zero causes P2P to use Reads rather than Writes
NCCL_PARAM(P2pReadEnable, "P2P_READ_ENABLE", -2);

static ncclResult_t p2pGetInfo(struct ncclTopoSystem* topo, struct ncclPeerInfo* info1, struct ncclPeerInfo* info2, int* read, int* intermediateRank) {
  int p2p;
  // Queries the topology to see if the GPUs are Ampere and
  // connected via NVLink, if so we enable P2P Read by default
  NCCLCHECK(ncclTopoCheckP2p(topo, info1->busId, info2->busId, &p2p, read, intermediateRank));

  int readEnable = ncclParamP2pReadEnable();
  if (readEnable != -2) *read = readEnable;
  return ncclSuccess;
}

static ncclResult_t p2pMap(struct ncclPeerInfo* myInfo, struct ncclPeerInfo* peerInfo, struct p2pConnectInfo* p2pInfo, void** devMem, void** ipcPtr) {
  if (myInfo->pidHash == peerInfo->pidHash) {
    if (peerInfo->cudaDev != myInfo->cudaDev) {
      // Enable P2P access
      hipError_t err = hipDeviceEnablePeerAccess(peerInfo->cudaDev, 0);
      if (err == hipErrorPeerAccessAlreadyEnabled) {
        hipGetLastError();
      } else if (err != hipSuccess) {
        WARN("failed to peer with device %d(=%lx): %d %s",
            peerInfo->cudaDev, peerInfo->busId, err, hipGetErrorString(err));
        return ncclInternalError;
      }
    }
    *devMem = p2pInfo->directPtr;
    *ipcPtr = NULL;
  } else {
    CUDACHECK(hipIpcOpenMemHandle(devMem, p2pInfo->devIpc, hipIpcMemLazyEnablePeerAccess));
    *ipcPtr = *devMem;
  }
  return ncclSuccess;
}

/* Send: Create and return connect structures for this peer to connect to me */
ncclResult_t p2pSendSetup(struct ncclComm* comm, struct ncclTopoGraph* graph, struct ncclPeerInfo* myInfo, struct ncclPeerInfo* peerInfo,
    struct ncclConnect* connectInfo, struct ncclConnector* send, int channelId, int connIndex) {
  struct p2pSendResources* resources;
  NCCLCHECK(ncclCalloc(&resources, 1));
  send->transportResources = resources;
  int useRead, intermediateRank;
  NCCLCHECK(p2pGetInfo(comm->topo, myInfo, peerInfo, &useRead, &intermediateRank));

  resources->next_hdp_reg = 0;
  RSMI_IO_LINK_TYPE linktype;
  int hops, bw;
  if (rocm_smi_getLinkInfo(myInfo->cudaDev, peerInfo->cudaDev, &linktype, &hops, &bw) != ncclSuccess) {
    INFO(NCCL_INIT|NCCL_P2P,"Ring %02d : %d -> %d failed to get link type and hop count", channelId, myInfo->rank, peerInfo->rank);
    return ncclInternalError;
  }
  if (linktype != RSMI_IOLINK_TYPE_XGMI) {
    CUDACHECK(hipDeviceGetAttribute((int*)&resources->next_hdp_reg, hipDeviceAttributeHdpMemFlushCntl,peerInfo->cudaDev));
    TRACE(NCCL_INIT|NCCL_P2P,"Ring %02d : %d -> %d HDP %p", channelId, myInfo->rank, peerInfo->rank, resources->next_hdp_reg);
  }

  struct p2pConnectInfo info;
  // For CollNet, we use write for scatter-reduce (conn 1), read for broadcast-gather (conn 0)
  info.read = (connIndex == 0) ? useRead : 0;
  const char* useReadStr = info.read ? "/read" : "";

  int sendSize = sizeof(struct ncclSendMem);
  // For P2P Read the SIMPLE buffer is tagged on the end of the ncclSendMem structure
  if (info.read) sendSize += send->comm->buffSizes[NCCL_PROTO_SIMPLE];
  ALIGN_SIZE(sendSize, CUDA_IPC_MIN);

  resources->remoteId = -1;
  resources->bootstrap = comm->bootstrap;
  if (intermediateRank == -1) {
    NCCLCHECK(ncclCudaCalloc((char**)&info.directPtr, sendSize, true));
    info.rank = myInfo->rank;
    if (myInfo->pidHash == peerInfo->pidHash) {
      if (info.read == 0) send->conn.direct |= NCCL_DIRECT_GPU;
      INFO(NCCL_INIT|NCCL_P2P, "Channel %02d : %d[%lx] -> %d[%lx] via P2P/direct pointer%s comm %p nRanks %02d",
          channelId, myInfo->rank, myInfo->busId, peerInfo->rank, peerInfo->busId, useReadStr, comm, comm->nRanks);
    } else {
      CUDACHECK(hipIpcGetMemHandle(&info.devIpc, info.directPtr));
      INFO(NCCL_INIT|NCCL_P2P,"Channel %02d : %d[%lx] -> %d[%lx] via P2P/IPC%s comm %p nRanks %02d",
          channelId, myInfo->rank, myInfo->busId, peerInfo->rank, peerInfo->busId, useReadStr, comm, comm->nRanks);
    }
  } else {
    NCCLCHECK(bootstrapRemAlloc(sendSize, intermediateRank, resources->bootstrap, &resources->remoteId, &info.devIpc, &info.directPtr));
    info.rank = intermediateRank;
    INFO(NCCL_INIT|NCCL_P2P, "Channel %02d : %d[%lx] -> %d[%lx] via P2P/indirect/%d[%lx]%s comm %p nRanks %02d",
        channelId, myInfo->rank, myInfo->busId, peerInfo->rank, peerInfo->busId, intermediateRank,
	comm->peerInfo[intermediateRank].busId, useReadStr, comm, comm->nRanks);
  }
  resources->memRank = info.rank;

  NCCLCHECK(p2pMap(myInfo, comm->peerInfo+info.rank, &info, (void**)&resources->devMem, &resources->ipcPtr));

  static_assert(sizeof(struct p2pConnectInfo) <= sizeof(struct ncclConnect), "p2p Connect Info is too big");
  memcpy(connectInfo, &info, sizeof(struct p2pConnectInfo));
  return ncclSuccess;
}

/* Create and return connect structures for this peer to connect to me */
ncclResult_t p2pRecvSetup(struct ncclComm* comm, struct ncclTopoGraph* graph, struct ncclPeerInfo* myInfo, struct ncclPeerInfo* peerInfo,
    struct ncclConnect* connectInfo, struct ncclConnector * recv, int channelId, int connIndex) {
  struct p2pRecvResources* resources;
  NCCLCHECK(ncclCalloc(&resources, 1));
  recv->transportResources = resources;
  int useRead, intermediateRank;
  NCCLCHECK(p2pGetInfo(comm->topo, myInfo, peerInfo, &useRead, &intermediateRank));

  struct p2pConnectInfo info;
  // For CollNet, we use write for scatter-reduce (conn 1), read for broadcast-gather (conn 0)
  info.read = (connIndex == 0) ? useRead : 0;

  int recvSize = offsetof(struct ncclRecvMem, buff);
  // For P2P Read the SIMPLE buffer is tagged on the end of the ncclSendMem structure
  for (int p=0; p<NCCL_NUM_PROTOCOLS; p++) if (!(info.read && p == NCCL_PROTO_SIMPLE)) recvSize += recv->comm->buffSizes[p];
  ALIGN_SIZE(recvSize, CUDA_IPC_MIN);

  resources->remoteId = -1;
  resources->bootstrap = comm->bootstrap;
  if (intermediateRank == -1) {
    NCCLCHECK(ncclCudaCalloc((char**)&info.directPtr, recvSize, true));
    info.rank = myInfo->rank;
    if (myInfo->pidHash == peerInfo->pidHash) {
      if (info.read == 0) recv->conn.direct |= NCCL_DIRECT_GPU;
    } else {
      CUDACHECK(hipIpcGetMemHandle(&info.devIpc, info.directPtr));
    }
  } else {
    NCCLCHECK(bootstrapRemAlloc(recvSize, intermediateRank, resources->bootstrap, &resources->remoteId, &info.devIpc, &info.directPtr));
    info.rank = intermediateRank;
  }
  resources->memRank = info.rank;

  NCCLCHECK(p2pMap(myInfo, comm->peerInfo+info.rank, &info, (void**)&resources->devMem, &resources->ipcPtr));

  static_assert(sizeof(struct p2pConnectInfo) <= sizeof(struct ncclConnect), "p2p Connect Info is too big");
  memcpy(connectInfo, &info, sizeof(struct p2pConnectInfo));
  return ncclSuccess;
}

/* Connect/Send to this peer */
static ncclResult_t p2pSendConnect(struct ncclComm* comm, struct ncclConnect* connectInfo, int nranks, int rank, struct ncclConnector* send) {
  struct p2pSendResources* resources = (struct p2pSendResources*)send->transportResources;
  struct ncclRecvMem* remDevMem;
  struct p2pConnectInfo* info = (struct p2pConnectInfo*)connectInfo;

  NCCLCHECK(p2pMap(comm->peerInfo+rank, comm->peerInfo+info->rank, info, (void**)&remDevMem, &resources->ipcPtr));

  int offset = 0;
  for (int p=0; p<NCCL_NUM_PROTOCOLS; p++) {
    if (info->read && p == NCCL_PROTO_SIMPLE) {
      /* For P2P Read the SIMPLE buffer is local (ncclSendMem) */
      send->conn.buffs[p] = resources->devMem->buff;
    } else {
      send->conn.buffs[p] = remDevMem->buff + offset;
      offset += send->comm->buffSizes[p];
    }
  }
  send->conn.tail = &remDevMem->tail;
  send->conn.head = &resources->devMem->head;
  send->conn.ptrExchange = &resources->devMem->ptrExchange;
  send->conn.next_hdp_reg = resources->next_hdp_reg;
  return ncclSuccess;
}

/* Connect/Recv from this peer */
ncclResult_t p2pRecvConnect(struct ncclComm* comm, struct ncclConnect* connectInfo, int nranks, int rank, struct ncclConnector* recv) {
  struct p2pRecvResources* resources = (struct p2pRecvResources*)recv->transportResources;
  struct ncclSendMem* remDevMem;
  struct p2pConnectInfo* info = (struct p2pConnectInfo*)connectInfo;

  NCCLCHECK(p2pMap(comm->peerInfo+rank, comm->peerInfo+info->rank, info, (void**)&remDevMem, &resources->ipcPtr));

  int offset = 0;
  for (int p=0; p<NCCL_NUM_PROTOCOLS; p++) {
    if (info->read && p == NCCL_PROTO_SIMPLE) {
      /* For P2P Read the SIMPLE buffer is remote (ncclSendMem) */
      recv->conn.buffs[p] = remDevMem->buff;
    } else {
      recv->conn.buffs[p] = resources->devMem->buff + offset;
      offset += recv->comm->buffSizes[p];
    }
  }
  recv->conn.tail = &resources->devMem->tail;
  recv->conn.head = &remDevMem->head;
  recv->conn.ptrExchange = &remDevMem->ptrExchange;
  return ncclSuccess;
}

ncclResult_t p2pSendFree(void* resources) {
  struct p2pSendResources* sendRes = (struct p2pSendResources*)resources;
  if (sendRes->ipcPtr)
    CUDACHECK(hipIpcCloseMemHandle(sendRes->ipcPtr));
  if (sendRes->remoteId != -1) {
    NCCLCHECK(bootstrapRemFree(sendRes->remoteId, sendRes->memRank, sendRes->bootstrap));
    sendRes->devMem = NULL;
  }
  CUDACHECK(hipFree(sendRes->devMem));
  free(sendRes);
  return ncclSuccess;
}

ncclResult_t p2pRecvFree(void* resources) {
  struct p2pRecvResources* recvRes = (struct p2pRecvResources*)resources;
  if (recvRes->ipcPtr)
    CUDACHECK(hipIpcCloseMemHandle(recvRes->ipcPtr));
  if (recvRes->remoteId != -1) {
    NCCLCHECK(bootstrapRemFree(recvRes->remoteId, recvRes->memRank, recvRes->bootstrap));
    recvRes->devMem = NULL;
  }
  CUDACHECK(hipFree(recvRes->devMem));
  free(recvRes);
  return ncclSuccess;
}

struct ncclTransport p2pTransport = {
  "P2P",
  p2pCanConnect,
  { p2pSendSetup, p2pSendConnect, p2pSendFree, NULL },
  { p2pRecvSetup, p2pRecvConnect, p2pRecvFree, NULL }
};
