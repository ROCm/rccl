/*************************************************************************
 * Copyright (c) 2016-2020, NVIDIA CORPORATION. All rights reserved.
 * Modifications Copyright (c) 2019-2020 Advanced Micro Devices, Inc. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

#include "comm.h"
#include "net.h"
#include "graph.h"
#include <sys/time.h>
#include <numaif.h>

struct netConnectInfo {
  ncclNetHandle_t netHandle;
};

#define LOC_HOSTMEM 0
#define LOC_DEVMEM  1
#define LOC_COUNT   2

struct netSendResources {
  void* netSendComm;
  struct ncclSendMem* sendMem;
  struct ncclRecvMem* recvMem;
  int netDev;
  int useGdr;
  char* buffers[LOC_COUNT];
  int buffSizes[LOC_COUNT];
  void* mhandles[LOC_COUNT];
  void** mhandlesProto[NCCL_NUM_PROTOCOLS];
  uint64_t step;
  uint64_t llLastCleaning;
};

struct netRecvResources {
  void* netListenComm;
  void* netRecvComm;
  struct ncclSendMem* sendMem;
  struct ncclRecvMem* recvMem;
  int netDev;
  int useGdr;
  char* buffers[LOC_COUNT];
  int buffSizes[LOC_COUNT];
  void* mhandles[LOC_COUNT];
  void** mhandlesProto[NCCL_NUM_PROTOCOLS];
  uint64_t step;
  uint64_t llLastCleaning;
  uint32_t* curr_hdp_reg;  // Curr GPU in ring (for rdma transport use only)
};

/* Determine if two peers can communicate with NET */
ncclResult_t netCanConnect(int* ret, struct ncclTopoSystem* topo, struct ncclTopoGraph* graph, struct ncclPeerInfo* info1, struct ncclPeerInfo* info2) {
  *ret = 1;
  return ncclSuccess;
}

/* Determine if we will use this transport for this peer and return connect
 * information for this peer */
ncclResult_t netSendSetup(struct ncclTopoSystem* topo, struct ncclTopoGraph* graph, struct ncclPeerInfo* myInfo, struct ncclPeerInfo* peerInfo, struct ncclConnect* connectInfo, struct ncclConnector* send, int channelId) {
  struct netSendResources* resources;
  NCCLCHECK(ncclCalloc(&resources, 1));
  send->transportResources = resources;

  NCCLCHECK(ncclTopoGetNetDev(topo, myInfo->rank, graph, channelId, &resources->netDev));
  NCCLCHECK(ncclTopoCheckGdr(topo, myInfo->busId, resources->netDev, 1, &resources->useGdr));

  NCCLCHECK(ncclCudaHostCalloc(&resources->sendMem, 1));
  NCCLCHECK(ncclCudaHostCalloc(&resources->recvMem, 1));

  send->conn.direct |= resources->useGdr ? NCCL_DIRECT_NIC : 0;
  send->conn.tail = &resources->recvMem->tail;
  send->conn.opCountRem = &resources->recvMem->opCount;
  send->conn.fifo = resources->recvMem->sizesFifo;
  send->conn.head = &resources->sendMem->head;
  send->conn.opCountLoc = &resources->sendMem->opCount;
  for (int i=0; i<NCCL_STEPS; i++) send->conn.fifo[i] = -1;

  int protoLoc[NCCL_NUM_PROTOCOLS];
  for (int p=0; p<NCCL_NUM_PROTOCOLS; p++) {
    protoLoc[p] = p != NCCL_PROTO_LL && resources->useGdr ? LOC_DEVMEM : LOC_HOSTMEM;
  }

  int buffSizes[NCCL_NUM_PROTOCOLS];
  for (int p=0; p<NCCL_NUM_PROTOCOLS; p++) {
    // Only allocate buffers for simple for p2p connections
    buffSizes[p] = graph == NULL && p != NCCL_PROTO_SIMPLE ? 0 : send->comm->buffSizes[p];
    resources->buffSizes[protoLoc[p]] += buffSizes[p];
  }

  if (resources->buffSizes[LOC_DEVMEM]) {
    NCCLCHECK(ncclCudaCalloc(resources->buffers+LOC_DEVMEM, resources->buffSizes[LOC_DEVMEM], resources->useGdr));
  }
  char line[16];
  if (resources->buffSizes[LOC_HOSTMEM]) {
    NCCLCHECK(ncclCudaHostCalloc(resources->buffers+LOC_HOSTMEM, resources->buffSizes[LOC_HOSTMEM]));
    int status[1] = {-1};
    line[0]= 0;
    if (!move_pages(0, 1, (void **)resources->buffers+LOC_HOSTMEM, NULL, status, 0))
      sprintf(line, "/MEM%d", status[0]);
  }

  int offsets[LOC_COUNT];
  offsets[LOC_HOSTMEM] = offsets[LOC_DEVMEM] = 0;
  for (int p=0; p<NCCL_NUM_PROTOCOLS; p++) {
    resources->mhandlesProto[p] = resources->mhandles+protoLoc[p];
    send->conn.buffs[p] = resources->buffers[protoLoc[p]] + offsets[protoLoc[p]];
    offsets[protoLoc[p]] += buffSizes[p];
  }

  INFO(NCCL_INIT|NCCL_NET,"Channel %02d : %d[%lx] -> %d[%lx] [send] via NET/%s/%d%s", channelId, myInfo->rank, myInfo->busId, peerInfo->rank, peerInfo->busId, ncclNetName(), resources->netDev,
      resources->useGdr ? "/GDRDMA" : line);
  return ncclSuccess;
}

ncclResult_t netRecvSetup(struct ncclTopoSystem* topo, struct ncclTopoGraph* graph, struct ncclPeerInfo* myInfo, struct ncclPeerInfo* peerInfo, struct ncclConnect* connectInfo, struct ncclConnector* recv, int channelId) {
  struct netRecvResources* resources;
  NCCLCHECK(ncclCalloc(&resources, 1));
  recv->transportResources = resources;

  NCCLCHECK(ncclTopoGetNetDev(topo, myInfo->rank, graph, channelId, &resources->netDev));
  NCCLCHECK(ncclTopoCheckGdr(topo, myInfo->busId, resources->netDev, 0, &resources->useGdr));

  NCCLCHECK(ncclCudaHostCalloc(&resources->sendMem, 1));
  NCCLCHECK(ncclCudaHostCalloc(&resources->recvMem, 1));

  recv->conn.direct |= resources->useGdr ? NCCL_DIRECT_NIC : 0;
  recv->conn.tail = &resources->recvMem->tail;
  recv->conn.opCountLoc = &resources->recvMem->opCount;
  recv->conn.head = &resources->sendMem->head;
  recv->conn.opCountRem = &resources->sendMem->opCount;

  int protoLoc[NCCL_NUM_PROTOCOLS];
  for (int p=0; p<NCCL_NUM_PROTOCOLS; p++) {
    protoLoc[p] = resources->useGdr ? LOC_DEVMEM : LOC_HOSTMEM;
  }

  int buffSizes[NCCL_NUM_PROTOCOLS];
  for (int p=0; p<NCCL_NUM_PROTOCOLS; p++) {
    // Only allocate buffers for simple for p2p connections
    buffSizes[p] = graph == NULL && p != NCCL_PROTO_SIMPLE ? 0 : recv->comm->buffSizes[p];
    resources->buffSizes[protoLoc[p]] += buffSizes[p];
  }

  if (resources->buffSizes[LOC_DEVMEM]) {
    NCCLCHECK(ncclCudaCalloc(resources->buffers+LOC_DEVMEM, resources->buffSizes[LOC_DEVMEM], resources->useGdr));
  }
  char line[16];
  if (resources->buffSizes[LOC_HOSTMEM]) {
    NCCLCHECK(ncclCudaHostCalloc(resources->buffers+LOC_HOSTMEM, resources->buffSizes[LOC_HOSTMEM]));
    int status[1] = {-1};
    line[0]= 0;
    if (!move_pages(0, 1, (void **)resources->buffers+LOC_HOSTMEM, NULL, status, 0))
      sprintf(line, "/MEM%d", status[0]);
  }

  int offsets[LOC_COUNT];
  offsets[LOC_HOSTMEM] = offsets[LOC_DEVMEM] = 0;
  for (int p=0; p<NCCL_NUM_PROTOCOLS; p++) {
    resources->mhandlesProto[p] = resources->mhandles+protoLoc[p];
    recv->conn.buffs[p] = resources->buffers[protoLoc[p]] + offsets[protoLoc[p]];
    offsets[protoLoc[p]] += buffSizes[p];
  }

  INFO(NCCL_INIT|NCCL_NET,"Channel %02d : %d[%lx] -> %d[%lx] [receive] via NET/%s/%d%s", channelId, peerInfo->rank, peerInfo->busId, myInfo->rank, myInfo->busId, ncclNetName(), resources->netDev,
      resources->useGdr ? "/GDRDMA" : line);
  struct netConnectInfo* info = (struct netConnectInfo*) connectInfo;
  NCCLCHECK(ncclNetListen(resources->netDev, &info->netHandle, &resources->netListenComm));

  return ncclSuccess;
}

ncclResult_t netSendConnect(struct ncclConnect* connectInfo, int nranks, int rank, struct ncclConnector* send) {
  // Setup device pointers
  struct netSendResources* resources = (struct netSendResources*)send->transportResources;
  struct netConnectInfo* info = (struct netConnectInfo*)connectInfo;

  // Connect to remote peer
  NCCLCHECK(ncclNetConnect(resources->netDev, info->netHandle, &resources->netSendComm));

  if (resources->buffSizes[LOC_DEVMEM]) {
    NCCLCHECK(ncclNetRegMr(resources->netSendComm, resources->buffers[LOC_DEVMEM], resources->buffSizes[LOC_DEVMEM], NCCL_PTR_CUDA, &resources->mhandles[LOC_DEVMEM]));
  }
  if (resources->buffSizes[LOC_HOSTMEM]) {
    NCCLCHECK(ncclNetRegMr(resources->netSendComm, resources->buffers[LOC_HOSTMEM], resources->buffSizes[LOC_HOSTMEM], NCCL_PTR_HOST, &resources->mhandles[LOC_HOSTMEM]));
  }
  return ncclSuccess;
}

/* Connect to this peer */
ncclResult_t netRecvConnect(struct ncclConnect* connectInfo, int nranks, int rank, struct ncclConnector* recv) {
  // Setup device pointers
  struct netRecvResources* resources = (struct netRecvResources*)recv->transportResources;

  // Finish connection establishment from remote peer
  NCCLCHECK(ncclNetAccept(resources->netListenComm, &resources->netRecvComm));
  NCCLCHECK(ncclNetCloseListen(resources->netListenComm));

  if (resources->buffSizes[LOC_DEVMEM]) {
    NCCLCHECK(ncclNetRegMr(resources->netRecvComm, resources->buffers[LOC_DEVMEM], resources->buffSizes[LOC_DEVMEM], NCCL_PTR_CUDA, &resources->mhandles[LOC_DEVMEM]));
  }
  if (resources->buffSizes[LOC_HOSTMEM]) {
    NCCLCHECK(ncclNetRegMr(resources->netRecvComm, resources->buffers[LOC_HOSTMEM], resources->buffSizes[LOC_HOSTMEM], NCCL_PTR_HOST, &resources->mhandles[LOC_HOSTMEM]));
  }
  return ncclSuccess;
}

ncclResult_t netSendFree(void* transportResources) {
  struct netSendResources* resources = (struct netSendResources*)transportResources;
  NCCLCHECK(ncclCudaHostFree(resources->sendMem));
  NCCLCHECK(ncclCudaHostFree(resources->recvMem));
  for (int l=0; l<LOC_COUNT; l++) {
    if (resources->buffers[l])
      NCCLCHECK(ncclNetDeregMr(resources->netSendComm, resources->mhandles[l]));
  }
  NCCLCHECK(ncclCudaHostFree(resources->buffers[LOC_HOSTMEM]));
  CUDACHECK(hipFree(resources->buffers[LOC_DEVMEM]));
  NCCLCHECK(ncclNetCloseSend(resources->netSendComm));
  free(resources);
  return ncclSuccess;
}

ncclResult_t netRecvFree(void* transportResources) {
  struct netRecvResources* resources = (struct netRecvResources*)transportResources;
  NCCLCHECK(ncclCudaHostFree(resources->sendMem));
  NCCLCHECK(ncclCudaHostFree(resources->recvMem));
  for (int l=0; l<LOC_COUNT; l++) {
    if (resources->buffers[l])
      NCCLCHECK(ncclNetDeregMr(resources->netRecvComm, resources->mhandles[l]));
  }
  NCCLCHECK(ncclCudaHostFree(resources->buffers[LOC_HOSTMEM]));
  CUDACHECK(hipFree(resources->buffers[LOC_DEVMEM]));
  NCCLCHECK(ncclNetCloseRecv(resources->netRecvComm));
  free(resources);
  return ncclSuccess;
}

ncclResult_t netSendProxy(struct ncclProxyArgs* args) {
  struct netSendResources* resources = (struct netSendResources*) (args->connector->transportResources);
  if (args->state == ncclProxyOpReady) {
    // Update opCount
    STORE(&resources->recvMem->opCount, args->opCount);

    // Round to next multiple of sliceSteps
    resources->step = ROUNDUP(resources->step, args->chunkSteps);
    args->head = resources->step;
    args->tail = resources->step;
    args->end = args->head + args->nsteps;
    args->state = ncclProxyOpProgress;
  }
  if (args->state == ncclProxyOpProgress) {
    int p = args->protocol;
    int stepSize = args->connector->comm->buffSizes[p] / NCCL_STEPS;
    char* localBuff = args->connector->conn.buffs[p];
    void* mhandle = *(resources->mhandlesProto[p]);
    args->idle = 1;
    if (args->head < args->end) {
      int buffSlot = args->tail%NCCL_STEPS;
      if (args->tail < args->end && args->tail < args->head + NCCL_STEPS) {
        volatile int* sizesFifo = resources->recvMem->sizesFifo;
        volatile uint64_t* recvTail = &resources->recvMem->tail;
        if (args->protocol == NCCL_PROTO_LL128) {
          if (args->tail < LOAD(recvTail)) {
            if (LOAD(sizesFifo+buffSlot) != -1) {
              int ready = resources->useGdr;
              if (!ready) {
                // When data is in sysmem, we need to wait until all flags are correct since the GPU only
                // called threadfence()
                uint64_t flag = args->tail + 1;
                int nFifoLines = DIVUP(LOAD(sizesFifo+buffSlot), sizeof(uint64_t)*NCCL_LL128_LINEELEMS);
                volatile uint64_t* lines = (volatile uint64_t*)(localBuff+buffSlot*stepSize);
                ready = 1;
                for (int i=0; i<nFifoLines; i++) {
                  if (LOAD(lines+i*NCCL_LL128_LINEELEMS+NCCL_LL128_DATAELEMS) != flag) { ready = 0; break; }
                }
              }
              if (ready) {
                // Send through network
                NCCLCHECK(ncclNetIsend(resources->netSendComm, localBuff+buffSlot*stepSize, LOAD(sizesFifo+buffSlot), mhandle, args->requests+buffSlot));
                if (args->requests[buffSlot] != NULL) {
                  STORE(sizesFifo+buffSlot, -1);
                  // Make sure size is reset to zero before we update the head.
                  __sync_synchronize();
                  args->tail += args->sliceSteps;
                  args->idle = 0;
                }
              }
            }
          }
        } else if (args->protocol == NCCL_PROTO_LL) {
          int size = LOAD(sizesFifo+buffSlot);
          if (size != -1) {
            uint32_t flag = NCCL_LL_FLAG(args->tail + 1);
            int nFifoLines = DIVUP(size, sizeof(union ncclLLFifoLine));
            size = nFifoLines * sizeof(union ncclLLFifoLine);
            union ncclLLFifoLine* lines = (union ncclLLFifoLine*)(localBuff+buffSlot*stepSize);
            int ready = 1;
            for (int i=0; i<nFifoLines; i++) {
              volatile uint32_t *f1 = &lines[i].flag1;
              volatile uint32_t *f2 = &lines[i].flag2;
              if (LOAD(f1) != flag || LOAD(f2) != flag) { ready = 0; break; }
            }
            if (ready) {
              NCCLCHECK(ncclNetIsend(resources->netSendComm, lines, size, mhandle, args->requests+buffSlot));
              if (args->requests[buffSlot] != NULL) {
                STORE(sizesFifo+buffSlot, -1);
                // Make sure size is reset to zero before we update the head.
                __sync_synchronize();
                args->tail += args->sliceSteps;
                args->idle = 0;
              }
            }
          }
        } else if (args->tail < LOAD(recvTail)) {
          // Send through network
          if (LOAD(sizesFifo+buffSlot) != -1) {
            NCCLCHECK(ncclNetIsend(resources->netSendComm, localBuff+buffSlot*stepSize, sizesFifo[buffSlot], mhandle, args->requests+buffSlot));
            if (args->requests[buffSlot] != NULL) {
#ifdef ENABLE_PROFILING
              if (args->channel->active_req == 0) {
                gettimeofday(&args->channel->tvs, NULL);
                args->channel->sizes = 0;
              }
              args->channel->active_req ++;
              args->channel->sizes += LOAD(sizesFifo+buffSlot);
              args->channel->send_byte += LOAD(sizesFifo+buffSlot);
#endif
              STORE(sizesFifo+buffSlot, -1);
              // Make sure size is reset to zero before we update the head.
              __sync_synchronize();
              args->tail += args->sliceSteps;
              args->idle = 0;
            }
          }
        }
      }
      if (args->head < args->tail) {
        int done;
        int buffSlot = args->head%NCCL_STEPS;
        NCCLCHECK(ncclNetTest(args->requests[buffSlot], &done, NULL));
        if (done) {
#ifdef ENABLE_PROFILING
          args->channel->active_req --;
          if (args->channel->active_req == 0) {
            struct timeval tv;
            gettimeofday(&tv, NULL);
            args->channel->bw_cumulative += (float)args->channel->sizes/((tv.tv_sec - args->channel->tvs.tv_sec)*1000*1000 + tv.tv_usec - args->channel->tvs.tv_usec)/1000.0;
            args->channel->bw_count ++;
          }
#endif
          args->head += args->sliceSteps;
          STORE(&resources->sendMem->head, args->head);
          args->idle = 0;
        }
      }
    }
    if (args->head == args->end) {
      resources->step = args->end;
      args->idle = 0;
      args->state = ncclProxyOpNone;
    }
  }
  return ncclSuccess;
}

ncclResult_t netRecvProxy(struct ncclProxyArgs* args) {
  struct netRecvResources* resources = (struct netRecvResources*) (args->connector->transportResources);
  if (args->state == ncclProxyOpReady) {
    // Update opCount
    STORE(&resources->sendMem->opCount, args->opCount);

    // Round to next multiple of sliceSteps
    resources->step = ROUNDUP(resources->step, args->chunkSteps);
    args->head = resources->step;
    args->tail = resources->step;
    args->end = args->head + args->nsteps;
    args->state = ncclProxyOpProgress;
  }
  if (args->state == ncclProxyOpProgress) {
    args->idle = 1;
    int p = args->protocol;
    int stepSize = args->connector->comm->buffSizes[p] / NCCL_STEPS;
    char* localBuff = args->connector->conn.buffs[p];
    void* mhandle = *(resources->mhandlesProto[p]);
    if (args->head < args->end) {
      volatile uint64_t* sendHead = &resources->sendMem->head;
      if ((args->tail < args->head + NCCL_STEPS) && (args->tail < LOAD(sendHead) + NCCL_STEPS) && (args->tail < args->end)) {
        int buffSlot = args->tail%NCCL_STEPS;
        int sliceSize = stepSize * args->sliceSteps;
        NCCLCHECK(ncclNetIrecv(resources->netRecvComm, localBuff+buffSlot*stepSize, sliceSize, mhandle, args->requests+buffSlot));
        if (args->requests[buffSlot] != NULL) {
#ifdef ENABLE_PROFILING
          if (args->channel->active_req == 0) {
            gettimeofday(&args->channel->tvs, NULL);
            args->channel->sizes = 0;
          }
          args->channel->active_req ++;
#endif
          args->tail += args->sliceSteps;
          args->idle = 0;
        }
      }
      if (args->tail > args->head) {
        int buffSlot = args->head%NCCL_STEPS;
        int done, size;
        NCCLCHECK(ncclNetTest(args->requests[buffSlot], &done, &size));
        if (done) {
          args->head += args->sliceSteps;
          if (args->protocol == NCCL_PROTO_SIMPLE) {
#ifdef ENABLE_PROFILING
          args->channel->active_req --;
          args->channel->sizes += size;
          args->channel->recv_byte += size;
          if (args->channel->active_req == 0) {
            struct timeval tv;
            gettimeofday(&tv, NULL);
            args->channel->bw_cumulative += (float)args->channel->sizes/((tv.tv_sec - args->channel->tvs.tv_sec)*1000*1000 + tv.tv_usec - args->channel->tvs.tv_usec)/1000.0;
            args->channel->bw_count ++;
          }
#endif
            if (resources->useGdr) NCCLCHECK(ncclNetFlush(resources->netRecvComm, localBuff+buffSlot*stepSize, size, mhandle));
            STORE(&resources->recvMem->tail, args->head);
          }
          args->idle = 0;
        }
      }
    }
    if (args->head == args->end) {
      resources->step = args->end;
      args->idle = 0;
      args->state = ncclProxyOpNone;
    }
  }
  return ncclSuccess;
}

struct ncclTransport netTransport = {
  "NET",
  netCanConnect,
  { netSendSetup, netSendConnect, netSendFree, netSendProxy },
  { netRecvSetup, netRecvConnect, netRecvFree, netRecvProxy }
};
