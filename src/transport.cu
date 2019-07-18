/*************************************************************************
 * Copyright (c) 2016-2018, NVIDIA CORPORATION. All rights reserved.
 * Modifications Copyright (c) 2019 Advanced Micro Devices, Inc. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

#include "core.h"
#include "common_coll.h"

extern struct ncclTransport p2pTransport;
extern struct ncclTransport shmTransport;
extern struct ncclTransport netTransport;

struct ncclTransport ncclTransports[NTRANSPORTS] = {
  p2pTransport,
  shmTransport,
  netTransport,
};

static void FifoPullArgs(struct transportProxyInfo* info, struct ncclProxyArgs *args) {
  struct ncclProxyArgs *fifoArgs = info->argsFifo + (LOAD(&info->argsFifoHead) % TRANSPORT_PROXY_FIFO_SIZE);
  pthread_mutex_lock(&info->mutex);
  while (LOAD(&fifoArgs->active) == 0)
    pthread_cond_wait(&info->cond, &info->mutex);
  __sync_synchronize();
  memcpy(args, fifoArgs, sizeof(struct ncclProxyArgs));
  __sync_synchronize();
  STORE(&fifoArgs->active, 0);
  pthread_cond_signal(&info->cond);
  pthread_mutex_unlock(&info->mutex);
  __atomic_fetch_add(&info->argsFifoHead, 1, __ATOMIC_SEQ_CST);
}

static struct ncclProxyArgs* FifoGetNextArgs(struct transportProxyInfo* info) {
  if (info == NULL) return NULL;
  struct ncclProxyArgs* fifoArgs = info->argsFifo + (LOAD(&info->argsFifoTail) % TRANSPORT_PROXY_FIFO_SIZE);
  pthread_mutex_lock(&info->mutex);
  while (LOAD(&fifoArgs->active) == 1)
    pthread_cond_wait(&info->cond, &info->mutex);
  pthread_mutex_unlock(&info->mutex);
  __atomic_fetch_add(&info->argsFifoTail, 1, __ATOMIC_SEQ_CST);
  return fifoArgs;
}

static void FifoPushArgs(struct transportProxyInfo* info) {
  if (info == NULL) return;

  struct ncclProxyArgs* fifoArgs = info->argsFifo + ((LOAD(&info->argsFifoTail)-1) % TRANSPORT_PROXY_FIFO_SIZE);
  if (LOAD(&fifoArgs->active) == 0) return;

  pthread_mutex_lock(&info->mutex);
  pthread_cond_signal(&info->cond);
  pthread_mutex_unlock(&info->mutex);
}

static void WaitProxyReady(struct transportProxyInfo* info) {
  pthread_mutex_lock(&info->mutex);
  while (LOAD(&info->proxyReady) == 0)
    pthread_cond_wait(&info->cond, &info->mutex);
  pthread_mutex_unlock(&info->mutex);
}

static void SetProxyReady(struct transportProxyInfo* info) {
  pthread_mutex_lock(&info->mutex);
  STORE(&info->proxyReady, 1);
  pthread_cond_signal(&info->cond);
  pthread_mutex_unlock(&info->mutex);
}

static void StopProxy(struct transportProxyInfo* info) {
  struct ncclProxyArgs* fifoArgs = FifoGetNextArgs(info);
  STORE(&fifoArgs->active, -1);
  FifoPushArgs(info);
}

#define RECV 0
#define SEND 1

static bool NeedProxy(int type, int pattern, struct ncclRing* ring, int nranks) {
  enum proxyMode mode = proxyPatternMode(pattern);
  if (mode == proxyRing) return true;

  /* In chains, one rank does not need a proxy. Let's figure out which one it is */
  int root = proxyPatternRoot(pattern);
  // Which index in the reorganized rings should we compare root against */
  const int myrank = 0, nextrank = 1, prevrank = nranks-1;
  int index = mode == proxyFrom ?
      /*                            no recv /  no send    if root = */
      /* bcast  */ (type == RECV ?   myrank : nextrank ):
      /* reduce */ (type == RECV ? prevrank :   myrank );
  int rank = ring->userRanks[index];
  return (root != rank);
}

static void SaveProxy(struct ncclConnector* connector, struct ncclProxyArgs* args, int needProxy) {
  struct transportProxyInfo* info = connector->proxyInfo;
  if (info == NULL) return;
  struct ncclProxyArgs* fifoArgs = FifoGetNextArgs(info);
  args->needProxy = needProxy;
  __sync_synchronize();
  memcpy(fifoArgs, args, sizeof(struct ncclProxyArgs));
  __sync_synchronize();
  STORE(&fifoArgs->active, 1);
}

ncclResult_t transportSaveProxies(int substeps, int subchunks, int nstepsPerRound, int nblocksPerRound, size_t nbytes, int pattern, struct ncclComm* comm) {
  int llMode, nrings, nthreads;
  ncclGetCollResource(comm, nbytes, &nrings, &nthreads, &llMode);
  nbytes       = llMode ? nbytes * 2    : nbytes;
  substeps     = llMode ? 1             : substeps;
  subchunks    = llMode ? NCCL_LL_CHUNKS : subchunks;
  int buffSize = llMode ? NCCL_LL_BUFF_SIZE : comm->rings[0].buffSize;

  int nrounds = (int)(DIVUP(nbytes, ((size_t)nrings * nblocksPerRound * (buffSize/subchunks)))); // Fixed 32-bit overflow
  int nsteps = nstepsPerRound * nrounds * substeps;
  TRACE(NCCL_NET,"opCount %lx substeps %d subchunks %d nrounds %d nsteps %d comm %p", comm->opCount, subchunks, subchunks, nrounds, nsteps, comm);
  TRACE(NCCL_NET,"opCount %lx nbytes %zi nrings %d buffSize %d pattern %d comm %p", comm->opCount, nbytes, nrings, buffSize, pattern, comm);
  for (int r=0; r<nrings; r++) {
    struct ncclRing* ring = comm->rings+((comm->myParams->gridDim.x+r)%comm->nRings);
    struct ncclProxyArgs args = { ring, substeps*subchunks, nsteps, comm->opCount, llMode, 0 };
    SaveProxy(&ring->recv, &args, NeedProxy(RECV, pattern, ring, comm->nRanks));
    SaveProxy(&ring->send, &args, NeedProxy(SEND, pattern, ring, comm->nRanks));
  }
  return ncclSuccess;
}

ncclResult_t transportStartProxies(ncclComm* comm) {
  for (int r=0; r<comm->nRings; r++) {
    FifoPushArgs(comm->rings[r].send.proxyInfo);
    FifoPushArgs(comm->rings[r].recv.proxyInfo);
  }
  pthread_yield(); // Let other threads run
  return ncclSuccess;
}

void* persistentThread(void *opaqueInfo) {
  struct transportProxyInfo* info = (struct transportProxyInfo*)opaqueInfo;
  // We need to initialize the context before launching any NCCL cuda kernel,
  // otherwise we would create it during the first hipMemcpyAsync inside the
  // proxy function and that would cause a deadlock
  hipSetDevice(info->comm->cudaDev);
  // Signal the main thread the context is created and it can proceed.
  SetProxyReady(info);
  while (1) {
    struct ncclProxyArgs args;
    FifoPullArgs(info, &args);
    if (args.active == -1) {
      // Main thread asked to stop
      return NULL;
    }
    ncclResult_t res = info->func(&args);
    if (res != ncclSuccess) {
      WARN("%s:%d -> %d [Proxy thread error]", __FILE__, __LINE__, res);
    }
  }
}

ncclResult_t transportCreateProxy(int type, struct ncclRing* ring, struct ncclComm* comm) {
  struct ncclConnector* connector = (type == RECV) ? &ring->recv : &ring->send;
  threadFunc_t proxyfunc = (threadFunc_t) ((type == RECV) ? connector->transport->recv.proxy : connector->transport->send.proxy);
  if (proxyfunc) {
    TRACE(NCCL_NET,"type %d ring %p proxyfunc %p comm %p", type, ring, proxyfunc, comm);
    struct transportProxyInfo* info;
    NCCLCHECK(ncclCalloc(&info, 1));
    connector->proxyInfo = info;
    info->comm = comm;
    info->cond = PTHREAD_COND_INITIALIZER;
    info->mutex = PTHREAD_MUTEX_INITIALIZER;
    info->func = proxyfunc;
    STORE(&info->argsFifoHead, 0); STORE(&info->argsFifoTail, 0);
    STORE(&info->proxyReady, 0);
    pthread_create(&connector->proxyInfo->thread, NULL, persistentThread, info);
    // Wait for thread to initialize its CUDA context.
    WaitProxyReady(info);
  }
  return ncclSuccess;
}

ncclResult_t transportDestroyProxy(struct ncclConnector* connector) {
  if (connector->proxyInfo) {
    StopProxy(connector->proxyInfo);
    pthread_join(connector->proxyInfo->thread, NULL);
    free(connector->proxyInfo);
    connector->proxyInfo = NULL;
  }
  return ncclSuccess;
}


ncclResult_t getGpuHdpReg(int cudaDev, uint32_t** hdp) {
  auto convert_bdf = [](const char *busId) {
    char bdf[9];
    strncpy(bdf, busId, 4);
    strncpy(bdf+4, busId+5, 2);
    strncpy(bdf+6, busId+8, 2);
    bdf[8] = '\0';
    uint16_t id = (uint16_t)strtol(bdf, NULL, 16);
    return id;
  };

  union find_agent_args {
    hsa_agent_t agent;
    uint16_t id;
  } args;

  const auto& find_agent = [](hsa_agent_t agent, void* arg) {
    uint16_t id = ((union find_agent_args *)arg)->id;
    hsa_device_type_t type;
    hsa_agent_get_info(agent, HSA_AGENT_INFO_DEVICE, (void*)&type);
    if(type == HSA_DEVICE_TYPE_GPU) {
      uint16_t bdf_id = 1;
      hsa_agent_get_info(agent, (hsa_agent_info_t)HSA_AMD_AGENT_INFO_BDFID, &bdf_id);
      if(bdf_id == id) {
        ((union find_agent_args *)arg)->agent=agent;
        return HSA_STATUS_INFO_BREAK;
      }
    }
    return HSA_STATUS_SUCCESS;
  };

#define PCI_BUS_ID_BUFFER_SIZE 16
  char busId[PCI_BUS_ID_BUFFER_SIZE];
  *hdp = NULL;
  CUDACHECK(hipDeviceGetPCIBusId(busId, PCI_BUS_ID_BUFFER_SIZE, cudaDev));
  args.id = convert_bdf(busId);
  hsa_status_t err = hsa_iterate_agents(find_agent, (void*)&args);
  if (err != HSA_STATUS_INFO_BREAK) {
    WARN("failed to get locate HSA agent for GPU %d", cudaDev);
    return ncclSystemError;
  }
  hsa_amd_hdp_flush_t hdpinfo;
  err = hsa_agent_get_info(args.agent, (hsa_agent_info_t)HSA_AMD_AGENT_INFO_HDP_FLUSH, &hdpinfo);
  if ((err != HSA_STATUS_SUCCESS) && (err != HSA_STATUS_INFO_BREAK)) {
    WARN("failed to get HSA_AMD_AGENT_INFO_HDP_FLUSH for GPU %d", cudaDev);
    return ncclSystemError;
  }
  *hdp = hdpinfo.HDP_MEM_FLUSH_CNTL;
  return ncclSuccess;
#undef PCI_BUS_ID_BUFFER_SIZE
}
