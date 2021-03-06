/*************************************************************************
 * Copyright (c) 2015-2020, NVIDIA CORPORATION. All rights reserved.
 * Modifications Copyright (c) 2019-2020 Advanced Micro Devices, Inc. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

#include "channel.h"
#include "param.h"

ncclResult_t initChannel(struct ncclComm* comm, int channelid) {
  struct ncclChannel* channel = comm->channels+channelid;
  if (channel->id != -1) return ncclSuccess;
  channel->id = channelid;

  // Ring index to user rank table.
  NCCLCHECK(ncclCudaCalloc(&channel->ring.devUserRanks, comm->nRanks));
  NCCLCHECK(ncclCalloc(&channel->ring.userRanks, comm->nRanks));

  // Communication structures with peers.
  NCCLCHECK(ncclCudaCalloc(&channel->devPeers, comm->nRanks+1)); // The extra one rank is for collnet root (i.e. network)
  NCCLCHECK(ncclCalloc(&channel->peers, comm->nRanks+1));
  for (size_t i=0; i<comm->nRanks+1; ++i) {
    channel->peers[i].send.comm = comm;
    channel->peers[i].recv.comm = comm;
    channel->peers[i].p2pSend.comm = comm;
    channel->peers[i].p2pRecv.comm = comm;
  }

  // Per-channel operation list.
  NCCLCHECK(ncclCudaHostCalloc(&channel->workFifo, NCCL_MAX_OPS));
  return ncclSuccess;
}

ncclResult_t freeChannel(struct ncclChannel* channel, int nRanks) {
  if (channel->id == -1) return ncclSuccess;
  // Operation list
  NCCLCHECK(ncclCudaHostFree(channel->workFifo));

  // Free Ring index to rank tables
  free(channel->ring.userRanks);
  CUDACHECK(hipFree(channel->ring.devUserRanks));

  // Free transport proxy resources
  // Note: free all send resources first due to CollNet arrangement
  for (int r=0; r<nRanks+1; r++) {
    struct ncclPeer* peer = channel->peers+r;
    if (peer->send.transportResources) NCCLCHECK(peer->send.transportComm->free(peer->send.transportResources));
    if (peer->send.transportResources == peer->p2pSend.transportResources) peer->p2pSend.transportResources = NULL;
    peer->send.transportResources = NULL;
    if (peer->p2pSend.transportResources) NCCLCHECK(peer->p2pSend.transportComm->free(peer->p2pSend.transportResources));
  }
  for (int r=0; r<nRanks+1; r++) {
    struct ncclPeer* peer = channel->peers+r;
    if (peer->recv.transportResources) NCCLCHECK(peer->recv.transportComm->free(peer->recv.transportResources));
    if (peer->recv.transportResources == peer->p2pRecv.transportResources) peer->p2pRecv.transportResources = NULL;
    peer->recv.transportResources = NULL;
    if (peer->p2pRecv.transportResources) NCCLCHECK(peer->p2pRecv.transportComm->free(peer->p2pRecv.transportResources));
  }

  // Free the peer structures.
  CUDACHECK(hipFree(channel->devPeers));
  free(channel->peers);

  return ncclSuccess;
}
