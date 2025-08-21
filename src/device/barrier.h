/*************************************************************************
 * Copyright (c) 2015-2022, NVIDIA CORPORATION. All rights reserved.
 * Modifications Copyright (c) 2019-2022 Advanced Micro Devices, Inc. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

#include "device.h"
#include "collectives.h"
#include "primitives.h"

__device__ __attribute__((noinline)) void  copyShmemData(struct ncclDevComm* comm, struct channelMasks channelMask, struct ncclDevKernelArgs const* args){
  const int tid = threadIdx.x;
  int tn = blockDim.x;
  int x = tid;
  int total = 0, y;
  int num = MAXCHANNELS/64 > 0 ? MAXCHANNELS/64 : 1;

  // Copy kernel args to shmem and then only read those. Otherwise the compiler
  // will end up putting the args into thread local stack which is very wasteful.
  if (tid < sizeof(ncclDevKernelArgs)/sizeof(uint32_t)) {
    ((uint32_t*)&ncclShmem.args)[tid] = ((uint32_t*)args)[tid];
  }

  // To map blockId to channelId, we need the n'th set bit of channelMask which
  // is the inverse of counting the number of set bits among the the first n.
  // PTX has the fns instruction which does this but is extremely slow. We can
  // do better when we know all threads are querying the same bitmask.
  switch (tid/WARP_SIZE) {
  case 0:
  //ncclShmem.channelId = blockIdx.x;
    for (int i = 0; i < num; i++) {
      if (channelMask.masks[i] & (1ull<<x)) {
        y = __popcll(channelMask.masks[i] & ((1ull<<x)-1));
        y = total + y;
        if (blockIdx.x == y) {
          ncclShmem.channelId = x + total;
          break;
        }
      }
      if (WARP_SIZE < 64) {
        x = WARP_SIZE + tid;
        if (channelMask.masks[i] & (1ull<<x)) {
          y = __popcll(channelMask.masks[i] & ((1ull<<x)-1));
          y = y + total;
          if (blockIdx.x == y) {
            ncclShmem.channelId = x + total;
            break;
          }
        }
      }
      total = total + __popcll(channelMask.masks[i]);
    }
    break;
  case 1:
    if (tid < WARP_SIZE + NCCL_MAX_GROUPS) {
      if (tid == WARP_SIZE) ncclShmem.barrier_pat = 0;
      ncclShmem.groups[tid-WARP_SIZE].barrier = 0;
    }
    break;
  case 2:
#ifdef ENABLE_FAULT_INJECTION
    /* load faults injection before first sync threads */
    if (tid == 2*WARP_SIZE) ncclShmem.faults = comm->faults;
#endif
    break;
  case 3:
    /* set abort flag to 0 */
    if (tid == 3*WARP_SIZE) ncclShmem.aborted = 0;
    break;
  default:
    break;
  }

  __syncthreads(); // publish ncclShmem.{args, channelId}

  /* set abort flag to 0 */
  if (tid == 0) {
    ncclShmem.aborted = 0;
    ncclShmem.channel.workCounter = comm->channels[ncclShmem.channelId].workCounter;
  }

  // Use first 2 warps to load comm and channel, and remaining load work batch.
  switch (tid/WARP_SIZE) {
  case 0:
    { void* dst = &ncclShmem.comm;
      void* src = comm;
      int bytes = sizeof(ncclDevComm);
      static_assert(sizeof(ncclDevComm) <= 16*WARP_SIZE, "ncclDevComm cannot be loaded by a single warp in one insn.");
      copyToShmem16(tid, dst, src, bytes);
    } break;
  case 1:
    { // Get address of channel without incurring indirect load from ncclDevComm::channels
      void* dst = &ncclShmem.channel;
      void* src = &((ncclDevCommAndChannels*)comm)->channels[ncclShmem.channelId];
      int bytes = sizeof(ncclDevChannel);
      static_assert(sizeof(ncclDevChannel) <= 16*WARP_SIZE, "ncclDevChannel cannot be loaded by a single warp in one insn.");
      copyToShmem16(tid-WARP_SIZE, dst, src, bytes);
    } break;
  default:
    { int subtid = tid - 2*WARP_SIZE;
      int subtn = tn - 2*WARP_SIZE;
      // Coverity reports a possible thread divergence due to not all threads participating in the collective.
      // However, the code ensures that the participation is on a per-warp basis.
      // coverity[device_thread_diverged:FALSE]
      //loadWorkBatchToShmem(subtid, subtn, args, /*batchIx=*/blockIdx.x);
    } break;
  }
  __syncthreads();
}



namespace {
  template<typename T>
#if defined(USE_INDIRECT_FUNCTION_CALL) && !defined(__gfx942__) && !defined(__gfx950__)
  __device__ void runRing(int tid, int nthreads, struct ncclDevWorkColl* work) {
#else
  __device__ __attribute__((noinline)) void runRing(int tid, int nthreads, struct ncclDevWorkColl* work) {
#endif

ncclRing *ring = &ncclShmem.channel.ring;
const int nranks = ncclShmem.comm.nRanks;
const int rank = ncclShmem.comm.rank;

const int prevRank = ring->userRanks[nranks-1];
const int root = work->root;
const size_t chunkCount = 4096;
const size_t channelCount = 1;
const size_t gridOffset = 0;
size_t offset;
int nelem;
Primitives<uint8_t, FuncSum<uint8_t>, FanSymmetric<1>, 0, ProtoLL, 0>
  prims(tid, nthreads, &ring->prev, &ring->next, work->sendbuff, work->recvbuff, 0, 0, 0, 0);

if (prevRank == root) {
  //printf("1111 prevRank = %d, rank = %d, root = %d channelCount = %zu, chunkCount = %zu\n", prevRank, rank, root, channelCount, chunkCount);
  for (size_t elemOffset = 0; elemOffset < channelCount; elemOffset += chunkCount) {
    offset = gridOffset + elemOffset;
    nelem = min(chunkCount, channelCount - elemOffset);
    prims.send(offset, nelem);
  }
}
else if (rank == root) {
  //printf("2222 prevRank = %d, rank = %d, root = %d channelCount = %zu, chunkCount = %zu\n", prevRank, rank, root, channelCount, chunkCount);
  for (size_t elemOffset = 0; elemOffset < channelCount; elemOffset += chunkCount) {
    offset = gridOffset + elemOffset;
    nelem = min(chunkCount, channelCount - elemOffset);
    prims.recvReduceCopy(offset, offset, nelem, /*postOp=*/true);
  }
}
else {
  for (size_t elemOffset = 0; elemOffset < channelCount; elemOffset += chunkCount) {
    offset = gridOffset + elemOffset;
    nelem = min(chunkCount, channelCount - elemOffset);
    prims.recvReduceSend(offset, nelem);
  }
}
}
}

__global__ __attribute__((noinline)) void rcclWaitForAllRanksBarrier(struct ncclDevComm* comm, struct channelMasks channelMask, struct ncclDevWorkColl* work,  struct ncclDevKernelArgs const* args)
{
  copyShmemData(comm, channelMask, args);
  int tid = threadIdx.x;
  int nthreads = blockDim.x;
  runRing<uint8_t>(tid, nthreads, work);
}
