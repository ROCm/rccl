/*************************************************************************
 * Copyright (c) 2015-2021, NVIDIA CORPORATION. All rights reserved.
 * Modifications Copyright (c) 2019-2021 Advanced Micro Devices, Inc. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

#include "devcomm.h"
#include "primitives.h"
#include "collectives.h"

template<class FUNC, typename T, int UNROLL>
class ncclFunction<ncclFuncSendRecv, NCCL_ALGO_RING, NCCL_PROTO_SIMPLE, FUNC, T, UNROLL> {
  public:
    __device__ __attribute__((noinline)) void run(struct ncclWorkElem* firstArgs) {
      struct ncclWorkElem* args = firstArgs;
      int tid = threadIdx.x;
      int group = 0;
      for (int s=0; s<NCCL_MAX_WORK_ELEMENTS; s++) {
        int nThreadsSegment = args->p2p.nThreads;
        if (nThreadsSegment == 0) return; // Nothing else to do
        int groupRecv = group;
        group += 1;
        int groupSend = group;
        group += 1;
        if (tid < nThreadsSegment) {
          const int nThreads = nThreadsSegment;

          // Compute pointers
          const T* sendbuff = (const T*)args->sendbuff;
          T* recvbuff = (T*)args->recvbuff;
          const ssize_t sendCount = args->p2p.sendCount;
          const ssize_t recvCount = args->p2p.recvCount;

          const int delta = args->p2p.delta;
          if (delta == 0) {
            if (tid < nThreads && sendbuff != recvbuff) {
              // local copy : ReduceOrCopyMulti takes an int as number of elements,
              // so we split it in blocks of 1G elements.
              int blockSize = 1<<30;
              for (size_t offset=0; offset<sendCount; offset += blockSize) {
                size_t remaining = sendCount - offset;
                if (remaining < blockSize) blockSize = remaining;
                ReduceOrCopyMulti<UNROLL, FUNC, T, 1, 1, 1, 1>(tid, nThreads, 1, &sendbuff, 1, &recvbuff, blockSize);
                sendbuff += blockSize; recvbuff += blockSize;
              }
            }
          } else {
            struct ncclDevComm* comm = args->comm;
            struct ncclChannel* channel = comm->channels+blockIdx.x;

            const int stepSize = comm->buffSizes[NCCL_PROTO_SIMPLE]/(sizeof(T)*NCCL_STEPS);

            int nThreadsSplit = nThreads/2;
            if ((tid < nThreadsSplit) && recvCount >= 0) {
	      const int chunkSize = args->p2p.recvChunkSize/sizeof(T);
              int peer = (comm->rank-delta+comm->nRanks)%comm->nRanks;
              int nt = nThreadsSplit;
              ncclPrimitives<UNROLL, 1, 1, T, 1, 0, 1, FUNC>
                prims(tid, nt, &peer, NULL, recvbuff, stepSize, channel, comm, ncclShmem->ptrs, PACK_GROUP(groupRecv, args->p2p.recvIdx));

              if (recvCount == 0) {
                prims.recv(recvbuff, 0);
              } else for (ssize_t offset = 0; offset < recvCount; offset += chunkSize) {
                int realChunkSize = min(chunkSize, recvCount-offset);
                ALIGN_SIZE(realChunkSize, nt*sizeof(uint64_t)/sizeof(T));
                int nelem = min(realChunkSize, recvCount-offset);
                prims.directRecv(recvbuff+offset, offset, nelem);
              }
            }
            if ((tid >= nThreadsSplit) && sendCount >= 0) {
	      const int chunkSize = args->p2p.sendChunkSize/sizeof(T);
              int peer = (comm->rank+delta)%comm->nRanks;
              int nt = nThreads-nThreadsSplit;
              ncclPrimitives<UNROLL, 1, 1, T, 0, 1, 1, FUNC>
                prims(tid-nThreadsSplit, nt, NULL, &peer, recvbuff, stepSize, channel, comm, ncclShmem->ptrs, PACK_GROUP(groupSend, args->p2p.sendIdx));

              if (sendCount == 0) {
                prims.send(sendbuff, 0);
              } else for (ssize_t offset = 0; offset < sendCount; offset += chunkSize) {
                int realChunkSize = min(chunkSize, sendCount-offset);
                ALIGN_SIZE(realChunkSize, nt*sizeof(uint64_t)/sizeof(T));
                int nelem = min(realChunkSize, sendCount-offset);
                prims.directSend(sendbuff+offset, offset, nelem);
              }
            }
          }
        }
        tid -= nThreadsSegment;
        if (tid < 0) return;
        args++;
      }
    }
};
