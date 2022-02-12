/*************************************************************************
 * Copyright (c) 2015-2021, NVIDIA CORPORATION. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

#include "devcomm.h"
#include "collectives.h"
#include "primitives.h"

namespace {
  template<typename T, typename RedOp, typename Proto>
  __device__ __attribute__((noinline)) void runRing(ncclWorkElem *args) {
    const int tid = threadIdx.x;
    const int nthreads = args->nThreads;
    const int bid = args->coll.bid;
    const int nranks = ncclShmem->comm.nRanks;
    const ncclRing *ring = &ncclShmem->channel.ring;
    // num_channels = (num_clock_cycle_rings + num_anti_clock_cycle_rings) * 4
    // chunk_id = [0, 1, 2, 0, 1, 2, 3, 4, 5, 3, 4, 5, 6, 7, 8, 6, 7, 8, 9, 10, 11, 9, 10, 11]
    const int num_channels = 24;
    const int num_uni_rings = 6;
    const int num_bi_rings = 3;
    const int num_chunks = 12;
    const int chunk_id = (bid % num_bi_rings) + (bid / num_uni_rings * num_bi_rings);
    const int elem_size = args->coll.count % 256 ? 1 : 256;
    const ssize_t num_elems = args->coll.count / elem_size;
    const int num_padding_chunks = num_elems % num_chunks;
    const ssize_t chunk_offset = elem_size * (num_elems / num_chunks * chunk_id + (chunk_id < num_padding_chunks ? chunk_id : num_padding_chunks));
    const ssize_t chunk_size = elem_size * (num_elems / num_chunks + (chunk_id < num_padding_chunks ? 1 : 0));
    const int pivot_half_idx = (bid % num_uni_rings) / num_bi_rings;
    const ssize_t prims_size = int(Proto::calcBytePerStep()/sizeof(T) * (Proto::Id == NCCL_PROTO_SIMPLE ? ALLREDUCE_CHUNKSTEPS : 1));

    Primitives<T, RedOp, FanSymmetric<1>, 0, Proto> prims
      (tid, nthreads, &ring->prev, &ring->next, args->sendbuff, args->recvbuff, /*redOpArg(ignored)=*/0, args->coll.connIndex << 16);

    for (int num_hops = 0; num_hops <= nranks / 2; num_hops++) {
      const int src_rank = ring->devUserRanks[(nranks - num_hops) % nranks];
      const int dst_rank = ring->devUserRanks[num_hops];
      const ssize_t send_offset =
          dst_rank * num_elems * elem_size + chunk_offset +
          (src_rank == dst_rank ? pivot_half_idx * chunk_size / 2 : 0);
      const ssize_t recv_offset =
          src_rank * num_elems * elem_size + chunk_offset +
          (src_rank == dst_rank ? pivot_half_idx * chunk_size / 2 : 0);
      const ssize_t send_recv_size =
          src_rank == dst_rank ?
          (pivot_half_idx == 0 ? chunk_size / 2 : chunk_size - chunk_size / 2) : chunk_size;

      if (num_hops == 0 && args->sendbuff != args->recvbuff) {
        const T* sendbuff = (const T*)args->sendbuff + send_offset;
        T* recvbuff = (T *)args->recvbuff + recv_offset;
        // const T* sendbuff = (const T*)args->sendbuff + bid * 21800 * 1024;
        // T* recvbuff = (T *)args->recvbuff + bid * 21800 * 1024;
        // for (ssize_t i = threadIdx.x; i * sizeof(ulong2) < send_recv_size; i += blockDim.x) {
        // for (ssize_t i = threadIdx.x; i * sizeof(ulong2) < 22369621; i += blockDim.x) {
          // ((ulong2*)recvbuff)[i] = ((ulong2*)sendbuff)[i];
        // }
        ReduceOrCopyMulti<COLL_UNROLL, RedOp, T, 1, 1, 1, 1, 0>(
            tid, nthreads, nullptr, false, 1, &sendbuff, 1, &recvbuff, send_recv_size);
        // if (blockIdx.x == 0 && threadIdx.x == 0) {
          // ((int*)args->recvbuff)[0] = send_offset;
          // ((int*)args->recvbuff)[1] = recv_offset;
          // ((int*)args->recvbuff)[2] = send_recv_size;
        // } else if (blockIdx.x == 1 && threadIdx.x == 0) {
          // ((int*)args->recvbuff)[3] = send_offset;
          // ((int*)args->recvbuff)[4] = recv_offset;
          // ((int*)args->recvbuff)[5] = send_recv_size;
        // }
      } else {
        for (ssize_t prims_offset = 0; prims_offset < send_recv_size; prims_offset += prims_size) {
          const int prims_nelem = min(prims_size, send_recv_size - prims_offset);

          // step 0: send
          prims.send(send_offset + prims_offset, prims_nelem);

          // num_hops - 1 steps: recv and copy to next gpu
          for (int i = 0; i < num_hops - 1; i++) {
            prims.recvSend(prims_nelem);
          }

          // final step: recv
          prims.directRecv(recv_offset + prims_offset, prims_nelem);

          // if (threadIdx.x == 0 && prims_offset < 4 * prims_size) {
          //   ((int*)args->recvbuff)[8 * prims_offset / prims_size] = send_offset + prims_offset;
          //   ((int*)args->recvbuff)[8 * prims_offset / prims_size + 1] = recv_offset + prims_offset;
          //   ((int*)args->recvbuff)[8 * prims_offset / prims_size + 2] = prims_nelem;
          //   ((int*)args->recvbuff)[8 * prims_offset / prims_size + 3] = int(Proto::calcBytePerStep());
          //   ((int*)args->recvbuff)[8 * prims_offset / prims_size + 4] = send_recv_size;
          //   ((int*)args->recvbuff)[8 * prims_offset / prims_size + 5] = ncclShmem->comm.buffSizes[0];
          //   ((int*)args->recvbuff)[8 * prims_offset / prims_size + 6] = ncclShmem->comm.buffSizes[1];
          //   ((int*)args->recvbuff)[8 * prims_offset / prims_size + 7] = ncclShmem->comm.buffSizes[2];
          // }
        }
      }
    }
  }
}

template<typename T, typename RedOp>
struct RunWorkElement<ncclFuncAllToAllPivot, T, RedOp, NCCL_ALGO_RING, NCCL_PROTO_SIMPLE> {
  __device__ __attribute__((noinline)) void run(ncclWorkElem *args) {
    using Proto = ProtoSimple<ALLREDUCE_CHUNKSTEPS/ALLREDUCE_SLICESTEPS, ALLREDUCE_SLICESTEPS>;
    runRing<T, RedOp, Proto>(args);
  }
};

template<typename T, typename RedOp>
struct RunWorkElement<ncclFuncAllToAllPivot, T, RedOp, NCCL_ALGO_RING, NCCL_PROTO_LL> {
  __device__ __attribute__((noinline)) void run(ncclWorkElem *args) {
    runRing<T, RedOp, ProtoLL>(args);
  }
};
