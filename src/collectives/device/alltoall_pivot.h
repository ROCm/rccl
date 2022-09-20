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
    const int nthreads = args->nWarps*WARP_SIZE;
    const int bid = args->bid;
    const int nranks = ncclShmem.comm.nRanks;
    const ncclRing *ring = &ncclShmem.channel.ring;
    const int num_bi_rings = args->pivotA2ANumBiRings;
    const int num_uni_rings = num_bi_rings * 2;
    const int num_chunks = args->nChannels / 2;
    const int chunk_id = (bid % num_bi_rings) + (bid / num_uni_rings * num_bi_rings);
    const int elem_size = min(256, args->count & (~(args->count) + 1));
    const ssize_t num_elems = args->count / elem_size;
    const int num_padding_chunks = num_elems % num_chunks;
    const ssize_t chunk_offset = elem_size * (num_elems / num_chunks * chunk_id + (chunk_id < num_padding_chunks ? chunk_id : num_padding_chunks));
    const ssize_t chunk_size = elem_size * (num_elems / num_chunks + (chunk_id < num_padding_chunks ? 1 : 0));
    const int pivot_direction = (bid % num_uni_rings) / num_bi_rings;
    const ssize_t prims_size = int(Proto::calcBytePerStep()/sizeof(T) * (Proto::Id == NCCL_PROTO_SIMPLE ? ALLTOALL_PIVOT_CHUNKSTEPS : 1));

    Primitives<T, RedOp, FanSymmetric<1>, 0, Proto, 0> prims
      (tid, nthreads, &ring->prev, &ring->next, args->sendbuff, args->recvbuff, /*redOpArg(ignored)=*/0);

    for (int num_hops = 0; num_hops <= nranks / 2; num_hops++) {
      const int src_rank = ring->userRanks[(nranks - num_hops) % nranks];
      const int dst_rank = ring->userRanks[num_hops];
      const ssize_t send_offset =
          dst_rank * num_elems * elem_size + chunk_offset +
          (src_rank == dst_rank ? pivot_direction * chunk_size / 2 : 0);
      const ssize_t recv_offset =
          src_rank * num_elems * elem_size + chunk_offset +
          (src_rank == dst_rank ? pivot_direction * chunk_size / 2 : 0);
      const ssize_t send_recv_size =
          src_rank == dst_rank ?
          (pivot_direction == 0 ? chunk_size / 2 : chunk_size - chunk_size / 2) : chunk_size;

      if (num_hops == 0 && args->sendbuff != args->recvbuff) {
        const T* sendbuff = (const T*)args->sendbuff + send_offset;
        T* recvbuff = (T *)args->recvbuff + recv_offset;
        ReduceOrCopyMulti<COLL_UNROLL, RedOp, T, 1, 1, 1, 1, 0>(
            tid, nthreads, nullptr, false, 1, &sendbuff, 1, &recvbuff, send_recv_size);
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
        }
      }
    }
  }
}

template<typename T, typename RedOp>
struct RunWorkElement<ncclFuncAllToAllPivot, T, RedOp, NCCL_ALGO_RING, NCCL_PROTO_SIMPLE> {
  __device__ __forceinline__ void run(ncclWorkElem *args) {
    using Proto = ProtoSimple<ALLTOALL_PIVOT_CHUNKSTEPS/ALLTOALL_PIVOT_SLICESTEPS, ALLTOALL_PIVOT_SLICESTEPS>;
    runRing<T, RedOp, Proto>(args);
  }
};
