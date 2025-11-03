/*************************************************************************
 * Copyright (c) 2015-2021, NVIDIA CORPORATION. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

#include "device.h"
#include "collectives.h"
#include "primitives.h"

#ifdef ENABLE_ROCSHMEM
#include <rocshmem/rocshmem.hpp>
#endif

template<typename T, typename RedOp>
struct RunWorkColl<ncclFuncAllToAllGda, T, RedOp, NCCL_ALGO_RING, NCCL_PROTO_SIMPLE> {
  __device__ __forceinline__ void run(int tid, int nThreads, struct ncclDevWorkColl* work) {
    using Proto = ProtoSimple<ALLTOALL_PIVOT_CHUNKSTEPS/ALLTOALL_PIVOT_SLICESTEPS, ALLTOALL_PIVOT_SLICESTEPS>;
	if (blockIdx.x == 0) {
           __shared__ rocshmem::rocshmem_ctx_t ctx;
           int64_t ctx_type = 0;

           rocshmem::rocshmem_wg_ctx_create(ctx_type, &ctx);
           int num_pes = rocshmem::rocshmem_ctx_n_pes(ctx);

	   reduceCopy<COLL_UNROLL, USE_ACC, RedOp, T, 0,1, 1, 0, 1, 1, 0>(
              tid, nThreads, 0, nullptr, false, 1, (void **)&work->sendbuff, 1, (void **)&work->sndbuff, 
	      (work->size*num_pes));

           rocshmem_ctx_char_alltoall_wg(ctx, work->team, ((char*)work->tempbuff), ((char*)work->sndbuff), work->size);

           //rocshmem_ctx_quiet(ctx);
           //__syncthreads();

           rocshmem_wg_ctx_destroy(&ctx);

           reduceCopy<COLL_UNROLL, USE_ACC, RedOp, T, 0,1, 1, 0, 1, 1, 0>(
              tid, nThreads, 0, nullptr, false, 1, (void **)&work->tempbuff, 1, (void **)&work->recvbuff, 
	      (work->size*num_pes));
        }
  }
};
