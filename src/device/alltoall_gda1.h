/*************************************************************************
 * Copyright (c) 2015-2021, NVIDIA CORPORATION. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

#include "device.h"
#include "collectives.h"
#include "primitives.h"
#include <hip/hip_cooperative_groups.h>

#ifdef ENABLE_ROCSHMEM
#include <rocshmem/rocshmem.hpp>

template<typename T, typename RedOp>
struct RunWorkColl<ncclFuncAllToAllGda1, T, RedOp, NCCL_ALGO_RING, NCCL_PROTO_SIMPLE> {
  __device__ __forceinline__ void run(int tid, int nThreads, struct ncclDevWorkColl* work) {
	   using namespace cooperative_groups;
	   grid_group grid = this_grid();
           int num_pes = rocshmem::rocshmem_n_pes();
           int numBlocks = gridDim.x;
                //int numBlocks = 1;

           int sizePerBlock = (work->size*num_pes)/numBlocks;
           void *src = (T*)work->sendbuff + blockIdx.x * sizePerBlock;
           void *dst = (T*)work->sndbuff + blockIdx.x * sizePerBlock;

           reduceCopy<COLL_UNROLL, USE_ACC, RedOp, T, 0,1, 1, 0, 1, 1, 0>(
           tid, nThreads, 0, nullptr, false, 1, (void **)&src, 1, (void **)&dst,
           sizePerBlock);

           grid.sync();
           if (blockIdx.x == 0) {
              rocshmem::rocshmem_char_alltoall_wg(work->team, ((char*)work->tempbuff), ((char*)work->sndbuff), work->size);
           }

           grid.sync();

           void *srcR = (T*)work->tempbuff + blockIdx.x * sizePerBlock;
           void *dstR = (T*)work->recvbuff + blockIdx.x * sizePerBlock;

           reduceCopy<COLL_UNROLL, USE_ACC, RedOp, T, 0,1, 1, 0, 1, 1, 0>(
           tid, nThreads, 0, nullptr, false, 1, (void **)&srcR, 1, (void **)&dstR,
           sizePerBlock);

  }
};
#endif

