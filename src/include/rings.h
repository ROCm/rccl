/*************************************************************************
 * Copyright (c) 2015-2018, NVIDIA CORPORATION. All rights reserved.
 * Modifications Copyright (c) 2019 Advanced Micro Devices, Inc. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

#ifndef NCCL_RINGS_H_
#define NCCL_RINGS_H_

static int getDefaultThreads() {
  // On Kepler, rings are doubled later.
#if defined(__HIP_PLATFORM_HCC__) || defined(__HCC__)
  return 256;
#else
  return ncclCudaCompCap() == 3 ? 128 : 256;
#endif
}

ncclResult_t ncclGetRings(int* nrings, int* nthreads, int rank, int nranks, int* transports, ncclTvalue_t* values, int* prev, int* next);

#endif
