/*************************************************************************
 * Copyright (c) 2019, NVIDIA CORPORATION. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

#ifndef NCCL_ALLOC_H_
#define NCCL_ALLOC_H_

#include "nccl.h"
#include "checks.h"
#include <sys/mman.h>

static inline ncclResult_t ncclCudaHostAlloc(void** ptr, void** devPtr, size_t size) {
  CUDACHECK(hipHostMalloc(ptr, size, hipHostMallocMapped));
  memset(*ptr, 0, size);
  *devPtr = *ptr;
  return ncclSuccess;
}

static inline ncclResult_t ncclCudaHostFree(void* ptr) {
  CUDACHECK(hipHostFree(ptr));
  return ncclSuccess;
}

template <typename T>
static ncclResult_t ncclCalloc(T** ptr, size_t nelem) {
  void* p = malloc(nelem*sizeof(T));
  if (p == NULL) {
    WARN("Failed to malloc %ld bytes", nelem*sizeof(T));
    return ncclSystemError;
  }
  memset(p, 0, nelem*sizeof(T));
  *ptr = (T*)p;
  return ncclSuccess;
}

template <typename T>
static ncclResult_t ncclCudaCalloc(T** ptr, size_t nelem, bool isFineGrain = false) {
  if (isFineGrain)
    CUDACHECK(hipExtMallocWithFlags((void**)ptr, nelem*sizeof(T), hipDeviceMallocFinegrained));
  else
    CUDACHECK(hipMalloc(ptr, nelem*sizeof(T)));
  CUDACHECK(hipMemset(*ptr, 0, nelem*sizeof(T)));
  return ncclSuccess;
}

template <typename T>
static ncclResult_t ncclCudaMemcpy(T* dst, T* src, size_t nelem) {
  CUDACHECK(hipMemcpy(dst, src, nelem*sizeof(T), hipMemcpyDefault));
  return ncclSuccess;
}

#endif
