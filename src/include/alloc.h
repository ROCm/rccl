/*************************************************************************
 * Copyright (c) 2019, NVIDIA CORPORATION. All rights reserved.
 * Modifications Copyright (c) 2019-2020 Advanced Micro Devices, Inc. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

#ifndef NCCL_ALLOC_H_
#define NCCL_ALLOC_H_

#include "nccl.h"
#include "checks.h"
#include "align.h"
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

static bool hasFineGrainVramPcie() {
  int *ptr;
  if (hipExtMallocWithFlags((void**)&ptr, sizeof(int), hipDeviceMallocFinegrained) == hipSuccess) {
    CUDACHECK(hipFree(ptr));
    return true;
  }
  else
    return false;
}

// Allocate memory to be potentially ibv_reg_mr'd. This needs to be
// allocated on separate pages as those pages will be marked DONTFORK
// and if they are shared, that could cause a crash in a child process
static ncclResult_t ncclIbMalloc(void** ptr, size_t size) {
  size_t page_size = sysconf(_SC_PAGESIZE);
  void* p;
  int size_aligned = ROUNDUP(size, page_size);
  int ret = posix_memalign(&p, page_size, size_aligned);
  if (ret != 0) return ncclSystemError;
  memset(p, 0, size);
  *ptr = p;
  return ncclSuccess;
}

#endif
