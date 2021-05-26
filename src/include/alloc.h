/*************************************************************************
 * Copyright (c) 2019-2021, NVIDIA CORPORATION. All rights reserved.
 * Modifications Copyright (c) 2019-2021 Advanced Micro Devices, Inc. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

#ifndef NCCL_ALLOC_H_
#define NCCL_ALLOC_H_

#include "nccl.h"
#include "checks.h"
#include "align.h"
#include <sys/mman.h>

template <typename T>
static ncclResult_t ncclCudaHostCalloc(T** ptr, size_t nelem) {
  CUDACHECK(hipHostMalloc(ptr, nelem*sizeof(T), hipHostMallocMapped));
  memset(*ptr, 0, nelem*sizeof(T));
  INFO(NCCL_ALLOC, "Cuda Host Alloc Size %ld pointer %p", nelem*sizeof(T), *ptr);
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
  INFO(NCCL_ALLOC, "Mem Alloc Size %ld pointer %p", nelem*sizeof(T), *ptr);
  return ncclSuccess;
}

struct __attribute__ ((aligned(64))) allocationTracker {
  union {
    struct {
      uint64_t totalAlloc;
      uint64_t totalAllocSize;
    };
    char align[64];
  };
};
static_assert(sizeof(struct allocationTracker) == 64, "allocationTracker must be size of 64 bytes");
#define MAX_ALLOC_TRACK_NGPU 32
extern struct allocationTracker allocTracker[];

template <typename T>
static ncclResult_t ncclCudaCalloc(T** ptr, size_t nelem, bool isFineGrain = false) {
  // Need async stream for P2P pre-connect + CUDA Graph
  hipStream_t stream;
  CUDACHECK(hipStreamCreateWithFlags(&stream, hipStreamNonBlocking));
  if (isFineGrain)
    CUDACHECK(hipExtMallocWithFlags((void**)ptr, nelem*sizeof(T), hipDeviceMallocFinegrained));
  else
    CUDACHECK(hipMalloc(ptr, nelem*sizeof(T)));
  CUDACHECK(hipMemsetAsync(*ptr, 0, nelem*sizeof(T), stream));
  CUDACHECK(hipStreamSynchronize(stream));
  CUDACHECK(hipStreamDestroy(stream));
  INFO(NCCL_ALLOC, "Cuda Alloc Size %ld pointer %p", nelem*sizeof(T), *ptr);
  int dev;
  CUDACHECK(hipGetDevice(&dev));
  if (dev < MAX_ALLOC_TRACK_NGPU) {
    __atomic_fetch_add(&allocTracker[dev].totalAlloc, 1, __ATOMIC_SEQ_CST);
    __atomic_fetch_add(&allocTracker[dev].totalAllocSize, nelem*sizeof(T), __ATOMIC_SEQ_CST);
  }
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
  INFO(NCCL_ALLOC, "Ib Alloc Size %ld pointer %p", size, *ptr);
  return ncclSuccess;
}

#endif
