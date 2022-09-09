/*************************************************************************
 * Copyright (c) 2019-2022, NVIDIA CORPORATION. All rights reserved.
 * Modifications Copyright (c) 2019-2022 Advanced Micro Devices, Inc. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

#ifndef NCCL_ALLOC_H_
#define NCCL_ALLOC_H_

#include "nccl.h"
#include "checks.h"
#include "align.h"
#include <sys/mman.h>
#include <unistd.h>
#include <stdlib.h>
#include <string.h>
#include "rccl_vars.h"

template <typename T>
static ncclResult_t ncclCudaHostCallocDebug(T** ptr, size_t nelem, const char *filefunc, int line) {
  CUDACHECK(hipHostMalloc(ptr, nelem*sizeof(T), hipHostMallocMapped));
  memset(*ptr, 0, nelem*sizeof(T));
  INFO(NCCL_ALLOC, "%s:%d Cuda Host Alloc Size %ld pointer %p", filefunc, line, nelem*sizeof(T), *ptr);
  return ncclSuccess;
}
#define ncclCudaHostCalloc(...) ncclCudaHostCallocDebug(__VA_ARGS__, __FILE__, __LINE__)

static inline ncclResult_t ncclCudaHostFree(void* ptr) {
  CUDACHECK(hipHostFree(ptr));
  return ncclSuccess;
}

template <typename T>
static ncclResult_t ncclCallocDebug(T** ptr, size_t nelem, const char *filefunc, int line) {
  void* p = malloc(nelem*sizeof(T));
  if (p == NULL) {
    WARN("Failed to malloc %ld bytes", nelem*sizeof(T));
    return ncclSystemError;
  }
  //INFO(NCCL_ALLOC, "%s:%d malloc Size %ld pointer %p", filefunc, line, nelem*sizeof(T), p);
  memset(p, 0, nelem*sizeof(T));
  *ptr = (T*)p;
  return ncclSuccess;
}
#define ncclCalloc(...) ncclCallocDebug(__VA_ARGS__, __FILE__, __LINE__)

template <typename T>
static ncclResult_t ncclRealloc(T** ptr, size_t oldNelem, size_t nelem) {
  if (nelem < oldNelem) return ncclInternalError;
  if (nelem == oldNelem) return ncclSuccess;

  T* oldp = *ptr;
  T* p = (T*)malloc(nelem*sizeof(T));
  if (p == NULL) {
    WARN("Failed to malloc %ld bytes", nelem*sizeof(T));
    return ncclSystemError;
  }
  memcpy(p, oldp, oldNelem*sizeof(T));
  free(oldp);
  memset(p+oldNelem, 0, (nelem-oldNelem)*sizeof(T));
  *ptr = (T*)p;
  INFO(NCCL_ALLOC, "Mem Realloc old size %ld, new size %ld pointer %p", oldNelem*sizeof(T), nelem*sizeof(T), *ptr);
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
static ncclResult_t ncclCudaCallocDebug(const char *filefunc, int line, T** ptr, size_t nelem, hipStream_t sideStream = nullptr, bool isFineGrain = false) {

  if (isFineGrain)
    CUDACHECK(hipExtMallocWithFlags((void**)ptr, nelem*sizeof(T), hipDeviceMallocFinegrained));
  else
    CUDACHECK(hipMalloc(ptr, nelem*sizeof(T)));

  if (rcclParamEnableHipGraph()) {
    hipStream_t stream = sideStream;
    if (stream == nullptr)
      CUDACHECK(hipStreamCreateWithFlags(&stream, hipStreamNonBlocking));

    CUDACHECK(hipMemsetAsync(*ptr, 0, nelem*sizeof(T), stream));
    CUDACHECK(hipStreamSynchronize(stream));
    if (sideStream == nullptr)
      CUDACHECK(hipStreamDestroy(stream));
  } else {
    CUDACHECK(hipMemset(*ptr, 0, nelem*sizeof(T)));
    CUDACHECK(hipStreamSynchronize(NULL));
  }

  INFO(NCCL_ALLOC, "%s:%d Cuda Alloc Size %ld pointer %p", filefunc, line, nelem*sizeof(T), *ptr);
  int dev;
  CUDACHECK(hipGetDevice(&dev));
  if (dev < MAX_ALLOC_TRACK_NGPU) {
    __atomic_fetch_add(&allocTracker[dev].totalAlloc, 1, __ATOMIC_SEQ_CST);
    __atomic_fetch_add(&allocTracker[dev].totalAllocSize, nelem*sizeof(T), __ATOMIC_SEQ_CST);
  }
  return ncclSuccess;
}
#define ncclCudaCalloc(...) ncclCudaCallocDebug(__FILE__, __LINE__, __VA_ARGS__)

template <typename T>
static ncclResult_t ncclCudaMemcpy(T* dst, T* src, size_t nelem) {
  CUDACHECK(hipMemcpy(dst, src, nelem*sizeof(T), hipMemcpyDefault));
  return ncclSuccess;
}

// Allocate memory to be potentially ibv_reg_mr'd. This needs to be
// allocated on separate pages as those pages will be marked DONTFORK
// and if they are shared, that could cause a crash in a child process
static ncclResult_t ncclIbMallocDebug(void** ptr, size_t size, const char *filefunc, int line) {
  size_t page_size = sysconf(_SC_PAGESIZE);
  void* p;
  int size_aligned = ROUNDUP(size, page_size);
  int ret = posix_memalign(&p, page_size, size_aligned);
  if (ret != 0) return ncclSystemError;
  memset(p, 0, size);
  *ptr = p;
  INFO(NCCL_ALLOC, "%s:%d Ib Alloc Size %ld pointer %p", filefunc, line, size, *ptr);
  return ncclSuccess;
}
#define ncclIbMalloc(...) ncclIbMallocDebug(__VA_ARGS__, __FILE__, __LINE__)

#endif
