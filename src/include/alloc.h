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
  memset(p, 0, nelem*sizeof(T));
  *ptr = (T*)p;
  INFO(NCCL_ALLOC, "%s:%d Mem Alloc Size %ld pointer %p", filefunc, line, nelem*sizeof(T), *ptr);
  return ncclSuccess;
}
#define ncclCalloc(...) ncclCallocDebug(__VA_ARGS__, __FILE__, __LINE__)

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
static ncclResult_t ncclCudaCallocDebug(const char *filefunc, int line, T** ptr, size_t nelem, bool isFineGrain = false) {
#if CUDART_VERSION >= 11030
  // Need async stream for P2P pre-connect + CUDA Graph
  hipStream_t stream;
  CUDACHECK(hipStreamCreateWithFlags(&stream, hipStreamNonBlocking));
#endif
  if (isFineGrain)
    CUDACHECK(hipExtMallocWithFlags((void**)ptr, nelem*sizeof(T), hipDeviceMallocFinegrained));
  else
    CUDACHECK(hipMalloc(ptr, nelem*sizeof(T)));
#if CUDART_VERSION >= 11030
  CUDACHECK(hipMemsetAsync(*ptr, 0, nelem*sizeof(T), stream));
  CUDACHECK(hipStreamSynchronize(stream));
  CUDACHECK(hipStreamDestroy(stream));
#else
  CUDACHECK(hipMemset(*ptr, 0, nelem*sizeof(T)));
#endif
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
