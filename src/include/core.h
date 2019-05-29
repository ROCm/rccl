/*************************************************************************
 * Copyright (c) 2015-2018, NVIDIA CORPORATION. All rights reserved.
 * Modifications Copyright (c) 2019 Advanced Micro Devices, Inc. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

#ifndef NCCL_CORE_H_
#define NCCL_CORE_H_

#define NCCL_MAX_OPS 2048

#include "nccl.h"
#include "transport.h"
#include "debug.h"
#include <cstdio>
#include <algorithm> // std::min/std::max
#include <unistd.h>
#include <stdlib.h>
#include <hip/hip_runtime_api.h>
#include <hip/hip_runtime.h>

#if CUDART_VERSION < 9000
struct cudaLaunchParams {
  void (*func)(struct ncclColl);
  dim3 gridDim;
  dim3 blockDim;
  struct ncclColl **args;
  size_t sharedMem;
  hipStream_t stream;
};
#endif

#define MAXRINGS 16
#define MAXTHREADS 256
#define DEFAULT_BUFFER_SIZE_BYTES (1LL << 22) /* 4MiB */

// Rings / LL tuning
#define NCCL_LL_RING_THRESHOLD 8 // Per thread size before we start increasing nrings
#define NCCL_THREAD_THRESHOLD 256 // Per thread size before we switch to non-LL for Volta and above
#define NCCL_THREAD_THRESHOLD_PREVOLTA 32 // Per thread size before we switch to non-LL for pre-Volta archs
#define NCCL_LL_MAX_NTHREADS 256
#define NCCL_LL_MIN_NTHREADS 256

#define DIVUP(x, y) \
    (((x)+(y)-1)/(y))
#define ROUNDUP(x, y) \
    (DIVUP((x), (y))*(y))

#define ALIGN_SIZE(size, align) \
  size = ((size + (align) - 1) / (align)) * (align);

union ncclLLFifoLine {
  /* Flags have to be *after* data, because otherwise, an incomplete receive
     from the network may receive the flag but not the data.
     Note this is assuming that either we receive contiguous chunks of data
     (sockets) or data is written with an atomicity of 8 bytes (IB/RDMA). */
  struct {
    uint32_t data1;
    uint32_t flag1;
    uint32_t data2;
    uint32_t flag2;
  };
  uint64_t v[2];
  int4 i4;
};

struct ncclConnInfo {
  // Regular comm mechanism
  char *buff;         // Local for recv, remote for send
  uint64_t *tail;     // Local for recv, remote for send
  uint64_t *head;     // Local for send, remote for recv
  uint64_t *opCount;  // Local for recv, remote for send

  int direct;         // Direct communication
  void **ptrExchange; // Pointer exchange for direct communication

  int *fifo;          // Size fifo for proxy

  // Low latency mechanism
  char *llBuff;       // Local for recv, remote for send
  uint64_t *llHead;   // Local for send, remote for recv
  int *llFifo;        // LL Size fifo for proxy
  uint64_t llStep;    // Keep where we are
  uint64_t llLastCleaning;
};

struct ncclConnector {
  struct transportProxyInfo* proxyInfo;
  struct ncclTransport* transport;
  void* transportResources; // Host-side resources
  struct ncclConnInfo conn;
};

#define CACHE_LINE_SIZE 64
#define MEM_ALIGN 4096
#define SIZES_FIFO_SIZE 16
#define CUDA_IPC_MIN 2097152UL /* 2MiB - not currently used */

#define NCCL_LL_CHUNKS 8
#define NUM_LINES_PER_THREAD 8
#define NCCL_LL_BUFF_SIZE (NUM_LINES_PER_THREAD*NCCL_LL_MAX_NTHREADS*NCCL_LL_CHUNKS*sizeof(union ncclLLFifoLine)) // 256K
#define NCCL_LL_BUFF_LINES (NCCL_LL_BUFF_SIZE / (2*sizeof(uint64_t)))
#define NCCL_LL_SLICE_LINES (NCCL_LL_BUFF_LINES / NCCL_LL_CHUNKS)
#define NCCL_LL_CLEAN_FREQ 0x10000000

struct ncclSendMem {
  union {
    struct {
      uint64_t head;
      char pad1[CACHE_LINE_SIZE-sizeof(uint64_t)];
      void* ptrExchange;
      char pad2[CACHE_LINE_SIZE-sizeof(void*)];
      uint64_t llHead;
    };
    char pad3[MEM_ALIGN];
  };
};

struct ncclRecvMem {
  union {
    struct {
      uint64_t tail;
      char pad2[CACHE_LINE_SIZE-sizeof(uint64_t)];
      uint64_t opCount;
      char pad4[CACHE_LINE_SIZE-sizeof(uint64_t)];
      int sizesFifo[SIZES_FIFO_SIZE];
      int llSizesFifo[SIZES_FIFO_SIZE];
    };
    char pad5[MEM_ALIGN];
  };
  char llBuff[NCCL_LL_BUFF_SIZE];
  char buff[1]; // Actually larger than that
};

struct ncclRing {
  union {
    struct {
      int id;
      int nthreads;
      // Per ring resources
      struct ncclSendMem* devMemSend;   // CUDA-size resources
      struct ncclRecvMem* devMemRecv;   // CUDA-size resources
      int buffSize;
      int devMemSendSize;    // Keep the size for IPCs
      int devMemRecvSize;    // Keep the size for IPCs
      struct ncclConnector send;
      struct ncclConnector recv;

      // Maps an internal nccl index to user-specified rank order. This is necessary
      // since we need to know how the user expects data to be ordered across
      // devices. Ordered from current device.
      int* userRanks;
      int* devUserRanks;

      // Next GPU's HDP_MEM_FLUSH_ADDR: HDP Memory Coherency Flush Control. This register
      // allows software to explicitly initiate a flush read to HDP memory. See more
      // descriptions in primitives.h.
      uint32_t* hdp_reg;

      // Operation list for aggregation
      struct ncclColl* collectives;
      struct ncclColl* devCollectives;
      int collStart;
      int collCount;
      int collFifoHead; // Only used by GPU
      int collFifoTail; // Only used by CPU
    };
    int data[0x80];
  };
};
static_assert(sizeof(struct ncclRing) == 0x80*sizeof(int), "ncclRing must have a pow2 size");

#pragma pack(push)  /* push current alignment to stack */
#pragma pack(4)     /* set alignment to 4 bytes boundary */
/* CollectiveArgs + ncclColl are to be a power of two, currently 64 bytes, */
/* to make sure reads to host from the CUDA kernel are aligned. */
/* Make sure to adjust padding at the end of ncclColl. */
struct CollectiveArgs {
  struct ncclComm* comm;
  uint64_t opCount;

  // local and remote input, output, and buffer
  const void * ThisInput;
  void * ThisOutput;

  // general parameters
  size_t N;
  uint32_t root;
  uint8_t bid;
  uint8_t nRings;
  uint16_t nThreads;

  int lastChunkSize;
};
struct ncclColl {
  union {
    struct {
      struct CollectiveArgs args;
      uint16_t nThreads;
      uint16_t funcIndex;
      uint16_t nextIndex;
      uint8_t  active;
    };
    int data[0x10];
  };
};
static_assert(sizeof(struct ncclColl) == (0x10*sizeof(int)), "ncclColl must have a pow2 size");
#pragma pack(pop)   /* restore original alignment from stack */

struct ncclComm {
  struct ncclRing rings[MAXRINGS];

  int rank;    // my rank in the communicator
  int nRanks;  // number of GPUs in communicator
  int cudaDev; // my cuda device index

  enum { GROUP, PARALLEL } launchMode;
  hipStream_t userStream;
  bool userStreamSet;
  hipEvent_t doneEvent;
  bool checkPointers;

  // Counter to make sure collectives match (needed for bcast/reduce
  // where syncs are not symmetric).
  uint64_t opCount;

  // Rings for collectives
  int nRings;
  int nThreads;

  // Low-latency algorithm threshold
  ssize_t llThreshold;
  ssize_t threadThreshold;

  // An internal CUDA stream for NCCL kernel CGMD launches
  int groupCudaStream;
  hipStream_t groupStream;

  // Device copy of the communicator
  struct ncclComm *devComm;

  // Intra-process sync
  int intraRank;
  int intraRanks;
  int* intraBarrier;
  int intraPhase;

  // Storage for deferred intra-process launch
  struct cudaLaunchParams * intraParams;
  struct cudaLaunchParams *myParams;
  int* intraCudaDevs;
  int* intraCGMode; // Whether we can use CUDA9 CGMD or not
  int* intraCC; // Only to check all have the same ComputeCap and disable CGMode if not
  struct ncclColl args;
  struct ncclColl* argsptr;
};

// Convert volatile access to atomic
#if defined(__HIP_PLATFORM_HCC__) || defined(__HCC__)
#define LOAD(VAR) __atomic_load_n((VAR), __ATOMIC_SEQ_CST)
#define STORE(DST, SRC) __atomic_store_n((DST), (SRC), __ATOMIC_SEQ_CST)
#else
#define LOAD(VAR) *(VAR)
#define STORE(DST, SRC) *(DST) = (SRC)
#endif

// Check CUDA calls
#define CUDACHECK(cmd) do {                                 \
    hipError_t e = cmd;                                    \
    if( e != hipSuccess ) {                                \
        WARN("Cuda failure '%s'", hipGetErrorString(e));   \
        return ncclUnhandledCudaError;                      \
    }                                                       \
} while(false)

#define CUDACHECKGOTO(cmd, res, label) do {                 \
    hipError_t e = cmd;                                    \
    if( e != hipSuccess ) {                                \
        WARN("Cuda failure '%s'", hipGetErrorString(e));   \
        res = ncclUnhandledCudaError;                       \
        goto label;                                         \
    }                                                       \
} while(false)

#include <errno.h>
// Check system calls
#define SYSCHECK(call, name) do { \
  int retval; \
  SYSCHECKVAL(call, name, retval); \
} while (false)

#define SYSCHECKVAL(call, name, retval) do { \
  SYSCHECKSYNC(call, name, retval); \
  if (retval == -1) { \
    WARN("Call to " name " failed : %s", strerror(errno)); \
    return ncclSystemError; \
  } \
} while (false)

#define SYSCHECKSYNC(call, name, retval) do { \
  retval = call; \
  if (retval == -1 && (errno == EINTR || errno == EWOULDBLOCK || errno == EAGAIN)) { \
    INFO(NCCL_ALL,"Call to " name " returned %s, retrying", strerror(errno)); \
  } else { \
    break; \
  } \
} while(true)

// Propagate errors up
#define NCCLCHECK(call) do { \
  ncclResult_t res = call; \
  if (res != ncclSuccess) { \
    /* Print the back trace*/ \
    INFO(NCCL_ALL,"%s:%d -> %d", __FILE__, __LINE__, res);    \
    return res; \
  } \
} while (0);

#define NCCLCHECKGOTO(call, res, label) do { \
  res = call; \
  if (res != ncclSuccess) { \
    /* Print the back trace*/ \
    INFO(NCCL_ALL,"%s:%d -> %d", __FILE__, __LINE__, res);    \
    goto label; \
  } \
} while (0);

#ifdef PROFAPI
#define NCCL_API(ret, func, args...)        \
    __attribute__ ((visibility("default"))) \
    __attribute__ ((alias(#func)))          \
    ret p##func (args);                     \
    extern "C"                              \
    __attribute__ ((visibility("default"))) \
    __attribute__ ((weak))                  \
    ret func(args)
#else
#define NCCL_API(ret, func, args...)        \
    extern "C"                              \
    __attribute__ ((visibility("default"))) \
    ret func(args)
#endif // end PROFAPI

int ncclCudaCompCap();

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

#endif // end include guard
