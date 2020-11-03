/*
Copyright (c) 2020 Advanced Micro Devices, Inc. All rights reserved.

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in
all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
THE SOFTWARE.
*/

#ifndef RCCL_CLIQUE_MANAGER_HPP_
#define RCCL_CLIQUE_MANAGER_HPP_

#include <semaphore.h>
#include <mutex>

#include "nccl.h"
#include "devcomm.h"
#include "CliqueCommon.h"
#include "HandleCache.h"
#include "HandleShm.h"

#define NUM_HANDLES_PER_RANK 2

class CliqueManager
{
public:
  typedef enum
  {
    CLIQUE_DISABLED       = 0,
    CLIQUE_SINGLE_PROCESS = 1,
    CLIQUE_SINGLE_NODE    = 2
  } cliqueMode_t;

  CliqueManager(int const rank, int const numRanks, cliqueMode_t const cliqueMode);

  ~CliqueManager();

  void CleanUp();

  ncclResult_t Init(ncclUniqueId const* commId, int suffix);

  // Returns true if the collective is supported via a clique-based kernel
  bool IsSupported(ncclFunc_t const coll,
                   size_t const count,
                   ncclDataType_t const datatype,
                   ncclRedOp_t const op) const;

  // Provide the pointers to be exchanged across the clique for the given rank / opCount
  ncclResult_t DeclarePointers(uint64_t opCount, void const* inputPtr, void* outputPtr);

  // Set pointers for where clique-related arguments will be found
  // This sets pointers to device-accessible memory where the arguments will eventually reside
  ncclResult_t SetCliqueCollectiveArgs(CollectiveArgs* args);

  // Blocking call which waits until exchanged pointers are ready for the given rank / opCount
  // Requires all participating ranks to
  ncclResult_t WaitForPointers(uint64_t const opCount);

  // Prepares shared memory files upon initialization
  static ncclResult_t BootstrapRootInit(int pid, unsigned long hash);

protected:
  ncclResult_t CheckCacheForPtr(void* devPtr,
                                NcclIpcHandleSendCache* cache,
                                int rank,
                                hipIpcMemHandle_t* handle);

  ncclResult_t CheckCacheForHandle(hipIpcMemHandle_t handle,
                                   NcclIpcHandleRecvCache* cache,
                                   void** ptr);

  // Race-condition helper functions
  void WaitForBarrier(int opIndex);

  int                          m_rank;                               // Associated rank
  int                          m_numRanks;                           // Total number of ranks
  cliqueMode_t                 m_cliqueMode;                         // Clique mode (off/single process/single node)
  bool                         m_init;                               // Whether CliqueManager has been initialized
  cliqueDevicePtrs_t*          m_pinnedCliquePtrs;                   // Pinned-host-memory (device accessible) containing device pointers
  int*                         m_cpuBarrierGlobalCount;              // Part of CPU barrier (count variable shared across ranks)
  int*                         m_cpuBarrierGlobalSense;              // Part of CPU barrier (reset variable shared across ranks)
  int                          m_cpuBarrierLocalSense[NCCL_MAX_OPS]; // Part of CPU barrier (reset variable local to this rank)
  int*                         m_gpuBarrierGlobalCount;              // Part of GPU barrier (count variable shared across ranks)
  int*                         m_gpuBarrierGlobalSense;              // Part of GPU barrier (reset variable shared across ranks)
  int*                         m_gpuBarrierLocalSense;               // Part of GPU barrier (reset variable local to this rank)

  // IPC-related (CLIQUE_SINGLE_NODE)
  NcclIpcHandleShm             m_shmHandles;
  NcclIpcHandleSendCache*      m_ipcHandleSendCache;                 // Caches pointers to IPC handles (to send to other processes)
  NcclIpcHandleRecvCache*      m_ipcHandleRecvCache;                 // Caches IPC handles to pointers (received from other processes)
  ShmObject<int32_t>           m_sharedCpuMemory;                    // Used to pass shared memory used for CPU barrier
  ShmObject<hipIpcMemHandle_t> m_sharedIpcHandle;                    // Used to pass fine-grained device memory buffer IPC handle
  int*                         m_fineGrainBarrierMem;                // Fine-grained GPU memory barrier (allocated only on 1st rank, shared on others)

  // Single-process (CLIQUE_SINGLE_PROCESS)
  static cliqueDevicePtrs_t    m_staticCliquePtrs[NCCL_MAX_OPS];
  static int32_t               m_staticGlobalCount[NCCL_MAX_OPS];
  static int32_t               m_staticGlobalSense[NCCL_MAX_OPS];
  static int*                  m_staticBarrierMem;
};

// For use in bootstrapping code
struct bootstrapRootStruct {
    void* listenComm;
    unsigned long hash;
};

#endif
