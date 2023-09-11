/*
Copyright (c) 2020-2021 Advanced Micro Devices, Inc. All rights reserved.

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

  void SetByteLimits();

  // Returns true if the collective is supported via a clique-based kernel
  bool IsSupported(ncclFunc_t const coll,
                   size_t const count,
                   ncclDataType_t const datatype,
                   ncclRedOp_t const op) const;

  // Provide the pointers to be exchanged across the clique for the given rank / opCount
  ncclResult_t DeclarePointers(void const* inputPtr, void* outputPtr);

  // Determine the number of channels / CUs to use for this call
  ncclResult_t GetNumChannelsToUse(ncclFunc_t const coll,
                                   size_t const count,
                                   ncclDataType_t const datatype,
                                   ncclRedOp_t const op,
                                   int const totalNumChannels,
                                   uint8_t* numChannelstoUse) const;

  // Blocking call that only returns the in-progress clique pointers are ready
  // This needs to be called in same order as DeclarePointers
  ncclResult_t WaitForPointers(ncclWorkElem* args);

  // Prepares shared memory files upon initialization
  static ncclResult_t BootstrapRootInit(int pid, unsigned long hash);

protected:
  ncclResult_t CheckCacheForPtr(void* devPtr,
				NcclIpcHandleSendCache* cache,
				int rank,
				std::pair<hipIpcMemHandle_t, size_t>* handlePair);

  ncclResult_t CheckCacheForHandle(std::pair<hipIpcMemHandle_t, size_t> const& handlePair,
				   NcclIpcHandleRecvCache* cache,
				   void** ptr);

  int                          m_rank;                               // Associated rank
  int                          m_numRanks;                           // Total number of ranks
  unsigned long                m_hash;                               // Hash used for identifying message queues & shared memory
  cliqueMode_t                 m_cliqueMode;                         // Clique mode (off/single process/single node)
  int32_t                      m_opIndexHead;                        // Track start of outstanding requests
  int32_t                      m_opIndexTail;                        // Track end of outstanding requests
  bool                         m_init;                               // Whether CliqueManager has been initialized
  int                          m_gcnArch;                            // Device GCN arch value
  size_t                       m_allReduceByteLimit;                 // Byte limit for AllReduce
  cliqueDevicePtrs_t*          m_pinnedCliquePtrs;                   // Pinned-host-memory (device accessible) containing device pointers
  int*                         m_gpuBarrierGlobalCount;              // Part of GPU barrier (count variable shared across ranks)
  int*                         m_gpuBarrierGlobalSense;              // Part of GPU barrier (reset variable shared across ranks)
  int*                         m_gpuBarrierLocalSense;               // Part of GPU barrier (reset variable local to this rank)
  int*                         m_cpuBarrierCount;                    // Points to either m_sharedBarrierCount or m_staticBarrierCount

  // IPC-related (CLIQUE_SINGLE_NODE)
  NcclIpcHandleShm             m_shmHandles;                         // Used to exchange IPC handles between ranks
  NcclIpcHandleSendCache*      m_ipcHandleSendCache;                 // Caches pointers to IPC handles (to send to other processes)
  NcclIpcHandleRecvCache*      m_ipcHandleRecvCache;                 // Caches IPC handles to pointers (received from other processes)
  ShmObject<int32_t>           m_sharedCpuMemory;                    // Used to pass shared memory used for CPU barrier
  ShmObject<hipIpcMemHandle_t> m_sharedIpcHandle;                    // Used to pass fine-grained device memory buffer IPC handle
  int*                         m_fineGrainBarrierMem;                // Fine-grained GPU memory barrier (allocated only on 1st rank, shared on others)
  int*                         m_sharedBarrierCount;                 // Part of CPU barrier (count variable shared across ranks)

  // Single-process (CLIQUE_SINGLE_PROCESS)
  static cliqueDevicePtrs_t    m_staticCliquePtrs[NCCL_MAX_OPS];     // Use shared static memory to exchange pointer info
  static int                   m_staticBarrierCount[2*NCCL_MAX_OPS]; // Part of CPU barrier (count variable shared across ranks)
  static int*                  m_staticGpuBarrierMem;                // Static storage backing for fine-grained gpu barrier
};

// For use in bootstrapping code
struct bootstrapRootStruct {
  int listenFd;
  unsigned long hash;
};

#endif
