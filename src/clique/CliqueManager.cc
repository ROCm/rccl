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

#include "CliqueManager.h"
#include "CliqueShmNames.h"
#include "MsgQueue.h"

#include "nccl.h"
#include "core.h"

#include "Hash.h"

#include "AllReduceCliqueKernel.h"

#include <hip/hip_runtime.h>
#include <stdio.h>
#include <stdlib.h>
#include <chrono>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <thread>

cliqueDevicePtrs_t CliqueManager::m_cliquePtrs[NCCL_MAX_OPS]     = {};
uint32_t           CliqueManager::m_staticCounters[NCCL_MAX_OPS] = {};
int*               CliqueManager::m_staticBarriers = NULL;

CliqueManager::CliqueManager(int          const  rank,
                             int          const  numRanks,
                             cliqueMode_t const  cliqueMode) :
  m_rank(rank),
  m_numRanks(numRanks),
  m_cliqueMode(cliqueMode),
  m_init(false),
  m_deviceBarriers(NULL)
{
}

CliqueManager::~CliqueManager()
{
  if (m_init)
  {
    CleanUp();
  }
}

void CliqueManager::CleanUp()
{
  if (m_cliqueMode == CLIQUE_SINGLE_NODE)
  {
    // Release caches
    if (m_ipcHandleSendCache) delete m_ipcHandleSendCache;
    if (m_ipcHandleSendCache) delete m_ipcHandleRecvCache;

    // Close shared memory
    m_shmHandles.Close();
    m_sharedCounters.Close();
    m_sharedBarrier.Close();

    if (m_rank == 0)
    {
      hipFree(m_deviceBarriers);
    }
  }
  else if (m_cliqueMode == CLIQUE_SINGLE_PROCESS)
  {
    if (m_staticBarriers) hipHostFree(m_staticBarriers);
  }
  m_init = false;
}

ncclResult_t CliqueManager::Init(ncclUniqueId const* commId, int suffix)
{
  ncclResult_t res;

  if (m_init) return ncclSuccess;
  m_init = true;

  // Check parameters
  if (m_rank < 0 || m_rank >= m_numRanks)
  {
    WARN("Invalid rank specified.  Expected 0 <= %d < %d for CliqueManager", m_rank, m_numRanks);
    return ncclInvalidUsage;
  }
  if (commId == NULL)
  {
    WARN("CommId should not be empty");
    return ncclInvalidUsage;
  }

  // For now, opt-into clique based kernels via RCCL_ENABLE_CLIQUE env var
  if (!getenv("RCCL_ENABLE_CLIQUE"))
  {
    if (m_rank == 0) INFO(NCCL_INIT, "Disabling clique-based kernels (did not find env var RCCL_ENABLE_CLIQUE)");
    m_cliqueMode = CLIQUE_DISABLED;
    return ncclSuccess;
  }

  unsigned long hash = djb2Hash(commId->internal);
  std::string shmSuffix = std::to_string(hash) + "_" + std::to_string(suffix);

  if (m_cliqueMode == CLIQUE_SINGLE_NODE)
  {
    // Initialize shared memory file for IPC handles (based on commId hash)
    m_shmHandles = NcclIpcHandleShm(m_rank, m_numRanks, hash, NUM_HANDLES_PER_RANK, NCCL_MAX_OPS, shmSuffix);
    NCCLCHECKGOTO(m_shmHandles.Open(), res, dropback);

    // Initialize IPC caches
    m_ipcHandleSendCache = new NcclIpcHandleSendCache(m_numRanks * NUM_HANDLES_PER_RANK * NCCL_MAX_OPS);
    m_ipcHandleRecvCache = new NcclIpcHandleRecvCache(m_numRanks * NUM_HANDLES_PER_RANK * NCCL_MAX_OPS,
                                                      100,
                                                      hipIpcMemHandleHash,
                                                      hipIpcMemHandleEqual);

    // Initialize shared host barrier counters
    m_sharedCounters = ShmObject<uint32_t>(NCCL_MAX_OPS * sizeof(uint32_t),
                                           CliqueShmNames["SharedCounters"] + shmSuffix,
                                           m_rank,
                                           m_numRanks,
                                           hash);
    NCCLCHECKGOTO(m_sharedCounters.Open(), res, dropback);
    m_arrivalCounter = m_sharedCounters.Get();

    // Initialized shared barriers
    m_sharedBarrier = ShmObject<hipIpcMemHandle_t>(std::max(4096LU, sizeof(hipIpcMemHandle_t)),
                                                   CliqueShmNames["Barriers"] + shmSuffix,
                                                   m_rank,
                                                   m_numRanks,
                                                   hash);
    NCCLCHECKGOTO(m_sharedBarrier.Open(), res, dropback);

    if (m_rank == 0)
    {
      hipIpcMemHandle_t handle;
      // Allocate fine-grained device memory on rank 0 and get handle for it and store in IPC
      NCCLCHECKGOTO(ncclCudaCalloc(&m_deviceBarriers, NCCL_MAX_OPS * sizeof(int), true), res, dropback);
      if (hipIpcGetMemHandle(&handle, m_deviceBarriers) != hipSuccess)
      {
        WARN("Unable to get IPC handle for barrier memory");
        goto dropback;
      }

      *m_sharedBarrier.Get() = handle;
    }
  }
  else if (m_cliqueMode == CLIQUE_SINGLE_PROCESS)
  {
    // Allocate and zero pinned host memory that all GPU kernels will have access to as a barrier
    if (hipHostMalloc(&m_staticBarriers, sizeof(int) * NCCL_MAX_OPS) != hipSuccess)
    {
      WARN("Unable to allocated pinned host memory for clique barrier.  Disabling clique-based kernels");
      m_cliqueMode = CLIQUE_DISABLED;
      m_init = true;
      return ncclSuccess;
    }
    memset(m_staticBarriers, 0, NCCL_MAX_OPS * sizeof(int));
    m_arrivalCounter = m_staticCounters;
  }
  m_init = true;
  return ncclSuccess;

dropback:
  WARN("Unable to initialize shared memory. Disabling clique-based kernels");
  CleanUp();
  m_cliqueMode = CLIQUE_DISABLED;
  return ncclSuccess;
}

bool CliqueManager::IsSupported(ncclFunc_t const coll,
                                size_t const count,
                                ncclDataType_t const datatype,
                                ncclRedOp_t const op) const
{
  if (m_cliqueMode == CLIQUE_DISABLED) return false;
  if (coll == ncclCollAllReduce) return true;

  // NOTE: Currently we only support allReduce
//#define ALL_REDUCE_COUNT 1048576
  //if (coll == ncclCollAllReduce && count < ALL_REDUCE_COUNT) return true;

  return false;
}

ncclResult_t CliqueManager::DeclarePointers(uint64_t opCount, void const* inputPtr, void* outputPtr)
{
  // Do nothing if disabled
  if (m_cliqueMode == CLIQUE_DISABLED) return ncclSuccess;

  if (!m_init)
  {
    WARN("CliqueManager must be initialized before use");
    return ncclInvalidUsage;
  }

  int const opIndex = opCount % NCCL_MAX_OPS;
  if (m_cliqueMode == CLIQUE_SINGLE_NODE)
  {
    // Get fine-grained device memory if not already done
    if (m_deviceBarriers == NULL)
    {
      hipIpcMemHandle_t handle = *m_sharedBarrier.Get();
      CUDACHECK(hipIpcOpenMemHandle((void**)&m_deviceBarriers, handle, hipIpcMemLazyEnablePeerAccess));
    }

    std::vector<hipIpcMemHandle_t> handles(NUM_HANDLES_PER_RANK);

    // Get IPC handles for input/output pointers from cache
    NCCLCHECK(CheckCacheForPtr(const_cast<void*>(inputPtr), m_ipcHandleSendCache, m_rank, &handles[0]));
    NCCLCHECK(CheckCacheForPtr(outputPtr                  , m_ipcHandleSendCache, m_rank, &handles[1]));

    // Write IPC handles to shared memory for given rank / opCount
    NCCLCHECK(m_shmHandles.WriteHandles(opIndex, handles));

  }
  else if (m_cliqueMode == CLIQUE_SINGLE_PROCESS)
  {
    // Store this rank's input/output pointers into static member
    m_cliquePtrs[opIndex].inputs[m_rank]  = inputPtr;
    m_cliquePtrs[opIndex].outputs[m_rank] = outputPtr;
  }
  return ncclSuccess;
}

ncclResult_t CliqueManager::QueueKernel(uint64_t       const opCount,
                                        ncclFunc_t     const coll,
                                        size_t         const count,
                                        ncclDataType_t const datatype,
                                        ncclRedOp_t    const op,
                                        int            const root,
                                        hipStream_t    const stream)
{
  // Do nothing if disabled
  if (m_cliqueMode == CLIQUE_DISABLED) return ncclSuccess;
  if (!m_init)
  {
    WARN("CliqueManager must be initialized before use");
    return ncclInvalidUsage;
  }

  // Wait for all ranks to declare pointers
  int opIndex = opCount % NCCL_MAX_OPS;
  WaitForBarrier(opIndex);

  // Get cliqueDevicePointers
  cliqueDevicePtrs_t cliquePtrs;
  NCCLCHECK(GetCliqueDevicePointers(opCount, cliquePtrs));

  // NOTE: The number of blocks to use per GPU will need to be further optimized
  int gridSize = (getenv("RCCL_CLIQUE_GRIDSIZE") ? atoi(getenv("RCCL_CLIQUE_GRIDSIZE")) : 2);

  // Launch kernel
  switch (coll)
  {
  case ncclCollAllReduce:
    return AllReduceCliqueKernel::Launch(m_rank, m_numRanks, gridSize, count, datatype, op, stream, cliquePtrs);
  default:
    WARN("Unsupported collective type");
    return ncclInvalidUsage;
  }
}

ncclResult_t CliqueManager::GetCliqueDevicePointers(uint64_t opCount, cliqueDevicePtrs_t& cliquePtrs)
{
  // Wait for completion for current opCount
  int opIndex = opCount % NCCL_MAX_OPS;

  if (m_cliqueMode == CLIQUE_SINGLE_NODE)
  {
    // Collect the ready handles from shared memory and convert them to device pointers
    int numHandles = m_numRanks * NUM_HANDLES_PER_RANK;
    std::vector<hipIpcMemHandle_t> handles(numHandles);

    NCCLCHECK(m_shmHandles.ReadHandles(opIndex, handles));

    for (int i = 0; i < m_numRanks; i++)
    {
      void *input;
      NCCLCHECK(CheckCacheForHandle(handles[i * NUM_HANDLES_PER_RANK],
                                    m_ipcHandleRecvCache, &input));
      cliquePtrs.inputs[i] = const_cast<const void *>(input);

      NCCLCHECK(CheckCacheForHandle(handles[(i * NUM_HANDLES_PER_RANK) + 1],
                                    m_ipcHandleRecvCache, &cliquePtrs.outputs[i]));
    }
    cliquePtrs.barrierCounter = &(m_deviceBarriers[opIndex]);
  }
  else if (m_cliqueMode == CLIQUE_SINGLE_PROCESS)
  {
    m_cliquePtrs[opIndex].barrierCounter = &m_staticBarriers[opIndex];
    cliquePtrs = m_cliquePtrs[opIndex];
  }
  return ncclSuccess;
}

ncclResult_t CliqueManager::CheckCacheForPtr(void* devPtr,
                                             NcclIpcHandleSendCache* cache,
                                             int rank,
                                             hipIpcMemHandle_t* handle)
{
  uint64_t addr = (uint64_t)devPtr;

  // handle NULL ptr case
  if (addr == 0)
  {
    WARN("Error while checking IPC memory handle cache for ptr: null pointer specified.\n");
    return ncclInternalError;
  }

  NcclIpcHandleSendCache::iterator it = cache->find(addr);

  if (it == cache->end())
  {
    CUDACHECK(hipIpcGetMemHandle(handle, devPtr));
    std::pair<uint64_t, hipIpcMemHandle_t> ptrHandleMap(addr, *handle) ;
    cache->insert(addr, *handle);
  }
  else
  {
    *handle = (it->second).first;
  }
  return ncclSuccess;
}

ncclResult_t CliqueManager::CheckCacheForHandle(hipIpcMemHandle_t handle,
                                                NcclIpcHandleRecvCache* cache,
                                                void** ptr)
{
  NcclIpcHandleRecvCache::iterator it = cache->find(handle);

  if (it == cache->end())
  {
    CUDACHECK(hipIpcOpenMemHandle(ptr, handle, hipIpcMemLazyEnablePeerAccess));
    cache->insert(handle, *ptr);
  }
  else
  {
    *ptr = (it->second).first;
  }
  return ncclSuccess;
}

void CliqueManager::WaitForBarrier(int opIndex)
{
  m_nextBarrierValue[opIndex] += m_numRanks;
  int const nextValue = m_nextBarrierValue[opIndex];

  __atomic_add_fetch(&m_arrivalCounter[opIndex], 1, __ATOMIC_SEQ_CST);
  while (m_arrivalCounter[opIndex] < nextValue)
  {
    std::this_thread::yield();
  }
}

ncclResult_t CliqueManager::BootstrapRootInit(int pid, unsigned long hash)
{
  for (auto it = CliqueShmNames.begin(); it != CliqueShmNames.end(); it++)
  {
    int msgid, fd;
    std::string msgQueueName = "/tmp/" + it->second + std::to_string(hash) + "_" + std::to_string(pid);
    SYSCHECKVAL(open(msgQueueName.c_str(), O_CREAT | O_RDWR), "open", fd);
    NCCLCHECK(MsgQueueGetId(msgQueueName, hash, true, msgid));
  }

  std::string shmDir = "/dev/shm/";

  for (auto it = CliqueShmNames.begin(); it != CliqueShmNames.end(); it++)
  {
    struct stat fileStatus;
    std::string shmFileName = it->second + std::to_string(hash) + "_" + std::to_string(pid);
    std::string shmFullPath = shmDir + shmFileName;

    // Check if shm file already exists; if so, unlink it
    if (stat(shmFullPath.c_str(), &fileStatus) == 0)
    {
      NCCLCHECK(shmUnlink(shmFileName.c_str()));
    }
  }
  return ncclSuccess;
}
