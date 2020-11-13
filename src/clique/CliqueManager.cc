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
#include <hsa/hsa_ext_amd.h>
#include <stdio.h>
#include <stdlib.h>
#include <chrono>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <thread>

cliqueDevicePtrs_t CliqueManager::m_staticCliquePtrs[NCCL_MAX_OPS]  = {};
int*               CliqueManager::m_staticGpuBarrierMem             = NULL;

// Define some environment variables that affect clique-based kernels
RCCL_PARAM(EnableClique, "ENABLE_CLIQUE", 0);                                  // Opt-in environment variable for clique-based kernels
RCCL_PARAM(AllReduceCliqueByteLimit, "CLIQUE_ALLREDUCE_BYTE_LIMIT", 2097152);  // Max number of bytes to use clique-based kernels for all reduce
RCCL_PARAM(AllReduceNumChannels,     "CLIQUE_ALLREDUCE_NCHANNELS", 4);         // Number of channels to use for all-reduce

CliqueManager::CliqueManager(int          const  rank,
                             int          const  numRanks,
                             cliqueMode_t const  cliqueMode) :
  m_rank(rank),
  m_numRanks(numRanks),
  m_cliqueMode(cliqueMode),
  m_init(false),
  m_pinnedCliquePtrs(NULL),
  m_fineGrainBarrierMem(NULL)
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
  if (m_cliqueMode == CLIQUE_DISABLED) return;

  // Free variables that are shared between SINGLE_PROCESS / SINGLE_NODE
  if (m_pinnedCliquePtrs) hipHostFree(m_pinnedCliquePtrs);
  if (m_gpuBarrierLocalSense) hipFree(m_gpuBarrierLocalSense);

  if (m_cliqueMode == CLIQUE_SINGLE_NODE)
  {
    // Release caches
    if (m_ipcHandleSendCache) delete m_ipcHandleSendCache;
    if (m_ipcHandleRecvCache)
    {
        m_ipcHandleRecvCache->close();
        delete m_ipcHandleRecvCache;
    }

    // Close shared memory
    m_shmHandles.Close();
    m_sharedCpuMemory.Close();
    m_sharedIpcHandle.Close();

    if (m_fineGrainBarrierMem)
    {
      if (m_rank == 0)
        hipFree(m_fineGrainBarrierMem);
      else
        hipIpcCloseMemHandle(m_fineGrainBarrierMem);
    }
  }
  else if (m_cliqueMode == CLIQUE_SINGLE_PROCESS)
  {
    if (m_rank == 0 && m_staticGpuBarrierMem)
      hipFree(m_staticGpuBarrierMem);
  }
  m_init = false;
}

ncclResult_t CliqueManager::Init(ncclUniqueId const* commId, int suffix)
{
  ncclResult_t res;

  if (m_init) return ncclSuccess;
  m_init = true;

  if (m_cliqueMode == CLIQUE_DISABLED) return ncclSuccess;

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
  if (!rcclParamEnableClique())
  {
    INFO(NCCL_INIT, "Disabling clique-based kernels (did not find env var RCCL_ENABLE_CLIQUE)");
    m_cliqueMode = CLIQUE_DISABLED;
    return ncclSuccess;
  }

  // Allocate pinned CPU memory for holding clique pointers, which kernels will have access to
  if (hipHostMalloc(&m_pinnedCliquePtrs, sizeof(cliqueDevicePtrs_t) * NCCL_MAX_OPS) != hipSuccess)
  {
    WARN("Unable to allocated pinned host memory for clique pointers.  Disabling clique-based kernels");
    m_cliqueMode = CLIQUE_DISABLED;
    m_init = true;
    return ncclSuccess;
  }

  unsigned long hash = djb2Hash(commId->internal);
  std::string shmSuffix = std::to_string(hash) + "_" + std::to_string(suffix);

  // Allocate sense barrier variable on local GPU
  NCCLCHECKGOTO(ncclCudaCalloc(&m_gpuBarrierLocalSense, NCCL_MAX_OPS * sizeof(int)), res, dropback);

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

    // Initialize shared object for GPU barrier IPC handle
    m_sharedIpcHandle = ShmObject<hipIpcMemHandle_t>(std::max(4096LU, sizeof(hipIpcMemHandle_t)),
                                                     CliqueShmNames["Barriers"] + shmSuffix,
                                                     m_rank,
                                                     m_numRanks,
                                                     hash);
    NCCLCHECKGOTO(m_sharedIpcHandle.Open(), res, dropback);

    if (m_rank == 0)
    {
      hipIpcMemHandle_t handle;
      // Allocate fine-grained device memory on rank 0 and get IPC handle for it
      // Re-usable barrier consists of (globalCount / globalSense) pair of integers
      NCCLCHECKGOTO(ncclCudaCalloc(&m_fineGrainBarrierMem, NCCL_MAX_OPS * 2 * sizeof(int), true), res, dropback);
      if (hipIpcGetMemHandle(&handle, m_fineGrainBarrierMem) != hipSuccess)
      {
        WARN("Unable to get IPC handle for barrier memory");
        goto dropback;
      }
      // Write IPC handle to shared memory for other ranks to receive
      *m_sharedIpcHandle.Get() = handle;

      // Set up global count/sense for first rank
      m_gpuBarrierGlobalCount = &m_fineGrainBarrierMem[0];
      m_gpuBarrierGlobalSense = &m_fineGrainBarrierMem[NCCL_MAX_OPS];
    }

    // Initialize shared CPU memory to be used for barrier variables
    m_sharedCpuMemory = ShmObject<int32_t>(2 * sizeof(int32_t),
                                           CliqueShmNames["SharedCounters"] + shmSuffix,
                                           m_rank,
                                           m_numRanks,
                                           hash);
    NCCLCHECKGOTO(m_sharedCpuMemory.Open(), res, dropback);

    // Split up the shared CPU memory for barrier counters / global sense
    m_cpuBarrierGlobalCount = &m_sharedCpuMemory.Get()[0];
    m_cpuBarrierGlobalSense = &m_sharedCpuMemory.Get()[1];

    // Initialize CPU barriers
    if (m_rank == 0)
    {
      *m_cpuBarrierGlobalCount = 0;
      *m_cpuBarrierGlobalSense = 0;
    }
    m_cpuBarrierLocalSense = 0;
  }
  else if (m_cliqueMode == CLIQUE_SINGLE_PROCESS)
  {
    // First rank prepares fine-grained memory shared across ranks used for the two barrier variables
    if (m_rank == 0)
    {
      NCCLCHECKGOTO(ncclCudaCalloc(&m_staticGpuBarrierMem, NCCL_MAX_OPS * 2 * sizeof(int), true), res, dropback);

      // Prepare all barriers
      for (int opIndex = 0; opIndex < NCCL_MAX_OPS; opIndex++)
      {
        m_staticCliquePtrs[opIndex].barrier.globalCount = &m_staticGpuBarrierMem[opIndex];
        m_staticCliquePtrs[opIndex].barrier.globalSense = &m_staticGpuBarrierMem[opIndex + NCCL_MAX_OPS];;
      }
    }
  }


  m_init = true;
  INFO(NCCL_INIT, "Clique-based kernels enabled (mode %d)", m_cliqueMode);
  return ncclSuccess;

dropback:
  // NOTE: This currently assumes that all ranks will fail the same way
  //       Additional support is required to handle cases when some processes succeed while others fail
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

  // Filter based on total input size for each collective type
  size_t totalBytes = count * ncclTypeSize(datatype);
  if (coll == ncclCollAllReduce && (totalBytes <= rcclParamAllReduceCliqueByteLimit())) return true;

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

  // Add opIndex to queue of in-progress collectives
  m_inProgress.push(opIndex);

  if (m_cliqueMode == CLIQUE_SINGLE_NODE)
  {
    // Get fine-grained device memory if not already done
    if (m_fineGrainBarrierMem == NULL)
    {
      hipIpcMemHandle_t handle = *m_sharedIpcHandle.Get();
      CUDACHECK(hipIpcOpenMemHandle((void**)&m_fineGrainBarrierMem, handle, hipIpcMemLazyEnablePeerAccess));

      // Prepare global count/sense barrier variables used the ipc-shared gpu device memory
      m_gpuBarrierGlobalCount = &m_fineGrainBarrierMem[0];
      m_gpuBarrierGlobalSense = &m_fineGrainBarrierMem[NCCL_MAX_OPS];
    }

    std::vector<std::pair<hipIpcMemHandle_t,size_t>> handles(NUM_HANDLES_PER_RANK);

    // Get IPC handles for input/output pointers from cache
    NCCLCHECK(CheckCacheForPtr(const_cast<void*>(inputPtr), m_ipcHandleSendCache, m_rank, &handles[0]));
    NCCLCHECK(CheckCacheForPtr(outputPtr                  , m_ipcHandleSendCache, m_rank, &handles[1]));

    // Prepare barrier pointers (done after the IpcOpenMemory)
    m_pinnedCliquePtrs[opIndex].barrier.globalCount = &m_gpuBarrierGlobalCount[opIndex];
    m_pinnedCliquePtrs[opIndex].barrier.globalSense = &m_gpuBarrierGlobalSense[opIndex];
    m_pinnedCliquePtrs[opIndex].barrier.localSense  = &m_gpuBarrierLocalSense[opIndex];

    // Write IPC handles to shared memory for given rank / opCount
    NCCLCHECK(m_shmHandles.WriteHandles(opIndex, handles));
  }
  else if (m_cliqueMode == CLIQUE_SINGLE_PROCESS)
  {
    // Store this rank's input/output pointers into static member
    m_staticCliquePtrs[opIndex].inputs[m_rank]  = inputPtr;
    m_staticCliquePtrs[opIndex].outputs[m_rank] = outputPtr;
  }

  return ncclSuccess;
}

ncclResult_t CliqueManager::GetNumChannelsToUse(ncclFunc_t const coll,
                                                size_t const count,
                                                ncclDataType_t const datatype,
                                                ncclRedOp_t const op,
                                                int const totalNumChannels,
                                                uint8_t* numChannelstoUse)
{
  size_t const totalBytes = count * ncclTypeSize(datatype);
  *numChannelstoUse = 1;

  if (coll == ncclCollAllReduce) {
    *numChannelstoUse = std::min((int)rcclParamAllReduceNumChannels(), totalNumChannels);
  }

  return ncclSuccess;
}



ncclResult_t CliqueManager::SetCliqueCollectiveArgs(CollectiveArgs* args)
{
  // Do nothing if disabled
  if (m_cliqueMode == CLIQUE_DISABLED) return ncclSuccess;
  if (!m_init)
  {
    WARN("CliqueManager must be initialized before use");
    return ncclInvalidUsage;
  }

  // Prepare clique argments (NOTE: clique pointers are not ready yet)
  int opIndex = args->opCount % NCCL_MAX_OPS;
  args->clique.ptrs = &m_pinnedCliquePtrs[opIndex];


  // Determine number of channels to use for this collective
  args->clique.nChannels = rcclParamAllReduceNumChannels();

  return ncclSuccess;
}

ncclResult_t CliqueManager::WaitForPointers()
{
  // Do nothing if disabled
  if (m_cliqueMode == CLIQUE_DISABLED) return ncclSuccess;

  if (!m_init)
  {
    WARN("CliqueManager must be initialized before use");
    return ncclInvalidUsage;
  }

  // Do nothing if there are no outstanding clique-kernels
  if (m_inProgress.empty()) return ncclSuccess;

  // Copy clique device pointers to pinned device memory
  if (m_cliqueMode == CLIQUE_SINGLE_NODE)
  {
    // Wait for all ranks to arrive
    WaitForBarrier();

    int numHandles = m_numRanks * NUM_HANDLES_PER_RANK;
    std::vector<std::pair<hipIpcMemHandle_t,size_t>> handles(numHandles);

    while (!m_inProgress.empty())
    {
      int const opIndex = m_inProgress.front();
      m_inProgress.pop();

      // Collect the ready handles from shared memory and convert them to device pointers
      NCCLCHECK(m_shmHandles.ReadHandles(opIndex, handles));
      for (int i = 0; i < m_numRanks; i++)
      {
        void *input;
        NCCLCHECK(CheckCacheForHandle(handles[i * NUM_HANDLES_PER_RANK],
                                      m_ipcHandleRecvCache, &input));
        m_pinnedCliquePtrs[opIndex].inputs[i] = const_cast<const void *>(input);

        NCCLCHECK(CheckCacheForHandle(handles[(i * NUM_HANDLES_PER_RANK) + 1],
                                      m_ipcHandleRecvCache, &m_pinnedCliquePtrs[opIndex].outputs[i]));
      }
    }
  }
  else if (m_cliqueMode == CLIQUE_SINGLE_PROCESS)
  {
    while (!m_inProgress.empty())
    {
      int const opIndex = m_inProgress.front();
      m_inProgress.pop();

      // Copy from static memory to pinned host memory and set local sense
      memcpy(&m_pinnedCliquePtrs[opIndex], &m_staticCliquePtrs[opIndex], sizeof(cliqueDevicePtrs_t));
      m_pinnedCliquePtrs[opIndex].barrier.localSense = &m_gpuBarrierLocalSense[opIndex];
    }
  }
  return ncclSuccess;
}

std::string HandleToString(hipIpcMemHandle_t handle)
{
  char mapping[17] = "0123456789ABCDEF";
  std::string result;
  for (int i = 0; i < 4; i++)
  {
    unsigned char val = (unsigned char)handle.reserved[i];
    result += mapping[val / 16];
    result += mapping[val % 16];
  }
  return result;
}


ncclResult_t CliqueManager::CheckCacheForPtr(void* devPtr,
                                             NcclIpcHandleSendCache* cache,
                                             int rank,
                                             std::pair<hipIpcMemHandle_t, size_t>* handlePair)
{
  // Get the base address for this device allocation
  hsa_status_t status;
  hsa_amd_pointer_info_t info;
  info.size = sizeof(hsa_amd_pointer_info_t);
  status = hsa_amd_pointer_info(devPtr, &info, NULL, NULL, NULL);
  if (status != HSA_STATUS_SUCCESS) {
    WARN("Uanble to get pointer information for %p", devPtr);
    return ncclInvalidArgument;
  }

  // Compute the offset between the device addres and the base address
  uint64_t baseAddr = (uint64_t)info.agentBaseAddress;
  uint64_t realAddr = (uint64_t)devPtr;
  handlePair->second = realAddr - baseAddr;

  // IPC handles are only supported for base address pointers
  NcclIpcHandleSendCache::iterator it = cache->find(baseAddr);

   if (it == cache->end())
   {
     CUDACHECK(hipIpcGetMemHandle(&handlePair->first, (void*)baseAddr));
     cache->insert(baseAddr, handlePair->first);
   }
   else
   {
     handlePair->first = (it->second).first;
   }
   return ncclSuccess;
}

ncclResult_t CliqueManager::CheckCacheForHandle(std::pair<hipIpcMemHandle_t, size_t> const& handlePair,
                                                NcclIpcHandleRecvCache* cache,
                                                void** ptr)
{
  NcclIpcHandleRecvCache::iterator it = cache->find(handlePair.first);

  // Get base address pointer from cache if it exists
  void* baseAddr;
  if (it == cache->end())
  {
    CUDACHECK(hipIpcOpenMemHandle(&baseAddr, handlePair.first, hipIpcMemLazyEnablePeerAccess));
    cache->insert(handlePair.first, baseAddr);
  }
  else
  {
    baseAddr = (it->second).first;
  }

  // Modify base address pointer with offset
  uint64_t realAddr = (uint64_t)baseAddr + handlePair.second;
  *ptr = (void*)realAddr;
  return ncclSuccess;
}

void CliqueManager::WaitForBarrier()
{
  // Sense inversion barrier
  m_cpuBarrierLocalSense = 1 - m_cpuBarrierLocalSense;

  if (__sync_add_and_fetch(m_cpuBarrierGlobalCount, 1) == m_numRanks)
  {
    // Reset the barrier
    STORE(m_cpuBarrierGlobalCount, 0);
    STORE(m_cpuBarrierGlobalSense, m_cpuBarrierLocalSense);
  } else {
    while (LOAD(m_cpuBarrierGlobalSense) != m_cpuBarrierLocalSense);
  }
}

ncclResult_t CliqueManager::BootstrapRootInit(int pid, unsigned long hash)
{
  for (auto it = CliqueShmNames.begin(); it != CliqueShmNames.end(); it++)
  {
    int msgid, fd;
    std::string msgQueueName = "/tmp/" + it->second + std::to_string(hash) + "_" + std::to_string(pid);
    SYSCHECKVAL(open(msgQueueName.c_str(), O_CREAT | O_RDWR, 0606), "open", fd);
    NCCLCHECK(MsgQueueGetId(msgQueueName, hash, true, msgid));
    SYSCHECK(close(fd), "close");
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
