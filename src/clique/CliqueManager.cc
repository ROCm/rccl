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
#include <unistd.h>

cliqueDevicePtrs_t CliqueManager::m_staticCliquePtrs[NCCL_MAX_OPS]     = {};
int                CliqueManager::m_staticBarrierCount[NCCL_MAX_OPS*2] = {};
int*               CliqueManager::m_staticGpuBarrierMem                = NULL;

// Define some environment variables that affect clique-based kernels
RCCL_PARAM(EnableClique, "ENABLE_CLIQUE", 0);                           // Opt-in environment variable for clique-based kernels
RCCL_PARAM(AllReduceCliqueByteLimit, "CLIQUE_ALLREDUCE_BYTE_LIMIT", 0); // Max number of bytes to use clique-based kernels for all reduce (0 for auto-select)
RCCL_PARAM(AllReduceNumChannels,     "CLIQUE_ALLREDUCE_NCHANNELS", 0);  // Number of channels to use for all-reduce. (0 for auto-select)

CliqueManager::CliqueManager(int          const  rank,
                             int          const  numRanks,
                             cliqueMode_t const  cliqueMode) :
  m_rank(rank),
  m_numRanks(numRanks),
  m_hash(0),
  m_cliqueMode(cliqueMode),
  m_opIndexHead(0),
  m_opIndexTail(0),
  m_init(false),
  m_gcnArch(0),
  m_allReduceByteLimit(0),
  m_pinnedCliquePtrs(NULL),
  m_gpuBarrierGlobalCount(NULL),
  m_gpuBarrierGlobalSense(NULL),
  m_gpuBarrierLocalSense(NULL),
  m_cpuBarrierCount(NULL),
  m_shmHandles(),
  m_ipcHandleSendCache(),
  m_ipcHandleRecvCache(),
  m_sharedCpuMemory(),
  m_sharedIpcHandle(),
  m_fineGrainBarrierMem(NULL),
  m_sharedBarrierCount(NULL)
{}

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
    INFO(NCCL_COLL, "Rank %d deleting IPC caches", m_rank);
    if (m_ipcHandleSendCache) delete m_ipcHandleSendCache;
    if (m_ipcHandleRecvCache) delete m_ipcHandleRecvCache;

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

  m_hash = djb2Hash(commId->internal);
  if (m_cliqueMode == CLIQUE_DISABLED)
  {
    INFO(NCCL_INIT, "Clique kernels disabled");
    return ncclSuccess;
  }

  // Check parameters
  if (m_rank < 0 || m_rank >= m_numRanks)
  {
    WARN("Invalid rank specified.  Expected 0 <= %d < %d for CliqueManager", m_rank, m_numRanks);
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

  std::string shmSuffix = std::to_string(m_hash) + "_" + std::to_string(suffix);

  // Allocate sense barrier variable on local GPU
  NCCLCHECKGOTO(ncclCudaCalloc(&m_gpuBarrierLocalSense, NCCL_MAX_OPS * sizeof(int)), res, dropback);

  if (m_cliqueMode == CLIQUE_SINGLE_NODE)
  {
    // Initialize shared memory file for IPC handles (based on commId hash)
    m_shmHandles = NcclIpcHandleShm(m_rank, m_numRanks, m_hash, NUM_HANDLES_PER_RANK, NCCL_MAX_OPS, shmSuffix);
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
                                                     m_hash);
    NCCLCHECKGOTO(m_sharedIpcHandle.Open(), res, dropback);

    if (m_rank == 0)
    {
      hipIpcMemHandle_t handle;
      // Allocate fine-grained device memory on rank 0 and get IPC handle for it
      // Re-usable barrier consists of (globalCount / globalSense) pair of integers
      NCCLCHECKGOTO(ncclCudaCalloc(&m_fineGrainBarrierMem, NCCL_MAX_OPS * 2 * sizeof(int), nullptr, true), res, dropback);
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
    m_sharedCpuMemory = ShmObject<int32_t>(NCCL_MAX_OPS * 2 * sizeof(int32_t),
                                           CliqueShmNames["SharedCounters"] + shmSuffix,
                                           m_rank,
                                           m_numRanks,
                                           m_hash);
    NCCLCHECKGOTO(m_sharedCpuMemory.Open(), res, dropback);

    // Split up the shared CPU memory for barrier counters / global sense
    m_cpuBarrierCount = m_sharedCpuMemory.Get();

    // Initialize CPU barriers
    if (m_rank == 0)
    {
      memset(m_cpuBarrierCount, 0, NCCL_MAX_OPS * 2 * sizeof(int32_t));
    }
  }
  else if (m_cliqueMode == CLIQUE_SINGLE_PROCESS)
  {
    m_cpuBarrierCount = &m_staticBarrierCount[0];

    // First rank prepares fine-grained memory shared across ranks used for the two barrier variables
    if (m_rank == 0)
    {
      NCCLCHECKGOTO(ncclCudaCalloc(&m_staticGpuBarrierMem, NCCL_MAX_OPS * 2 * sizeof(int), nullptr, true), res, dropback);
      // Prepare all barriers
      for (int opIndex = 0; opIndex < NCCL_MAX_OPS; opIndex++)
      {
        m_staticCliquePtrs[opIndex].barrier.globalCount = &m_staticGpuBarrierMem[opIndex];
        m_staticCliquePtrs[opIndex].barrier.globalSense = &m_staticGpuBarrierMem[opIndex + NCCL_MAX_OPS];;
      }
    }
  }

  // Figure out device arch for tuning
  int deviceId;
  CUDACHECK(hipGetDevice(&deviceId));
  hipDeviceProp_t devProp;
  CUDACHECK(hipGetDeviceProperties(&devProp, deviceId));
  m_gcnArch = devProp.gcnArch;

  // Establish when to use clique-based kernels based on input size
  SetByteLimits();

  m_init = true;
  INFO(NCCL_INIT, "Clique-based kernels enabled (mode %d) [GCN %d]", m_cliqueMode, m_gcnArch);
  return ncclSuccess;

dropback:
  // NOTE: This currently assumes that all ranks will fail the same way
  //       Additional support is required to handle cases when some processes succeed while others fail
  WARN("Unable to initialize shared memory. Disabling clique-based kernels");
  CleanUp();
  m_cliqueMode = CLIQUE_DISABLED;
  return ncclSuccess;
}

void CliqueManager::SetByteLimits()
{
  m_allReduceByteLimit = rcclParamAllReduceCliqueByteLimit();
  if (m_allReduceByteLimit == 0)
  {
    switch (m_gcnArch)
    {
    case 906: m_allReduceByteLimit =  16777216; break;
    case 908: m_allReduceByteLimit =   8388608; break;
    default:  m_allReduceByteLimit =  16777216; break;
    }
  }
}

bool CliqueManager::IsSupported(ncclFunc_t const coll,
                                size_t const count,
                                ncclDataType_t const datatype,
                                ncclRedOp_t const op) const
{
  if (m_cliqueMode == CLIQUE_DISABLED) return false;

  // Filter based on total input size for each collective type and ops sum/prod/min/max
  size_t totalBytes = count * ncclTypeSize(datatype);
  if (coll == ncclFuncAllReduce && (totalBytes <= m_allReduceByteLimit) && op < ncclAvg) return true;
  return false;
}

ncclResult_t CliqueManager::DeclarePointers(void const* inputPtr, void* outputPtr)
{
  // Do nothing if disabled
  if (m_cliqueMode == CLIQUE_DISABLED) return ncclSuccess;

  if (!m_init)
  {
    WARN("CliqueManager must be initialized before use");
    return ncclInvalidUsage;
  }

  // Add to queue of in-progress collectives
  int32_t const opIndex = m_opIndexTail;
  m_opIndexTail = (m_opIndexTail + 1) % NCCL_MAX_OPS;

  INFO(NCCL_COLL, "Rank %d declaring pointers for opIndex %d", m_rank, opIndex);
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

  // Increment entry barrier counter - must not block
  volatile int* entryCounter = &m_cpuBarrierCount[2 * opIndex];
  int entryVal = LOAD(entryCounter);
  // Loop until successful atomic update to counter
  bool done = false;
  while (done == false) {
    // Last rank resets exit barrier counter prior to incrementing entry count to numRanks
    if (entryVal+1 == m_numRanks)
      m_cpuBarrierCount[2 * opIndex + 1] = 0;
    done = __sync_bool_compare_and_swap(entryCounter, entryVal, entryVal+1);
    entryVal++;
  }
  return ncclSuccess;
}

ncclResult_t CliqueManager::GetNumChannelsToUse(ncclFunc_t const coll,
                                                size_t const count,
                                                ncclDataType_t const datatype,
                                                ncclRedOp_t const op,
                                                int const totalNumChannels,
                                                uint8_t* numChannelstoUse) const
{
  size_t const totalBytes = count * ncclTypeSize(datatype);
  *numChannelstoUse = 1;

  if (coll == ncclFuncAllReduce) {
    if (rcclParamAllReduceNumChannels() == 0)
    {
      // NOTE: These are currently based on collected data and not necessarily ideal for all hardware
      int numChannels;
      switch (m_gcnArch)
      {
      case 906:
        if      (totalBytes <=   16384) numChannels =  1;
        else                            numChannels =  2;
        break;
      case 908:
        if      (totalBytes <=  131072) numChannels =  2;
        else if (totalBytes <=  524288) numChannels =  6;
        else if (totalBytes <= 1048576) numChannels = 13;
        else                            numChannels = 16;
        break;
      case 910:
        if      (totalBytes <=  262144) numChannels =  4;
        else                            numChannels =  8;
        break;
      default:
        if      (totalBytes <=   65536) numChannels =  1;
        else if (totalBytes <=  262144) numChannels =  2;
        else if (totalBytes <=  524288) numChannels =  4;
        else if (totalBytes <= 2097152) numChannels =  8;
        else                            numChannels = 11;
      }
      *numChannelstoUse = std::min(numChannels, totalNumChannels);
    }
    else
    {
      *numChannelstoUse = std::min((int)rcclParamAllReduceNumChannels(), totalNumChannels);
    }
  }
  return ncclSuccess;
}

ncclResult_t CliqueManager::WaitForPointers(ncclWorkElem* args)
{
  // Do nothing if disabled
  if (m_cliqueMode == CLIQUE_DISABLED) return ncclSuccess;

  if (!m_init)
  {
    WARN("CliqueManager must be initialized before use");
    return ncclInvalidUsage;
  }

  // Check that collective queue is not empty
  if (m_opIndexHead == m_opIndexTail)
  {
    WARN("WaitForPointers must be called after DeclarePointers");
    return ncclInvalidUsage;
  }

  // Pop first collective off queue
  int32_t const opIndex = m_opIndexHead;
  INFO(NCCL_COLL, "Rank %d waiting for pointers for opIndex %d", m_rank, opIndex);

  m_opIndexHead = (m_opIndexHead + 1) % NCCL_MAX_OPS;
  args->clique.ptrs = &m_pinnedCliquePtrs[opIndex];

  // Wait for all ranks to declare pointers for this opIndex
  volatile int* entryCounter = (volatile int*)(&m_cpuBarrierCount[2 * opIndex]);
  int entryVal = LOAD(entryCounter);
  while (entryVal != m_numRanks) entryVal = LOAD(entryCounter);

  // Last rank to past barrier resets entry barrier
  // NOTE: There is another GPU-barrier performed during the kernels therefore it should
  //       not be possible for any rank to modify entry count prior to being reset
  volatile int* exitCounter = &m_cpuBarrierCount[2 * opIndex + 1];
  int exitVal = LOAD(exitCounter);
  // Loop until successful atomic update to counter
  bool done = false;
  while (done == false) {
    // Last rank resets entry counter
    if (exitVal+1 == m_numRanks)
      m_cpuBarrierCount[2 * opIndex] = 0;
    done = __sync_bool_compare_and_swap(exitCounter, exitVal, exitVal+1);
    exitVal++;
  }
  INFO(NCCL_COLL, "Rank %d past opIndex barrier %d", m_rank, opIndex);

  // Collect pointers
  if (m_cliqueMode == CLIQUE_SINGLE_NODE)
  {
    int numHandles = m_numRanks * NUM_HANDLES_PER_RANK;
    std::vector<std::pair<hipIpcMemHandle_t,size_t>> handles(numHandles);

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
  else if (m_cliqueMode == CLIQUE_SINGLE_PROCESS)
  {
    // Copy from static memory to pinned host memory and set local sense
    memcpy(&m_pinnedCliquePtrs[opIndex], &m_staticCliquePtrs[opIndex], sizeof(cliqueDevicePtrs_t));
    m_pinnedCliquePtrs[opIndex].barrier.localSense = &m_gpuBarrierLocalSense[opIndex];
  }
  return ncclSuccess;
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

  CUDACHECK(hipIpcGetMemHandle(&handlePair->first, (void*)baseAddr));

  /* Disabling cache until proper deallocation methods are available
  // IPC handles are only supported for base address pointers
  NcclIpcHandleSendCache::iterator it = cache->find(baseAddr);

   if (it == cache->end())
   {
     INFO(NCCL_COLL, "Rank %d searching IPC handle cache for %p (not found)", rank, devPtr);
     CUDACHECK(hipIpcGetMemHandle(&handlePair->first, (void*)baseAddr));
     cache->insert(baseAddr, handlePair->first);
   }
   else
   {
     INFO(NCCL_COLL, "Rank %d searching IPC handle cache for %p (found!)", rank, devPtr);
     handlePair->first = (it->second).first;
   }
  */
   return ncclSuccess;
}

ncclResult_t CliqueManager::CheckCacheForHandle(std::pair<hipIpcMemHandle_t, size_t> const& handlePair,
                                                NcclIpcHandleRecvCache* cache,
                                                void** ptr)
{
  // Until proper deallocation hooks are implemented, receive cache can not be used
  // Handles will need to be extract each time
  void* baseAddr;
  CUDACHECK(hipIpcOpenMemHandle(&baseAddr, handlePair.first, hipIpcMemLazyEnablePeerAccess));

  /*
  NcclIpcHandleRecvCache::iterator it = cache->find(handlePair.first);

  // Get base address pointer from cache if it exists

  if (it == cache->end())
  {
    CUDACHECK(hipIpcOpenMemHandle(&baseAddr, handlePair.first, hipIpcMemLazyEnablePeerAccess));
    cache->insert(handlePair.first, baseAddr);
  }
  else
  {
    baseAddr = (it->second).first;
  }
  */

  // Modify base address pointer with offset
  uint64_t realAddr = (uint64_t)baseAddr + handlePair.second;
  *ptr = (void*)realAddr;
  return ncclSuccess;
}

ncclResult_t CliqueManager::BootstrapRootInit(int pid, unsigned long hash)
{
  if (rcclParamEnableClique())
  {
      for (auto it = CliqueShmNames.begin(); it != CliqueShmNames.end(); it++)
      {
        mqd_t mq_desc;
        std::string msgQueueName = it->second + std::to_string(hash) + "_" + std::to_string(pid);
        NCCLCHECK(MsgQueueGetId(msgQueueName, true, mq_desc));
        NCCLCHECK(MsgQueueClose(msgQueueName, mq_desc, true));
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
  }
  else
  {
    INFO(NCCL_INIT, "Not performing bootstrap root for clique kernels as clique mode not enabled.");
  }
  return ncclSuccess;
}
