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

  // Launch a clique based kernel
  ncclResult_t QueueKernel(uint64_t       const opCount,
                           ncclFunc_t     const coll,
                           size_t         const count,
                           ncclDataType_t const datatype,
                           ncclRedOp_t    const op,
                           int            const root,
                           hipStream_t    const stream);

  ncclResult_t CloseSharedMemory();

  static ncclResult_t BootstrapRootInit(int pid, unsigned long hash);

protected:
  // Collect the device pointers from all GPUs for specified opCount
  ncclResult_t GetCliqueDevicePointers(uint64_t opCount, cliqueDevicePtrs_t& cliquePtrs);


  ncclResult_t CheckCacheForPtr(void* devPtr,
                                NcclIpcHandleSendCache* cache,
                                int rank,
                                hipIpcMemHandle_t* handle);

  ncclResult_t CheckCacheForHandle(hipIpcMemHandle_t handle,
                                   NcclIpcHandleRecvCache* cache,
                                   void** ptr);

  // Race-condition helper functions
  void WaitForBarrier(int opIndex);

  cliqueMode_t  m_cliqueMode;
  int           m_rank;
  int           m_numRanks;
  bool          m_init;
  int           m_nextBarrierValue[NCCL_MAX_OPS];
  uint32_t*     m_arrivalCounter;

  // IPC-related (CLIQUE_SINGLE_NODE)
  NcclIpcHandleShm             m_shmHandles;
  NcclIpcHandleSendCache*      m_ipcHandleSendCache;
  NcclIpcHandleRecvCache*      m_ipcHandleRecvCache;
  ShmObject<uint32_t>          m_sharedCounters; // Tracks # of ranks that have finished declaring pointers
  ShmObject<hipIpcMemHandle_t> m_sharedBarrier;  // Used to pass fine-grained device memory buffer
  int*                         m_deviceBarriers; // fine-grained barrier

  // Single-process (CLIQUE_SINGLE_PROCESS)
  static cliqueDevicePtrs_t    m_cliquePtrs[NCCL_MAX_OPS];
  static uint32_t              m_staticCounters[NCCL_MAX_OPS];
  static int*                  m_staticBarriers;
};

// For use in bootstrapping code
struct bootstrapRootStruct {
    void* listenComm;
    unsigned long hash;
};

#endif
