#include <hip/hip_runtime.h>

#include "HandleShm.h"
#include "CliqueShmNames.h"
#include "core.h"
#include "Hash.h"
#include "shm.h"

NcclIpcHandleShm::NcclIpcHandleShm(int rank, int numRanks, int projid, int numHandlesPerRank, int capacity, std::string suffix) :
    ShmObject<hipIpcMemHandle_t>(numRanks * numHandlesPerRank * capacity * sizeof(hipIpcMemHandle_t),
                                 CliqueShmNames["IpcHandles"] + suffix,
                                 rank,
                                 numRanks,
                                 projid),
    m_numHandlesPerRank(numHandlesPerRank),
    m_numHandlesPerOpCount(numRanks * numHandlesPerRank)
    {
    }

NcclIpcHandleShm::NcclIpcHandleShm()
{
}

NcclIpcHandleShm::~NcclIpcHandleShm()
{
}

ncclResult_t NcclIpcHandleShm::Open()
{
    return ShmObject::Open();
}

ncclResult_t NcclIpcHandleShm::WriteHandles(uint64_t opCount, std::vector<hipIpcMemHandle_t> const& sendHandles)
{
    size_t idx = (opCount * m_numHandlesPerOpCount) + (m_rank *  m_numHandlesPerRank);
    memcpy(m_shmPtr + idx, sendHandles.data(), sizeof(hipIpcMemHandle_t) * m_numHandlesPerRank);
    return ncclSuccess;
}

ncclResult_t NcclIpcHandleShm::ReadHandles(uint64_t opCount, std::vector<hipIpcMemHandle_t>& recvHandles)
{
    size_t idx = opCount * m_numHandlesPerOpCount;
    memcpy(recvHandles.data(), m_shmPtr + idx, m_numHandlesPerOpCount * sizeof(hipIpcMemHandle_t));
    return ncclSuccess;
}