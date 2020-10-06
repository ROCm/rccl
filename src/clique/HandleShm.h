#ifndef NCCL_IPC_HANDLE_SHM_H_
#define NCCL_IPC_HANDLE_SHM_H_

#include <hip/hip_runtime.h>
#include <vector>
#include <string>

#include "nccl.h"
#include "ShmObject.h"

class NcclIpcHandleShm : public ShmObject<hipIpcMemHandle_t>
{
public:
    NcclIpcHandleShm(int rank, int numRanks, int projid, int numHandlesPerRank, int capacity, std::string suffix);

    NcclIpcHandleShm();

    ~NcclIpcHandleShm();

    ncclResult_t Open();

    ncclResult_t WriteHandles(uint64_t opCount, std::vector<hipIpcMemHandle_t> const& sendHandles);

    ncclResult_t ReadHandles(uint64_t opCount, std::vector<hipIpcMemHandle_t>& recvHandles);

private:
    int m_numHandlesPerRank;
    int m_numHandlesPerOpCount;
};

#endif