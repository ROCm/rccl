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

#include <hip/hip_runtime.h>

#include "HandleShm.h"
#include "CliqueShmNames.h"
#include "core.h"
#include "Hash.h"
#include "shm.h"

NcclIpcHandleShm::NcclIpcHandleShm(int rank, int numRanks, int projid, int numHandlesPerRank, int capacity, std::string const& suffix) :
  ShmObject<std::pair<hipIpcMemHandle_t,size_t>>(numRanks * numHandlesPerRank * capacity * sizeof(std::pair<hipIpcMemHandle_t,size_t>),
                                                 CliqueShmNames["IpcHandles"] + suffix,
                                                 rank,
                                                 numRanks,
                                                 projid),
  m_numHandlesPerRank(numHandlesPerRank),
  m_numHandlesPerOpCount(numRanks * numHandlesPerRank)
{
}

NcclIpcHandleShm::NcclIpcHandleShm() :
  m_numHandlesPerRank(0),
  m_numHandlesPerOpCount(0)
{
}

NcclIpcHandleShm::~NcclIpcHandleShm()
{
}

ncclResult_t NcclIpcHandleShm::Open()
{
  return ShmObject::Open();
}

ncclResult_t NcclIpcHandleShm::WriteHandles(uint64_t opCount, std::vector<std::pair<hipIpcMemHandle_t,size_t>> const& sendHandles)
{
  size_t idx = (opCount * m_numHandlesPerOpCount) + (m_rank *  m_numHandlesPerRank);
  memcpy(m_shmPtr + idx, sendHandles.data(), sizeof(std::pair<hipIpcMemHandle_t,size_t>) * m_numHandlesPerRank);
  return ncclSuccess;
}

ncclResult_t NcclIpcHandleShm::ReadHandles(uint64_t opCount, std::vector<std::pair<hipIpcMemHandle_t,size_t>>& recvHandles)
{
  size_t idx = opCount * m_numHandlesPerOpCount;
  memcpy(recvHandles.data(), m_shmPtr + idx, m_numHandlesPerOpCount * sizeof(std::pair<hipIpcMemHandle_t,ssize_t>));
  return ncclSuccess;
}
