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

#ifndef NCCL_IPC_HANDLE_SHM_H_
#define NCCL_IPC_HANDLE_SHM_H_

#include <hip/hip_runtime.h>
#include <vector>
#include <string>

#include "nccl.h"
#include "ShmObject.h"

class NcclIpcHandleShm : public ShmObject<std::pair<hipIpcMemHandle_t,size_t>>
{
public:
    NcclIpcHandleShm(int rank, int numRanks, int projid, int numHandlesPerRank, int capacity, std::string const& suffix);

    NcclIpcHandleShm();

    ~NcclIpcHandleShm();

    ncclResult_t Open();

    ncclResult_t WriteHandles(uint64_t opCount, std::vector<std::pair<hipIpcMemHandle_t,size_t>> const& sendHandles);

    ncclResult_t ReadHandles(uint64_t opCount, std::vector<std::pair<hipIpcMemHandle_t,size_t>>& recvHandles);

private:
    int m_numHandlesPerRank;
    int m_numHandlesPerOpCount;
};

#endif
