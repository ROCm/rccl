/*************************************************************************
 * Copyright (c) 2021 Advanced Micro Devices, Inc. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/
#include "test_AllReduceGroupMultiProcess.hpp"

namespace CorrectnessTests
{
    TEST_P(AllReduceGroupMultiProcessCorrectnessTest, Correctness)
    {
        // Important: Make sure the order of ncclFunc_t's here match the order of ncclFunc_ts
        // as they appear in TestGroupCalls()
        std::vector<ncclFunc_t> ncclFuncs;
        ncclFuncs.push_back(ncclCollAllReduce);
        ncclFuncs.push_back(ncclCollAllReduce);
        ncclFuncs.push_back(ncclCollAllReduce);

        // Create multiple datasets for combined operation
        std::vector<Dataset*> datasets(ncclFuncs.size());
        for (int i = 0; i < datasets.size(); i++)
        {
            datasets[i] = (Dataset*)mmap(NULL, sizeof(Dataset), PROT_READ|PROT_WRITE, MAP_SHARED|MAP_ANONYMOUS, -1, 0);
            datasets[i]->InitializeRootProcess(numDevices, numElements, dataType, inPlace, ncclFuncs[i]);
        }

        int const numGpusPerProcess = 2;
        int const numProcesses = numDevices / numGpusPerProcess;
        std::vector<int> pids(numProcesses);
        int process = -1;

        for (int i = 0; i < numDevices; i+= numGpusPerProcess)
        {
            process++;
            int pid = fork();
            if (pid == 0)
            {
                int gpuIdx = i;
                int maxIdx = gpuIdx + (numGpusPerProcess - 1) >= numDevices ? numDevices : gpuIdx + numGpusPerProcess;

                std::vector<int> ranks;
                for (; gpuIdx < maxIdx; gpuIdx++)
                {
                    ranks.push_back(gpuIdx);
                }

                bool pass;
                TestGroupCalls(process, ranks, datasets, ncclFuncs, pass);
                TerminateChildProcess(pass);
            }
            else
            {
                pids[process] = pid;
            }
        }

        ValidateProcesses(pids);

        for (int i = 0; i < datasets.size(); i++)
        {
            munmap(datasets[i], sizeof(Dataset));
        }
    }

    INSTANTIATE_TEST_SUITE_P(AllReduceGroupMultiProcessCorrectnessSweep,
                            AllReduceGroupMultiProcessCorrectnessTest,
                            testing::Combine(
                                // Reduction operator (not used)
                                testing::Values(ncclSum),
                                // Data types
                                testing::Values(ncclFloat32,
                                                ncclFloat64),
                                // Number of elements
                                testing::Values(3072, 3145728),
                                // Number of devices
                                testing::Values(4,8),
                                // In-place or not
                                testing::Values(false, true),
                                testing::Values("RCCL_ENABLE_CLIQUE=0", "RCCL_ENABLE_CLIQUE=1")),
                            CorrectnessTest::PrintToStringParamName());
} // namespace
