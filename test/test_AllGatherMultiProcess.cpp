/*************************************************************************
 * Copyright (c) 2019-2021 Advanced Micro Devices, Inc. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/
#include "test_AllGatherMultiProcess.hpp"

namespace CorrectnessTests
{
    TEST_P(AllGatherMultiProcessCorrectnessTest, Correctness)
    {
        dataset->InitializeRootProcess(numDevices, numElements, dataType, inPlace, ncclCollAllGather);
        std::vector<int> pids(numDevices);

        int gpu = -1;
        for (int i = 0; i < numDevices; i++)
        {
            gpu++;
            int pid = fork();
            if (pid == 0)
            {
                bool pass;
                TestAllGather(gpu, *dataset, pass);
                TerminateChildProcess(pass);
            }
            else
            {
                pids[gpu] = pid;
            }
        }

        ValidateProcesses(pids);
        dataset->ReleaseRootProcess();
    }

    INSTANTIATE_TEST_SUITE_P(AllGatherMultiProcessCorrectnessSweep,
                            AllGatherMultiProcessCorrectnessTest,
                            testing::Combine(
                                // Reduction operator (not used)
                                testing::Values(ncclSum),
                                // Data types
                                testing::Values(ncclInt8,
                                                ncclUint8,
                                                ncclInt32,
                                                ncclUint32,
                                                ncclInt64,
                                                ncclUint64,
                                                //ncclFloat16,
                                                ncclFloat32,
                                                ncclFloat64,
                                                ncclBfloat16),
                                // Number of elements
                                testing::Values(3072, 3145728),
                                // Number of devices
                                testing::Values(2,3,4,8),
                                // In-place or not
                                testing::Values(false, true),
                                testing::Values("")),
                            CorrectnessTest::PrintToStringParamName());
} // namespace
