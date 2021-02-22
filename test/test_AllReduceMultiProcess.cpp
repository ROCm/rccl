/*************************************************************************
 * Copyright (c) 2019-2021 Advanced Micro Devices, Inc. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

#include "test_AllReduceMultiProcess.hpp"

namespace CorrectnessTests
{
    TEST_P(AllReduceMultiProcessCorrectnessTest, Correctness)
    {
        dataset->InitializeRootProcess(numDevices, numElements, dataType, inPlace, ncclCollAllReduce);
        std::vector<int> pids(numDevices);

        int gpu = -1;
        for (int i = 0; i < numDevices; i++)
        {
            gpu++;
            int pid = fork();
            if (pid == 0)
            {
                bool pass;
                TestAllReduce(gpu, *dataset, pass);
                TerminateChildProcess(pass);
            }
            else
            {
                pids[gpu] = pid;
            }
        }

        ValidateProcesses(pids);
    }

    INSTANTIATE_TEST_SUITE_P(AllReduceMultiProcessCorrectnessSweep,
                            AllReduceMultiProcessCorrectnessTest,
                            testing::Combine(
                                // Reduction operator
                                testing::Values(ncclSum, ncclProd, ncclMax, ncclMin),
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
                                testing::Values(1024, 1048576),
                                // Number of devices
                                testing::Values(2,3,4,8),
                                // In-place or not
                                testing::Values(false, true),
                                testing::Values("")),
                            CorrectnessTest::PrintToStringParamName());
} // namespace
