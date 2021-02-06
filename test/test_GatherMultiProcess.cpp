/*************************************************************************
 * Copyright (c) 2019-2021 Advanced Micro Devices, Inc. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

#include "test_GatherMultiProcess.hpp"

namespace CorrectnessTests
{
    TEST_P(GatherMultiProcessCorrectnessTest, Correctness)
    {
        dataset->InitializeRootProcess(numDevices, numElements, dataType, inPlace, ncclCollGather);
        std::vector<int> pids(numDevices);

        int gpu = -1;
        for (int i = 0; i < numDevices; i++)
        {
            gpu++;
            int pid = fork();
            if (pid == 0)
            {
                bool pass;
                TestGather(gpu, *dataset, pass);
                TerminateChildProcess(pass);
            }
            else
            {
                pids[gpu] = pid;
            }
        }

        ValidateProcesses(pids);
    }

    INSTANTIATE_TEST_SUITE_P(GatherMultiProcessCorrectnessSweep,
                            GatherMultiProcessCorrectnessTest,
                            testing::Combine(
                                // Reduction operator is not used
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
                                testing::Values(1024, 1048576),
                                // Number of devices
                                testing::Values(2,3,4,8),
                                // In-place or not
                                testing::Values(false),
                                testing::Values("RCCL_ALLTOALL_KERNEL_DISABLE=0", "RCCL_ALLTOALL_KERNEL_DISABLE=1")),
                            CorrectnessTest::PrintToStringParamName());
} // namespace
