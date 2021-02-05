/*************************************************************************
 * Copyright (c) 2019-2021 Advanced Micro Devices, Inc. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/
#ifndef TEST_ALLTOALL_MULTI_PROCESS_HPP
#define TEST_ALLTOALL_MULTI_PROCESS_HPP

#include "CorrectnessTest.hpp"

namespace CorrectnessTests
{
    class AllToAllMultiProcessCorrectnessTest : public MultiProcessCorrectnessTest
    {
    public:
        static void ComputeExpectedResults(Dataset& dataset, int const rank)
        {
            for (int i = 0; i < dataset.numDevices; i++)
            {
                HIP_CALL(hipMemcpy((int8_t *)dataset.expected[i]+dataset.NumBytes()*rank, (int8_t *)dataset.inputs[rank]+dataset.NumBytes()*i,
                               dataset.NumBytes(), hipMemcpyDeviceToHost));
            }
        }

        void TestAllToAll(int rank, Dataset& dataset)
        {
            SetUpPerProcess(rank, ncclCollAllToAll, comms[rank], streams[rank], dataset);

            if (numDevices > numDevicesAvailable) return;

            // Prepare input / output / expected results
            FillDatasetWithPattern(dataset, rank);
            ComputeExpectedResults(dataset, rank);

            // Launch the reduction
            ncclAllToAll(dataset.inputs[rank],
                         dataset.outputs[rank],
                         numElements, dataType,
                         comms[rank], streams[rank]);

            // Wait for reduction to complete
            HIP_CALL(hipStreamSynchronize(streams[rank]));

            // Check results
            ValidateResults(dataset, rank);

            TearDownPerProcess(comms[rank], streams[rank]);
            dataset.Release(rank);
        }
    };
}

#endif
