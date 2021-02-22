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
        static void ComputeExpectedResults(Dataset& dataset, std::vector<int> const& ranks)
        {
            for (int i = 0; i < ranks.size(); i++)
            {
                int rank = ranks[i];
                for (int j = 0; j < dataset.numDevices; j++)
                {
                    HIP_CALL(hipMemcpy((int8_t *)dataset.expected[j]+dataset.NumBytes()*rank, (int8_t *)dataset.inputs[rank]+dataset.NumBytes()*j,
                                   dataset.NumBytes(), hipMemcpyDeviceToHost));
                }
            }
        }

        void TestAllToAll(int rank, Dataset& dataset, bool& pass)
        {
            SetUpPerProcess(rank, ncclCollAllToAll, comms[rank], streams[rank], dataset);

            if (numDevices > numDevicesAvailable)
            {
                pass = true;
                return;
            }

            // Prepare input / output / expected results
            FillDatasetWithPattern(dataset, rank);
            ComputeExpectedResults(dataset, std::vector<int>(1, rank));

            // Launch the reduction
            ncclAllToAll(dataset.inputs[rank],
                         dataset.outputs[rank],
                         numElements, dataType,
                         comms[rank], streams[rank]);

            // Wait for reduction to complete
            HIP_CALL(hipStreamSynchronize(streams[rank]));

            // Check results
            pass = ValidateResults(dataset, rank);

            TearDownPerProcess(comms[rank], streams[rank]);
            dataset.Release(rank);
        }
    };
}

#endif
