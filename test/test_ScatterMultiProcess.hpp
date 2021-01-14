/*************************************************************************
 * Copyright (c) 2019-2021 Advanced Micro Devices, Inc. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/
#ifndef TEST_SCATTER_MULTI_PROCESS_HPP
#define TEST_SCATTER_MULTI_PROCESS_HPP

#include "CorrectnessTest.hpp"

namespace CorrectnessTests
{
    class ScatterMultiProcessCorrectnessTest : public MultiProcessCorrectnessTest
    {
    public:
        static void ComputeExpectedResults(Dataset& dataset, int const root, int const rank)
        {
            if (rank == root)
            {
                for (int i = 0; i < dataset.numDevices; i++)
                    HIP_CALL(hipMemcpy(dataset.expected[i], (int8_t *)dataset.inputs[root]+dataset.NumBytes()*i,
                                       dataset.NumBytes(), hipMemcpyDeviceToHost));
            }
        }

        void TestScatter(int rank, Dataset& dataset)
        {
            // Prepare input / output / expected results
            SetUpPerProcess(rank, ncclCollScatter, comms[rank], streams[rank], dataset);

            if (numDevices > numDevicesAvailable) return;

            Barrier barrier(rank, numDevices, std::atoi(getenv("NCCL_COMM_ID")));

            // Test each possible root
            for (int root = 0; root < numDevices; root++)
            {
                // Prepare input / output / expected results
                FillDatasetWithPattern(dataset, rank);

                ComputeExpectedResults(dataset, root, rank);

                // Launch the reduction (1 process per GPU)
                ncclScatter(dataset.inputs[rank],
                            dataset.outputs[rank],
                            numElements, dataType,
                            root, comms[rank], streams[rank]);

                // Wait for reduction to complete
                HIP_CALL(hipStreamSynchronize(streams[rank]));

                // Check results
                ValidateResults(dataset, rank);

                barrier.Wait();
            }

            TearDownPerProcess(comms[rank], streams[rank]);
            dataset.Release(rank);
        }
    };
}

#endif
