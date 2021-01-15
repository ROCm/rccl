/*************************************************************************
 * Copyright (c) 2019-2021 Advanced Micro Devices, Inc. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/
#ifndef TEST_BROADCAST_MULTI_PROCESS_HPP
#define TEST_BROADCAST_MULTI_PROCESS_HPP

#include "CorrectnessTest.hpp"

namespace CorrectnessTests
{
    class BroadcastMultiProcessCorrectnessTest : public MultiProcessCorrectnessTest
    {
    public:
        static void ComputeExpectedResults(Dataset& dataset, int const root, int const rank)
        {
            // Root has the answer; share it via host memcpy's
            if (rank == root)
            {
                HIP_CALL(hipMemcpy(dataset.expected[rank], dataset.inputs[rank],
                                   dataset.NumBytes(), hipMemcpyDeviceToHost));
                for (int i = 0; i < dataset.numDevices; i++)
                {
                    if (i == rank) continue;
                    memcpy(dataset.expected[i], dataset.expected[root], dataset.NumBytes());
                }
            }
        }

        void TestBroadcast(int rank, Dataset& dataset)
        {
            SetUpPerProcess(rank, ncclCollBroadcast, comms[rank], streams[rank], dataset);

            if (numDevices > numDevicesAvailable) return;

            Barrier barrier(rank, numDevices, std::atoi(getenv("NCCL_COMM_ID")));

            // Test each possible root
            for (int root = 0; root < numDevices; root++)
            {
                // Prepare input / output / expected results
                FillDatasetWithPattern(dataset, rank);
                ComputeExpectedResults(dataset, root, rank);

                // Launch the reduction (1 process per GPU)
                ncclResult_t res = ncclBroadcast(dataset.inputs[rank],
                              dataset.outputs[rank],
                              numElements, dataType,
                              root, comms[rank], streams[rank]);

                // Wait for reduction to complete
                HIP_CALL(hipStreamSynchronize(streams[rank]));

                // Check results
                ValidateResults(dataset, rank);

                // Ensure all processes have finished current iteration before proceeding
                barrier.Wait();
            }

            TearDownPerProcess(comms[rank], streams[rank]);
            dataset.Release(rank);
        }
    };
}

#endif
