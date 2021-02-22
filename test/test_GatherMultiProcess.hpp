/*************************************************************************
 * Copyright (c) 2019-2021 Advanced Micro Devices, Inc. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/
#ifndef TEST_GATHER_MULTI_PROCESS_HPP
#define TEST_GATHER_MULTI_PROCESS_HPP

#include "CorrectnessTest.hpp"

namespace CorrectnessTests
{
    class GatherMultiProcessCorrectnessTest : public MultiProcessCorrectnessTest
    {
    public:
        static void ComputeExpectedResults(Dataset& dataset, int const root, int const rank)
        {
            HIP_CALL(hipMemcpy((int8_t *)dataset.expected[root]+dataset.NumBytes()*rank, dataset.inputs[rank],
                               dataset.NumBytes(), hipMemcpyDeviceToHost));
        }

        void TestGather(int rank, Dataset& dataset, bool& pass)
        {
            SetUpPerProcess(rank, ncclCollGather, comms[rank], streams[rank], dataset);

            if (numDevices > numDevicesAvailable)
            {
                pass = true;
                return;
            }

            Barrier barrier(rank, numDevices, std::atoi(getenv("NCCL_COMM_ID")));

            // Test each possible root
            for (int root = 0; root < numDevices; root++)
            {
                // Prepare input / output / expected results
                FillDatasetWithPattern(dataset, rank);
                ComputeExpectedResults(dataset, root, rank);

                // Launch the reduction (1 process per GPU)
                ncclGather(dataset.inputs[rank],
                           dataset.outputs[rank],
                           numElements, dataType,
                           root, comms[rank], streams[rank]);

                // Wait for reduction to complete
                HIP_CALL(hipStreamSynchronize(streams[rank]));

                // Check results
                pass = ValidateResults(dataset, rank, root);

                // Ensure all processes have finished current iteration before proceeding
                barrier.Wait();
            }

            TearDownPerProcess(comms[rank], streams[rank]);
            dataset.Release(rank);
        }
    };
}

#endif
