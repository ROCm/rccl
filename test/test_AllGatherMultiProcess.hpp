/*************************************************************************
 * Copyright (c) 2019-2021 Advanced Micro Devices, Inc. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/
#ifndef TEST_ALLGATHER_MULTI_PROCESS_HPP
#define TEST_ALLGATHER_MULTI_PROCESS_HPP

#include "CorrectnessTest.hpp"

namespace CorrectnessTests
{
    class AllGatherMultiProcessCorrectnessTest : public MultiProcessCorrectnessTest
    {
    public:
        static void ComputeExpectedResults(Dataset& dataset, Barrier& barrier, int const numDevices, std::vector<int> const& ranks)
        {
            size_t const byteCount = dataset.NumBytes() / dataset.numDevices;

            for (int i = 0; i < ranks.size(); i++)
            {
                int rank = ranks[i];
                HIP_CALL(hipMemcpy(static_cast<char*>(dataset.expected[0]) + rank * byteCount, (int8_t *)dataset.inputs[rank] + (rank * byteCount),
                                   byteCount, hipMemcpyDeviceToHost));
            }
            barrier.Wait();

            // Rank 0 sends answer to other ranks
            for (int i = 0; i < ranks.size(); i++)
            {
                int rank = ranks[i];
                if (rank == 0)
                {
                    for (int i = 0; i < dataset.numDevices; i++)
                    {
                        if (i == rank) continue;
                        memcpy(dataset.expected[i], dataset.expected[0], dataset.NumBytes());
                    }
                }
            }
        }

        void TestAllGather(int rank, Dataset& dataset, bool& pass)
        {
            // Prepare input / output / expected results
            SetUpPerProcess(rank, ncclCollAllGather, comms[rank], streams[rank], dataset);

            if (numDevices > numDevicesAvailable || numElements % numDevices != 0)
            {
                pass = true;
                return;
            }

            Barrier barrier(rank, numDevices, std::atoi(getenv("NCCL_COMM_ID")));

            // Prepare input / output / expected results
            FillDatasetWithPattern(dataset, rank);

            ComputeExpectedResults(dataset, barrier, numDevices, std::vector<int>(1, rank));

            size_t const byteCount = dataset.NumBytes() / numDevices;
            size_t const sendCount = dataset.numElements / numDevices;

            // Launch the reduction (1 process per GPU)
            ncclAllGather((int8_t *)dataset.inputs[rank] + (rank * byteCount),
                          dataset.outputs[rank], sendCount,
                          dataType, comms[rank], streams[rank]);

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
