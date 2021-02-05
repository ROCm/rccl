/*************************************************************************
 * Copyright (c) 2019-2021 Advanced Micro Devices, Inc. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

#ifndef TEST_COMBINEDCALLS_MULTI_PROCESS_HPP
#define TEST_COMBINEDCALLS_MULTI_PROCESS_HPP

#include "CorrectnessTest.hpp"

#include "test_AllGatherMultiProcess.hpp"
#include "test_AllReduceMultiProcess.hpp"
#include "test_BroadcastMultiProcess.hpp"
#include "test_ReduceMultiProcess.hpp"
#include "test_ReduceScatterMultiProcess.hpp"

namespace CorrectnessTests
{
    class CombinedCallsMultiProcessCorrectnessTest : public MultiProcessCorrectnessTest
    {
    public:
        void TestCombinedCalls(int rank, std::vector<Dataset*>& datasets, std::vector<ncclFunc_t> const& funcs)
        {
            SetUpPerProcess(rank, funcs, comms[rank], streams[rank], datasets);

            if (numDevices > numDevicesAvailable) return;

            Barrier barrier(rank, numDevices, std::atoi(getenv("NCCL_COMM_ID")));

            // Compute expected results for each dataset in combined
            int const root = 0;
            AllGatherMultiProcessCorrectnessTest::ComputeExpectedResults(*datasets[0], barrier, rank, numDevices);
            AllReduceMultiProcessCorrectnessTest::ComputeExpectedResults(*datasets[1], barrier, op, rank);
            BroadcastMultiProcessCorrectnessTest::ComputeExpectedResults(*datasets[2], root, rank);
            ReduceMultiProcessCorrectnessTest::ComputeExpectedResults(*datasets[3], barrier, op, root, rank);
            ReduceScatterMultiProcessCorrectnessTest::ComputeExpectedResults(*datasets[4], barrier, op, rank);

            size_t const byteCount = datasets[0]->NumBytes() / numDevices;
            size_t const elemCount = numElements / numDevices;

            ncclAllGather((int8_t *)datasets[0]->inputs[rank] + (rank * byteCount),
                          datasets[0]->outputs[rank], elemCount,
                          dataType, comms[rank], streams[rank]);

            ncclAllReduce(datasets[1]->inputs[rank], datasets[1]->outputs[rank],
                          numElements, dataType, op, comms[rank], streams[rank]);

            ncclBroadcast(datasets[2]->inputs[rank],
                          datasets[2]->outputs[rank],
                          numElements, dataType,
                          root, comms[rank], streams[rank]);

            ncclReduce(datasets[3]->inputs[rank],
                       datasets[3]->outputs[rank],
                       numElements, dataType, op,
                       root, comms[rank], streams[rank]);

            ncclReduceScatter(datasets[4]->inputs[rank],
                              (int8_t *)datasets[4]->outputs[rank] + (rank * byteCount),
                              elemCount, dataType, op,
                              comms[rank], streams[rank]);

            // Wait for reduction to complete
            HIP_CALL(hipStreamSynchronize(streams[rank]));

            // Check results for each collective in the combined
            for (int i = 0; i < 5; i++)
            {
                ValidateResults(*datasets[i], rank);
                barrier.Wait();
                datasets[i]->Release(rank);
            }
        }
    };
}

#endif
