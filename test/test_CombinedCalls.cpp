/*************************************************************************
 * Copyright (c) 2019-2020 Advanced Micro Devices, Inc. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/
#include "test_CombinedCalls.hpp"

#include "test_AllGather.hpp"
#include "test_AllReduce.hpp"
#include "test_Broadcast.hpp"
#include "test_Reduce.hpp"
#include "test_ReduceScatter.hpp"
#include "test_Scatter.hpp"

namespace CorrectnessTests
{
    TEST_P(CombinedCallsCorrectnessTest, Correctness)
    {
        if (numDevices > numDevicesAvailable) return;

        // Create multiple datasets for combined operation
        std::vector<Dataset> datasets(5);
        for (int i = 0; i < datasets.size(); i++)
        {
            datasets[i].Initialize(numDevices, numElements, dataType, inPlace);
            FillDatasetWithPattern(datasets[i]);
        }

        Dataset scatter_dataset;
        scatter_dataset.Initialize(numDevices, numElements, dataType, inPlace, ncclCollScatter);
        FillDatasetWithPattern(scatter_dataset);

        // Compute expected results for each dataset in combined
        int const root = 0;
        AllGatherCorrectnessTest::ComputeExpectedResults(datasets[0]);
        AllReduceCorrectnessTest::ComputeExpectedResults(datasets[1], op);
        BroadcastCorrectnessTest::ComputeExpectedResults(datasets[2], root);
        ReduceCorrectnessTest::ComputeExpectedResults(datasets[3], op, root);
        ReduceScatterCorrectnessTest::ComputeExpectedResults(datasets[4], op);
        ScatterCorrectnessTest::ComputeExpectedResults(scatter_dataset, root);

        size_t const byteCount = datasets[0].NumBytes() / numDevices;
        size_t const elemCount = numElements / numDevices;

        for (int j = 0; j < 10; j++)
        {
            ncclGroupStart();
            for (int i = 0; i < numDevices; i++)
            {
                ncclScatter(scatter_dataset.inputs[i],
                              scatter_dataset.outputs[i],
                              numElements, dataType,
                              root, comms[i], streams[i]);
            }
            ncclGroupEnd();
            ncclGroupStart();
            for (int i = 0; i < numDevices; i++)
            {
                ncclAllGather((int8_t *)datasets[0].inputs[i] + (i * byteCount),
                              datasets[0].outputs[i], elemCount,
                              dataType, comms[i], streams[i]);

                ncclAllReduce(datasets[1].inputs[i], datasets[1].outputs[i],
                              numElements, dataType, op, comms[i], streams[i]);

                ncclBroadcast(datasets[2].inputs[i],
                              datasets[2].outputs[i],
                              numElements, dataType,
                              root, comms[i], streams[i]);

                ncclReduce(datasets[3].inputs[i],
                           datasets[3].outputs[i],
                           numElements, dataType, op,
                           root, comms[i], streams[i]);

                ncclReduceScatter(datasets[4].inputs[i],
                                  (int8_t *)datasets[4].outputs[i] + (i * byteCount),
                                  elemCount, dataType, op,
                                  comms[i], streams[i]);
            }
            ncclGroupEnd();
            // Wait for reduction to complete
            Synchronize();
            // Check results for each collective in the combined
            for (int i = 0; i < 5; i++)
                ValidateResults(datasets[i]);

            ValidateResults(scatter_dataset);
        }

        for (int i = 0; i < 5; i++)
            datasets[i].Release();
        scatter_dataset.Release();
    }

    INSTANTIATE_TEST_SUITE_P(CombinedCallsCorrectnessSweep,
                            CombinedCallsCorrectnessTest,
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
                                testing::Values(2520, 3026520),
                                // Number of devices
                                testing::Values(2,3,4,5,6,7,8),
                                // In-place or not
                                testing::Values(false),
                                testing::Values("RCCL_ENABLE_CLIQUE=0", "RCCL_ENABLE_CLIQUE=1", "RCCL_P2P_NET_DISABLE=0", "RCCL_P2P_NET_DISABLE=1")),
                            CorrectnessTest::PrintToStringParamName());
} // namespace
