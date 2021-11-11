/*************************************************************************
 * Copyright (c) 2019-2020 Advanced Micro Devices, Inc. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

#include "test_ReduceScatter.hpp"

namespace CorrectnessTests
{
    TEST_P(ReduceScatterCorrectnessTest, Correctness)
    {
        if (numDevices > numDevicesAvailable) return;
        if (numElements % numDevices != 0) return;

        // Prepare input / output / expected results
        Dataset dataset;
        dataset.Initialize(numDevices, numElements, dataType, inPlace, ncclCollReduceScatter);
        FillDatasetWithPattern(dataset);
        ComputeExpectedResults(dataset, op);

        size_t const byteCount = dataset.NumBytes() / dataset.numDevices;
        size_t const recvCount = dataset.numElements / dataset.numDevices;

        // Launch the reduction (1 thread per GPU)
        ncclGroupStart();
        for (int i = 0; i < numDevices; i++)
        {
            ncclReduceScatter(dataset.inputs[i],
                              (int8_t *)dataset.outputs[i] + (i * byteCount),
                              recvCount, dataType, op,
                              comms[i], streams[i]);
        }
        ncclGroupEnd();

        // Wait for reduction to complete
        Synchronize();

        // Check results
        ValidateResults(dataset);

        dataset.Release();
    }

    INSTANTIATE_TEST_SUITE_P(ReduceScatterCorrectnessSweep,
                             ReduceScatterCorrectnessTest,
                             testing::Combine(
                                // Reduction operator
                                testing::Values(ncclSum, ncclProd, ncclMax, ncclMin, ncclAvg),
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
                                testing::Range(2,(GTESTS_NUM_GPUS+1)),
                                // In-place or not
                                testing::Values(false, true),
                                testing::Values("")),
                             CorrectnessTest::PrintToStringParamName());
} // namespace
