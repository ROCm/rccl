/*************************************************************************
 * Copyright (c) 2019-2020 Advanced Micro Devices, Inc. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

#include "test_AllToAll.hpp"

namespace CorrectnessTests
{
    TEST_P(AllToAllCorrectnessTest, Correctness)
    {
        if (numDevices > numDevicesAvailable) return;

        // Allocate data
        Dataset dataset;
        dataset.Initialize(numDevices, numElements, dataType, inPlace, ncclCollAllToAll);

        // Prepare input / output / expected results
        FillDatasetWithPattern(dataset);
        ComputeExpectedResults(dataset);

        // Launch the reduction (1 thread per GPU)
        ncclGroupStart();
        for (int i = 0; i < numDevices; i++)
        {
            ncclAllToAll(dataset.inputs[i],
                          dataset.outputs[i],
                          numElements, dataType,
                          comms[i], streams[i]);
        }
        ncclGroupEnd();

        // Wait for reduction to complete
        Synchronize();

        // Check results
        ValidateResults(dataset);

        dataset.Release();
    }

    INSTANTIATE_TEST_SUITE_P(AllToAllCorrectnessSweep,
                            AllToAllCorrectnessTest,
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
                                testing::Range(2,(GTESTS_NUM_GPUS+1)),
                                // In-place or not
                                testing::Values(false),
                                testing::Values("")),
                            CorrectnessTest::PrintToStringParamName());
} // namespace
