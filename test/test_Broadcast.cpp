/*************************************************************************
 * Copyright (c) 2019 Advanced Micro Devices, Inc. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

#include "test_Broadcast.hpp"
#include <omp.h>

namespace CorrectnessTests
{
    TEST_P(BroadcastCorrectnessTest, Correctness)
    {
        if (numDevices > numDevicesAvailable) return;

        // Allocate data
        Dataset dataset;
        dataset.Initialize(numDevices, numElements, dataType, inPlace);

        // Test each possible root
        for (int root = 0; root < numDevices; root++)
        {
            // Prepare input / output / expected results
            FillDatasetWithPattern(dataset);
            ComputeExpectedResults(dataset, root);

            // Launch the reduction (1 thread per GPU)
            #pragma omp parallel for num_threads(numDevices)
            for (int i = 0; i < numDevices; i++)
            {
                ncclBroadcast(dataset.inputs[i],
                              dataset.outputs[i],
                              numElements, dataType,
                              root, comms[i], streams[i]);
            }


            // Wait for reduction to complete
            Synchronize();

            // Check results
            ValidateResults(dataset);
        }

        dataset.Release();
    }

    INSTANTIATE_TEST_CASE_P(BroadcastCorrectnessSweep,
                            BroadcastCorrectnessTest,
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
                                testing::Values(2,3,4),
                                // In-place or not
                                testing::Values(false, true)));
} // namespace
