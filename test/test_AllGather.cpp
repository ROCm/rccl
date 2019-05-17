/*************************************************************************
 * Copyright (c) 2019 Advanced Micro Devices, Inc. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/
#include "test_AllGather.hpp"
#include <omp.h>

namespace CorrectnessTests
{
    TEST_P(AllGatherCorrectnessTest, Correctness)
    {
        if (numDevices > numDevicesAvailable) return;
        if (numElements % numDevices != 0) return;

        // Prepare input / output / expected results
        Dataset dataset;
        dataset.Initialize(numDevices, numElements, dataType, inPlace);
        FillDatasetWithPattern(dataset);
        ComputeExpectedResults(dataset);

        size_t const byteCount = dataset.NumBytes() / dataset.numDevices;
        size_t const sendCount = dataset.numElements / dataset.numDevices;

        // Launch the reduction (1 thread per GPU)
        #pragma omp parallel for num_threads(numDevices)
        for (int i = 0; i < numDevices; i++)
        {
            ncclAllGather((int8_t *)dataset.inputs[i] + (i * byteCount),
                          dataset.outputs[i], sendCount,
                          dataType, comms[i], streams[i]);
        }

        // Wait for reduction to complete
        Synchronize();

        // Check results
        ValidateResults(dataset);
        dataset.Release();
    }

    TEST_P(AllGatherCorrectnessTest, Alignment)
    {
        if (numDevices > numDevicesAvailable) return;
        if (numElements % numDevices != 0) return;

        // Allocate dataset
        Dataset dataset;
        dataset.Initialize(numDevices, numElements, dataType, inPlace);

        // Loop over several offsets (so that device pointers are not aligned)
        for (int firstElement = 1; firstElement <= 11; firstElement += 2)
        {
            if (firstElement < numElements)
            {
                // Select last element so that total number of elements is multiple of numDevices
                int const lastElement = firstElement + ((numElements - firstElement) / numDevices) * numDevices - 1;
                if (lastElement >= numElements) break;

                Dataset subDataset;
                dataset.ExtractSubDataset(firstElement, lastElement, subDataset);

                // Compute reference results for sub-dataset
                FillDatasetWithPattern(subDataset);
                ComputeExpectedResults(subDataset);

                size_t const byteCount = subDataset.NumBytes() / subDataset.numDevices;
                size_t const sendCount = subDataset.numElements / subDataset.numDevices;

                // Launch the reduction (1 thread per GPU)
                #pragma omp parallel for num_threads(numDevices)
                for (int i = 0; i < numDevices; i++)
                {
                    ncclAllGather((int8_t *)subDataset.inputs[i] + (i * byteCount),
                                  subDataset.outputs[i], sendCount,
                                  dataType, comms[i], streams[i]);
                }

                // Wait for reduction to complete
                Synchronize();

                // Check results
                ValidateResults(subDataset);
            }
        }
        dataset.Release();
    }


    INSTANTIATE_TEST_CASE_P(AllGatherCorrectnessSweep,
                            AllGatherCorrectnessTest,
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
                                                ncclFloat64),
                                // Number of elements
                                testing::Values(3072, 3145728),
                                // Number of devices
                                testing::Values(2,3,4),
                                // In-place or not
                                testing::Values(false, true)));
} // namespace
