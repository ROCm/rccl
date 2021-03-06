/*************************************************************************
 * Copyright (c) 2019-2020 Advanced Micro Devices, Inc. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

#include "test_AllToAllv.hpp"

namespace CorrectnessTests
{
    TEST_P(AllToAllvCorrectnessTest, Correctness)
    {
        if (numDevices > numDevicesAvailable) return;

        // Allocate data
        Dataset dataset;
        dataset.Initialize(numDevices, numElements, dataType, inPlace, ncclCollAllToAll);

        // Prepare input / output / expected results
        FillDatasetWithPattern(dataset);
        ComputeExpectedResults(dataset);

        size_t chunksize = numElements*2/numDevices;
        #define MAX_ALLTOALLV_RANKS 16
        static size_t sendcounts[MAX_ALLTOALLV_RANKS*MAX_ALLTOALLV_RANKS], recvcounts[MAX_ALLTOALLV_RANKS*MAX_ALLTOALLV_RANKS], sdispls[MAX_ALLTOALLV_RANKS*MAX_ALLTOALLV_RANKS], rdispls[MAX_ALLTOALLV_RANKS*MAX_ALLTOALLV_RANKS];
        // Launch the reduction (1 thread per GPU)
        ncclGroupStart();
        for (int r = 0; r < numDevices; r++) {
            size_t disp = 0;
            for (int i = 0; i < numDevices; i++) {
                size_t scount = ((i+r)%numDevices)*chunksize;
                if (i+r == numDevices-1)
                  scount += (numElements*numDevices-chunksize*(numDevices-1)*numDevices/2);
                sendcounts[i+r*MAX_ALLTOALLV_RANKS] = recvcounts[i+r*MAX_ALLTOALLV_RANKS] = scount;
                sdispls[i+r*MAX_ALLTOALLV_RANKS] = rdispls[i+r*MAX_ALLTOALLV_RANKS] = disp;
                disp += scount;
            }
            ncclAllToAllv((char*)dataset.inputs[r], sendcounts+r*MAX_ALLTOALLV_RANKS, sdispls+r*MAX_ALLTOALLV_RANKS,
              (char*)dataset.outputs[r], recvcounts+r*MAX_ALLTOALLV_RANKS, rdispls+r*MAX_ALLTOALLV_RANKS, dataType, comms[r], streams[r]);
        }
        ncclGroupEnd();
        // Wait for reduction to complete
        Synchronize();

        // Check results
        ValidateResults(dataset);

        dataset.Release();
    }

    INSTANTIATE_TEST_SUITE_P(AllToAllvCorrectnessSweep,
                            AllToAllvCorrectnessTest,
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
                                testing::Values(2520, 3026520),
                                // Number of devices
                                testing::Values(2,3,4,5,6,7,8),
                                // In-place or not
                                testing::Values(false),
                                testing::Values("")),
                            CorrectnessTest::PrintToStringParamName());
} // namespace
