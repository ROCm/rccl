/*************************************************************************
 * Copyright (c) 2019 Advanced Micro Devices, Inc. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/
#ifndef TEST_ALLGATHER_HPP
#define TEST_ALLGATHER_HPP

#include "CorrectnessTest.hpp"

namespace CorrectnessTests
{
    class AllGatherCorrectnessTest : public CorrectnessTest
    {
    public:
        static void ComputeExpectedResults(Dataset& dataset)
        {
            size_t const byteCount = dataset.NumBytes() / dataset.numDevices;

            int8_t* result = (int8_t *)malloc(dataset.NumBytes());

            for (int i = 0; i < dataset.numDevices; i++)
                HIP_CALL(hipMemcpy(result + i * byteCount, (int8_t *)dataset.inputs[i] + (i * byteCount),
                                   byteCount, hipMemcpyDeviceToHost));

            for (int i = 0; i < dataset.numDevices; i++)
                memcpy(dataset.expected[i], result, dataset.NumBytes());
        }
    };
}

#endif
