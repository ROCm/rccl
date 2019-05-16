/*************************************************************************
 * Copyright (c) 2019 Advanced Micro Devices, Inc. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/
#ifndef TEST_BROADCAST_HPP
#define TEST_BROADCAST_HPP

#include "CorrectnessTest.hpp"
#include <omp.h>

namespace CorrectnessTests
{
    class BroadcastCorrectnessTest : public CorrectnessTest
    {
    public:
        static void ComputeExpectedResults(Dataset& dataset, int const root)
        {
            for (int i = 0; i < dataset.numDevices; i++)
                HIP_CALL(hipMemcpy(dataset.expected[i], dataset.inputs[root],
                                   dataset.NumBytes(), hipMemcpyDeviceToHost));
        }
    };
}

#endif
