/*************************************************************************
 * Copyright (c) 2019-2020 Advanced Micro Devices, Inc. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/
#ifndef TEST_GATHER_HPP
#define TEST_GATHER_HPP

#include "CorrectnessTest.hpp"

namespace CorrectnessTests
{
    class GatherCorrectnessTest : public CorrectnessTest
    {
    public:
        static void ComputeExpectedResults(Dataset& dataset, int const root)
        {
            for (int i = 0; i < dataset.numDevices; i++)
                HIP_CALL(hipMemcpy((int8_t *)dataset.expected[root]+dataset.NumBytes()*i, dataset.inputs[i],
                                   dataset.NumBytes(), hipMemcpyDeviceToHost));
        }
    };
}

#endif
