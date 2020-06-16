/*************************************************************************
 * Copyright (c) 2019-2020 Advanced Micro Devices, Inc. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/
#ifndef TEST_SCATTER_HPP
#define TEST_SCATTER_HPP

#include "CorrectnessTest.hpp"

namespace CorrectnessTests
{
    class ScatterCorrectnessTest : public CorrectnessTest
    {
    public:
        static void ComputeExpectedResults(Dataset& dataset, int const root)
        {
            for (int i = 0; i < dataset.numDevices; i++)
                HIP_CALL(hipMemcpy(dataset.expected[i], (int8_t *)dataset.inputs[root]+dataset.NumBytes()*i,
                                   dataset.NumBytes(), hipMemcpyDeviceToHost));
        }
    };
}

#endif
