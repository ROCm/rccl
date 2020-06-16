/*************************************************************************
 * Copyright (c) 2019-2020 Advanced Micro Devices, Inc. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/
#ifndef TEST_ALLTOALL_HPP
#define TEST_ALLTOALL_HPP

#include "CorrectnessTest.hpp"

namespace CorrectnessTests
{
    class AllToAllCorrectnessTest : public CorrectnessTest
    {
    public:
        static void ComputeExpectedResults(Dataset& dataset)
        {
            for (int i = 0; i < dataset.numDevices; i++)
                for (int j = 0; j < dataset.numDevices; j++)
                    HIP_CALL(hipMemcpy((int8_t *)dataset.expected[i]+dataset.NumBytes()*j, (int8_t *)dataset.inputs[j]+dataset.NumBytes()*i,
                                   dataset.NumBytes(), hipMemcpyDeviceToHost));
        }
    };
}

#endif
