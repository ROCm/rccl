/*************************************************************************
 * Copyright (c) 2019-2020 Advanced Micro Devices, Inc. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/
#ifndef TEST_ALLTOALLV_HPP
#define TEST_ALLTOALLV_HPP

#include "CorrectnessTest.hpp"

namespace CorrectnessTests
{
    class AllToAllvCorrectnessTest : public CorrectnessTest
    {
    public:
        static void ComputeExpectedResults(Dataset& dataset)
        {
            for (int i = 0; i < dataset.numDevices; i++) {
                size_t rdisp = 0;
                size_t chunksize = dataset.numElements*2/dataset.numDevices;
                for (int j = 0; j < dataset.numDevices; j++) {
                    size_t scount = 0, rcount = ((j+i)%dataset.numDevices)*chunksize;
                    if (j+i == dataset.numDevices-1)
                        rcount += (dataset.numElements*dataset.numDevices-chunksize*(dataset.numDevices-1)*dataset.numDevices/2);
                    size_t sdisp = 0;
                    for (int k=0; k<dataset.numDevices; k++) {
                        scount = ((k+j)%dataset.numDevices)*chunksize;
                        if (k+j == dataset.numDevices-1)
                          scount += (dataset.numElements*dataset.numDevices-chunksize*(dataset.numDevices-1)*dataset.numDevices/2);
                        if (k == i)
                          break;
                        sdisp += scount;
                    }
                    HIP_CALL(hipMemcpy((int8_t *)dataset.expected[i]+rdisp*DataTypeToBytes(dataset.dataType),
                        (int8_t *)dataset.inputs[j]+sdisp*DataTypeToBytes(dataset.dataType),
                        rcount*DataTypeToBytes(dataset.dataType), hipMemcpyDeviceToHost));
                    rdisp += rcount;
                }
            }
        }
    };
}

#endif
