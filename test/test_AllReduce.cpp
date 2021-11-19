/*************************************************************************
 * Copyright (c) 2019-2020 Advanced Micro Devices, Inc. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

#include "test_AllReduce.hpp"

namespace CorrectnessTests
{
  TEST_P(AllReduceCorrectnessTest, Correctness)
  {
    if (numDevices > numDevicesAvailable) return;

    // Prepare input / output / expected results
    Dataset dataset;
    dataset.Initialize(numDevices, numElements, dataType, inPlace, ncclCollAllReduce);
    FillDatasetWithPattern(dataset);
    ComputeExpectedResults(dataset, op);

    // Launch the reduction (1 thread per GPU)
    ncclGroupStart();
    for (int i = 0; i < numDevices; i++)
    {
      ncclAllReduce(dataset.inputs[i], dataset.outputs[i],
                    numElements, dataType, op, comms[i], streams[i]);
    }
    ncclGroupEnd();

    // Wait for reduction to complete
    Synchronize();

    // Check results
    ValidateResults(dataset);

    dataset.Release();
  }
#if defined(BUILD_ALLREDUCE_ONLY)
  INSTANTIATE_TEST_SUITE_P(AllReduceCorrectnessSweep,
                           AllReduceCorrectnessTest,
                           testing::Combine(
                             // Reduction operator
                             testing::Values(ncclSum),
                             // Data types
                             testing::Values(ncclFloat32),
                             // Number of elements
                             testing::Values(1024, 1048576),
                             // Number of devices
                             testing::Range(2,(GTESTS_NUM_GPUS+1)),
                             // In-place or not
                             testing::Values(false, true),
                             testing::Values("RCCL_ENABLE_CLIQUE=0", "RCCL_ENABLE_CLIQUE=1")),
                           CorrectnessTest::PrintToStringParamName());
#else
  INSTANTIATE_TEST_SUITE_P(AllReduceCorrectnessSweep,
                           AllReduceCorrectnessTest,
                           testing::Combine(
                             // Reduction operator
                             testing::Values(ncclSum, ncclProd, ncclMax, ncclMin, ncclAvg),
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
                             testing::Values(false, true),
                             testing::Values("RCCL_ENABLE_CLIQUE=0", "RCCL_ENABLE_CLIQUE=1")),
                           CorrectnessTest::PrintToStringParamName());
#endif
} // namespace
