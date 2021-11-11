/*************************************************************************
 * Copyright (c) 2021 Advanced Micro Devices, Inc. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

#include "test_AllReduceGroup.hpp"

namespace CorrectnessTests
{
  // This tests aggregated AllReduce calls within a group
  TEST_P(AllReduceGroupCorrectnessTest, Correctness)
  {
    if (numDevices > numDevicesAvailable) return;

    // Prepare input / output / expected results
    Dataset dataset1, dataset2, dataset3;
    dataset1.Initialize(numDevices, numElements, dataType, inPlace, ncclCollAllReduce);
    dataset2.Initialize(numDevices, numElements, dataType, inPlace, ncclCollAllReduce);
    dataset3.Initialize(numDevices, numElements, dataType, inPlace, ncclCollAllReduce);
    FillDatasetWithPattern(dataset1);
    FillDatasetWithPattern(dataset2);
    FillDatasetWithPattern(dataset3);
    ComputeExpectedResults(dataset1, op);
    ComputeExpectedResults(dataset2, op);
    ComputeExpectedResults(dataset3, op);

    // Launch the reduction (1 thread per GPU)
    ncclGroupStart();
    for (int i = 0; i < numDevices; i++)
    {
      ncclAllReduce(dataset1.inputs[i], dataset1.outputs[i], numElements, dataType, op, comms[i], streams[i]);
      ncclAllReduce(dataset2.inputs[i], dataset2.outputs[i], numElements, dataType, op, comms[i], streams[i]);
      ncclAllReduce(dataset3.inputs[i], dataset3.outputs[i], numElements, dataType, op, comms[i], streams[i]);
    }
    ncclGroupEnd();

    // Wait for reduction to complete
    Synchronize();

    // Check results
    ValidateResults(dataset1);
    ValidateResults(dataset2);
    ValidateResults(dataset3);

        dataset1.Release();
        dataset2.Release();
        dataset3.Release();
  }
#if defined(BUILD_ALLREDUCE_ONLY)
  INSTANTIATE_TEST_SUITE_P(AllReduceGroupCorrectnessSweep,
                           AllReduceGroupCorrectnessTest,
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
  INSTANTIATE_TEST_SUITE_P(AllReduceGroupCorrectnessSweep,
                           AllReduceGroupCorrectnessTest,
                           testing::Combine(
                             // Reduction operator
                             testing::Values(ncclSum),
                             // Data types
                             testing::Values(ncclFloat32, ncclFloat64),
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
