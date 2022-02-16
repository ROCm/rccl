/*************************************************************************
 * Copyright (c) 2022 Advanced Micro Devices, Inc. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/
#include "TestBed.hpp"

namespace RcclUnitTesting
{
  // This tests using custom pre-mult scalars reductions
  // NOTE: Currently disabled due to issues with > 4 GPUs
  TEST(AllReduce, PreMultScalar)
  {
    TestBed testBed;

    // Configuration
    ncclFunc_t                  const  funcType      = ncclCollAllReduce;
    std::vector<ncclDataType_t> const& dataTypes     = {ncclInt32, ncclFloat32, ncclFloat64};
    ncclRedOp_t                 const  redOp         = ncclSum;
    std::vector<int>            const  numElements   = {1048576, 1024};
    int                         const  root          = 0;
    bool                        const  inPlace       = false;
    bool                        const  useManagedMem = false;

    // Terminate the test as soon as first failure occurs
    bool isCorrect = true;
    for (int totalRanks = testBed.ev.minGpus; totalRanks <= testBed.ev.maxGpus && isCorrect; ++totalRanks)
    for (int isMultiProcess = 0; isMultiProcess <= 1; ++isMultiProcess)
    {
      int const numProcesses = isMultiProcess ? totalRanks : 1;
      testBed.InitComms(TestBed::GetDeviceIdsList(numProcesses, totalRanks));

      for (int dataIdx = 0; dataIdx < dataTypes.size() && isCorrect; ++dataIdx)
      {
        ncclDataType_t const dataType = dataTypes[dataIdx];

        // Set scalars per rank
        PtrUnion scalarsPerRank;
        scalarsPerRank.AllocateCpuMem(totalRanks * DataTypeToBytes(dataType));
        for (int i = 0;  i < totalRanks; i++)
        {
          double F = i;
          scalarsPerRank.Set(dataType, i, i, F);
        }

        // Test various scalar residence modes
        for (int scalarMode = 0; scalarMode <= 1 && isCorrect; ++scalarMode)
        {
          if (testBed.ev.showNames)
            INFO("%s process %2d-ranks AllReduce (custom-scalar Mode %d %s)\n",
                 isMultiProcess ? "Multi " : "Single",
                 totalRanks, scalarMode, ncclDataTypeNames[dataType]);

          for (int i = 0; i < numElements.size() && isCorrect; ++i)
          {
            testBed.SetCollectiveArgs(funcType, dataType, redOp, root,
                                      numElements[i], numElements[i],
                                      -1, -1, scalarsPerRank, scalarMode);
            // For performance, only allocate and prepare data on largest size
            if (i == 0)
            {
              testBed.AllocateMem(inPlace, useManagedMem);
              testBed.PrepareData();
            }
            testBed.ExecuteCollectives();
            testBed.ValidateResults(isCorrect);
          }
          testBed.DeallocateMem();
        }
      }
      testBed.DestroyComms();
    }
    testBed.Finalize();
  }
}
