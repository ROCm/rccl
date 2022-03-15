/*************************************************************************
 * Copyright (c) 2022 Advanced Micro Devices, Inc. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/
#include "TestBed.hpp"

namespace RcclUnitTesting
{
  TEST(AllReduce, GroupCall)
  {
    TestBed testBed;

    // Configuration
    ncclFunc_t                  const  funcType        = ncclCollAllReduce;
    std::vector<ncclDataType_t> const& dataTypes       = {ncclFloat};
    std::vector<ncclRedOp_t>    const& redOps          = {ncclSum};
    std::vector<int>            const  numElements     = {1048576, 53327, 1024};
    bool                        const  inPlace         = false;
    bool                        const  useManagedMem   = false;
    int                         const  numCollPerGroup = numElements.size();

    OptionalColArgs options;
    // This tests runs 3 collectives in the same group call
    bool isCorrect = true;
    for (int totalRanks = testBed.ev.minGpus; totalRanks <= testBed.ev.maxGpus && isCorrect; ++totalRanks)
    for (int isMultiProcess = 0; isMultiProcess <= 1 && isCorrect; ++isMultiProcess)
    {
      if (!(testBed.ev.processMask & (1 << isMultiProcess))) continue;

      // Test either single process all GPUs, or 1 process per GPU
      int const numProcesses = isMultiProcess ? totalRanks : 1;
      testBed.InitComms(TestBed::GetDeviceIdsList(numProcesses, totalRanks), numCollPerGroup);

      for (int redOpIdx = 0; redOpIdx < redOps.size() && isCorrect; ++redOpIdx)
      {
        options.redOp = redOps[redOpIdx];
        for (int dataIdx = 0; dataIdx < dataTypes.size() && isCorrect; ++dataIdx)
        {
          if (testBed.ev.showNames)
            INFO("%s %d-ranks AllReduce %d Grouped Calls (%s-%s)\n",
                 isMultiProcess ? "MP" : "SP",
                 totalRanks, numCollPerGroup,
                 ncclRedOpNames[redOps[redOpIdx]], ncclDataTypeNames[dataTypes[dataIdx]]);

          // Run all element sizes in parallel as single group
          for (int collIdx = 0; collIdx < numCollPerGroup; ++collIdx)
          {
            testBed.SetCollectiveArgs(funcType,
                                      dataTypes[dataIdx],
                                      numElements[collIdx],
                                      numElements[collIdx],
                                      options,
                                      collIdx);
          }
          testBed.AllocateMem(inPlace, useManagedMem);
          testBed.PrepareData();
          testBed.ExecuteCollectives();
          testBed.ValidateResults(isCorrect);
          testBed.DeallocateMem();
        }
      }
      testBed.DestroyComms();
    }
    testBed.Finalize();
  }
}
