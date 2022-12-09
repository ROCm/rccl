/*************************************************************************
 * Copyright (c) 2022 Advanced Micro Devices, Inc. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/
#include "TestBed.hpp"
namespace RcclUnitTesting
{
  TEST(AllReduce, NonBlocking)
  {
    TestBed testBed;
    // Configuration
    ncclFunc_t                  const  funcType      = ncclCollAllReduce;
    std::vector<ncclDataType_t> const& dataTypes     = {ncclFloat};
    std::vector<ncclRedOp_t>    const& redOps        = {ncclSum};
    std::vector<int>            const  numElements   = {1048576, 1024};
    bool                        const  inPlace       = false;
    bool                        const  useManagedMem = false;
    bool                        const  useBlocking   = false;
    
    OptionalColArgs options;
    // Terminate the test as soon as first failure occurs
    bool isCorrect = true;
    for (int totalRanks = testBed.ev.minGpus; totalRanks <= testBed.ev.maxGpus && isCorrect; ++totalRanks)
    for (int isMultiProcess = 0; isMultiProcess <= 1 && isCorrect; ++isMultiProcess)
    {
      if (!(testBed.ev.processMask & (1 << isMultiProcess))) continue;

      // Test either single process all GPUs, or 1 process per GPU
      int const numProcesses = isMultiProcess ? totalRanks : 1;
      testBed.InitComms(TestBed::GetDeviceIdsList(numProcesses, totalRanks), 1, useBlocking);

      for (int redOpIdx = 0; redOpIdx < redOps.size() && isCorrect; ++redOpIdx)
      {
        options.redOp = redOps[redOpIdx];
        for (int dataIdx = 0; dataIdx < dataTypes.size() && isCorrect; ++dataIdx)
        {
          if (testBed.ev.showNames)
            INFO("%s %d-ranks AllReduce %s Blocking Config (%s-%s)\n",
                 isMultiProcess ? "MP" : "SP",
                 totalRanks, useBlocking ? "true" : "false",
                 ncclRedOpNames[redOps[redOpIdx]], ncclDataTypeNames[dataTypes[dataIdx]]);

          
          for (int numIdx = 0; numIdx < numElements.size() && isCorrect; ++numIdx)
          {
            testBed.SetCollectiveArgs(funcType,
                                      dataTypes[dataIdx],
                                      numElements[numIdx],
                                      numElements[numIdx],
                                      options);
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
