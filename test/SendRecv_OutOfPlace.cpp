/*************************************************************************
 * Copyright (c) 2022 Advanced Micro Devices, Inc. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/
#include "TestBed.hpp"

namespace RcclUnitTesting
{
  TEST(SendRecv, OutOfPlace)
  {
    TestBed testBed;

    // Configuration
    std::vector<ncclFunc_t>     const  funcType        = {ncclCollSend, ncclCollRecv};
    std::vector<ncclDataType_t> const& dataTypes       = {ncclInt32};
    std::vector<ncclRedOp_t>    const& redOps          = {ncclSum}; //Not important for send receive tests
    std::vector<int>            const  numElements     = {1048576, 53327, 1024};
    bool                        const  inPlace         = false;
    bool                        const  useManagedMem   = false;
    int                         const  numCollPerGroup = numElements.size();

    bool isCorrect = true;
    int totalRanks = testBed.ev.maxGpus;
    int const numProcesses = 1;
    int const isMultiProcess = 0;
    testBed.InitComms(TestBed::GetDeviceIdsList(numProcesses, totalRanks), numCollPerGroup);

    for (int dataIdx = 0; dataIdx < dataTypes.size() && isCorrect; ++dataIdx)
    {
      if (testBed.ev.showNames) // Show test names
        INFO("%s process %2d-ranks SendRec v%d out Of place (%s)\n",
             isMultiProcess ? "Multi " : "Single",
             numCollPerGroup,
             totalRanks, ncclDataTypeNames[dataTypes[dataIdx]]);

      // Run all element sizes in parallel as single group
      for (int root = 0; root < totalRanks; ++root)
      {
        testBed.SetCollectiveArgs(funcType[0],
                                  dataTypes[dataIdx],
                                  redOps[0],
                                  0, // 0?
                                  numElements[0],
                                  numElements[0],
                                  0,
                                  root);
        testBed.AllocateMem(inPlace, useManagedMem, -1, root);
        testBed.PrepareData(-1, root);
        for (int currentRank = 0; currentRank < totalRanks; ++currentRank)
        {

          if (currentRank != root)
          {
            testBed.SetCollectiveArgs(funcType[0],
                                      dataTypes[dataIdx],
                                      redOps[0],
                                      currentRank,
                                      numElements[0],
                                      numElements[0],
                                      0,
                                      root);

            testBed.SetCollectiveArgs(funcType[1],
                                      dataTypes[dataIdx],
                                      redOps[0],
                                      root,
                                      numElements[0],
                                      numElements[0],
                                      0,
                                      currentRank);
            testBed.AllocateMem(inPlace, useManagedMem, -1, currentRank);
            testBed.PrepareData(-1, currentRank);
            testBed.ExecuteCollectives({root,currentRank});
            testBed.ValidateResults(isCorrect, -1, currentRank);
            testBed.DeallocateMem(-1, currentRank);
          }

        }
        testBed.DeallocateMem(-1, root);
      }
    }
    testBed.DestroyComms();
    testBed.Finalize();
  }
}