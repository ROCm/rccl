/*************************************************************************
 * Copyright (c) 2022 Advanced Micro Devices, Inc. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/
#include "TestBed.hpp"

namespace RcclUnitTesting
{
  TEST(SendRecv, SinglePairs)
  {
    TestBed testBed;

    // Configuration
    std::vector<ncclDataType_t> const& dataTypes       = {ncclInt32, ncclFloat64};
    std::vector<int>            const  numElements     = {1048576, 53327, 1024};
    bool                        const  inPlace         = false;
    bool                        const  useManagedMem   = false;

    bool isCorrect = true;
    int totalRanks = testBed.ev.maxGpus;
    for (int isMultiProcess = 0; isMultiProcess <= 1 && isCorrect; ++isMultiProcess)
    {
      if (!(testBed.ev.processMask & (1 << isMultiProcess))) continue;

      int const numProcesses = isMultiProcess ? totalRanks : 1;
      testBed.InitComms(TestBed::GetDeviceIdsList(numProcesses, totalRanks), 1);

      for (int dataIdx = 0; dataIdx < dataTypes.size() && isCorrect; ++dataIdx)
      for (int numIdx = 0; numIdx < numElements.size() && isCorrect; ++numIdx)
      for (int sendRank = 0; sendRank < totalRanks; ++sendRank)
      {
        for (int recvRank = 0; recvRank  < totalRanks; ++recvRank)
        {
          testBed.SetCollectiveArgs(ncclCollSend,
                                    dataTypes[dataIdx],
                                    ncclSum,
                                    recvRank,
                                    numElements[numIdx],
                                    numElements[numIdx],
                                    0,
                                    sendRank);
          if (recvRank == 0)
          {
            testBed.AllocateMem(inPlace, useManagedMem, 0, sendRank);
            testBed.PrepareData(0, sendRank);
          }
          if (recvRank  != sendRank)
          {
            if (testBed.ev.showNames) // Show test names
              INFO("%s Datatype: %s SendReceive test Rank %d -> Rank %d for %d Elements\n",
                  isMultiProcess ? "MP" : "SP",
                  ncclDataTypeNames[dataTypes[dataIdx]],
                  sendRank,
                  recvRank,
                  numElements[numIdx]);


            testBed.SetCollectiveArgs(ncclCollRecv,
                                      dataTypes[dataIdx],
                                      ncclSum,
                                      sendRank,
                                      numElements[numIdx],
                                      numElements[numIdx],
                                      0,
                                      recvRank);
            testBed.AllocateMem(inPlace, useManagedMem, 0, recvRank);
            testBed.PrepareData(0, recvRank);
            testBed.ExecuteCollectives({sendRank,recvRank });
            testBed.ValidateResults(isCorrect, 0, recvRank);
            testBed.DeallocateMem(0, recvRank);
          }
        }
        testBed.DeallocateMem(0, sendRank);
      }
      testBed.DestroyComms();
    }
    testBed.Finalize();
  }
}
