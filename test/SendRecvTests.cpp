/*************************************************************************
 * Copyright (c) 2022-2023 Advanced Micro Devices, Inc. All rights reserved.
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
    std::vector<ncclDataType_t> const& dataTypes       = {ncclInt32, ncclFloat16, ncclFloat64};
    std::vector<int>            const  numElements     = {1048576, 53327, 1024, 0};
    bool                        const  inPlace         = false;
    bool                        const  useManagedMem   = false;
    int                         const  groupCallId     = 0;

    OptionalColArgs options;
    bool isCorrect = true;
    int numGpus = testBed.ev.maxGpus;
    for (int rpg=0; rpg < 2 && isCorrect; ++rpg)
    for (int isMultiProcess = 0; isMultiProcess <= 1 && isCorrect; ++isMultiProcess)
    {
      if (!(testBed.ev.processMask & (1 << isMultiProcess))) continue;
      int ranksPerGpu = rpg == 0 ? 1 : testBed.ev.maxRanksPerGpu;
      int totalRanks = numGpus * ranksPerGpu;
      int const numProcesses = isMultiProcess ? numGpus : 1;
      testBed.InitComms(TestBed::GetDeviceIdsList(numProcesses, numGpus, ranksPerGpu), 1);

      for (int dataIdx = 0; dataIdx < dataTypes.size() && isCorrect; ++dataIdx)
      for (int numIdx = 0; numIdx < numElements.size() && isCorrect; ++numIdx)
      for (int sendRank = 0; sendRank < totalRanks; ++sendRank)
      {
        for (int recvRank = 0; recvRank  < totalRanks; ++recvRank)
        {
          options.root = recvRank;
          testBed.SetCollectiveArgs(ncclCollSend,
                                    dataTypes[dataIdx],
                                    numElements[numIdx],
                                    numElements[numIdx],
                                    options,
                                    0,
                                    groupCallId,
                                    sendRank);
          if (recvRank == 0)
          {
            testBed.AllocateMem(inPlace, useManagedMem, groupCallId, 0, sendRank);
            testBed.PrepareData(groupCallId, 0, sendRank);
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

            options.root = sendRank;
            testBed.SetCollectiveArgs(ncclCollRecv,
                                      dataTypes[dataIdx],
                                      numElements[numIdx],
                                      numElements[numIdx],
                                      options,
                                      0,
                                      groupCallId,
                                      recvRank);
            testBed.AllocateMem(inPlace, useManagedMem, groupCallId, 0, recvRank);
            testBed.PrepareData(groupCallId, 0, recvRank);
            testBed.ExecuteCollectives({sendRank, recvRank});
            testBed.ValidateResults(isCorrect, groupCallId, 0, recvRank);
            testBed.DeallocateMem(groupCallId, 0, recvRank);
          }
        }
        testBed.DeallocateMem(groupCallId, 0, sendRank);
      }
      testBed.DestroyComms();
    }
    testBed.Finalize();
  }

  TEST(SendRecv, UserBufferRegister)
  {
    setenv("RCCL_ENABLE_INTRANET", "1", 1);
    TestBed testBed;

    // Configuration
    std::vector<ncclDataType_t> const& dataTypes       = {ncclInt32, ncclFloat16, ncclFloat64};
    std::vector<int>            const  numElements     = {1048576, 53327, 1024, 0};
    bool                        const  inPlace         = false;
    bool                        const  useManagedMem   = false;
    bool                        const  userRegistered  = true;
    int                         const  groupCallId     = 0;

    OptionalColArgs options;
    bool isCorrect = true;
    int numGpus = testBed.ev.maxGpus;
    for (int rpg=0; rpg < 2 && isCorrect; ++rpg)
    for (int isMultiProcess = 0; isMultiProcess <= 1 && isCorrect; ++isMultiProcess)
    {
      if (!(testBed.ev.processMask & (1 << isMultiProcess))) continue;
      int ranksPerGpu = rpg == 0 ? 1 : testBed.ev.maxRanksPerGpu;
      int totalRanks = numGpus * ranksPerGpu;
      int const numProcesses = isMultiProcess ? numGpus : 1;
      testBed.InitComms(TestBed::GetDeviceIdsList(numProcesses, numGpus, ranksPerGpu), 1);

      for (int dataIdx = 0; dataIdx < dataTypes.size() && isCorrect; ++dataIdx)
      for (int numIdx = 0; numIdx < numElements.size() && isCorrect; ++numIdx)
      for (int sendRank = 0; sendRank < totalRanks; ++sendRank)
      {
        for (int recvRank = 0; recvRank  < totalRanks; ++recvRank)
        {
          options.root = recvRank;
          testBed.SetCollectiveArgs(ncclCollSend,
                                    dataTypes[dataIdx],
                                    numElements[numIdx],
                                    numElements[numIdx],
                                    options,
                                    0,
                                    groupCallId,
                                    sendRank);
          if (recvRank == 0)
          {
            testBed.AllocateMem(inPlace, useManagedMem, groupCallId, 0, sendRank, userRegistered);
            testBed.PrepareData(groupCallId, 0, sendRank);
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

            options.root = sendRank;
            testBed.SetCollectiveArgs(ncclCollRecv,
                                      dataTypes[dataIdx],
                                      numElements[numIdx],
                                      numElements[numIdx],
                                      options,
                                      0,
                                      groupCallId,
                                      recvRank);
            testBed.AllocateMem(inPlace, useManagedMem, groupCallId, 0, recvRank, userRegistered);
            testBed.PrepareData(groupCallId, 0, recvRank);
            testBed.ExecuteCollectives({sendRank, recvRank});
            testBed.ValidateResults(isCorrect, groupCallId, 0, recvRank);
            testBed.DeallocateMem(groupCallId, 0, recvRank);
          }
        }
        testBed.DeallocateMem(groupCallId, 0, sendRank);
      }
      testBed.DestroyComms();
    }
    testBed.Finalize();
    unsetenv("RCCL_ENABLE_INTRANET");
  }
}
