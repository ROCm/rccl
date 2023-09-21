/*************************************************************************
 * Copyright (c) 2023 Advanced Micro Devices, Inc. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

 // Note: InPlace is not supported for All-To-Allv
 
#include "TestBed.hpp"

namespace RcclUnitTesting
{
  // Prepare sendcount/recvcounts, sdispls/rdispls arrays within options
  void PrepareCounts(int const totalRanks, int const chunkSize,
                     OptionalColArgs& options,
                     std::vector<size_t>& numInputElements,
                     std::vector<size_t>& numOutputElements)
  {
    numInputElements.clear();
    numOutputElements.clear();
    numInputElements.resize(totalRanks, 0);
    numOutputElements.resize(totalRanks, 0);

    // Decide how many elements each pair send/recv
    for (int sendRank = 0; sendRank < totalRanks; ++sendRank)
    for (int recvRank = 0; recvRank < totalRanks; ++recvRank)
    {
      // Get linear indices into sendcounts/recvcounts array
      int const sendIdx = sendRank * totalRanks + recvRank;
      int const recvIdx = recvRank * totalRanks + sendRank;

      // Each pair sends slightly different amounts of elements (based on chunkSize)
      int const numElements = (1 + sendRank + recvRank) * chunkSize;
      options.sendcounts[sendIdx]  = options.recvcounts[recvIdx] = numElements;
    }

    // Compute displacements
    for (int sendRank = 0; sendRank < totalRanks; ++sendRank)
    {
      int totalSend = 0;
      int totalRecv = 0;

      for (int recvRank = 0; recvRank < totalRanks; ++recvRank)
      {
        int const pairIdx = sendRank * totalRanks + recvRank;

        options.sdispls[pairIdx] = totalSend;
        options.rdispls[pairIdx] = totalRecv;

        totalSend += options.sendcounts[pairIdx];
        totalRecv += options.recvcounts[pairIdx];
      }

      numInputElements[sendRank] = totalSend;
      numOutputElements[sendRank] = totalRecv;
    }
  }

  TEST(AllToAllv, OutOfPlace)
  {
    TestBed testBed;

    // Configuration
    std::vector<ncclDataType_t> const& dataTypes       = {ncclInt32, ncclFloat64, ncclFloat16};
    bool                        const  inPlace         = false;
    bool                        const  useManagedMem   = false;
    bool                        const  useHipGraph     = false;

    OptionalColArgs options;

    bool isCorrect = true;
    for (int totalRanks : testBed.ev.GetNumGpusList())
    for (int isMultiProcess : testBed.ev.GetIsMultiProcessList())
    {
      int const numProcesses = isMultiProcess ? totalRanks : 1;
      testBed.InitComms(TestBed::GetDeviceIdsList(numProcesses, totalRanks));

      // Prepare AllToAllV options
      std::vector<size_t> numInputElements;
      std::vector<size_t> numOutputElements;
      PrepareCounts(totalRanks, 256, options, numInputElements, numOutputElements);

      for (int dataIdx = 0; dataIdx < dataTypes.size() && isCorrect; ++dataIdx)
      {
        if (testBed.ev.showNames)
        {
          std::string name = testBed.GetTestCaseName(totalRanks, isMultiProcess,
                                                     ncclCollAllToAllv, dataTypes[dataIdx],
                                                     ncclSum, -1, inPlace, useManagedMem, useHipGraph);
          INFO("%s\n", name.c_str());
        }

        for (int rank = 0; rank < totalRanks; ++rank)
        {
          testBed.SetCollectiveArgs(ncclCollAllToAllv,
                                    dataTypes[dataIdx],
                                    numInputElements[rank],
                                    numOutputElements[rank],
                                    options,
                                    -1,
                                    rank);
        }
        testBed.AllocateMem(inPlace, useManagedMem);
        testBed.PrepareData();
        testBed.ExecuteCollectives({}, useHipGraph);
        testBed.ValidateResults(isCorrect);
        testBed.DeallocateMem();
      }
      testBed.DestroyComms();
    }
    testBed.Finalize();
  }


  TEST(AllToAllv, OutOfPlaceGraph)
  {
    TestBed testBed;

    // Configuration
    std::vector<ncclDataType_t> const& dataTypes       = {ncclFloat32, ncclInt8};
    bool                        const  inPlace         = false;
    bool                        const  useManagedMem   = false;
    bool                        const  useHipGraph     = false;

    OptionalColArgs options;

    bool isCorrect = true;
    for (int totalRanks : testBed.ev.GetNumGpusList())
    for (int isMultiProcess : testBed.ev.GetIsMultiProcessList())
    {
      int const numProcesses = isMultiProcess ? totalRanks : 1;
      testBed.InitComms(TestBed::GetDeviceIdsList(numProcesses, totalRanks));

      // Prepare AllToAllV options
      std::vector<size_t> numInputElements;
      std::vector<size_t> numOutputElements;
      PrepareCounts(totalRanks, 256, options, numInputElements, numOutputElements);

      for (int dataIdx = 0; dataIdx < dataTypes.size() && isCorrect; ++dataIdx)
      {
        if (testBed.ev.showNames)
        {
          std::string name = testBed.GetTestCaseName(totalRanks, isMultiProcess,
                                                     ncclCollAllToAllv, dataTypes[dataIdx],
                                                     ncclSum, -1, inPlace, useManagedMem, useHipGraph);
          INFO("%s\n", name.c_str());
        }

        for (int rank = 0; rank < totalRanks; ++rank)
        {
          testBed.SetCollectiveArgs(ncclCollAllToAllv,
                                    dataTypes[dataIdx],
                                    numInputElements[rank],
                                    numOutputElements[rank],
                                    options,
                                    -1,
                                    rank);
        }
        testBed.AllocateMem(inPlace, useManagedMem);
        testBed.PrepareData();
        testBed.ExecuteCollectives({}, useHipGraph);
        testBed.ValidateResults(isCorrect);
        testBed.DeallocateMem();
      }
      testBed.DestroyComms();
    }
    testBed.Finalize();
  }
}
