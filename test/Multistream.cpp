/*************************************************************************
 * Copyright (c) 2022 Advanced Micro Devices, Inc. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/
#include "TestBed.hpp"
#include <cstdlib>
namespace RcclUnitTesting
{
  TEST(Multistream, NoGraph)
  {
    TestBed testBed;

    // Configuration
    int  const  numElements        = 1048576;
    bool const  inPlace            = false;
    bool const  useManagedMem      = false;

    OptionalColArgs options;

    // This test runs multiple AllReduce collectives on different streams within the same group call
    bool isCorrect = true;
    for (int totalRanks = testBed.ev.minGpus; totalRanks <= testBed.ev.maxGpus && isCorrect; ++totalRanks)
    for (int isMultiProcess = 0; isMultiProcess <= 1 && isCorrect; ++isMultiProcess)
    {
      if (!(testBed.ev.processMask & (1 << isMultiProcess))) continue;

      // Test either single process all GPUs, or 1 process per GPU
      int const numProcesses = isMultiProcess ? totalRanks : 1;

      for (int numCollPerGroup = 2; numCollPerGroup <= 6; numCollPerGroup += 2)
      {
        for (int numStreamsPerGroup = numCollPerGroup; numStreamsPerGroup >= 2; numStreamsPerGroup -= 3)
        {
          if (testBed.ev.showNames)
            INFO("%s %d-ranks Multistream %d-Group Calls across %d streams\n",
                 isMultiProcess ? "MP" : "SP", totalRanks, numCollPerGroup, numStreamsPerGroup);

          testBed.InitComms(TestBed::GetDeviceIdsList(numProcesses, totalRanks),
                            numCollPerGroup, false, numStreamsPerGroup);

          // Set up each collective in group in different stream (modulo numStreamsPerGroup)
          options.redOp = ncclSum;
          for (int collIdx = 0; collIdx < numCollPerGroup; ++collIdx)
          {
            testBed.SetCollectiveArgs(ncclCollAllReduce, ncclFloat, numElements, numElements,
                                      options, collIdx, -1, collIdx % numStreamsPerGroup);
          }

          testBed.AllocateMem(inPlace, useManagedMem);
          testBed.PrepareData();
          testBed.ExecuteCollectives();
          testBed.ValidateResults(isCorrect);
          testBed.DeallocateMem();
          testBed.DestroyComms();
        }
      }
    }
    testBed.Finalize();
  }
}
