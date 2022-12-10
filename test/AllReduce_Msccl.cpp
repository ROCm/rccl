/*************************************************************************
 * Copyright (c) 2022 Advanced Micro Devices, Inc. All rights reserved.
 * Modifications Copyright (c) Microsoft Corporation. Licensed under the MIT License.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

#include <cstdlib>

#include "TestBed.hpp"

namespace RcclUnitTesting
{
  TEST(AllReduce, MscclSingleCall)
  {
    TestBed testBed;

    // Configuration
    std::vector<ncclFunc_t>     const funcTypes       = {ncclCollAllReduce};
    std::vector<ncclDataType_t> const dataTypes       = {ncclInt8, ncclInt32, ncclFloat32};
    std::vector<ncclRedOp_t>    const redOps          = {ncclSum, ncclProd};
    std::vector<int>            const roots           = {0};
    std::vector<int>            const numElements     = {384 * 1024, 384};
    std::vector<bool>           const inPlaceList     = {true, false};
    std::vector<bool>           const managedMemList  = {true, false};
    std::vector<bool>           const useHipGraphList = {true, false};

    testBed.RunSimpleSweep(funcTypes, dataTypes, redOps, roots, numElements, inPlaceList, managedMemList, useHipGraphList);
    testBed.Finalize();
  }

  TEST(AllReduce, MscclGroupCall)
  {
    TestBed testBed;

    // Configuration
    ncclFunc_t                  const  funcType        = ncclCollAllReduce;
    std::vector<ncclDataType_t> const& dataTypes       = {ncclFloat};
    std::vector<ncclRedOp_t>    const& redOps          = {ncclSum};
    std::vector<int>            const  numElements     = {384 * 1024, 384};
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

  TEST(AllReduce, MscclPreMultScalar)
  {
    TestBed testBed;

    // Configuration
    ncclFunc_t                  const  funcType      = ncclCollAllReduce;
    std::vector<ncclDataType_t> const& dataTypes     = {ncclInt32, ncclFloat32, ncclFloat64};
    ncclRedOp_t                 const  redOp         = ncclSum;
    std::vector<int>            const  numElements   = {384 * 1024, 384};
    bool                        const  inPlace       = false;
    bool                        const  useManagedMem = false;

    OptionalColArgs options;
    // Terminate the test as soon as first failure occurs
    bool isCorrect = true;
    for (int totalRanks = testBed.ev.minGpus; totalRanks <= testBed.ev.maxGpus && isCorrect; ++totalRanks)
    for (int isMultiProcess = 0; isMultiProcess <= 1; ++isMultiProcess)
    {
      if (!(testBed.ev.processMask & (1 << isMultiProcess))) continue;

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
        int const numBytes = totalRanks * DataTypeToBytes(dataType);
        memcpy(options.scalarTransport.ptr, scalarsPerRank.ptr, numBytes);

        // Test various scalar residence modes
        for (int scalarMode = 0; scalarMode <= 1 && isCorrect; ++scalarMode)
        {
          if (testBed.ev.showNames)
            INFO("%s %d-ranks AllReduce (custom-scalar Mode %d %s)\n",
                 isMultiProcess ? "MP" : "SP",
                 totalRanks, scalarMode, ncclDataTypeNames[dataType]);

          for (int i = 0; i < numElements.size() && isCorrect; ++i)
          {
            options.scalarMode = scalarMode;
            options.redOp = redOp;
            testBed.SetCollectiveArgs(funcType, dataType,
                                      numElements[i], numElements[i],
                                      options);
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
