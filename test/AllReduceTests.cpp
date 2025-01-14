/*************************************************************************
 * Copyright (c) 2023 Advanced Micro Devices, Inc. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/
#include "TestBed.hpp"

namespace RcclUnitTesting
{
  TEST(AllReduce, OutOfPlace)
  {
    TestBed testBed;

    // Configuration
    std::vector<ncclFunc_t>     const funcTypes       = {ncclCollAllReduce};
    std::vector<ncclDataType_t> const dataTypes       = {ncclFloat32, ncclFp8E4M3, ncclFp8E5M2};
    std::vector<ncclRedOp_t>    const redOps          = {ncclSum};
    std::vector<int>            const roots           = {0};
    std::vector<int>            const numElements     = {393216, 384};
    std::vector<bool>           const inPlaceList     = {false};
    std::vector<bool>           const managedMemList  = {false};
    std::vector<bool>           const useHipGraphList = {false};

    testBed.RunSimpleSweep(funcTypes, dataTypes, redOps, roots, numElements,
                           inPlaceList, managedMemList, useHipGraphList);
    testBed.Finalize();
  }

  TEST(AllReduce, OutOfPlaceGraph)
  {
    TestBed testBed;

    // Configuration
    std::vector<ncclFunc_t>     const funcTypes       = {ncclCollAllReduce};
    std::vector<ncclDataType_t> const dataTypes       = {ncclFloat16, ncclFloat64, ncclFp8E4M3, ncclFp8E5M2};
    std::vector<ncclRedOp_t>    const redOps          = {ncclMin};
    std::vector<int>            const roots           = {0};
    std::vector<int>            const numElements     = {12888};
    std::vector<bool>           const inPlaceList     = {false};
    std::vector<bool>           const managedMemList  = {false};
    std::vector<bool>           const useHipGraphList = {true};

    testBed.RunSimpleSweep(funcTypes, dataTypes, redOps, roots, numElements,
                           inPlaceList, managedMemList, useHipGraphList);
    testBed.Finalize();
  }

  TEST(AllReduce, InPlace)
  {
    TestBed testBed;

    // Configuration
    std::vector<ncclFunc_t>     const funcTypes       = {ncclCollAllReduce};
    std::vector<ncclDataType_t> const dataTypes       = {ncclInt32, ncclInt8};
    std::vector<ncclRedOp_t>    const redOps          = {ncclProd};
    std::vector<int>            const roots           = {0};
    std::vector<int>            const numElements     = {384};
    std::vector<bool>           const inPlaceList     = {true};
    std::vector<bool>           const managedMemList  = {false};
    std::vector<bool>           const useHipGraphList = {false};

    testBed.RunSimpleSweep(funcTypes, dataTypes, redOps, roots, numElements,
                           inPlaceList, managedMemList, useHipGraphList);
    testBed.Finalize();
  }

  TEST(AllReduce, InPlaceGraph)
  {
    TestBed testBed;

    // Configuration
    std::vector<ncclFunc_t>     const funcTypes       = {ncclCollAllReduce};
    std::vector<ncclDataType_t> const dataTypes       = {ncclInt32, ncclFp8E4M3, ncclFp8E5M2};
    std::vector<ncclRedOp_t>    const redOps          = {ncclMax};
    std::vector<int>            const roots           = {0};
    std::vector<int>            const numElements     = {393216, 12888, 384};
    std::vector<bool>           const inPlaceList     = {true};
    std::vector<bool>           const managedMemList  = {false};
    std::vector<bool>           const useHipGraphList = {true};

    testBed.RunSimpleSweep(funcTypes, dataTypes, redOps, roots, numElements,
                           inPlaceList, managedMemList, useHipGraphList);
    testBed.Finalize();
  }

  TEST(AllReduce, ManagedMem)
  {
    TestBed testBed;

    // Configuration
    std::vector<ncclFunc_t>     const funcTypes       = {ncclCollAllReduce};
    std::vector<ncclDataType_t> const dataTypes       = {ncclUint8, ncclUint64};
    std::vector<ncclRedOp_t>    const redOps          = {ncclSum};
    std::vector<int>            const roots           = {0};
    std::vector<int>            const numElements     = {2500};
    std::vector<bool>           const inPlaceList     = {false};
    std::vector<bool>           const managedMemList  = {true};
    std::vector<bool>           const useHipGraphList = {false};

    testBed.RunSimpleSweep(funcTypes, dataTypes, redOps, roots, numElements,
                           inPlaceList, managedMemList, useHipGraphList);
    testBed.Finalize();
  }

  TEST(AllReduce, Channels)
  {
    TestBed testBed;
    if(testBed.ev.maxGpus >= 8) {
      if(testBed.ev.isGfx94) {
        // Configuration
        std::vector<ncclFunc_t>     const funcTypes       = {ncclCollAllReduce};
        std::vector<ncclDataType_t> const dataTypes       = {ncclBfloat16};
        std::vector<ncclRedOp_t>    const redOps          = {ncclSum};
        std::vector<int>            const roots           = {0};
        std::vector<int>            const numElements     = {64 * 1024 * 1024, 1024};
        std::vector<bool>           const inPlaceList     = {false};
        std::vector<bool>           const managedMemList  = {false};
        std::vector<bool>           const useHipGraphList = {false, true};
        std::vector<const char *>   const channelList     = {"84", "112"};
        bool                        const enableSweep     = false; 
        for (auto channel : channelList) {
          setenv("NCCL_MIN_NCHANNELS", channel, 1);
          testBed.RunSimpleSweep(funcTypes, dataTypes, redOps, roots, numElements,
                                inPlaceList, managedMemList, useHipGraphList, enableSweep);
          testBed.Finalize();
          unsetenv("NCCL_MIN_NCHANNELS");
        }
      }
    }
  }

  TEST(AllReduce, ManagedMemGraph)
  {
    TestBed testBed;

    // Configuration
    std::vector<ncclFunc_t>     const funcTypes       = {ncclCollAllReduce};
    std::vector<ncclDataType_t> const dataTypes       = {ncclFloat64, ncclBfloat16};
    std::vector<ncclRedOp_t>    const redOps          = {ncclSum};
    std::vector<int>            const roots           = {0};
    std::vector<int>            const numElements     = {4314};
    std::vector<bool>           const inPlaceList     = {false};
    std::vector<bool>           const managedMemList  = {true};
    std::vector<bool>           const useHipGraphList = {true};

    testBed.RunSimpleSweep(funcTypes, dataTypes, redOps, roots, numElements,
                           inPlaceList, managedMemList, useHipGraphList);
    testBed.Finalize();
  }

  TEST(AllReduce, DISABLED_Clique)
  {
    // Set clique env var prior to TestBed
    setenv("RCCL_ENABLE_CLIQUE", "1", 1);

    TestBed testBed;

    // Configuration
    std::vector<ncclFunc_t>     const funcTypes       = {ncclCollAllReduce};
    std::vector<ncclDataType_t> const dataTypes       = testBed.GetAllSupportedDataTypes();
    std::vector<ncclRedOp_t>    const redOps          = testBed.GetAllSupportedRedOps();
    std::vector<int>            const roots           = {0};
    std::vector<int>            const numElements     = {1048576, 1024};
    std::vector<bool>           const inPlaceList     = {false, true};
    std::vector<bool>           const managedMemList  = {false};
    std::vector<bool>           const useHipGraphList = {false, true};

    testBed.RunSimpleSweep(funcTypes, dataTypes, redOps, roots, numElements,
                           inPlaceList, managedMemList, useHipGraphList);
    testBed.Finalize();

    unsetenv("RCCL_ENABLE_CLIQUE");
  }

  // This tests using custom pre-mult scalars reductions
  TEST(AllReduce, PreMultScalar)
  {
    TestBed testBed;

    // Configuration
    ncclFunc_t                  const  funcType      = ncclCollAllReduce;
    std::vector<ncclDataType_t> const& dataTypes     = {ncclFloat32};
    ncclRedOp_t                 const  redOp         = ncclSum;
    std::vector<int>            const  numElements   = {384 * 1024, 384 * 32, 384};
    bool                        const  inPlace       = false;
    bool                        const  useManagedMem = false;

    OptionalColArgs options;

    // Terminate the test as soon as first failure occurs
    bool isCorrect = true;
    for (int totalRanks : testBed.ev.GetNumGpusList())
    for (int isMultiProcess : testBed.ev.GetIsMultiProcessList())
    {
      int const numProcesses = isMultiProcess ? totalRanks : 1;
      const std::vector<int>& gpuPriorityOrder = testBed.ev.GetGpuPriorityOrder();
      testBed.InitComms(TestBed::GetDeviceIdsList(numProcesses, totalRanks, gpuPriorityOrder));

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

  TEST(AllReduce, UBR3)
  {          
    setenv("UT_PROCESS_MASK", "2", 1);
    const int nranks = 8;
    std::vector<pid_t> children(nranks);
    std::vector<std::vector<int>> childPipes(nranks, std::vector<int>(2,0));
    std::vector<bool> ResultsCorrect(nranks, false);
    ncclUniqueId id;

    size_t count = 32;
    std::vector<int> data(count, 0);
    std::vector<int> expected(count, 0);
    for (int i = 0; i < count; ++i){
        data[i] = i;
        expected[i] = i * nranks;
    }
    
    for(int r = 0; r < nranks; ++r){
      if(pipe(childPipes[r].data()) == -1)
        printf("child %i pipe Failed\n", r);
    } 
    
    auto createNCCLid = [&](int rank){
        ncclGetUniqueId(&id);
        close(childPipes[rank][0]);
        write(childPipes[rank][1], &id, sizeof(ncclUniqueId));
        close(childPipes[rank][1]);
    };

    auto getNCCLidFromParent = [&](int rank){
      close(childPipes[rank][1]); //close write to child0
      read(childPipes[rank][0], &id, sizeof(ncclUniqueId));
      close(childPipes[rank][0]);
    };

    auto getAndDistributeNCCLid = [&](int nranks){
      close(childPipes[0][1]); //close write to child0
      read(childPipes[0][0], &id, sizeof(ncclUniqueId)); //read from child0
      for(int r = 1; r < nranks; ++r){
        write(childPipes[r][1], &id, sizeof(ncclUniqueId));
        close(childPipes[r][1]);
      }
    };

    for(int r = 0; r < nranks; ++r){
      children[r] = fork();
      if(children[r] == 0){
        //child processes
        if(r == 0)
          createNCCLid(r);
        else
          getNCCLidFromParent(r);
        call_RCCL(id, r, nranks);
        break;
      }
    }

    getAndDistributeNCCLid(nranks);
    for(int r = 0; r < nranks; ++r)
      wait(NULL); // Wait for all children
  }
}
